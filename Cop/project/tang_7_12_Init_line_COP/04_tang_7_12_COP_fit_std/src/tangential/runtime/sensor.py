"""面向用户的最小压力传感器 API。

本模块把压力传感器帧解码、CoP/梯度计算和可选标定组合成一个不依赖
Qt/Matplotlib 的 Python API。终端输出由示例或调用方自行决定。
"""

import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import PressureConfig, ProcessingConfig
from ..processing.calibration import FitCalibrationModel
from ..processing.cop import PRSensorAngle
from ..processing.slip import SlipDetector, TangentialMotionState
from ..sensors.pressure import PressureSensor


def compute_vector_angle(x: float, y: float) -> float:
    """计算二维向量的方向角并归一化到 ``[0, 360)`` 度。

    Args:
        x (float): 向量 X 分量，无量纲或与 ``y`` 相同单位。
        y (float): 向量 Y 分量，无量纲或与 ``x`` 相同单位。

    Returns:
        float: 以度为单位的方向角；沿正 X 轴为 0 度，负角会加 360 度。

    Notes:
        实现给 X 分量增加极小量以避免在零附近出现除零/符号边界问题。
    """
    angle = float(np.degrees(np.arctan2(y, x + 1e-8)))
    return angle + 360.0 if angle < 0 else angle


def angle_difference(a: float, b: float) -> float:
    """计算两个方向角之间的最小绝对环绕误差。

    Args:
        a (float): 第一个方向角，单位为度。
        b (float): 第二个方向角，单位为度。

    Returns:
        float: 单位为度、范围为 ``[0, 180]`` 的最小绝对角差。
    """
    difference = abs(float(a) - float(b)) % 360.0
    return min(difference, 360.0 - difference)


@dataclass
class TangentialFrame:
    """公开的单帧压力结果。

    公开采集 API 只返回这八个字段。``raw`` 保持为 84 通道的一维数组；
    需要终端显示或自定义矩阵计算时，调用方可以自行 ``reshape(12, 7)``。
    ``adc_sum`` 是对象中唯一的 84 通道 ADC 总和字段；108 列 CSV 使用同名
    ``adc_sum`` 列，二者语义一致。
    """

    raw: np.ndarray
    adc_sum: float
    cop_x: float
    cop_y: float
    angle: float
    dx: float
    dy: float
    motion_state: TangentialMotionState


@dataclass
class TangentialSample:
    """完整应用内部使用的单帧结果。

    该类型不是公共 API。它保留 GUI、CSV、同步和滑移显示所需的 canonical
    字段；公开处理器会从同一次计算结果投影出 ``TangentialFrame``，不会再次
    执行 CoP、梯度、滑移或标定算法。
    """

    raw: np.ndarray
    gradient: np.ndarray
    adc_sum: float
    cop_x: float
    cop_y: float
    angle: float
    dx: float
    dy: float
    state: int
    calibrated_fx: float
    calibrated_fy: float
    calibrated_fz: float
    calibrated_angle: float
    request_seq: int = -1
    tx_t: float = float("nan")
    rx_t: float = float("nan")
    latency_s: float = float("nan")
    origin_x: float | None = None
    origin_y: float | None = None
    contact: bool = False
    display_contact: bool = False
    refined: bool = False
    region_mask: np.ndarray | None = None
    regions: list[dict] = field(default_factory=list)
    centroid: tuple[float, float] | None = None
    rel_ms: int = 0
    motion_state: TangentialMotionState = TangentialMotionState.NO_CONTACT
    is_slipping: bool = False
    slip_motion_distance: float = 0.0
    slip_confidence: float = 0.0
    angle_vector_magnitude: float = 0.0


class TangentialSampleProcessor:
    """把一个 84 通道压力帧处理为完整的内部 ``TangentialSample``。

    该类负责完整应用需要的动态阈值、CoP、梯度、区域、滑移和标定状态；
    具体算法仍委托给 ``PRSensorAngle``、``SlipDetector`` 和
    ``FitCalibrationModel``。它是运行时内部实现，不属于公开 API。

    Attributes:
        cop_sensor (PRSensorAngle): CoP、状态机和梯度计算器。
        calibration: 具有 ``predict`` 方法的标定模型，或 ``None``。
        cal_dim (str): 标定模型使用的输出维度模式。
        region_mode (str): ``full``、``region`` 或 ``both``。
    """

    def __init__(self, cop_sensor=None, calibration=None, cal_dim=None,
                 region_mode=None, median_window=None,
                 processing_config: ProcessingConfig | None = None,
                 slip_detector: SlipDetector | None = None):
        """初始化单帧处理器和偏移量中值滤波状态。

        Args:
            cop_sensor (PRSensorAngle | None): 可注入的 CoP 计算器；为
                ``None`` 时创建默认实例。
            calibration (object | None): 标定模型；应提供 ``predict`` 方法。
                ``None`` 时输出标定值为 ``NaN``。
            cal_dim (str): 传给 ``FitCalibrationModel.predict`` 的维度模式，
                默认 ``"3D"``。
            region_mode (str): 区域计算模式，必须是 ``"full"``、``"region"``
                或 ``"both"``，默认 ``"full"``。
            median_window (int): dx/dy 中值滤波窗口长度，默认 5，必须为正数。
            processing_config (ProcessingConfig | None): CoP、区域、滤波和滑移
                参数的集中配置；显式 ``cal_dim``、``region_mode`` 和
                ``median_window`` 参数优先于其中的对应字段。
            slip_detector (SlipDetector | None): 可选的独立滑移检测器；传入后
                由调用方负责保证其阵列尺寸匹配，否则按配置为本处理器创建独立实例。

        Returns:
            None: 初始化处理器状态。

        Raises:
            ValueError: ``region_mode`` 不支持或 ``median_window <= 0``。
        """
        defaults = (processing_config or ProcessingConfig()).validate()
        cal_dim = defaults.cal_dim if cal_dim is None else cal_dim
        region_mode = defaults.region_mode if region_mode is None else region_mode
        median_window = defaults.median_window if median_window is None else median_window
        if region_mode not in ("full", "region", "both"):
            raise ValueError("region_mode 必须是 full、region 或 both")
        if median_window <= 0:
            raise ValueError("median_window 必须大于0")
        self.cop_sensor = cop_sensor or PRSensorAngle(config=defaults.cop)
        self.slip_config = defaults.slip
        self.slip_detector = slip_detector or SlipDetector(
            config=defaults.slip,
            rows=self.cop_sensor.rows,
            cols=self.cop_sensor.cols,
        )
        self.calibration = calibration
        self.cal_dim = cal_dim
        self.region_mode = region_mode
        self._dx_values = deque(maxlen=median_window)
        self._dy_values = deque(maxlen=median_window)

    def _predict(self, dx, dy, adc_sum):
        """调用注入的标定模型并规范化为三个力分量。

        Args:
            dx (float): 平滑后的 CoP X 偏移。
            dy (float): 平滑后的 CoP Y 偏移。
            adc_sum (float): 84 通道 ADC 总和。

        Returns:
            tuple[float, float, float]: Fx、Fy、Fz 预测值；没有模型或模型
                输出不足三维时，缺失值补为 ``NaN``。

        Raises:
            Exception: 注入模型的 ``predict`` 方法抛出的模型相关异常会向上
                传播。
        """
        if self.calibration is None:
            return (float("nan"),) * 3
        if isinstance(self.calibration, FitCalibrationModel):
            return self.calibration.predict(dx, dy, adc_sum, self.cal_dim)
        values = list(self.calibration.predict([dx, dy, adc_sum]))
        values.extend([float("nan")] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])

    def _process_sample(self, raw, frame=None) -> TangentialSample:
        """处理一帧原始压力数据并生成完整内部结果。

        Args:
            raw (array-like): 原始 ADC 通道序列；长度必须等于
                ``cop_sensor.rows * cop_sensor.cols``，当前为 84。
            frame (Mapping | None): 可选传感器元数据，读取其中的
                ``request_seq``、``tx_t``、``rx_t`` 和 ``latency_s``；默认
                ``None`` 表示使用内部结果的缺省元数据。

        Returns:
            TangentialSample: 包含完整应用所需的 ADC、CoP、角度、梯度、区域、
                滑移、标定和时间元数据。

        Raises:
            ValueError: ``raw`` 通道数不是处理器期望的数量。
            Exception: CoP 或标定实现内部发生错误时向上传播。

        Side Effects:
            更新 ``PRSensorAngle`` 的动态阈值/状态机和本对象的 dx/dy 中值
            滤波队列。
        """
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        expected = self.cop_sensor.rows * self.cop_sensor.cols
        if values.size != expected:
            raise ValueError(f"压力帧通道数必须为{expected}，实际为{values.size}")
        matrix = values.reshape(self.cop_sensor.rows, self.cop_sensor.cols)
        self.cop_sensor.dynamic_threshold(matrix)

        use_full = self.region_mode in ("full", "both")
        use_region = self.region_mode in ("region", "both")
        if use_full:
            angle, dx, dy, cop_x, cop_y = self.cop_sensor.get_all(values)
            origin_x, origin_y = self.cop_sensor.get_origin()
            state = self.cop_sensor.get_state()
            gradient = np.asarray(
                self.cop_sensor.get_gradient(values), dtype=np.float64
            )
            centroid = self.cop_sensor._compute_centroid(matrix)
        else:
            angle = dx = dy = 0.0
            cop_x = cop_y = float("nan")
            origin_x = origin_y = None
            state = 0
            gradient = np.zeros(
                (self.cop_sensor.rows, self.cop_sensor.cols, 2),
                dtype=np.float64,
            )
            centroid = None

        # region-only 仍可用整帧聚合 CoP 做全局滑移检测；不为每个 region
        # 创建或共享滑移状态。get_cop 是无状态查询，不推进全局 origin。
        if use_full:
            global_cop_x, global_cop_y = cop_x, cop_y
        elif hasattr(self.cop_sensor, "get_cop"):
            global_cop_x, global_cop_y = self.cop_sensor.get_cop(values)
        else:
            global_cop_x = global_cop_y = float("nan")

        if use_region:
            regions = self.cop_sensor._compute_region_delta_cop(matrix)
            region_mask = np.zeros(matrix.shape, dtype=np.int32)
            for region in regions:
                for row, col in region["coords"]:
                    region_mask[row, col] = region["id"]
        else:
            regions = []
            region_mask = np.zeros(matrix.shape, dtype=np.int32)

        detector_contact = bool(state > 0) if use_full else any(
            region.get("contact_init", False) for region in regions
        )
        motion_ready = (
            self.cop_sensor.is_motion_ready()
            if use_full and hasattr(self.cop_sensor, "is_motion_ready")
            else True
        )
        slip_result = self.slip_detector.update(
            matrix,
            global_cop_x,
            global_cop_y,
            contact=detector_contact,
            ready=motion_ready and np.isfinite(global_cop_x) and np.isfinite(global_cop_y),
        )
        if slip_result.reanchored and use_full:
            # 仅在滑移退出时重锁既有全局 CoP origin；滑移期间 detector
            # 自己维护 anchor，避免滑移运动污染静态 dx/dy。
            self.cop_sensor.reanchor_origin(global_cop_x, global_cop_y)
            origin_x, origin_y = global_cop_x, global_cop_y

        if slip_result.reanchored:
            angle = 0.0
            angle_vector_magnitude = 0.0
        elif slip_result.is_slipping:
            angle_vector_magnitude = slip_result.angle_vector_magnitude
            angle = (
                PRSensorAngle._compute_cop_angle(
                    slip_result.direction_x, slip_result.direction_y
                )
                if angle_vector_magnitude >= self.slip_config.angle_deadband else 0.0
            )
        else:
            angle_vector_magnitude = float(np.hypot(dx, dy)) if use_full else 0.0
            if angle_vector_magnitude < self.slip_config.angle_deadband:
                angle = 0.0
        if float(angle) == 0.0:
            angle = 0.0

        self._dx_values.append(dx)
        self._dy_values.append(dy)
        filtered_dx = float(np.median(self._dx_values))
        filtered_dy = float(np.median(self._dy_values))
        adc_sum = float(np.sum(values))
        cal_fx, cal_fy, cal_fz = self._predict(
            filtered_dx, filtered_dy, adc_sum
        )
        cal_angle = (
            compute_vector_angle(cal_fx, cal_fy)
            if np.isfinite(cal_fx) and np.isfinite(cal_fy)
            else float("nan")
        )
        contact = state > 0
        display_contact = contact
        if use_region and not use_full:
            display_contact = any(
                region.get("contact_init", False) for region in regions
            )

        metadata = frame or {}
        sample = TangentialSample(
            raw=values.copy(),
            gradient=gradient,
            adc_sum=adc_sum,
            cop_x=float(cop_x),
            cop_y=float(cop_y),
            angle=float(angle),
            dx=filtered_dx,
            dy=filtered_dy,
            state=int(state),
            calibrated_fx=cal_fx,
            calibrated_fy=cal_fy,
            calibrated_fz=cal_fz,
            calibrated_angle=cal_angle,
            request_seq=int(metadata.get("request_seq", -1)),
            tx_t=float(metadata.get("tx_t", float("nan"))),
            rx_t=float(metadata.get("rx_t", float("nan"))),
            latency_s=float(metadata.get("latency_s", float("nan"))),
            origin_x=origin_x,
            origin_y=origin_y,
            contact=bool(contact),
            display_contact=bool(display_contact),
            refined=state == 2,
            region_mask=region_mask,
            regions=regions,
            centroid=centroid,
            motion_state=slip_result.motion_state,
            is_slipping=slip_result.is_slipping,
            slip_motion_distance=slip_result.motion_distance,
            slip_confidence=slip_result.confidence,
            angle_vector_magnitude=angle_vector_magnitude,
        )
        return sample


class TangentialFrameProcessor:
    """面向用户的 ``TangentialFrame`` 薄处理门面。

    本类不持有 CoP、滑移或标定算法实现；它只调用一个
    ``TangentialSampleProcessor`` 生成完整内部结果，再通过自身的私有静态方法
    ``_to_tangential_frame`` 投影为八字段公开结果。每个门面都根据传入的
    CoP、标定和处理配置参数创建并独占一个 ``TangentialSampleProcessor``，
    不允许从公开构造函数注入或共享内部样本处理器。

    Attributes:
        _sample_processor (TangentialSampleProcessor): 本门面独占的内部样本处理器。
    """

    def __init__(self, cop_sensor=None, calibration=None, cal_dim=None,
                 region_mode=None, median_window=None,
                 processing_config: ProcessingConfig | None = None,
                 slip_detector: SlipDetector | None = None):
        """初始化公开门面及其内部样本处理器。

        Args:
            cop_sensor (PRSensorAngle | None): 传给内部
                ``TangentialSampleProcessor`` 的 CoP 计算器；为 ``None`` 时
                创建默认实例。
            calibration (object | None): 内部样本处理器使用的标定模型。
            cal_dim (str | None): 标定维度模式。
            region_mode (str | None): ``full``、``region`` 或 ``both``。
            median_window (int | None): dx/dy 中值滤波窗口长度。
            processing_config (ProcessingConfig | None): CoP、区域、滤波和
                滑移的集中配置。
            slip_detector (SlipDetector | None): 可注入的独立滑移检测器。

        Returns:
            None: 创建门面并保存一个内部样本处理器。

        Raises:
            ValueError: 处理配置中的区域模式或滤波窗口非法。
        """
        self._sample_processor = TangentialSampleProcessor(
            cop_sensor=cop_sensor,
            calibration=calibration,
            cal_dim=cal_dim,
            region_mode=region_mode,
            median_window=median_window,
            processing_config=processing_config,
            slip_detector=slip_detector,
        )

    @staticmethod
    def _to_tangential_frame(sample: TangentialSample) -> TangentialFrame:
        """从完整内部结果中挑选八个稳定公开字段。

        Args:
            sample (TangentialSample): 当前处理器刚刚计算出的内部详细结果。

        Returns:
            TangentialFrame: 只包含 raw、adc_sum、CoP、角度、偏移和运动状态。

        Notes:
            本函数只复制已经算出的字段，不调用 CoP、滑移或标定算法，也不属于
            SDK 公共导出。
        """
        return TangentialFrame(
            raw=sample.raw.copy(),
            adc_sum=sample.adc_sum,
            cop_x=sample.cop_x,
            cop_y=sample.cop_y,
            angle=sample.angle,
            dx=sample.dx,
            dy=sample.dy,
            motion_state=sample.motion_state,
        )

    def process_frame(self, raw, frame=None) -> TangentialFrame:
        """处理一帧原始压力数据并返回简化公开结果。

        Args:
            raw (array-like): 原始 ADC 通道序列；当前必须包含 84 个通道。
            frame (Mapping | None): 可选压力帧元数据；完整应用使用它保存
                真实时间和请求序号，公开结果不暴露这些元数据。

        Returns:
            TangentialFrame: 只有八个稳定公开字段的单帧结果。

        Raises:
            ValueError: ``raw`` 通道数不是处理器期望的数量。
            Exception: CoP、滑移或标定实现内部发生错误时向上传播。

        Side Effects:
            通过内部样本处理器更新 CoP 状态机、滑移检测器和 dx/dy 中值滤波状态。
        """
        sample = self._sample_processor._process_sample(raw, frame)
        return self._to_tangential_frame(sample)


class TangentialSensorAPI:
    """最小压力采集 API；管理压力设备并返回 ``TangentialFrame``。

    Attributes:
        sensor: 提供 ``read_frame``、``decode`` 和可选 ``close`` 的压力传感器。
        processor (TangentialFrameProcessor): 只返回公开 ``TangentialFrame`` 的
            单帧处理门面。
        _closed (bool): 是否已经关闭；关闭后不能继续读取。
    """

    def __init__(self, sensor=None, processor=None, sensor_factory=None,
                 model_path=None, pressure_port=None,
                 config: PressureConfig | None = None,
                 processing_config: ProcessingConfig | None = None):
        """创建压力采集 API，并按需构造传感器和标定处理器。

        Args:
            sensor (object | None): 已创建的传感器对象；传入后不使用
                ``sensor_factory`` 创建新对象。
            processor (TangentialFrameProcessor | None): 已创建的公开处理门面；
                为 ``None`` 时按 ``model_path`` 创建默认门面。不能注入内部
                ``TangentialSampleProcessor``，以保证本 API 始终返回 ``Frame``。
            sensor_factory (callable | None): 接受 ``port=...`` 的传感器工厂；
                仅在 ``sensor`` 为 ``None`` 时使用。
            model_path (str | os.PathLike | None): 外部模型路径；为 ``None``
                时加载内置 package resource 模型。
            pressure_port (str): 压力传感器串口路径，默认
                ``/dev/ttyUSB0``。
            config (PressureConfig | None): 压力设备配置；传入时覆盖端口、
                周期、响应超时、队列和启动超时默认值。
            processing_config (ProcessingConfig | None): 单帧处理配置；未注入
                ``processor`` 时用于创建 CoP 和标定处理器。

        Returns:
            None: 保存传感器、处理器和关闭状态。

        Raises:
            TypeError: 注入的 ``processor`` 不是 ``TangentialFrameProcessor``。
            Exception: 传感器工厂、模型加载或处理器创建失败时向上传播。

        Side Effects:
            若未注入 ``sensor``，会立即调用传感器工厂；若未注入
            ``processor``，会立即加载标定模型。
        """
        if config is None:
            config = PressureConfig()
            if pressure_port is not None:
                config.port = pressure_port
        config.validate()
        pressure_port = config.port
        if sensor is None:
            if sensor_factory is None:
                sensor_factory = PressureSensor
            if sensor_factory is PressureSensor:
                sensor = sensor_factory(
                    port=pressure_port,
                    period_s=config.period_s,
                    response_timeout_s=config.response_timeout_s,
                    queue_size=config.frame_queue_size,
                    baudrate=config.baudrate,
                    _startup_timeout_s=config.startup_timeout_s,
                )
            else:
                sensor = sensor_factory(port=pressure_port)
        if processor is None:
            calibration = (
                FitCalibrationModel.from_default()
                if model_path is None
                else FitCalibrationModel.from_path(model_path)
            )
            processing_config = processing_config or ProcessingConfig()
            processing_config.validate()
            processor = TangentialFrameProcessor(
                cop_sensor=PRSensorAngle(**processing_config.cop.as_kwargs()),
                calibration=calibration,
                processing_config=processing_config,
            )
        elif not isinstance(processor, TangentialFrameProcessor):
            raise TypeError(
                "TangentialSensorAPI.processor 必须是 TangentialFrameProcessor"
            )
        self.sensor = sensor
        self.processor = processor
        self._closed = False

    def read(self, timeout_s=0.1) -> TangentialFrame | None:
        """读取并处理下一帧压力数据。

        Args:
            timeout_s (float): 等待合法压力帧的最长时间，单位为秒，默认
                0.1；传给传感器的 ``read_frame``。

        Returns:
            TangentialFrame | None: 收到合法帧时返回八字段处理结果；传感器在超时
                或无帧时返回 ``None``。

        Raises:
            RuntimeError: API 已经由 ``close`` 关闭。
            Exception: 传感器读取/解码或单帧处理失败时向上传播。

        Side Effects:
            读取传感器并推进其内部请求/响应状态，同时更新处理器的状态机
            和 dx/dy 滤波状态。
        """
        if self._closed:
            raise RuntimeError("TangentialSensorAPI 已关闭")
        frame = self.sensor.read_frame(timeout_s=timeout_s)
        if frame is None:
            return None
        raw = self.sensor.decode(frame["raw"])
        return self.processor.process_frame(raw, frame)

    def close(self):
        """幂等地关闭压力传感器。

        Args:
            None: 此方法不接收业务参数。

        Returns:
            None: 首次调用关闭资源，重复调用直接返回。

        Side Effects:
            将 API 标记为关闭，并在传感器提供 ``close`` 方法时调用它；传感器
            的串口、线程或子进程由传感器实现负责释放。
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self.sensor, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        """进入上下文管理器并返回当前 API 对象。

        Returns:
            TangentialSensorAPI: 当前实例，可用于 ``read``。
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文管理器并释放传感器资源。

        Args:
            exc_type (type | None): 上下文内异常类型。
            exc_value (BaseException | None): 上下文内异常实例。
            traceback (TracebackType | None): 上下文内异常回溯。

        Returns:
            bool: 始终返回 ``False``，不抑制上下文内异常。
        """
        self.close()
        return False


def format_terminal_sample(sample: TangentialFrame) -> str:
    """把公开帧格式化为固定布局的终端文本。

    Args:
        sample (TangentialFrame): 要显示的单帧结果；``raw`` 必须包含 84
            个 ADC 通道，并会在此处 reshape 为 12×7。

    Returns:
        str: 包含 12 行 ADC 以及 adc_sum、CoP、角度、dx/dy 和运动状态的
        换行文本；不会写入终端。

    Raises:
        AttributeError: ``sample`` 缺少所需字段时抛出。
        ValueError: 格式化字段不是可格式化数值时抛出。
    """
    matrix = np.asarray(sample.raw).reshape(12, 7)
    rows = [" ".join(f"{value:7.0f}" for value in row) for row in matrix]
    rows.extend([
        f"adc_sum={sample.adc_sum:14.3f}",
        f"cop_x={sample.cop_x:11.4f} cop_y={sample.cop_y:11.4f} "
        f"angle={sample.angle:10.3f}",
        f"dx={sample.dx:11.4f} dy={sample.dy:11.4f} "
        f"motion_state={sample.motion_state.name}",
    ])
    return "\n".join(rows)


class FixedTerminalRenderer:
    """每帧只执行一次 ``write``/``flush`` 的固定布局终端渲染器。"""

    def __init__(self, stream=None):
        """初始化终端渲染器。

        Args:
            stream (TextIO | None): 可写文本流；为 ``None`` 时使用当前
                ``sys.stdout``。
        """
        self.stream = stream or sys.stdout
        self._first_frame = True

    def render(self, sample: TangentialFrame) -> str:
        """格式化并立即刷新一帧终端输出。"""
        text = format_terminal_sample(sample)
        prefix = "\x1b[2J\x1b[H" if self._first_frame else "\x1b[H"
        self._first_frame = False
        self.stream.write(prefix + text + "\n")
        self.stream.flush()
        return text
