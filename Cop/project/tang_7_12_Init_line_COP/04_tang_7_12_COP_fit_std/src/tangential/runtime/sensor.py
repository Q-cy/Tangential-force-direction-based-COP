"""面向用户的最小压力传感器 API。

本模块把压力传感器帧解码、CoP/梯度计算和可选标定组合成一个不依赖
Qt/Matplotlib 的 Python API，并提供固定布局的终端渲染器。
"""

import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import PressureConfig, ProcessingConfig
from ..processing.calibration import FitCalibrationModel
from ..processing.cop import PRSensorAngle
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
class TangentialSample:
    """一个合法压力帧及其全部最小 API 计算结果。

    Attributes:
        raw (np.ndarray): 原始 84 通道 ADC 一维数组。
        matrix (np.ndarray): 形状为 ``(12, 7)`` 的 ADC 矩阵。
        gradient (np.ndarray): 压力梯度数组，通常形状为 ``(12, 7, 2)``。
        minimum/maximum/total/mean (float): ADC 最小值、最大值、总和和均值。
        cop_x/cop_y (float): CoP 坐标；未计算或无效时可能为 ``NaN``。
        angle (float): 压力阵列方向角，单位为度。
        dx/dy (float): 平滑后的 CoP 偏移分量。
        state (int): CoP 状态机状态。
        calibrated_fx/calibrated_fy/calibrated_fz (float): 标定力分量，单位
            由模型定义；没有模型时为 ``NaN``。
        calibrated_angle (float): 标定 Fx/Fy 的方向角，单位为度。
        request_seq (int): 传感器请求序号，默认 ``-1`` 表示无元数据。
        tx_t/rx_t (float): 发送和接收时间，使用 ``perf_counter`` 的秒数；
            无元数据时为 ``NaN``。
        latency_s (float): 请求到响应的延迟，单位为秒。
        origin_x/origin_y (float | None): 接触 origin 坐标。
        contact/display_contact/refined (bool): 接触、显示接触和精修状态。
        region_mask (np.ndarray | None): 区域编号矩阵。
        regions (list[dict]): 区域信息列表。
        centroid (tuple[float, float] | None): 压力质心坐标。
        rel_ms (int): 相对首帧时间，单位为毫秒；默认值为 0。
    """

    raw: np.ndarray
    matrix: np.ndarray
    gradient: np.ndarray
    minimum: float
    maximum: float
    total: float
    mean: float
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

    @property
    def raw_2d(self) -> np.ndarray:
        """返回 12×7 ADC 矩阵的直观别名。

        Returns:
            np.ndarray: 与 ``matrix`` 相同的 12×7 数组引用。
        """
        return self.matrix

    @property
    def adc_sum(self) -> float:
        """返回 84 个 ADC 通道的总和。

        Returns:
            float: ``total`` 字段的值。
        """
        return self.total

    @property
    def min(self) -> float:
        """返回 ADC 通道最小值。

        Returns:
            float: ``minimum`` 字段的值。
        """
        return self.minimum

    @property
    def max(self) -> float:
        """返回 ADC 通道最大值。

        Returns:
            float: ``maximum`` 字段的值。
        """
        return self.maximum

    @property
    def sum(self) -> float:
        """返回 ADC 通道总和的别名。

        Returns:
            float: ``total`` 字段的值。
        """
        return self.total

    @property
    def copX(self) -> float:
        """返回 CoP 的 X 坐标。

        Returns:
            float: ``cop_x`` 字段的值，可能为 ``NaN``。
        """
        return self.cop_x

    @property
    def copY(self) -> float:
        """返回 CoP 的 Y 坐标。

        Returns:
            float: ``cop_y`` 字段的值，可能为 ``NaN``。
        """
        return self.cop_y


class TangentialFrameProcessor:
    """复用 CoP 和标定实现处理一个 84 通道压力帧。

    该类只编排既有算法：动态阈值、CoP、梯度、区域和模型预测均委托给
    ``PRSensorAngle`` 或 ``FitCalibrationModel``。

    Attributes:
        cop_sensor (PRSensorAngle): CoP、状态机和梯度计算器。
        calibration: 具有 ``predict`` 方法的标定模型，或 ``None``。
        cal_dim (str): 标定模型使用的输出维度模式。
        region_mode (str): ``full``、``region`` 或 ``both``。
    """

    def __init__(self, cop_sensor=None, calibration=None, cal_dim="3D",
                 region_mode="full", median_window=5):
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

        Returns:
            None: 初始化处理器状态。

        Raises:
            ValueError: ``region_mode`` 不支持或 ``median_window <= 0``。
        """
        if region_mode not in ("full", "region", "both"):
            raise ValueError("region_mode 必须是 full、region 或 both")
        if median_window <= 0:
            raise ValueError("median_window 必须大于0")
        self.cop_sensor = cop_sensor or PRSensorAngle()
        self.calibration = calibration
        self.cal_dim = cal_dim
        self.region_mode = region_mode
        self._dx_values = deque(maxlen=median_window)
        self._dy_values = deque(maxlen=median_window)

    def _predict(self, dx, dy, total):
        """调用注入的标定模型并规范化为三个力分量。

        Args:
            dx (float): 平滑后的 CoP X 偏移。
            dy (float): 平滑后的 CoP Y 偏移。
            total (float): 84 通道 ADC 总和。

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
            return self.calibration.predict(dx, dy, total, self.cal_dim)
        values = list(self.calibration.predict([dx, dy, total]))
        values.extend([float("nan")] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])

    def process(self, raw, frame=None) -> TangentialSample:
        """处理一帧原始压力数据并生成 ``TangentialSample``。

        Args:
            raw (array-like): 原始 ADC 通道序列；长度必须等于
                ``cop_sensor.rows * cop_sensor.cols``，当前为 84。
            frame (Mapping | None): 可选传感器元数据，读取其中的
                ``request_seq``、``tx_t``、``rx_t`` 和 ``latency_s``；默认
                ``None`` 表示使用 ``TangentialSample`` 的缺省值。

        Returns:
            TangentialSample: 包含原始矩阵、统计值、CoP、角度、梯度、区域和
                标定结果的单帧结果。

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

        if use_region:
            regions = self.cop_sensor._compute_region_delta_cop(matrix)
            region_mask = np.zeros(matrix.shape, dtype=np.int32)
            for region in regions:
                for row, col in region["coords"]:
                    region_mask[row, col] = region["id"]
        else:
            regions = []
            region_mask = np.zeros(matrix.shape, dtype=np.int32)

        self._dx_values.append(dx)
        self._dy_values.append(dy)
        filtered_dx = float(np.median(self._dx_values))
        filtered_dy = float(np.median(self._dy_values))
        total = float(np.sum(values))
        cal_fx, cal_fy, cal_fz = self._predict(filtered_dx, filtered_dy, total)
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
            matrix=matrix.copy(),
            gradient=gradient,
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            total=total,
            mean=float(np.mean(values)),
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
        )
        return sample


class TangentialSensorAPI:
    """最小压力采集 API；管理压力设备并返回 ``TangentialSample``。

    Attributes:
        sensor: 提供 ``read_frame``、``decode`` 和可选 ``close`` 的压力传感器。
        processor (TangentialFrameProcessor): 单帧处理器。
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
            processor (TangentialFrameProcessor | None): 已创建的处理器；为
                ``None`` 时按 ``model_path`` 创建默认处理器。
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
                cal_dim=processing_config.cal_dim,
                region_mode=processing_config.region_mode,
                median_window=processing_config.median_window,
            )
        self.sensor = sensor
        self.processor = processor
        self._closed = False

    def read(self, timeout_s=0.1) -> TangentialSample | None:
        """读取并处理下一帧压力数据。

        Args:
            timeout_s (float): 等待合法压力帧的最长时间，单位为秒，默认
                0.1；传给传感器的 ``read_frame``。

        Returns:
            TangentialSample | None: 收到合法帧时返回处理结果；传感器在超时
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
        return self.processor.process(raw, frame)

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


def format_terminal_sample(sample: TangentialSample) -> str:
    """把样本格式化为固定布局的终端文本。

    Args:
        sample (TangentialSample): 要显示的单帧结果，必须包含 12×7
            ``matrix`` 和统计/CoP/角度/标定字段。

    Returns:
        str: 包含 12 行 ADC 和统计、CoP/角度、标定结果的换行文本；不会
        写入终端。

    Raises:
        AttributeError: ``sample`` 缺少所需字段时抛出。
        ValueError: 格式化字段不是可格式化数值时抛出。
    """
    rows = [" ".join(f"{value:7.0f}" for value in row) for row in sample.matrix]
    rows.extend([
        f"min={sample.minimum:12.3f} max={sample.maximum:12.3f} "
        f"sum={sample.total:14.3f} mean={sample.mean:12.3f}",
        f"copX={sample.cop_x:11.4f} copY={sample.cop_y:11.4f} "
        f"angle={sample.angle:10.3f}",
        f"Fx_cal={sample.calibrated_fx:10.4f} "
        f"Fy_cal={sample.calibrated_fy:10.4f} "
        f"Fz_cal={sample.calibrated_fz:10.4f}",
    ])
    return "\n".join(rows)


class FixedTerminalRenderer:
    """每帧只执行一次 ``write``/``flush`` 的固定布局终端渲染器。

    Attributes:
        stream (TextIO): 输出文本流，默认是 ``sys.stdout``。
        _first_frame (bool): 是否尚未渲染首帧，用于决定清屏控制序列。
    """

    def __init__(self, stream=None):
        """初始化终端渲染器。

        Args:
            stream (TextIO | None): 可写文本流；为 ``None`` 时使用当前
                ``sys.stdout``。

        Returns:
            None: 保存输出流并将首帧标志设为 ``True``。

        Side Effects:
            不写入输出流；真正的写入发生在 ``render`` 中。
        """
        self.stream = stream or sys.stdout
        self._first_frame = True

    def render(self, sample: TangentialSample) -> str:
        """格式化并立即刷新一帧终端输出。

        Args:
            sample (TangentialSample): 要渲染的压力样本。

        Returns:
            str: 不含 ANSI 光标控制前缀的格式化样本文本。

        Raises:
            AttributeError/ValueError: ``format_terminal_sample`` 无法读取或
                格式化样本字段时抛出。
            OSError: 输出流写入或刷新失败时可能抛出。

        Side Effects:
            首帧写入清屏并回到左上角控制序列，后续帧写入回到左上角序列；
            每次调用都会写入一次并调用一次 ``flush``，并清除首帧状态。
        """
        text = format_terminal_sample(sample)
        prefix = "\x1b[2J\x1b[H" if self._first_frame else "\x1b[H"
        self._first_frame = False
        self.stream.write(prefix + text + "\n")
        self.stream.flush()
        return text
