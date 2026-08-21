# Tangential Sensor SDK 0.4.0

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器，提供 CoP、角度、梯度、切向力标定、实时 GUI、固定 108 列 CSV 和离线分析。

本文面向安装和使用 SDK 的用户，介绍硬件连接、命令行、Python API、参数配置、
滑移检测、CSV 行为和常见故障。用户可以安装 wheel，也可以在获得源码后直接运行。

## 系统要求与安装

当前 wheel 适用于 Linux x86_64 和 CPython 3.11。压力传感器是必需设备，
六维力传感器是可选设备；默认端口分别为 ``/dev/ttyUSB0`` 和
``/dev/ttyUSB1``。

完整功能包含实时 GUI 和离线绘图，推荐安装：

```bash
python -m pip install "./dist/tangential_sensor-0.4.0-cp311-cp311-linux_x86_64.whl[full]"
```

只使用压力采集、CoP、标定等核心 API 时，可以不安装 GUI 可选依赖：

```bash
python -m pip install ./dist/tangential_sensor-0.4.0-cp311-cp311-linux_x86_64.whl
```

安装后检查：

```bash
tangential --version
python -c "import tangential; print(tangential.__version__)"
```

## 从源码运行

在项目根目录安装依赖并通过 ``PYTHONPATH=src`` 运行：

~~~bash
python -m pip install -r requirements.txt

PYTHONPATH=src python -m tangential.examples.minimal
PYTHONPATH=src python -m tangential.examples.full
~~~

也可以直接运行 CLI：

~~~bash
PYTHONPATH=src python -m tangential.cli --version
PYTHONPATH=src python -m tangential.cli example --help
PYTHONPATH=src python -m tangential.cli app --help
~~~

``minimal`` 需要压力传感器；``full`` 需要完整 GUI 依赖。安装 wheel 的用户
不需要设置 ``PYTHONPATH``。

## 同时连接两个压力传感器

双传感器示例模块为 ``tangential.examples.dual_sensor``。它启动一个 Qt 应用
和两个完整窗口；A/B 各自执行压力采集、CoP、角度、梯度、标定、实时曲线、
压力表、完整 108 列 CSV，并在退出时生成各自的分析图。不再是终端摘要循环。
默认只连接压力传感器；只有显式提供对应 ``--force-port-a`` 或
``--force-port-b`` 才启用六维力通道，避免两路同时打开默认 ``/dev/ttyUSB1``。

### 第1步：插入设备并识别两个端口

插入两只压力传感器后运行：

~~~bash
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
~~~

优先选择 ``/dev/serial/by-id/`` 下两个不同的设备路径，因为它们通常不会随
重插或重启改变。若该目录不存在，再根据 ``serial.tools.list_ports`` 的
输出确认两只设备分别对应哪个 ``/dev/ttyUSB*`` 或 ``/dev/ttyACM*``。

本机当前如果没有列出任何端口，说明设备尚未接入、USB未识别或串口驱动尚未
创建，不能继续启动示例。

### 第2步：设置本次运行使用的端口

把下面两行中的 ``DEVICE_A_ID`` 和 ``DEVICE_B_ID`` 替换为第1步看到的真实文件名，
再执行后续命令。例如，真实路径可能类似
``/dev/serial/by-id/usb-FTDI_A1-if00-port0`` 和
``/dev/serial/by-id/usb-FTDI_B2-if00-port0``；下面的名称只是示意：

~~~bash
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
printf 'A=%s\nB=%s\n' "$PORT_A" "$PORT_B"
~~~

不要把 ``<sensor-a>`` 或 ``<sensor-b>`` 原样输入命令，也不要把它们写进
变量赋值。Bash会把尖括号解释成输入/输出重定向符号，从而产生
``syntax error near unexpected token 'newline'``。只有替换成第1步实际查到的
路径后，才能继续执行 ``printf`` 和启动命令。

如果没有 ``by-id`` 路径，且已经确认端口映射，可以改成：

~~~bash
PORT_A=/dev/ttyUSB0
PORT_B=/dev/ttyUSB1
~~~

两个变量必须对应不同物理设备。示例会解析符号链接，并在打开串口前拒绝
两个变量最终指向同一物理串口。

### 第3步：检查权限和端口占用

~~~bash
ls -l "$PORT_A" "$PORT_B"
groups
fuser "$PORT_A" "$PORT_B"
~~~

- ``ls`` 必须能找到两个路径。
- 当前用户通常需要属于 ``dialout`` 组；若没有权限，可执行
  ``sudo usermod -aG dialout "$USER"``，然后注销并重新登录。
- ``fuser`` 没有输出通常表示端口空闲；若显示进程号，应先关闭正在占用
  传感器的旧采集程序，不要让两个程序同时读取同一串口。

### 第4步：启动双传感器示例

从源码运行：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
~~~

安装wheel后运行，不需要 ``PYTHONPATH``：

~~~bash
python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
~~~

默认输出目录为 ``./data/sensor_a`` 和 ``./data/sensor_b``。指定父目录时：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" --port-b "$PORT_B" \
  --save-dir ./data/dual
~~~

如果两路都要连接六维力传感器，必须显式提供两个不同的力端口：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" --port-b "$PORT_B" \
  --force-port-a /dev/serial/by-id/FORCE_A \
  --force-port-b /dev/serial/by-id/FORCE_B
~~~

也可以分别覆盖输出目录：``--save-dir-a`` 和 ``--save-dir-b``；模型使用
``--model MODEL_PATH``，或分别使用 ``--model-a``、``--model-b``。

查看全部参数：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor --help
~~~

### 第5步：确认输出并停止

运行后会出现两个窗口，标题分别包含 ``Sensor A`` 和 ``Sensor B``。每个窗口
都包含压力/六维力实时曲线、方向和幅值、12×7 压力表、CoP 标记、梯度箭头
以及状态显示；状态变化不会覆盖 A/B 标签。每路目录会保存一个完整 108 列
CSV，退出后还会保存 ``full_analysis_cop_<n>.png``。

按 ``Ctrl+C`` 或关闭 Qt 应用时，两路会同时停止；任一路采集线程异常都会
报告具体的 A/B，并联动安全关闭另一路。不要直接拔线代替正常退出。

Python调用：

~~~python
from tangential import FullApplicationConfig
from tangential.config import ForceConfig, GuiConfig, OutputConfig, PressureConfig
from tangential.examples.dual_sensor import run

run(
    FullApplicationConfig(
        pressure=PressureConfig(port="/dev/serial/by-id/DEVICE_A_ID"),
        force=ForceConfig(enabled=False),
        output=OutputConfig(save_dir="./data/sensor_a"),
        gui=GuiConfig(window_title="Sensor A"),
    ),
    FullApplicationConfig(
        pressure=PressureConfig(port="/dev/serial/by-id/DEVICE_B_ID"),
        force=ForceConfig(enabled=False),
        output=OutputConfig(save_dir="./data/sensor_b"),
        gui=GuiConfig(window_title="Sensor B"),
    ),
)
~~~

更直接的公共入口是 ``run_dual_application(config_a, config_b)``。每一路都有
独立串口、采集进程、IPC队列、读取线程、缓存、CoP状态机、标定处理器、停止
事件、GUI和输出目录；一个设备的读取超时不会占用另一个设备的串口消费者。
软件状态互相隔离，但 USB 控制器带宽、CPU 调度和供电仍是共享硬件资源，实际
帧率应分别验收。

### 常见错误

| 现象 | 原因 | 处理方法 |
| --- | --- | --- |
| Bash报告 ``unexpected token newline`` | 原样复制了带 ``<...>`` 的占位符 | 按第1、2步设置真实 ``PORT_A``/``PORT_B`` |
| ``No such file or directory`` | 设备未连接或端口名已变化 | 重新运行 ``serial.tools.list_ports -v`` |
| ``Permission denied`` | 当前用户没有串口权限 | 加入 ``dialout`` 后重新登录 |
| 提示两个传感器使用同一物理串口 | 两个路径相同，或两个符号链接指向同一设备 | 为A、B选择两个不同设备路径 |
| 某一路窗口持续无数据 | 端口选错、设备无响应、供电或USB带宽异常 | 单独运行最小示例验证该端口，再检查USB连接 |

## 命令行

安装 wheel 后使用统一命令：

~~~bash
tangential --version
tangential example
tangential app
tangential plot --help
tangential fit --help
~~~

### 最小压力采集

~~~bash
tangential example \
  --pressure-port /dev/ttyUSB0 \
  --timeout 0.1
~~~

终端每帧显示 12×7 原始 ADC、min、max、sum、mean、CoP X/Y 和角度。此路径不启动六维力、CSV 或 Qt GUI。

### 完整采集

~~~bash
tangential app \
  --pressure-port /dev/ttyUSB0 \
  --force-port /dev/ttyUSB1 \
  --save-dir ./data \
  --max-time-diff-ms 15
~~~

压力传感器是必需设备；连接失败时程序退出且不创建空 CSV。六维力传感器是可选设备；连接或普通数据帧校零失败时降级为压力模式，力相关列写入 NaN。两路设备由独立采集进程读取，父进程按真实接收时间完成匹配和 CSV 保存。

### 双路完整采集

~~~bash
tangential dual \
  --port-a /dev/serial/by-id/PRESSURE_A \
  --port-b /dev/serial/by-id/PRESSURE_B \
  --save-dir ./data/dual
~~~

该命令显示两个完整 GUI 窗口，默认把 CSV 和退出分析图分别保存到
``./data/dual/sensor_a``、``./data/dual/sensor_b``。只有显式增加
``--force-port-a``、``--force-port-b`` 才启用对应六维力通道；两个力端口也
必须是不同物理设备。

### 离线绘图

绘图按 CSV 实际表头解析列名，不依赖旧版硬编码列索引：

~~~bash
tangential plot \
  --dir ./data \
  --files capture.csv \
  --columns Fy_cal,delta_Force_Y \
  --rows 100:500 \
  --save ./data/capture.png
~~~

使用 --list 列出 CSV，使用 --mode full_analysis 生成完整分析图。空文件、缺列和空行范围会返回明确错误。

### 离线训练

~~~bash
tangential fit \
  --xy-csv ./data/fx_fy.csv \
  --z-csv ./data/fz.csv \
  --output-model ./fit_coefs.bin \
  --output-plot ./fit_report.png
~~~

默认只生成模型和评估图，不修改输入 CSV。只有明确提供 --write-back PATH 才会写回；目标已存在时必须额外提供 --force。

## Python API

所有稳定公共名称都可以直接从 ``tangential`` 导入。普通采集优先使用
``TangentialSensor``；需要完整 GUI 时使用 ``run_application`` 或
``run_dual_application``。``PressureSensor``、``PRSensorAngle`` 和
``TangentialFrameProcessor`` 面向需要自行编排数据流的高级用户。

### 最小采集示例

~~~python
from tangential import PressureConfig, TangentialSensor

pressure = PressureConfig(port="/dev/ttyUSB0")
with TangentialSensor(config=pressure) as sensor:
    while True:
        sample = sensor.read(timeout_s=0.1)
        if sample is not None:
            print(sample.matrix)
            print(sample.minimum, sample.maximum, sample.total, sample.mean)
            print(sample.cop_x, sample.cop_y, sample.angle)
~~~

### 完整应用示例

~~~python
from tangential import (
    ForceConfig,
    FullApplicationConfig,
    OutputConfig,
    PressureConfig,
    run_application,
)

config = FullApplicationConfig(
    pressure=PressureConfig(port="/dev/ttyUSB0"),
    force=ForceConfig(enabled=True, port="/dev/ttyUSB1"),
    output=OutputConfig(save_dir="./data"),
)
run_application(config)
~~~

### 公共 API 总览

以下表格覆盖当前 ``tangential.__all__`` 的全部33个公共名称。

#### 采集、处理与终端输出

| API | 作用 | 主要输入 | 返回值或输出 |
| --- | --- | --- | --- |
| ``TangentialSensor`` | 推荐的单压力传感器高级 API；是 ``TangentialSensorAPI`` 的别名，支持上下文管理器 | ``PressureConfig``、可选 ``ProcessingConfig``、模型路径 | ``read(timeout_s)`` 返回 ``TangentialSample`` 或 ``None`` |
| ``TangentialSensorAPI`` | 管理压力设备生命周期并串联解码、CoP、滑移和标定 | 传感器/工厂注入、压力配置、处理配置 | 逐帧 ``TangentialSample``；``close()`` 释放设备 |
| ``TangentialSample`` | 保存一帧压力数据和全部计算结果 | 通常由处理器创建，不建议用户手工构造 | ADC、CoP、角度、标定、时间戳、区域和滑移字段 |
| ``TangentialFrameProcessor`` | 对已有84通道 ADC 做单帧计算，不负责串口 | ``raw``、``ProcessingConfig``、可选标定模型 | ``process()`` 返回 ``TangentialSample`` |
| ``FixedTerminalRenderer`` | 在终端固定位置刷新12×7矩阵和指标 | 输出流、``TangentialSample`` | ``render()`` 写入并刷新终端，同时返回文本 |
| ``format_terminal_sample`` | 将样本格式化为固定布局文本，不直接采集 | ``TangentialSample`` | ``str`` |

#### 算法、模型与底层压力驱动

| API | 作用 | 主要输入 | 返回值或输出 |
| --- | --- | --- | --- |
| ``FitCalibrationModel`` | 加载内置或外部 ``fit_coefs.bin``，预测 Fx/Fy/Fz | ``from_default()`` 或 ``from_path(path)``；``predict(dx, dy, total, dim)`` | 三个标定力分量及模型状态 |
| ``PRSensorAngle`` | 动态阈值、接触状态、CoP、origin、角度、梯度和区域计算 | 84通道 ADC、``CopConfig`` | CoP/角度/梯度/状态；高级用户使用 |
| ``PressureSensor`` | 底层 PZT 串口请求、帧解析、CRC和时序统计 | 串口、周期、超时、队列等 | ``read_frame()`` 帧字典、``decode()`` 84通道数据 |
| ``SlipDetector`` | 独立的逐帧全局滑移状态机 | 压力矩阵、CoP、接触/ready状态、``SlipConfig`` | ``SlipResult`` |
| ``SlipResult`` | 不可变的单帧滑移检测结果 | 由 ``SlipDetector`` 生成 | 状态、位移、置信度、方向、斑块平移和重锚定标志 |
| ``TangentialMotionState`` | 滑移状态枚举 | 无 | ``NO_CONTACT``、``STICK``、``SLIP`` |
| ``compute_vector_angle`` | 计算二维向量方向角 | ``x``、``y`` | ``[0, 360)`` 度角 |
| ``angle_difference`` | 计算两个方向角的最小环绕差 | 两个角度 | ``[0, 180]`` 度差 |

#### 完整应用入口

| API | 作用 | 主要输入 | 返回值或输出 |
| --- | --- | --- | --- |
| ``run_application`` | 启动一路完整采集、CSV和实时 GUI | ``FullApplicationConfig`` | 阻塞运行至窗口关闭；正常退出返回0并输出CSV/分析图 |
| ``run_dual_application`` | 在同一 Qt 应用中启动两路相互隔离的完整采集 | ``config_a``、``config_b`` | 正常退出返回0，并生成两个GUI、两套CSV和分析图 |

#### 配置对象

| API | 作用 | 主要输入 | 返回值或输出 |
| --- | --- | --- | --- |
| ``FullApplicationConfig`` | 聚合完整应用的全部分类配置 | pressure、force、processing、calibration、sync、output、gui | 经校验的完整配置对象 |
| ``PressureConfig`` | 压力串口、请求频率、超时和队列配置 | 端口及轮询参数 | 压力设备配置；``period_s`` 返回周期 |
| ``ForceConfig`` | 六维力启用、串口、频率和软件校零配置 | 端口、频率、校零样本/超时 | 六维力设备配置；``enabled=False`` 禁用通道 |
| ``CopConfig`` | CoP、动态阈值、接触、精修和区域参数 | 各阈值、帧数和区域参数 | CoP配置；``as_kwargs()`` 返回算法参数字典 |
| ``ProcessingConfig`` | 组合单帧处理模式、滤波、CoP和滑移参数 | cal_dim、region_mode、cop、slip等 | 单帧处理配置 |
| ``SlipConfig`` | 控制滑移窗口、判定阈值、滞回和方向平滑 | 12项滑移参数 | 经 ``validate()`` 校验的滑移配置 |
| ``CalibrationConfig`` | 选择内置模型或外部模型 | ``model_path`` | 模型路径配置；``None`` 使用内置模型 |
| ``SyncConfig`` | 主循环、GUI刷新、缓存和压力—力匹配窗口 | 频率、15 ms窗口等 | 同步配置 |
| ``OutputConfig`` | 指定CSV输出目录 | ``save_dir`` | 输出配置 |
| ``GuiConfig`` | 控制窗口、历史长度、热图和区域显示 | GUI参数 | GUI配置 |
| ``TrainingConfig`` | 定义离线标定训练任务 | XY/Z CSV、模型类型、输出路径等 | 传给 ``train_model`` 的训练配置 |
| ``PlotConfig`` | 定义离线CSV绘图任务 | 文件、列、行范围、模式和输出路径 | 传给绘图API的配置 |

#### 训练与绘图

| API | 作用 | 主要输入 | 返回值或输出 |
| --- | --- | --- | --- |
| ``TrainingResult`` | 描述训练产物和评估结果 | 由 ``train_model`` 创建 | 模型路径、评估图、指标和写回信息 |
| ``train_model`` | 从XY和Z训练CSV拟合标定模型 | ``TrainingConfig`` | ``TrainingResult``；默认不修改输入CSV |
| ``PlotResult`` | 描述离线绘图产物 | 由绘图函数创建 | 图片、分析文件和处理信息 |
| ``plot_csv`` | 按CSV真实表头绘制指定列和范围 | ``PlotConfig`` | ``PlotResult`` |
| ``plot_full_analysis`` | 生成完整采集数据的综合分析图 | ``PlotConfig`` 或完整分析参数 | ``PlotResult`` |

### TangentialSample 字段

| 字段 | 类型/单位 | 含义 |
| --- | --- | --- |
| ``raw`` | ndarray，84 | 原始一维 ADC 数据副本 |
| ``matrix`` / ``raw_2d`` | ndarray，12×7 | 按阵列布局排列的 ADC |
| ``gradient`` | ndarray，12×7×2 | 每个压力单元的二维梯度 |
| ``minimum`` / ``min`` | float，ADC | 当前帧最小值 |
| ``maximum`` / ``max`` | float，ADC | 当前帧最大值 |
| ``total`` / ``sum`` / ``adc_sum`` | float，ADC | 84通道总和 |
| ``mean`` | float，ADC | 84通道均值 |
| ``cop_x`` / ``copX`` | float，cell | CoP列坐标；无效时可能为NaN |
| ``cop_y`` / ``copY`` | float，cell | CoP行坐标；无效时可能为NaN |
| ``angle`` | float，度 | 当前静态切向或滑移方向角 |
| ``dx``、``dy`` | float，cell | 中值滤波后的CoP相对origin偏移 |
| ``state`` | int | CoP状态：0未接触、1粗略、2精修完成 |
| ``calibrated_fx``、``calibrated_fy``、``calibrated_fz`` | float | 模型预测的三轴力；模型不可用时为NaN |
| ``calibrated_angle`` | float，度 | 标定Fx/Fy方向角；不可用时为NaN |
| ``request_seq`` | int | 压力请求序号；无元数据时为-1 |
| ``tx_t``、``rx_t`` | float，秒 | ``perf_counter`` 发送/合法响应接收时间 |
| ``latency_s`` | float，秒 | 单次压力请求响应延迟 |
| ``rel_ms`` | int，毫秒 | 相对首个合法压力帧的真实接收时间 |
| ``origin_x``、``origin_y`` | float或None，cell | 当前静态CoP基准 |
| ``contact`` | bool | 全局CoP状态机是否接触 |
| ``display_contact`` | bool | GUI是否应显示接触；region模式可与contact不同 |
| ``refined`` | bool | 全局CoP二次精修是否完成 |
| ``region_mask`` | ndarray或None | 每个cell对应的区域编号 |
| ``regions`` | list[dict] | 每个区域的CoP、delta、坐标和状态 |
| ``centroid`` | (x, y)或None | 当前压力区域形心 |
| ``motion_state`` | ``TangentialMotionState`` | NO_CONTACT、STICK或SLIP |
| ``is_slipping`` | bool | 当前帧是否正在滑移 |
| ``slip_motion_distance`` | float，cell | 滑移短窗首尾CoP位移 |
| ``slip_confidence`` | float，0..1 | 斑块平移确认后的余弦相关置信度 |
| ``angle_vector_magnitude`` | float，cell | angle所用向量模长；STICK为静态delta，SLIP为EMA滑移向量 |

不要用 ``rel_ms`` 反推请求发送时间；需要分析设备延迟时使用
``tx_t``、``rx_t`` 和 ``latency_s``。

### 按功能分类配置

~~~python
from tangential import (
    FullApplicationConfig, PressureConfig, ForceConfig, ProcessingConfig,
    SlipConfig, SyncConfig, OutputConfig,
)

config = FullApplicationConfig(
    pressure=PressureConfig(port="/dev/ttyUSB0", target_hz=200),
    force=ForceConfig(port="/dev/ttyUSB1", target_hz=200),
    processing=ProcessingConfig(
        region_mode="full",
        slip=SlipConfig(),
    ),
    sync=SyncConfig(max_time_diff_s=0.015),
    output=OutputConfig(save_dir="./data"),
)
~~~

## 配置与环境变量

用户不需要、也不建议直接修改安装包中的 ``config.py``。推荐在代码中创建配置
对象，或在启动前设置 ``TANGENTIAL_*`` 环境变量。配置对象在应用启动前统一
校验，非法端口、频率、超时、队列或阈值会抛出 ``ValueError``。

配置优先级：

```text
CLI显式参数 > 代码显式传入的配置对象 > TANGENTIAL_*环境变量 > 默认值
```

### 设备配置

| 配置 | 字段（默认值） | 用户用途 |
| --- | --- | --- |
| ``PressureConfig`` | ``port=/dev/ttyUSB0``、``baudrate=921600``、``target_hz=200``、``response_timeout_s=0.050``、``frame_queue_size=256``、``startup_timeout_s=2.0`` | 压力设备端口和请求—响应轮询；实际帧率受设备响应速度影响 |
| ``ForceConfig`` | ``enabled=True``、``port=/dev/ttyUSB1``、``baudrate=460800``、``target_hz=200``、``response_timeout_s=0.050``、``frame_queue_size=256``、``startup_timeout_s=2.0``、``zero_sample_count=10``、``zero_timeout_s=1.0``、``rezero_timeout_s=1.0`` | 六维力设备、启动软件校零和运行期重新归零；不需要力传感器时设置 ``enabled=False`` |

``target_hz`` 是请求上限，不代表设备一定能返回同样帧率。增大队列可吸收短时
消费延迟，但不能修复串口断开或持续处理过慢。

### CoP与处理配置

| 配置 | 字段（默认值） | 用户用途 |
| --- | --- | --- |
| ``CopConfig`` | ``rows=12``、``cols=7``、``total_threshold_factor=3.0``、``pixel_threshold_factor=5.0``、``collect_frames=10``、``stability_frames=5``、``reset_at_frame=0``、``refine_cnt=10``、``refine_distance=0.1``、``merge_ratio=0.6``、``region_match_dist=5.0``、``region_min_area=4``、``region_peak_ratio=1.0``、``region_peak_dist=3`` | 动态阈值、接触稳定、origin精修和区域跟踪。标准硬件固定12×7，不要修改rows/cols |
| ``ProcessingConfig`` | ``cal_dim=3D``、``region_mode=full``、``median_window=5``、``refine_rezero_force=True``、``cop=CopConfig()``、``slip=SlipConfig()`` | 选择1D/2D/3D标定、full/region/both模式、CoP偏移滤波以及滑移配置 |
| ``CalibrationConfig`` | ``model_path=None`` | ``None`` 加载SDK内置模型；传路径时加载外部 ``fit_coefs.bin`` |

常用调节原则：增大 ``collect_frames`` 会延长启动背景学习；增大
``stability_frames`` 会降低短时卸载导致的接触复位；增大 ``refine_cnt`` 或减小
``refine_distance`` 会让origin精修更严格，但完成更慢。

### 同步、输出与GUI配置

| 配置 | 字段（默认值） | 用户用途 |
| --- | --- | --- |
| ``SyncConfig`` | ``target_fps=100``、``plot_fps=60``、``max_time_diff_s=0.015``、``timing_log_interval_s=1.0``、``buffer_size=500`` | 主循环、GUI刷新上限、压力—力一对一匹配窗口和时间戳缓存 |
| ``OutputConfig`` | ``save_dir=当前目录/data`` | CSV及退出分析图保存目录 |
| ``GuiConfig`` | ``window_title=RealTime``、``timer_interval_ms=10``、``history_size=100``、``error_history_size=100``、``max_region_arrows=8``、``heat_vmax=500``、``window_width=1900``、``window_height=1050``、8色 ``region_palette`` | 窗口标题、刷新定时、历史长度、热图范围和区域颜色 |

``max_time_diff_s`` 只用于压力帧和六维力帧匹配，不控制压力请求频率。

### 训练和绘图配置

| 配置 | 字段（默认值） | 用户用途 |
| --- | --- | --- |
| ``TrainingConfig`` | 必填 ``xy_csv``、``z_csv``；``output_model=fit_coefs.bin``、``output_plot=fit_report.png``、``dim=1``、``poly_order=3``、``fx=sym_log``、``fy=sym_log``、``fz=exp``、``valid_only=True``、``split_sign=True``、``one_on_one=True``、``write_back=None``、``force=False`` | 选择训练数据、模型形式和输出；默认不回写输入CSV |
| ``PlotConfig`` | ``files=None``、``directory=当前目录/data``、``columns=(Fy_cal, delta_Force_Y)``、``rows=None``、``x_column=rel_ms``、``title=None``、``save_path=None``、``error_ref=None``、``mode=plot``、``highlight_valid=True``、``show_annotations=True``、``force_min=0.2`` | 选择文件、列、行范围、横轴、绘图模式和保存位置 |

``FullApplicationConfig`` 将 pressure、force、processing、calibration、sync、
output 和 gui 七类配置组合为完整应用的唯一配置入口。

可用环境变量示例：

~~~bash
export TANGENTIAL_PRESSURE_PORT=/dev/ttyUSB0
export TANGENTIAL_FORCE_PORT=/dev/ttyUSB1
export TANGENTIAL_MAX_TIME_DIFF_S=0.015
export TANGENTIAL_DATA_DIR=./data
export TANGENTIAL_MODEL_PATH=/path/to/fit_coefs.bin
~~~

协议帧头、CRC、固定 12×7/84 通道布局、固定 108 列 CSV 和设备帧长度属于协议不变量，不通过配置修改。

## 滑移检测

0.4.0 增加了可复用的 ``SlipDetector``。它不改变 108 列 CSV，不修改
``fit_coefs.bin``，也不改变标定模型输入；结果只出现在 ``TangentialSample``、
终端输出和实时 GUI 中。每个处理器/传感器实例拥有独立 detector，双传感器
不会共享滑移历史。

### SlipConfig全部可调参数

距离和搜索半径的单位都是压力阵列 cell。参数修改只影响之后创建的处理器；
运行中的 detector 不会自动读取新配置。

| 参数 | 默认值；范围/单位 | 调大后的影响 | 调小后的影响 | 环境变量 |
| --- | --- | --- | --- | --- |
| ``enabled`` | ``True``；布尔 | 开启滑移状态机 | ``False`` 时接触帧保持STICK，不判定SLIP | ``TANGENTIAL_SLIP_ENABLED`` |
| ``window_frames`` | ``5``；整数≥2，帧 | 短窗更平滑、抗噪更强，但检测更慢 | 响应更快，但更容易受单帧抖动影响 | ``TANGENTIAL_SLIP_WINDOW_FRAMES`` |
| ``enter_distance`` | ``0.20``；≥0，cell | 需要更明显运动才进入SLIP，误报更少 | 微小运动更容易触发，灵敏但可能误报 | ``TANGENTIAL_SLIP_ENTER_DISTANCE`` |
| ``exit_distance`` | ``0.05``；≥0，cell | 更容易把残余小运动视为停止，较快退出 | 必须更稳定才退出，SLIP保持更久 | ``TANGENTIAL_SLIP_EXIT_DISTANCE`` |
| ``reanchor_distance`` | ``1.0``；≥``enter_distance``，cell | 斑块相关不足时的大位移兜底更严格 | CoP累计位移更容易绕过斑块确认触发SLIP | ``TANGENTIAL_SLIP_REANCHOR_DISTANCE`` |
| ``enter_frames`` | ``3``；整数>0，窗口 | 连续证据要求更久，降低瞬态误报、增加延迟 | 更快进入SLIP，但抗噪降低 | ``TANGENTIAL_SLIP_ENTER_FRAMES`` |
| ``exit_frames`` | ``8``；整数>0，窗口 | 短暂停顿不会立即退出，方向保持更稳 | 停止后更快回到STICK | ``TANGENTIAL_SLIP_EXIT_FRAMES`` |
| ``direction_smoothing`` | ``0.35``；``(0,1]``，EMA系数 | 更跟随最新方向，转向快但更抖 | 方向更平滑，但转向响应更慢 | ``TANGENTIAL_SLIP_DIRECTION_SMOOTHING`` |
| ``patch_search_radius`` | ``2``；整数≥0，cell | 可识别更大斑块平移，但计算量和误匹配机会增加 | 计算更少，但可能漏掉大步移动 | ``TANGENTIAL_SLIP_PATCH_SEARCH_RADIUS`` |
| ``patch_min_correlation`` | ``0.75``；``[0,1]`` | 压力形状匹配更严格，误报更少 | 更容忍形变和噪声，但误确认概率增加 | ``TANGENTIAL_SLIP_PATCH_MIN_CORRELATION`` |
| ``patch_min_improvement`` | ``0.03``；≥0 | 非零平移必须明显优于零平移，判定更保守 | 更容易接受微弱平移证据 | ``TANGENTIAL_SLIP_PATCH_MIN_IMPROVEMENT`` |
| ``angle_deadband`` | ``0.1``；≥0，cell | 更多小方向向量被置为0，箭头更稳定 | 更容易显示弱方向，也更容易抖动 | ``TANGENTIAL_SLIP_ANGLE_DEADBAND`` |

关键关系：``reanchor_distance`` 不能小于 ``enter_distance``；
``direction_smoothing`` 必须大于0且不超过1；相关阈值必须在0到1之间。
进入SLIP既需要短窗位移达到 ``enter_distance``，又需要斑块平移确认或
``reanchor_distance`` 大位移兜底，并连续满足 ``enter_frames`` 次。

### 三套调参预设

默认配置适合先验证硬件：

```python
from tangential import ProcessingConfig, SlipConfig

processing = ProcessingConfig(slip=SlipConfig())
```

保守抗噪配置适合振动、接触形变较大或误报较多的场景：

```python
processing = ProcessingConfig(slip=SlipConfig(
    window_frames=7,
    enter_distance=0.35,
    exit_distance=0.04,
    reanchor_distance=1.50,
    enter_frames=4,
    exit_frames=10,
    direction_smoothing=0.25,
    patch_search_radius=2,
    patch_min_correlation=0.85,
    patch_min_improvement=0.06,
    angle_deadband=0.15,
))
```

灵敏快速配置适合低噪声、短促滑移和希望较低延迟的场景：

```python
processing = ProcessingConfig(slip=SlipConfig(
    window_frames=3,
    enter_distance=0.10,
    exit_distance=0.08,
    reanchor_distance=0.60,
    enter_frames=2,
    exit_frames=4,
    direction_smoothing=0.55,
    patch_search_radius=2,
    patch_min_correlation=0.65,
    patch_min_improvement=0.015,
    angle_deadband=0.05,
))
```

把预设传给完整应用：

```python
from tangential import FullApplicationConfig, run_application

run_application(FullApplicationConfig(processing=processing))
```

环境变量适合命令行部署。例如提高进入距离和连续窗口数：

```bash
export TANGENTIAL_SLIP_ENTER_DISTANCE=0.35
export TANGENTIAL_SLIP_ENTER_FRAMES=4
tangential app --pressure-port /dev/ttyUSB0
```

算法按以下顺序工作：

- 对当前压力斑块按总压力归一化，保存 ``window_frames`` 帧的 CoP 和斑块历史。
- 比较短窗首尾 CoP 位移；在 ``±patch_search_radius`` 范围做零填充平移，使用
  余弦相关，并要求相对零平移提升 ``patch_min_improvement``。
- CoP 位移达到 ``enter_distance`` 且斑块确认，或相对 detector anchor 达到
  ``reanchor_distance`` 的大位移兜底时，连续 ``enter_frames`` 个窗口进入 SLIP。
- SLIP 期间用 ``direction_smoothing`` 做运动方向 EMA；短窗位移连续低于
  ``exit_distance`` 达到 ``exit_frames`` 后退出，当前位置重新锁定全局静摩擦
  origin，退出帧角度为 0。
- ``angle_deadband`` 以下的方向向量输出 0。无接触或 CoP 不可用时完整 reset，
  状态为 ``NO_CONTACT``；接触但未滑移为 ``STICK``。

实时 GUI 的两个方向面板语义不同：

- ``Direction`` 的红色 PZT 箭头保持固定显示长度，只表达 ``sample.angle``
  的方向，不表达位移或力的大小。
- ``Pressure Snapshot`` 的红色 PZT 箭头同样沿 ``sample.angle``，但长度来自
  ``sample.angle_vector_magnitude``：STICK 时是静态 CoP delta 模长，SLIP 时
  是 EMA 滑移向量模长。显示时乘 0.5 并限制到 0.65，避免超出面板。
- ``Pressure Snapshot`` 蓝色箭头仍使用六维力 Fx/Fy 的模长；Pressure Table
  中实际 origin、当前 CoP、delta 和区域几何不受上述显示缩放影响。

### 滑移方向与 CoP 重锚定时序

这里没有异步、延时执行的“重置 CoP 任务”。滑移状态、方向更新和重锚定都在
当前压力帧的单帧处理过程中同步完成：

- 一旦进入 ``SLIP``，detector 把短窗运动向量保存到独立的方向缓存，并在后续
  帧使用 EMA 更新；方向不再依赖静态 CoP origin，因此更新 origin 不会把正在
  输出的滑移方向清零。
- ``SLIP`` 期间 detector 的内部 anchor 每帧跟随当前 CoP，避免累计位移无限
  增长；这只是几次数值赋值，不需要等待硬件或后台线程。
- 只有短窗位移连续 ``exit_frames`` 次低于 ``exit_distance``，才确认滑移结束。
  确认退出的同一帧会同步重锚定 ``PRSensorAngle`` 的全局 origin，清空滑移方向，
  并按设计输出 0°；下一帧从新 origin 计算静态切向方向。

需要注意：如果一次运动短到不足以形成
``window_frames`` 短窗并连续满足 ``enter_frames`` 次进入证据，它可能不会进入
``SLIP``；这是抗噪滞回带来的检测下限，不是 CoP 重锚定速度造成的。当前 API
表示“当前是否正在滑移”，不会在滑移结束后继续保持历史方向；需要记录事件方向
时，应在 ``is_slipping`` 为真时由上层保存 ``sample.angle``。

公开调用示例：

~~~python
from tangential import ProcessingConfig, SlipDetector, TangentialMotionState

config = ProcessingConfig()
detector = SlipDetector(config.slip, rows=12, cols=7)
result = detector.update(matrix, cop_x, cop_y, contact=True, ready=True)
if result.motion_state is TangentialMotionState.SLIP:
    print(result.motion_distance, result.confidence)
~~~

``region``/``both`` 处理模式只支持用整帧聚合 CoP 做全局滑移检测，不对每个
region 单独检测滑移。多接触点的 CoP 可能互相抵消，因此全局检测在多接触、
接触区域分离或形变明显时可能低估运动；需要 per-region 滑移时应另行设计
独立跟踪算法，不能把当前全局结果误认为每个 region 的结果。

### 配置何时生效

- 推荐在代码中显式传入 ``PressureConfig``、``SlipConfig`` 等对象，参数最清楚。
- 纯命令行部署可以设置 ``TANGENTIAL_*`` 环境变量，再启动新进程。
- 配置在对象创建和应用启动时读取；修改环境变量后必须重新创建配置并重启程序。
- 不要为了调参修改安装包中的 ``config.py``，升级或重新安装会覆盖此类修改。
- 多传感器场景应分别创建 ``config_a`` 和 ``config_b``，不要共享同一个可变
  ``FullApplicationConfig`` 实例。

## 数据和时序不变量

- 压力和六维力均以 200 Hz 为请求目标，单请求在途；响应较慢时实际频率自然下降，不插值、不重复请求补发。
- 合法压力帧解析完成后立即记录真实 rx_t。rel_ms 和 delta_ms 基于真实压力接收时间，不由 GUI 刷新或 CSV 写入节拍生成。
- 压力帧是主顺序；每个合法压力帧最多处理和保存一次。
- 六维力帧最多匹配一次，匹配窗口为 abs(force_t - press_t) <= 0.015 秒。
- 力通道不可用时，压力帧仍逐行保存，力和同步字段写 NaN。
- 双传感器模式下，压力帧超过15 ms仍未匹配时不写CSV，但仍推进CoP状态机并更新GUI；这是当前数据语义，不能在文档或调用方中误写成NaN行。
- 压力设备必需，六维力设备可选；启动校零和运行期重新归零使用普通力数据帧，不发送额外置零命令。
- CSV 由唯一的 TABLE_CSV_HEADER 和 build_csv_row 生成，保持 108 列和既有模型输出。

## 用户二次开发

二次开发应从 ``tangential`` 顶层导入公共 API，不要依赖未在公共 API 表中列出
的内部模块路径。常见方式包括：

- 用 ``TangentialSensor`` 编写自己的实时控制或数据分析循环。
- 用 ``TangentialFrameProcessor`` 处理已有84通道压力帧。
- 用 ``SlipDetector`` 将滑移结果接入机器人控制状态机。
- 用 ``run_application`` 和分类配置快速启动标准GUI。
- 用 ``train_model``、``plot_csv`` 和 ``plot_full_analysis`` 构建离线流程。

每只压力传感器必须创建独立的 ``TangentialSensor`` 或处理器实例，不能共享
``PRSensorAngle`` 或 ``SlipDetector``，否则接触origin、历史窗口和滑移状态会
相互污染。

## 常见故障

| 现象 | 常见原因 | 处理方法 |
| --- | --- | --- |
| ``Permission denied`` | 当前用户没有串口权限 | 将用户加入 ``dialout``，注销后重新登录 |
| ``No such file or directory`` | 端口路径错误或设备重插后编号改变 | 运行 ``python -m serial.tools.list_ports -v``，优先使用 ``/dev/serial/by-id`` |
| 压力传感器连接失败 | 端口错误、被占用或设备无响应 | 用 ``fuser PORT`` 检查占用，压力设备失败时应用会退出且不创建空CSV |
| 六维力不可用但GUI仍运行 | 力端口连接或启动校零失败 | 这是预期降级行为；压力继续采集，力字段写NaN |
| 实际帧率低于200 Hz | 设备响应、USB调度或系统负载超过5 ms | 查看每秒时序日志中的响应延迟、超时和跳过周期；不要用插值伪造频率 |
| ``delta_ms`` 有少量抖动 | Linux/Python和串口不是硬实时系统 | 使用真实 ``rx_t`` 分析；关闭不必要负载并检查设备响应延迟 |
| 滑移误报较多 | 阈值过灵敏或压力斑块噪声较大 | 使用保守预设，优先增大 ``enter_distance``、``enter_frames`` 和相关阈值 |
| 短促滑移检测不到 | 窗口或进入滞回过长 | 使用灵敏预设，减小 ``window_frames``、``enter_distance`` 或 ``enter_frames`` |
| 滑移停止后仍保持SLIP | ``exit_frames`` 较大或 ``exit_distance`` 较小 | 增大 ``exit_distance`` 或减小 ``exit_frames`` |
| 两路传感器互相影响 | 两个配置指向同一物理端口或共享状态对象 | 使用不同 ``by-id`` 路径并为A/B分别创建完整配置和会话 |
| Bash提示 ``unexpected token newline`` | 原样输入了带尖括号的占位符 | 用真实端口路径替换示例名称，不要输入 ``<`` 或 ``>`` |

如果问题仍存在，先分别运行单传感器最小示例验证两只设备，再运行完整或双路GUI，
这样可以区分硬件通信、算法配置和GUI负载问题。
