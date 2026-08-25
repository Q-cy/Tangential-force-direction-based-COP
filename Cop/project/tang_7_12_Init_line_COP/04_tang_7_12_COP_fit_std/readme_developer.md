# Tangential Sensor SDK 0.4.0 开发者完整指南

本文是 `readme.md` 的严格内容超集：完整保留用户安装、运行、硬件策略、示例、API、配置、滑移、训练、绘图、wheel、二次开发和故障排查内容，并在后半部分追加内部实现与项目维护说明。本文可以独立阅读，用户不需要跳转到其他文档才能完成使用或二次开发。

本文分为两部分：第一部分面向SDK用户和二次开发者；第二部分面向需要阅读源码、修改协议或算法、运行测试和构建wheel的项目维护者。

## 第一部分：用户使用与二次开发

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器，提供 CoP、角度、梯度、切向力标定、实时 GUI、固定 108 列 CSV 和离线分析。

本文面向安装和使用 SDK 的用户，介绍硬件连接、命令行、Python API、参数配置、滑移检测、CSV 行为和常见故障。用户可以安装 wheel，也可以在获得源码后直接运行。

### 系统要求与安装

当前 wheel 适用于 Linux x86_64 和 CPython 3.11。压力传感器是必需设备，六维力传感器是可选设备；默认端口分别为 ``/dev/ttyUSB0`` 和 ``/dev/ttyUSB1``。

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

### 从源码运行

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

``minimal`` 需要压力传感器；``full`` 需要完整 GUI 依赖。安装 wheel 的用户不需要设置 ``PYTHONPATH``。

### 同时连接两个压力传感器

双传感器示例模块为 ``tangential.examples.dual_sensor``。它启动一个 Qt 应用和两个完整窗口；A/B 各自执行压力采集、CoP、角度、梯度、标定、实时曲线、压力表、完整 108 列 CSV，并在退出时生成各自的分析图。不再是终端摘要循环。默认只连接压力传感器；只有显式提供对应 ``--force-port-a`` 或 ``--force-port-b`` 才启用六维力通道，避免两路同时打开默认 ``/dev/ttyUSB1``。

#### 第1步：插入设备并识别两个端口

插入两只压力传感器后运行：

~~~bash
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
~~~

优先选择 ``/dev/serial/by-id/`` 下两个不同的设备路径，因为它们通常不会随重插或重启改变。若该目录不存在，再根据 ``serial.tools.list_ports`` 的输出确认两只设备分别对应哪个 ``/dev/ttyUSB*`` 或 ``/dev/ttyACM*``。

本机当前如果没有列出任何端口，说明设备尚未接入、USB未识别或串口驱动尚未创建，不能继续启动示例。

#### 第2步：设置本次运行使用的端口

把下面两行中的 ``DEVICE_A_ID`` 和 ``DEVICE_B_ID`` 替换为第1步看到的真实文件名，再执行后续命令。例如，真实路径可能类似 ``/dev/serial/by-id/usb-FTDI_A1-if00-port0`` 和 ``/dev/serial/by-id/usb-FTDI_B2-if00-port0``；下面的名称只是示意：

~~~bash
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
printf 'A=%s\nB=%s\n' "$PORT_A" "$PORT_B"
~~~

不要把 ``<sensor-a>`` 或 ``<sensor-b>`` 原样输入命令，也不要把它们写进变量赋值。Bash会把尖括号解释成输入/输出重定向符号，从而产生 ``syntax error near unexpected token 'newline'``。只有替换成第1步实际查到的路径后，才能继续执行 ``printf`` 和启动命令。

如果没有 ``by-id`` 路径，且已经确认端口映射，可以改成：

~~~bash
PORT_A=/dev/ttyUSB0
PORT_B=/dev/ttyUSB1
~~~

两个变量必须对应不同物理设备。示例会解析符号链接，并在打开串口前拒绝两个变量最终指向同一物理串口。

#### 第3步：检查权限和端口占用

~~~bash
ls -l "$PORT_A" "$PORT_B"
groups
fuser "$PORT_A" "$PORT_B"
~~~

- ``ls`` 必须能找到两个路径。
- 当前用户通常需要属于 ``dialout`` 组；若没有权限，可执行 ``sudo usermod -aG dialout "$USER"``，然后注销并重新登录。
- ``fuser`` 没有输出通常表示端口空闲；若显示进程号，应先关闭正在占用传感器的旧采集程序，不要让两个程序同时读取同一串口。

#### 第4步：启动双传感器示例

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

也可以分别覆盖输出目录：``--save-dir-a`` 和 ``--save-dir-b``；模型使用 ``--model MODEL_PATH``，或分别使用 ``--model-a``、``--model-b``。

查看全部参数：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor --help
~~~

#### 第5步：确认输出并停止

运行后会出现两个窗口，标题分别包含 ``Sensor A`` 和 ``Sensor B``。每个窗口都包含压力/六维力实时曲线、方向和幅值、12×7 压力表、CoP 标记、梯度箭头以及状态显示；状态变化不会覆盖 A/B 标签。每路目录会保存一个完整 108 列 CSV，退出后还会保存 ``full_analysis_cop_<n>.png``。

按 ``Ctrl+C`` 或关闭 Qt 应用时，两路会同时停止；任一路采集线程异常都会报告具体的 A/B，并联动安全关闭另一路。不要直接拔线代替正常退出。

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

更直接的公共入口是 ``run_dual_application(config_a, config_b)``。每一路都有独立串口、采集进程、IPC队列、读取线程、缓存、CoP状态机、标定处理器、停止事件、GUI和输出目录；一个设备的读取超时不会占用另一个设备的串口消费者。软件状态互相隔离，但 USB 控制器带宽、CPU 调度和供电仍是共享硬件资源，实际帧率应分别验收。

#### 常见错误

<table>
<thead>
<tr>
<th style="min-width:180px">现象</th>
<th>原因</th>
<th>处理方法</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal">Bash报告<code>unexpected token newline</code></td>
<td style="white-space:normal">原样复制了带<code>&lt;...&gt;</code> 的占位符</td>
<td style="white-space:normal">按第1、2步设置真实<code>PORT_A</code>/<code>PORT_B</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>No such file or directory</code></td>
<td style="white-space:normal">设备未连接或端口名已变化</td>
<td style="white-space:normal">重新运行<code>serial.tools.list_ports -v</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>Permission denied</code></td>
<td style="white-space:normal">当前用户没有串口权限</td>
<td style="white-space:normal">加入<code>dialout</code> 后重新登录</td>
</tr>
<tr>
<td style="white-space:normal">提示两个传感器使用同一物理串口</td>
<td style="white-space:normal">两个路径相同，或两个符号链接指向同一设备</td>
<td style="white-space:normal">为A、B选择两个不同设备路径</td>
</tr>
<tr>
<td style="white-space:normal">某一路窗口持续无数据</td>
<td style="white-space:normal">端口选错、设备无响应、供电或USB带宽异常</td>
<td style="white-space:normal">单独运行最小示例验证该端口，再检查USB连接</td>
</tr>
</tbody>
</table>

### 命令行

安装 wheel 后使用统一命令：

~~~bash
tangential --version
tangential example
tangential app
tangential plot --help
tangential fit --help
~~~

#### 最小压力采集

~~~bash
tangential example \
  --pressure-port /dev/ttyUSB0 \
  --timeout 0.1
~~~

终端每帧显示 12×7 原始 ADC、min、max、sum、mean、CoP X/Y 和角度。此路径不启动六维力、CSV 或 Qt GUI。

#### 完整采集

~~~bash
tangential app \
  --pressure-port /dev/ttyUSB0 \
  --force-port /dev/ttyUSB1 \
  --save-dir ./data \
  --max-time-diff-ms 15
~~~

压力传感器是必需设备；连接失败时程序退出且不创建空 CSV。六维力传感器是可选设备；连接或普通数据帧校零失败时降级为压力模式，力相关列写入 NaN。两路设备由独立采集进程读取，父进程按真实接收时间完成匹配和 CSV 保存。

普通 ``app`` 命令使用 ``ForceConfig.enabled=True`` 的默认配置，因此没有提供 ``--force-port`` 时仍会尝试打开默认 ``/dev/ttyUSB1``；如果该设备不存在或校零失败，程序会关闭力通道并继续压力采集。需要明确只采集压力时，在 Python API 中传入 ``ForceConfig(enabled=False)``，或设置 ``TANGENTIAL_FORCE_ENABLED=false`` 后再启动。

#### 双路完整采集

~~~bash
tangential dual \
  --port-a /dev/serial/by-id/PRESSURE_A \
  --port-b /dev/serial/by-id/PRESSURE_B \
  --save-dir ./data/dual
~~~

该命令显示两个完整 GUI 窗口，默认把 CSV 和退出分析图分别保存到 ``./data/dual/sensor_a``、``./data/dual/sensor_b``。只有显式增加 ``--force-port-a``、``--force-port-b`` 才启用对应六维力通道；两个力端口也必须是不同物理设备。

#### 离线绘图

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

#### 离线训练

~~~bash
tangential fit \
  --xy-csv ./data/fx_fy.csv \
  --z-csv ./data/fz.csv \
  --output-model ./fit_coefs.bin \
  --output-plot ./fit_report.png
~~~

默认只生成模型和评估图，不修改输入 CSV。只有明确提供 --write-back PATH 才会写回；目标已存在时必须额外提供 --force。

### Python API

所有稳定公共名称都可以直接从 ``tangential`` 导入。普通采集优先使用 ``TangentialSensor``；需要完整 GUI 时使用 ``run_application`` 或 ``run_dual_application``。``PressureSensor``、``PRSensorAngle`` 和 ``TangentialFrameProcessor`` 面向需要自行编排数据流的高级用户。

#### 最小采集示例

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

#### 完整应用示例

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

#### 公共 API 总览

以下表格覆盖当前 ``tangential.__all__`` 的全部33个公共名称。

##### 采集、处理与终端输出

<table>
<thead>
<tr>
<th style="min-width:180px">API</th>
<th>作用</th>
<th>主要输入</th>
<th>返回值或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TangentialSensor</code></td>
<td style="white-space:normal">串口采集 → 解码 → 单帧处理 → TangentialSample</td>
<td style="white-space:normal"><code>PressureConfig</code>、可选 <code>ProcessingConfig</code>、模型路径</td>
<td style="white-space:normal"><code>read(timeout_s)</code> 返回 <code>TangentialSample</code> 或 <code>None</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialSensorAPI</code></td>
<td style="white-space:normal">压力设备生命周期 → 调用 TangentialFrameProcessor → TangentialSample</td>
<td style="white-space:normal">传感器/工厂注入、压力配置、处理配置</td>
<td style="white-space:normal">逐帧<code>TangentialSample</code>；<code>close()</code> 释放设备</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialSample</code></td>
<td style="white-space:normal">原始帧与处理结果 → 统一封装 → 单帧数据对象</td>
<td style="white-space:normal">通常由处理器创建，不建议用户手工构造</td>
<td style="white-space:normal">ADC、CoP、角度、标定、时间戳、区域和滑移字段</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrameProcessor</code></td>
<td style="white-space:normal">84通道 ADC → CoP/梯度/滑移/标定 → TangentialSample</td>
<td style="white-space:normal"><code>raw</code>、<code>ProcessingConfig</code>、可选标定模型</td>
<td style="white-space:normal"><code>process()</code> 返回 <code>TangentialSample</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>FixedTerminalRenderer</code></td>
<td style="white-space:normal">TangentialSample → 固定布局文本 → 原位刷新终端</td>
<td style="white-space:normal">输出流、<code>TangentialSample</code></td>
<td style="white-space:normal"><code>render()</code> 写入并刷新终端，同时返回文本</td>
</tr>
<tr>
<td style="white-space:normal"><code>format_terminal_sample</code></td>
<td style="white-space:normal">TangentialSample → 12×7矩阵与指标 → str</td>
<td style="white-space:normal"><code>TangentialSample</code></td>
<td style="white-space:normal"><code>str</code></td>
</tr>
</tbody>
</table>

##### 算法、模型与底层压力驱动

<table>
<thead>
<tr>
<th style="min-width:180px">API</th>
<th>作用</th>
<th>主要输入</th>
<th>返回值或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>FitCalibrationModel</code></td>
<td style="white-space:normal">fit_coefs.bin → dx/dy/ADC总和 → Fx/Fy/Fz</td>
<td style="white-space:normal"><code>from_default()</code> 或 <code>from_path(path)</code>；<code>predict(dx, dy, total, cal_dim="3D")</code></td>
<td style="white-space:normal">三个标定力分量及模型状态</td>
</tr>
<tr>
<td style="white-space:normal"><code>PRSensorAngle</code></td>
<td style="white-space:normal">84通道 ADC → 阈值/接触/CoP/区域计算 → 角度与梯度</td>
<td style="white-space:normal">84通道 ADC、<code>CopConfig</code></td>
<td style="white-space:normal">CoP/角度/梯度/状态；高级用户使用</td>
</tr>
<tr>
<td style="white-space:normal"><code>PressureSensor</code></td>
<td style="white-space:normal">发送请求 → 接收/校验/解码压力帧 → 84通道 ADC与时间戳</td>
<td style="white-space:normal">串口、周期、超时、队列等</td>
<td style="white-space:normal"><code>read_frame()</code> 帧字典、<code>decode()</code> 84通道数据</td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipDetector</code></td>
<td style="white-space:normal">压力矩阵与CoP序列 → 位移/相关性/滞回判断 → SlipResult</td>
<td style="white-space:normal">压力矩阵、CoP、接触/ready状态、<code>SlipConfig</code></td>
<td style="white-space:normal"><code>SlipResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipResult</code></td>
<td style="white-space:normal">滑移检测结果 → 封装状态/方向/置信度 → 不可变对象</td>
<td style="white-space:normal">由<code>SlipDetector</code> 生成</td>
<td style="white-space:normal">状态、位移、置信度、方向、斑块平移和重锚定标志</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialMotionState</code></td>
<td style="white-space:normal">接触与滑移判断 → NO_CONTACT/STICK/SLIP</td>
<td style="white-space:normal">无</td>
<td style="white-space:normal"><code>NO_CONTACT</code>、<code>STICK</code>、<code>SLIP</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>compute_vector_angle</code></td>
<td style="white-space:normal">二维向量 x/y → atan2角度 → [0, 360)度</td>
<td style="white-space:normal"><code>x</code>、<code>y</code></td>
<td style="white-space:normal"><code>[0, 360)</code> 度角</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle_difference</code></td>
<td style="white-space:normal">两个方向角 → 环绕差计算 → [0, 180]度</td>
<td style="white-space:normal">两个角度</td>
<td style="white-space:normal"><code>[0, 180]</code> 度差</td>
</tr>
</tbody>
</table>

##### 完整应用入口

<table>
<thead>
<tr>
<th style="min-width:180px">API</th>
<th>作用</th>
<th>主要输入</th>
<th>返回值或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>run_application</code></td>
<td style="white-space:normal">FullApplicationConfig → 设备/会话/GUI/CSV → 单路完整应用</td>
<td style="white-space:normal"><code>FullApplicationConfig</code></td>
<td style="white-space:normal">阻塞运行至窗口关闭；正常退出返回0并输出CSV/分析图</td>
</tr>
<tr>
<td style="white-space:normal"><code>run_dual_application</code></td>
<td style="white-space:normal">两份独立配置 → 两路会话与GUI → 两套CSV/分析图</td>
<td style="white-space:normal"><code>config_a</code>、<code>config_b</code></td>
<td style="white-space:normal">正常退出返回0，并生成两个GUI、两套CSV和分析图</td>
</tr>
</tbody>
</table>

##### 配置对象

<table>
<thead>
<tr>
<th style="min-width:180px">API</th>
<th>作用</th>
<th>主要输入</th>
<th>返回值或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>FullApplicationConfig</code></td>
<td style="white-space:normal">分类配置对象 → 组合与校验 → 完整应用配置</td>
<td style="white-space:normal">pressure、force、processing、calibration、sync、output、gui</td>
<td style="white-space:normal">经校验的完整配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>PressureConfig</code></td>
<td style="white-space:normal">端口/频率/超时/队列 → 校验 → 压力采集配置</td>
<td style="white-space:normal">端口及轮询参数</td>
<td style="white-space:normal">压力设备配置；<code>period_s</code> 返回周期</td>
</tr>
<tr>
<td style="white-space:normal"><code>ForceConfig</code></td>
<td style="white-space:normal">启用开关/端口/频率/校零参数 → 校验 → 六维力配置</td>
<td style="white-space:normal">端口、频率、校零样本/超时</td>
<td style="white-space:normal">六维力设备配置；<code>enabled=False</code> 禁用通道</td>
</tr>
<tr>
<td style="white-space:normal"><code>CopConfig</code></td>
<td style="white-space:normal">阈值/帧数/区域参数 → 校验 → CoP算法配置</td>
<td style="white-space:normal">各阈值、帧数和区域参数</td>
<td style="white-space:normal">CoP配置；<code>as_kwargs()</code> 返回算法参数字典</td>
</tr>
<tr>
<td style="white-space:normal"><code>ProcessingConfig</code></td>
<td style="white-space:normal">处理模式/滤波/CoP/滑移参数 → 组合 → 单帧处理配置</td>
<td style="white-space:normal">cal_dim、region_mode、cop、slip等</td>
<td style="white-space:normal">单帧处理配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipConfig</code></td>
<td style="white-space:normal">窗口/阈值/滞回/平滑参数 → 校验 → 滑移检测配置</td>
<td style="white-space:normal">12项滑移参数</td>
<td style="white-space:normal">经<code>validate()</code> 校验的滑移配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>CalibrationConfig</code></td>
<td style="white-space:normal">模型路径 → 选择内置或外部模型 → 标定配置</td>
<td style="white-space:normal"><code>model_path</code></td>
<td style="white-space:normal">模型路径配置；<code>None</code> 使用内置模型</td>
</tr>
<tr>
<td style="white-space:normal"><code>SyncConfig</code></td>
<td style="white-space:normal">循环/刷新/缓存/匹配窗口 → 校验 → 同步配置</td>
<td style="white-space:normal">频率、15 ms窗口等</td>
<td style="white-space:normal">同步配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>OutputConfig</code></td>
<td style="white-space:normal">保存目录 → 指定CSV与分析图位置 → 输出配置</td>
<td style="white-space:normal"><code>save_dir</code></td>
<td style="white-space:normal">输出配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>GuiConfig</code></td>
<td style="white-space:normal">窗口/历史/热图/区域参数 → 校验 → GUI配置</td>
<td style="white-space:normal">GUI参数</td>
<td style="white-space:normal">GUI配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>TrainingConfig</code></td>
<td style="white-space:normal">训练数据/模型类型/输出选项 → 统一封装 → train_model输入</td>
<td style="white-space:normal">XY/Z CSV、模型类型、输出路径等</td>
<td style="white-space:normal">传给<code>train_model</code> 的训练配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotConfig</code></td>
<td style="white-space:normal">CSV/列/行范围/输出选项 → 统一封装 → plot_csv输入</td>
<td style="white-space:normal">文件、列、行范围、模式和输出路径</td>
<td style="white-space:normal">传给绘图API的配置</td>
</tr>
</tbody>
</table>

##### 训练与绘图

<table>
<thead>
<tr>
<th style="min-width:180px">API</th>
<th>作用</th>
<th>主要输入</th>
<th>返回值或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TrainingResult</code></td>
<td style="white-space:normal">训练产物与指标 → 统一封装 → 训练结果对象</td>
<td style="white-space:normal">由<code>train_model</code> 创建</td>
<td style="white-space:normal">模型路径、评估图、指标和写回信息</td>
</tr>
<tr>
<td style="white-space:normal"><code>train_model</code></td>
<td style="white-space:normal">TrainingConfig → 读取CSV并拟合 → 模型/评估结果</td>
<td style="white-space:normal"><code>TrainingConfig</code></td>
<td style="white-space:normal"><code>TrainingResult</code>；默认不修改输入CSV</td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotResult</code></td>
<td style="white-space:normal">绘图产物与处理信息 → 统一封装 → 绘图结果对象</td>
<td style="white-space:normal">由绘图函数创建</td>
<td style="white-space:normal">图片、分析文件和处理信息</td>
</tr>
<tr>
<td style="white-space:normal"><code>plot_csv</code></td>
<td style="white-space:normal">PlotConfig → 按表头读取并绘图 → PlotResult</td>
<td style="white-space:normal"><code>PlotConfig</code></td>
<td style="white-space:normal"><code>PlotResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>plot_full_analysis</code></td>
<td style="white-space:normal">完整CSV与绘图配置 → 综合分析 → PlotResult</td>
<td style="white-space:normal"><code>PlotConfig</code> 或 CSV 路径</td>
<td style="white-space:normal"><code>PlotResult</code></td>
</tr>
</tbody>
</table>

#### TangentialSample 字段

<table>
<thead>
<tr>
<th style="min-width:180px">字段</th>
<th>类型/单位</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>raw</code></td>
<td style="white-space:normal">ndarray，84</td>
<td style="white-space:normal">原始一维 ADC 数据副本</td>
</tr>
<tr>
<td style="white-space:normal"><code>matrix</code> / <code>raw_2d</code></td>
<td style="white-space:normal">ndarray，12×7</td>
<td style="white-space:normal">按阵列布局排列的 ADC</td>
</tr>
<tr>
<td style="white-space:normal"><code>gradient</code></td>
<td style="white-space:normal">ndarray，12×7×2</td>
<td style="white-space:normal">每个压力单元的二维梯度</td>
</tr>
<tr>
<td style="white-space:normal"><code>minimum</code> / <code>min</code></td>
<td style="white-space:normal">float，ADC</td>
<td style="white-space:normal">当前帧最小值</td>
</tr>
<tr>
<td style="white-space:normal"><code>maximum</code> / <code>max</code></td>
<td style="white-space:normal">float，ADC</td>
<td style="white-space:normal">当前帧最大值</td>
</tr>
<tr>
<td style="white-space:normal"><code>total</code> / <code>sum</code> / <code>adc_sum</code></td>
<td style="white-space:normal">float，ADC</td>
<td style="white-space:normal">84通道总和</td>
</tr>
<tr>
<td style="white-space:normal"><code>mean</code></td>
<td style="white-space:normal">float，ADC</td>
<td style="white-space:normal">84通道均值</td>
</tr>
<tr>
<td style="white-space:normal"><code>cop_x</code> / <code>copX</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">CoP列坐标；无效时可能为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>cop_y</code> / <code>copY</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">CoP行坐标；无效时可能为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle</code></td>
<td style="white-space:normal">float，度</td>
<td style="white-space:normal">当前静态切向或滑移方向角</td>
</tr>
<tr>
<td style="white-space:normal"><code>dx</code>、<code>dy</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">中值滤波后的CoP相对origin偏移</td>
</tr>
<tr>
<td style="white-space:normal"><code>state</code></td>
<td style="white-space:normal">int</td>
<td style="white-space:normal">CoP状态：0未接触、1粗略、2精修完成</td>
</tr>
<tr>
<td style="white-space:normal"><code>calibrated_fx</code>、<code>calibrated_fy</code>、<code>calibrated_fz</code></td>
<td style="white-space:normal">float</td>
<td style="white-space:normal">模型预测的三轴力；模型不可用时为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>calibrated_angle</code></td>
<td style="white-space:normal">float，度</td>
<td style="white-space:normal">标定Fx/Fy方向角；不可用时为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>request_seq</code></td>
<td style="white-space:normal">int</td>
<td style="white-space:normal">压力请求序号；无元数据时为-1</td>
</tr>
<tr>
<td style="white-space:normal"><code>tx_t</code>、<code>rx_t</code></td>
<td style="white-space:normal">float，秒</td>
<td style="white-space:normal"><code>perf_counter</code> 发送/合法响应接收时间</td>
</tr>
<tr>
<td style="white-space:normal"><code>latency_s</code></td>
<td style="white-space:normal">float，秒</td>
<td style="white-space:normal">单次压力请求响应延迟</td>
</tr>
<tr>
<td style="white-space:normal"><code>rel_ms</code></td>
<td style="white-space:normal">int，毫秒</td>
<td style="white-space:normal">相对首个合法压力帧的真实接收时间</td>
</tr>
<tr>
<td style="white-space:normal"><code>origin_x</code>、<code>origin_y</code></td>
<td style="white-space:normal">float或None，cell</td>
<td style="white-space:normal">当前静态CoP基准</td>
</tr>
<tr>
<td style="white-space:normal"><code>contact</code></td>
<td style="white-space:normal">bool</td>
<td style="white-space:normal">全局CoP状态机是否接触</td>
</tr>
<tr>
<td style="white-space:normal"><code>display_contact</code></td>
<td style="white-space:normal">bool</td>
<td style="white-space:normal">GUI是否应显示接触；region模式可与contact不同</td>
</tr>
<tr>
<td style="white-space:normal"><code>refined</code></td>
<td style="white-space:normal">bool</td>
<td style="white-space:normal">全局CoP二次精修是否完成</td>
</tr>
<tr>
<td style="white-space:normal"><code>region_mask</code></td>
<td style="white-space:normal">ndarray或None</td>
<td style="white-space:normal">每个cell对应的区域编号</td>
</tr>
<tr>
<td style="white-space:normal"><code>regions</code></td>
<td style="white-space:normal">list[dict]</td>
<td style="white-space:normal">每个区域的CoP、delta、坐标和状态</td>
</tr>
<tr>
<td style="white-space:normal"><code>centroid</code></td>
<td style="white-space:normal">(x, y)或None</td>
<td style="white-space:normal">当前压力区域形心</td>
</tr>
<tr>
<td style="white-space:normal"><code>motion_state</code></td>
<td style="white-space:normal"><code>TangentialMotionState</code></td>
<td style="white-space:normal">NO_CONTACT、STICK或SLIP</td>
</tr>
<tr>
<td style="white-space:normal"><code>is_slipping</code></td>
<td style="white-space:normal">bool</td>
<td style="white-space:normal">当前帧是否正在滑移</td>
</tr>
<tr>
<td style="white-space:normal"><code>slip_motion_distance</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">滑移短窗首尾CoP位移</td>
</tr>
<tr>
<td style="white-space:normal"><code>slip_confidence</code></td>
<td style="white-space:normal">float，0..1</td>
<td style="white-space:normal">斑块平移确认后的余弦相关置信度</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle_vector_magnitude</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">angle所用向量模长；STICK为静态delta，SLIP为EMA滑移向量</td>
</tr>
</tbody>
</table>

不要用 ``rel_ms`` 反推请求发送时间；需要分析设备延迟时使用 ``tx_t``、``rx_t`` 和 ``latency_s``。

#### 按功能分类配置

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

### 配置与环境变量

用户不需要、也不建议直接修改安装包中的 ``config.py``。推荐在代码中创建配置对象，或在启动前设置 ``TANGENTIAL_*`` 环境变量。配置对象在应用启动前统一校验，非法端口、频率、超时、队列或阈值会抛出 ``ValueError``。

配置优先级：

```text
CLI显式参数 > 代码显式传入的配置对象 > TANGENTIAL_*环境变量 > 默认值
```

#### 设备配置

<table>
<thead>
<tr>
<th style="min-width:180px">配置</th>
<th>字段（默认值）</th>
<th>用户用途</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>PressureConfig</code></td>
<td style="white-space:normal"><code>port=/dev/ttyUSB0</code>、<code>baudrate=921600</code>、<code>target_hz=200</code>、<code>response_timeout_s=0.050</code>、<code>frame_queue_size=256</code>、<code>startup_timeout_s=2.0</code></td>
<td style="white-space:normal">压力设备端口和请求—响应轮询；实际帧率受设备响应速度影响</td>
</tr>
<tr>
<td style="white-space:normal"><code>ForceConfig</code></td>
<td style="white-space:normal"><code>enabled=True</code>、<code>port=/dev/ttyUSB1</code>、<code>baudrate=460800</code>、<code>target_hz=200</code>、<code>response_timeout_s=0.050</code>、<code>frame_queue_size=256</code>、<code>startup_timeout_s=2.0</code>、<code>zero_sample_count=10</code>、<code>zero_timeout_s=1.0</code>、<code>rezero_timeout_s=1.0</code></td>
<td style="white-space:normal">六维力设备、启动软件校零和运行期重新归零；不需要力传感器时设置<code>enabled=False</code></td>
</tr>
</tbody>
</table>

``target_hz`` 是请求上限，不代表设备一定能返回同样帧率。增大队列可吸收短时消费延迟，但不能修复串口断开或持续处理过慢。

#### CoP与处理配置

<table>
<thead>
<tr>
<th style="min-width:180px">配置</th>
<th>字段（默认值）</th>
<th>用户用途</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>CopConfig</code></td>
<td style="white-space:normal"><code>rows=12</code>、<code>cols=7</code>、<code>total_threshold_factor=3.0</code>、<code>pixel_threshold_factor=5.0</code>、<code>collect_frames=10</code>、<code>stability_frames=5</code>、<code>reset_at_frame=0</code>、<code>refine_cnt=10</code>、<code>refine_distance=0.1</code>、<code>merge_ratio=0.6</code>、<code>region_match_dist=5.0</code>、<code>region_min_area=4</code>、<code>region_peak_ratio=1.0</code>、<code>region_peak_dist=3</code></td>
<td style="white-space:normal">动态阈值、接触稳定、origin精修和区域跟踪。标准硬件固定12×7，不要修改rows/cols</td>
</tr>
<tr>
<td style="white-space:normal"><code>ProcessingConfig</code></td>
<td style="white-space:normal"><code>cal_dim=3D</code>、<code>region_mode=full</code>、<code>median_window=5</code>、<code>refine_rezero_force=True</code>、<code>cop=CopConfig()</code>、<code>slip=SlipConfig()</code></td>
<td style="white-space:normal">选择1D/2D/3D标定、full/region/both模式、CoP偏移滤波以及滑移配置</td>
</tr>
<tr>
<td style="white-space:normal"><code>CalibrationConfig</code></td>
<td style="white-space:normal"><code>model_path=None</code></td>
<td style="white-space:normal"><code>None</code> 加载SDK内置模型；传路径时加载外部 <code>fit_coefs.bin</code></td>
</tr>
</tbody>
</table>

常用调节原则：增大 ``collect_frames`` 会延长启动背景学习；增大 ``stability_frames`` 会降低短时卸载导致的接触复位；增大 ``refine_cnt`` 或减小 ``refine_distance`` 会让origin精修更严格，但完成更慢。

#### 同步、输出与GUI配置

<table>
<thead>
<tr>
<th style="min-width:180px">配置</th>
<th>字段（默认值）</th>
<th>用户用途</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>SyncConfig</code></td>
<td style="white-space:normal"><code>target_fps=100</code>、<code>plot_fps=60</code>、<code>max_time_diff_s=0.015</code>、<code>timing_log_interval_s=1.0</code>、<code>buffer_size=500</code></td>
<td style="white-space:normal">主循环、GUI刷新上限、压力—力一对一匹配窗口和时间戳缓存</td>
</tr>
<tr>
<td style="white-space:normal"><code>OutputConfig</code></td>
<td style="white-space:normal"><code>save_dir=当前目录/data</code></td>
<td style="white-space:normal">CSV及退出分析图保存目录</td>
</tr>
<tr>
<td style="white-space:normal"><code>GuiConfig</code></td>
<td style="white-space:normal"><code>window_title=RealTime</code>、<code>timer_interval_ms=10</code>、<code>history_size=100</code>、<code>error_history_size=100</code>、<code>max_region_arrows=8</code>、<code>heat_vmax=500</code>、<code>window_width=1900</code>、<code>window_height=1050</code>、8色 <code>region_palette</code></td>
<td style="white-space:normal">窗口标题、刷新定时、历史长度、热图范围和区域颜色</td>
</tr>
</tbody>
</table>

``max_time_diff_s`` 只用于压力帧和六维力帧匹配，不控制压力请求频率。

#### 训练和绘图配置

<table>
<thead>
<tr>
<th style="min-width:180px">配置</th>
<th>字段（默认值）</th>
<th>用户用途</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TrainingConfig</code></td>
<td style="white-space:normal">必填<code>xy_csv</code>、<code>z_csv</code>；<code>output_model=fit_coefs.bin</code>、<code>output_plot=fit_report.png</code>、<code>dim=1</code>、<code>poly_order=3</code>、<code>fx=sym_log</code>、<code>fy=sym_log</code>、<code>fz=exp</code>、<code>valid_only=True</code>、<code>split_sign=True</code>、<code>one_on_one=True</code>、<code>write_back=None</code>、<code>force=False</code></td>
<td style="white-space:normal">选择训练数据、模型形式和输出；默认不回写输入CSV</td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotConfig</code></td>
<td style="white-space:normal"><code>files=None</code>、<code>directory=当前目录/data</code>、<code>columns=(Fy_cal, delta_Force_Y)</code>、<code>rows=None</code>、<code>x_column=rel_ms</code>、<code>title=None</code>、<code>save_path=None</code>、<code>error_ref=None</code>、<code>mode=plot</code>、<code>highlight_valid=True</code>、<code>show_annotations=True</code>、<code>force_min=0.2</code></td>
<td style="white-space:normal">选择文件、列、行范围、横轴、绘图模式和保存位置</td>
</tr>
</tbody>
</table>

``FullApplicationConfig`` 将 pressure、force、processing、calibration、sync、 output 和 gui 七类配置组合为完整应用的唯一配置入口。

可用环境变量示例：

~~~bash
export TANGENTIAL_PRESSURE_PORT=/dev/ttyUSB0
export TANGENTIAL_FORCE_PORT=/dev/ttyUSB1
export TANGENTIAL_MAX_TIME_DIFF_S=0.015
export TANGENTIAL_DATA_DIR=./data
export TANGENTIAL_MODEL_PATH=/path/to/fit_coefs.bin
~~~

协议帧头、CRC、固定 12×7/84 通道布局、固定 108 列 CSV 和设备帧长度属于协议不变量，不通过配置修改。

### 滑移检测

0.4.0 增加了可复用的 ``SlipDetector``。它不改变 108 列 CSV，不修改 ``fit_coefs.bin``，也不改变标定模型输入；结果只出现在 ``TangentialSample``、终端输出和实时 GUI 中。每个处理器/传感器实例拥有独立 detector，双传感器不会共享滑移历史。

#### SlipConfig全部可调参数

距离和搜索半径的单位都是压力阵列 cell。参数修改只影响之后创建的处理器；运行中的 detector 不会自动读取新配置。

<table>
<thead>
<tr>
<th style="min-width:180px">参数</th>
<th>默认值；范围/单位</th>
<th>调大后的影响</th>
<th>调小后的影响</th>
<th>环境变量</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>enabled</code></td>
<td style="white-space:normal"><code>True</code>；布尔</td>
<td style="white-space:normal">开启滑移状态机</td>
<td style="white-space:normal"><code>False</code> 时接触帧保持STICK，不判定SLIP</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_ENABLED</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>window_frames</code></td>
<td style="white-space:normal"><code>5</code>；整数≥2，帧</td>
<td style="white-space:normal">短窗更平滑、抗噪更强，但检测更慢</td>
<td style="white-space:normal">响应更快，但更容易受单帧抖动影响</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_WINDOW_FRAMES</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>enter_distance</code></td>
<td style="white-space:normal"><code>0.20</code>；≥0，cell</td>
<td style="white-space:normal">需要更明显运动才进入SLIP，误报更少</td>
<td style="white-space:normal">微小运动更容易触发，灵敏但可能误报</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_ENTER_DISTANCE</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>exit_distance</code></td>
<td style="white-space:normal"><code>0.05</code>；≥0，cell</td>
<td style="white-space:normal">更容易把残余小运动视为停止，较快退出</td>
<td style="white-space:normal">必须更稳定才退出，SLIP保持更久</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_EXIT_DISTANCE</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>reanchor_distance</code></td>
<td style="white-space:normal"><code>1.0</code>；≥<code>enter_distance</code>，cell</td>
<td style="white-space:normal">斑块相关不足时的大位移兜底更严格</td>
<td style="white-space:normal">CoP累计位移更容易绕过斑块确认触发SLIP</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_REANCHOR_DISTANCE</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>enter_frames</code></td>
<td style="white-space:normal"><code>3</code>；整数>0，窗口</td>
<td style="white-space:normal">连续证据要求更久，降低瞬态误报、增加延迟</td>
<td style="white-space:normal">更快进入SLIP，但抗噪降低</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_ENTER_FRAMES</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>exit_frames</code></td>
<td style="white-space:normal"><code>8</code>；整数>0，窗口</td>
<td style="white-space:normal">短暂停顿不会立即退出，方向保持更稳</td>
<td style="white-space:normal">停止后更快回到STICK</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_EXIT_FRAMES</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>direction_smoothing</code></td>
<td style="white-space:normal"><code>0.35</code>；<code>(0,1]</code>，EMA系数</td>
<td style="white-space:normal">更跟随最新方向，转向快但更抖</td>
<td style="white-space:normal">方向更平滑，但转向响应更慢</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_DIRECTION_SMOOTHING</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>patch_search_radius</code></td>
<td style="white-space:normal"><code>2</code>；整数≥0，cell</td>
<td style="white-space:normal">可识别更大斑块平移，但计算量和误匹配机会增加</td>
<td style="white-space:normal">计算更少，但可能漏掉大步移动</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_PATCH_SEARCH_RADIUS</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>patch_min_correlation</code></td>
<td style="white-space:normal"><code>0.75</code>；<code>[0,1]</code></td>
<td style="white-space:normal">压力形状匹配更严格，误报更少</td>
<td style="white-space:normal">更容忍形变和噪声，但误确认概率增加</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_PATCH_MIN_CORRELATION</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>patch_min_improvement</code></td>
<td style="white-space:normal"><code>0.03</code>；≥0</td>
<td style="white-space:normal">非零平移必须明显优于零平移，判定更保守</td>
<td style="white-space:normal">更容易接受微弱平移证据</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_PATCH_MIN_IMPROVEMENT</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>angle_deadband</code></td>
<td style="white-space:normal"><code>0.1</code>；≥0，cell</td>
<td style="white-space:normal">更多小方向向量被置为0，箭头更稳定</td>
<td style="white-space:normal">更容易显示弱方向，也更容易抖动</td>
<td style="white-space:normal"><code>TANGENTIAL_SLIP_ANGLE_DEADBAND</code></td>
</tr>
</tbody>
</table>

关键关系：``reanchor_distance`` 不能小于 ``enter_distance``； ``direction_smoothing`` 必须大于0且不超过1；相关阈值必须在0到1之间。进入SLIP既需要短窗位移达到 ``enter_distance``，又需要斑块平移确认或 ``reanchor_distance`` 大位移兜底，并连续满足 ``enter_frames`` 次。

#### 三套调参预设

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
- 比较短窗首尾 CoP 位移；在 ``±patch_search_radius`` 范围做零填充平移，使用余弦相关，并要求相对零平移提升 ``patch_min_improvement``。
- CoP 位移达到 ``enter_distance`` 且斑块确认，或相对 detector anchor 达到 ``reanchor_distance`` 的大位移兜底时，连续 ``enter_frames`` 个窗口进入 SLIP。
- SLIP 期间用 ``direction_smoothing`` 做运动方向 EMA；短窗位移连续低于 ``exit_distance`` 达到 ``exit_frames`` 后退出，当前位置重新锁定全局静摩擦 origin，退出帧角度为 0。
- ``angle_deadband`` 以下的方向向量输出 0。无接触或 CoP 不可用时完整 reset，状态为 ``NO_CONTACT``；接触但未滑移为 ``STICK``。

实时 GUI 的两个方向面板语义不同：

- ``Direction`` 的红色 PZT 箭头保持固定显示长度，只表达 ``sample.angle`` 的方向，不表达位移或力的大小。
- ``Pressure Snapshot`` 的红色 PZT 箭头同样沿 ``sample.angle``，但长度来自 ``sample.angle_vector_magnitude``：STICK 时是静态 CoP delta 模长，SLIP 时是 EMA 滑移向量模长。显示时乘 0.5 并限制到 0.65，避免超出面板。
- ``Pressure Snapshot`` 蓝色箭头仍使用六维力 Fx/Fy 的模长；Pressure Table 中实际 origin、当前 CoP、delta 和区域几何不受上述显示缩放影响。

#### 滑移方向与 CoP 重锚定时序

这里没有异步、延时执行的“重置 CoP 任务”。滑移状态、方向更新和重锚定都在当前压力帧的单帧处理过程中同步完成：

- 一旦进入 ``SLIP``，detector 把短窗运动向量保存到独立的方向缓存，并在后续帧使用 EMA 更新；方向不再依赖静态 CoP origin，因此更新 origin 不会把正在输出的滑移方向清零。
- ``SLIP`` 期间 detector 的内部 anchor 每帧跟随当前 CoP，避免累计位移无限增长；这只是几次数值赋值，不需要等待硬件或后台线程。
- 只有短窗位移连续 ``exit_frames`` 次低于 ``exit_distance``，才确认滑移结束。确认退出的同一帧会同步重锚定 ``PRSensorAngle`` 的全局 origin，清空滑移方向，并按设计输出 0°；下一帧从新 origin 计算静态切向方向。

需要注意：如果一次运动短到不足以形成 ``window_frames`` 短窗并连续满足 ``enter_frames`` 次进入证据，它可能不会进入 ``SLIP``；这是抗噪滞回带来的检测下限，不是 CoP 重锚定速度造成的。当前 API 表示“当前是否正在滑移”，不会在滑移结束后继续保持历史方向；需要记录事件方向时，应在 ``is_slipping`` 为真时由上层保存 ``sample.angle``。

公开调用示例：

~~~python
from tangential import ProcessingConfig, SlipDetector, TangentialMotionState

config = ProcessingConfig()
detector = SlipDetector(config.slip, rows=12, cols=7)
result = detector.update(matrix, cop_x, cop_y, contact=True, ready=True)
if result.motion_state is TangentialMotionState.SLIP:
    print(result.motion_distance, result.confidence)
~~~

``region``/``both`` 处理模式只支持用整帧聚合 CoP 做全局滑移检测，不对每个 region 单独检测滑移。多接触点的 CoP 可能互相抵消，因此全局检测在多接触、接触区域分离或形变明显时可能低估运动；需要 per-region 滑移时应另行设计独立跟踪算法，不能把当前全局结果误认为每个 region 的结果。

#### 配置何时生效

- 推荐在代码中显式传入 ``PressureConfig``、``SlipConfig`` 等对象，参数最清楚。
- 纯命令行部署可以设置 ``TANGENTIAL_*`` 环境变量，再启动新进程。
- 配置在对象创建和应用启动时读取；修改环境变量后必须重新创建配置并重启程序。
- 不要为了调参修改安装包中的 ``config.py``，升级或重新安装会覆盖此类修改。
- 多传感器场景应分别创建 ``config_a`` 和 ``config_b``，不要共享同一个可变 ``FullApplicationConfig`` 实例。

### 数据和时序不变量

- 压力和六维力均以 200 Hz 为请求目标，单请求在途；响应较慢时实际频率自然下降，不插值、不重复请求补发。
- 合法压力帧解析完成后立即记录真实 rx_t。rel_ms 和 delta_ms 基于真实压力接收时间，不由 GUI 刷新或 CSV 写入节拍生成。
- 压力帧是主顺序；每个合法压力帧最多处理和保存一次。
- 六维力帧最多匹配一次，匹配窗口为 abs(force_t - press_t) <= 0.015 秒。
- 力通道不可用时，压力帧仍逐行保存，力和同步字段写 NaN。
- 双传感器模式下，压力帧超过15 ms仍未匹配时不写CSV，但仍推进CoP状态机并更新GUI；这是当前数据语义，不能在文档或调用方中误写成NaN行。
- 压力设备必需，六维力设备可选；启动校零和运行期重新归零使用普通力数据帧，不发送额外置零命令。
- CSV 由唯一的 TABLE_CSV_HEADER 和 build_csv_row 生成，保持 108 列和既有模型输出。

### 用户二次开发

二次开发应从 ``tangential`` 顶层导入公共 API，不要依赖未在公共 API 表中列出的内部模块路径。常见方式包括：

- 用 ``TangentialSensor`` 编写自己的实时控制或数据分析循环。
- 用 ``TangentialFrameProcessor`` 处理已有84通道压力帧。
- 用 ``SlipDetector`` 将滑移结果接入机器人控制状态机。
- 用 ``run_application`` 和分类配置快速启动标准GUI。
- 用 ``train_model``、``plot_csv`` 和 ``plot_full_analysis`` 构建离线流程。

每只压力传感器必须创建独立的 ``TangentialSensor`` 或处理器实例，不能共享 ``PRSensorAngle`` 或 ``SlipDetector``，否则接触origin、历史窗口和滑移状态会相互污染。

### 常见故障

<table>
<thead>
<tr>
<th style="min-width:180px">现象</th>
<th>常见原因</th>
<th>处理方法</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>Permission denied</code></td>
<td style="white-space:normal">当前用户没有串口权限</td>
<td style="white-space:normal">将用户加入<code>dialout</code>，注销后重新登录</td>
</tr>
<tr>
<td style="white-space:normal"><code>No such file or directory</code></td>
<td style="white-space:normal">端口路径错误或设备重插后编号改变</td>
<td style="white-space:normal">运行<code>python -m serial.tools.list_ports -v</code>，优先使用 <code>/dev/serial/by-id</code></td>
</tr>
<tr>
<td style="white-space:normal">压力传感器连接失败</td>
<td style="white-space:normal">端口错误、被占用或设备无响应</td>
<td style="white-space:normal">用<code>fuser PORT</code> 检查占用，压力设备失败时应用会退出且不创建空CSV</td>
</tr>
<tr>
<td style="white-space:normal">六维力不可用但GUI仍运行</td>
<td style="white-space:normal">力端口连接或启动校零失败</td>
<td style="white-space:normal">这是预期降级行为；压力继续采集，力字段写NaN</td>
</tr>
<tr>
<td style="white-space:normal">实际帧率低于200 Hz</td>
<td style="white-space:normal">设备响应、USB调度或系统负载超过5 ms</td>
<td style="white-space:normal">查看每秒时序日志中的响应延迟、超时和跳过周期；不要用插值伪造频率</td>
</tr>
<tr>
<td style="white-space:normal"><code>delta_ms</code> 有少量抖动</td>
<td style="white-space:normal">Linux/Python和串口不是硬实时系统</td>
<td style="white-space:normal">使用真实<code>rx_t</code> 分析；关闭不必要负载并检查设备响应延迟</td>
</tr>
<tr>
<td style="white-space:normal">滑移误报较多</td>
<td style="white-space:normal">阈值过灵敏或压力斑块噪声较大</td>
<td style="white-space:normal">使用保守预设，优先增大<code>enter_distance</code>、<code>enter_frames</code> 和相关阈值</td>
</tr>
<tr>
<td style="white-space:normal">短促滑移检测不到</td>
<td style="white-space:normal">窗口或进入滞回过长</td>
<td style="white-space:normal">使用灵敏预设，减小<code>window_frames</code>、<code>enter_distance</code> 或 <code>enter_frames</code></td>
</tr>
<tr>
<td style="white-space:normal">滑移停止后仍保持SLIP</td>
<td style="white-space:normal"><code>exit_frames</code> 较大或 <code>exit_distance</code> 较小</td>
<td style="white-space:normal">增大<code>exit_distance</code> 或减小 <code>exit_frames</code></td>
</tr>
<tr>
<td style="white-space:normal">两路传感器互相影响</td>
<td style="white-space:normal">两个配置指向同一物理端口或共享状态对象</td>
<td style="white-space:normal">使用不同<code>by-id</code> 路径并为A/B分别创建完整配置和会话</td>
</tr>
<tr>
<td style="white-space:normal">Bash提示<code>unexpected token newline</code></td>
<td style="white-space:normal">原样输入了带尖括号的占位符</td>
<td style="white-space:normal">用真实端口路径替换示例名称，不要输入<code>&lt;</code> 或 <code>&gt;</code></td>
</tr>
</tbody>
</table>

如果问题仍存在，先分别运行单传感器最小示例验证两只设备，再运行完整或双路GUI，这样可以区分硬件通信、算法配置和GUI负载问题。

## 第二部分：内部实现与项目维护

本文第二部分面向维护 Tangential SDK 源码的开发者，说明系统为什么这样分层、数据如何穿过各模块、哪些状态必须相互隔离、修改某项功能时应进入哪个文件，以及怎样验证源码和 wheel 没有破坏既有协议与数据语义。第一部分已经包含安装、公共 API、命令行、用户二次开发和操作排障说明，本文可以脱离其他文档独立阅读。

### 1. 开发目标与不可破坏边界

项目处理 12×7 PZT 压力阵列与可选六维力传感器，完整功能包括串口请求—响应采集、时间戳、压力—力匹配、CoP、角度、梯度、区域、滑移、标定、108列CSV、实时GUI、离线训练和绘图。

维护时必须保留以下边界：

- `src/tangential/`是唯一正式源码，不在根目录恢复旧实现或创建第二套算法。
- 压力和六维力协议分别只在`sensors/pressure.py`与`sensors/force.py`实现。
- CoP、区域和梯度只在`processing/cop.py`实现，滑移只在`processing/slip.py`实现，标定只在`processing/calibration.py`实现。
- 108列CSV只能由`storage/csv.py`中的`TABLE_CSV_HEADER`与`build_csv_row()`生成。
- `fit_coefs.bin`是package resource，运行时通过`importlib.resources`加载。
- 压力与六维力的发送、接收和合法帧完成时间使用单调时钟；不得由GUI刷新时间、主循环周期或重采样伪造。
- 一个物理串口只能有一个消费者；启动校零和运行期重新归零都读取普通六维力帧，不向设备发送额外置零命令。
- 每只压力传感器必须拥有独立串口、进程、队列、缓存、处理器、CoP状态、滑移状态、GUI和输出目录。
- 源码模式必须可以直接运行；`.so`只是wheel构建产物，不能替代仓库中的`.py`源码。

维护时以实际代码为事实来源，判断顺序为：`pyproject.toml`决定版本、依赖、入口和资源声明；`setup.py`决定编译模块与wheel过滤；`src/tangential/`决定运行时行为；`tests/`把协议、时序、API、GUI、分发和模型回归固化为可执行契约；`readme.md`和本文只解释这些事实，不创建第二套默认值或算法定义。本文的第一部分与第二部分必须保持同一份实现事实。

### 2. 文档分层与维护边界

<table>
<thead>
<tr>
<th style="min-width:180px">文档/部分</th>
<th>读者</th>
<th>内容流程</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>readme.md</code></td>
<td style="white-space:normal">SDK用户与二次开发者</td>
<td style="white-space:normal">发行页面 → 安装 → 公共API → 配置 → 示例 → 常见故障</td>
</tr>
<tr>
<td style="white-space:normal"><code>第一部分</code></td>
<td style="white-space:normal">需要使用SDK或进行二次开发的用户</td>
<td style="white-space:normal">用户操作说明 → API调用 → 参数配置 → 采集与离线工具</td>
</tr>
<tr>
<td style="white-space:normal"><code>第二部分</code></td>
<td style="white-space:normal">项目维护者和需要修改内部实现的开发者</td>
<td style="white-space:normal">架构 → 数据流 → 修改路由 → 测试 → 构建与排障</td>
</tr>
<tr>
<td style="white-space:normal"><code>AGENTS.md</code></td>
<td style="white-space:normal">自动化开发代理</td>
<td style="white-space:normal">强制约束 → 不变量 → 验收命令 → Git安全</td>
</tr>
</tbody>
</table>

`pyproject.toml`继续使用`readme.md`作为发行说明，因此用户安装页面展示用户指南；本文同时包含同样完整的用户说明和内部维护说明，用户与维护者都可以在本文件内完成阅读。

### 3. 一分钟理解整个系统

最小压力API的数据流：

```text
压力串口
→ PressureSensor请求、收包、校验、时间戳
→ decode得到84通道ADC
→ TangentialFrameProcessor计算CoP、梯度、滑移和标定
→ TangentialSample
→ 用户循环或FixedTerminalRenderer
```

完整应用的数据流：

```text
压力采集进程 → PressureThread → TimestampedBuffer ┐
                                                     ├→ FullAcquisitionSession
六维力采集进程 → ForceThread → TimestampedBuffer ───┘
→ 压力帧按seq顺序推进处理器
→ 在15 ms窗口内一对一匹配六维力
→ build_csv_row生成108列
→ CSV与RealTimePlot
```

当`ForceConfig.enabled=False`，完整会话只建立压力缓存和压力消费线程，并为每个已保存压力帧把力相关字段写成`NaN`；当力通道启用但连接或普通帧校零失败时，会关闭力传感器并采用同样的压力模式降级路径。

完整应用入口的数据流：

```text
用户代码 / CLI / examples
→ application.py
→ FullApplicationRunner或DualApplicationRunner
→ acquisition_loop
→ FullAcquisitionSession
→ 设备、处理、同步、CSV、GUI和清理
```

### 4. 目录结构与职责

```text
04_tang_7_12_COP_fit_std/
├── readme.md                      用户与二次开发指南，发行页面使用
├── readme_developer.md            用户与内部维护合一的完整指南
├── AGENTS.md                      自动化修改约束
├── pyproject.toml                 包元数据、依赖、CLI入口和package data
├── setup.py                       Cython扩展清单与wheel源码过滤
├── MANIFEST.in                    源码分发清单
├── requirements.txt               完整开发与GUI环境依赖
├── src/tangential/
│   ├── __init__.py                顶层稳定公共API
│   ├── api.py                     可读公共API门面
│   ├── application.py             单路/双路完整应用公共入口
│   ├── cli.py                     统一命令解析与分发
│   ├── config.py                  分类配置、环境默认和校验
│   ├── acquisition/               顺序缓存与一次性匹配
│   ├── sensors/                   压力和六维力协议采集
│   ├── processing/                CoP、滑移和标定
│   ├── runtime/                   最小API、完整会话和同步编排
│   ├── storage/                   唯一CSV结构
│   ├── gui/                       PyQtGraph实时显示
│   ├── tools/                     离线训练与绘图
│   ├── examples/                  最小、完整和双路调用示例
│   └── resources/                 内置fit_coefs.bin
└── tests/                          协议、时序、API、GUI、分发和回归测试
```

目录层级按职责划分，不表示重要程度。`runtime`、`acquisition`、`sensors`、`processing`和`storage`在发布wheel中编译为多个同名CPython扩展；仓库中的`.py`仍是唯一维护源。

### 5. 推荐源码阅读顺序

第一次阅读不要从最长的`runtime/session.py`或`processing/cop.py`开始，建议按以下顺序建立心智模型：

1. `src/tangential/__init__.py`：先确认稳定公共名称。
2. `src/tangential/config.py`：理解设备、处理、同步、输出和GUI有哪些可调边界。
3. `src/tangential/examples/minimal.py`：观察最小用户循环。
4. `src/tangential/runtime/sensor.py`：理解`TangentialSample`、单帧处理器和高级传感器API。
5. `src/tangential/sensors/pressure.py`：理解压力请求、收包、校验、时间戳和独立进程。
6. `src/tangential/processing/cop.py`、`slip.py`、`calibration.py`：分别阅读CoP状态、滑移状态和模型预测。
7. `src/tangential/storage/csv.py`：确认完整应用最终写出的108列语义。
8. `src/tangential/acquisition/buffer.py`与`runtime/synchronization.py`：理解seq消费和一次性时间匹配。
9. `src/tangential/runtime/session.py`：把设备、处理、匹配、CSV、GUI和清理串起来。
10. `src/tangential/application.py`、`examples/full.py`、`examples/dual_sensor.py`与`cli.py`：理解公共入口如何复用完整会话。
11. `src/tangential/tools/training.py`与`plotting.py`：最后阅读离线工具和模型生产流程。
12. `setup.py`与`tests/test_distribution.py`：理解源码怎样变成独立wheel。

### 6. 分层职责总表

<table>
<thead>
<tr>
<th style="min-width:180px">模块</th>
<th>职责流程</th>
<th>明确不负责</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>config.py</code></td>
<td style="white-space:normal">环境默认/显式参数 → 分类dataclass → 启动前校验</td>
<td style="white-space:normal">协议常量、算法执行和硬件连接</td>
</tr>
<tr>
<td style="white-space:normal"><code>sensors/pressure.py</code></td>
<td style="white-space:normal">发送请求 → 接收并校验响应 → 168字节payload与真实时间戳</td>
<td style="white-space:normal">CoP、滑移、模型、CSV和GUI</td>
</tr>
<tr>
<td style="white-space:normal"><code>sensors/force.py</code></td>
<td style="white-space:normal">发送普通请求 → 校验28字节帧 → 六轴物理量与软件零点</td>
<td style="white-space:normal">压力匹配、CoP和CSV格式</td>
</tr>
<tr>
<td style="white-space:normal"><code>acquisition/buffer.py</code></td>
<td style="white-space:normal">时间戳数据 → 单调seq缓存 → 顺序消费或一次性最近匹配</td>
<td style="white-space:normal">串口读取和业务计算</td>
</tr>
<tr>
<td style="white-space:normal"><code>processing/cop.py</code></td>
<td style="white-space:normal">84通道ADC → 动态阈值/接触/origin/区域 → CoP、角度和梯度</td>
<td style="white-space:normal">串口、模型文件和CSV</td>
</tr>
<tr>
<td style="white-space:normal"><code>processing/slip.py</code></td>
<td style="white-space:normal">压力矩阵与CoP短窗 → 斑块相关和滞回 → STICK/SLIP与方向</td>
<td style="white-space:normal">修改CSV列和直接操作GUI</td>
</tr>
<tr>
<td style="white-space:normal"><code>processing/calibration.py</code></td>
<td style="white-space:normal">fit_coefs.bin → 输入特征与拟合类型 → Fx/Fy/Fz预测</td>
<td style="white-space:normal">训练数据拟合和硬件读取</td>
</tr>
<tr>
<td style="white-space:normal"><code>runtime/sensor.py</code></td>
<td style="white-space:normal">PressureSensor帧 → TangentialFrameProcessor → TangentialSample</td>
<td style="white-space:normal">完整Qt生命周期、六维力匹配和CSV</td>
</tr>
<tr>
<td style="white-space:normal"><code>runtime/session.py</code></td>
<td style="white-space:normal">压力缓存与可选力缓存 → 顺序处理与匹配 → CSV、GUI、统计和统一清理</td>
<td style="white-space:normal">复制协议、CoP公式和CSV字段定义</td>
</tr>
<tr>
<td style="white-space:normal"><code>storage/csv.py</code></td>
<td style="white-space:normal">压力样本与可选力帧 → 固定映射 → 108列CSV行</td>
<td style="white-space:normal">决定采集节拍和匹配策略</td>
</tr>
<tr>
<td style="white-space:normal"><code>gui/realtime.py</code></td>
<td style="white-space:normal">最新样本与历史序列 → PyQtGraph项目 → 实时窗口与分析图</td>
<td style="white-space:normal">读取串口和推进算法状态</td>
</tr>
<tr>
<td style="white-space:normal"><code>application.py</code></td>
<td style="white-space:normal">FullApplicationConfig → 惰性加载完整运行器 → 单路或双路应用</td>
<td style="white-space:normal">命令行解析和重复实现采集循环</td>
</tr>
<tr>
<td style="white-space:normal"><code>examples/</code></td>
<td style="white-space:normal">用户参数 → 构造公共配置 → 调用公共API</td>
<td style="white-space:normal">成为SDK内部依赖或保存第二套业务逻辑</td>
</tr>
<tr>
<td style="white-space:normal"><code>tools/</code></td>
<td style="white-space:normal">CSV与离线配置 → 训练或绘图 → 模型、指标和图片</td>
<td style="white-space:normal">实时采集和Qt事件循环</td>
</tr>
</tbody>
</table>

### 7. 配置系统

所有用户可调参数集中在`config.py`，调用方不得重新定义相同默认值。配置优先级为：

```text
CLI显式参数 > 显式配置对象 > TANGENTIAL_*环境默认 > dataclass内置默认
```

<table>
<thead>
<tr>
<th style="min-width:180px">配置类</th>
<th>输入 → 输出</th>
<th>主要消费者</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>PressureConfig</code></td>
<td style="white-space:normal">端口/波特率/频率/超时/队列 → 压力采集配置</td>
<td style="white-space:normal"><code>PressureSensor</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>ForceConfig</code></td>
<td style="white-space:normal">启用开关/端口/轮询/校零 → 六维力配置</td>
<td style="white-space:normal"><code>SixAxisForceSensor</code>与完整会话</td>
</tr>
<tr>
<td style="white-space:normal"><code>CopConfig</code></td>
<td style="white-space:normal">阈值/帧数/区域/精修 → CoP算法参数</td>
<td style="white-space:normal"><code>PRSensorAngle</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipConfig</code></td>
<td style="white-space:normal">窗口/距离/相关性/滞回/平滑 → 滑移状态机参数</td>
<td style="white-space:normal"><code>SlipDetector</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>ProcessingConfig</code></td>
<td style="white-space:normal">维度/区域模式/滤波/CoP/滑移 → 单帧处理配置</td>
<td style="white-space:normal"><code>TangentialFrameProcessor</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>CalibrationConfig</code></td>
<td style="white-space:normal">模型路径 → 内置或外部标定模型选择</td>
<td style="white-space:normal">完整会话</td>
</tr>
<tr>
<td style="white-space:normal"><code>SyncConfig</code></td>
<td style="white-space:normal">循环/绘图频率/15 ms窗口/缓存 → 同步配置</td>
<td style="white-space:normal">完整会话</td>
</tr>
<tr>
<td style="white-space:normal"><code>OutputConfig</code></td>
<td style="white-space:normal">保存目录 → CSV与分析图位置</td>
<td style="white-space:normal">完整会话与GUI</td>
</tr>
<tr>
<td style="white-space:normal"><code>GuiConfig</code></td>
<td style="white-space:normal">窗口/历史/色阶/箭头/配色 → 实时显示配置</td>
<td style="white-space:normal"><code>RealTimePlot</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>FullApplicationConfig</code></td>
<td style="white-space:normal">上述运行时配置 → 组合与校验 → 完整应用配置</td>
<td style="white-space:normal"><code>run_application</code>与<code>run_dual_application</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>TrainingConfig</code></td>
<td style="white-space:normal">数据/拟合类型/输出/写回选项 → 训练任务</td>
<td style="white-space:normal"><code>train_model</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotConfig</code></td>
<td style="white-space:normal">文件/列/范围/模式/输出 → 绘图任务</td>
<td style="white-space:normal"><code>plot_csv</code>与<code>plot_full_analysis</code></td>
</tr>
</tbody>
</table>

协议帧头、CRC、多字节顺序、12×7布局、84通道、固定帧长度和108列CSV属于协议或格式不变量，不放入配置。单次操作参数，例如`read(timeout_s)`的超时，也不作为全局配置。

新增配置时必须同步完成：在正确dataclass中增加字段和类型 → 如需环境默认则增加`TANGENTIAL_*`解析 → 在`validate()`或完整配置启动校验中拒绝非法值 → 把配置传到唯一消费者 → 更新用户文档和测试。不得只增加字段而不让实际运行路径读取它。

### 8. 压力采集实现

`PressureSensor`负责硬件通信，不负责业务算法。生产模式下父进程对象创建`spawn`采集子进程；子进程中的本地`PressureSensor`使用单一I/O线程执行请求、接收和解析，再通过IPC队列把帧与统计发回父进程。

每轮压力采集流程：

```text
记录cycle_start
→ 清空本轮串口输入/输出与解析缓存
→ 记录tx_t并发送14字节CMD_BYTES
→ 最长等待response_timeout_s
→ select等待可读并批量读取最多1024字节
→ 持久化缓存查找AA 55帧头
→ 读取小端payload_len
→ 验证长度、CRC、状态和168字节传感器payload
→ 记录rx_t与latency_s
→ 写入request_seq/tx_t/rx_t/latency_s/raw
→ 本轮不足period_s时等待剩余时间，超期则直接进入下一轮并计数
```

解析器支持分包、单轮粘包、前导噪声、错误长度、CRC错误和状态错误恢复。当前策略每轮只接受一个合法响应，轮末清空残留，避免上一轮晚到数据被错误归属到下一请求。

`read_frame()`返回168字节payload与时序元数据，`decode()`只执行84个little-endian `uint16`解码并保持设备原始线序。左右翻转、基线、增益、CoP和标定不属于该模块。

重要统计包括`requests`、`frames`、`response_timeouts`、`crc_errors`、`length_errors`、`status_errors`、`serial_read_errors`、`serial_write_errors`、`serial_flush_errors`、`queue_drops`和`schedule_skips`，以及最近发送间隔、接收间隔和响应延迟。目标200 Hz是请求上限；设备响应约6 ms时实际频率约166 Hz属于正常物理结果。

压力驱动的生产结构是“父进程`PressureSensor` → spawn子进程 → 子进程内本地`PressureSensor` → 单一压力I/O线程 → 串口”。父进程只从IPC帧队列读取，父进程的`PressureThread`负责解码并追加到`TimestampedBuffer`；因此业务处理、GUI刷新和CSV写入不会成为串口消费者。

### 9. 六维力采集与软件校零

`SixAxisForceSensor`与压力驱动采用相同的单请求在途思想和独立进程边界，但协议解析针对28字节六维力帧。合法普通帧完成后记录`request_seq`、`tx_t`、`rx_t`、`latency_s`和六轴数据。

六维力协议请求为`49 AA 0D 0A`，完整帧长度为28字节，帧头为`49 AA`，字节2到25为6个little-endian `float32`，每个值乘以9.8并保留两位小数，帧尾必须为`0D 0A`。生产结构是“父进程`SixAxisForceSensor` → spawn子进程 → 子进程内本地驱动 → 单一I/O线程 → 串口”；父进程`ForceThread`只消费已解析的普通帧。

启动校零流程：

```text
普通六维力帧
→ 收集zero_sample_count个新样本
→ 六轴逐项求均值
→ 保存软件zero_data
→ 后续普通帧减去zero_data
```

样本不足或超过`zero_timeout_s`时返回失败，完整会话关闭力通道并继续压力模式。运行期Fx/Fy重新归零仍从同一帧流读取，`schedule_rezero()`使用锁合并精修和卸载触发，避免多个任务同时修改零点或竞争串口。

力子进程只传递未扣零点的六轴物理量；父进程在`read_frame()`取帧时用`_zero_lock`复制当前`zero_data`并应用软件零点。这样重新归零线程只读取父进程缓存中的新帧，不会直接调用串口`read()`，且零点更新不会与当前帧解析产生部分读写。

### 10. 单帧处理、CoP与滑移

`TangentialFrameProcessor`是脱离串口也能使用的处理入口，适合回放CSV、自定义采集源或算法测试。每个实例都持有自己的`PRSensorAngle`、`SlipDetector`和dx/dy中值窗口，因此一个实例不能跨物理传感器共享。

单帧处理流程：

```text
84通道ADC
→ reshape为12×7
→ 更新动态总压与像素阈值
→ 按full/region/both模式计算CoP、origin、区域和梯度
→ 更新SlipDetector
→ 必要时同步重锚定PRSensorAngle
→ dx/dy中值滤波
→ FitCalibrationModel预测Fx/Fy/Fz
→ TangentialSample
```

`PRSensorAngle`维护接触状态、origin、二次精修和区域历史。首次接触建立粗origin，满足稳定与精修条件后进入状态2；卸载会重置接触相关状态。调用`reanchor_origin()`时必须保留已经完成的精修状态，只更新全局参考位置。

滑移检测流程：

```text
接触且motion ready
→ 保存短窗CoP与归一化压力斑块
→ 比较窗口首尾CoP位移
→ 在patch_search_radius内搜索零填充斑块平移
→ 相关性与改善量确认运动
→ enter_frames连续证据进入SLIP
→ EMA保存独立滑移方向
→ exit_frames连续低位移退出
→ 同帧重锚定CoP并输出0°
```

`sample.angle`在STICK时来自静态CoP delta，在SLIP时来自滑移EMA方向；`sample.angle_vector_magnitude`必须与该角度使用同一向量。GUI Snapshot红色箭头不能重新使用`hypot(sample.dx, sample.dy)`覆盖滑移向量模长。

### 11. 缓存、seq与压力—力同步

`TimestampedBuffer.append()`为缓存项分配单调递增seq；`get_after(seq)`按顺序返回未消费项；`find_closest(ts, max_diff_s, min_seq)`只寻找未使用且满足窗口的候选项。`runtime/synchronization.py`只是该匹配能力的薄适配层，不保存第二套匹配算法。

完整会话以压力帧为唯一业务驱动：

```text
get_after(last_press_seq)
→ 按seq逐帧调用TangentialFrameProcessor
→ 每帧推进阈值、CoP、滑移、标定和GUI状态
→ 无力通道时立即写NaN力字段
→ 有力通道时进入pending_press队列
→ 队首压力帧在±15 ms内匹配一个未使用力帧
→ 匹配成功写108列CSV
→ 超过等待窗口仍未匹配则不写该CSV行
```

有力通道时，即使某个压力帧最终没有CSV行，它也已经推进了压力状态机并可更新GUI。每个力帧最多匹配一次，后到压力帧不能越过pending队首。修改该语义会影响数据量、训练筛选和时间连续性，必须同时修改测试与本文第一部分的用户说明。

`rel_ms`以第一帧合法压力`rx_t`为起点，`delta_ms`来自相邻已保存压力帧的真实接收时间差。不得把它们写成固定0、5、10网格，也不得使用GUI调用时间或文件flush时间。

当前实现的未匹配语义必须特别保留：无力通道时每个合法压力帧都写一行并填充NaN力字段；力通道启用时，压力样本先进入`pending_press`，队首样本只有在15 ms窗口内找到尚未使用的力帧才写CSV，超过窗口会移出队列但不写该行。该语义与“压力状态机和GUI仍继续推进”同时成立，不能只根据CSV行数判断压力帧是否被处理。

### 12. 完整会话与并发模型

单路完整应用的线程与进程关系：

```text
Qt主线程
├── RealTimePlot与QTimer
└── 错误轮询

full-acquisition工作线程
└── acquisition_loop
    └── FullAcquisitionSession
        ├── pressure-consumer线程
        │   └── 父PressureSensor.read_frame
        │       └── IPC ← 压力采集子进程 ← 本地I/O线程 ← 压力串口
        ├── force-consumer线程
        │   └── 父SixAxisForceSensor.read_frame
        │       └── IPC ← 六维力采集子进程 ← 本地I/O线程 ← 力串口
        └── 主业务循环：处理、匹配、CSV、统计与GUI数据转发
```

`acquisition_loop`必须保持显式顺序：

```python
session.start()
try:
    while not session.should_stop():
        session.check_errors()
        session.process_new_pressure_frames()
        session.drain_force_matches()
        session.log_timing_stats()
        session.update_plot()
        session.wait_for_next_iteration()
finally:
    session.close()
```

该顺序确保线程错误先暴露、所有新压力帧按序处理、力匹配随后排空、GUI只消费最新状态、循环等待不会成为时间戳来源，并且任何异常都会进入统一清理。

双路应用只共享一个`QApplication`，不共享设备或算法状态：

```text
一个QApplication
├── Sensor A：config/窗口/工作线程/session/进程/缓存/CSV/输出目录
└── Sensor B：config/窗口/工作线程/session/进程/缓存/CSV/输出目录
```

`_validate_dual_configs()`会解析物理路径并拒绝压力端口冲突、启用的力端口冲突、压力与力交叉冲突以及输出目录冲突。任一路工作线程异常会设置两路停止事件并由Qt主线程报告和退出。

### 13. 资源生命周期与错误传播

压力设备是必需资源，连接或启动握手失败时`FullAcquisitionSession.start()`抛出异常，不创建空CSV。六维力是可选资源，连接或启动校零失败时关闭该通道，压力采集继续运行。

数据线程异常保存在`PressureThread.error`或`ForceThread.error`，`check_errors()`在业务循环中抛出；`FullApplicationRunner`通过线程安全队列把异常交给Qt主线程显示并退出，避免GUI仍存活但采集已经停止。

`close()`必须幂等并按依赖顺序释放：设置停止事件 → 等待消费线程 → 等待重新归零任务 → 关闭传感器与IPC → flush/关闭CSV → 删除确实没有数据行的本次CSV。新增资源时必须把释放逻辑加入同一个生命周期，并新增异常退出测试。

### 14. CSV与模型

`storage/csv.py`是108列格式的唯一来源。业务代码只传参数给`build_csv_row()`，不得手写列索引、复制表头或在GUI中拼接第二套行结构。

修改CSV时的最低要求：

- 同时修改`TABLE_CSV_HEADER`和`build_csv_row()`，保持长度一致。
- 更新训练、绘图、模型回归和集成测试。
- 明确新旧CSV兼容策略，离线工具必须按实际表头解析。
- 如果项目要求继续兼容108列，则不得增加、删除或重排列。

`FitCalibrationModel.from_default()`从`tangential.resources`读取内置`fit_coefs.bin`，`from_path()`读取用户外部模型。运行时预测与离线训练共享模型格式，修改序列化结构前必须通过现有模型回归测试证明旧模型行为没有变化。

### 15. application、examples与CLI为什么分开

`application.py`是稳定的库入口，`examples/`是调用示范，`cli.py`是字符串参数到配置对象的适配层，三者不能互相复制完整应用逻辑。

```text
用户Python代码 → run_application(config) ┐
examples/full.py → run_application(config) ├→ application.py → runtime/session.py
CLI app → examples/full.main(config) ──────┘
```

`application.py`只导入轻量配置；Qt、PyQtGraph和完整会话在真正调用`run_application()`或`run_dual_application()`时惰性加载。这样基础`import tangential`不会加载可选GUI和绘图库。

`examples/minimal.py`保留唯一最小压力循环；`examples/full.py`只调用完整应用公共入口；`examples/dual_sensor.py`只负责两份独立配置与命令行示范。示例不是SDK内部实现层，生产模块不得反向依赖示例。

当前命令分工固定为：`tangential example`惰性调用`examples/minimal.py`并只显示压力样本；`tangential app`通过`examples/full.py`调用`run_application`；`tangential dual`调用双路示例并复用`run_dual_application`；`tangential plot`惰性加载`tools.plotting`；`tangential fit`惰性加载`tools.training`。基础`import tangential`不应创建Qt窗口，也不应把Matplotlib/PyQtGraph加载为运行时副作用。

### 16. 公共API维护规则

`tangential.__all__`定义稳定顶层公共边界。用户通过`from tangential import ...`、`help()`、IDE类型提示和`py.typed/.pyi`了解API；内部模块路径不承诺稳定。

当前顶层共有33个导出名称：`TangentialSensor`、`TangentialSensorAPI`、`TangentialSample`、`TangentialFrameProcessor`、`FixedTerminalRenderer`、`FitCalibrationModel`、`FullApplicationConfig`、`PressureConfig`、`ForceConfig`、`CopConfig`、`ProcessingConfig`、`SlipConfig`、`CalibrationConfig`、`SyncConfig`、`OutputConfig`、`GuiConfig`、`PRSensorAngle`、`PressureSensor`、`TangentialMotionState`、`SlipResult`、`SlipDetector`、`compute_vector_angle`、`angle_difference`、`format_terminal_sample`、`TrainingConfig`、`TrainingResult`、`train_model`、`PlotConfig`、`PlotResult`、`plot_csv`、`plot_full_analysis`、`run_application`和`run_dual_application`。其中`TangentialSensor`是`TangentialSensorAPI`的推荐别名；两者当前指向同一个实现，修改导出时必须同步用户文档和API测试。

新增或修改公共API时必须同步：

1. 在唯一实现模块写完整类型标注与docstring，至少包含作用、参数、返回值、异常和副作用。
2. 通过`api.py`或对应公共门面导出。
3. 更新`__init__.py`导入与`__all__`。
4. 编译模块同步更新同名`.pyi`签名。
5. 更新`readme.md`公共API流程、输入和输出，并同步本文第一部分。
6. 增加API导入、签名、行为和基础导入惰性测试。

不要为了让用户“看到更多功能”把所有内部类都放进顶层。判断标准是：用户是否存在无需依赖内部会话即可稳定复用的场景。`TangentialSensor`适合硬件采集，`TangentialFrameProcessor`适合自定义数据源和离线84通道ADC；内部线程、会话辅助函数和协议解析私有方法不应公开。

### 17. 常见扩展任务

#### 17.1 增加配置参数

```text
确定唯一消费者
→ 在对应Config增加字段和默认值
→ 增加环境变量解析与validate
→ 从调用入口传到消费者
→ 添加默认/显式/非法值测试
→ 更新readme.md与本文对应章节
```

#### 17.2 修改压力或六维力协议

只修改对应`sensors/*.py`，同时覆盖分包、粘包、噪声、错误长度、CRC或帧尾、超时、慢响应和恢复。不得把协议解析放入`runtime/session.py`。

#### 17.3 修改CoP、区域或滑移

CoP与区域修改进入`processing/cop.py`，滑移修改进入`processing/slip.py`，`TangentialFrameProcessor`只负责编排。必须验证无接触、首次接触、精修、卸载、滑移进入、方向平滑、退出重锚定和多实例状态隔离。

#### 17.4 接入自定义ADC数据源

自定义来源只需提供84通道数据并调用`TangentialFrameProcessor.process()`；如果要复用`TangentialSensor`生命周期，可注入实现`read_frame()`、`decode()`和`close()`的sensor对象。不要修改`PressureSensor`来适配与现有协议无关的数据源。

#### 17.5 增加第三只或更多传感器

为每一路分别构造`FullApplicationConfig`、端口、输出目录、处理器和停止事件；复用现有单路会话，不共享`PRSensorAngle`或`SlipDetector`。在扩展运行器中统一校验所有物理端口和目录唯一性。

#### 17.6 增加新的编译模块

```text
新增唯一.py源码
→ 增加同名.pyi
→ 加入setup.py COMPILED_MODULES
→ 确认wheel排除该内部.py
→ 检查.so、.pyi、签名和docstring
→ 更新分发测试
```

### 18. 测试结构与修改路由

<table>
<thead>
<tr>
<th style="min-width:180px">修改内容</th>
<th>首选源码</th>
<th>最低联动测试</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal">压力协议、CRC、调度、进程和队列</td>
<td style="white-space:normal"><code>sensors/pressure.py</code></td>
<td style="white-space:normal"><code>test_data.py</code></td>
</tr>
<tr>
<td style="white-space:normal">六维力协议、校零和进程</td>
<td style="white-space:normal"><code>sensors/force.py</code></td>
<td style="white-space:normal"><code>test_data.py</code></td>
</tr>
<tr>
<td style="white-space:normal">seq、缓存和时间匹配</td>
<td style="white-space:normal"><code>acquisition/buffer.py</code>、<code>runtime/synchronization.py</code></td>
<td style="white-space:normal"><code>test_data.py</code>、<code>test_main_integration.py</code></td>
</tr>
<tr>
<td style="white-space:normal">CoP、阈值、梯度和区域</td>
<td style="white-space:normal"><code>processing/cop.py</code></td>
<td style="white-space:normal"><code>test_tangential_api.py</code>、GUI与集成测试</td>
</tr>
<tr>
<td style="white-space:normal">滑移状态和方向</td>
<td style="white-space:normal"><code>processing/slip.py</code>、<code>runtime/sensor.py</code></td>
<td style="white-space:normal"><code>test_slip.py</code>、<code>test_plot_and_gui.py</code></td>
</tr>
<tr>
<td style="white-space:normal">模型读取与预测</td>
<td style="white-space:normal"><code>processing/calibration.py</code></td>
<td style="white-space:normal"><code>test_model_and_table.py</code>、<code>test_calibration_multidim.py</code></td>
</tr>
<tr>
<td style="white-space:normal">最小API与示例</td>
<td style="white-space:normal"><code>runtime/sensor.py</code>、<code>api.py</code>、<code>examples/minimal.py</code></td>
<td style="white-space:normal"><code>test_tangential_api.py</code>、<code>test_stage2_structure.py</code></td>
</tr>
<tr>
<td style="white-space:normal">完整采集、CSV、清理和Qt生命周期</td>
<td style="white-space:normal"><code>runtime/session.py</code>、<code>application.py</code></td>
<td style="white-space:normal"><code>test_main_integration.py</code>、<code>test_dual_sensor_example.py</code></td>
</tr>
<tr>
<td style="white-space:normal">CSV结构</td>
<td style="white-space:normal"><code>storage/csv.py</code></td>
<td style="white-space:normal"><code>test_model_and_table.py</code>、绘图测试</td>
</tr>
<tr>
<td style="white-space:normal">GUI</td>
<td style="white-space:normal"><code>gui/realtime.py</code></td>
<td style="white-space:normal"><code>test_plot_and_gui.py</code></td>
</tr>
<tr>
<td style="white-space:normal">训练与绘图</td>
<td style="white-space:normal"><code>tools/training.py</code>、<code>tools/plotting.py</code></td>
<td style="white-space:normal"><code>test_training.py</code>、<code>test_plotting.py</code></td>
</tr>
<tr>
<td style="white-space:normal">CLI</td>
<td style="white-space:normal"><code>cli.py</code></td>
<td style="white-space:normal"><code>test_cli.py</code></td>
</tr>
<tr>
<td style="white-space:normal">wheel内容、资源和惰性导入</td>
<td style="white-space:normal"><code>pyproject.toml</code>、<code>setup.py</code>、<code>MANIFEST.in</code></td>
<td style="white-space:normal"><code>test_distribution.py</code>、<code>test_stage1_resources.py</code></td>
</tr>
</tbody>
</table>

### 19. 本地开发与测试

安装完整环境：

```bash
python -m pip install -r requirements.txt
```

基础语法检查：

```bash
PYTHONPATH=src python -m compileall -q src/tangential tests
```

完整测试：

```bash
PYTHONPATH=src \
QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -q
```

只运行相关测试时使用模块名，例如：

```bash
PYTHONPATH=src python -m unittest tests.test_data -q
PYTHONPATH=src python -m unittest tests.test_slip -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m unittest tests.test_plot_and_gui -q
```

提交前至少执行：

```bash
git diff --check
PYTHONPATH=src python -m compileall -q src/tangential tests
```

如果工作树已有用户修改，测试失败时必须区分本次变更和预存变更，不能用`git reset --hard`或覆盖式恢复清除用户内容。

### 20. Wheel构建与隔离验收

构建依赖由`pyproject.toml`声明，开发环境可以直接执行：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

当前10个编译模块由`setup.py`的`COMPILED_MODULES`定义：`runtime/sensor`、`runtime/session`、`runtime/synchronization`、`acquisition/buffer`、`sensors/pressure`、`sensors/force`、`processing/cop`、`processing/calibration`、`processing/slip`和`storage/csv`。

`setup.py`的Cython指令必须保持`language_level=3`、`annotation_typing=False`、`binding=True`、`embedsignature=True`和`always_allow_keywords=True`。其中`annotation_typing=False`保证源码中的类型注解不会被错误解释为运行时强类型约束，尤其不能破坏对`bytearray`、NumPy数组和测试注入对象的兼容输入。

构建流程：

```text
.py唯一源码
→ Cython生成并编译同名扩展
→ BinaryWheelBuildPy清理旧build/lib*/tangential
→ wheel保留公开Python层、配置、CLI、示例、GUI、tools、.pyi和资源
→ wheel排除10个内部实现.py与生成的C源码
```

预期产物：

```text
dist/tangential_sensor-0.4.0-cp311-cp311-linux_x86_64.whl
```

分发验收必须确认：

- wheel包含10个内部`.so`和10个同名`.pyi`。
- wheel包含`py.typed`与`tangential/resources/fit_coefs.bin`。
- wheel不包含对应内部`.py`、生成的C源码或外部share模型目录。
- 脱离源码目录后可以`import tangential`、加载内置模型并完成回归预测。
- `help()`、函数签名和IDE类型提示在安装wheel后仍可用。
- 基础`import tangential`不加载Qt、PyQtGraph或Matplotlib。
- 源码模式和隔离安装模式都通过协议、CoP、同步、CSV和模型回归测试。

当前`requirements.txt`锁定完整开发/GUI环境：Cython 3.2.9、NumPy 2.4.3、SciPy 1.17.1、pyserial 3.5、pyqtgraph 0.14.0、Matplotlib 3.10.8和PyQt5 5.15.11；`pyproject.toml`只声明核心运行依赖`numpy`、`scipy`、`pyserial`，GUI和离线绘图库属于`full`可选依赖，Cython只属于构建依赖。

不要手工提交`build/`、`dist/`、生成的`.so`或C文件；它们是可重建产物。

### 21. 常见故障定位

<table>
<thead>
<tr>
<th style="min-width:180px">现象</th>
<th>排查流程</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal">压力无数据</td>
<td style="white-space:normal">端口存在与权限 → 端口占用 → startup握手 → requests/frames/timeout统计 → 协议响应</td>
</tr>
<tr>
<td style="white-space:normal">实际频率低于200 Hz</td>
<td style="white-space:normal">响应延迟P50/P95 → timeout → schedule_skips → USB与设备响应能力</td>
</tr>
<tr>
<td style="white-space:normal"><code>delta_ms</code>抖动</td>
<td style="white-space:normal">确认使用rx_t → 对照响应延迟 → 排除timeout/错误 → 再检查系统负载</td>
</tr>
<tr>
<td style="white-space:normal">有压力但CSV行少</td>
<td style="white-space:normal">确认力通道是否启用 → 检查15 ms匹配 → 检查力帧率与时间戳 → 检查pending超窗</td>
</tr>
<tr>
<td style="white-space:normal">六维力降级</td>
<td style="white-space:normal">端口与权限 → 普通帧数量 → zero_sample_count → zero_timeout_s → 子进程错误</td>
</tr>
<tr>
<td style="white-space:normal">滑移误报或漏报</td>
<td style="white-space:normal">接触与motion ready → CoP短窗 → patch相关性 → enter/exit滞回 → angle_deadband</td>
</tr>
<tr>
<td style="white-space:normal">GUI仍在但采集停止</td>
<td style="white-space:normal">消费线程error → check_errors → runner错误队列 → Qt错误定时器与退出事件</td>
</tr>
<tr>
<td style="white-space:normal">wheel出现旧模块</td>
<td style="white-space:normal">清理build输出 → 检查COMPILED_MODULES → 重建wheel → 运行分发内容测试</td>
</tr>
<tr>
<td style="white-space:normal">双路相互影响</td>
<td style="white-space:normal">物理端口唯一性 → 配置对象独立 → 处理器与状态机独立 → 输出目录独立</td>
</tr>
</tbody>
</table>

### 22. Git检查点与回退

开始前：

```bash
git status --short
git log --oneline -n 10
```

完成后：

```bash
git diff --check
git diff --stat
```

每个独立阶段使用单独提交，提交前只暂存属于该阶段的文件。需要撤销已提交阶段时使用：

```bash
git revert <commit-hash>
```

不要使用`git reset --hard`覆盖用户修改。看到与当前任务无关的修改、数据文件删除或相邻目录未跟踪文件时，应保留并在交付报告中说明。

### 23. 修改完成的定义

一次修改只有同时满足以下条件才算完成：

- 修改位于唯一职责模块，没有复制协议、算法或CSV实现。
- 配置从`config.py`进入实际运行路径，没有散落第二套默认值。
- 多传感器状态与资源仍然隔离。
- 时间戳、seq、匹配窗口和CSV语义没有被GUI或循环节拍改变。
- 异常路径可以关闭线程、进程、串口、队列、CSV和Qt资源。
- 相关单元测试、集成测试和回归测试通过。
- 公共签名、`.pyi`、`readme.md`与本文的用户部分和维护部分同步。
- `git diff --check`通过，提交只包含本阶段文件。
