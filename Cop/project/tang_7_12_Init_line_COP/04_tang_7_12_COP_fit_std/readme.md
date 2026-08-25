# Tangential Sensor SDK 0.5.0

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器，提供 CoP、角度、梯度、切向力标定、实时 GUI、固定 108 列 CSV 和离线分析。

本文面向安装 wheel 后使用 SDK 和进行二次开发的用户，介绍硬件连接、命令行、Python API、参数配置、滑移检测、CSV 行为和常见故障。

## 系统要求与安装

当前 wheel 适用于 Linux x86_64 和 CPython 3.11。压力传感器是必需设备，六维力传感器是可选设备；默认端口分别为 ``/dev/ttyUSB0`` 和 ``/dev/ttyUSB1``。

完整功能包含实时 GUI 和离线绘图，推荐安装：

```bash
python -m pip install "./dist/tangential_sensor-0.5.0-cp311-cp311-linux_x86_64.whl[full]"
```

只使用压力采集、CoP、标定等核心 API 时，可以不安装 GUI 可选依赖：

```bash
python -m pip install ./dist/tangential_sensor-0.5.0-cp311-cp311-linux_x86_64.whl
```

安装后检查：

```bash
tangential --version
python -c "import tangential; print(tangential.__version__)"
```

用户代码只需从 ``tangential`` 顶层导入本指南列出的公共 API；Python 会自动加载 wheel 内的编译核心，不需要也不应直接导入某个 ``.so`` 文件。

## 同时连接两个压力传感器

双传感器示例模块为 ``tangential.examples.dual_sensor``。它启动一个 Qt 应用和两个完整窗口；A/B 各自执行压力采集、CoP、角度、梯度、标定、实时曲线、压力表、完整 108 列 CSV，并在退出时生成各自的分析图。不再是终端摘要循环。默认只连接压力传感器；只有显式提供对应 ``--force-port-a`` 或 ``--force-port-b`` 才启用六维力通道，避免两路同时打开默认 ``/dev/ttyUSB1``。

### 第1步：插入设备并识别两个端口

插入两只压力传感器后运行：

~~~bash
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
~~~

优先选择 ``/dev/serial/by-id/`` 下两个不同的设备路径，因为它们通常不会随重插或重启改变。若该目录不存在，再根据 ``serial.tools.list_ports`` 的输出确认两只设备分别对应哪个 ``/dev/ttyUSB*`` 或 ``/dev/ttyACM*``。

本机当前如果没有列出任何端口，说明设备尚未接入、USB未识别或串口驱动尚未创建，不能继续启动示例。

### 第2步：设置本次运行使用的端口

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

### 第3步：检查权限和端口占用

~~~bash
ls -l "$PORT_A" "$PORT_B"
groups
fuser "$PORT_A" "$PORT_B"
~~~

- ``ls`` 必须能找到两个路径。
- 当前用户通常需要属于 ``dialout`` 组；若没有权限，可执行 ``sudo usermod -aG dialout "$USER"``，然后注销并重新登录。
- ``fuser`` 没有输出通常表示端口空闲；若显示进程号，应先关闭正在占用传感器的旧采集程序，不要让两个程序同时读取同一串口。

### 第4步：启动双传感器示例

~~~bash
tangential dual \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
~~~

默认输出目录为 ``./data/sensor_a`` 和 ``./data/sensor_b``。指定父目录时：

~~~bash
tangential dual \
  --port-a "$PORT_A" --port-b "$PORT_B" \
  --save-dir ./data/dual
~~~

如果两路都要连接六维力传感器，必须显式提供两个不同的力端口：

~~~bash
tangential dual \
  --port-a "$PORT_A" --port-b "$PORT_B" \
  --force-port-a /dev/serial/by-id/FORCE_A \
  --force-port-b /dev/serial/by-id/FORCE_B
~~~

也可以分别覆盖输出目录：``--save-dir-a`` 和 ``--save-dir-b``；模型使用 ``--model MODEL_PATH``，或分别使用 ``--model-a``、``--model-b``。

查看全部参数：

~~~bash
tangential dual --help
~~~

### 第5步：确认输出并停止

运行后会出现两个窗口，标题分别包含 ``Sensor A`` 和 ``Sensor B``。每个窗口都包含压力/六维力实时曲线、方向和幅值、12×7 压力表、CoP 标记、梯度箭头以及状态显示；状态变化不会覆盖 A/B 标签。每路目录会保存一个完整 108 列 CSV，退出后还会保存 ``full_analysis_cop_<n>.png``。

按 ``Ctrl+C`` 或关闭 Qt 应用时，两路会同时停止；任一路采集线程异常都会报告具体的 A/B，并联动安全关闭另一路。不要直接拔线代替正常退出。

Python调用：

~~~python
from tangential import (
    ForceConfig,
    FullApplicationConfig,
    GuiConfig,
    OutputConfig,
    PressureConfig,
    run_dual_application,
)

run_dual_application(
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

### 常见错误

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

终端每帧显示 12×7 原始 ADC、adc_sum、CoP X/Y、角度、dx、dy 和运动状态。此路径不启动六维力、CSV 或 Qt GUI。

### 完整采集

~~~bash
tangential app \
  --pressure-port /dev/ttyUSB0 \
  --force-port /dev/ttyUSB1 \
  --save-dir ./data \
  --max-time-diff-ms 15
~~~

压力传感器是必需设备；连接失败时程序退出且不创建空 CSV。六维力传感器是可选设备；连接或普通数据帧校零失败时降级为压力模式，力相关列写入 NaN。两路设备由独立采集进程读取，父进程按真实接收时间完成匹配和 CSV 保存。

普通 ``app`` 命令使用 ``ForceConfig.enabled=True`` 的默认配置，因此没有提供 ``--force-port`` 时仍会尝试打开默认 ``/dev/ttyUSB1``；如果该设备不存在或校零失败，程序会关闭力通道并继续压力采集。需要明确只采集压力时，在 Python API 中传入 ``ForceConfig(enabled=False)``，或设置 ``TANGENTIAL_FORCE_ENABLED=false`` 后再启动。

### 双路完整采集

~~~bash
tangential dual \
  --port-a /dev/serial/by-id/PRESSURE_A \
  --port-b /dev/serial/by-id/PRESSURE_B \
  --save-dir ./data/dual
~~~

该命令显示两个完整 GUI 窗口，默认把 CSV 和退出分析图分别保存到 ``./data/dual/sensor_a``、``./data/dual/sensor_b``。只有显式增加 ``--force-port-a``、``--force-port-b`` 才启用对应六维力通道；两个力端口也必须是不同物理设备。

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

所有稳定公共名称都可以直接从 ``tangential`` 导入。普通采集优先使用 ``TangentialSensor``；需要完整 GUI 时使用 ``run_application`` 或 ``run_dual_application``。``PressureSensor``、``PRSensorAngle`` 和 ``TangentialFrameProcessor`` 面向需要自行编排数据流的高级用户。

### 最小采集示例

~~~python
from tangential import PressureConfig, TangentialSensor

pressure = PressureConfig(port="/dev/ttyUSB0")
with TangentialSensor(config=pressure) as sensor:
    while True:
        frame = sensor.read(timeout_s=0.1)
        if frame is not None:
            print(frame.raw.reshape(12, 7))
            print(frame.adc_sum)
            print(frame.cop_x, frame.cop_y, frame.angle)
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
<td style="white-space:normal">PressureSensor → decode → TangentialFrameProcessor → TangentialFrame</td>
<td style="white-space:normal"><code>PressureConfig</code>、可选 <code>ProcessingConfig</code>、模型路径</td>
<td style="white-space:normal"><code>read(timeout_s)</code> 返回 <code>TangentialFrame</code> 或 <code>None</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialSensorAPI</code></td>
<td style="white-space:normal">压力设备生命周期 → 调用 TangentialFrameProcessor → TangentialFrame</td>
<td style="white-space:normal">传感器/工厂注入、压力配置、处理配置</td>
<td style="white-space:normal">逐帧<code>TangentialFrame</code>；<code>close()</code> 释放设备</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrame</code></td>
<td style="white-space:normal">84通道ADC与处理结果 → 八字段公开帧</td>
<td style="white-space:normal">通常由处理器创建，不建议用户手工构造</td>
<td style="white-space:normal"><code>raw</code>、<code>adc_sum</code>、CoP、角度、dx/dy和运动状态</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrameProcessor</code></td>
<td style="white-space:normal">84通道ADC → CoP/梯度/滑移/标定 → TangentialFrame</td>
<td style="white-space:normal"><code>raw</code>、<code>ProcessingConfig</code>、可选标定模型</td>
<td style="white-space:normal"><code>process()</code> 返回 <code>TangentialFrame</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>FixedTerminalRenderer</code></td>
<td style="white-space:normal">TangentialFrame → 固定布局文本 → 原位刷新终端</td>
<td style="white-space:normal">输出流、<code>TangentialFrame</code></td>
<td style="white-space:normal"><code>render()</code> 写入并刷新终端，同时返回文本</td>
</tr>
<tr>
<td style="white-space:normal"><code>format_terminal_sample</code></td>
<td style="white-space:normal">TangentialFrame → 12×7矩阵与指标 → str</td>
<td style="white-space:normal"><code>TangentialFrame</code></td>
<td style="white-space:normal"><code>str</code></td>
</tr>
</tbody>
</table>

#### 算法、模型与底层压力驱动

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
<td style="white-space:normal">fit_coefs.bin → dx/dy/adc_sum → Fx/Fy/Fz</td>
<td style="white-space:normal"><code>from_default()</code> 或 <code>from_path(path)</code>；<code>predict(dx, dy, adc_sum, cal_dim="3D")</code></td>
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

#### 完整应用入口

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

#### 配置对象

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

#### 训练与绘图

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

### TangentialFrame 字段

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
<td style="white-space:normal">原始一维 ADC 数据；终端显示时可 reshape 为12×7</td>
</tr>
<tr>
<td style="white-space:normal"><code>adc_sum</code></td>
<td style="white-space:normal">float，ADC</td>
<td style="white-space:normal">84通道 ADC 之和；对象中唯一的 ADC 总和名称，CSV 也使用同名列</td>
</tr>
<tr>
<td style="white-space:normal"><code>cop_x</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">CoP列坐标；无效时可能为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>cop_y</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">CoP行坐标；无效时可能为NaN</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle</code></td>
<td style="white-space:normal">float，度</td>
<td style="white-space:normal">当前静态切向或滑移方向角</td>
</tr>
<tr>
<td style="white-space:normal"><code>dx</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">中值滤波后的 CoP X 相对 origin 偏移</td>
</tr>
<tr>
<td style="white-space:normal"><code>dy</code></td>
<td style="white-space:normal">float，cell</td>
<td style="white-space:normal">中值滤波后的 CoP Y 相对 origin 偏移</td>
</tr>
<tr>
<td style="white-space:normal"><code>motion_state</code></td>
<td style="white-space:normal"><code>TangentialMotionState</code></td>
<td style="white-space:normal">NO_CONTACT、STICK 或 SLIP</td>
</tr>
</tbody>
</table>

完整应用保存的 108 列 CSV 中包含 ``rel_ms``、``delta_ms``、``press_t``、``force_t`` 等时序列；这些列不属于 ``TangentialFrame``。分析设备时序时应读取 CSV 对应列并结合采集日志。

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

用户不需要、也不建议直接修改安装包中的 ``config.py``。推荐在代码中创建配置对象，或在启动前设置 ``TANGENTIAL_*`` 环境变量。配置对象在应用启动前统一校验，非法端口、频率、超时、队列或阈值会抛出 ``ValueError``。

配置优先级：

```text
CLI显式参数 > 代码显式传入的配置对象 > TANGENTIAL_*环境变量 > 默认值
```

### 设备配置

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

### CoP与处理配置

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

### 同步、输出与GUI配置

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

### 训练和绘图配置

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

## 滑移检测

当前版本提供可复用的 ``SlipDetector``。它不改变 108 列 CSV，不修改 ``fit_coefs.bin``，也不改变标定模型输入；公开 ``TangentialFrame`` 只暴露 ``motion_state`` 和 ``angle``，滑移距离、置信度和方向向量等详细结果仅供完整 GUI 内部使用。每个处理器/传感器实例拥有独立 detector，双传感器不会共享滑移历史。

### SlipConfig全部可调参数

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
- 比较短窗首尾 CoP 位移；在 ``±patch_search_radius`` 范围做零填充平移，使用余弦相关，并要求相对零平移提升 ``patch_min_improvement``。
- CoP 位移达到 ``enter_distance`` 且斑块确认，或相对 detector anchor 达到 ``reanchor_distance`` 的大位移兜底时，连续 ``enter_frames`` 个窗口进入 SLIP。
- SLIP 期间用 ``direction_smoothing`` 做运动方向 EMA；短窗位移连续低于 ``exit_distance`` 达到 ``exit_frames`` 后退出，当前位置重新锁定全局静摩擦 origin，退出帧角度为 0。
- ``angle_deadband`` 以下的方向向量输出 0。无接触或 CoP 不可用时完整 reset，状态为 ``NO_CONTACT``；接触但未滑移为 ``STICK``。

实时 GUI 的两个方向面板语义不同：

- ``Direction`` 的红色 PZT 箭头保持固定显示长度，只表达 ``sample.angle`` 的方向，不表达位移或力的大小。
- ``Pressure Snapshot`` 的红色 PZT 箭头同样沿 ``sample.angle``，但长度来自 ``sample.angle_vector_magnitude``：STICK 时是静态 CoP delta 模长，SLIP 时是 EMA 滑移向量模长。显示时乘 0.5 并限制到 0.65，避免超出面板。
- ``Pressure Snapshot`` 蓝色箭头仍使用六维力 Fx/Fy 的模长；Pressure Table 中实际 origin、当前 CoP、delta 和区域几何不受上述显示缩放影响。

### 滑移方向与 CoP 重锚定时序

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

### 配置何时生效

- 推荐在代码中显式传入 ``PressureConfig``、``SlipConfig`` 等对象，参数最清楚。
- 纯命令行部署可以设置 ``TANGENTIAL_*`` 环境变量，再启动新进程。
- 配置在对象创建和应用启动时读取；修改环境变量后必须重新创建配置并重启程序。
- 不要为了调参修改安装包中的 ``config.py``，升级或重新安装会覆盖此类修改。
- 多传感器场景应分别创建 ``config_a`` 和 ``config_b``，不要共享同一个可变 ``FullApplicationConfig`` 实例。

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

二次开发应从 ``tangential`` 顶层导入公共 API，不要依赖未在公共 API 表中列出的内部模块路径。常见方式包括：

- 用 ``TangentialSensor`` 编写自己的实时控制或数据分析循环。
- 用 ``TangentialFrameProcessor`` 处理已有84通道压力帧。
- 用 ``SlipDetector`` 将滑移结果接入机器人控制状态机。
- 用 ``run_application`` 和分类配置快速启动标准GUI。
- 用 ``train_model``、``plot_csv`` 和 ``plot_full_analysis`` 构建离线流程。

每只压力传感器必须创建独立的 ``TangentialSensor`` 或处理器实例，不能共享 ``PRSensorAngle`` 或 ``SlipDetector``，否则接触origin、历史窗口和滑移状态会相互污染。

## 常见故障

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
