# Tangential Sensor SDK 0.6.0

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器。安装 wheel 后，普通用户可以直接运行现成示例，也可以通过高层 Python API 获取 `TangentialFrame` 并接入自己的程序。

当前 wheel 适用于 Linux x86_64 和 CPython 3.11。压力传感器是完整采集的必需设备，六维力传感器可选；默认端口分别为 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`。

## 系统要求与安装

推荐安装完整功能，包括实时 GUI 和离线绘图：

```bash
python -m pip install "./dist/tangential_sensor-0.6.0-cp311-cp311-linux_x86_64.whl[full]"
```

只使用压力采集和单帧结果时，可以不安装 GUI 可选依赖：

```bash
python -m pip install ./dist/tangential_sensor-0.6.0-cp311-cp311-linux_x86_64.whl
```

安装后检查：

```bash
tangential --version
python -c "import tangential; print(tangential.__version__)"
```

## A. 运行示例

### 命令总览

<table>
<thead>
<tr>
<th style="min-width:180px">命令</th>
<th>作用</th>
<th>主要输入</th>
<th>输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>tangential example</code></td>
<td style="white-space:normal">运行单压力传感器终端示例</td>
<td style="white-space:normal">压力端口和单帧超时时间</td>
<td style="white-space:normal">终端打印每个 <code>TangentialFrame</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>tangential app</code></td>
<td style="white-space:normal">运行单路完整采集应用</td>
<td style="white-space:normal">压力/六维力端口、保存目录和同步窗口</td>
<td style="white-space:normal">实时 GUI、108 列 CSV、CoP 频谱窗口和退出分析图</td>
</tr>
<tr>
<td style="white-space:normal"><code>tangential dual</code></td>
<td style="white-space:normal">同时运行两路相互隔离的完整应用</td>
<td style="white-space:normal">两只压力传感器的不同端口，可选两只六维力传感器</td>
<td style="white-space:normal">两个 GUI、两套 CSV 和两套分析图</td>
</tr>
<tr>
<td style="white-space:normal"><code>tangential plot</code></td>
<td style="white-space:normal">绘制已有 CSV</td>
<td style="white-space:normal">CSV 目录、文件、列和行范围</td>
<td style="white-space:normal">绘图文件或完整分析图</td>
</tr>
<tr>
<td style="white-space:normal"><code>tangential fit</code></td>
<td style="white-space:normal">根据训练 CSV 生成标定结果</td>
<td style="white-space:normal">训练数据、模型输出路径和评估图路径</td>
<td style="white-space:normal">模型文件和评估图；默认不改写输入 CSV</td>
</tr>
</tbody>
</table>

### 单压力传感器示例

```bash
tangential example \
  --pressure-port /dev/ttyUSB0 \
  --timeout 0.1
```

示例会在终端固定位置持续刷新 12×7 压力矩阵，并显示 `adc_sum`、CoP、角度、dx/dy 和当前运动状态；它不会打开 GUI、六维力传感器或 CSV 文件。

### 单路完整采集

```bash
tangential app \
  --pressure-port /dev/ttyUSB0 \
  --force-port /dev/ttyUSB1 \
  --save-dir ./data \
  --max-time-diff-ms 15
```

压力设备连接失败时程序退出且不创建空 CSV。六维力设备连接或校零失败时关闭六维力通道并继续压力采集，力相关 CSV 字段写入 `NaN`。关闭窗口或按 `Ctrl+C` 后，线程、进程、串口和 CSV 都会释放。

单路完整应用还会打开独立的 `CoP Spectrum` 窗口。窗口上方显示 CoP X、CoP Y 和合成幅值的实时频谱，下方显示最近约 30 秒的频谱瀑布图；频谱需要先积累约 2 秒稳定接触数据。采集结束后，CSV 同目录会生成与 CSV 同名的 `_spectrum.npz` 文件，保存频率轴、频谱时间、三个幅值矩阵以及采样配置。关闭频谱窗口只关闭显示，不会停止采集；关闭主窗口会同时关闭频谱窗口。

### 双传感器完整采集

双传感器示例为 `tangential dual`。两路拥有独立串口、采集进程、缓存、处理状态、GUI 和输出目录，不会共享同一个压力传感器实例或串口读取循环。

#### 第1步：识别两个物理端口

插入两只压力传感器后运行：

```bash
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

优先使用 `/dev/serial/by-id/` 下两个不同的实际设备路径。如果没有该目录，先根据设备枚举结果确认两个不同的 `/dev/ttyUSB*` 或 `/dev/ttyACM*`。

#### 第2步：设置端口变量

将下面的 `DEVICE_A_ID` 和 `DEVICE_B_ID` 替换为第1步看到的真实名称：

```bash
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
printf 'A=%s\nB=%s\n' "$PORT_A" "$PORT_B"
```

不要原样输入 `<sensor-a>` 或 `<sensor-b>`。Bash 会把尖括号解释为重定向符号并产生语法错误；两个变量必须指向不同的物理设备。

#### 第3步：检查权限和占用

```bash
ls -l "$PORT_A" "$PORT_B"
groups
fuser "$PORT_A" "$PORT_B"
```

当前用户通常需要属于 `dialout` 组。`fuser` 如果显示旧采集进程，应先关闭旧程序，不能让两个程序同时读取同一个串口。

#### 第4步：启动双传感器

```bash
tangential dual \
  --port-a "$PORT_A" \
  --port-b "$PORT_B" \
  --save-dir ./data/dual
```

默认输出目录为 `./data/dual/sensor_a` 和 `./data/dual/sensor_b`。如果两路都连接六维力传感器，显式指定两个不同的力端口：

```bash
tangential dual \
  --port-a "$PORT_A" \
  --port-b "$PORT_B" \
  --force-port-a /dev/serial/by-id/FORCE_A \
  --force-port-b /dev/serial/by-id/FORCE_B
```

使用 `tangential dual --help` 查看全部参数，也可以用 `--save-dir-a` 和 `--save-dir-b` 分别指定输出目录。

#### 第5步：确认和停止

启动后应看到标题包含 `Sensor A` 和 `Sensor B` 的两个完整 GUI。每一路都会保存 108 列 CSV，并在退出时生成自己的分析图。双传感器模式不创建 CoP 频谱窗口，也不生成频谱 NPZ。按 `Ctrl+C` 或关闭窗口停止；如果某一路异常，程序会报告对应的 A/B 并关闭两路资源。

### 离线绘图

```bash
tangential plot \
  --dir ./data \
  --files capture.csv \
  --columns Fy_cal,delta_Force_Y \
  --rows 100:500 \
  --save ./data/capture.png
```

绘图工具按 CSV 实际表头读取列名，不依赖旧版固定索引。空文件、缺失列和空行范围会返回明确错误；使用 `--list` 查看可用 CSV，使用 `--mode full_analysis` 生成完整分析图。

### 离线训练

```bash
tangential fit \
  --xy-csv ./data/fx_fy.csv \
  --z-csv ./data/fz.csv \
  --output-model ./fit_coefs.bin \
  --output-plot ./fit_report.png
```

默认只生成模型和评估图，不修改输入 CSV。只有显式提供 `--write-back PATH` 才会写回 CSV；覆盖已有目标时还必须提供 `--force`。

### 常见故障排查

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
<td style="white-space:normal"><code>unexpected token newline</code></td>
<td style="white-space:normal">原样复制了带尖括号的端口占位符</td>
<td style="white-space:normal">替换成实际端口路径后再执行命令</td>
</tr>
<tr>
<td style="white-space:normal"><code>No such file or directory</code></td>
<td style="white-space:normal">设备未连接或端口名称改变</td>
<td style="white-space:normal">重新运行 <code>serial.tools.list_ports</code> 并核对路径</td>
</tr>
<tr>
<td style="white-space:normal"><code>Permission denied</code></td>
<td style="white-space:normal">当前用户没有串口权限</td>
<td style="white-space:normal">加入 <code>dialout</code> 组并重新登录</td>
</tr>
<tr>
<td style="white-space:normal">双路提示端口冲突</td>
<td style="white-space:normal">两个参数指向同一物理设备</td>
<td style="white-space:normal">为 A/B 选择两个不同的设备路径</td>
</tr>
<tr>
<td style="white-space:normal">某一路持续无数据</td>
<td style="white-space:normal">端口、供电、设备响应或 USB 资源异常</td>
<td style="white-space:normal">先单独运行 <code>tangential example</code> 验证该端口</td>
</tr>
<tr>
<td style="white-space:normal">频谱窗口始终等待且没有曲线</td>
<td style="white-space:normal">稳定接触状态未连续积满约2秒，或相邻有效帧间隔超过75 ms而重新计数</td>
<td style="white-space:normal">保持稳定接触并检查USB连接和响应超时；默认75 ms可覆盖一次50 ms超时恢复，超过75 ms仍会重置当前频谱窗口</td>
</tr>
</tbody>
</table>

## B. Python 二次开发

用户代码只需要从 `tangential` 导入稳定高层 API。底层设备驱动、内部处理和资源读取由 SDK 完成，不属于用户二次开发接口；用户可以选择直接读取 `TangentialFrame`，或把自己的 84 通道 ADC 数据交给 `TangentialFrameProcessor`。

### 公共 API

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
<td style="white-space:normal"><code>TangentialSensorAPI</code></td>
<td style="white-space:normal">压力设备读取 → 单帧处理 → <code>TangentialFrame</code></td>
<td style="white-space:normal">压力配置、处理配置和可选资源注入</td>
<td style="white-space:normal">逐帧 <code>TangentialFrame</code>；<code>close()</code> 释放资源</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrameProcessor</code></td>
<td style="white-space:normal">84 通道 ADC → 单帧处理 → <code>TangentialFrame</code></td>
<td style="white-space:normal"><code>process_frame(raw_data, frame=None)</code> 和处理配置</td>
<td style="white-space:normal">返回 <code>TangentialFrame</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrame</code></td>
<td style="white-space:normal">保存一帧可直接使用的压力数据和结果</td>
<td style="white-space:normal">通常由 <code>TangentialSensorAPI</code> 或 <code>TangentialFrameProcessor</code> 创建</td>
<td style="white-space:normal">固定八个字段，详见下表</td>
</tr>
</tbody>
</table>

### TangentialFrame 字段

公开采集和处理接口始终只返回 `TangentialFrame`，不会要求用户接触内部详细样本。`base_data` 是 SDK 实际使用的长度为 84 的一维压力数组；需要矩阵时可以在用户代码中执行 `frame.base_data.reshape(12, 7)`。

<table>
<thead>
<tr>
<th style="min-width:180px">字段</th>
<th>作用</th>
<th>类型</th>
<th>语义</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>base_data</code></td>
<td style="white-space:normal">保存实际进入算法的压力通道</td>
<td style="white-space:normal"><code>numpy.ndarray</code></td>
<td style="white-space:normal">长度 84，按 12×7 传感器布局排列</td>
</tr>
<tr>
<td style="white-space:normal"><code>adc_sum</code></td>
<td style="white-space:normal">保存 ADC 总和</td>
<td style="white-space:normal"><code>float</code></td>
<td style="white-space:normal">84 个通道之和，CSV 中使用同名列</td>
</tr>
<tr>
<td style="white-space:normal"><code>cop_x</code>、<code>cop_y</code></td>
<td style="white-space:normal">保存压力中心位置</td>
<td style="white-space:normal"><code>float</code></td>
<td style="white-space:normal">坐标单位由传感器布局约定</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle</code></td>
<td style="white-space:normal">保存切向方向角</td>
<td style="white-space:normal"><code>float</code></td>
<td style="white-space:normal">单位为度</td>
</tr>
<tr>
<td style="white-space:normal"><code>dx</code>、<code>dy</code></td>
<td style="white-space:normal">保存 CoP 相对位移</td>
<td style="white-space:normal"><code>float</code></td>
<td style="white-space:normal">方向和大小由当前处理状态决定</td>
</tr>
<tr>
<td style="white-space:normal"><code>motion_state</code></td>
<td style="white-space:normal">保存运动状态</td>
<td style="white-space:normal">枚举值</td>
<td style="white-space:normal">可比较 <code>NO_CONTACT</code>、<code>STICK</code>、<code>SLIP</code></td>
</tr>
</tbody>
</table>

### 配置对象

所有用户可调参数集中在分类配置中。显式 CLI 参数优先于代码传入的配置对象，配置对象优先于环境变量，环境变量优先于内置默认值。

<table>
<thead>
<tr>
<th style="min-width:180px">配置</th>
<th>作用</th>
<th>主要消费者</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>PressureConfig</code></td>
<td style="white-space:normal">压力端口、波特率、频率、超时和队列</td>
<td style="white-space:normal"><code>TangentialSensorAPI</code> 和完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>ForceConfig</code></td>
<td style="white-space:normal">六维力开关、端口、频率、超时和校零</td>
<td style="white-space:normal">完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>CopConfig</code>、<code>ProcessingConfig</code></td>
<td style="white-space:normal">单帧处理相关阈值、区域、滤波和策略</td>
<td style="white-space:normal"><code>TangentialFrameProcessor</code> 和完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipConfig</code></td>
<td style="white-space:normal">单帧运动状态相关窗口、阈值、平滑和滞回参数</td>
<td style="white-space:normal"><code>TangentialFrameProcessor</code> 和完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>CalibrationConfig</code></td>
<td style="white-space:normal">选择默认或外部标定配置</td>
<td style="white-space:normal">完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>SyncConfig</code></td>
<td style="white-space:normal">主循环、GUI、匹配窗口和缓存</td>
<td style="white-space:normal">完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>OutputConfig</code>、<code>GuiConfig</code></td>
<td style="white-space:normal">保存目录、窗口和显示参数</td>
<td style="white-space:normal">完整应用</td>
</tr>
<tr>
<td style="white-space:normal"><code>TrainingConfig</code>、<code>PlotConfig</code></td>
<td style="white-space:normal">离线训练和绘图参数</td>
<td style="white-space:normal"><code>train_model()</code> 和绘图入口</td>
</tr>
</tbody>
</table>

### 常见二次开发流程

已有压力设备时，使用 `TangentialSensorAPI`：

```python
from tangential import PressureConfig, TangentialSensorAPI

with TangentialSensorAPI(config=PressureConfig(port="/dev/ttyUSB0")) as sensor:
    while True:
        frame = sensor.read(timeout_s=0.1)
        if frame is not None:
            print(frame.adc_sum, frame.cop_x, frame.cop_y, frame.angle)
```

已有 84 通道 ADC 数据时，使用 `TangentialFrameProcessor`：

```python
import numpy as np
from tangential import TangentialFrameProcessor

processor = TangentialFrameProcessor()
raw_data = np.zeros(84, dtype=np.uint16)
frame = processor.process_frame(raw_data)
print(frame)
```

需要完整 GUI、CSV 和双传感器隔离时，请按“运行示例”部分的 `app` 或 `dual` 命令启动；不要在用户代码中复制设备协议、帧解析或 108 列 CSV 字段映射。

### 数据与 CSV 语义

压力帧在合法协议帧完成解析时记录真实接收时间。`rel_ms` 和 `delta_ms` 来自压力帧时间戳，不由 GUI 刷新、主循环睡眠或重采样生成；首个有效压力帧的时间从 0 开始。

完整应用保持现有 108 列 CSV 格式。`adc_sum` 是 84 通道 ADC 之和；六维力通道关闭、连接失败或校零失败时，力相关字段写入 `NaN`。启用六维力时，每个力帧最多匹配一次，匹配窗口由 `SyncConfig` 控制。

`TangentialFrame.base_data` 是 SDK 实际用于 CoP、角度、梯度、滑移、模型和 CSV 的 84 通道压力数据。

绘图工具按 CSV 表头读取列名，因此用户可以使用现有 CSV，也可以在离线分析中明确选择列和行范围。训练命令默认不写回输入 CSV，写回必须显式指定目标路径。

### 配置生效时机

修改 `config.py` 的默认值只影响之后创建的配置对象和重新构建的 wheel；已经安装的旧 wheel 和已经创建的配置对象不会自动更新。多传感器程序必须为每一路创建独立的配置对象。
