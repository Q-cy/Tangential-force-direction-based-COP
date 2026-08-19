# PZT + 六维力实时采集与标定

本项目采集 12×7 压阻阵列和六维力传感器，计算整帧/分区 CoP，加载
`fit_coefs.bin` 实时标定 Fx、Fy、Fz，并保存 108 列 CSV。

## 安装

- Python 3.11
- Linux 串口：压力 `/dev/ttyUSB0`，六维力 `/dev/ttyUSB1`
- `requirements.txt` 是开发和完整 GUI 环境

普通用户推荐安装 wheel。核心压力 API 不安装 GUI：

```bash
python -m pip install tangential_sensor-0.1.0-py3-none-any.whl
```

需要完整 GUI 时安装可选依赖：

```bash
python -m pip install "tangential_sensor-0.1.0-py3-none-any.whl[full]"
```

推荐使用独立环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本机已验证的 Conda 环境是 `TimeDrift_GRU`。

## API 示例

最小示例通过公开的 `tangential` 包采集压力帧并计算 CoP、角度、梯度和
`fit_coefs.bin` 标定结果。终端每帧原位刷新固定的 12×7 ADC 矩阵以及
min、max、sum、mean、copX、copY、angle 和标定力：

```bash
python example.py
```

用户代码只需要一个公开入口：

```python
from tangential import TangentialSensor

with TangentialSensor() as sensor:
    sample = sensor.read()
    print(sample.matrix, sample.cop_x, sample.cop_y, sample.angle)
```

完整示例保留原来的双传感器、时间匹配、CSV 和 PyQtGraph 功能：

```bash
python main.py
```

`main.py` 明确保留采集 `while` 循环；设备、CoP/标定处理、重新归零、CSV、
同步、统计和 GUI 生命周期由 `TangentialFrameProcessor`、
`FullAcquisitionSession` 等类封装。完整应用直接复用最小 API 的单帧处理器，
不会维护第二套 CoP、梯度或拟合算法。

## 项目架构

规范实现位于 `src/tangential/`：

```text
src/tangential/
├── __init__.py                  # 用户公共 API
├── api.py                       # TangentialSensor / TangentialSample
├── config.py                    # 配置和模型资源路径
├── sensors/                     # 压力与六维力串口驱动
├── processing/                  # CoP、梯度和运行时标定
├── acquisition/                 # 时间戳缓存与同步
├── storage/                     # 108列 CSV
├── gui/                         # 可选 PyQtGraph GUI
└── full.py                      # 完整采集会话
```

根目录的兼容模块只转发到 `src/tangential/`，新的功能只在标准包中实现。

构建 wheel：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

wheel 可以同时包含 Python 模块和平台相关 `.so`。当前版本保持纯 Python，
因此生成的 `py3-none-any.whl` 更容易跨机器安装；以后只需把确有必要的核心算法
编译成 `.so` 并继续封装在同一个 wheel 中。

压力传感器是必需设备：连接失败时程序退出，且不会创建空 CSV。六维力传感器
是可选设备；启动时需要在 1 秒内收集 10 个有效帧完成零点校准。连接或校零失败
时程序会明确报警并降级为压力模式，力、力时间戳和同步差列写 `NaN`。

压力和六维力通道分别在独立的 `spawn` 子进程中，以 200 Hz（5 ms）为目标执行单请求在途的
请求—响应循环，避免 GUI、CoP 和标定计算通过 Python GIL 延迟串口轮询。每轮发送
一次请求，最多等待 50 ms；完整压力帧和六维力帧
通过状态和 CRC 校验后立即记录接收时间；CSV 前两列 `rel_ms` 和 `delta_ms` 分别表示
从本文件首个实际保存行开始的相对时间（毫秒）以及与上一保存行的压力时间差（毫秒），
首行均为 `0.0`。`press_t` 仍保存压力帧原始时间戳；
不会被强制改写为等间隔。设备响应超过目标周期时会跳过已错过的发送周期，不会
密集补发请求；每秒控制台会输出实际帧率、请求间隔、响应延迟及错误统计。

力传感器校零属于软件校零：启动时对 10 个普通读取帧取平均作为六轴零点，运行期
触发重新归零时从 ForceThread 已接收的新帧计算 Fx/Fy 偏置，不发送专用硬件置零
命令，也不会创建第二个串口消费者。

双传感器模式以压力帧驱动，每个压力帧只处理一次，并在 ±15 ms 窗口内匹配
一个尚未使用的力帧。超窗压力帧仍参与 CoP 状态和 GUI 更新，但不写入双传感器
CSV，保证保存行内的 `dt <= 0.015` 且力帧不重复使用。

关闭窗口后程序会停止采集线程、关闭串口和 CSV；本次没有任何数据行时删除 CSV。

## 标定与离线分析

训练脚本：

```bash
python fit.py
```

当前配置使用以下有效行（`valid != 0`；没有 `valid` 时回退到
`CoP_state != 0`）：

- Fx/Fy：`/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0713_1.csv`
- Fz：`/home/qcy/Project/data/2.PZT_tangential/weight/concat/concat_5_10_15.csv`

离线绘图：

```bash
python plot_static.py -f COP_0713_1.csv -c Fy_cal,delta_Force_Y -r 100:500
```

## 串口与时序策略

- 压力串口以 `timeout=0` 打开；使用 `select.select()` 等待文件描述符可读，
  每次最多等待 10 ms，再用一次非阻塞读取取出最多 1024B。
- 每轮开始清空压力串口输入/输出和解析缓存，然后发送一个请求；不会并发或突发补发请求。
- 使用 `bytearray` 累积本轮分包数据，解析出完整帧后才移除；本轮结束时丢弃残留字节。
- 帧头错位时逐字节滑动；长度、CRC 或状态错误时丢弃错误帧并继续同步。
- 压力发送和接收由独立采集进程内的同一个 I/O 线程管理，同一时间最多一个压力请求
  在途；父进程只接收已经带有原始 `rx_t` 的完整帧。
- 六维力发送和接收同样由独立采集进程内的同一个 I/O 线程管理，同一时间最多一个力请求
  在途；父进程使用力帧真实 `rx_t` 与压力帧做 ±15 ms 匹配。两路 200 Hz 都是目标上限，
  实际帧率由设备响应时间决定，不插值或重采样。
- 使用单调时钟按 5 ms 周期运行；整轮耗时超过周期时立即进入下一轮，不突发补发。
- 接收缓冲区设置上限，防止异常数据无限增长。

## 测试

测试不需要真实硬件：

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -v
```
