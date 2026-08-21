# Tangential Sensor SDK 0.3.0

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器，提供 CoP、角度、梯度、切向力标定、实时 GUI、固定 108 列 CSV 和离线分析。

项目同时支持源码开发和 wheel 交付：仓库完整保留 Python 源码；普通用户通过公开 API、配置、CLI 和示例使用 wheel。

## 安装与交付边界

当前发布目标为 Linux x86_64、CPython 3.11：

~~~text
tangential_sensor-0.3.0-cp311-cp311-linux_x86_64.whl
~~~

安装完整功能：

~~~bash
python -m pip install "./dist/tangential_sensor-0.3.0-cp311-cp311-linux_x86_64.whl[full]"
~~~

只使用压力采集、CoP、标定和 CSV 核心能力时，可不安装 GUI extra：

~~~bash
python -m pip install ./dist/tangential_sensor-0.3.0-cp311-cp311-linux_x86_64.whl
~~~

wheel 分为两层：

- runtime、acquisition、sensors、processing、storage 的内部实现编译为多个 CPython 3.11 .so。
- __init__.py、api.py、config.py、application.py、cli.py、examples/、gui/、tools/ 和类型声明保留为可读 Python。

wheel 内部实现的 Python 源文件不随 wheel 发布，但源码仓库完整保留这些 .py 文件。当前共编译9个扩展模块：runtime 3个、acquisition 1个、sensors 2个、processing 2个、storage 1个；每个扩展都有同名 .pyi 类型声明。

.so 是 Python 的 CPython 扩展，不是稳定的 C++ ABI，不能直接作为 C++ 链接。需要原生 ABI 时，应另行设计 C ABI 或原生 SDK 层。

为保护内部实现，不要向保密用户发布 sdist；sdist 必然包含 Python 源码。源码仓库可以继续作为内部开发和维护源。

## 从源码直接运行

源码仓库不要求预先生成 .so。项目根目录执行：

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

minimal 需要压力传感器；full 需要完整 GUI 依赖和实际硬件。源码运行不依赖预编译 .so，wheel 用户不需要 PYTHONPATH。

## 同时连接两个压力传感器

双传感器示例模块为 ``tangential.examples.dual_sensor``。它只采集两路压力
传感器，分别计算CoP、角度、梯度和标定结果，并在终端打印两路摘要；不会
启动六维力、GUI或108列CSV保存。

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

查看全部参数：

~~~bash
PYTHONPATH=src python -m tangential.examples.dual_sensor --help
~~~

### 第5步：确认输出并停止

正常运行时每行同时显示A、B两路的端口、序号、ADC总和、CoP和角度，例如：

~~~text
A(...): seq=10 sum=12345 cop=(3.100,5.200) angle=42.00° | B(...): seq=9 sum=12001 cop=(2.900,5.000) angle=39.50°
~~~

正常输出应持续出现A、B两路的 ``seq``、ADC总和、CoP和角度，并且两路的
``seq`` 会随着有效压力帧逐步增加。某一路显示 ``timeout`` 表示该路本次
``read`` 在配置的等待时间内没有收到合法压力帧；它不表示 Bash 命令错误，
可能原因包括端口选错、设备无响应、串口被占用、供电或USB带宽问题。偶发
一次可以继续观察；持续出现时应回到第1步重新确认端口，并单独验证该设备。

按 ``Ctrl+C`` 正常停止。示例会退出读取循环，并通过上下文管理器关闭A/B
两路的读取线程、采集进程、IPC队列和串口；不要直接拔线代替正常退出。

Python调用：

~~~python
from tangential import PressureConfig
from tangential.examples.dual_sensor import run

run(
    PressureConfig(
        port="/dev/serial/by-id/DEVICE_A_ID",
        target_hz=200,
        frame_queue_size=256,
    ),
    PressureConfig(
        port="/dev/serial/by-id/DEVICE_B_ID",
        target_hz=200,
        frame_queue_size=256,
    ),
)
~~~

每一路都有独立串口、采集进程、IPC队列、读取线程、CoP状态机和标定处理器，
一个设备的读取超时不会占用另一个设备的串口消费者。软件状态互相隔离，但
USB控制器带宽、CPU调度和供电仍是共享硬件资源，实际帧率应分别验收。按
``Ctrl+C`` 退出时，两路进程、线程和串口都会通过上下文管理器关闭。

### 常见错误

| 现象 | 原因 | 处理方法 |
| --- | --- | --- |
| Bash报告 ``unexpected token newline`` | 原样复制了带 ``<...>`` 的占位符 | 按第1、2步设置真实 ``PORT_A``/``PORT_B`` |
| ``No such file or directory`` | 设备未连接或端口名已变化 | 重新运行 ``serial.tools.list_ports -v`` |
| ``Permission denied`` | 当前用户没有串口权限 | 加入 ``dialout`` 后重新登录 |
| 提示两个传感器使用同一物理串口 | 两个路径相同，或两个符号链接指向同一设备 | 为A、B选择两个不同设备路径 |
| 某一路持续 ``timeout`` | 端口选错、设备无响应、供电或USB带宽异常 | 单独运行最小示例验证该端口，再检查USB连接 |

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

最小采集：

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

TangentialSensor 返回 TangentialSample。顶层公开 API 还包括 TangentialFrameProcessor、PRSensorAngle、FitCalibrationModel、终端渲染器、训练 API、绘图 API 以及所有配置 dataclass；稳定导出清单位于 src/tangential/__init__.py。

完整应用：

~~~python
from tangential import FullApplicationConfig, run_application

config = FullApplicationConfig()
run_application(config)
~~~

按功能分类配置：

~~~python
from tangential import (
    FullApplicationConfig, PressureConfig, ForceConfig,
    ProcessingConfig, SyncConfig, OutputConfig,
)

config = FullApplicationConfig(
    pressure=PressureConfig(port="/dev/ttyUSB0", target_hz=200),
    force=ForceConfig(port="/dev/ttyUSB1", target_hz=200),
    processing=ProcessingConfig(region_mode="full"),
    sync=SyncConfig(max_time_diff_s=0.015),
    output=OutputConfig(save_dir="./data"),
)
~~~

## 配置与环境变量

所有用户可调参数集中在 src/tangential/config.py：

- PressureConfig：压力端口、波特率、目标频率、响应超时、队列和启动超时。
- ForceConfig：六维力端口、波特率、目标频率、响应超时、队列、启动超时和校零参数。
- CopConfig：动态阈值、背景学习、CoP 稳定、区域和二次精修参数。
- ProcessingConfig：标定维度、区域模式、中值窗口和精修归零策略。
- CalibrationConfig：外部模型路径。
- SyncConfig：主循环频率、GUI 频率、15 ms 匹配窗口、统计周期和缓存容量。
- OutputConfig：CSV 保存目录。
- GuiConfig：GUI 历史数据和区域箭头显示参数。
- TrainingConfig、PlotConfig：训练和离线绘图参数。
- FullApplicationConfig：组合完整应用配置。

可用环境变量示例：

~~~bash
export TANGENTIAL_PRESSURE_PORT=/dev/ttyUSB0
export TANGENTIAL_FORCE_PORT=/dev/ttyUSB1
export TANGENTIAL_MAX_TIME_DIFF_S=0.015
export TANGENTIAL_DATA_DIR=./data
export TANGENTIAL_MODEL_PATH=/path/to/fit_coefs.bin
~~~

配置优先级：

~~~text
CLI 显式参数 > 显式传入的配置对象 > TANGENTIAL_* 环境默认 > config.py 内置默认值
~~~

协议帧头、CRC、固定 12×7/84 通道布局、固定 108 列 CSV 和设备帧长度属于协议不变量，不通过配置修改。

### 修改config.py是否会直接生效

- 使用源码运行时，直接修改 ``src/tangential/config.py`` 中的默认值，会
  影响修改后新建且没有显式覆盖对应字段的配置对象。
- 已经创建的配置对象不会因为文件随后被修改而自动变化，需要重新启动程序。
- 已安装wheel的用户修改源码仓库不会影响已安装包；应传入配置对象、设置
  ``TANGENTIAL_*`` 环境变量，或者修改源码后重新构建并安装wheel。
- 显式传入的 ``PressureConfig`` 等分类配置优先于环境变量和源码默认值。
- 多传感器场景必须为每个传感器分别创建配置对象，不要修改并复用同一个
  可变配置实例。

## 数据和时序不变量

- 压力和六维力均以 200 Hz 为请求目标，单请求在途；响应较慢时实际频率自然下降，不插值、不重复请求补发。
- 合法压力帧解析完成后立即记录真实 rx_t。rel_ms 和 delta_ms 基于真实压力接收时间，不由 GUI 刷新或 CSV 写入节拍生成。
- 压力帧是主顺序；每个合法压力帧最多处理和保存一次。
- 六维力帧最多匹配一次，匹配窗口为 abs(force_t - press_t) <= 0.015 秒。
- 力通道不可用时，压力帧仍逐行保存，力和同步字段写 NaN。
- 双传感器模式下，压力帧超过15 ms仍未匹配时不写CSV，但仍推进CoP状态机并更新GUI；这是当前数据语义，不能在文档或调用方中误写成NaN行。
- 压力设备必需，六维力设备可选；启动校零和运行期重新归零使用普通力数据帧，不发送额外置零命令。
- CSV 由唯一的 TABLE_CSV_HEADER 和 build_csv_row 生成，保持 108 列和既有模型输出。

## 构建 wheel

构建系统在 pyproject.toml 中声明 Cython>=3.1,<4 为构建依赖：

~~~bash
python -m pip install -r requirements.txt
python -m pip wheel . --no-deps --no-build-isolation -w dist
~~~

构建结果：

~~~text
dist/tangential_sensor-0.3.0-cp311-cp311-linux_x86_64.whl
~~~

构建时会生成多个内部 .so；生成的 C 文件位于 build/。build/、dist/、*.egg-info/ 和 .so 都是构建产物，不提交 Git。源码仓库仍保留完整 .py，可在没有 .so 的情况下运行。

`.so` 的位置随使用阶段不同：

- 本地构建缓存：``build/lib.linux-x86_64-cpython-311/tangential/...``。
- 最终wheel内部：``tangential/runtime/``、``acquisition/``、``sensors/``、
  ``processing/`` 和 ``storage/`` 下的同名扩展。
- 安装后：当前Python环境的 ``site-packages/tangential/...``。

安装后可以查询实际加载路径：

~~~bash
python -c "import tangential.sensors.pressure as m; print(m.__file__)"
~~~

输出应以 ``pressure.cpython-311-x86_64-linux-gnu.so`` 结尾。不要手工复制
``build/`` 中的单个扩展；应安装完整wheel，以保证公开Python层、类型声明和
``fit_coefs.bin`` 版本一致。

检查 wheel 是否符合交付边界：

~~~bash
unzip -l dist/tangential_sensor-0.3.0-cp311-cp311-linux_x86_64.whl
~~~

应看到9个内部 .so、9个同名 .pyi、py.typed 和 resources/fit_coefs.bin；不应看到这些内部模块的 .py、生成的 .c/.cpp 或外部 share/ 模型目录。

## 二次开发与保密边界

用户可以：

- 从 tangential 顶层导入公开 API 和配置对象。
- 编写自己的采集、分析和 GUI 脚本。
- 复制 examples/minimal.py 或 examples/full.py 作为应用入口。
- 使用 tools 中的训练和绘图 Python API。

内部采集、同步、协议、算法和 CSV 实现以 .so 形式随 wheel 交付，普通 wheel 用户不直接获得这些 .py 源码；源码仓库则完整保留它们，供内部维护和有权限的开发者二次修改。内部 .so 和 fit_coefs.bin 都可能被逆向或提取，因此这是提高获取门槛和控制交付边界，不是绝对保密。具体使用、逆向和再分发限制还应通过许可证约束。

如果二次开发只组合现有能力，优先使用 tangential 顶层 API 和 config.py；如果需要修改串口协议、CoP、同步或CSV实现，应使用私有源码仓库，不要修改已安装 wheel 内的文件。

## 测试、回退与工作区检查

完整测试：

~~~bash
PYTHONPATH=src \
QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -q
~~~

分发测试会在临时目录重新构建并隔离安装 wheel，验证内部实现确实从 .so 加载。发布前必须同时通过源码模式和 wheel 安装模式，不能只验证其中一种。

源码语法和差异检查：

~~~bash
PYTHONPATH=src python -m compileall -q src/tangential tests
git diff --check
~~~

Git 检查和回退：

~~~bash
git status --short
git log --oneline -n 10
git revert <commit-hash>
~~~

git revert 会创建反向提交并保留历史；不要使用 git reset --hard 覆盖未保存的用户修改。
