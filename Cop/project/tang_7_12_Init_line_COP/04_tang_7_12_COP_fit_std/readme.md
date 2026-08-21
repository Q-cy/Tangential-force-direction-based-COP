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
