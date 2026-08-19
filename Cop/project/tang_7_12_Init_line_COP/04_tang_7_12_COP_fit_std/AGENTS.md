# 04_tang_7_12_COP_fit_std：Agent 工作手册

本文按当前 `0.2.0` 代码编写，是本目录中 Agent 修改、测试和交付的执行约束。修改前先核对实际实现；不要恢复已删除的旧根目录入口，也不要在多个模块复制同一套协议、算法或 CSV 格式。

## 1. 项目定位与分发边界

本项目是 12×7 PZT 压力阵列与可选六维力传感器的采集、CoP/角度/梯度计算、切向力标定、实时 GUI 和离线分析 SDK。

- 唯一正式源码目录是 `src/tangential/`。
- 用户唯一交付物是 wheel；当前包名为 `tangential-sensor`，版本为 `0.2.0`。
- Python 最低版本为 3.11。
- 核心依赖由 `pyproject.toml` 声明：NumPy、SciPy、pyserial；可导入最小 API 和串口采集。
- GUI/绘图依赖仅属于 `full` extra：PyQt5、PyQtGraph、Matplotlib。
- `requirements.txt` 是完整开发/GUI 环境锁定清单，不替代 wheel 元数据。
- 根目录不再提供旧版 `main.py`、`example.py`、`fit.py`、`plot_static.py` 或兼容导入层；源码和测试统一使用 `tangential.*`。

## 2. 当前目录树与文件职责

目录树只列当前正式源码、配置和测试文件；`__pycache__`、构建缓存和本地生成的 `dist/` 不属于源码架构。

```text
04_tang_7_12_COP_fit_std/
├── .gitignore                         # 忽略 __pycache__、字节码、build/dist 和 egg-info 构建/缓存产物
├── .vscode/
│   └── settings.json                  # VS Code 选择 Conda 作为 Python 环境管理器和包管理器
├── AGENTS.md                         # 本 Agent 工作手册：架构、不变量、修改路由和验收命令
├── MANIFEST.in                       # 源码分发时包含 readme、Python 源码和 package 资源
├── pyproject.toml                    # wheel 元数据、依赖、src 布局、entry point 和 package-data
├── readme.md                         # 对外安装、CLI、API、设备和运行说明
├── requirements.txt                  # 完整开发/GUI 环境的锁定依赖
├── src/
│   └── tangential/
│       ├── __init__.py               # 公共 Python API 导出边界和版本；不得隐式加载 Qt/Matplotlib
│       ├── api.py                    # 最小压力采集 API、Sample、单帧处理和终端渲染
│       ├── cli.py                    # 唯一命令行入口及 example/app/plot/fit 四个子命令
│       ├── config.py                 # 默认端口、保存目录、模型路径和完整应用配置
│       ├── full.py                   # 完整采集会话、线程消费、同步、CSV、GUI 调度和资源清理
│       ├── plotting.py               # 离线 CSV 读取、列解析、误差计算、绘图和完整分析
│       ├── training.py               # 离线数据筛选、拟合、模型写出、评估报告和可选 CSV 写回
│       ├── acquisition/
│       │   ├── __init__.py           # 导出缓存/同步 API
│       │   └── buffer.py             # TimestampedBuffer、seq 顺序消费和最近时间帧匹配
│       ├── gui/
│       │   ├── __init__.py           # 可选 GUI 子包标记；不导出核心 API
│       │   └── realtime.py           # PyQtGraph 实时窗口、阵列网格、箭头、曲线和历史数据
│       ├── processing/
│       │   ├── __init__.py           # 导出 CoP 和运行时标定类
│       │   ├── calibration.py        # fit_coefs.bin 解析、模型预测和 package resource 加载
│       │   └── cop.py                # PRSensorAngle：阈值、CoP、状态机、梯度、区域和精修
│       ├── resources/
│       │   ├── __init__.py           # 静态资源 package，使资源可由 importlib.resources 定位
│       │   └── fit_coefs.bin         # wheel 内置拟合模型；运行时不可改写
│       ├── sensors/
│       │   ├── __init__.py           # 导出 PressureSensor、SixAxisForceSensor
│       │   ├── force.py              # 六维力串口协议、200 Hz 轮询、帧解析、校零、独立进程和统计
│       │   └── pressure.py           # PZT 串口协议、200 Hz 轮询、帧解析、CRC、独立进程和统计
│       └── storage/
│           ├── __init__.py           # 导出 CSV 表头、行构造和文件初始化函数
│           └── csv.py                # 唯一 108 列表头、CSV 路径、文件初始化和行构造
└── tests/
    ├── test_calibration_multidim.py  # 多输入 poly、符号分段和混合标定预测
    ├── test_cli.py                   # CLI 版本/帮助、参数映射、子命令输出、退出码和异常
    ├── test_data.py                  # 压力/六维力协议、分包粘包、CRC、超时、200 Hz 调度、队列和缓存
    ├── test_distribution.py          # wheel 元数据、资源、entry point、旧入口缺失和隔离安装
    ├── test_main_integration.py      # 完整会话：必需压力设备、力通道降级、匹配、异常和资源清理
    ├── test_model_and_table.py       # 内置模型回归预测和 108 列 CSV 契约
    ├── test_plot_and_gui.py          # CSV 列解析、空数据和离屏实时 GUI 箭头清理
    ├── test_plotting.py              # 离线绘图 API、真实表头、行区间、PNG、完整分析和角度误差
    ├── test_stage1_resources.py      # package resource 模型、环境覆盖、默认目录、端口传递
    ├── test_tangential_api.py        # 最小 API、Sample/处理器、终端渲染、生命周期和无 GUI 导入
    └── test_training.py              # 训练配置、valid 筛选、多维拟合、模型加载和写回保护
```

`src/tangential/__init__.py` 是公共导出边界，不是普通聚合目录：用户应从这里导入 `TangentialSensor`、`TangentialSample`、`TangentialFrameProcessor`、`FitCalibrationModel`、`FullApplicationConfig`、训练和绘图 API。子模块是实现分层；只有明确列出的符号才视为稳定公共接口。

## 3. 四条数据流

### 3.1 最小 `example`

`tangential example` → `TangentialSensorAPI` → `PressureSensor.read_frame()` → `decode()` 得到 84 个 ADC → `TangentialFrameProcessor` → `PRSensorAngle`/标定模型 → `TangentialSample` → `FixedTerminalRenderer`。

每个合法压力帧只处理一次，终端显示 12×7 原始 ADC、min/max/sum/mean、CoP、角度和标定结果。此路径不启动六维力、CSV 或 Qt GUI。

### 3.2 完整 `app`

```text
PZT 串口 ──> PressureSensor spawn 进程
                         │
六维力串口 ─> SixAxisForceSensor spawn 进程
                         │  各自 read_frame()，带 rx_t/request_seq
                         v
              父进程 TimestampedBuffer
                         │
              压力 seq 顺序驱动处理
                         │
        CoP / 梯度 / 状态机 / 标定 / GUI 最新画面
                         │
      六维力 find_closest：15 ms 内一对一匹配
                         │
           build_csv_row：固定 108 列 CSV
```

生产采集由 `FullAcquisitionSession` 管理，`FullApplicationRunner` 创建 Qt 应用和窗口，`acquisition_loop()` 驱动 `start`、检查错误、处理压力帧、排空匹配、统计、绘图和最终清理。压力设备必需；六维力连接或校零失败时降级为压力模式，力相关字段写 `NaN`。GUI 最高 60 Hz，只显示最新状态，不参与串口读取或时间戳生成。

### 3.3 `plot`

`tangential plot` → `PlotConfig` → `resolve_csvs/load_csv` 按真实表头解析 → 列名/列号、行范围和 `rel_ms` 配置 → `plot_csv()` 或 `plot_full_analysis()` → PNG，必要时另写角度/误差 CSV。空文件、缺列、坏行和非法范围必须给出明确错误。

### 3.4 `fit`

`tangential fit` → `TrainingConfig` → 读取 XY/Z CSV → 优先使用 `valid != 0`，无 `valid` 时回退 `CoP_state != 0` → 按配置拟合 `sym_log`、`sym_exp`、`exp_log`、`exp`、`poly`、`sigmoid` 或 `pchip` → 保存 `fit_coefs.bin` 和可选评估图。默认不修改输入 CSV；只有显式 `--write-back` 才写回，已有目标必须再加 `--force`。

## 4. 关键公开类和函数

### 公共 API：`api.py` 与顶层 `__init__.py`

- `TangentialSensor` / `TangentialSensorAPI`：管理压力传感器生命周期，读取并解码合法帧，返回 `TangentialSample`；支持 `pressure_port`、外部 `model_path`、上下文管理和幂等 `close()`。
- `TangentialSample`：保存 84 通道一维/12×7 矩阵、梯度、统计值、CoP、角度、状态、时间戳、seq、标定结果和显示状态。
- `TangentialFrameProcessor`：单帧算法编排；复用 `PRSensorAngle` 和 `FitCalibrationModel`，不复制 CoP 或拟合公式。
- `compute_vector_angle()`、`angle_difference()`：统一二维方向角和环绕误差计算。
- `FixedTerminalRenderer`、`format_terminal_sample()`：固定布局终端输出，每帧只写入并刷新一次。

### 传感器与采集：`sensors/`、`acquisition/`

- `PressureSensor`：921600 波特率、现有 14 B 请求、200 Hz/5 ms 调度上限、单请求在途、每轮最多 50 ms 响应等待、分包/粘包/噪声/CRC/状态错误恢复、84 通道 `decode()`、`read_frame()` 和时序统计。
- `SixAxisForceSensor`：460800 波特率、`49 AA ... 0D 0A` 28 B 普通帧、独立 spawn 进程、持久化解析缓存、普通帧校零、运行期零偏更新和时序统计。
- `TimestampedBuffer`：带锁 deque；`append()` 自动生成单调 seq，`get_after(seq)` 顺序取未消费帧，`find_closest()` 查找未使用时间帧。
- `match_closest()`：对 `TimestampedBuffer` 的 15 ms 一对一匹配薄封装。

### 处理与标定：`processing/`

- `PRSensorAngle`：动态阈值、CoP、接触状态、origin 锁定、二次精修、压力梯度、区域分割/区域 CoP 和角度。
- `FitCalibrationModel`：从外部路径或内置 package resource 读取模型，执行现有标定模型预测；`apply_fit_predict_multi()` 和模型解析函数属于运行时实现。

### 完整应用：`full.py`

- `FullApplicationConfig`：设备端口、模型/保存路径、采样/绘图频率、匹配窗口、校零和区域配置。
- `PressureThread`、`ForceThread`：父进程中的消费线程，从各自传感器进程接口读取带时间戳帧并写入对应 `TimestampedBuffer`，通过 `error` 暴露线程异常。
- `FullAcquisitionSession`：完整会话的启动、压力驱动处理、force 匹配、CSV 写入、统计、重归零、GUI 更新、错误检查和关闭。
- `FullApplicationRunner`：创建/管理 Qt 应用和实时窗口，并执行会话。
- `acquisition_loop()`：完整应用的显式循环编排；不要把循环隐藏到新的 API 中。

### 离线能力：`training.py`、`plotting.py`

- `TrainingConfig`、`TrainingResult`、`train_model()`：训练参数、训练产物和拟合流程。
- `PlotConfig`、`PlotResult`、`plot_csv()`、`plot_full_analysis()`：绘图参数、输出结果、普通绘图和 108 列完整分析。
- `scan_csv()`、`list_files()`、`resolve_csvs()`、`load_csv()`、`resolve_column()`：文件发现、真实表头解析和输入校验。

### CSV 和 GUI：`storage/`、`gui/`

- `TABLE_CSV_HEADER`：唯一 108 列表头。
- `auto_get_csv_path()`、`init_csv_file()`、`build_csv_row()`：创建不重复文件、写表头和按既定顺序构造行；禁止在其他模块手工拼 108 列。
- `RealTimePlot`：PyQtGraph 实时窗口；`CellGridItem` 绘制 12×7 阵列和区域，`GridLinesItem` 绘制网格。该模块只能按需导入。

## 5. 公共 API、CLI 和内部实现边界

- 公共 Python API：从 `tangential` 顶层导入；核心采集和处理不应要求 Qt 或 Matplotlib。
- CLI 入口：`pyproject.toml` 将 `tangential` 命令注册到 `tangential.cli:main`；命令内部按子命令惰性导入 GUI、绘图或训练依赖。
- 内部实现：`sensors`、`acquisition`、`processing`、`storage`、`full` 的未列出私有函数可调整，但必须保持上层公开契约和数据不变量。
- `import tangential` 不得隐式加载 `pyqtgraph`、PyQt5 或 `matplotlib`。`full` 仅在运行 `app` 时加载 Qt；`plot`/绘图函数仅在真正绘图时加载 Matplotlib。
- 不允许新增第二套传感器解析器、CoP 公式、标定预测器或 CSV 行格式；优先复用已有定义。

## 6. 配置、资源和时间参数

- 默认压力端口：`/dev/ttyUSB0`；默认六维力端口：`/dev/ttyUSB1`。
- 默认输出目录：当前工作目录下的 `data/`；可用 `TANGENTIAL_DATA_DIR` 覆盖。
- `TANGENTIAL_MODEL_PATH` 指定外部模型；未设置时使用 wheel 内置模型。
- 压力和六维力目标采样配置均为 200 Hz、5 ms 周期；设备响应较慢时实际频率自然下降，不插值、不重复写伪造时间。
- 压力响应单轮最多等待 50 ms；压力帧完成合法解析后立即用 `time.perf_counter()` 记录 `rx_t`。
- 压力为必需设备；六维力是可选设备，启动校零和运行期重新归零都使用普通数据帧，不发送额外“置零命令”。校零默认收集 10 帧，超时 1 s 则禁用力通道。
- 压力与六维力同步窗口为 `0.015 s`；GUI 更新上限为 60 Hz。
- `src/tangential/resources/fit_coefs.bin` 是静态 package data。`pyproject.toml` 通过 `[tool.setuptools.package-data]` 声明，`MANIFEST.in` 同步声明；运行时必须通过 `importlib.resources.files("tangential.resources")` 加载。

## 7. 必须维护的不变量

- 压力协议：921600 波特率、14 B 请求、84 个原始线序 ADC；CRC、状态、长度和分包/粘包恢复不可被静默放宽。
- 力协议：460800 波特率、28 B 普通帧、持久化接收缓存；串口只能有一个消费者。
- 每个合法压力帧按 `request_seq`/缓存 `seq` 顺序最多处理和保存一次；父进程不能只取最新帧而跳过中间帧。
- 每个力帧最多匹配一次；匹配必须满足绝对时间差不超过 15 ms。匹配失败时保留压力行，力字段按当前逻辑为 `NaN`。
- CSV 必须由 `TABLE_CSV_HEADER` 和 `build_csv_row()` 产生，固定 108 列、固定列顺序。
- `rel_ms` 从首个已保存压力行的真实 `rx_t` 计算；`delta_ms` 是相邻已保存压力行的真实时间差；首行均为 0。禁止固定网格、插值或重采样改变采集时间语义。
- 六维力校零只从已接收的新帧计算；不可启动第二个串口消费者或并发归零任务竞争零点。
- 线程、spawn 进程、停止事件、串口、CSV 和 Qt 窗口必须在异常、Ctrl+C、窗口关闭和无数据退出时通过 `try/finally` 释放。
- `fit_coefs.bin` 的现有模型格式和回归预测必须保持；除非用户明确要求，不重新训练或改变模型输出。

## 8. 修改路由与联动测试

先定位唯一实现，再修改对应模块：

| 需求 | 首选修改位置 | 必须联动检查 |
| --- | --- | --- |
| 压力请求、串口帧、CRC、调度、队列 | `sensors/pressure.py` | `tests/test_data.py`、分发测试、完整会话测试 |
| 六维力帧、校零、独立进程 | `sensors/force.py` | `tests/test_data.py`、`tests/test_main_integration.py` |
| seq、缓存、时间匹配 | `acquisition/buffer.py` | `tests/test_data.py`、匹配/CSV 集成测试 |
| CoP、阈值、状态机、梯度、区域 | `processing/cop.py` | `tests/test_tangential_api.py`、完整 GUI/集成测试 |
| 模型读取、预测或模型格式 | `processing/calibration.py`、`resources/fit_coefs.bin` | `test_model_and_table.py`、`test_calibration_multidim.py`、资源测试 |
| 最小 API 单帧结果 | `api.py` | `test_tangential_api.py`，确认不加载 Qt/Matplotlib |
| 完整采集、同步、CSV 生命周期、GUI 调度 | `full.py` | `test_main_integration.py`、`test_data.py`、离屏 GUI 测试 |
| 108 列表头或行顺序 | `storage/csv.py` | `test_model_and_table.py`、`test_plotting.py`、完整集成测试 |
| 实时显示 | `gui/realtime.py` | `test_plot_and_gui.py`，保持 GUI 不参与采集时钟 |
| 离线绘图和 CSV 解析 | `plotting.py` | `test_plotting.py`、`test_plot_and_gui.py` |
| 拟合训练和写回保护 | `training.py` | `test_training.py`、`test_cli.py`、模型回归测试 |
| CLI 参数、退出码、懒加载 | `cli.py` | `test_cli.py`、wheel 隔离安装测试 |
| 公共导出或版本 | `__init__.py`、`pyproject.toml` | `test_tangential_api.py`、`test_distribution.py` |
| 默认端口、目录、环境变量 | `config.py`、必要时 `cli.py` | `test_stage1_resources.py`、`test_cli.py` |

修改后不要在调用方重新实现算法；先复用目标模块的公开函数/类，再补薄适配器。涉及协议或时间戳的改动必须同时检查压力/力独立进程、队列、匹配和 CSV 时间列。

## 9. 构建、运行、测试和 Git

在项目根目录执行。开发环境建议使用 Python 3.11，并安装完整依赖：

```bash
python -m pip install -r requirements.txt
```

构建 wheel：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

安装完整功能：

```bash
python -m pip install "./dist/tangential_sensor-0.2.0-py3-none-any.whl[full]"
```

运行四个入口：

```bash
tangential example
tangential app
tangential plot --help
tangential fit --help
```

完整测试和编译检查：

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -v

PYTHONPATH=src python -m compileall -q src/tangential tests
git diff --check
```

分发测试必须在临时源码副本构建，避免污染仓库的 `build/`、`dist/` 和 `egg-info`。至少验证 wheel 包含全部 `tangential` 模块、`tangential/resources/fit_coefs.bin` 和 `dist-info/entry_points.txt`，不包含旧根模块或 `share/` 模型路径；隔离安装后验证四个子命令的 `--help` 和：

```bash
PYTHONPATH=src python -m tangential.cli --version
```

每次完成修改都应检查 `git diff --check` 并创建提交；如果用户明确要求不提交，则保留工作区修改，在最终报告中列出变更文件和验证结果。
