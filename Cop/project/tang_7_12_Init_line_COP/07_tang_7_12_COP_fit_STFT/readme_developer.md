# Tangential Sensor SDK 0.6.0 开发者维护指南

本文仅面向 Tangential SDK 0.6.0 源码维护者，说明内部架构、实现边界、修改路由、测试、构建与排障流程。安装 wheel、使用命令行和公共 API 的用户请阅读 [readme.md](readme.md)。

## 1. 开发目标与不可破坏边界

项目处理可配置行列数的 PZT 压力阵列与可选六维力传感器，默认尺寸为12×7。完整功能包括串口请求—响应采集、时间戳、压力—力匹配、CoP、角度、梯度、区域、滑移、标定、动态通道CSV、实时GUI、CoP实时频谱与时频瀑布、离线训练和绘图。

维护时必须保留以下边界：

- `src/tangential/`是唯一正式源码，不在根目录恢复旧实现或创建第二套算法。
- 压力和六维力协议分别只在`sensors/pressure.py`与`sensors/force.py`实现。
- CoP、区域和梯度只在`processing/cop.py`实现，滑移只在`processing/slip.py`实现，标定只在`processing/calibration.py`实现。
- CSV只能由`storage/csv.py`中的`build_csv_header()`、`TABLE_CSV_HEADER`与`build_csv_row()`生成；默认84通道为108列，任意尺寸为`rows*cols+24`列。

### 1.1 动态阵列尺寸

阵列尺寸的唯一配置源是项目最基础的`ArrayConfig.rows`和`ArrayConfig.cols`，也可分别通过`TANGENTIAL_ARRAY_ROWS`、`TANGENTIAL_ARRAY_COLS`提供环境默认值。两者必须是严格正整数，并满足`2*rows*cols+10<=65535`。请求帧`data[11:13]`自动写入`2*rows*cols`，应答帧的payload长度、返回字节数、接收缓存、uint16解码、CoP/滑移矩阵、终端布局、CSV的`ch1...chN`和GUI图元数量均从同一个`ArrayConfig`推导。`CopConfig`只保存CoP算法参数，不再保存布局。

源码配置示例：

```python
from tangential import ArrayConfig, FullApplicationConfig

config = FullApplicationConfig(array=ArrayConfig(rows=14, cols=5))
```

默认内置`fit_coefs.bin`是在原12×7硬件上训练的。其他尺寸在程序上可以计算`dx/dy/adc_sum`，但标定输出不具备自动的物理可信度，应使用对应阵列重新训练的外部模型。一致性系数同样必须与`rows*cols`一致；旧84通道NPZ只适用于84通道布局，尺寸不符会在启动时明确报错。

实时显示必须和采集状态解耦：Pressure Table、每个单元值、PZT_Z以及有限的PZT_X/Y和Force_X/Y/Z每帧更新；`contact_init`只控制依赖origin的基准点、位移箭头、region和gradient叠加层。未建立origin时不得清空已经收到的压力矩阵。源码验收使用`PYTHONPATH=src python -m tangential.examples.full`；当前`dist/`不在本轮重建范围，不能用于验证本轮源码修改。
- `fit_coefs.bin`是package resource，运行时通过`importlib.resources`加载。
- 一致性系数由`src/tangential/processing/calconsistence.py`离线生成并以安全 NPZ 加载；运行时不读取标定 CSV。
- 压力与六维力的发送、接收和合法帧完成时间使用单调时钟；不得由GUI刷新时间、主循环周期或重采样伪造。
- 一个物理串口只能有一个消费者；启动校零和运行期重新归零都读取普通六维力帧，不向设备发送额外置零命令。
- 每只压力传感器必须拥有独立串口、进程、队列、缓存、处理器、CoP状态、滑移状态、GUI和输出目录。
- 频谱只属于单路完整会话；双路完整会话不得创建频谱分析器、窗口或 NPZ。
- 频谱使用合法压力帧的真实 `rx_t` 和未经过 `dx/dy` 中值滤波的绝对 CoP；它不改变现有 CSV、模型、滑移或 GUI 主窗口数据。
- 源码模式必须可以直接运行；`.so`只是wheel构建产物，不能替代仓库中的`.py`源码。

维护时以实际代码为事实来源，判断顺序为：`pyproject.toml`决定版本、依赖、入口和资源声明；`setup.py`决定编译模块与wheel过滤；`src/tangential/`决定运行时行为；`tests/`把协议、时序、API、GUI、分发和模型回归固化为可执行契约；`readme.md`只说明wheel用户使用方式，本文只说明源码维护事实，两者都不创建第二套默认值或算法定义，也不互相复制。

## 2. 文档分层与维护边界

<table>
<thead>
<tr>
<th style="min-width:180px">文档</th>
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
<td style="white-space:normal"><code>readme_developer.md</code></td>
<td style="white-space:normal">项目源码维护者</td>
<td style="white-space:normal">架构 → 数据流 → 修改路由 → 测试 → 构建与排障</td>
</tr>
<tr>
<td style="white-space:normal"><code>AGENTS.md</code></td>
<td style="white-space:normal">自动化开发代理</td>
<td style="white-space:normal">强制约束 → 不变量 → 验收命令 → 版本控制限制</td>
</tr>
</tbody>
</table>

`pyproject.toml`继续使用`readme.md`作为发行说明，因此用户安装页面只展示wheel用户指南。功能、API、命令或配置修改默认只同步本文中的架构、实现、修改路由和验收信息，不复制用户指南；只有用户明确要求更新wheel用户文档时才修改`readme.md`。

## 3. 一分钟理解整个系统

最小压力API的数据流：

```text
压力串口
→ PressureSensor请求、收包、校验、时间戳
→ decode得到rows×cols通道ADC
→ raw_data →（可选）consistence_data → base_data
→ TangentialSampleProcessor._process_sample()使用base_data计算CoP、梯度、滑移和标定
→ TangentialSample（内部详细结果）
→ TangentialFrameProcessor._to_tangential_frame()挑选八个公开字段
→ TangentialFrame（只公开base_data）
→ 用户循环自行决定终端或其他输出
```

完整应用的数据流：

```text
压力采集进程 → PressureThread → TimestampedBuffer ┐
                                                     ├→ FullAcquisitionSession
六维力采集进程 → ForceThread → TimestampedBuffer ───┘
→ 压力帧按seq顺序推进处理器
→ FullAcquisitionSession直接消费TangentialSample
→ 在15 ms窗口内一对一匹配六维力
→ build_csv_row生成rows×cols+24列（默认108列）
→ CSV与RealTimePlot
```

单路完整应用的频谱数据流独立于原有 CSV 和主窗口刷新：

```text
合法压力帧的 rx_t
→ TangentialSampleProcessor 计算未经过 dx/dy 中值滤波的绝对 cop_x/cop_y
→ CopSpectrumAnalyzer 按真实时间线性重采样到 160 Hz
→ 81点CoP位置 → 80点速度 → 0.5秒周期Hann短窗STFT
→ 2–70 Hz X/Y/合成单边速度谱
→ slip_band_power_ratio = 默认24–28 Hz功率 / 完整2–70 Hz总功率
→ 同一阈值与连续窗时间滞回
→ 旁路逐频点静态基线 → relative_power_db（只显示、记录和研究）
→ 唯一 SpectrumSnapshot → Qt主线程速度谱、相对谱/瀑布和状态文本
→ 会话 finally 原子写出 <CSV stem>_spectrum.npz
```

频谱只消费`state == 2`的有限且严格递增时间帧。第一份完整窗直接输出`STICK`并开始累计ratio证据，不等待旁路基线。接触结束会清除基线；通信gap保留已冻结基线、丢弃未完成基线。`max_gap_s`默认`0.160`秒；不超过该值的间隔线性补齐160 Hz网格，补点不是硬件测量值。频谱状态不覆盖`TangentialMotionState`、`SlipDetector`、方向、origin、CSV或模型结果。分析器保留完整快照历史用于NPZ，窗口只保留最近30秒；手动关闭窗口后采集仍继续。双路运行器不会创建分析器、窗口或NPZ。

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

## 4. 目录结构与职责

本节只面向源码维护者，说明每个目录和文件的唯一职责。`readme.md` 是安装 wheel 后的用户与二次开发文档；`readme_developer.md` 是源码、测试、构建和维护流程文档，两者不再合并成一份“用户与维护者混合指南”。

目录树从项目根目录开始展开；例如压力协议文件的完整相对路径是 `src/tangential/sensors/pressure.py`，不能只根据 `sensors/` 目录名定位实现。

```text
04_tang_7_12_COP_fit_std/
├── AGENTS.md
│   └── Agent 修改边界、架构不变量、文件路由、测试和版本控制限制。
├── readme.md
│   └── wheel 用户安装、公开 API、CLI 和二次开发说明；不描述内部实现。
├── analysis/
│   ├── offline_friction_spectrum_0827.py
│   │   └── 回放0827前三次摩擦CSV，复用运行时CoP和STFT并保存对比图、逐窗特征与JSON报告。
│   ├── offline_friction_spectrum_0827_4.py
│   │   └── 分析0827_4同次接触的静止保持、完整静摩擦和滑动候选阶段；不被运行时导入。
│   ├── offline_friction_feature_evaluation_0827_4.py
│   │   └── 评估CoP速度CV、频谱平坦度、高频功率占比和频谱质心的区分能力与误报连续性。
│   ├── offline_target_band_evaluation_0827_4.py
│   │   └── 评估20–30、22–30、24–28 Hz占比、绝对量和局部峰突出度。
│   └── offline_score2_audit_0827_4.py
│       └── 保存历史fraction-only 0.30阈值及连续窗参数审计，不被实时运行时导入。
├── data/
│   ├── spectrum_feature_research_summary.md
│   │   └── 历史候选特征、公式、阶段、指标、失败原因和后续验证建议。
│   ├── offline_spectrum_0827/
│   │   └── 0827前三次记录的既有离线结果；只读保留。
│   └── offline_spectrum_0827_4/
│       └── 0827_4完整过程的既有离线结果与feature_evaluation子目录；只读保留。
├── readme_developer.md
│   └── 本维护者文档；记录源码架构、内部数据流、测试、构建和维护约束。
├── pyproject.toml
│   └── 包元数据、Python 版本、运行/可选依赖、CLI 入口、资源声明和构建依赖。
├── setup.py
│   └── Cython 扩展模块清单、编译指令、源码过滤和 wheel 构建辅助逻辑。
├── MANIFEST.in
│   └── 源码分发清单；声明资源文件和维护源码在源码包中的包含范围。
├── requirements.txt
│   └── Python 3.11 开发、测试、串口、GUI、绘图和构建环境的完整依赖列表。
├── .gitignore
│   └── 忽略 __pycache__、Cython 中间文件、build/、dist/、wheel 和本地运行产物。
├── .vscode/
│   └── settings.json
│       └── 项目编辑器设置；不参与运行时 API、采集协议或 wheel 内容。
├── src/
│   └── tangential/
│       ├── __init__.py
│       │   └── 包边界和稳定顶层导出；集中声明版本与允许用户从 tangential 导入的名称。
│       ├── api.py
│       │   └── 可读公开 API 门面；把公开对象组织为用户入口，不复制采集或算法实现。
│       ├── application.py
│       │   └── FullApplicationConfig → 完整运行器 → 单路/双路应用退出码；提供公共应用入口。
│       ├── application.pyi
│       │   └── application.py 对应公开函数的静态类型和签名；不是第二套实现，也不执行逻辑。
│       ├── cli.py
│       │   └── 命令行参数 → 配置对象/工具参数 → example、app、dual、plot、fit 公共入口分发。
│       ├── config.py
│       │   └── 环境默认/显式配置 → 分类 dataclass → 启动前校验；集中管理所有可调参数。
│       ├── py.typed
│       │   └── PEP 561 类型提示标记；告诉类型检查器该包随附可用的类型信息。
│       ├── acquisition/
│       │   ├── __init__.py
│       │   │   └── 缓存子包边界和导出；暴露 TimestampedBuffer 与最近帧匹配入口。
│       │   ├── buffer.py
│       │   │   └── 时间戳帧 → 单调 seq 缓存 → 顺序消费/get_after 或一次性最近匹配。
│       │   └── buffer.pyi
│       │       └── buffer.py 对应 Cython .so 的公开静态类型和签名；不是第二套缓存实现。
│       ├── sensors/
│       │   ├── __init__.py
│       │   │   └── 设备子包边界和导出；集中暴露 PressureSensor 与 SixAxisForceSensor。
│       │   ├── pressure.py
│       │   │   └── 串口请求 → 清缓存/分批收包 → 动态帧长、CRC、状态和载荷校验 → rows×cols 通道 ADC、真实接收时间；提供独立压力采集进程入口。
│       │   ├── pressure.pyi
│       │   │   └── pressure.py 对应 Cython .so 的公开静态类型和签名；不是第二套压力协议实现。
│       │   ├── force.py
│       │   │   └── 普通六维力请求 → 分包组帧/帧尾与校验 → 六轴物理量 → 普通帧软件校零；提供独立力采集进程入口。
│       │   └── force.pyi
│       │       └── force.py 对应 Cython .so 的公开静态类型和签名；不是第二套力传感器实现。
│       ├── processing/
│       │   ├── __init__.py
│       │   │   └── 算法子包边界和导出；集中组织 CoP、滑移和标定的公开类型。
│       │   ├── cop.py
│       │   │   └── rows×cols 通道 ADC → 动态阈值、接触区域、origin 和区域合并 → CoP、角度、梯度及状态。
│       │   ├── cop.pyi
│       │   │   └── cop.py 对应 Cython .so 的公开静态类型和签名；不是第二套 CoP 算法实现。
│       │   ├── slip.py
│       │   │   └── 压力矩阵/CoP 短窗 → 斑块相关、位移、EMA 方向和连续帧滞回 → STICK/SLIP 与同步重锚结果。
│       │   ├── slip.pyi
│       │   │   └── slip.py 对应 Cython .so 的公开静态类型和签名；不是第二套滑移状态机实现。
│       │   ├── calibration.py
│       │   │   └── 内置或外部 fit_coefs.bin → 特征和拟合类型解析 → Fx/Fy/Fz 标定预测。
│       │   ├── calibration.pyi
│       │   │   └── calibration.py 对应 Cython .so 的公开静态类型和签名；不是第二套模型加载/预测实现。
│       │   ├── spectrum.py
│       │   │   └── 合法稳定CoP → 真实时间重采样 → 0.5秒CoP速度STFT、相对基线、摩擦检测与30秒历史；只供单路完整会话使用。
│       │   ├── spectrum.pyi
│       │   │   └── spectrum.py对应Cython .so的静态签名；分析器、快照、检测结果和内部摩擦状态不进入顶层用户API。
│       │   ├── calconsistence.py
│       │   │   └── config.py 配置的维护者 CSV → 两状态通道中位数 → 两点仿射系数 → 安全 NPZ；运行时加载并修正 raw_data。
│       │   └── calconsistence.pyi
│       │       └── calconsistence.py 对应 Cython .so 的静态签名；不包含第二套标定实现。
│       ├── runtime/
│       │   ├── __init__.py
│       │   │   └── 运行时子包边界和导出；区分用户可见 Frame 与完整应用内部编排对象。
│       │   ├── sensor.py
│       │   │   └── PressureSensor 帧 → decode → TangentialSampleProcessor._process_sample() → TangentialSample → TangentialFrameProcessor._to_tangential_frame() → 公开 TangentialFrame；每个 TangentialFrameProcessor 门面独占一个样本处理器。
│       │   ├── sensor.pyi
│       │   │   └── sensor.py 对应 Cython .so 的公开静态类型和签名；只声明 TangentialFrame 等公开接口，不声明内部 TangentialSample。
│       │   ├── session.py
│       │   │   └── 配置/设备工厂 → 采集线程或进程 → 内部详细样本、力匹配、CSV、GUI、统计和资源清理；承载完整应用会话与单/双路运行器。
│       │   ├── session.pyi
│       │   │   └── session.py 对应 Cython .so 的公开静态类型和签名；不是第二套完整会话实现。
│       │   ├── synchronization.py
│       │   │   └── 压力时间戳与力缓存 → 15 ms 窗口内一对一匹配；只提供同步薄适配层，不读取串口。
│       │   └── synchronization.pyi
│       │       └── synchronization.py 对应 Cython .so 的公开静态类型和签名；不是第二套同步算法实现。
│       ├── storage/
│       │   ├── __init__.py
│       │   │   └── 存储子包边界和导出；集中暴露固定 CSV 表头与行构造入口。
│       │   ├── csv.py
│       │   │   └── 内部压力/力结果 → 动态通道字段映射 → 唯一 CSV 表头和数据行，并提供 CSV→同stem PNG 命名；不决定采样节拍。
│       │   └── csv.pyi
│       │       └── csv.py 对应 Cython .so 的公开静态类型和签名；不是第二套 CSV 格式实现。
│       ├── gui/
│       │   ├── __init__.py
│       │   │   └── GUI 子包边界；保持基础 import tangential 不主动加载 Qt 和绘图库。
│       │   ├── realtime.py
│       │   │   └── 最新样本/历史序列 → PyQtGraph 图元、压力快照和方向/力箭头 → 主实时窗口与分析图；不读取串口。
│       │   └── spectrum.py
│       │       └── 线程安全频谱快照 → Qt 主线程三条频谱曲线和时频瀑布图；忽略频段保留数据并用背景框标识，不执行 FFT 或读取串口。
│       ├── tools/
│       │   ├── __init__.py
│       │   │   └── 离线工具子包边界；供 CLI 按需加载训练和绘图模块。
│       │   ├── training.py
│       │   │   └── 训练 CSV/有效行 → 拟合参数和评估 → fit_coefs.bin 或显式写回目标；不参与实时采集。
│       │   └── plotting.py
│       │       └── CSV 实际表头/列选择 → 行范围与分析计算 → 静态图、同CSV stem的完整分析图和结果对象；不参与实时采集。
│       ├── examples/
│       │   ├── __init__.py
│       │   │   └── 示例子包边界；不增加第二套内部业务实现。
│       │   ├── minimal.py
│       │   │   └── PressureConfig → TangentialSensorAPI → 逐帧读取 TangentialFrame → 终端摘要；最小 API 示例唯一循环。
│       │   ├── full.py
│       │   │   └── FullApplicationConfig → run_application；完整示例只展示公共入口，不复制完整循环。
│       │   └── dual_sensor.py
│       │       └── 两组独立端口/配置 → 两个隔离采集会话和 GUI → 独立 CSV 与清理；展示双压力传感器用法。
│       └── resources/
│           ├── __init__.py
│           │   └── package resource 命名空间；使内置模型可通过 importlib.resources 定位。
│           ├── fit_coefs.bin
│           │   └── 随 wheel 安装的静态标定模型；由 calibration.py 作为 package resource 加载，不是 Python 源码。
│           └── consistence_coeffs.npz
│               └── 由维护者按统一配置生成并随 wheel 安装的默认一致性系数；不包含标定 CSV。
└── tests/
    └── 覆盖压力/六维力协议与时序、缓存同步、公开 API、CoP/滑移/标定、CSV、GUI、CLI、训练、wheel 分发和回归契约；测试文件不属于运行时包。
```

`src/tangential/**/__pycache__/*.pyc`、`build/` 和 `dist/` 是运行或构建生成物，不属于正式源码树；`data/` 是采集和离线分析输出目录，不作为包实现。发布 wheel 时，`runtime`、`acquisition`、`sensors`、`processing` 和 `storage` 的 `.py` 由 `setup.py` 编译为同名 Cython 扩展；仓库中的 `.py` 仍是唯一维护源，`.pyi` 只提供静态签名，不能替代实现。

使用这棵树定位修改时，协议采集只改 `src/tangential/sensors/pressure.py` 或 `force.py`；公开 `TangentialFrame` 与内部 `TangentialSample` 的边界只改 `src/tangential/runtime/sensor.py`；CoP、滑移和标定分别改 `processing/cop.py`、`slip.py` 和 `calibration.py`；压力—力同步只改 `acquisition/buffer.py` 与 `runtime/synchronization.py`；108 列 CSV 只改 `storage/csv.py`；实时窗口只改 `gui/realtime.py`；CLI 与示例入口分别改 `cli.py`、`examples/minimal.py`、`full.py` 和 `dual_sensor.py`；训练和离线绘图只改 `tools/training.py` 与 `tools/plotting.py`；构建、扩展过滤和 wheel 内容只改 `pyproject.toml`、`setup.py` 与 `MANIFEST.in`。每类功能先定位这里列出的唯一实现，再更新对应测试和本维护者文档，禁止在调用方复制协议、算法或 CSV 映射。

## 4.1 公共 API 与内部实现边界

`src/tangential/__init__.py`当前真实导出 33 个名称，其中`ArrayConfig`是全项目最基础的公共阵列布局配置。`readme.md`只向普通用户推荐其中 3 个：`TangentialSensorAPI`、`TangentialFrameProcessor`和`TangentialFrame`。一致性标定完全属于维护者内部边界：`ConsistenceCalibrationConfig`只从`tangential.config`导入，`ConsistenceCalibrator`和`fit_consistence`只从`tangential.processing.calconsistence`导入，不进入`tangential`、`tangential.api`或用户CLI。`TangentialSensor`别名已删除，`TangentialSensorAPI`是唯一正式压力采集类名称。

### 用户推荐 API

<table>
<thead>
<tr>
<th style="min-width:180px">公共 API 名称</th>
<th>用户用途/流程</th>
<th>实现入口</th>
<th>返回类型或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TangentialSensorAPI</code></td>
<td style="white-space:normal">管理压力读取生命周期 → 解码 → 调用公开单帧处理门面</td>
<td style="white-space:normal"><code>runtime/sensor.py:TangentialSensorAPI</code></td>
<td style="white-space:normal">逐帧<code>TangentialFrame</code>；<code>close()</code>释放资源</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrame</code></td>
<td style="white-space:normal">保存用户可消费的单帧压力结果</td>
<td style="white-space:normal"><code>runtime/sensor.py:TangentialFrame</code></td>
<td style="white-space:normal">固定八字段的数据类</td>
</tr>
<tr>
<td style="white-space:normal"><code>TangentialFrameProcessor</code></td>
<td style="white-space:normal">已有rows×cols通道ADC → 单帧处理 → <code>TangentialFrame</code></td>
<td style="white-space:normal"><code>runtime/sensor.py:TangentialFrameProcessor.process_frame</code></td>
<td style="white-space:normal"><code>TangentialFrame</code></td>
</tr>
</tbody>
</table>

### 配置 API

下列名称是真实顶层导出，负责完整应用的设备、处理、同步、输出和 GUI 参数；普通用户运行示例时不需要把它们当作单帧采集 API。

<table>
<thead>
<tr>
<th style="min-width:180px">公共 API 名称</th>
<th>用户用途/流程</th>
<th>实现入口</th>
<th>返回类型或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>ArrayConfig</code></td>
<td style="white-space:normal">阵列行列 → shape、通道数、协议字节数和全链路布局</td>
<td style="white-space:normal"><code>config.py:ArrayConfig</code></td>
<td style="white-space:normal">全项目唯一阵列布局对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>FullApplicationConfig</code></td>
<td style="white-space:normal">组合设备、处理、同步、输出和GUI配置</td>
<td style="white-space:normal"><code>config.py:FullApplicationConfig</code></td>
<td style="white-space:normal">完整应用配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>PressureConfig</code></td>
<td style="white-space:normal">配置压力端口、波特率、频率、超时和队列</td>
<td style="white-space:normal"><code>config.py:PressureConfig</code></td>
<td style="white-space:normal">压力配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>ForceConfig</code></td>
<td style="white-space:normal">配置六维力开关、端口、频率、超时和校零</td>
<td style="white-space:normal"><code>config.py:ForceConfig</code></td>
<td style="white-space:normal">六维力配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>CopConfig</code></td>
<td style="white-space:normal">配置CoP相关阈值、稳定和精修参数</td>
<td style="white-space:normal"><code>config.py:CopConfig</code></td>
<td style="white-space:normal">CoP配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>ProcessingConfig</code></td>
<td style="white-space:normal">组合维度、区域模式、滤波和单帧处理策略</td>
<td style="white-space:normal"><code>config.py:ProcessingConfig</code></td>
<td style="white-space:normal">处理配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipConfig</code></td>
<td style="white-space:normal">配置滑移窗口、阈值、平滑和滞回参数</td>
<td style="white-space:normal"><code>config.py:SlipConfig</code></td>
<td style="white-space:normal">滑移配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>CalibrationConfig</code></td>
<td style="white-space:normal">选择默认或外部标定配置</td>
<td style="white-space:normal"><code>config.py:CalibrationConfig</code></td>
<td style="white-space:normal">标定配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>SyncConfig</code></td>
<td style="white-space:normal">配置主循环、GUI、匹配窗口和缓存</td>
<td style="white-space:normal"><code>config.py:SyncConfig</code></td>
<td style="white-space:normal">同步配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>OutputConfig</code></td>
<td style="white-space:normal">配置CSV和分析图的保存目录</td>
<td style="white-space:normal"><code>config.py:OutputConfig</code></td>
<td style="white-space:normal">输出配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>GuiConfig</code></td>
<td style="white-space:normal">配置窗口、历史数据、色阶和显示参数</td>
<td style="white-space:normal"><code>config.py:GuiConfig</code></td>
<td style="white-space:normal">GUI配置对象</td>
</tr>
</tbody>
</table>

### 公共支撑类型

<table>
<thead>
<tr>
<th style="min-width:180px">公共 API 名称</th>
<th>用户用途/流程</th>
<th>实现入口</th>
<th>返回类型或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TangentialMotionState</code></td>
<td style="white-space:normal">读取<code>frame.motion_state</code>并判断接触运动状态</td>
<td style="white-space:normal"><code>processing/slip.py:TangentialMotionState</code></td>
<td style="white-space:normal"><code>NO_CONTACT</code>、<code>STICK</code>或<code>SLIP</code></td>
</tr>
</tbody>
</table>

### 应用与工具 API

<table>
<thead>
<tr>
<th style="min-width:180px">公共 API 名称</th>
<th>用户用途/流程</th>
<th>实现入口</th>
<th>返回类型或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>TrainingConfig</code></td>
<td style="white-space:normal">配置离线训练数据、拟合和输出选项</td>
<td style="white-space:normal"><code>config.py:TrainingConfig</code></td>
<td style="white-space:normal">训练配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>TrainingResult</code></td>
<td style="white-space:normal">读取训练入口返回的模型和评估信息</td>
<td style="white-space:normal"><code>tools/training.py:TrainingResult</code></td>
<td style="white-space:normal">训练结果对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>train_model</code></td>
<td style="white-space:normal">训练数据 → 标定结果和评估信息</td>
<td style="white-space:normal"><code>tools/training.py:train_model</code></td>
<td style="white-space:normal"><code>TrainingResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotConfig</code></td>
<td style="white-space:normal">配置CSV、列、行范围、模式和输出路径</td>
<td style="white-space:normal"><code>config.py:PlotConfig</code></td>
<td style="white-space:normal">绘图配置对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>PlotResult</code></td>
<td style="white-space:normal">读取绘图入口生成的图像和分析路径</td>
<td style="white-space:normal"><code>tools/plotting.py:PlotResult</code></td>
<td style="white-space:normal">绘图结果对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>plot_csv</code></td>
<td style="white-space:normal">CSV表头和绘图配置 → 指定列图像</td>
<td style="white-space:normal"><code>tools/plotting.py:plot_csv</code></td>
<td style="white-space:normal"><code>PlotResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>plot_full_analysis</code></td>
<td style="white-space:normal">完整CSV → 全部分析图和统计结果</td>
<td style="white-space:normal"><code>tools/plotting.py:plot_full_analysis</code></td>
<td style="white-space:normal"><code>PlotResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>run_application</code></td>
<td style="white-space:normal">完整配置 → 单路会话、GUI、CSV和清理</td>
<td style="white-space:normal"><code>application.py:run_application</code></td>
<td style="white-space:normal">正常退出返回<code>int 0</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>run_dual_application</code></td>
<td style="white-space:normal">两份独立配置 → 两路隔离会话、GUI和CSV</td>
<td style="white-space:normal"><code>application.py:run_dual_application</code></td>
<td style="white-space:normal">正常退出返回<code>int 0</code></td>
</tr>
</tbody>
</table>

### 高级/底层公共 API

下表 9 个名称仍保留在源码、顶层导出、运行时导出和类型声明中，主要由高层采集、完整会话、离线工具或维护者代码使用；它们不是普通用户的首选入口。它们的实现入口和内部用途必须保持稳定，新增调用应优先复用前三项推荐 API。

<table>
<thead>
<tr>
<th style="min-width:180px">公共 API 名称</th>
<th>用户用途/流程</th>
<th>实现入口</th>
<th>返回类型或输出</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>FitCalibrationModel</code></td>
<td style="white-space:normal">加载内置或外部标定模型 → 对压力结果预测切向力</td>
<td style="white-space:normal"><code>processing/calibration.py:FitCalibrationModel</code></td>
<td style="white-space:normal">模型对象；<code>predict()</code>输出标定值</td>
</tr>
<tr>
<td style="white-space:normal"><code>PRSensorAngle</code></td>
<td style="white-space:normal">压力矩阵 → CoP、接触状态、区域、梯度和角度</td>
<td style="white-space:normal"><code>processing/cop.py:PRSensorAngle</code></td>
<td style="white-space:normal">有状态 CoP 处理器及其计算结果</td>
</tr>
<tr>
<td style="white-space:normal"><code>PressureSensor</code></td>
<td style="white-space:normal">压力串口 → CRC/状态校验 → 合法压力帧</td>
<td style="white-space:normal"><code>sensors/pressure.py:PressureSensor</code></td>
<td style="white-space:normal">原始帧、时间戳和时序统计</td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipDetector</code></td>
<td style="white-space:normal">压力斑块短窗 → 滑移距离、方向、置信度和状态</td>
<td style="white-space:normal"><code>processing/slip.py:SlipDetector</code></td>
<td style="white-space:normal"><code>SlipResult</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>SlipResult</code></td>
<td style="white-space:normal">保存一次滑移检测的距离、方向、置信度和状态</td>
<td style="white-space:normal"><code>processing/slip.py:SlipResult</code></td>
<td style="white-space:normal">不可变滑移结果对象</td>
</tr>
<tr>
<td style="white-space:normal"><code>compute_vector_angle</code></td>
<td style="white-space:normal">二维向量 → 方向角</td>
<td style="white-space:normal"><code>runtime/sensor.py:compute_vector_angle</code></td>
<td style="white-space:normal">角度 <code>float</code>，单位为度</td>
</tr>
<tr>
<td style="white-space:normal"><code>angle_difference</code></td>
<td style="white-space:normal">两个方向角 → 最小环绕差值</td>
<td style="white-space:normal"><code>runtime/sensor.py:angle_difference</code></td>
<td style="white-space:normal">角度差 <code>float</code>，单位为度</td>
</tr>
<tr>
<td style="white-space:normal"><code>FixedTerminalRenderer</code></td>
<td style="white-space:normal"><code>TangentialFrame</code> → 固定终端布局</td>
<td style="white-space:normal"><code>runtime/sensor.py:FixedTerminalRenderer</code></td>
<td style="white-space:normal"><code>render()</code>输出文本并刷新终端</td>
</tr>
<tr>
<td style="white-space:normal"><code>format_terminal_sample</code></td>
<td style="white-space:normal"><code>TangentialFrame</code> → 固定布局文本</td>
<td style="white-space:normal"><code>runtime/sensor.py:format_terminal_sample</code></td>
<td style="white-space:normal">格式化后的 <code>str</code></td>
</tr>
</tbody>
</table>

## 5. 推荐源码阅读顺序

第一次阅读不要从最长的`runtime/session.py`或`processing/cop.py`开始，建议按以下顺序建立心智模型：

1. `src/tangential/__init__.py`：先确认稳定公共名称。
2. `src/tangential/config.py`：理解设备、处理、同步、输出和GUI有哪些可调边界。
3. `src/tangential/examples/minimal.py`：观察最小用户循环。
4. `src/tangential/runtime/sensor.py`：理解`TangentialSampleProcessor`的完整处理、`TangentialFrameProcessor`的公开薄门面、`TangentialFrame`和高级传感器API。
5. `src/tangential/sensors/pressure.py`：理解压力请求、收包、校验、时间戳和独立进程。
6. `src/tangential/processing/cop.py`、`slip.py`、`calibration.py`：分别阅读CoP状态、滑移状态和模型预测。
7. `src/tangential/processing/calconsistence.py`：理解维护者离线拟合、安全NPZ和运行时一致性修正；默认输入、输出和开关只在`config.py`的统一配置类中维护。
8. `src/tangential/storage/csv.py`：确认完整应用最终写出的动态通道列与固定24个结果列语义。
9. `src/tangential/acquisition/buffer.py`与`runtime/synchronization.py`：理解seq消费和一次性时间匹配。
10. `src/tangential/runtime/session.py`：把设备、处理、匹配、CSV、GUI和清理串起来。
11. `src/tangential/application.py`、`src/tangential/examples/full.py`、`src/tangential/examples/dual_sensor.py`与`src/tangential/cli.py`：理解公共入口如何复用完整会话。
12. `src/tangential/tools/training.py`与`plotting.py`：最后阅读离线工具和模型生产流程。
13. `setup.py` 与 `tests/`：理解源码怎样变成独立 wheel，并由分发契约验证 wheel 内容。

## 6. 分层职责总表

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
<td style="white-space:normal">发送请求 → 接收并校验响应 → rows×cols×2字节payload与真实时间戳</td>
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
<td style="white-space:normal">rows×cols通道ADC → 动态阈值/接触/origin/区域 → CoP、角度和梯度</td>
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
<td style="white-space:normal"><code>processing/calconsistence.py</code></td>
<td style="white-space:normal">维护者CSV与统一配置 → 两状态中位数仿射拟合 → 安全NPZ；运行时NPZ与raw_data → base_data</td>
<td style="white-space:normal">用户CLI、硬件采集和替换未经授权的标定数据</td>
</tr>
<tr>
<td style="white-space:normal"><code>runtime/sensor.py</code></td>
<td style="white-space:normal">PressureSensor帧 → TangentialSampleProcessor → TangentialSample → TangentialFrameProcessor → TangentialFrame</td>
<td style="white-space:normal">完整Qt生命周期、六维力匹配和CSV</td>
</tr>
<tr>
<td style="white-space:normal"><code>runtime/session.py</code></td>
<td style="white-space:normal">压力缓存与可选力缓存 → 顺序处理与匹配 → CSV、GUI、统计和统一清理</td>
<td style="white-space:normal">复制协议、CoP公式和CSV字段定义</td>
</tr>
<tr>
<td style="white-space:normal"><code>storage/csv.py</code></td>
<td style="white-space:normal">压力样本与可选力帧 → 动态通道映射 → rows×cols+24列CSV行</td>
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

## 7. 配置系统

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
<td style="white-space:normal"><code>ArrayConfig</code></td>
<td style="white-space:normal">rows/cols → shape、channel_count、sensor_bytes</td>
<td style="white-space:normal">压力、处理、CSV、终端和GUI全链路</td>
</tr>
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
<td style="white-space:normal"><code>TangentialFrameProcessor</code>、<code>TangentialSampleProcessor</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>ConsistenceCalibrationConfig</code></td>
<td style="white-space:normal">维护者开关/CSV/NPZ/状态/目标/裁剪 → 离线拟合与运行时一致性修正统一配置</td>
<td style="white-space:normal"><code>calconsistence.py</code>与<code>ProcessingConfig.consistence</code></td>
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
<td style="white-space:normal"><code>SpectrumConfig</code></td>
<td style="white-space:normal">160 Hz重采样 → 0.5秒CoP速度STFT → 默认24–28 Hz/2–70 Hz功率占比 → 同阈值连续窗滞回 → GUI与精简NPZ</td>
<td style="white-space:normal"><code>CopSpectrumAnalyzer</code>与<code>SpectrumWindow</code></td>
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

协议帧头、CRC、多字节顺序、设备地址和uint16通道编码属于协议不变量。行列数属于全项目基础配置，统一存放在`ArrayConfig.rows/cols`；帧长度和CSV列数由其动态推导。单次操作参数，例如`read(timeout_s)`的超时，不作为全局配置。

新增配置时必须同步完成：在正确dataclass中增加字段和类型 → 如需环境默认则增加`TANGENTIAL_*`解析 → 在`validate()`或完整配置启动校验中拒绝非法值 → 把配置传到唯一消费者 → 更新本文相应配置说明和测试。不得只增加字段而不让实际运行路径读取它；`readme.md`只在用户明确要求时更新。

## 8. 压力采集实现

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
→ 验证长度、CRC、状态和`rows*cols*2`字节传感器payload
→ 记录rx_t与latency_s
→ 写入request_seq/tx_t/rx_t/latency_s/payload
→ 本轮不足period_s时等待剩余时间，超期则直接进入下一轮并计数
```

解析器支持分包、单轮粘包、前导噪声、错误长度、CRC错误和状态错误恢复。当前策略每轮只接受一个合法响应，轮末清空残留，避免上一轮晚到数据被错误归属到下一请求。

`read_frame()`返回`rows*cols*2`字节payload与时序元数据，`decode()`只执行`rows*cols`个little-endian `uint16`解码并保持设备原始线序。左右翻转、基线、增益、CoP和标定不属于该模块。

重要统计包括`requests`、`frames`、`response_timeouts`、`crc_errors`、`length_errors`、`status_errors`、`serial_read_errors`、`serial_write_errors`、`serial_flush_errors`、`queue_drops`和`schedule_skips`，以及最近发送间隔、接收间隔和响应延迟。目标200 Hz是请求上限；设备响应约6 ms时实际频率约166 Hz属于正常物理结果。

压力驱动的生产结构是“父进程`PressureSensor` → spawn子进程 → 子进程内本地`PressureSensor` → 单一压力I/O线程 → 串口”。父进程只从IPC帧队列读取，父进程的`PressureThread`负责解码并追加到`TimestampedBuffer`；因此业务处理、GUI刷新和CSV写入不会成为串口消费者。

## 8.1 一致性标定实现

一致性标定只供源码维护者使用，不能通过用户 CLI 触发。运行时和离线拟合的全部参数只允许放在 `src/tangential/config.py` 的 `ConsistenceCalibrationConfig` 中，不得再建立第二个平行配置类。路径默认基于 `_SOURCE_PROJECT_ROOT`，不依赖当前工作目录。

### 维护者需要修改的配置

直接编辑 `src/tangential/config.py` 中 `ConsistenceCalibrationConfig` 类体内的字段默认值。下面的字段全部属于同一个配置对象：

<table>
<thead>
<tr>
<th style="min-width:180px">字段</th>
<th>作用</th>
<th>当前默认或填写规则</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space:normal"><code>enabled</code></td>
<td style="white-space:normal">控制实时处理是否加载并应用一致性系数</td>
<td style="white-space:normal"><code>True</code>；关闭时运行时直接使用原始数据</td>
</tr>
<tr>
<td style="white-space:normal"><code>csv_path</code></td>
<td style="white-space:normal">离线拟合的输入 CSV</td>
<td style="white-space:normal">实际路径始终以 <code>config.py</code> 中该字段为准；替换时指向项目 <code>data/</code> 下已确认的 <code>.csv</code> 文件</td>
</tr>
<tr>
<td style="white-space:normal"><code>output_path</code></td>
<td style="white-space:normal">离线拟合生成的 NPZ 输出路径</td>
<td style="white-space:normal">通常为项目根目录 <code>src/tangential/resources/consistence_coeffs.npz</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>coefficients_path</code></td>
<td style="white-space:normal">实时运行时加载的 NPZ 系数</td>
<td style="white-space:normal"><code>None</code> 使用包内资源；填写外部 NPZ 路径则使用外部文件</td>
</tr>
<tr>
<td style="white-space:normal"><code>state_column</code></td>
<td style="white-space:normal">CSV 中表示状态的列名</td>
<td style="white-space:normal">默认 <code>CoP_state</code>，必须与输入表头一致</td>
</tr>
<tr>
<td style="white-space:normal"><code>baseline_state</code></td>
<td style="white-space:normal">基准/卸载状态值</td>
<td style="white-space:normal">默认 <code>0</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>loaded_state</code></td>
<td style="white-space:normal">加载状态值</td>
<td style="white-space:normal">默认 <code>2</code>，不能与 <code>baseline_state</code> 相同</td>
</tr>
<tr>
<td style="white-space:normal"><code>target_min</code>、<code>target_max</code></td>
<td style="white-space:normal">两点拟合的目标范围</td>
<td style="white-space:normal">默认 <code>0.0</code> 和 <code>4000.0</code>，<code>target_max</code> 必须更大</td>
</tr>
<tr>
<td style="white-space:normal"><code>clip_min</code>、<code>clip_max</code></td>
<td style="white-space:normal">运行时修正结果的可选裁剪范围</td>
<td style="white-space:normal">默认下限 <code>0.0</code>、上限为 <code>None</code>；允许把下限设为 <code>None</code></td>
</tr>
<tr>
<td style="white-space:normal"><code>force</code></td>
<td style="white-space:normal">是否覆盖已经存在的输出 NPZ</td>
<td style="white-space:normal">默认 <code>True</code>，无参数源码命令会更新同名文件；需要保护已有文件时显式改为 <code>False</code></td>
</tr>
</tbody>
</table>

例如，维护者只需要在类体中调整路径和参数，保持现有 `field(default_factory=...)` 写法：

```python
# src/tangential/config.py
class ConsistenceCalibrationConfig:
    enabled = True
    csv_path = (_SOURCE_PROJECT_ROOT / "data" / "my_consistence_data.csv").resolve()
    output_path = (_SOURCE_PROJECT_ROOT / "src" / "tangential" / "resources" / "consistence_coeffs.npz").resolve()
    coefficients_path = None
    state_column = "CoP_state"
    baseline_state = 0
    loaded_state = 2
    target_min = 0.0
    target_max = 4000.0
    clip_min = 0.0
    clip_max = None
    force = True
```

上面的代码只展示需要修改的字段；实际源码中的路径默认值通过 `_SOURCE_PROJECT_ROOT` 和 `field(default_factory=...)` 构造，不能把项目根目录写成依赖当前 shell 工作目录的相对路径。输入 CSV 必须包含 `state_column` 以及 `ch1` 到 `ch84`，两种状态都必须有有效、有限数据。

### 生成 NPZ 的源码步骤

1. 在 `ConsistenceCalibrationConfig` 中确认 `csv_path`、状态列、目标范围、裁剪范围和 `output_path`。
2. 默认 `force=True`，输出文件已存在时会直接更新；需要人工保护已有文件时，把 `force` 显式改为 `False`。
3. 在项目根目录执行下面的无参数源码命令：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.processing.calconsistence
```

该命令没有位置参数或选项，不使用 `argparse`，不连接硬件；模块的 `main()` 只构造 `ConsistenceCalibrationConfig()`，读取类中配置的 CSV，调用 `fit_consistence()` 并打印输入/输出路径。默认配置会把 `force=True` 沿 `main() → fit_consistence() → ConsistenceCalibrator.save()` 传递，因此连续运行会覆盖并更新同一路径下的旧 NPZ。不要给用户增加对应 CLI 子命令或路径参数。

CSV 表头会去除首尾空白，再按 `state_column`、`ch1` 到 `ch84` 定位列；每个通道分别计算 `baseline_state` 和 `loaded_state` 的中位数，并执行：

```text
scale = (target_max - target_min) / (loaded_median - baseline_median)
offset = target_min - baseline_median * scale
```

拟合前必须验证 84 个通道、两种状态均有样本、所有值有限且每个加载中位数严格大于对应卸载中位数。输出是 `allow_pickle=False` 可读取的压缩 NPZ，至少包含 `scale`、`offset`、`states`、`targets`、`sample_counts` 和源 CSV 的 `source_sha256`。维护者无参数入口默认覆盖同名 NPZ；底层 `ConsistenceCalibrator.save()` 的 `force` 参数仍默认 `False`，显式配置 `ConsistenceCalibrationConfig(force=False)` 也会在目标已存在时抛出 `FileExistsError`。维护测试使用临时 CSV，不能使用正在增长或未经确认的采集文件。

### 运行时语义和资源选择

- `enabled=True`：每个 `TangentialSampleProcessor` 创建时加载系数，并把每帧 `raw_data` 修正为下游使用的数据；CoP、梯度、滑移、模型、GUI、终端和 CSV 都使用修正后的结果。
- `enabled=False`：运行时不加载系数、不执行修正，`base_data` 直接复制 `raw_data`；不会删除已有 NPZ，也不会自动生成新的 NPZ。离线生成命令是否执行由维护者单独决定，与这个运行期开关分离。
- `coefficients_path=None`：调用 `ConsistenceCalibrator.from_default()`，从 `tangential.resources/consistence_coeffs.npz` 加载包内资源。
- `coefficients_path` 为外部路径：调用 `ConsistenceCalibrator.from_path()` 加载指定 NPZ；每个会话可以使用自己的系数文件。
- `enabled=True` 时，如果内置资源缺失、外部路径不存在、NPZ 损坏、形状错误或包含非有限系数，启动阶段立即失败；完整会话会在创建 CSV 前关闭已创建资源，不得静默回退到未修正数据。`enabled=False` 时不会加载或检查系数文件。

环境变量只作为构造配置对象时的默认覆盖，不能替代类字段的维护配置。当前支持：`TANGENTIAL_CONSISTENCE_ENABLED`、`TANGENTIAL_CONSISTENCE_COEFFICIENTS`、`TANGENTIAL_CONSISTENCE_CLIP_MIN` 和 `TANGENTIAL_CONSISTENCE_CLIP_MAX`。字段显式传值优先于环境变量；环境变量会在 `ConsistenceCalibrationConfig()` 实例化时读取，修改环境变量或源码后必须重新创建配置对象。可以用下面的代码检查最终生效值：

```python
from tangential.config import ConsistenceCalibrationConfig

config = ConsistenceCalibrationConfig()
print(config.enabled, config.coefficients_path, config.clip_min, config.clip_max)
```

### 双传感器的独立配置

双传感器不能共享同一个 `FullApplicationConfig`、处理器或校正器。为 Sensor A/B 分别创建完整配置，并分别设置 `processing.consistence`；即使两路使用相同的 NPZ，也要创建两个独立的配置对象：

```python
from tangential import FullApplicationConfig, run_dual_application

config_a = FullApplicationConfig()
config_b = FullApplicationConfig()

config_a.processing.consistence.enabled = True
config_b.processing.consistence.enabled = True
config_a.processing.consistence.coefficients_path = None
config_b.processing.consistence.coefficients_path = None

# 若两路需要不同系数，则分别填写两个已确认的外部 NPZ 路径。
run_dual_application(config_a, config_b)
```

两份配置还必须使用不同的压力端口、独立输出目录；启用六维力时，力端口也必须不同。修改 `config_a` 不会改变 `config_b`。运行时只加载 NPZ，不读取标定 CSV。

`FullAcquisitionSession.start()`先加载模型、构造 `TangentialSampleProcessor` 并完成系数验证，之后才调用 `auto_get_csv_path()` 和 `init_csv_file()`；若验证失败，`close()` 会关闭已经创建的压力/力传感器，输出目录不会留下空 CSV。

## 8.2 CoP频谱、旁路相对基线与滑移频带功率占比

频谱是单路完整采集的附属分析链，不参与压力请求调度、CoP主状态机、原有空间滑移、标定、动态CSV或主窗口刷新。`FullAcquisitionSession`把同一次`TangentialSample`中未经过`dx/dy`中值滤波的绝对`cop_x/cop_y`与真实`rx_t`交给`CopSpectrumAnalyzer`，不存在第二次CoP计算。

```text
合法压力帧 rx_t、未滤波绝对 cop_x/cop_y、state
→ 只接受 state == required_cop_state（默认2）的有限、严格递增连续段
→ 相邻真实帧 gap <= max_gap_s（默认160 ms）时线性重采样到160 Hz
→ 81个CoP位置点 → 80个速度点 → X/Y分别去均值
→ 0.5秒周期Hann窗 → rfft → 完整2–70 Hz X/Y/合成单边速度幅值谱
├→ slip_band_power_ratio = 24–28 Hz功率 / 完整2–70 Hz总功率
│  → 同一0.30阈值与3/5连续窗时间滞回 → WAITING/STICK/SLIP
└→ 旁路收集1秒逐频点功率中位数并冻结 → relative_power_db
→ 唯一SpectrumSnapshot → GUI速度谱/相对谱/relative dB瀑布/状态 → 新会话NPZ
```

功率逐频点定义为`velocity_amplitude_x**2 + velocity_amplitude_y**2`。目标带边界包含在内，完整分析频带没有ignored mask，所有2–70 Hz频点都进入ratio分母。运行时保留逐频点冻结基线和`relative_power_db=10log10((power+floor)/(baseline+floor))`，但它们只用于显示、NPZ记录和未来研究，绝不进入ratio、阈值或状态。score1/score2、CV、高频辅助占比、质心、prominence和`motion/active/entropy/flatness/peak/flux`仍已删除。

第一份完整0.5秒窗到达前状态为`WAITING`；第一份完整窗立即以`STICK`输出，并把该窗ratio计入连续进入证据，不等待`baseline_duration_s`。STICK中`slip_band_power_ratio >= slip_band_power_ratio_threshold`连续`enter_windows`窗进入SLIP；SLIP中`ratio <`同一阈值连续`exit_windows`窗回到STICK。默认阈值0.30、连续窗3/5只是当前可调初值，不是生产验证结论。该内部状态只进入频谱GUI和NPZ，不覆盖`TangentialMotionState`、`SlipDetector`、方向、origin、CSV或公开`TangentialFrame.motion_state`。

旁路基线从每次接触的第一份完整频谱窗开始收集，达到`baseline_duration_s`后对每个频点取时间中位数并冻结，之后不再更新。基线建立前，快照中的`baseline_power`和`relative_power_db`均为NaN，`baseline_established=False`；GUI显示“相对基线收集中”，但ratio状态照常推进。基线冻结后，GUI显示relative曲线和relative dB瀑布。

状态不匹配、非法数值或时间倒退按接触结束处理：清空短窗、状态及冻结/未完成基线。严格超过`max_gap_s`按通信gap处理：清空短窗和滞回，保留已冻结基线；若基线尚未完成则丢弃部分样本并在新完整窗后重新收集。已经生成的会话快照历史不会删除。频谱窗口只保留最近`history_duration_s`的显示历史，瀑布色阶使用`color_percentile`；手动关闭窗口不停止采集，主窗口关闭时联动关闭。双路运行器强制不创建频谱分析器、窗口或NPZ。

### SpectrumConfig 当前字段

`SpectrumConfig`保留实际依赖：`enabled`、`enabled_in_dual`、`sample_rate_hz`、`window_duration_s`、`update_interval_s`、`analysis_min_frequency_hz`、`analysis_max_frequency_hz`、`slip_band_hz`、`slip_band_power_ratio_threshold`、`enter_windows`、`exit_windows`、`baseline_duration_s`、`baseline_power_floor`、`max_gap_s`、`required_cop_state`、`history_duration_s`、`color_percentile`、`save_npz`、`output_suffix`、`window_width`和`window_height`。每项默认值、单位、环境变量与校验规则写在`src/tangential/config.py`字段注释中。

### SpectrumSnapshot 与新会话 NPZ

`SpectrumSnapshot`字段为`frequency_hz`、`spectrum_time_s`、三份`velocity_amplitude_*`、`baseline_power`、`relative_power_db`、`baseline_established`、`slip_band_power_ratio`、`friction_state`、`threshold`和`revision`。它、`SpectralFrictionState`和`CopSpectrumAnalyzer`都属于内部实现，不从顶层公共API导出。

新会话NPZ写：`frequency_hz (F,) float64`、`spectrum_time_s (T,) float64`、三份`velocity_amplitude_* (T,F) float32`、`baseline_power (T,F) float32`、`relative_power_db (T,F) float32`、`baseline_established (T,) bool`、`slip_band_power_ratio (T,) float64`、`friction_state (T,) int8`、标量`threshold`，以及采样率、窗长、更新周期、分析/滑移频带、进入/退出窗数、基线时长/地板、最大gap、目标CoP状态、窗名与CSV文件名。保存使用同目录临时文件和`os.replace()`原子替换；无快照不创建文件。旧NPZ不迁移，读取方必须按新schema处理新会话文件。

### 历史频谱研究归档

`analysis/`中的离线脚本和`data/offline_spectrum_0827*/`中的既有结果永久保留，不被运行时、CLI或wheel入口导入。历史上测试过的score1六特征、CV/高频占比/质心/平坦度、20–30/22–30/24–28 Hz占比、绝对量、局部峰突出度、归一化AND组合和fraction-only 0.30审计，已集中整理到`data/spectrum_feature_research_summary.md`。该文档明确记录公式、候选阶段、AUC、中位数、最长误报窗、延迟、失败原因与后续验证要求；所有结论来自单记录候选边界而非ground truth，不得直接当作生产规则。

离线脚本继续使用项目唯一验收环境：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/pzt-mplconfig \
/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python analysis/<script_name>.py
```

这些历史脚本可能重现旧研究特征，其字段不是当前实时`SpectrumSnapshot`或新NPZ契约。维护历史研究时不得让它们回写已有采集CSV/NPZ，也不得重新把已删除分类概念接入实时判定。

## 9. 六维力采集与软件校零

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

## 10. 单帧处理、CoP与滑移

`TangentialSampleProcessor`是完整算法处理器，负责把`rows*cols`通道ADC变成内部`TangentialSample`，并独占自己的`PRSensorAngle`、`SlipDetector`和dx/dy中值窗口。`TangentialFrameProcessor`是面向用户的薄门面，适合回放CSV、自定义采集源或算法测试；它根据自己的CoP、标定和处理配置创建并持有一个独立样本处理器，公开构造函数不提供内部处理器注入点，因此不同门面不会共享算法状态。

单帧处理流程：

```text
rows×cols通道ADC
→ reshape为配置的(rows, cols)
→ 更新动态总压与像素阈值
→ 按full/region/both模式计算CoP、origin、区域和梯度
→ 更新SlipDetector
→ 必要时同步重锚定PRSensorAngle
→ dx/dy中值滤波
→ FitCalibrationModel预测Fx/Fy/Fz
→ TangentialSampleProcessor._process_sample()
→ TangentialSample
```

``TangentialSample`` 是完整应用内部详细结果，ADC 总和的 canonical 字段仍只叫 ``adc_sum``，不提供 ``total``、``sum``、``raw_2d``、``min``、``max``、``copX`` 或 ``copY`` 等别名。公开 ``TangentialFrameProcessor.process_frame(raw_data, frame=None)`` 只调用其内部独占的 ``_sample_processor._process_sample()`` 一次，再经 ``TangentialFrameProcessor._to_tangential_frame()`` 私有静态方法挑选八个字段并返回 ``TangentialFrame``；完整会话直接使用 ``TangentialSampleProcessor`` 生成的同一次 ``TangentialSample``，用于真实时间戳、梯度、区域、标定、GUI、同步和 108 列 CSV，不会再次执行 CoP、滑移或标定算法。``TangentialSampleProcessor``、``TangentialSample`` 和 ``TangentialFrameProcessor._to_tangential_frame()`` 仅属于内部实现，不进入顶层、``tangential.api``、``tangential.runtime.__all__`` 或公开 ``sensor.pyi``；``sensor.pyi`` 也不声明内部 ``_sample_processor`` 属性；安装 wheel 的用户始终只通过正式 API 得到 ``TangentialFrame``；``TangentialFrameProcessor`` 不再提供 ``_process_sample()`` 或旧的 ``process()``。

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

## 11. 缓存、seq与压力—力同步

`TimestampedBuffer.append()`为缓存项分配单调递增seq；`get_after(seq)`按顺序返回未消费项；`find_closest(ts, max_diff_s, min_seq)`只寻找未使用且满足窗口的候选项。`runtime/synchronization.py`只是该匹配能力的薄适配层，不保存第二套匹配算法。

完整会话以压力帧为唯一业务驱动：

```text
get_after(last_press_seq)
→ 按seq逐帧调用TangentialSampleProcessor
→ 每帧推进阈值、CoP、滑移、标定和GUI状态
→ 无力通道时立即写NaN力字段
→ 有力通道时进入pending_press队列
→ 队首压力帧在±15 ms内匹配一个未使用力帧
→ 匹配成功写rows×cols+24列CSV
→ 超过等待窗口仍未匹配则不写该CSV行
```

有力通道时，即使某个压力帧最终没有CSV行，它也已经推进了压力状态机并可更新GUI。每个力帧最多匹配一次，后到压力帧不能越过pending队首。修改该语义会影响数据量、训练筛选和时间连续性，必须同时修改测试与本文对应的架构、实现和验收说明。

`rel_ms`以第一帧合法压力`rx_t`为起点，`delta_ms`来自相邻已保存压力帧的真实接收时间差。不得把它们写成固定0、5、10网格，也不得使用GUI调用时间或文件flush时间。

当前实现的未匹配语义必须特别保留：无力通道时每个合法压力帧都写一行并填充NaN力字段；力通道启用时，压力样本先进入`pending_press`，队首样本只有在15 ms窗口内找到尚未使用的力帧才写CSV，超过窗口会移出队列但不写该行。该语义与“压力状态机和GUI仍继续推进”同时成立，不能只根据CSV行数判断压力帧是否被处理。

## 12. 完整会话与并发模型

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

## 13. 资源生命周期与错误传播

压力设备是必需资源，连接或启动握手失败时`FullAcquisitionSession.start()`抛出异常，不创建空CSV。六维力是可选资源，连接或启动校零失败时关闭该通道，压力采集继续运行。

数据线程异常保存在`PressureThread.error`或`ForceThread.error`，`check_errors()`在业务循环中抛出；`FullApplicationRunner`通过线程安全队列把异常交给Qt主线程显示并退出，避免GUI仍存活但采集已经停止。

`close()`必须幂等并按依赖顺序释放：设置停止事件 → 等待消费线程 → 等待重新归零任务 → 关闭传感器与IPC → flush/关闭CSV → 删除确实没有数据行的本次CSV。新增资源时必须把释放逻辑加入同一个生命周期，并新增异常退出测试。

## 14. CSV与模型

`storage/csv.py`是CSV格式的唯一来源。`build_csv_header(channel_count)`生成动态表头，默认`TABLE_CSV_HEADER`仍代表84通道108列格式；业务代码只传参数给`build_csv_row()`，不得手写列索引、复制表头或在GUI中拼接第二套行结构。

修改CSV时的最低要求：

- 同时修改`TABLE_CSV_HEADER`和`build_csv_row()`，保持长度一致。
- 更新训练、绘图、模型回归和集成测试。
- 明确新旧CSV兼容策略，离线工具必须按实际表头解析。
- 默认12×7必须继续兼容原108列顺序；非默认尺寸只改变连续`ch1...chN`数量，其余24列的名称和顺序不得改变。

`FitCalibrationModel.from_default()`从`tangential.resources`读取内置`fit_coefs.bin`，`from_path()`读取用户外部模型。运行时预测与离线训练共享模型格式，修改序列化结构前必须通过现有模型回归测试证明旧模型行为没有变化。

## 15. application、examples与CLI为什么分开

`application.py`是稳定的库入口，`examples/`是调用示范，`cli.py`是字符串参数到配置对象的适配层，三者不能互相复制完整应用逻辑。

```text
用户Python代码 → run_application(config) ┐
examples/full.py → run_application(config) ├→ application.py → runtime/session.py
CLI app → examples/full.main(config) ──────┘
```

`application.py`只导入轻量配置；Qt、PyQtGraph和完整会话在真正调用`run_application()`或`run_dual_application()`时惰性加载。这样基础`import tangential`不会加载可选GUI和绘图库。

`examples/minimal.py`保留唯一最小压力循环；`examples/full.py`只调用完整应用公共入口；`examples/dual_sensor.py`只负责两份独立配置与命令行示范。示例不是SDK内部实现层，生产模块不得反向依赖示例。

用户CLI固定为五个命令：`tangential example`惰性调用`examples/minimal.py`并只显示压力样本；`tangential app`通过`examples/full.py`调用`run_application`；`tangential dual`调用双路示例并复用`run_dual_application`；`tangential plot`惰性加载`tools.plotting`；`tangential fit`惰性加载`tools.training`。CLI和`examples/dual_sensor.py`都不得出现一致性标定子命令、开关或系数路径参数。基础`import tangential`不应创建Qt窗口，也不应把Matplotlib/PyQtGraph加载为运行时副作用。

## 16. 公共API维护规则

`tangential.__all__`定义稳定顶层公共边界。用户通过`from tangential import ...`、`help()`、IDE类型提示和`py.typed/.pyi`了解API；内部模块路径不承诺稳定。

当前顶层共有33个导出名称；新增的`ArrayConfig`是全项目唯一阵列布局来源。`ConsistenceCalibrationConfig`、`ConsistenceCalibrator`和`fit_consistence`均为维护者模块级名称。`TangentialSensor`别名不再导出；`TangentialSensorAPI`是唯一正式压力采集类名称。上方公共边界表按用户推荐、配置/应用/工具和高级/底层三类列出全部33个名称，`readme.md`不得出现一致性标定实现、命令、配置或术语；修改导出时必须同步本文和API测试。

新增或修改公共API时必须同步：

1. 在唯一实现模块写完整类型标注与docstring，至少包含作用、参数、返回值、异常和副作用。
2. 通过`api.py`或对应公共门面导出。
3. 更新`__init__.py`导入与`__all__`。
4. 编译模块同步更新同名`.pyi`签名。
5. 更新本文中的公共边界、内部调用链、修改路由和验收说明；用户明确要求时同步更新`readme.md`，并确保两份文档的公共名称集合一致。
6. 增加API导入、签名、行为和基础导入惰性测试。

不要为了让用户“看到更多功能”把所有内部类都放进顶层。判断标准是：用户是否存在无需依赖内部会话即可稳定复用的场景。`TangentialSensorAPI`适合硬件采集，`TangentialFrameProcessor`适合自定义数据源和离线`rows*cols`通道ADC；`TangentialSampleProcessor`只供完整会话维护内部详细结果，内部线程、会话辅助函数和协议解析私有方法不应公开。

## 17. 常见扩展任务

### 17.1 增加配置参数

```text
确定唯一消费者
→ 在对应Config增加字段和默认值
→ 增加环境变量解析与validate
→ 从调用入口传到消费者
→ 添加默认/显式/非法值测试
→ 更新readme.md与本文对应章节
```

### 17.2 修改压力或六维力协议

只修改对应`sensors/*.py`，同时覆盖分包、粘包、噪声、错误长度、CRC或帧尾、超时、慢响应和恢复。不得把协议解析放入`runtime/session.py`。

### 17.3 修改CoP、区域或滑移

CoP与区域修改进入`processing/cop.py`，滑移修改进入`processing/slip.py`，`TangentialSampleProcessor`负责编排完整算法，`TangentialFrameProcessor`只负责公开结果投影。必须验证无接触、首次接触、精修、卸载、滑移进入、方向平滑、退出重锚定和多实例状态隔离。

### 17.4 接入自定义ADC数据源

自定义来源只需提供与处理配置`rows*cols`一致的通道数据并调用`TangentialFrameProcessor.process_frame(raw_data, frame=None)`；如果要复用`TangentialSensorAPI`生命周期，可注入实现`read_frame()`、`decode()`和`close()`的sensor对象。完整应用测试若需要检查详细结果，应直接注入或构造提供`_process_sample()`的`TangentialSampleProcessor`对象。不要修改`PressureSensor`来适配与现有协议无关的数据源。

### 17.5 增加第三只或更多传感器

为每一路分别构造`FullApplicationConfig`、端口、输出目录、处理器和停止事件；复用现有单路会话，不共享`PRSensorAngle`或`SlipDetector`。在扩展运行器中统一校验所有物理端口和目录唯一性。

### 17.6 增加新的编译模块

```text
新增唯一.py源码
→ 增加同名.pyi
→ 加入setup.py COMPILED_MODULES
→ 确认wheel排除该内部.py
→ 检查.so、.pyi、签名和docstring
→ 更新分发测试
```

## 18. 测试结构与修改路由

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
<td style="white-space:normal"><code>tests/</code> 中的压力协议、调度和队列契约</td>
</tr>
<tr>
<td style="white-space:normal">六维力协议、校零和进程</td>
<td style="white-space:normal"><code>sensors/force.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的六维力协议、校零和进程契约</td>
</tr>
<tr>
<td style="white-space:normal">seq、缓存和时间匹配</td>
<td style="white-space:normal"><code>acquisition/buffer.py</code>、<code>runtime/synchronization.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的缓存、匹配和集成契约</td>
</tr>
<tr>
<td style="white-space:normal">CoP、阈值、梯度和区域</td>
<td style="white-space:normal"><code>processing/cop.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的 API、GUI 和集成契约</td>
</tr>
<tr>
<td style="white-space:normal">滑移状态和方向</td>
<td style="white-space:normal"><code>processing/slip.py</code>、<code>runtime/sensor.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的滑移和 GUI 契约</td>
</tr>
<tr>
<td style="white-space:normal">模型读取与预测</td>
<td style="white-space:normal"><code>processing/calibration.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的模型、CSV 和多维预测契约</td>
</tr>
<tr>
<td style="white-space:normal">最小API与示例</td>
<td style="white-space:normal"><code>runtime/sensor.py</code>、<code>api.py</code>、<code>examples/minimal.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的公开 API、结构和示例契约</td>
</tr>
<tr>
<td style="white-space:normal">完整采集、CSV、清理和Qt生命周期</td>
<td style="white-space:normal"><code>runtime/session.py</code>、<code>application.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的完整会话、双传感器和资源生命周期契约</td>
</tr>
<tr>
<td style="white-space:normal">CSV结构</td>
<td style="white-space:normal"><code>storage/csv.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的 108 列、表头和绘图解析契约</td>
</tr>
<tr>
<td style="white-space:normal">GUI</td>
<td style="white-space:normal"><code>gui/realtime.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的离屏 GUI 和显示状态契约</td>
</tr>
<tr>
<td style="white-space:normal">CoP频谱、时频窗口和NPZ</td>
<td style="white-space:normal"><code>processing/spectrum.py</code>、<code>gui/spectrum.py</code>、<code>runtime/session.py</code></td>
<td style="white-space:normal"><code>tests/test_spectrum.py</code> 及完整会话、分发测试</td>
</tr>
<tr>
<td style="white-space:normal">训练与绘图</td>
<td style="white-space:normal"><code>tools/training.py</code>、<code>tools/plotting.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的训练、列解析和绘图契约</td>
</tr>
<tr>
<td style="white-space:normal">CLI</td>
<td style="white-space:normal"><code>cli.py</code></td>
<td style="white-space:normal"><code>tests/</code> 中的命令参数、分发和退出码契约</td>
</tr>
<tr>
<td style="white-space:normal">wheel内容、资源和惰性导入</td>
<td style="white-space:normal"><code>pyproject.toml</code>、<code>setup.py</code>、<code>MANIFEST.in</code></td>
<td style="white-space:normal"><code>tests/</code> 中的 wheel、资源和隔离导入契约</td>
</tr>
</tbody>
</table>

## 19. 本地开发与测试

项目唯一开发和验收 Conda 环境是 `TimeDrift_GRU`，固定解释器为 `/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python`。维护者和 Agent 不得用裸 `python`、base 环境或系统解释器执行源码、测试、`compileall` 和构建；以下命令统一先设置项目专用变量：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
"$TANGENTIAL_PYTHON" --version
"$TANGENTIAL_PYTHON" -m pip install -r requirements.txt
```

源码模式从项目根目录运行，不要求预先生成 ``.so``：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.examples.minimal
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.examples.full
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.cli --version
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.cli example --help
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.cli app --help
```

源码模式双传感器示例必须使用两个真实且不同的端口：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
```

基础语法检查：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m compileall -q src/tangential tests
```

完整测试：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src \
QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/tmp/pzt-mplconfig \
"$TANGENTIAL_PYTHON" -m unittest discover -s tests -q
```

只运行相关测试时使用模块名，例如：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.processing.calconsistence
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m unittest tests.test_data -q
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m unittest tests.test_slip -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=src "$TANGENTIAL_PYTHON" -m unittest tests.test_plot_and_gui -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=src "$TANGENTIAL_PYTHON" -m unittest tests.test_spectrum -q
```

修改后至少执行：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m compileall -q src/tangential tests
```

如果目录中已有用户修改，测试失败时必须区分本次变更和预存变更，不得覆盖或清除无关内容。

## 20. Wheel构建与隔离验收

构建依赖由`pyproject.toml`声明，开发环境可以直接执行：

```bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
"$TANGENTIAL_PYTHON" -m pip wheel . --no-deps --no-build-isolation -w dist
```

当前12个编译模块由`setup.py`的`COMPILED_MODULES`定义：`runtime/sensor`、`runtime/session`、`runtime/synchronization`、`acquisition/buffer`、`sensors/pressure`、`sensors/force`、`processing/cop`、`processing/calibration`、`processing/calconsistence`、`processing/slip`、`processing/spectrum`和`storage/csv`。

`setup.py`的Cython指令必须保持`language_level=3`、`annotation_typing=False`、`binding=True`、`embedsignature=True`和`always_allow_keywords=True`。其中`annotation_typing=False`保证源码中的类型注解不会被错误解释为运行时强类型约束，尤其不能破坏对`bytearray`、NumPy数组和测试注入对象的兼容输入。

构建流程：

```text
.py唯一源码
→ Cython生成并编译同名扩展
→ BinaryWheelBuildPy清理旧build/lib*/tangential
→ wheel保留公开Python层、配置、CLI、示例、GUI、tools、.pyi和资源
→ wheel排除12个内部实现.py与生成的C源码
```

预期产物：

```text
dist/tangential_sensor-0.6.0-cp311-cp311-linux_x86_64.whl
```

分发验收必须确认：

- wheel包含12个内部`.so`和12个同名`.pyi`。
- wheel包含`py.typed`、`tangential/resources/fit_coefs.bin`和已经由维护者确认来源的`tangential/resources/consistence_coeffs.npz`；不得把标定 CSV 本身打包进 wheel。
- wheel不包含对应内部`.py`、生成的C源码或外部share模型目录。
- 脱离源码目录后可以`import tangential`、加载内置模型并完成回归预测。
- `help()`、函数签名和IDE类型提示在安装wheel后仍可用。
- 基础`import tangential`不加载Qt、PyQtGraph或Matplotlib。
- 源码模式和隔离安装模式都通过协议、CoP、同步、CSV和模型回归测试。

当前`requirements.txt`锁定完整开发/GUI环境：Cython 3.2.9、NumPy 2.4.3、SciPy 1.17.1、pyserial 3.5、pyqtgraph 0.14.0、Matplotlib 3.10.8和PyQt5 5.15.11；`pyproject.toml`只声明核心运行依赖`numpy`、`scipy`、`pyserial`，GUI和离线绘图库属于`full`可选依赖，Cython只属于构建依赖。

不要手工提交`build/`、`dist/`、生成的`.so`或C文件；它们是可重建产物。

## 21. 常见故障定位

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

## 22. 版本控制限制与经授权回退

默认不执行Git写操作，不运行`add`、`commit`、`revert`、`restore`或`reset`，也不创建提交。只读状态检查也只在用户明确要求或任务确有必要且当前指令允许时进行；如果当前任务明确禁止Git命令，则不得运行任何Git命令。

只有用户之后明确重新授权版本控制操作时，才能在核对精确目标后执行授权范围内的动作。撤销已经提交的独立阶段时优先使用`git revert <commit-hash>`保留历史；不得使用`git reset --hard`覆盖用户修改。与当前任务无关的修改、数据文件删除或相邻目录未跟踪内容必须原样保留。

## 23. 修改完成的定义

一次修改只有同时满足以下条件才算完成：

- 修改位于唯一职责模块，没有复制协议、算法或CSV实现。
- 配置从`config.py`进入实际运行路径，没有散落第二套默认值。
- 多传感器状态与资源仍然隔离。
- 时间戳、seq、匹配窗口和CSV语义没有被GUI或循环节拍改变。
- 异常路径可以关闭线程、进程、串口、队列、CSV和Qt资源。
- 相关单元测试、集成测试和回归测试通过。
- 公共签名、`.pyi`和本文中的架构、实现、修改路由与验收信息同步；`readme.md`仅在用户明确要求时更新。
- 未执行未获授权的版本控制操作，也未覆盖任何预存工作区内容。
