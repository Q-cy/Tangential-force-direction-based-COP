# Tangential Sensor SDK 0.5.0 开发者维护指南

本文仅面向 Tangential SDK 源码维护者，说明内部架构、实现边界、修改路由、测试、构建与排障流程。安装 wheel、使用命令行和公共 API 的用户请阅读 [readme.md](readme.md)。

## 1. 开发目标与不可破坏边界

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
→ decode得到84通道ADC
→ TangentialSampleProcessor._process_sample()计算CoP、梯度、滑移和标定
→ TangentialSample（内部详细结果）
→ TangentialFrameProcessor._to_tangential_frame()挑选八个公开字段
→ TangentialFrame
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

## 4. 目录结构与职责

本节只面向源码维护者，说明每个目录和文件的唯一职责。`readme.md` 是安装 wheel 后的用户与二次开发文档；`readme_developer.md` 是源码、测试、构建和维护流程文档，两者不再合并成一份“用户与维护者混合指南”。

目录树从项目根目录开始展开；例如压力协议文件的完整相对路径是 `src/tangential/sensors/pressure.py`，不能只根据 `sensors/` 目录名定位实现。

```text
04_tang_7_12_COP_fit_std/
├── AGENTS.md
│   └── Agent 修改边界、架构不变量、文件路由、测试和版本控制限制。
├── readme.md
│   └── wheel 用户安装、公开 API、CLI 和二次开发说明；不描述内部实现。
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
│       │   │   └── 串口请求 → 清缓存/分批收包 → 动态帧长、CRC、状态和载荷校验 → 84 通道 ADC、真实接收时间；提供独立压力采集进程入口。
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
│       │   │   └── 84 通道 ADC → 动态阈值、接触区域、origin 和区域合并 → CoP、角度、梯度及状态。
│       │   ├── cop.pyi
│       │   │   └── cop.py 对应 Cython .so 的公开静态类型和签名；不是第二套 CoP 算法实现。
│       │   ├── slip.py
│       │   │   └── 压力矩阵/CoP 短窗 → 斑块相关、位移、EMA 方向和连续帧滞回 → STICK/SLIP 与同步重锚结果。
│       │   ├── slip.pyi
│       │   │   └── slip.py 对应 Cython .so 的公开静态类型和签名；不是第二套滑移状态机实现。
│       │   ├── calibration.py
│       │   │   └── 内置或外部 fit_coefs.bin → 特征和拟合类型解析 → Fx/Fy/Fz 标定预测。
│       │   └── calibration.pyi
│       │       └── calibration.py 对应 Cython .so 的公开静态类型和签名；不是第二套模型加载/预测实现。
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
│       │   │   └── 内部压力/力结果 → 固定字段映射 → 唯一 108 列 CSV 表头和数据行；不决定采样节拍。
│       │   └── csv.pyi
│       │       └── csv.py 对应 Cython .so 的公开静态类型和签名；不是第二套 CSV 格式实现。
│       ├── gui/
│       │   ├── __init__.py
│       │   │   └── GUI 子包边界；保持基础 import tangential 不主动加载 Qt 和绘图库。
│       │   └── realtime.py
│       │       └── 最新样本/历史序列 → PyQtGraph 图元、压力快照和方向/力箭头 → 实时窗口与分析图；不读取串口。
│       ├── tools/
│       │   ├── __init__.py
│       │   │   └── 离线工具子包边界；供 CLI 按需加载训练和绘图模块。
│       │   ├── training.py
│       │   │   └── 训练 CSV/有效行 → 拟合参数和评估 → fit_coefs.bin 或显式写回目标；不参与实时采集。
│       │   └── plotting.py
│       │       └── CSV 实际表头/列选择 → 行范围与分析计算 → 静态图、完整分析图和结果对象；不参与实时采集。
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
│           └── fit_coefs.bin
│               └── 随 wheel 安装的静态标定模型；由 calibration.py 作为 package resource 加载，不是 Python 源码。
└── tests/
    └── 覆盖压力/六维力协议与时序、缓存同步、公开 API、CoP/滑移/标定、CSV、GUI、CLI、训练、wheel 分发和回归契约；测试文件不属于运行时包。
```

`src/tangential/**/__pycache__/*.pyc`、`build/` 和 `dist/` 是运行或构建生成物，不属于正式源码树；`data/` 是采集和离线分析输出目录，不作为包实现。发布 wheel 时，`runtime`、`acquisition`、`sensors`、`processing` 和 `storage` 的 `.py` 由 `setup.py` 编译为同名 Cython 扩展；仓库中的 `.py` 仍是唯一维护源，`.pyi` 只提供静态签名，不能替代实现。

使用这棵树定位修改时，协议采集只改 `src/tangential/sensors/pressure.py` 或 `force.py`；公开 `TangentialFrame` 与内部 `TangentialSample` 的边界只改 `src/tangential/runtime/sensor.py`；CoP、滑移和标定分别改 `processing/cop.py`、`slip.py` 和 `calibration.py`；压力—力同步只改 `acquisition/buffer.py` 与 `runtime/synchronization.py`；108 列 CSV 只改 `storage/csv.py`；实时窗口只改 `gui/realtime.py`；CLI 与示例入口分别改 `cli.py`、`examples/minimal.py`、`full.py` 和 `dual_sensor.py`；训练和离线绘图只改 `tools/training.py` 与 `tools/plotting.py`；构建、扩展过滤和 wheel 内容只改 `pyproject.toml`、`setup.py` 与 `MANIFEST.in`。每类功能先定位这里列出的唯一实现，再更新对应测试和本维护者文档，禁止在调用方复制协议、算法或 CSV 映射。

## 4.1 公共 API 与内部实现边界

`src/tangential/__init__.py`当前真实导出 32 个名称。`readme.md`只向普通用户推荐其中 3 个：`TangentialSensorAPI`、`TangentialFrameProcessor`和`TangentialFrame`。配置、完整应用、训练和绘图名称属于可选的维护者/命令实现边界；高级协议、算法和终端输出名称仍保留源码与顶层导出，但不作为普通用户的首选接口。`TangentialSensor`别名已删除，`TangentialSensorAPI`是唯一正式压力采集类名称。

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
<td style="white-space:normal">已有84通道ADC → 单帧处理 → <code>TangentialFrame</code></td>
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
7. `src/tangential/storage/csv.py`：确认完整应用最终写出的108列语义。
8. `src/tangential/acquisition/buffer.py`与`runtime/synchronization.py`：理解seq消费和一次性时间匹配。
9. `src/tangential/runtime/session.py`：把设备、处理、匹配、CSV、GUI和清理串起来。
10. `src/tangential/application.py`、`src/tangential/examples/full.py`、`src/tangential/examples/dual_sensor.py`与`src/tangential/cli.py`：理解公共入口如何复用完整会话。
11. `src/tangential/tools/training.py`与`plotting.py`：最后阅读离线工具和模型生产流程。
12. `setup.py` 与 `tests/`：理解源码怎样变成独立 wheel，并由分发契约验证 wheel 内容。

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
→ 验证长度、CRC、状态和168字节传感器payload
→ 记录rx_t与latency_s
→ 写入request_seq/tx_t/rx_t/latency_s/raw
→ 本轮不足period_s时等待剩余时间，超期则直接进入下一轮并计数
```

解析器支持分包、单轮粘包、前导噪声、错误长度、CRC错误和状态错误恢复。当前策略每轮只接受一个合法响应，轮末清空残留，避免上一轮晚到数据被错误归属到下一请求。

`read_frame()`返回168字节payload与时序元数据，`decode()`只执行84个little-endian `uint16`解码并保持设备原始线序。左右翻转、基线、增益、CoP和标定不属于该模块。

重要统计包括`requests`、`frames`、`response_timeouts`、`crc_errors`、`length_errors`、`status_errors`、`serial_read_errors`、`serial_write_errors`、`serial_flush_errors`、`queue_drops`和`schedule_skips`，以及最近发送间隔、接收间隔和响应延迟。目标200 Hz是请求上限；设备响应约6 ms时实际频率约166 Hz属于正常物理结果。

压力驱动的生产结构是“父进程`PressureSensor` → spawn子进程 → 子进程内本地`PressureSensor` → 单一压力I/O线程 → 串口”。父进程只从IPC帧队列读取，父进程的`PressureThread`负责解码并追加到`TimestampedBuffer`；因此业务处理、GUI刷新和CSV写入不会成为串口消费者。

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

`TangentialSampleProcessor`是完整算法处理器，负责把84通道ADC变成内部`TangentialSample`，并独占自己的`PRSensorAngle`、`SlipDetector`和dx/dy中值窗口。`TangentialFrameProcessor`是面向用户的薄门面，适合回放CSV、自定义采集源或算法测试；它根据自己的CoP、标定和处理配置创建并持有一个独立样本处理器，公开构造函数不提供内部处理器注入点，因此不同门面不会共享算法状态。

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
→ TangentialSampleProcessor._process_sample()
→ TangentialSample
```

``TangentialSample`` 是完整应用内部详细结果，ADC 总和的 canonical 字段仍只叫 ``adc_sum``，不提供 ``total``、``sum``、``raw_2d``、``min``、``max``、``copX`` 或 ``copY`` 等别名。公开 ``TangentialFrameProcessor.process_frame(raw, frame=None)`` 只调用其内部独占的 ``_sample_processor._process_sample()`` 一次，再经 ``TangentialFrameProcessor._to_tangential_frame()`` 私有静态方法挑选八个字段并返回 ``TangentialFrame``；完整会话直接使用 ``TangentialSampleProcessor`` 生成的同一次 ``TangentialSample``，用于真实时间戳、梯度、区域、标定、GUI、同步和 108 列 CSV，不会再次执行 CoP、滑移或标定算法。``TangentialSampleProcessor``、``TangentialSample`` 和 ``TangentialFrameProcessor._to_tangential_frame()`` 仅属于内部实现，不进入顶层、``tangential.api``、``tangential.runtime.__all__`` 或公开 ``sensor.pyi``；``sensor.pyi`` 也不声明内部 ``_sample_processor`` 属性；安装 wheel 的用户始终只通过正式 API 得到 ``TangentialFrame``；``TangentialFrameProcessor`` 不再提供 ``_process_sample()`` 或旧的 ``process()``。

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
→ 匹配成功写108列CSV
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

`storage/csv.py`是108列格式的唯一来源。业务代码只传参数给`build_csv_row()`，不得手写列索引、复制表头或在GUI中拼接第二套行结构。

修改CSV时的最低要求：

- 同时修改`TABLE_CSV_HEADER`和`build_csv_row()`，保持长度一致。
- 更新训练、绘图、模型回归和集成测试。
- 明确新旧CSV兼容策略，离线工具必须按实际表头解析。
- 如果项目要求继续兼容108列，则不得增加、删除或重排列。

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

当前命令分工固定为：`tangential example`惰性调用`examples/minimal.py`并只显示压力样本；`tangential app`通过`examples/full.py`调用`run_application`；`tangential dual`调用双路示例并复用`run_dual_application`；`tangential plot`惰性加载`tools.plotting`；`tangential fit`惰性加载`tools.training`。基础`import tangential`不应创建Qt窗口，也不应把Matplotlib/PyQtGraph加载为运行时副作用。

## 16. 公共API维护规则

`tangential.__all__`定义稳定顶层公共边界。用户通过`from tangential import ...`、`help()`、IDE类型提示和`py.typed/.pyi`了解API；内部模块路径不承诺稳定。

当前顶层共有32个导出名称：`TangentialSensorAPI`、`TangentialFrame`、`TangentialFrameProcessor`、`FixedTerminalRenderer`、`FitCalibrationModel`、`FullApplicationConfig`、`PressureConfig`、`ForceConfig`、`CopConfig`、`ProcessingConfig`、`SlipConfig`、`CalibrationConfig`、`SyncConfig`、`OutputConfig`、`GuiConfig`、`PRSensorAngle`、`PressureSensor`、`TangentialMotionState`、`SlipResult`、`SlipDetector`、`compute_vector_angle`、`angle_difference`、`format_terminal_sample`、`TrainingConfig`、`TrainingResult`、`train_model`、`PlotConfig`、`PlotResult`、`plot_csv`、`plot_full_analysis`、`run_application`和`run_dual_application`。`TangentialSensor`别名不再导出；`TangentialSensorAPI`是唯一正式压力采集类名称。上方公共边界表按用户推荐、配置/应用/工具和高级/底层三类列出全部32个名称，`readme.md`只保留前三项的普通用户介绍；修改导出时必须同步本文、`readme.md`和API测试。

新增或修改公共API时必须同步：

1. 在唯一实现模块写完整类型标注与docstring，至少包含作用、参数、返回值、异常和副作用。
2. 通过`api.py`或对应公共门面导出。
3. 更新`__init__.py`导入与`__all__`。
4. 编译模块同步更新同名`.pyi`签名。
5. 更新本文中的公共边界、内部调用链、修改路由和验收说明；用户明确要求时同步更新`readme.md`，并确保两份文档的公共名称集合一致。
6. 增加API导入、签名、行为和基础导入惰性测试。

不要为了让用户“看到更多功能”把所有内部类都放进顶层。判断标准是：用户是否存在无需依赖内部会话即可稳定复用的场景。`TangentialSensorAPI`适合硬件采集，`TangentialFrameProcessor`适合自定义数据源和离线84通道ADC；`TangentialSampleProcessor`只供完整会话维护内部详细结果，内部线程、会话辅助函数和协议解析私有方法不应公开。

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

自定义来源只需提供84通道数据并调用`TangentialFrameProcessor.process_frame(raw, frame=None)`；如果要复用`TangentialSensorAPI`生命周期，可注入实现`read_frame()`、`decode()`和`close()`的sensor对象。完整应用测试若需要检查详细结果，应直接注入或构造提供`_process_sample()`的`TangentialSampleProcessor`对象。不要修改`PressureSensor`来适配与现有协议无关的数据源。

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

安装完整环境：

```bash
python -m pip install -r requirements.txt
```

源码模式从项目根目录运行，不要求预先生成 ``.so``：

```bash
PYTHONPATH=src python -m tangential.examples.minimal
PYTHONPATH=src python -m tangential.examples.full
PYTHONPATH=src python -m tangential.cli --version
PYTHONPATH=src python -m tangential.cli example --help
PYTHONPATH=src python -m tangential.cli app --help
```

源码模式双传感器示例必须使用两个真实且不同的端口：

```bash
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
PYTHONPATH=src python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
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

修改后至少执行：

```bash
PYTHONPATH=src python -m compileall -q src/tangential tests
```

如果目录中已有用户修改，测试失败时必须区分本次变更和预存变更，不得覆盖或清除无关内容。

## 20. Wheel构建与隔离验收

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
dist/tangential_sensor-0.5.0-cp311-cp311-linux_x86_64.whl
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
