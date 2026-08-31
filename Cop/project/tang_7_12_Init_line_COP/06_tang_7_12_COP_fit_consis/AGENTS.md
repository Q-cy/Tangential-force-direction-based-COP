# 04_tang_7_12_COP_fit_std：Agent 工作手册

本文对应 Tangential SDK 0.5.0 的实际架构。修改前必须读取并核对源码；不得恢复旧入口，不得复制协议、CoP、标定或 108 列 CSV 实现。

## 1. 项目定位和交付边界

项目提供 12×7 PZT 压力阵列与可选六维力传感器的采集、CoP、角度、梯度、切向力标定、实时 GUI、CSV 保存、训练和离线绘图。

- 正式源码目录是 src/tangential/。
- 源码仓库完整保留所有 Python .py；源码模式可通过 PYTHONPATH=src 直接运行，不要求预编译 .so。
- 发布 wheel 目标是 Linux x86_64、CPython 3.11，名称为 tangential_sensor-0.5.0-cp311-cp311-linux_x86_64.whl。
- wheel 内部 runtime、acquisition、sensors、processing、storage 是多个 CPython 扩展 .so；公开 API、config、CLI、examples、gui、tools 和类型声明保留可读 Python。
- .so 只是 Python 扩展，不提供稳定 C++/Rust ABI；不得把它描述为原生跨语言 SDK。
- 保密用户不要发布 sdist，因为 sdist 包含 Python 源码。源码仓库是内部维护源。
- Python 最低版本为 3.11；核心运行依赖和 full 可选依赖由 pyproject.toml 声明，requirements.txt 是完整开发/GUI 环境。

## 2. 实际目录和职责

~~~text
04_tang_7_12_COP_fit_std/
├── AGENTS.md                         # Agent 架构、不变量、修改路由和验收约束
├── readme.md                         # wheel用户安装、API、CLI 和二次开发说明
├── requirements.txt                  # Python 3.11 完整开发/GUI 依赖
├── pyproject.toml                    # 包元数据、依赖、入口、资源和构建依赖，关于
├── setup.py                          # Cython扩展清单、编译指令和wheel源码过滤
├── MANIFEST.in                       # 源码分发清单
├── src/tangential/
│   ├── __init__.py                   # 稳定顶层公共 API 和版本
│   ├── api.py                        # 可读公开 API 门面
│   ├── application.py                # run_application 公共完整应用入口
│   ├── application.pyi               # 完整应用入口的静态类型签名
│   ├── cli.py                        # example/app/dual/plot/fit 命令分发
│   ├── config.py                     # 分类配置、默认值、环境变量和校验
│   ├── py.typed                      # 类型提示标记
│   ├── runtime/
│   │   ├── sensor.py                 # TangentialSensorAPI、Frame/内部Sample、单帧处理和终端渲染
│   │   ├── sensor.pyi                # 编译后模块的公开类型签名
│   │   ├── session.py                # 完整采集会话、消费线程、CSV、GUI 和清理
│   │   └── synchronization.py        # 压力—六维力匹配薄适配层
│   ├── acquisition/
│   │   └── buffer.py                 # TimestampedBuffer、seq、顺序消费和匹配
│   ├── sensors/
│   │   ├── pressure.py               # 压力串口协议、CRC、200 Hz 轮询和独立进程
│   │   └── force.py                  # 六维力协议、轮询、校零和独立进程
│   ├── processing/
│   │   ├── cop.py                    # PRSensorAngle、CoP、状态机、梯度和区域
│   │   ├── slip.py                   # 归一化斑块平移、CoP短窗和STICK/SLIP状态
│   │   ├── calibration.py            # fit_coefs.bin 读取和模型预测
│   │   ├── calconsistence.py         # 离线 CSV 拟合、NPZ 加载和84通道一致性修正
│   │   └── calconsistence.pyi       # 一致性标定扩展的静态签名
│   ├── storage/
│   │   └── csv.py                    # 唯一 108 列表头和行构造
│   ├── gui/
│   │   └── realtime.py               # PyQtGraph 实时显示
│   ├── tools/
│   │   ├── training.py               # 离线拟合和模型写出
│   │   └── plotting.py               # CSV 解析、绘图和完整分析
│   ├── examples/
│   │   ├── minimal.py                # 唯一最小压力采集循环
│   │   ├── dual_sensor.py            # 两只压力传感器的隔离并发采集示例
│   │   └── full.py                   # 调用 run_application 的完整示例
│   └── resources/
│       ├── fit_coefs.bin              # package resource 静态模型
│       └── consistence_coeffs.npz     # 由 consistence/ 多段CSV生成的默认v2一致性系数
└── tests/                             # 协议、API、GUI、分发和回归测试
~~~

用户代码 / CLI / examples -> application.py -> runtime/session.py -> 采集进程、CSV、GUI、同步和资源清理

runtime、acquisition、sensors、processing、storage 是运行时核心实现；api、config、application、cli、examples、gui、tools 是用户可读层或按需加载层。目录层级不是“重要/不重要”标记，而是按运行时、设备、算法、界面、离线工具和资源职责划分。所有被编译的 .py 在仓库中继续作为唯一源码，相应 .so 只由构建生成。

## 3. 公共 API 和示例边界

稳定源码 API 从 tangential 顶层导入，包括以下 32 个名称；普通用户推荐其中 3 个：TangentialSensorAPI、TangentialFrameProcessor、TangentialFrame。配置、应用、工具和高级/底层名称仍保留在源码与顶层导出中，供完整程序和维护者使用：

- TangentialSensorAPI、TangentialFrame、TangentialFrameProcessor。
- ``TangentialFrame`` 的字段严格为 ``base_data``、``adc_sum``、``cop_x``、``cop_y``、
  ``angle``、``dx``、``dy``、``motion_state``；其中 ``adc_sum`` 是 84 通道
  ADC 之和，也是公开对象中唯一的 ADC 总和名称。
- FixedTerminalRenderer、format_terminal_sample、compute_vector_angle、angle_difference。
- FitCalibrationModel、PRSensorAngle、PressureSensor、TangentialMotionState、SlipResult、SlipDetector、SlipConfig。
- PressureConfig、ForceConfig、CopConfig、ProcessingConfig、CalibrationConfig、SyncConfig、OutputConfig、GuiConfig、TrainingConfig、PlotConfig、FullApplicationConfig。
- train_model、TrainingResult、plot_csv、plot_full_analysis、PlotResult、run_application。
- run_dual_application：在一个 QApplication 中启动两路完整、相互隔离的 GUI 会话。

上述列表中的其余 29 个名称继续保留源码、顶层导出、runtime 导出和类型声明，但不放入面向普通用户的 readme.md 三项推荐 API 表；它们由完整会话、离线工具或维护者场景使用。一致性标定完全属于维护者内部能力，``ConsistenceCalibrationConfig`` 只从 ``tangential.config`` 导入，``ConsistenceCalibrator`` 和 ``fit_consistence`` 只从 ``tangential.processing.calconsistence`` 导入；这些名称不得进入 ``tangential``、``tangential.api`` 或用户 CLI。TangentialSensor 别名已删除，TangentialSensorAPI 是唯一正式压力采集类名称。minimal.py 继续使用 FixedTerminalRenderer 输出固定终端布局；format_terminal_sample 保持为终端文本格式化函数。

examples/minimal.py 是唯一最小循环；CLI example 必须调用它，不得在 cli.py 复制 while 循环。examples/full.py 只调用公开 run_application；CLI app 必须复用该入口。plot 和 fit 必须惰性导入 tools，基础 import tangential 不得加载 Qt、PyQtGraph 或 Matplotlib。

examples/dual_sensor.py 展示两路完整应用用法：每个 ``FullApplicationConfig``
必须拥有独立 ``PressureConfig``、串口、采集进程、IPC队列、读取线程、处理器、
``FullAcquisitionSession``、停止事件、GUI和输出目录；
启动前必须拒绝指向同一物理串口的配置。不要为多设备引入共享传感器实例、
共享CoP状态机或单一阻塞读取循环。

双传感器运行和验收必须严格按以下顺序执行：

1. 插入两只压力设备。运行 ``python -m serial.tools.list_ports -v`` 和
   ``ls -l /dev/serial/by-id/``，取得两个不同物理设备的实际路径；没有列出
   端口时不得继续启动示例。
2. 用第1步查到的真实路径分别设置 ``PORT_A``、``PORT_B``。不得原样输入
   ``<sensor-a>`` 或 ``<sensor-b>``：Bash会把尖括号解释为重定向，导致
   ``syntax error near unexpected token 'newline'``。设置后用
   ``printf 'A=%s\nB=%s\n' "$PORT_A" "$PORT_B"`` 复核变量内容。
3. 用 ``ls -l "$PORT_A" "$PORT_B"`` 检查路径和权限；用
   ``groups`` 检查当前用户是否有串口权限，用 ``fuser "$PORT_A" "$PORT_B"``
   检查端口是否被旧进程占用。两个变量必须对应两个不同的物理设备。
4. 按下面的源码模式或 wheel 模式命令启动示例，并确认终端持续打印A/B两路
   的压力数据和完整 GUI。某一路持续没有数据时应重新核对端口、权限、占用
   和设备连接；任一路采集线程异常都会报告 Sensor A/B 并联动停止两路。
5. 按 ``Ctrl+C`` 停止。必须确认两路读取线程、采集进程、IPC队列和串口都已
   关闭；异常退出排查时也要检查没有遗留进程继续占用端口。

源码运行命令：

~~~bash
PORT_A=/dev/serial/by-id/DEVICE_A_ID
PORT_B=/dev/serial/by-id/DEVICE_B_ID
PYTHONPATH=src python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
~~~

安装wheel后运行命令：

~~~bash
python -m tangential.examples.dual_sensor \
  --port-a "$PORT_A" \
  --port-b "$PORT_B"
~~~

文档中的 ``DEVICE_A_ID``/``DEVICE_B_ID`` 是待替换文本，不得使用带尖括号
的shell占位符；Bash会把 ``<``/``>`` 解释为输入输出重定向。若系统没有
``/dev/serial/by-id/``，可在确认设备映射后使用不同的 ``/dev/ttyUSB*``。

该示例启动一个 ``QApplication``、两个 ``RealTimePlot`` 和两个
``acquisition_loop``，两路都完整执行压力采集、CoP、角度、梯度、标定、实时
曲线/压力表、108列 CSV 和退出后的分析图；它不再是终端摘要示例。默认只启用
压力传感器，只有显式提供 ``--force-port-a``/``--force-port-b`` 才启用对应
六维力通道，避免两路同时打开默认 ``/dev/ttyUSB1``。修改示例时必须复用
``run_dual_application``、``FullAcquisitionSession`` 和 ``acquisition_loop``，
保留并发读取、端口唯一性校验和两路资源的异常清理。

统一 CLI 也提供同一入口：

~~~bash
PYTHONPATH=src python -m tangential.cli dual \
  --port-a "$PORT_A" --port-b "$PORT_B" \
  --save-dir ./data/dual
~~~

提供 ``--force-port-a``/``--force-port-b`` 后才会分别启用力通道；两路力端口
必须不同。默认输出为 ``<base>/sensor_a`` 和 ``<base>/sensor_b``，也可用
``--save-dir-a``、``--save-dir-b`` 分别覆盖。窗口标题固定保留 ``Sensor A``
和 ``Sensor B`` 标签，状态文本不能覆盖标签。

公开 API 的签名、输入、输出、异常和资源生命周期必须有文档字符串。新增用户可调用符号时同步更新 api.py、__init__.py 和测试。

功能、API、命令或配置修改默认不改 ``readme.md``，但必须同步更新 ``readme_developer.md`` 中的架构、实现、修改路由和验收信息。``readme.md`` 只面向安装 wheel 后的用户和二次开发者；``readme_developer.md`` 只面向源码维护者，两者不得复制。只有用户明确要求更新wheel用户文档时才修改 ``readme.md``。

## 4. 配置规则

所有用户可调参数集中在 config.py，按功能使用 dataclass：

- PressureConfig：压力端口、波特率、目标频率、响应超时、队列和启动超时。
- ForceConfig：六维力 enabled 开关、端口、波特率、目标频率、响应超时、队列、启动超时和校零参数。
- CopConfig：阈值、背景学习、稳定帧、区域和二次精修。
- ProcessingConfig：标定维度、区域模式、中值窗口和精修归零策略。
- CalibrationConfig：外部模型路径。
- SyncConfig：主循环频率、GUI 频率、匹配窗口、统计周期和缓存。
- OutputConfig：CSV 目录。
- GuiConfig：Qt刷新周期、窗口标题、历史长度、热力图色阶、窗口尺寸、区域箭头和配色。
- TrainingConfig、PlotConfig：离线训练和绘图。
- ConsistenceCalibrationConfig：维护者内部统一配置，包含运行期开关/系数路径和离线输入、输出、状态、目标与裁剪参数；不得再建立第二个一致性配置类。
- FullApplicationConfig：以上配置的组合。

环境变量只提供默认值。优先级必须保持：

~~~text
CLI 显式参数 > 显式配置对象 > TANGENTIAL_* 环境默认 > config.py 内置默认
~~~

协议帧头、CRC、固定 12×7/84 通道布局、固定帧长度和 108 列 CSV 属于协议不变量，不复制到 config，也不在调用方重新定义。

修改 ``config.py`` 默认值只影响之后从该源码或重新构建wheel创建的默认配置；
已经安装的旧wheel和已经构造的对象不会自动更新。用户代码优先显式传入分类
配置对象，避免依赖全局默认值；多传感器场景必须为每个实例分别构造配置。

## 5. 数据和时序不变量

- 压力和六维力请求目标均为 200 Hz、5 ms 周期；单请求在途，设备响应慢时实际频率自然下降。
- 压力合法帧解析完成后立即记录真实 rx_t；rel_ms/delta_ms 不得由 GUI、主循环 sleep 或重采样生成。
- 压力帧按 seq 顺序驱动；每个合法压力帧最多处理和保存一次。
- 最小 API 的数据流固定为 ``TangentialSensorAPI → PressureSensor/decode → TangentialSampleProcessor._process_sample() → TangentialSample → TangentialFrameProcessor._to_tangential_frame() → TangentialFrame``；``TangentialSample`` 只供完整应用内部直接消费，公开处理器和传感器始终只返回 ``TangentialFrame``，不得从顶层、``tangential.api``、``tangential.runtime.__all__`` 或公开 ``sensor.pyi`` 导出内部类型。
- 每个力帧最多匹配一次；匹配窗口为 0.015 秒。
- 力通道不可用时，压力帧保存为NaN力字段；双传感器模式下，超过窗口仍未匹配的压力帧不写CSV，但必须继续推进状态机和GUI。
- 压力设备必需；六维力连接或普通帧校零失败时降级为压力模式。
- 启动校零和运行期重新归零只使用普通力数据帧，不发送额外置零命令；串口只能有一个消费者。
- CSV 只能由 storage/csv.py 的 TABLE_CSV_HEADER 和 build_csv_row 生成。
- CSV 固定列 ``adc_sum`` 表示同一帧 84 通道 ADC 之和；对象字段与 CSV 列均
  使用 ``adc_sum``，不得重新引入 ``total``、``sum`` 或其它别名。
- 压力数据流固定为 ``raw_data → consistence_data → base_data``：传输帧使用
  ``payload``，解码数组使用 ``raw_data``；启用一致性标定时所有 CoP、梯度、
  滑移、模型、GUI、终端和 CSV 都只消费 ``base_data=consistence_data``，关闭时
  ``consistence_data=None`` 且 ``base_data=raw_data``。
- 维护者只在 ``config.py`` 的 ``ConsistenceCalibrationConfig`` 类体内编辑运行时
  开关、CSV目录/匹配模式、末尾行数、断点间距、局部最大增益、NPZ路径、裁剪和
  覆盖策略。默认 ``csv_directory`` 必须解析为项目
  ``src/tangential/resources/consistence``，其中每个 ``*-<数值>G.csv`` 代表一个
  载荷段；默认 ``output_path`` 解析为项目
  ``src/tangential/resources/consistence_coeffs.npz``。
- 每个一致性CSV只取最后 ``tail_rows`` 个非空数据行，并分别计算84通道端点；
  该档公共ADC目标是84通道端点均值，文件名克重仅用于排序和审计。每通道按自身
  原始ADC排序，对对应公共目标做保序回归，添加零锚点，以
  ``minimum_breakpoint_step`` 保证输入断点严格递增，并用
  ``max_segment_scale`` 限制局部增益，且该上限不得超过100。运行时直接按每通道
  原始ADC选择分段，不得先猜测克重；非单调矛盾数据允许产生平台和残差，不得用
  失控增益强行精确通过。``target_breakpoints``是补偿曲线的公共ADC目标断点，
  不是旧式全局目标范围或状态列。
- 离线标定唯一命令是源码模式 ``PYTHONPATH=src
  /home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python -m
  tangential.processing.calconsistence``。该模块不使用 argparse、不访问硬件，按统一
  配置读取 CSV 和写出 NPZ；``force`` 默认 ``True``，连续运行必须覆盖并更新同名
  NPZ。底层 ``ConsistenceCalibrator.save(force=False)`` 和显式配置
  ``ConsistenceCalibrationConfig(force=False)`` 必须继续拒绝覆盖。用户 CLI 不得
  提供对应子命令或任何一致性开关/路径参数。
- ``FullAcquisitionSession.start()`` 必须先加载模型、构造处理器并验证一致性
  NPZ，再调用 ``auto_get_csv_path()``/``init_csv_file()``；系数缺失、损坏或形状
  错误时要关闭已经创建的传感器且目录中不得留下空 CSV。
- fit_coefs.bin 通过 package resource 加载，模型格式和预测输出不得改变。
- 滑移检测是运行时状态，不增加或删除 CSV 列；算法使用短窗 CoP、零填充斑块
  平移相关性、相对 anchor 大位移兜底和连续帧滞回。SLIP 期间 sample.angle
  使用 EMA 运动方向，退出帧为 0，并通过 PRSensorAngle.reanchor_origin 重锁
  全局 origin。region-only 只做整帧聚合 CoP 的全局检测，不做 per-region 滑移；
  多接触时 CoP 可能互相抵消，结果可能低估真实运动。
- 滑移方向必须保存在 detector 独立的 EMA 向量中，不能直接依赖会变化的静态
  CoP origin。SLIP 期间内部 anchor 每帧同步跟随当前 CoP；达到退出滞回后，
  必须在同一压力帧处理调用内同步重锚定 PRSensorAngle、清空方向并输出 0°，
  禁止改成异步“稍后重置”任务。短于窗口和进入滞回要求的运动可能不会被识别，
  这是当前算法的抗噪检测下限；如需退出后保留历史方向，应由上层事件记录实现，
  不得改变 ``is_slipping`` 表示当前状态的语义。
- GUI ``Direction`` 面板的 PZT 箭头保持固定长度，只表示 ``sample.angle``；
  ``Pressure Snapshot`` 红色 PZT 箭头方向也使用 ``sample.angle``，长度必须来自
  ``sample.angle_vector_magnitude``（STICK=静态 CoP delta 模长，SLIP=EMA滑移
  向量模长），不得重新用 ``hypot(sample.dx, sample.dy)`` 计算。Snapshot 蓝色
  力箭头和 Pressure Table 的 origin/current CoP 几何保持原语义。只有旧代码
  直接调用 ``RealTimePlot.set_data`` 且 ``angle_vector_magnitude=None`` 时允许
  回退到 ``hypot(cop_delta_x, cop_delta_y)``；正式会话必须显式传 sample 字段。
- 异常、Ctrl+C、窗口关闭和无数据退出必须释放停止事件、线程、进程、串口、CSV 和 Qt 资源。

## 6. 修改路由


| 需求                          | 首选位置                                          | 联动测试                           |
| ----------------------------- | ------------------------------------------------- | ---------------------------------- |
| 压力协议、CRC、调度和队列     | sensors/pressure.py                               | test_data.py、分发和集成测试       |
| 六维力协议、校零和进程        | sensors/force.py                                  | test_data.py、集成测试             |
| seq、缓存和时间匹配           | acquisition/buffer.py、runtime/synchronization.py | test_data.py、集成测试             |
| CoP、阈值、状态机、梯度、区域 | processing/cop.py                                 | API、GUI、集成测试                 |
| 模型读取和预测                | processing/calibration.py                         | 模型回归测试                       |
| 一致性标定                    | processing/calconsistence.py、config.py           | 标定器、源码入口、资源和数据流测试 |
| 最小 API 和示例               | api.py、runtime/sensor.py、examples/minimal.py    | test_tangential_api.py、结构测试   |
| 完整采集和清理                | runtime/session.py、application.py                | test_main_integration.py           |
| CSV 格式                      | storage/csv.py                                    | test_model_and_table.py、绘图测试  |
| GUI                           | gui/realtime.py                                   | test_plot_and_gui.py               |
| 训练和绘图                    | tools/training.py、tools/plotting.py              | test_training.py、test_plotting.py |
| CLI                           | cli.py                                            | test_cli.py                        |
| 公共导出和配置                | __init__.py、config.py                            | API、资源和分发测试                |

修改时先定位唯一实现，再复用已有函数/类；不得在调用方复制串口解析、CoP 公式、标定预测或 CSV 行格式。

## 7. 源码运行、构建和测试

源码运行：

~~~bash
TANGENTIAL_PYTHON=/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python
"$TANGENTIAL_PYTHON" -m pip install -r requirements.txt
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.examples.minimal
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.examples.full
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m tangential.processing.calconsistence
~~~

Cython>=3.1,<4 是构建依赖，已在 pyproject.toml 声明；requirements.txt 供 no-build-isolation 开发构建使用。

setup.py 当前把以下11个内部模块分别编译为同名扩展：runtime/sensor、runtime/session、runtime/synchronization、acquisition/buffer、sensors/pressure、sensors/force、processing/cop、processing/calibration、processing/calconsistence、processing/slip、storage/csv。新增或移动编译模块时必须同步更新 setup.py、同名 .pyi、package-data 和分发测试。

``BinaryWheelBuildPy.run`` 必须在复制package前清理旧 ``build/lib*/tangential``
输出，防止已删除或重命名的模块残留进wheel。分发测试必须明确检查旧模块名
不存在，不能只检查新模块存在。

Cython必须保持 language_level=3、annotation_typing=False、binding=True、embedsignature=True 和 always_allow_keywords=True。尤其不能删除 annotation_typing=False，否则类型注解会被当作运行时强类型，破坏原Python代码对 bytearray、NumPy数组等兼容输入的接受行为。

构建 wheel：

~~~bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
~~~

构建结果应为：

~~~text
dist/tangential_sensor-0.5.0-cp311-cp311-linux_x86_64.whl
~~~

完整测试：

~~~bash
PYTHONPATH=src \
QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -q
~~~

附加检查：

~~~bash
PYTHONPYCACHEPREFIX=/tmp/tangential-pycache-06 \
PYTHONPATH=src "$TANGENTIAL_PYTHON" -m compileall -q src/tangential tests
~~~

分发验收应检查 wheel 中存在11个内部 .so、11个同名 .pyi、py.typed、已确认的资源文件和CLI入口，同时不存在对应内部 .py、生成的 C/C++ 文件或 share/ 模型目录。`consistence_coeffs.npz` 只有在 `ConsistenceCalibrationConfig.csv_directory` 中的多段标定数据来源确认并生成后才能加入 wheel；不得把标定 CSV 本身或未经确认的数据打包。确认模型可在脱离源码目录加载，函数签名和文档可查看；源码模式和隔离安装模式都要运行完整测试。不要生成或向保密用户发布 sdist。

dist/中的wheel和build/中的中间产物不属于源码修改内容。最终交付时必须在报告中给出wheel绝对路径、平台标签和测试结果。

## 8. 版本控制限制

除非用户之后明确重新授权，否则不得执行Git写操作，包括 ``add``、``commit``、``revert``、``restore`` 和 ``reset``，也不得创建提交。只读状态检查仅在用户明确要求或任务确有必要且当前指令允许时进行；当前任务明确禁止Git命令时，不得运行任何Git命令。

用户未来明确授权回退已经提交的阶段时，应先核对精确目标，并优先使用 ``git revert <commit-hash>`` 保留历史；不得使用 ``git reset --hard`` 覆盖用户修改。无论是否允许版本控制操作，都必须保留与当前任务无关的预存修改。
