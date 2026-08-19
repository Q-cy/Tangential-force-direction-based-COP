# 04_tang_7_12_COP_fit_std：Codex/Agent 工作指南

本文是本目录的执行约束。修改前先核对实际实现、测试和用户要求；不要把本文件当作泛化项目介绍。

## 架构和公开接口

- `src/tangential/` 是唯一正式源码，wheel 是唯一用户交付物。
- `src/tangential/__init__.py` 暴露公共 Python API；导入 `tangential` 不得隐式加载 Qt、PyQtGraph 或 Matplotlib。
- `src/tangential/sensors/` 负责压力和六维力串口协议。
- `src/tangential/acquisition/` 负责带锁时间戳缓存和同步。
- `src/tangential/processing/` 负责 CoP、梯度和运行时标定。
- `src/tangential/storage/` 负责固定 108 列 CSV。
- `src/tangential/gui/` 只包含可选 GUI 实现。
- `src/tangential/resources/fit_coefs.bin` 是 wheel 内置静态模型，必须通过 package resource 加载。
- `src/tangential/training.py` 和 `src/tangential/plotting.py` 提供离线训练及绘图 API；`src/tangential/cli.py` 提供统一命令行入口。
- wheel 注册四个子命令：`tangential example`、`tangential app`、`tangential plot`、`tangential fit`。

## 环境和分发

- Python 最低版本为 3.11。
- `pyproject.toml` 是正式构建和运行依赖来源：核心依赖为 NumPy、SciPy、pyserial；GUI/绘图依赖只在 `full` extra 中声明。
- `requirements.txt` 只用于完整开发/GUI 锁定环境，不替代 wheel 元数据。
- 默认压力端口为 `/dev/ttyUSB0`，默认六维力端口为 `/dev/ttyUSB1`；默认输出目录为当前工作目录下的 `data/`。
- `TANGENTIAL_MODEL_PATH` 可以覆盖 wheel 内置模型路径。
- 训练默认不修改输入 CSV；只有显式 `--write-back` 才写回，覆盖已有目标还必须使用 `--force`。
- 不兼容旧根目录导入路径；测试和源码统一使用 `tangential.*`。

## 压力采集不变量

- 压力设备是必需设备，串口为 921600 波特率；生产采集使用 `multiprocessing` 的 `spawn` 独立进程，进程内部由单一 I/O 线程完成压力串口读写。
- 目标请求频率为 200 Hz、周期 5 ms；这是调度上限，不伪造固定 5 ms 的数据间隔。设备响应约 6 ms 时实际频率自然下降。
- 单请求在途：合法响应或 50 ms 超时后才能进入下一轮；每轮先清理输入、输出和解析缓存，再发送现有 14 B 请求。
- 使用 `select` 等待，单次等待最多 10 ms，单次读取最多 1024 B；解析支持分包、前导噪声、粘包、错误 CRC 和截断帧恢复。
- 合法完整帧解析完成后立即以 `time.perf_counter()` 记录 `rx_t`。`read_frame()` 的 `request_seq`、`tx_t`、`rx_t`、`latency_s`、`raw` 字段保持不变。
- 84 个 ADC 通道保持原始线序，不做 C++ 左右翻转、基线、增益矩阵或阈值处理。
- IPC 队列容量为 256；`queue_drops` 必须可观测且完整采集验收要求为 0。出现丢帧时不得宣称 CSV 完整。

## 六维力和同步不变量

- 六维力为可选设备，串口为 460800 波特率，使用 `49 AA ... 0D 0A` 的 28 B 普通帧和持久化接收缓存。
- 启动校零及运行期重新归零均使用普通力数据帧，不发送额外置零命令；校零收集 10 个有效帧，超时 1 s 或样本不足则禁用力通道。
- 串口只能有一个消费者；重新归零从已接收的新帧读取，不能由另一个线程直接读串口。
- 压力和六维力分别由独立 `spawn` 进程采集；父进程只消费带真实接收时间的帧。
- 压力帧具有单调 `seq`，通过 `get_after(seq)` 顺序消费；六维力通过 `find_closest(..., max_diff_s=0.015)` 一对一匹配。
- GUI 最高 60 Hz，只负责显示，不参与串口读取、采样调度或时间戳生成。
- 所有采集生命周期使用 `try/finally` 关闭停止事件、线程/进程、串口和 CSV；异常必须通知主线程。

## CSV、时间和模型不变量

- CSV 固定 108 列，表头和行顺序由 `src/tangential/storage/csv.py` 定义，不得复制第二套格式。
- `rel_ms` 是从首个已保存压力行开始的真实相对毫秒时间，`delta_ms` 是相邻已保存压力行的真实时间差；首行均为 0，禁止插值、重采样或固定网格伪造时间。
- `valid` 为接触/训练有效状态；训练优先筛选 `valid != 0`，缺少该列时回退到 `CoP_state != 0`。
- 不修改现有 CoP、标定算法、模型格式及内置 `fit_coefs.bin` 的预测输出，除非用户明确要求重新训练。

## 测试和 Git

修改后至少运行：

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/pzt-mplconfig \
python -m unittest discover -s tests -v

PYTHONPATH=src python -m compileall -q src/tangential tests
git diff --check
```

分发测试必须在临时源码副本构建 wheel，不能污染仓库的 `build/`、`dist/` 或 `egg-info`。至少验证：wheel 内包含所有 `tangential` 模块和 `tangential/resources/fit_coefs.bin`，包含 `dist-info/entry_points.txt`，不包含旧根模块或 `share/` 模型路径；隔离安装后四个子命令的 `--help` 和 `python -m tangential.cli --version` 均成功。

每次完成修改都创建 Git 提交，除非用户明确要求不提交。本阶段若用户要求不提交，保持工作区修改并由主代理最后提交。
