# 04_tang_7_12_COP_fit_std：Codex/Agent 工作指南

本文是本目录的执行约束。修改代码前先阅读本文和相关模块；不要把它当作泛化的项目介绍。若代码现状与本文冲突，先核对实际实现、测试和用户要求，再更新本文或提出风险。

## 0. 代码风格

- 规范源码位于 `src/tangential/`；根目录旧模块只能作为兼容转发层，不得新增第二套实现。
- 公共压力 API 位于 `src/tangential/api.py`，通过 `src/tangential/__init__.py`
  暴露；导入 `tangential` 不得隐式加载 Qt。
- 六维力、同步、CSV、GUI 和完整会话按功能分层，并复用公共 API。
- `example.py` 是最小示例：终端固定显示 12×7 ADC、min、max、sum、mean、copX、copY、angle。
- `main.py` 是完整示例，必须保留显式采集 `while` 循环；循环只编排会话对象的公开方法。

## 1. 项目边界与环境

- 项目名和工作目录：`04_tang_7_12_COP_fit_std`。
- 运行时：Python 3.11；本机验证环境为 Conda 环境 `TimeDrift_GRU`。
- 项目用途：12×7 PZT 压力阵列与可选六维力传感器采集、CoP/角度计算、拟合标定、实时显示和 CSV 保存。
- 当前模块树：

  ```text
  pyproject.toml
  src/tangential/
  main.py
  example.py
  fit.py
  plot_static.py
  data.py / table.py / realtime.py             # 兼容层
  tangential_*_package.py                      # 兼容层
  fit_coefs.bin
  requirements.txt
  tests/
  ```
- 本项目可能与兄弟目录共用 Git 工作树。保护无关改动，不使用宽泛路径执行覆盖、删除或回退操作。

## 2. 压力采集约束

设备路径为 `/dev/ttyUSB0`，波特率 `921600`。生产环境的 `PressureSensor` 使用 `multiprocessing` 的 `spawn` 独立进程；该子进程内部运行本地压力 I/O 线程，采集过程不能被 GUI、CoP 或标定计算拖延。

当前协议和调度参数必须保持：

- 目标频率 `200 Hz`，目标周期 `5 ms`。这是上限/调度目标，不是伪造固定5 ms数据间隔；设备实际响应约 `6 ms` 时，实际帧率自然低于200 Hz。
- 单请求在途：收到合法响应或发生超时后，才能进入下一轮；禁止并发请求、补发密集请求或积累未完成请求。
- 每轮先 flush 输入、输出和解析缓存，再发送现有 `14 B` 请求；响应等待上限 `50 ms`。
- 使用 `select` 等待，单次等待最多 `10 ms`，单次读取最多 `1024 B`。
- 响应按 `AA 55` 帧头和动态长度解析；校验 CRC-8-ITU、状态字段以及 `168 B` 传感器载荷。载荷解码为原始线序的 `84` 个 `uint16` 通道，不做 C++ 左右翻转、基线、增益矩阵或阈值处理。
- 完整合法帧解析完成后立即用 `time.perf_counter()`记录 `rx_t`。`read_frame()` 返回接口及字段 `request_seq`、`tx_t`、`rx_t`、`latency_s`、`raw` 必须保持不变。
- IPC 压力帧队列容量为 `256`。当前实现溢出时增加 `queue_drops` 并继续运行，
  此后 CSV 已不能视为完整采集；日志和硬件验收必须明确检查该值且要求为 `0`，
  不得忽略后继续宣称数据完整。

同一轮的接收缓存必须支持分包、前导噪声、粘包、错误 CRC 后重新找帧头和截断帧恢复。正常轮次按逐轮隔离策略清理残留数据；不要随意改成无条件每5 ms发送，也不要在没有协议请求序号的情况下把跨请求的晚到帧混入下一请求。

## 3. 六维力采集约束

设备路径为 `/dev/ttyUSB1`，波特率 `460800`。协议为 `28 B` 普通帧，帧头/帧尾为 `49 AA ... 0D 0A`；接收使用持久化字节缓存，不能依赖每轮清空串口来处理粘包。

- 六维力是可选设备。连接失败、普通帧软件校零失败或有效样本不足时，关闭/禁用力通道并降级为压力单独模式。
- 启动校零和运行期重新归零都使用普通力数据帧，不额外发送“置零命令”。校零收集 `10` 个有效原始帧，超时 `1 s` 或样本不足返回失败。
- 串口只能有一个消费者。运行期归零从 `ForceThread` 缓冲区读取新帧，不允许另一个线程直接调用串口 `read()`。

## 4. 主流程与同步语义

数据流为：

```text
PressureSensor → PressureThread → TimestampedBuffer
SixAxisForceSensor → ForceThread → TimestampedBuffer
压力帧 → CoP/角度/梯度/标定/GUI/CSV
压力帧与力帧 → 时间匹配 → CSV中的力字段和标定字段
```

- 压力和六维力生产路径分别由独立 `spawn` 子进程负责串口 I/O；父进程的
  `PressureThread`/`ForceThread` 只消费带真实 `rx_t` 的帧。测试注入串口时可退回本地线程。
  线程间共享数据必须通过带锁的 `TimestampedBuffer`。
- `TimestampedBuffer` 的帧具有单调递增 `seq`，使用 `get_after(seq)` 获取尚未处理的帧，使用 `find_closest(ts, max_diff_s, min_seq)` 做未使用力帧匹配。
- 压力是主驱动。主循环约 `100 Hz` 批量处理全部新的压力帧；不能只取最新帧，也不能让 GUI 刷新节拍决定采样节拍。
- GUI最高 `60 Hz`，只负责显示，不参与压力请求、串口读取或采样调度。
- 六维力目标频率同为 `200 Hz`、周期 `5 ms`、响应超时 `50 ms`；实际频率由设备响应限制。
  力帧的 `rx_t` 必须在完整 `49 AA ... 0D 0A` 帧确定后立即记录，用于 ±15 ms 匹配。
- 有六维力时，在 `±15 ms`（`MAIN_MAX_TIME_DIFF_S=0.015`）内一对一匹配；一个力帧最多匹配一次。当前实现中，超过窗口未匹配的压力行不写入 CSV，但该压力帧仍必须推进阈值、CoP、精修、标定状态和 GUI。
- 没有六维力时，每个合法压力帧都写入一行；力字段及其派生字段写 `NaN`。
- 所有采集生命周期使用 `try/finally`：设置停止事件、停止并等待线程/进程、关闭串口和 CSV；仅在确实没有数据行时删除本次空文件。线程异常必须通知主线程，不能让 GUI在采集已静默死亡后继续显示。

## 5. CSV、时间和训练数据

CSV格式固定为 `108` 列，不得随意增删或重排：

- 第1列 `rel_ms`：相对第一条已保存压力行的毫秒时间。
- 第2列 `delta_ms`：与上一条已保存压力行的时间差。
- 第3列 `adc_sum`。
- 后续 `84` 列：PZT原始 ADC 通道，顺序必须保持。
- 其余列为现有 CoP、力、标定、角度、状态等字段，表头和顺序由 `table.py` 维护。

`rel_ms`、`delta_ms`均基于已保存压力行的原始 `rx_t`，首行固定为 `0`；`press_t`保留原始 `perf_counter`时间语义。禁止通过 sleep、插值、重采样或固定时间网格伪造压力时间。需要重采样时只能在离线分析中另行完成。

`valid` 表示接触/训练有效状态（`1`为有效）；训练筛选由 `TRAIN_VALID_ONLY=True` 控制。缺少 `valid` 列时按既有兼容逻辑回退到 `CoP_state != 0`。

当前训练数据结论：

- Fx/Fy：`/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0713_1.csv`，使用 `valid != 0` 的 `20,954` 行。
- Fz：`/home/qcy/Project/data/2.PZT_tangential/weight/concat/concat_5_10_15.csv`，使用 `valid != 0` 的 `23,508` 行。
- `fit_coefs.bin` 是当前运行模型；稳定性修改不得改写、替换或重新训练它，除非用户明确要求训练。

## 6. 必须保持的不变量

- `PressureSensor.read_frame()`、`decode()`及其下游接口兼容。
- 84个压力通道的原始线序不变。
- CSV为108列，表头、列顺序和字段含义不变；规范定义位于
  `src/tangential/storage/csv.py`。
- 不改变现有 CoP、标定算法、模型加载方式和 `fit_coefs.bin` 的模型输出。
- 压力时间戳必须是真实合法帧解析时间；不能用 GUI时间、CSV写盘时间或固定频率替代。
- GUI不能参与采样调度；压力和力串口分别只能有一个消费者。
- 资源必须在异常、窗口关闭、无数据退出时通过 `finally` 关闭。
- 不静默丢帧。任何队列溢出、序号断层、协议错误、超时和线程异常都必须可观测；
  一旦 `queue_drops > 0`，必须明确报告本次 CSV 不完整。若任务要求完整记录，应停止本次
  记录，而不是继续生成看似正常的数据。

## 7. 诊断字段和判读

运行日志中的字段按以下含义解释：

- `pressure fps`/帧率：统计窗口内收到或处理的合法压力帧率，需明确区分“收到”和“写入 CSV”。
- `request interval P50/P95`：相邻请求发送时间的中位数/P95；反映请求调度，不等于传感器真实帧间隔。
- `response latency P50/P95`：单次请求从 `tx_t` 到对应合法响应 `rx_t` 的时间；优先用它判断设备响应、串口和调度是否变慢。
- `rx intervals`：相邻合法压力帧 `rx_t` 的差值；这是判断 CSV采样节拍的主要指标。
- `timeout`：等待合法响应超过 `50 ms` 的轮次数。
- `CRC`：收到候选完整帧但 CRC校验失败的次数。
- `status`：帧结构和 CRC正确但协议状态字段非0而被丢弃的次数。
- `queue_drops`/`queue_overruns`：IPC或缓存满导致的帧丢弃/溢出；这是数据完整性错误。
- `schedule_skips`：本轮实际耗时达到或超过目标 `5 ms`，因此没有可补的周期睡眠次数；不是丢帧、不是跳过合法响应，也不是数据缺失计数。设备稳定约6 ms时它持续增加是正常的。

验收时优先查看 `latency P50/P95`、连续 `rx intervals`、超时/协议错误和队列溢出；不要只用 GUI刷新率或主循环频率判断采集稳定性。

## 8. 修改、测试与 Git 协作

修改前：

- 先确认目标文件和调用链，保护工作树中的兄弟目录及用户已有改动。
- 主代理负责拆解计划、确定边界、复核结果和提交；Luna负责边界清晰、可验证的执行任务（可用时）。
- 使用 `apply_patch` 做局部修改；不要用 `git reset --hard`、`git checkout --`或宽泛删除命令覆盖用户数据。

修改后至少运行：

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/pzt-mplconfig \
/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python -m unittest discover -s tests -v

/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python -m py_compile \
  main.py example.py data.py table.py realtime.py fit.py plot_static.py \
  tangential_other_package.py tangential_package.py \
  $(find src/tangential -name '*.py' -print)

git diff --check
```

硬件验收至少连续采集1000个合法压力帧：观察实际 `rx_t` 间隔、响应延迟、超时、CRC/状态错误、序号连续性和队列溢出。正常设备约6 ms响应时，不要求伪造200 Hz；应确认没有请求堆积、重复帧、静默丢帧，异常长间隔能由超时或串口错误解释。双传感器模式还需确认有效匹配满足 `abs(force_t - press_t) <= 0.015`。

每次完成项目修改（包括代码、测试和文档）都必须创建一个新的 Git 提交，便于恢复；
提交时只 stage 当前任务涉及的文件，不要把兄弟目录或无关改动带入提交。恢复优先使用：

```bash
git revert <commit>
```

不要使用 `git reset --hard` 破坏工作树。若用户明确要求本次只改文档且不提交，则遵从该要求。

## 9. 常用命令与离线绘图

启动采集：

```bash
/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python main.py
```

训练/重新生成模型（只有用户明确要求时执行）：

```bash
/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python fit.py
```

离线绘图按实际 CSV表头解析，不使用可能错位的硬编码列索引：

```bash
/home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python plot_static.py \
  -f /path/to/data.csv -c col1,col2 -r 100:500
```

离线工具遇到空 CSV、空区间或缺失列时应给出明确错误，不得产生难以定位的索引异常。
