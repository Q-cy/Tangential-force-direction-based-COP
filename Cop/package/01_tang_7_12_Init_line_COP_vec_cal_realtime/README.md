# 压阻传感器 CoP 角度实时估计（Realtime）

一个 Python 实时包：把 **84 通道 PZT 触觉传感器** 的原始流（来自 USB 串口 / `libeskin_finger_sdk.so`）喂给 `PZTSensorAngle` 算法，逐帧输出 `(angle, dx, dy)`。

---

## ✨ 特性

- **真实硬件读帧**：通过 `libeskin_finger_sdk.so` 的 ctypes 绑定读串口
- **生产者-消费者架构**：后台线程 ~1kHz 读帧 + 主循环算算法，解耦
- **算法 (`PZTSensorAngle`)**：动态阈值学习 + 二次精修 + 按帧号 reset
- **纯 NumPy**：算法层不依赖 PyTorch / ROS

---

## 📂 项目结构

| 文件 | 作用 |
|---|---|
| `tang_7_12_InitCOP_realtime_package.py` | 算法实现（无 docstring） |
| `tang_7_12_InitCOP_realtime_package_note.py` | 同算法带中文 docstring（推荐 import 这个） |
| `run_realtime.py` | 主循环入口（生产者 + 消费者 + 5s 一次性 reset） |
| `data.py` | 串口读帧 + 时间戳缓冲（从 vaule 兄弟包拷贝） |
| `eskin_ffi.py` | `libeskin_finger_sdk.so` 的 ctypes 绑定 |
| `libeskin_finger_sdk.so` | 硬件抽象层共享库 |
| `example.py` | 算法本身的多 sensor 演示（**不连硬件**，纯算法 demo） |

---

## 🚀 快速开始

### 1) 跑实时（需要硬件在 `/dev/ttyUSB2`）

```bash
cd 01_…_realtime/
python run_realtime.py
```

预期输出：

```
压阻传感器就绪，开始采集... (Ctrl+C 退出)
frame     1  angle=  86.20°  dx=+0.10  dy=+0.00
frame     2  angle=  87.10°  dx=+0.12  dy=-0.05
...
[reset] 5s 触发 reset_origin()
...
```

### 2) 跑算法本身（不需要硬件）

```bash
python example.py
```

---

## 🔌 硬件依赖

- **PZT 触觉传感器**：12×7 = 84 cell，串口协议见 `libeskin_finger_sdk`
- **Linux 设备**：`/dev/ttyUSB2`（默认；改 `run_realtime.py` 里的 `data.PressureSensor(port=...)` 即可切别的串口）
- **共享库**：`libeskin_finger_sdk.so`（同目录，外部 SDK 提供）

---

## 📚 算法 API（`PZTSensorAngle`）

完整 API 文档见算法类的 docstring（`tang_7_12_InitCOP_realtime_package_note.py`）。摘要：

```python
PZTSensorAngle(
    rows: int = 12,
    cols: int = 7,
    threshold_factor: float = 5,    # thresh = threshold_factor × mean(前 collect_frames 帧总压力)
    collect_frames: int = 20,        # 阈值学习窗口 (0 = raw 模式, _thresh=0)
    stability_frames: int = 5,       # 连续低压 N 帧自动 reset_origin
    reset_at_frame: int = 0,         # 第 N 帧自动 reset_origin (0 = 禁用)
    refine_cnt: int = 10,            # 二次精修所需稳定帧数 (0 = 禁用)
    refine_distance: float = 0.1,    # 二次精修"稳定"距离阈值 (0 = 禁用)
)
```

| 公开方法 | 用途 |
|---|---|
| `get_all(adc_data) -> (angle, dx, dy)` | 一帧三件套（推荐） |
| `get_angle(adc_data) -> angle` | 仅角度（`get_all(...)[0]` 的简写） |
| `reset_origin()` | 清 origin + 二次精修状态，阈值保留 |

每实例独立状态——同进程多 sensor 不互相串扰。

---

## 🧮 主循环（`run_realtime.py`）

主循环是生产者-消费者模式：

```
┌─────────────────────┐                ┌──────────────────────────┐
│  PressureThread       │  ~1kHz         │  主循环 (main loop)        │
│  (后台 daemon 线程)    │ ────────────► │  buf.get_latest()         │
│  sensor.read_data()   │   Timestamped  │  algorithm.get_all(adc)   │
│  sensor.decode(raw)   │   Buffer (500) │  print(angle, dx, dy)    │
└─────────────────────┘                └──────────────────────────┘
```

- **后台线程 `PressureThread`**：从串口读帧、解码 84 通道、塞入 `TimestampedBuffer`（带锁线程安全队列）
- **主循环 `while True`**：从 buffer 拿最新帧（`get_latest()`），调 `algorithm.get_all(adc)` 算 `(angle, dx, dy)`，打印
- **5 秒一次性 `reset_origin()`**：`run_realtime.py` line 71-74——`start_ts` 之后 5 秒调一次，目的是"避免前几秒闲置帧污染基准"

> ⚠️ **5s reset 是外层硬编码**——不是 `PZTSensorAngle` 本身的功能。如果需要基于帧号或稳定性的自动 reset，直接用算法自带的 `reset_at_frame` / `refine_cnt` 参数。

---

## ⚠️ 注意事项

- **串口端口**：默认 `/dev/ttyUSB2`；改 `run_realtime.py` line 40 里的 `data.PressureSensor(port=...)` 即可。
- **共享库位置**：`libeskin_finger_sdk.so` 必须在同目录（`eskin_ffi.py` 用相对路径 `LIB_PATH = "libeskin_finger_sdk.so"` 加载）。
- **`collect_frames=0` 是 raw 模式**：`_thresh` 立刻 = 0，首帧非零压力即锁 origin。适合"无阈值学习"场景。
- **5s reset 是临时方案**：生产环境建议改用 `reset_at_frame=N` 或 `refine_cnt=N` 让算法自己控制。

---

## 📜 许可证

本目录暂无许可文件。除非上游仓库另有说明，请把代码视为
**项目作者保留全部权利**。如需公开发布请补充 `LICENSE` 文件。
