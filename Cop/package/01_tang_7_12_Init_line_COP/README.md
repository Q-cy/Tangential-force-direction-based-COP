# PZT 压力中心（CoP）方位角估计

一个轻量的 Python 类，把 **84 通道压阻（PZT）触觉传感器帧** 转成二维
**压力中心（Center-of-Pressure, CoP）位移** 与 **方位角**。
面向机器人 / 触觉项目里的指尖切向力估计场景。

---

## ✨ 特性

- **纯 NumPy**——算法层不依赖 PyTorch / ROS 。
- **每实例独立状态**——一个传感器一个实例，同进程内多 sensor 数据流不会互相串扰（基线、origin 各自独立）。
- **自学习低压阈值**——启动期累积前 *N* 帧总压力，阈值取 *K × 平均*。无需为环境压力手动校准。
- **极简公开 API**——三个方法：`get_all`（一次性三件套）/ `get_angle`（仅角度）/ `reset_origin`（重新锁定首次接触）。
- **几何可配**——构造参数 `rows` / `cols`，默认 12×7 = 84 个 cell。

---

## 🚀 快速开始

```python
import numpy as np
from tang_7_12_InitCOP_realtime_package import PZTSensorAngle

sensor = PZTSensorAngle()                       # 默认：12×7, k=5

# 1) 前 `collect_frames=20` 帧作为初始帧
#    这里用任何"无接触"帧都行
for _ in range(sensor.collect_frames):
    sensor.get_angle([np.random.randint(0, 50) for _ in range(84)])

# 2) 真实接触帧：84 个 ADC 值，按行优先 (rows × cols) 排列
contact = [200] * 84                             # length 84
angle, dx, dy = sensor.get_all(contact)

print(f"angle={angle:.1f}°  dx={dx:+.2f}  dy={dy:+.2f}")
#   →  例如  "angle=42.7°  dx=+1.32  dy=-0.45"
```

可运行的多 sensor 演示见 [`example.py`](./example.py)。

---

## 📚 API 参考

### `class PZTSensorAngle(...)`

```python
PZTSensorAngle(
    rows: int = 12,
    cols: int = 7,
    k: float = 5,
    collect_frames: int = 20,
    stability_frames: int = 5,
)
```

| 参数               | 默认值 | 含义                                                                       |
|--------------------|--------|----------------------------------------------------------------------------|
| `rows`             | `12`   | 传感器行数（构成 (rows × cols) 数组的一维）。                              |
| `cols`             | `7`    | 传感器列数。输入 ADC 序列的期望长度 = `rows * cols`。                      |
| `k`                | `5`    | 阈值倍数：`low_thresh = k × mean(前 N 帧总压力)`。值越大越不容易被判为低压。 |
| `collect_frames`   | `20`   | 启动期用于学习低压阈值的帧数。                                              |
| `stability_frames` | `5`    | 连续多少帧低压后自动重新锁定 origin。                                       |

### `sensor.get_all(adc_data) -> tuple[float, float, float]`

```python
angle, dx, dy = sensor.get_all(adc_data)
```

- **`adc_data`** —— 长度 `rows * cols`（默认 `84`）的序列，原始 ADC 值。
- **返回** `(angle, dx, dy)`：
  - `angle`：方位角，范围 `[0°, 360°)`，方向由 CoP 相对首次有效接触帧的位移给出。
  - `dx`：CoP X 方向位移（列方向），单位 **cell**。
  - `dy`：CoP Y 方向位移（行方向），单位 **cell**。
- **`len(adc_data) != rows * cols` 时抛 `ValueError`**。

### `sensor.get_angle(adc_data) -> float`

便捷访问器——完全等价于 `sensor.get_all(adc_data)[0]`。
当只关心角度、不想解包元组时用这个。

### `sensor.reset_origin() -> None`

丢弃已锁定的首次接触 origin 和低压计数。**已经学好的 `k × 均值` 阈值会保留**。

调用后，下一帧有效接触会成为新的参考原点：那一帧 `dx = dy = 0`；
之后帧返回相对该新原点的位移。

这是"重校但不再学阈值"的推荐做法（例如两次抓取动作之间）。

---

## 🧮 算法

每帧流水线：

```
adc_data (1D, length rows*cols)
        │
        ▼
compute_cop(raw_frame)            ──► (dx, dy)
   ├─ 重塑为 (rows, cols)
   ├─ update_dynamic_threshold(total_pressure)
   │       · 启动期：append 到 deque(maxlen=N)
   │       · 满后：    thresh = K × mean(buf)
   ├─ 若阈值已确定：低压计数器，N 帧连续低压 → reset
   ├─ total CoP = Σ(p · coord) / Σp
   └─ 首次有效帧锁 origin；后续帧返回 (current − origin)
        │
        ▼
compute_cop_angle(dx, dy)         ──► 角度（度，[0, 360)）
   └─ compute_angle(dx, -dy)                  # PZT 阵列 Y 朝向翻转
        └─ arctan2 → 度 → 归一到 [0, 360)
```

**CoP**（压力中心）是传感器表面上标准的压力加权重心。返回的 `angle`
是 `ΔCoP` 相对锁定原点的方位——其方向指示用户接触偏移到了哪里
（在指尖场景下等价于切向力推动的方向）。

二维 Y 轴做了翻转（角度计算里的 `-dy`），原因是 PZT 阵列读数是自顶向下排列，
而世界坐标的 Y 指向上方。

---

## 📂 项目结构

| 文件                                              | 作用                                                            |
|---------------------------------------------------|----------------------------------------------------------------|
| `tang_7_12_InitCOP_realtime_package.py`           | 算法实现
| `example.py`                                      | 可运行的多 sensor 演示，含 `reset_origin` 用法。               |
---

## 🧪 运行示例

```bash
cd 01_tang_7_12_Init_line_COP_vec_cal/
python example.py
```

预期输出（数字每次随机，运行结果会变）：

```
sensor_a (get_angle): angle=48.09°
sensor_b (get_all): angle=79.33° dx=+0.13 dy=-0.69
```

demo 流程：

1. 构造两个 `PZTSensorAngle` 实例（各自独立状态）；
2. 每路各喂 `collect_frames=20` 闲置帧，让其低压阈值收敛；
3. 每路再喂接触数据：`sensor_a` 用 `get_angle`，`sensor_b` 用 `get_all`；
4. 调用 `sensor_a.reset_origin()`，注释说明新 origin 行为。

---

## 📜 许可证

本目录暂无许可文件。除非上游仓库另有说明，请把代码视为
**项目作者保留全部权利**。如需公开发布请补充 `LICENSE` 文件。
