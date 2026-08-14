# CLAUDE.md

> 给Claude Code使用的工作手册。

## 项目画像
### 项目名: PZT_Hall（03_tang_7_12_COP_fit_sd）
### 定位: 压阻阵列(PZT) + 六维力传感器 数据采集 / 标定 / 实时显示 / 数据保存
### 技术栈: python（PyQtGraph 实时绘图，scipy/numpy 拟合，pyserial 串口）

## 架构分层

```
03_tang_7_12_COP_fit_sd/
├── main.py                              # 主入口：实时采集线程 + 标定引擎选择 + CSV 保存 + 绘图调度
├── realtime.py                          # PyQtGraph 实时绘图窗口（RealTimePlot + CellGridItem + GridLinesItem）
├── data.py                              # 串口采集：PressureSensor（14×5 PZT） + SixAxisForceSensor + TimestampedBuffer
├── fit.py                               # 拟合标定（sym_log/sym_exp/exp_log/pchip/sigmoid/poly/exp） + 训练脚本(__main__)
├── table.py                             # CSV 表头(TABLE_CSV_HEADER) + 行构造(build_csv_row)
├── plot_static.py                       # 离线 CSV 静态图绘制 CLI 工具
├── tang_7_12_InitCOP_realtime_other_package.py   # 标量角度工具：compute_vector_angle / compute_6Dforce_angle / angle_difference
├── tang_7_12_InitCOP_realtime_package_note.py    # PZTSensorAngle 类：CoP 计算 / 角度估计 / 压力梯度 / 阈值+origin+二次精修
├── fit_coefs.bin                        # 训练好的 fit 模型（运行时由 main.py 加载）
├── readme.md                            # 串口粘包问题说明（持久化接收缓冲区方案）
└── CLAUDE.md                            # 本文件
```

## 数据流

```
压阻(PZT)串口 /dev/ttyUSB0
  → PressureSensor（CRC8 校验 + 持久化缓冲区 + 10ms 轮询）
  → TimestampedBuffer
  → PZTSensorAngle.get_all → (angle, dx, dy, cop_x, cop_y)
  → PZTSensorAngle.get_gradient → (14, 5, 2) 梯度

六维力串口 /dev/ttyUSB1
  → SixAxisForceSensor（28B 帧，持久化缓冲 + 帧头尾校验）
  → TimestampedBuffer

两者经严格时间匹配后（MAIN_MAX_TIME_DIFF_S=15ms 窗口内配对，超窗跳过该行）
  → 标定引擎（fit.py）
  → CSV 持久化（table.py）+ PyQtGraph 实时绘图（realtime.py）
```

## 标定模式

唯一标定引擎：`fit.py`（拟合标定，加载 `fit_coefs.bin`）。

## 编码规范
### 命名约定
- 模块化函数命名：模块名.功能名

### 代码风格
- 把功能封装成函数/类，主函数只调用
- 线程间共享数据用 `TimestampedBuffer`（自带锁）
- 全局运行标志可用模块级 `threading.Event`（如 `g_main_stop_flag`），但状态对象尽量用类封装

## 注意事项
- 同时接入压阻阵列（14×5 = 70 通道 PZT）+ 六维力传感器
- 压阻帧：14B header + 140B payload + 1B CRC，CRC-8 ITU 校验，10ms 轮询
- 数据读取时掩码到 2×2 活跃区（0-based 行 3-4、列 1-2 = 1-based 坐标 (4,2)(4,3)(5,2)(5,3)），其余 cell 置 0；PRSensorAngle 内亦有活跃区掩码，总压/阈值/CoP/梯度/形心显式只针对掩码内 cell，其他 cell 不参与；接触阈值 total_threshold_factor≈0.6（掩码后基线≈区域总压）
- 梯度差值：首次接触锁 origin / 二次精修完成时记录初始梯度，输出当前−初始差值（实时梯度图 + CSV 8 列 grad_x/grad_y_d{r}c{c}，0-based 坐标）
- 六维力帧：28B，持久化缓冲 + 帧头 49 AA 帧尾 0D 0A 校验（不再每次 reset_input_buffer）
- CoP 状态机：阈值→首次接触 origin 锁定→稳定 N 帧触发二次精修（精修后重新归零 Fx/Fy）
- CSV 表头含 valid 列（1=接触帧有效），fit.py 训练筛选 / plot_static.py 高亮依赖该列
- 训练产出物：`python fit.py` → fit_coefs.bin（需手动 copy 到项目目录）
- 离线分析：`python plot_static.py -f xxx.csv -c col1,col2 -r 100:500`
