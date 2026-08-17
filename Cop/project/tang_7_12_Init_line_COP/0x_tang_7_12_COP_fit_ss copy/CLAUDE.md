# CLAUDE.md

> 给Claude Code使用的工作手册。

## 项目画像
### 项目名: PZT_Hall
### 定位: 压阻阵列数据读取+数据处理+实时显示+数据保存
### 技术栈: python (pyqtgraph + pyserial + numpy + scipy + Qt)

## 架构分层
03_tang_7_12_COP_fit_ss/
├── main.py                       #主函数，Qt事件循环+多线程编排
├── data.py                       #用于读取数据
├── tangential_7_12_package.py    #PZT阵列算法封装（PZTSensorAngle类：CoP/角度/梯度/二次精修）
├── tangential_7_12_other_package.py  #通用角度工具（compute_vector_angle等）
├── calibrate.py                  #用于查找表标定（lookup最近邻 / discrete双线性插值）
├── fit.py                        #用于函数拟合标定（poly/log/exp/sigmoid/pchip/sym_*）
├── table.py                      #用于保存数据至csv文件
├── realtime.py                   #用于实时显示各种数据
├── plot_static.py                #用于把数据绘制成静态图
├── fit_coefs.bin                 #fit模型二进制持久化（fit.py产出，main.py热加载）
├── readme.md                     #粘包问题解决方案说明
└── CLAUDE.md                     (给Claude Code使用的工作手册)

## 编码规范
### 命名约定
- 函数命名：模块名.功能名，如 table.auto_get_csv_path
- 类封装多实例算法（如 PZTSensorAngle 支持多 sensor 实例）
- 全局配置常量加模块前缀：MAIN_*（main.py）、CAL_*（calibrate.py）、FIT_*（fit.py）、PLOT_*（realtime.py）
- 私有方法/字段：单下划线前缀（如 _compute_cop、_thresh）

### 代码风格
- 把各个功能封装成函数/类，在main中都只是调用
- 少用全局变量，建议用类封装
- 多线程模型：采集线程（PressureThread/ForceThread）+ 主循环（data_loop）+ Qt GUI，通过 TimestampedBuffer 跨线程传数据
- 传感器连接失败时不中断程序，跳过该数据源继续运行

## 注意事项
- 本项目同时使用压阻阵列（PZT 12×7）和六维力传感器，两个数据源独立采集
- 标定分两条路径：calibrate.py（查找表）vs fit.py（函数拟合），由 main.py 中 MAIN_CAL_MODE 选择引擎
- PressureSensor 收发解耦（_tx_loop + _rx_loop + 持久化 _rx_buf），解决串口粘包/分包
- fit_coefs.bin 是 fit.py 的训练产物；若重训练需重新生成，main.py 会自动热加载