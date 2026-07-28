# CLAUDE.md

> 给Claude Code使用的工作手册。

## 项目画像
### 项目名: PZT_Hall
### 定位: 压阻阵列数据读取+数据处理+实时显示+数据保存
### 技术栈: python

## 架构分层

tang_7_12_Init_line_COP_vec_cal/01_tang_7_12_COP_cal_fit_realtime/
├── main.py          #主函数，
├── realtime.py      #用于实时显示各种数据
├── data.py          #用于读取各种数据
├── fit.py           #用于标定数据
├── angle.py         #用于计算角度
├── COP.py           #用于计算COP(Center of Pressure)
├── table.py         #用于保存数据至csv文件
├── plot_static.py   #用于把数据绘制成静态图
├── calibrate.py     #查找表标定(lookup + discrete)
├── eskin_ffi.py     #libeskin_finger_sdk.so 的 ctypes 封装
├── libeskin_finger_sdk.so
└── CLAUDE.md        (给Claude Code使用的工作手册)

## 编码规范
### 命名约定
函数命名：模块名.功能名，如

### 代码风格
- 把各个功能封装成函数，在各个主函数中都只是调用函数
- 少用全局变量，建议用类封装

## 注意事项
- 本项目只有压阻阵列，所以只需要处理压阻数据的代码
