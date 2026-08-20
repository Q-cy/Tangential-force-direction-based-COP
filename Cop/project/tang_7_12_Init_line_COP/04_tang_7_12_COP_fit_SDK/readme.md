# Tangential Sensor SDK

Tangential Sensor SDK 用于采集 12×7 PZT 压力阵列和可选六维力传感器，提供 CoP、角度、梯度、标定、实时显示、CSV 保存以及离线分析能力。

用户只需要一个 wheel。源码包的内部模块不是使用接口。

## 安装

运行环境为 Python 3.11 或更高版本。构建好的 wheel 位于 `dist/`：

```bash
python -m pip install ./dist/tangential_sensor-0.2.0-py3-none-any.whl
```

上面的核心安装提供压力采集 API、传感器协议、CoP、标定和 CSV 能力，不安装 GUI 绘图库。需要完整采集应用和离线绘图时安装完整 extra：

```bash
python -m pip install "./dist/tangential_sensor-0.2.0-py3-none-any.whl[full]"
```

`requirements.txt` 是项目维护者使用的完整开发/GUI 锁定环境；正式安装依赖以 wheel 的 `pyproject.toml` 元数据为准。

安装后可用以下命令验证版本：

```bash
tangential --version
```

## 命令行

安装后统一使用 `tangential` 命令：

```bash
tangential example
tangential app
tangential plot
tangential fit
```

查看帮助：

```bash
tangential --help
tangential app --help
```

### `example`

采集压力阵列并在终端显示当前帧。默认压力串口为 `/dev/ttyUSB0`，可用参数覆盖：

```bash
tangential example --pressure-port /dev/ttyUSB0 --timeout 0.1
```

### `app`

启动完整采集、六维力同步、标定、CSV 和实时 GUI。默认设备为：

- 压力传感器：`/dev/ttyUSB0`
- 六维力传感器：`/dev/ttyUSB1`

默认输出目录为当前工作目录下的 `data/`，可以指定其他目录和匹配窗口：

```bash
tangential app \
  --pressure-port /dev/ttyUSB0 \
  --force-port /dev/ttyUSB1 \
  --save-dir ./data \
  --max-time-diff-ms 15
```

压力传感器是必需设备。压力连接失败时程序退出，不生成空 CSV。六维力传感器是可选设备；连接或软件校零失败时降级为压力模式，力相关字段写入 `NaN`。

### `plot`

从 CSV 表头按列名进行离线绘图：

```bash
tangential plot \
  --dir ./data \
  --files capture.csv \
  --columns Fy_cal,delta_Force_Y \
  --rows 100:500 \
  --save ./data/capture.png
```

使用 `--list` 查看目录中的 CSV，使用 `--mode full_analysis` 运行完整分析。

### `fit`

根据两个 CSV 训练模型。默认只写模型和评估图，不修改输入 CSV：

```bash
tangential fit \
  --xy-csv ./data/fx_fy.csv \
  --z-csv ./data/fz.csv \
  --output-model ./fit_coefs.bin \
  --output-plot ./fit_report.png
```

只有显式提供 `--write-back PATH` 时才会写回标定列；目标文件已存在时还必须提供 `--force`：

```bash
tangential fit \
  --xy-csv ./data/fx_fy.csv \
  --z-csv ./data/fz.csv \
  --write-back ./data/calibrated.csv \
  --force
```

## Python API

最小压力采集程序：

```python
from tangential import TangentialSensor

with TangentialSensor(pressure_port="/dev/ttyUSB0") as sensor:
    while True:
        sample = sensor.read(timeout_s=0.1)
        if sample is not None:
            print(sample.matrix)
            print(sample.cop_x, sample.cop_y, sample.angle)
```

内置模型是 package resource `tangential/resources/fit_coefs.bin`。如需使用外部模型，可设置环境变量：

```bash
export TANGENTIAL_MODEL_PATH=/path/to/fit_coefs.bin
tangential example
```

也可以在 Python 中显式传入 `model_path`。未设置覆盖路径时使用 wheel 内置模型。

## 数据和时序

完整应用保存固定 108 列 CSV。`rel_ms` 和 `delta_ms` 使用合法压力帧的真实接收时间计算，不插值、不重采样，也不由 GUI 刷新节拍生成。压力和六维力分别使用独立采集进程；两路以约 200 Hz 为请求目标，单请求在途，实际频率由设备响应时间决定。

六维力可用时，压力帧和力帧在 ±15 ms 内一对一匹配。六维力不可用时，每个合法压力帧仍保存，力相关字段为 `NaN`。压力连接或采集错误会被明确报告；六维力软件校零失败时降级为压力模式，资源在退出或异常时关闭。

## 从源码构建 wheel

维护者可以在项目根目录执行：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

wheel 显式包含静态模型 `tangential/resources/fit_coefs.bin`。当前发行版是纯 Python 的 `py3-none-any.whl`；以后若有必要，也可以把局部计算模块编译为平台相关 `.so` 并继续封装进 wheel。
