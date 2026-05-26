# Eskin Finger SDK

E-Skin 手指力传感器 Rust SDK，提供串口通信、寄存器读写、流式采集能力，支持 Rust / C/C++ / Python / ROS2 调用。

## 目录结构

```text
src/
  lib.rs          — Rust 库入口
  device.rs       — 设备管理（open/close/read/write）
  stream.rs       — 流式采集（PollingSampleCollector）
  protocol.rs     — 协议帧编解码（CRC-8/X25）
  register.rs     — 寄存器地址定义与解析
  transport.rs    — 串口传输抽象
  channel.rs      — 线程间 Channel
  config.rs       — 配置与设备信息
  error.rs        — 错误类型
  types.rs        — 数据类型定义
  ffi/mod.rs      — C FFI 导出函数

include/
  eskin_ffi.h     — C/C++ 头文件

example/
  cpp/main.cpp       — 独立 C++ 使用示例（含流式采集）
  python/eskin_ffi.py — Python FFI 包装器
  python/example.py   — Python 使用示例（含流式采集）
  ros-cpp/            — ROS2 C++ 示例（publisher/subscriber）
```

---

## 快速开始（Rust）

```rust
use eskin_finger_sdk::config::DeviceConfig;
use eskin_finger_sdk::device::{EskinDevice, EskinDeviceInner};
use eskin_finger_sdk::transport::SerialPortTransport;

let transport = SerialPortTransport::new("/dev/ttyUSB0", 921600);
let config = DeviceConfig::default();
let mut device = EskinDeviceInner::new(config, Box::new(transport));
device.open().unwrap();

// 读寄存器（原始字节）
let data = device.read_register(0x0000, 4).unwrap();
println!("Serial: {:?}", data);

// 写寄存器
let count = device.write_register(0x0030, &[0x01, 0x00, 0x00, 0x00]).unwrap();
println!("Wrote {} bytes", count);

device.close().unwrap();
```

### 流式采集（Rust）

```rust
// 启动流式采集
device.start_stream().unwrap();

// 读取采样数据
let sample = device.read_sample(Some(std::time::Duration::from_millis(200))).unwrap();
println!("force: fx={} fy={} fz={}", sample.force.fx, sample.force.fy, sample.force.fz);

// 停止采集
device.stop_stream().unwrap();
```

---

## 使用 C/C++

### 构建动态库

```bash
# 安装依赖（Ubuntu）
sudo apt install pkg-config libudev-dev

# 构建
cargo build --release
# 输出: target/release/libeskin_finger_sdk.so
```

### 编译 C++ 示例

```bash
g++ -std=c++17 example/cpp/main.cpp -I include -L target/release -leskin_finger_sdk -lpthread -o example_cpp
LD_LIBRARY_PATH=target/release ./example_cpp
```

### C++ 代码示例

完整示例见 `example/cpp/main.cpp`，包含 Command 模式和 Streaming 模式：

```cpp
#include "eskin_ffi.h"
#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <queue>

int main() {
    // 打开设备
    EskinDeviceHandle dev = eskin_open("/dev/ttyUSB0", nullptr);
    if (!dev) return 1;

    // Command 模式：读取设备信息
    char hw_buf[64] = {};
    uint32_t hw_len = 0;
    eskin_read_hdw_version(dev, hw_buf, sizeof(hw_buf), &hw_len);
    printf("Hardware version: %.*s\n", (int)hw_len, hw_buf);

    // Streaming 模式：持续采集力数据
    eskin_start_stream(dev);

    CFingerSample sample;
    memset(&sample, 0, sizeof(sample));
    if (eskin_read_sample(dev, 200, &sample) == ESkinSuccess) {
        printf("force: fx=%u fy=%u fz=%u\n",
               sample.combined_force.force.fx,
               sample.combined_force.force.fy,
               sample.combined_force.force.fz);
    }

    eskin_stop_stream(dev);
    eskin_close(dev);
    return 0;
}
```

---

## 使用 Python

### 构建 & 准备

```bash
cargo build --release
# 将生成的 .so 文件复制到 python 示例目录下
cp target/release/libeskin_finger_sdk.so example/python/
cd example/python
python3 example.py
```

> **注意：** Python 示例默认从当前目录加载 `libeskin_finger_sdk.so`，请确保 `.so` 文件已复制到 `example/python/` 目录下，或修改 `eskin_ffi.py` 中的 `LIB_PATH` 指向正确的路径。

### Python 代码示例

完整示例见 `example/python/example.py`，包含 Command 模式和 Streaming 模式：

```python
from eskin_ffi import EskinDevice

with EskinDevice() as dev:
    dev.open("/dev/ttyUSB0")

    # Command 模式：读取设备信息
    print(f"Hardware version: {dev.read_hdw_version()}")
    print(f"Matrix: {dev.read_matrix_row()} x {dev.read_matrix_col()}")

    # Streaming 模式：持续采集力数据
    dev.start_stream()
    for _ in range(10):
        sample = dev.read_sample(timeout_ms=200)
        f = sample.combined_force.force
        print(f"fx={f.fx}  fy={f.fy}  fz={f.fz}")
    dev.stop_stream()
```

---

## ROS2 示例

ROS2 C++ 示例位于 `example/ros-cpp/`，包含：

- `eskin_publisher.cpp` — 独立线程读取力数据，定时发布到 `/comb_force` topic（`std_msgs/UInt32`）
- `eskin_subscriber.cpp` — 订阅 `/comb_force` topic 并打印
- `CMakeLists.txt` / `package.xml` — ROS2 包配置

### 构建

```bash
cd your_ros2_ws/src
ln -s /path/to/eskin-finger-sdk/example/ros-cpp eskin_example
cd .. && colcon build --packages-select eskin_example
```

### 运行

```bash
# 终端1：启动 publisher（指定设备路径）
ros2 run eskin_example eskin_publisher /dev/ttyUSB0

# 终端2：订阅数据
ros2 run eskin_example eskin_subscriber
```

---

## 数据类型

```c
typedef struct {
    uint32_t fx;  // X 轴力（uint32）
    uint32_t fy;  // Y 轴力（uint32）
    uint32_t fz;  // Z 轴力（uint32）
} CForce3D;

typedef struct {
    uint32_t module;   // 模块编号
    CForce3D force;    // 三维力
} CCombinedForce;

typedef struct {
    uint64_t timestamp_us;    // 时间戳（微秒）
    uint32_t sequence;        // 序列号
    CCombinedForce combined_force;  // 组合力数据
} CFingerSample;
```

---

## FFI 接口一览

| 函数 | 说明 |
|------|------|
| `eskin_version()` | 获取 SDK 版本 |
| `eskin_open(path, config)` | 打开串口设备，返回 handle |
| `eskin_close(handle)` | 关闭设备，释放 handle |
| `eskin_read_register(handle, addr, length, buf, buf_len, actual_len)` | 读寄存器原始字节 |
| `eskin_write_register(handle, addr, data, data_len, return_count)` | 写寄存器原始字节 |
| `eskin_read_hdw_version(handle, buf, buf_len, actual_len)` | 读取硬件版本号 |
| `eskin_read_matrix_row(handle, out)` | 读取矩阵行数 |
| `eskin_read_matrix_col(handle, out)` | 读取矩阵列数 |
| `eskin_read_device_config1(handle, out)` | 读取设备配置寄存器1 |
| `eskin_read_device_config2(handle, out)` | 读取设备配置寄存器2 |
| `eskin_write_device_config1(handle, enable, return_count)` | 写入设备配置寄存器1 |
| `eskin_write_device_config2(handle, enable, return_count)` | 写入设备配置寄存器2 |
| `eskin_write_matrix_row(handle, row, return_count)` | 写入矩阵行数 |
| `eskin_write_matrix_col(handle, col, return_count)` | 写入矩阵列数 |
| `eskin_start_stream(handle)` | 启动流式采集 |
| `eskin_stop_stream(handle)` | 停止流式采集 |
| `eskin_read_sample(handle, timeout_ms, out)` | 读取一个采样数据 |
| `eskin_get_mode(handle, out)` | 查询当前设备模式（0=Command, 1=Streaming） |

---

## 协议格式

### Request 帧

```text
[start:2B] [data_len:2B LE] [dev_addr:1B] [reserved:1B] [func:1B]
[start_addr:4B LE] [read/write_len:2B LE] [payload:NB] [crc:1B]

start = [0x55, 0xAA]
func: READ=0xFB, WRITE=0x79
```

### Response 帧

```text
[start:2B] [data_len:2B LE] [dev_addr:1B] [reserved:1B] [func:1B]
[start_addr:4B LE] [read/write_len:2B LE] [payload:NB] [status:1B] [crc:1B]

start = [0x55, 0xAA]  (注: 0xAA55 LE)
func: RESPONSE_READ=0xFF, RESPONSE_WRITE=0xF9
status: 0x00=Success
```

---

## 错误码

| 名称 | 值 | 说明 |
|------|----|------|
| `Success` | 0 | 成功 |
| `InvalidPointer` | 1 | 空指针 |
| `DeviceNotFound` | 2 | 设备未找到 |
| `DeviceAlreadyOpen` | 3 | 设备已打开 |
| `NotInitialized` | 4 | 未初始化 |
| `AlreadyStreaming` | 5 | 已在采集中 |
| `NotStreaming` | 6 | 未在采集 |
| `Timeout` | 9 | 读超时 |
| `ChannelClosed` | 10 | 通道断开 |
| `CrcError` | 14 | CRC 校验失败 |
| `FrameError` | 15 | 帧格式错误 |
| `DeviceError` | 17 | 设备返回错误 |

---

## 测试

```bash
cargo test                   # 全部测试
cargo test protocol::tests   # 仅协议层
```

---

## 架构

```text
User / C / Python / ROS2
        ↓ FFI
   DeviceWrapper (handle)
        ↓
   EskinDeviceInner
        ↓ read_register / write_register / start_stream / read_sample
   EskinProtocolCodec (encode/decode + CRC8)
        ↓
   SerialTransport (write/read bytes)
        ↓
   Hardware (UART)
```

详细设计见 `docs/PROGRESS.md` 和 `docs/ARCHITECTURE.md`。

---

## 依赖

- `serialport` — 串口访问（需要 `pkg-config` + `libudev-dev`）
- `crc` — CRC-8/X25 校验
- `crossbeam-channel` — 高性能线程间通信
- `chrono` — 时间戳
- `serde` / `serde_json` — 序列化（可选）