# Changelog

## v0.1.0 (2026-05-07)

E-Skin 手指力传感器 SDK 首个正式版本，支持 Rust / C/C++ / Python 多语言调用。

### ✨ 核心功能

- **串口通信**：基于 `serialport` 的串口传输层，支持 UART 连接
- **协议编解码**：完整的请求/响应帧编解码，内置 CRC-8/X25 校验
- **寄存器读写**：底层寄存器原始字节读写接口
- **设备配置管理**：硬件版本读取、矩阵行列尺寸读写、设备配置寄存器读写
- **流式采集**：基于 `crossbeam-channel` 的高性能线程间数据传输
- **设备管理**：设备打开/关闭状态机，支持 Open / Streaming / Error 状态

### 🔌 FFI 接口

提供完整的 C FFI 导出，支持 C/C++ 和 Python 调用：

| 接口 | 说明 |
|------|------|
| `eskin_version` | SDK 版本 |
| `eskin_open` / `eskin_close` | 设备打开/关闭 |
| `eskin_read_register` / `eskin_write_register` | 寄存器原始读写 |
| `eskin_read_hdw_version` | 硬件版本号 |
| `eskin_read_matrix_row` / `col` | 矩阵行列读取 |
| `eskin_write_matrix_row` / `col` | 矩阵行列写入 |
| `eskin_read_device_config1` / `2` | 设备配置寄存器读取 |
| `eskin_write_device_config1` / `2` | 设备配置寄存器写入 |

### 📦 示例代码

- **C++ 示例**：`example/cpp/main.cpp`
- **Python 示例**：`example/python/example.py` + `example/python/eskin_ffi.py`

### 🛠 构建

```bash
# 安装依赖（Ubuntu）
sudo apt install pkg-config libudev-dev

# 构建
cargo build --release
# 输出: target/release/libeskin_finger_sdk.so
```

### 📋 协议

- 请求帧：`[0x55, 0xAA] + data_len(LE) + dev_addr + func + addr(LE) + len(LE) + payload + crc8`
- 响应帧：同上格式 + status 字段，`0x00` 表示成功
- 支持功能码：`READ=0xFB`、`WRITE=0x79`、`RESPONSE_READ=0xFF`、`RESPONSE_WRITE=0xF9`

### ⚠️ 已知限制

- 当前仅支持串口传输（UART）
- 版本号 `0.1.0`，API 可能在后续版本中调整