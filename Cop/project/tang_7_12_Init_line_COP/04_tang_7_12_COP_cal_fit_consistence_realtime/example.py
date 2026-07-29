import time
import threading
from collections import deque

from eskin_ffi import EskinDevice, CFingerSample


def demo_command_mode(dev: EskinDevice):
    """Command 模式：读取设备信息、寄存器等"""
    print("=== Command Mode ===")

    hdw_ver = dev.read_hdw_version()
    print(f"Hardware version: {hdw_ver}")

    row = dev.read_matrix_row()
    col = dev.read_matrix_col()
    print(f"Matrix size: {row} x {col}")

    cfg1 = dev.read_device_config1()
    print(f"Device config1: 0x{cfg1:02X}")

    data = dev.read_register(0x1C00, 168)
    print(f"Serial number: {data.hex().upper()}")


def demo_streaming(dev: EskinDevice, duration_sec: float = 5.0):
    """Streaming 模式：持续采集力数据（参考 ROS C++ publisher）"""
    print("\n=== Streaming Mode ===")

    # 启动流式采集
    dev.start_stream()
    print(f"Streaming started, will run for {duration_sec}s ...")

    # 线程安全的队列（参考 ROS demo 的 read_loop + publish_callback 分离模式）
    queue: deque = deque(maxlen=1014)
    raw_queue: deque = deque(maxlen=1024)
    running = True

    def read_loop():
        """独立读取线程：持续从设备读取 sample 和原始帧"""
        while running:
            try:
                sample = dev.read_sample(timeout_ms=50)
                queue.append(sample)
                raw_frame = dev.read_stream_frame(timeout_ms=1)
                print(raw_frame)
                raw_queue.append(raw_frame)
            except RuntimeError:
                # 超时等非致命错误，继续读取
                pass

    # 启动读取线程
    reader = threading.Thread(target=read_loop, daemon=True)
    reader.start()

    # 主线程：从队列取数据并打印（类似 ROS 的 publish_callback）
    start = time.monotonic()
    count = 0
    while True:
        if queue:
            sample: CFingerSample = queue.popleft()
            raw_frame = raw_queue.popleft() if raw_queue else b""
            f = sample.combined_force.force
            mod = sample.combined_force.module
            print(
                f"[{sample.sequence:5d}] "
                f"module={mod}  "
                f"fx={f.fx}  fy={f.fy}  fz={f.fz}  "
                f"raw_len={len(raw_frame)}"
            )
            count += 1
        else:
            time.sleep(0.005)

    running = False
    reader.join(timeout=1.0)
    dev.stop_stream()
    print(f"Streaming stopped. Total samples: {count}")


def main():
    ver = EskinDevice().version()
    print(f"ESkin SDK version: {ver[0]}.{ver[1]}.{ver[2]}")

    device_path = "/dev/ttyUSB0"

    with EskinDevice() as dev:
        dev.open(device_path)
        print(f"Device opened: {device_path}")

        # demo_command_mode(dev)
        demo_streaming(dev, duration_sec=5.0)

    print("Device closed")


if __name__ == "__main__":
    main()
