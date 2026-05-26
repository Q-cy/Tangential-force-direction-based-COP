#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "../../include/eskin_ffi.h"

using namespace std::chrono_literals;

// ── Command 模式：读取设备信息 ─────────────────────────────

static void demo_command_mode(EskinDeviceHandle device) {
    printf("=== Command Mode ===\n");

    // 硬件版本
    char hw_buf[64] = {};
    uint32_t hw_len = 0;
    if (eskin_read_hdw_version(device, hw_buf, sizeof(hw_buf), &hw_len) == ESkinSuccess) {
        printf("Hardware version: %.*s\n", (int)hw_len, hw_buf);
    }

    // 矩阵尺寸
    uint8_t row = 0, col = 0;
    if (eskin_read_matrix_row(device, &row) == ESkinSuccess &&
        eskin_read_matrix_col(device, &col) == ESkinSuccess) {
        printf("Matrix size: %u x %u\n", row, col);
    }

    // 设备配置
    uint8_t cfg1 = 0;
    if (eskin_read_device_config1(device, &cfg1) == ESkinSuccess) {
        printf("Device config1: 0x%02X\n", cfg1);
    }

    // 序列号（原始寄存器读取）
    uint8_t buf[256] = {};
    uint32_t actual = 0;
    if (eskin_read_register(device, 0x1C00, 168, buf, sizeof(buf), &actual) == ESkinSuccess) {
        printf("Serial number (raw): ");
        for (uint32_t i = 0; i < actual; i++) {
            printf("%02X", buf[i]);
        }
        printf("\n");
    }
}

// ── Streaming 模式：持续采集力数据 ────────────────────────

static void demo_streaming(EskinDeviceHandle device, double duration_sec = 5.0) {
    printf("\n=== Streaming Mode ===\n");

    auto err = eskin_start_stream(device);
    if (err != ESkinSuccess) {
        printf("Failed to start stream, error: %d\n", (int)err);
        return;
    }
    printf("Streaming started, will run for %.1fs ...\n", duration_sec);

    // 线程安全队列（参考 ROS publisher 的 read_loop + publish_callback 分离模式）
    std::mutex mtx;
    std::queue<CFingerSample> queue;
    bool running = true;

    // 读取线程：持续从设备读取 sample 放入队列
    std::thread read_thread([&]() {
        while (running) {
            CFingerSample sample;
            memset(&sample, 0, sizeof(sample));
            auto e = eskin_read_sample(device, 50, &sample);
            if (e == ESkinSuccess) {
                std::lock_guard<std::mutex> lock(mtx);
                queue.push(sample);
                while (queue.size() > 100) {
                    queue.pop();  // 防止堆积
                }
            }
            // 超时等非致命错误忽略，继续读取
        }
    });

    // 主线程：从队列取数据并打印（类似 ROS 的 publish_callback）
    auto start = std::chrono::steady_clock::now();
    int count = 0;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= duration_sec) break;

        {
            std::lock_guard<std::mutex> lock(mtx);
            while (!queue.empty()) {
                const auto &s = queue.front();
                printf("[%5u] module=%u  fx=%u  fy=%u  fz=%u\n",
                       s.sequence, s.combined_force.module,
                       s.combined_force.force.fx,
                       s.combined_force.force.fy,
                       s.combined_force.force.fz);
                queue.pop();
                count++;
            }
        }

        std::this_thread::sleep_for(5ms);
    }

    running = false;
    read_thread.join();
    eskin_stop_stream(device);
    printf("Streaming stopped. Total samples: %d\n", count);
}

// ── Main ──────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    std::string device_path = "/dev/ttyUSB0";
    if (argc > 1) {
        device_path = argv[1];
    }

    // SDK 版本
    auto ver = eskin_version();
    printf("ESkin SDK version: %u.%u.%u\n", ver.major, ver.minor, ver.patch);

    // 打开设备
    EskinDeviceHandle device = eskin_open(device_path.c_str(), nullptr);
    if (!device) {
        fprintf(stderr, "Failed to open device: %s\n", device_path.c_str());
        return 1;
    }
    printf("Device opened: %s\n", device_path.c_str());

    // Command 模式演示
    demo_command_mode(device);

    // Streaming 模式演示
    demo_streaming(device, 5.0);

    // 关闭设备
    eskin_close(device);
    printf("Device closed\n");
    return 0;
}