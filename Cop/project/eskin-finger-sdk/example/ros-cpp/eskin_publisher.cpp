#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int32.hpp"

#include "../include/eskin_ffi.h"

using namespace std::chrono_literals;

class EskinPublisher : public rclcpp::Node {
public:
  EskinPublisher(const std::string & device_path)
  : Node("eskin_publisher"), running_(true)
  {
    // 打开设备
    device_ = eskin_open(device_path.c_str(), nullptr);
    if (!device_) {
      RCLCPP_FATAL(this->get_logger(), "Failed to open device: %s", device_path.c_str());
      rclcpp::shutdown();
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Device opened: %s", device_path.c_str());

    // 启动 streaming
    auto err = eskin_start_stream(device_);
    if (err != ESkinSuccess) {
      RCLCPP_FATAL(this->get_logger(), "Failed to start stream, error: %d", (int)err);
      eskin_close(device_);
      rclcpp::shutdown();
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Streaming started");

    // 创建 publisher
    publisher_ = this->create_publisher<std_msgs::msg::UInt32>("comb_force", 10);

    // 启动独立读取线程
    read_thread_ = std::thread(&EskinPublisher::read_loop, this);

    // 10ms 定时器：从队列取数据并发布
    timer_ = this->create_wall_timer(10ms, std::bind(&EskinPublisher::publish_callback, this));

    RCLCPP_INFO(this->get_logger(), "EskinPublisher setup success");
  }

  ~EskinPublisher() {
    // 停止读取线程
    running_ = false;
    cv_.notify_all();
    if (read_thread_.joinable()) {
      read_thread_.join();
    }

    // 停止 streaming 并关闭设备
    if (device_) {
      eskin_stop_stream(device_);
      eskin_close(device_);
    }
    RCLCPP_INFO(this->get_logger(), "EskinPublisher destroyed");
  }

private:
  // 独立读取线程：持续从设备读取 sample 放入队列
  void read_loop() {
    while (running_) {
      CFingerSample sample;
      memset(&sample, 0, sizeof(sample));
      EskinSdkErrorCode err = eskin_read_sample(device_, 50, &sample);
      if (err == ESkinSuccess) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        sample_queue_.push(sample);
        // 限制队列大小，防止堆积
        while (sample_queue_.size() > 100) {
          sample_queue_.pop();
        }
      } else if (err != ESkinTimeout) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "eskin_read_sample error: %d", (int)err);
      }
    }
  }

  // 定时器回调：从队列取数据并发布到 ROS topic
  void publish_callback() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!sample_queue_.empty()) {
      const auto & sample = sample_queue_.front();
      auto msg = std_msgs::msg::UInt32();
      // 使用 combined_force 中的 fz（法向力）作为发布值
      msg.data = static_cast<uint32_t>(sample.combined_force.force.fz);
      publisher_->publish(msg);
      sample_queue_.pop();
    }
  }

  bool running_;
  EskinDeviceHandle device_;
  std::thread read_thread_;
  std::mutex queue_mutex_;
  std::queue<CFingerSample> sample_queue_;
  std::condition_variable cv_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr publisher_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);

  // 设备路径可通过命令行参数传入，默认 /dev/ttyUSB0
  std::string device_path = "/dev/ttyUSB0";
  if (argc > 1) {
    device_path = argv[1];
  }

  auto node = std::make_shared<EskinPublisher>(device_path);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}