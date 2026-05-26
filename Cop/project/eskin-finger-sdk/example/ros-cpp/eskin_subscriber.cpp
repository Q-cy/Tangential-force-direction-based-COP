#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int32.hpp"

using std::placeholders::_1;

class EskinSubscriber : public rclcpp::Node {
public:
  EskinSubscriber()
  : Node("eskin_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::UInt32>(
      "comb_force", 10,
      std::bind(&EskinSubscriber::topic_callback, this, _1));
    RCLCPP_INFO(this->get_logger(), "EskinSubscriber listening on /comb_force");
  }

private:
  void topic_callback(const std_msgs::msg::UInt32::SharedPtr msg) const {
    RCLCPP_INFO(this->get_logger(), "Received force: %u", msg->data);
  }

  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EskinSubscriber>());
  rclcpp::shutdown();
  return 0;
}