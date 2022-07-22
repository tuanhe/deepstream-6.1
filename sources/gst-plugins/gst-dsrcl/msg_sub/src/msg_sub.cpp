#include <functional>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "dsmsg/msg/num.hpp"                                       // CHANGE

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
       auto callback_factory = [](const std::string & a_topic_name) {
            return [a_topic_name](const dsmsg::msg::Num::SharedPtr msg) -> void {
            printf("topic_name: %s   I heard : %ld\n", a_topic_name.c_str(), msg->num);
            //RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->num);
            };
        };
        subscription_ = this->create_subscription<dsmsg::msg::Num>("metadata", 10,  callback_factory("metadata"));
  }

private:
  rclcpp::Subscription<dsmsg::msg::Num>::SharedPtr subscription_;  // CHANGE
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
