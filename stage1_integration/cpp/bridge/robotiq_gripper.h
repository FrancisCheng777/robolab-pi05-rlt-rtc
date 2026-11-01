// Robotiq 2F-85 / 2F-140 夹爪驱动（Modbus RTU）
// 基于 ros-industrial/robotiq Python 代码改写为 C++，不依赖 ROS
#pragma once
#include <cstdint>
#include <string>

namespace robotiq {

struct GripperStatus {
  bool activated = false;        // gACT: 夹爪已激活
  bool goto_requested = false;   // gGTO: 正在执行 goto
  uint8_t status = 0;            // gSTA: 0=reset 1=activation 3=ready
  uint8_t object_detection = 0;  // gOBJ: 0=运动中 1=外侧停止 2=内侧停止 3=夹住
  uint8_t fault = 0;             // gFLT: 错误码（0=正常）
  uint8_t position = 0;          // gPO: 实际位置 0-255（0=全开 255=全闭）
  uint8_t current = 0;           // gCU: 电流 0-255（单位 10mA）
};

class Robotiq2FGripper {
 public:
  explicit Robotiq2FGripper(const std::string& device = "/dev/ttyUSB0");
  ~Robotiq2FGripper();

  // 连接与关闭
  bool connect();
  void close();
  bool is_connected() const { return connected_; }

  // 激活（首次使用或 reset 后必须调用）
  bool activate();

  // 位置控制（position: 0-255，0=全开 255=全闭）
  // speed: 0-255（速度），force: 0-255（夹持力）
  bool move(uint8_t position, uint8_t speed = 255, uint8_t force = 128);

  // 按实际宽度控制（单位 米，对 2F-85: 0=全开 0.085=全闭）
  // 内部转换为 0-255
  bool move_width(double width_m, uint8_t speed = 255, uint8_t force = 128);

  // 读取状态
  GripperStatus get_status();

  // 等待运动完成（返回 false = 超时）
  bool wait_for_move(double timeout_s = 5.0);

  // 紧急停止
  bool stop();

  // 获取实际宽度（米）
  double get_width() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  bool connected_ = false;
  std::string device_;
  double max_width_ = 0.085;  // 2F-85 的最大开度（米）

  bool send_command(const uint8_t data[6]);
  bool read_status(uint8_t data[6]);
};

}  // namespace robotiq
