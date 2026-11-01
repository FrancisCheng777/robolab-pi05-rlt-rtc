// Robotiq 2F 夹爪驱动实现（Modbus RTU over libmodbus）
#include "robotiq_gripper.h"
#include <modbus/modbus.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>

namespace robotiq {

struct Robotiq2FGripper::Impl {
  modbus_t* ctx = nullptr;
};

Robotiq2FGripper::Robotiq2FGripper(const std::string& device)
    : impl_(new Impl), device_(device) {}

Robotiq2FGripper::~Robotiq2FGripper() {
  close();
  delete impl_;
}

bool Robotiq2FGripper::connect() {
  if (connected_) return true;

  impl_->ctx = modbus_new_rtu(device_.c_str(), 115200, 'N', 8, 1);
  if (!impl_->ctx) {
    std::cerr << "modbus_new_rtu 失败: " << modbus_strerror(errno) << std::endl;
    return false;
  }

  // Robotiq 的 Modbus 单元地址是 0x09
  if (modbus_set_slave(impl_->ctx, 0x09) < 0) {
    std::cerr << "modbus_set_slave 失败" << std::endl;
    modbus_free(impl_->ctx);
    impl_->ctx = nullptr;
    return false;
  }

  if (modbus_connect(impl_->ctx) < 0) {
    std::cerr << "modbus_connect 失败: " << modbus_strerror(errno) << std::endl;
    modbus_free(impl_->ctx);
    impl_->ctx = nullptr;
    return false;
  }

  // 设置超时：200 ms（和 Python 代码对齐）
  modbus_set_response_timeout(impl_->ctx, 0, 200000);  // 200 ms
  connected_ = true;
  return true;
}

void Robotiq2FGripper::close() {
  if (impl_->ctx) {
    modbus_close(impl_->ctx);
    modbus_free(impl_->ctx);
    impl_->ctx = nullptr;
  }
  connected_ = false;
}

bool Robotiq2FGripper::send_command(const uint8_t data[6]) {
  if (!connected_) return false;

  // 6 字节打包成 3 个 16-bit 寄存器
  uint16_t regs[3];
  for (int i = 0; i < 3; i++) {
    regs[i] = (data[2*i] << 8) | data[2*i + 1];
  }

  // 写入寄存器 0x03E8 (1000)
  int rc = modbus_write_registers(impl_->ctx, 0x03E8, 3, regs);
  if (rc != 3) {
    std::cerr << "modbus_write_registers 失败: " << modbus_strerror(errno) << std::endl;
    return false;
  }
  return true;
}

bool Robotiq2FGripper::read_status(uint8_t data[6]) {
  if (!connected_) return false;

  uint16_t regs[3];
  // 读取寄存器 0x07D0 (2000)
  int rc = modbus_read_registers(impl_->ctx, 0x07D0, 3, regs);
  if (rc != 3) {
    std::cerr << "modbus_read_registers 失败: " << modbus_strerror(errno) << std::endl;
    return false;
  }

  // 解包：高字节在前
  for (int i = 0; i < 3; i++) {
    data[2*i]     = (regs[i] >> 8) & 0xFF;
    data[2*i + 1] = regs[i] & 0xFF;
  }
  return true;
}

bool Robotiq2FGripper::activate() {
  if (!connected_) return false;

  // 1. 发送 reset (rACT=0)
  uint8_t cmd[6] = {0, 0, 0, 0, 0, 0};
  if (!send_command(cmd)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // 2. 发送激活 (rACT=1, rGTO=1)
  cmd[0] = 0x01 | (0x01 << 3);  // rACT=1, rGTO=1
  cmd[3] = 0;    // position=0 (全开)
  cmd[4] = 255;  // speed=max
  cmd[5] = 128;  // force=default
  if (!send_command(cmd)) return false;

  // 3. 等待 gSTA=3 (ready)
  for (int i = 0; i < 50; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto st = get_status();
    if (st.activated && st.status == 3) {
      std::cout << "Robotiq 夹爪已激活" << std::endl;
      return true;
    }
  }
  std::cerr << "激活超时" << std::endl;
  return false;
}

bool Robotiq2FGripper::move(uint8_t position, uint8_t speed, uint8_t force) {
  if (!connected_) return false;

  uint8_t cmd[6];
  cmd[0] = 0x01 | (0x01 << 3);  // rACT=1, rGTO=1
  cmd[1] = 0;
  cmd[2] = 0;
  cmd[3] = position;
  cmd[4] = speed;
  cmd[5] = force;
  return send_command(cmd);
}

bool Robotiq2FGripper::move_width(double width_m, uint8_t speed, uint8_t force) {
  // 2F-85: 0 m = position 0 (全开), 0.085 m = position 255 (全闭)
  // width 越大 => position 越小
  width_m = std::max(0.0, std::min(max_width_, width_m));
  uint8_t pos = static_cast<uint8_t>((1.0 - width_m / max_width_) * 255);
  return move(pos, speed, force);
}

GripperStatus Robotiq2FGripper::get_status() {
  GripperStatus st;
  uint8_t data[6] = {0};
  if (!read_status(data)) throw std::runtime_error("Robotiq feedback read failed");

  st.activated = (data[0] & 0x01);
  st.goto_requested = (data[0] >> 3) & 0x01;
  st.status = (data[0] >> 4) & 0x03;
  st.object_detection = (data[0] >> 6) & 0x03;
  st.fault = data[2];
  st.position = data[3];  // gPR (请求位置)
  st.position = data[4];  // gPO (实际位置，覆盖前一行）
  st.current = data[5];
  return st;
}

double Robotiq2FGripper::get_width() const {
  // 当前无法直接拿到 gPO，需在 get_status() 里缓存
  // 这里暂时返回 0（TODO: 需要在类里加个 last_status_ 缓存）
  return 0.0;
}

bool Robotiq2FGripper::wait_for_move(double timeout_s) {
  auto start = std::chrono::steady_clock::now();
  while (true) {
    auto st = get_status();
    // gOBJ != 0 表示运动结束（1=外侧 2=内侧 3=夹住）
    if (st.object_detection != 0) return true;

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed > timeout_s) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

bool Robotiq2FGripper::stop() {
  // rGTO=0 停止运动
  uint8_t cmd[6] = {0x01, 0, 0, 0, 0, 0};  // rACT=1, rGTO=0
  return send_command(cmd);
}

}  // namespace robotiq
