#include "core.hpp"
#include <iostream>
#include <thread>

using namespace pi05;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
json fixture() {
  // Synthetic test limits; never a physical-robot calibration.
  return {{"robot_model", "panda"}, {"action_contract_id", "test-only"},
    {"joint_lower", Joints{-2,-2,-2,-2,-2,-2,-2}}, {"joint_upper", Joints{2,2,2,2,2,2,2}},
    {"joint_delta_scale", Joints{.1,.1,.1,.1,.1,.1,.1}}, {"max_velocity", Joints{.5,.5,.5,.5,.5,.5,.5}},
    {"max_acceleration", Joints{1,1,1,1,1,1,1}}, {"max_jerk", Joints{10,10,10,10,10,10,10}},
    {"joint_impedance", Joints{100,100,100,100,100,100,100}}, {"max_tracking_error", Joints{.1,.1,.1,.1,.1,.1,.1}},
    {"collision_torque", Joints{1,1,1,1,1,1,1}}, {"collision_force", std::array<double,6>{1,1,1,1,1,1}},
    {"command_timeout_s", .3}, {"state_timeout_s", .05}, {"max_state_sequence_lag", 50},
    {"gripper_max_width", .08}, {"gripper_speed", .01}, {"gripper_force", 1},
    {"gripper_command_deadband", .001}, {"gripper_grasp_epsilon_inner", .005}, {"gripper_grasp_epsilon_outer", .005}};
}
int main() {
  try {
    const RobotiqCalibration aperture(.085, 12, 240);
    require(aperture.position(0) == 240, "closed aperture direction");
    require(aperture.position(.085) == 12, "open aperture direction");
    require(std::abs(aperture.width(aperture.position(.0425)) - .0425) < .0004,
            "calibrated aperture round trip");
    const Config cfg(fixture());
    Motion motion(cfg, Joints{});
    motion.target(Joints{.1,.1,.1,.1,.1,.1,.1});
    double last_acc = 0;
    for (int i = 0; i < 300; ++i) {
      motion.update();
      require(std::abs(motion.velocity()[0]) <= .500001, "velocity bound");
      require(std::abs(motion.acceleration()[0]) <= 1.000001, "acceleration bound");
      require(std::abs(motion.acceleration()[0] - last_acc) <= .010001, "jerk bound");
      last_acc = motion.acceleration()[0];
    }
    motion.stop();
    bool finished = false;
    for (int i = 0; i < 2000; ++i) {
      finished = motion.update();
      require(std::abs(motion.acceleration()[0] - last_acc) <= .010001, "stop jerk discontinuity");
      last_acc = motion.acceleration()[0];
      if (finished) break;
    }
    require(finished && std::abs(motion.velocity()[0]) < 1e-8, "stop did not brake to zero");
    bool rejected = false;
    try { Motion invalid(cfg, Joints{3,0,0,0,0,0,0}); } catch (...) { rejected = true; }
    require(rejected, "initial joint limit not enforced");
    SpscQueue<int, 32> queue;
    std::thread producer([&] { for (int i = 0; i < 100000; ++i) while (!queue.push(i)) std::this_thread::yield(); });
    for (int i = 0; i < 100000; ++i) { int value; while (!queue.pop(value)) std::this_thread::yield(); require(value == i, "mailbox ordering"); }
    producer.join();
    std::cout << "PASS: trajectory bounds, continuous stop, limits, 100000 ordered concurrent messages\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
