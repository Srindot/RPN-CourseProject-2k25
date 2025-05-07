#include <random>
#include <string>
#include <memory>
#include <algorithm>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_core/controller.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <eigen3/Eigen/Dense>

using nav2_util::declare_parameter_if_not_declared;

namespace mppic
{

class mppic : public nav2_core::Controller
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
  {
    node_ = parent;
    auto node = node_.lock();
    name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;

    declare_parameter_if_not_declared(node, name + ".horizon_steps", rclcpp::ParameterValue(10));
    declare_parameter_if_not_declared(node, name + ".horizon_time", rclcpp::ParameterValue(1.0));
    declare_parameter_if_not_declared(node, name + ".N", rclcpp::ParameterValue(100));
    declare_parameter_if_not_declared(node, name + ".lambda", rclcpp::ParameterValue(1.0));
    declare_parameter_if_not_declared(node, name + ".sigma", rclcpp::ParameterValue(0.2));
    declare_parameter_if_not_declared(node, name + ".wheel_radius", rclcpp::ParameterValue(0.1));
    declare_parameter_if_not_declared(node, name + ".wheel_base", rclcpp::ParameterValue(0.5));
    declare_parameter_if_not_declared(node, name + ".speed_limit", rclcpp::ParameterValue(0.5));

    node->get_parameter(name + ".horizon_steps", horizon_steps_);
    node->get_parameter(name + ".horizon_time", horizon_time_);
    node->get_parameter(name + ".N", N_);
    node->get_parameter(name + ".lambda", lambda_);
    node->get_parameter(name + ".sigma", sigma_);
    node->get_parameter(name + ".wheel_radius", wheel_radius_);
    node->get_parameter(name + ".wheel_base", wheel_base_);
    node->get_parameter(name + ".speed_limit", speed_limit_);

    dt_ = horizon_time_ / horizon_steps_;

    Q_ = Eigen::Matrix3d::Identity();
    Q_(2,2) = 0.001;  // Penalize yaw error more
    R_ = Eigen::Matrix2d::Identity() * 0.01;
    P1_ = Eigen::Matrix3d::Identity();

    initial_action_ = Eigen::MatrixXd::Zero(2, horizon_steps_);
    rng_.seed(std::random_device()());
    logger_ = node->get_logger();
    clock_ = node->get_clock();
  }

  void activate() override
  {
    RCLCPP_INFO(logger_, "mppic controller activated");
  }

  void deactivate() override
  {
    RCLCPP_INFO(logger_, "mppic controller deactivated");
  }

  void cleanup() override
  {
    RCLCPP_INFO(logger_, "mppic controller cleaned up");
  }

  void setPlan(const nav_msgs::msg::Path & path) override
  {
    if (!path.poses.empty()) {
      global_plan_ = path;
      goal_pose_ = path.poses.back();
    
    initial_action_.setConstant(0.3); 
    }
  }
    
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override
  {
    (void) velocity;
    (void) goal_checker;

    Eigen::Vector3d state(
      pose.pose.position.x,
      pose.pose.position.y,
      tf2::getYaw(pose.pose.orientation));

    Eigen::Vector3d goal(
      goal_pose_.pose.position.x,
      goal_pose_.pose.position.y,
      tf2::getYaw(goal_pose_.pose.orientation));

   RCLCPP_INFO(logger_, "State: %.2f, %.2f, %.2f", state(0), state(1), state(2));
   RCLCPP_INFO(logger_, "Goal: %.2f, %.2f, %.2f", goal(0), goal(1), goal(2));
  
   Eigen::MatrixXd new_action = runMPPI(state, goal);

   double v = wheel_radius_ / 2.0 * (new_action(0, 0) + new_action(1, 0));
   double w = wheel_radius_ / wheel_base_ * (new_action(0, 0) - new_action(1, 0));
    

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.frame_id = pose.header.frame_id;
    cmd.header.stamp = clock_->now();
    cmd.twist.linear.x = std::clamp(v, -speed_limit_, speed_limit_);
    cmd.twist.angular.z = std::clamp(w, -speed_limit_, speed_limit_);
    
    RCLCPP_INFO(logger_, "Publishing velocity: linear=%.2f angular=%.2f", cmd.twist.linear.x, cmd.twist.angular.z);

    return cmd;
  }

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override
  {
    speed_limit_ = percentage ? speed_limit * speed_limit_ : speed_limit;
  }

  bool isGoalReached(const geometry_msgs::msg::PoseStamped & pose,
                     const geometry_msgs::msg::Twist & velocity)
  {
    (void) velocity;
    return nav2_util::geometry_utils::euclidean_distance(pose.pose.position, goal_pose_.pose.position) < 0.2;
  }

private:
    Eigen::MatrixXd runMPPI(const Eigen::Vector3d & state, const Eigen::Vector3d & goal)
    {
      std::normal_distribution<double> noise(0.0, sigma_);

      std::vector<Eigen::MatrixXd> eps_list(horizon_steps_);
      std::vector<Eigen::VectorXd> cost_list(horizon_steps_ + 1);

      Eigen::MatrixXd temp_state = state.replicate(1, N_).transpose();

      for (int t = 0; t < horizon_steps_; ++t) {
        eps_list[t] = Eigen::MatrixXd::NullaryExpr(N_, 2, [&]() { return noise(rng_); });
        Eigen::MatrixXd control = initial_action_.col(t).replicate(1, N_).transpose() + eps_list[t];
        cost_list[t] = computeCost(temp_state, goal, control, eps_list[t]);
        temp_state = step(temp_state, control);
      }

      cost_list[horizon_steps_] = terminalCost(temp_state, goal);

      for (int t = horizon_steps_ - 1; t >= 0; --t) {
        cost_list[t] += cost_list[t + 1];

        // Numerically stable softmax
        double min_cost = cost_list[t].minCoeff();
        Eigen::VectorXd weights = (-1.0 / lambda_ * (cost_list[t] - Eigen::VectorXd::Constant(N_, min_cost))).array().exp();
        weights /= weights.sum();

        Eigen::RowVector2d delta = weights.transpose() * eps_list[t];

        // Clamp updated action to avoid unbounded growth
        initial_action_.col(t) = (initial_action_.col(t) + delta.transpose()).cwiseMax(-1.0).cwiseMin(1.0);
      }

      return initial_action_;
    }


    Eigen::MatrixXd step(const Eigen::MatrixXd & x, const Eigen::MatrixXd & u)
    {
      Eigen::MatrixXd out = x;
      for (int i = 0; i < x.rows(); ++i) {
        auto dynamics = [&](const Eigen::Vector3d & state, const Eigen::Vector2d & control) -> Eigen::Vector3d {
          double th = state(2);
          double vl = control(0);
          double vr = control(1);
          double v = wheel_radius_ / 2.0 * (vl + vr);
          double w = wheel_radius_ / wheel_base_ * (vl - vr);
          return Eigen::Vector3d(
            v * std::cos(th),
            v * std::sin(th),
            w
          );
        };

        Eigen::Vector3d xi = x.row(i).transpose();
        Eigen::Vector2d ui = u.row(i).transpose();

        Eigen::Vector3d k1 = dynamics(xi, ui);
        Eigen::Vector3d k2 = dynamics(xi + 0.5 * dt_ * k1, ui);
        Eigen::Vector3d k3 = dynamics(xi + 0.5 * dt_ * k2, ui);
        Eigen::Vector3d k4 = dynamics(xi + dt_ * k3, ui);

        Eigen::Vector3d next = xi + dt_ / 6.0 * (k1 + 2*k2 + 2*k3 + k4);

        out(i, 0) = next(0);
        out(i, 1) = next(1);
        out(i, 2) = next(2);
      }

      return out;
    }

  Eigen::VectorXd computeCost(
    const Eigen::MatrixXd & x,
    const Eigen::Vector3d & goal,
    const Eigen::MatrixXd & u,
    const Eigen::MatrixXd & eps)
  {
    Eigen::VectorXd cost(N_);
    for (int i = 0; i < N_; ++i) {
      Eigen::Vector3d dx = x.row(i).transpose() - goal;
    double state_cost = dx.transpose() * Q_ * dx;
    double control_cost = u.row(i) * R_ * u.row(i).transpose();
    double noise_cost = lambda_ * sigma_ * eps.row(i).squaredNorm();
    cost(i) = state_cost + control_cost + noise_cost;
    }
    return cost;
  }

  Eigen::VectorXd terminalCost(const Eigen::MatrixXd & x, const Eigen::Vector3d & goal)
  {
    Eigen::VectorXd cost(N_);
    for (int i = 0; i < N_; ++i) {
      Eigen::Vector3d dx = x.row(i).transpose() - goal;
      cost(i) = dx.transpose() * P1_ * dx;
    }
    return cost;
  }

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Logger logger_ {rclcpp::get_logger("mppic")};
  rclcpp::Clock::SharedPtr clock_;

  std::string name_;
  nav_msgs::msg::Path global_plan_;
  geometry_msgs::msg::PoseStamped goal_pose_;

  int horizon_steps_, N_;
  double horizon_time_, dt_, lambda_, sigma_;
  double wheel_radius_, wheel_base_, speed_limit_;

  Eigen::MatrixXd initial_action_;
  Eigen::Matrix3d Q_, P1_;
  Eigen::Matrix2d R_;

  std::mt19937 rng_;
};

}  // namespace mppic

PLUGINLIB_EXPORT_CLASS(mppic::mppic, nav2_core::Controller)

