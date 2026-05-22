#!/usr/bin/env python3
"""
drone_env.py
------------
Task B: Random Target Navigation
Wraps the NSYSU Drone simulation environment into a Gymnasium-compatible Env.

Architecture Modifications:
    1. Observation Space: Reduced from 13D (Allo-centric) to 6D (Ego-centric) using Yaw rotation.
    2. Reward Shaping: Introduced outer deceleration penalty and 1.5s stable hovering criteria.
    3. Parameters: Adopted from AirPilot (Zhang et al., 2024) and Shen & Huang (2024).
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv

import gymnasium as gym
from gymnasium import spaces


class DroneROSInterface(Node):
    """
    ROS 2 Interface Layer.
    Wraps ROS 2 publisher/subscriber/service clients for the Gym environment.
    """
    def __init__(self):
        super().__init__('rl_drone_interface')

        self.current_pose  = np.zeros(3, dtype=np.float32)
        self.current_vel   = np.zeros(3, dtype=np.float32)
        self.current_quat  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.pose_received = False

        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.soft_reset_pub = self.create_publisher(Empty, '/simple_drone/reset', 10)

        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        self.pose_sub = self.create_subscription(Pose, '/simple_drone/gt_pose', self._pose_cb, 10)
        self.vel_sub  = self.create_subscription(Twist, '/simple_drone/gt_vel', self._vel_cb, 10)

        self.get_logger().info('DroneROSInterface initialized')

    def _pose_cb(self, msg: Pose):
        self.current_pose = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        self.current_quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        self.current_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)

    def send_velocity(self, vx: float, vy: float, vz: float):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        msg.angular.x, msg.angular.y, msg.angular.z = 0.0, 0.0, 0.0
        self.cmd_vel_pub.publish(msg)

    def reset_world(self) -> bool:
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_world service not available, skipping hard reset')
            return False
        req = EmptySrv.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result() is not None

    def takeoff(self):
        self.takeoff_pub.publish(Empty())


class DroneGymEnv(gym.Env):
    """
    Task B: Random Target Navigation Environment.
    """
    MAX_SPEED = 1.0
    ARRIVE_DIST = 0.5  # 設定為 0.5m 進入懸停判定區 (Paper 2 標準)
    MAX_STEPS = 400

    BOUNDARY_XY = 4.0
    BOUNDARY_Z_MAX = 5.0
    BOUNDARY_Z_MIN = -1.0

    # 擴大生成範圍以確保良好的泛化能力 (Domain Randomization)
    TARGET_LOW  = np.array([-2.0, -2.0, 1.0], dtype=np.float32)
    TARGET_HIGH = np.array([ 2.0,  2.0, 3.0], dtype=np.float32)      

    MIN_HOVER_Z = 0.8

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # Action Space: [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-self.MAX_SPEED, high=self.MAX_SPEED, shape=(3,), dtype=np.float32
        )

        # Observation Space: [dx_b, dy_b, dz_w, vx, vy, vz] (6維本體視角)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(6,), dtype=np.float32
        )

        self.target = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_dist = None
        self.success_steps = 0  # 紀錄連續懸停的步數

        self.ep_components = {
            'progress': 0.0, 'arrive': 0.0, 'time': 0.0, 
            'boundary': 0.0, 'hover': 0.0, 'decel': 0.0
        }

    def _get_tilt_angle(self) -> float:
        """計算機身 Z 軸與世界 Z 軸的夾角 (弧度)"""
        x, y, z, w = self.ros.current_quat
        z_z = 1.0 - 2.0 * (x**2 + y**2)
        z_z = np.clip(z_z, -1.0, 1.0)
        return float(np.arccos(z_z))

    def _get_yaw(self) -> float:
        """從四元數提取 Yaw 角 (偏航角)"""
        x, y, z, w = self.ros.current_quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.success_steps = 0

        self.ros.reset_world()
        for _ in range(20):
            rclpy.spin_once(self.ros, timeout_sec=0.1)

        self.ros.takeoff()
        for _ in range(100):
            self.ros.send_velocity(0.0, 0.0, 0.0)
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            if self.ros.current_pose[2] > self.MIN_HOVER_Z:
                vel_norm = float(np.linalg.norm(self.ros.current_vel))
                if vel_norm < 0.2:
                    break

        self.target = self.np_random.uniform(low=self.TARGET_LOW, high=self.TARGET_HIGH).astype(np.float32)
        rclpy.spin_once(self.ros, timeout_sec=0.1)

        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0

        for key in self.ep_components.keys():
            self.ep_components[key] = 0.0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        real_velocity = action * 0.8

        self.ros.send_velocity(*real_velocity)
        rclpy.spin_once(self.ros, timeout_sec=0.1)
        self.step_count += 1

        obs = self._get_obs()
        pos = self.ros.current_pose.copy()

        reward, terminated = self._compute_reward(action, pos)
        truncated = (self.step_count >= self.MAX_STEPS)

        # 超時懲罰
        if truncated and not terminated:
            r_timeout = -50.0  
            reward += r_timeout
            self.ep_components['time'] += r_timeout

        info = {}
        if terminated or truncated:
            info['ep_components'] = self.ep_components.copy()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        terminated = False
        curr_dist  = float(np.linalg.norm(pos - self.target))
        if np.isnan(curr_dist):
            curr_dist = 10.0

        vel_norm = float(np.linalg.norm(self.ros.current_vel))

        # 1. 基礎時間與距離獎勵
        r_time = -0.1
        r_progress = 10.0 * (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        # 2. 外圍減速網 (Braking Coach)
        r_decel = 0.0
        if curr_dist < 4.5:
            r_decel = -2.0 * vel_norm * (4.5 - curr_dist)

        # 3. 進圈強制煞停與懸停紅包 (Stable Hovering)
        # 依據 Paper 2 (AirPilot) 的穩定懸停判定邏輯 ：
        # 論文定義懸停穩定需維持 50 steps (0.04s/step = 2.0s)。
        # 本環境步長為 0.1s/step，故轉換為 20 steps (2.0s) 以維持相同判定嚴格度。
        REQUIRED_HOVER_STEPS = 20
        r_hover = 0.0
        r_arrive = 0.0
        
        if curr_dist < self.ARRIVE_DIST:
            self.success_steps += 1
            # 懸停獎勵設計：結合距離對齊與速度懲罰 (Paper 2: 減速以達成精確懸停) [cite: 967, 986]
            r_hover = (5.0 * self.success_steps) + 10.0 * (self.ARRIVE_DIST - curr_dist) - (8.0 * vel_norm)
            
            # 懸停達標時間 (依據 Paper 2: 穩定懸停判定) 
            if self.success_steps >= self.REQUIRED_HOVER_STEPS:
                r_arrive = 100.0  # 依據 Paper 3: 到達目標獎勵 
                terminated = True
        else:
            self.success_steps = 0

        # 4. 邊界與姿態墜機保護
        r_boundary = 0.0
        tilt_angle = self._get_tilt_angle()
        is_crashed = tilt_angle > (np.pi / 4.0)
        
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )
        
        if out_of_bounds or is_crashed:
            r_boundary = -100.0
            terminated = True

        self.ep_components['progress'] += r_progress
        self.ep_components['arrive']   += r_arrive
        self.ep_components['time']     += r_time
        self.ep_components['decel']    += r_decel
        self.ep_components['hover']    += r_hover
        self.ep_components['boundary'] += r_boundary

        reward = r_time + r_progress + r_decel + r_hover + r_arrive + r_boundary
        return reward, terminated

    def _get_obs(self) -> np.ndarray:
        """
        降維至 6D 本體視角觀測向量: 
        [dx_b, dy_b, dz_w, vx, vy, vz]
        利用 Yaw 角旋轉矩陣，讓神經網路專注於第一人稱的相對特徵。
        """
        pos = self.ros.current_pose
        vel = self.ros.current_vel
        target = self.target
        yaw = self._get_yaw()
        
        dx_w = target[0] - pos[0]
        dy_w = target[1] - pos[1]
        dz_w = target[2] - pos[2]

        # 轉換為本體座標系 (Ego-centric)
        dx_b = dx_w * np.cos(yaw) + dy_w * np.sin(yaw)
        dy_b = -dx_w * np.sin(yaw) + dy_w * np.cos(yaw)

        # 尺度正規化處理
        rel_pos_scaled = np.array([dx_b / 5.0, dy_b / 5.0, dz_w / 5.0], dtype=np.float32)
        vels_scaled = vel / 2.0
        
        obs = np.concatenate([rel_pos_scaled, vels_scaled]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs

    def get_logger(self):
        return self.ros.get_logger()