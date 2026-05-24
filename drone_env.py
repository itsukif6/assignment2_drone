#!/usr/bin/env python3
"""
drone_env.py
------------
Task B: 隨機目標導航 + 懸停確認
- 觀測空間: 6 維 [相對位置(本體系) x3, 速度 x3]
- Curriculum: ARRIVE_DIST 由外部 Callback 動態調降
- 重置: land -> reset_world -> takeoff
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty, Bool
from std_srvs.srv import Empty as EmptySrv
import gymnasium as gym
from gymnasium import spaces


# ================================================================
# Part 1: ROS 2 介面層
# ================================================================
class DroneROSInterface(Node):
    def __init__(self):
        super().__init__(
            'drone_rl_node',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )
        # Publishers
        self.pub_cmd     = self.create_publisher(Twist,  '/simple_drone/cmd_vel',       10)
        self.pub_takeoff = self.create_publisher(Empty,  '/simple_drone/takeoff',        10)
        self.pub_land    = self.create_publisher(Empty,  '/simple_drone/land',           10)
        self.pub_reset   = self.create_publisher(Empty,  '/simple_drone/reset',          10)
        self.pub_velmode = self.create_publisher(Bool,   '/simple_drone/dronevel_mode',  10)

        # Service client
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        # Subscribers
        self.create_subscription(Pose,  '/simple_drone/gt_pose', self._pose_cb, 10)
        self.create_subscription(Twist, '/simple_drone/gt_vel',  self._vel_cb,  10)

        # State
        self.current_pose  = np.zeros(3, dtype=np.float32)
        self.current_vel   = np.zeros(3, dtype=np.float32)
        self.current_yaw   = 0.0
        self.pose_received = False

    # ---- callbacks ----
    def _pose_cb(self, msg: Pose):
        self.current_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z], dtype=np.float32
        )
        q = msg.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        self.current_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32
        )

    # ---- commands ----
    def send_velocity(self, vx: float, vy: float, vz: float):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        self.pub_cmd.publish(msg)

    def _spin_seconds(self, sec: float):
        """spin ROS callbacks for approximately `sec` wall-clock seconds."""
        deadline = time.time() + sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)

    def reset_episode(self):
        """
        完整 episode 重置流程:
          1. 停速 + 降落，等待觸地
          2. reset_world (物理歸零)
          3. 起飛，最多重試 5 次
          4. 啟用速度模式，等待懸停穩定
        """
        # --- 1. 停速 + 降落 ---
        self.send_velocity(0.0, 0.0, 0.0)
        self.pub_land.publish(Empty())
        deadline = time.time() + 3.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.current_pose[2] < 0.15:
                break

        # --- 2. 物理世界重置 ---
        if self.reset_world_client.wait_for_service(timeout_sec=2.0):
            future = self.reset_world_client.call_async(EmptySrv.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self._spin_seconds(0.3)

        # --- 3. 起飛 (最多重試 5 次) ---
        for _ in range(5):
            self.pub_takeoff.publish(Empty())
            t0 = self.get_clock().now().nanoseconds
            while self.get_clock().now().nanoseconds - t0 < 5e8:   # 等 0.5 s
                rclpy.spin_once(self, timeout_sec=0.01)
            if self.current_pose[2] > 0.2:
                break

        # --- 4. 速度模式 + 等待穩定懸停 ---
        vm = Bool(); vm.data = True
        self.pub_velmode.publish(vm)
        self.pub_reset.publish(Empty())

        deadline = time.time() + 3.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            in_place = (
                abs(self.current_pose[0]) < 0.5 and
                abs(self.current_pose[1]) < 0.5 and
                self.current_pose[2] > 0.4
            )
            if in_place and np.linalg.norm(self.current_vel) < 0.4:
                break
            self.send_velocity(0.0, 0.0, 0.0)
            self.pub_reset.publish(Empty())

        self._spin_seconds(0.2)


# ================================================================
# Part 2: Gymnasium 環境
# ================================================================
class DroneGymEnv(gym.Env):
    """
    觀測 (6 維, 本體座標系):
        [dx_body, dy_body, dz_world, vx, vy, vz]
        dx/dy 正規化 ÷ 10, dz ÷ 5, vel ÷ 2  -> 大約落在 [-1, 1]

    動作 (3 維): [ax, ay, az] ∈ [-1, 1]
        實際速度 = action * MAX_SPEED

    Curriculum:
        ARRIVE_DIST 由 RewardLoggerCallback 在達標時動態調降
        預設從 1.0 開始
    """

    MAX_SPEED   = 2.0           # m/s，xy 方向
    MAX_SPEED_Z = 1.5           # m/s，z 方向
    MAX_STEPS   = 150           # 每回合最大步數

    # Curriculum 起始值，Callback 會動態修改
    ARRIVE_DIST = 1.0

    # 懸停確認門檻：必須在目標範圍內連續 N 步才算成功
    HOVER_STEPS_REQUIRED = 15

    # 邊界
    BOUNDARY_XY  = 9.8
    BOUNDARY_Z_H = 9.8
    BOUNDARY_Z_L = 0.05

    # 目標隨機範圍
    TARGET_XY_RANGE = 8.0       # ±8 m
    TARGET_Z_LOW    = 2.0
    TARGET_Z_HIGH   = 8.0

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(6,), dtype=np.float32
        )

        self.target_pos    = np.zeros(3, dtype=np.float32)
        self.prev_dist     = 0.0
        self.step_count    = 0
        self.hover_steps   = 0     # 連續在目標範圍內的步數
        self.ep_components = {
            'progress': 0.0, 'potential': 0.0, 'hover': 0.0,
            'boundary': 0.0, 'decel':     0.0, 'time':  0.0,
        }

    # ---- Gymnasium API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.reset_episode()

        # 隨機目標
        self.target_pos = np.array([
            np.random.uniform(-self.TARGET_XY_RANGE, self.TARGET_XY_RANGE),
            np.random.uniform(-self.TARGET_XY_RANGE, self.TARGET_XY_RANGE),
            np.random.uniform(self.TARGET_Z_LOW, self.TARGET_Z_HIGH),
        ], dtype=np.float32)

        self.prev_dist   = float(np.linalg.norm(self.target_pos - self.ros.current_pose))
        self.step_count  = 0
        self.hover_steps = 0

        for k in self.ep_components:
            self.ep_components[k] = 0.0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 執行動作，持續 0.1 s
        vx = float(action[0]) * self.MAX_SPEED
        vy = float(action[1]) * self.MAX_SPEED
        vz = float(action[2]) * self.MAX_SPEED_Z

        t0 = self.ros.get_clock().now().nanoseconds
        while self.ros.get_clock().now().nanoseconds - t0 < 1e8:
            self.ros.send_velocity(vx, vy, vz)
            rclpy.spin_once(self.ros, timeout_sec=0.01)

        curr_pos = self.ros.current_pose.copy()
        curr_vel = self.ros.current_vel.copy()
        dist     = float(np.linalg.norm(self.target_pos - curr_pos))

        reward, terminated, hover_success, dist, vel_norm = self._compute_reward(curr_pos, curr_vel, dist, action)

        truncated = self.step_count >= self.MAX_STEPS
        if truncated and not terminated:
            reward -= 30.0

        if not np.isfinite(reward):
            reward        = 0.0
            terminated    = True
            hover_success = False

        info = {}
        if terminated or truncated:
            info['ep_components'] = self.ep_components.copy()
            info['ep_dist']       = dist
            info['ep_vel']        = vel_norm
            info['ep_steps']      = self.step_count
        if hover_success:
            info['hover_success'] = True

        return self._get_obs(), reward, terminated, truncated, info

    # ---- Reward ----
    def _compute_reward(self, pos, vel, dist, action):
        terminated = False
        vel_norm   = float(np.linalg.norm(vel))

        # 1. 時間懲罰（距離越近懲罰越輕）
        r_time = -1.5 + 0.6 * math.exp(-dist / 3.0)

        # 2. 進度獎勵（靠近 +，遠離 -，不對稱係數抑制套利）
        delta      = self.prev_dist - dist
        r_progress = 35.0 * delta if delta >= 0 else 12.0 * delta
        self.prev_dist = dist

        # 3. 近距離潛力場（平滑，不突變）
        r_potential = 1.2 / (dist ** 2 + 0.15)

        # 4. 近距離強制減速懲罰
        r_decel = 0.0
        if dist < 4.0:
            r_decel = -1.8 * vel_norm * (4.0 - dist)

        # 5. 動作平滑懲罰（抑制抖振）
        r_smooth = -0.02 * float(np.sum(np.square(action)))

        # 6. 懸停確認（在目標範圍內連續停留）
        r_hover       = 0.0
        hover_success = False
        if dist < self.ARRIVE_DIST:
            self.hover_steps += 1
            r_hover = 4.0 * self.hover_steps + 8.0 * (self.ARRIVE_DIST - dist) - 6.0 * vel_norm

            if self.hover_steps >= self.HOVER_STEPS_REQUIRED:
                r_hover      += 600.0
                terminated    = True
                hover_success = True
                print(f'[Success] Hover {self.HOVER_STEPS_REQUIRED} steps | '
                      f'dist={dist:.2f} | ARRIVE={self.ARRIVE_DIST}')
        else:
            self.hover_steps = 0

        # 7. 邊界終止懲罰
        r_boundary = 0.0
        out = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_H     or
            pos[2] < self.BOUNDARY_Z_L
        )
        if out and not terminated:
            r_boundary = -100.0
            terminated = True
            print(f'[Fail] Out of bounds | pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})')

        reward = r_time + r_progress + r_potential + r_decel + r_smooth + r_hover + r_boundary

        # 累計本回合各分量（step() 呼叫後可從 ep_components 讀取）
        self.ep_components['time']      += r_time
        self.ep_components['progress']  += r_progress
        self.ep_components['potential'] += r_potential
        self.ep_components['decel']     += r_decel
        self.ep_components['hover']     += r_hover
        self.ep_components['boundary']  += r_boundary

        return reward, terminated, hover_success, dist, float(np.linalg.norm(vel))

    # ---- Observation ----
    def _get_obs(self):
        dx_w = self.target_pos[0] - self.ros.current_pose[0]
        dy_w = self.target_pos[1] - self.ros.current_pose[1]
        dz_w = self.target_pos[2] - self.ros.current_pose[2]

        yaw   = self.ros.current_yaw
        dx_b  =  dx_w * math.cos(yaw) + dy_w * math.sin(yaw)
        dy_b  = -dx_w * math.sin(yaw) + dy_w * math.cos(yaw)

        obs = np.array([
            dx_b / 10.0,
            dy_b / 10.0,
            dz_w / 5.0,
            self.ros.current_vel[0] / 2.0,
            self.ros.current_vel[1] / 2.0,
            self.ros.current_vel[2] / 2.0,
        ], dtype=np.float32)

        if not np.isfinite(obs).all():
            obs = np.zeros(6, dtype=np.float32)

        return np.clip(obs, -2.0, 2.0)