#!/usr/bin/env python3
"""
drone_env.py
------------
無人機隨機目標導航 (Random Target Navigation) 環境。
將 ROS 2 模擬環境封裝為相容於 Gymnasium 的強化學習環境。
支援 Level 1 ~ 13 漸進式課程學習 (Curriculum Learning)。
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty, Bool
from std_srvs.srv import Empty as EmptySrv

import gymnasium as gym
from gymnasium import spaces

class DroneROSInterface(Node):
    """
    負責處理與 Gazebo/ROS 2 底層通訊的節點介面。
    包含感測器狀態訂閱 (Subscribe) 與控制指令發布 (Publish)。
    """
    def __init__(self):
        super().__init__('rl_drone_interface')

        self.current_pose  = np.zeros(3, dtype=np.float32)
        self.current_vel   = np.zeros(3, dtype=np.float32)
        self.current_quat  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.current_yaw   = 0.0
        self.pose_received = False

        # 定義 ROS 2 發布者 (Publisher)
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)
        self.vel_mode_pub = self.create_publisher(Bool, '/simple_drone/dronevel_mode', 10)
        self.soft_reset_pub = self.create_publisher(Empty, '/simple_drone/reset', 10)

        # 定義 ROS 2 服務客戶端 (Service Client) 用於硬重置世界
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        # 定義 ROS 2 訂閱者 (Subscriber) 接收真實狀態資訊 (Ground Truth)
        self.pose_sub = self.create_subscription(Pose, '/simple_drone/gt_pose', self._pose_cb, 10)
        self.vel_sub  = self.create_subscription(Twist, '/simple_drone/gt_vel',  self._vel_cb,  10)

        self.get_logger().info('DroneROSInterface 初始化完成')

    def _pose_cb(self, msg: Pose):
        """處理無人機位置與四元數姿態更新"""
        self.current_pose = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        q = msg.orientation
        self.current_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        
        # 將四元數轉換為尤拉角中的偏航角 (Yaw)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        """處理無人機線性速度更新"""
        self.current_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)

    def send_velocity(self, vx: float, vy: float, vz: float):
        """封裝 Twist 訊息並發布速度指令"""
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.linear.z  = float(vz)
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def reset_world(self) -> bool:
        """呼叫 Gazebo 服務進行模擬器物理狀態的完全重置"""
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_world 服務無法使用，跳過硬重置')
            return False
        req    = EmptySrv.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result() is not None

    def takeoff(self):
        """發送起飛指令"""
        self.takeoff_pub.publish(Empty())

class DroneGymEnv(gym.Env):
    """
    符合 Gymnasium 標準的無人機強化學習環境。
    整合獎勵機制、狀態空間與動作空間的定義。
    """
    MAX_SPEED = 1.0
    ARRIVE_DIST = 1.0  
    MAX_STEPS = 600
    BOUNDARY_XY    = 3.0
    BOUNDARY_Z_MAX = 6.5 # 放寬最大高度限制，配合 Level 13 的 5.5m 目標
    BOUNDARY_Z_MIN = 0.0

    # 定義 13 個 Curriculum Levels (課程學習難度階層)，參數採線性漸進式收緊
    CURRICULUM_LEVELS = {
        1:  {'target_low': np.array([-1.0, -1.0, 1.5], dtype=np.float32),  'target_high': np.array([ 1.0,  1.0, 2.5], dtype=np.float32),  'boundary_xy': 3.0,  'arrive_dist': 1.50, 'hover_steps': 3,  'hover_max_spd': 0.80},
        2:  {'target_low': np.array([-1.75, -1.75, 1.5], dtype=np.float32),'target_high': np.array([ 1.75,  1.75, 2.75], dtype=np.float32),'boundary_xy': 3.5,  'arrive_dist': 1.35, 'hover_steps': 3,  'hover_max_spd': 0.75},
        3:  {'target_low': np.array([-2.5, -2.5, 1.5], dtype=np.float32),  'target_high': np.array([ 2.5,  2.5, 3.0], dtype=np.float32),  'boundary_xy': 4.0,  'arrive_dist': 1.20, 'hover_steps': 5,  'hover_max_spd': 0.60},
        4:  {'target_low': np.array([-3.25, -3.25, 1.5], dtype=np.float32),'target_high': np.array([ 3.25,  3.25, 3.25], dtype=np.float32),'boundary_xy': 4.5,  'arrive_dist': 1.10, 'hover_steps': 8,  'hover_max_spd': 0.45},
        5:  {'target_low': np.array([-4.0, -4.0, 1.5], dtype=np.float32),  'target_high': np.array([ 4.0,  4.0, 3.5], dtype=np.float32),  'boundary_xy': 5.0,  'arrive_dist': 1.00, 'hover_steps': 12, 'hover_max_spd': 0.35},
        6:  {'target_low': np.array([-4.75, -4.75, 1.5], dtype=np.float32),'target_high': np.array([ 4.75,  4.75, 3.75], dtype=np.float32),'boundary_xy': 5.75, 'arrive_dist': 0.95, 'hover_steps': 14, 'hover_max_spd': 0.33},
        7:  {'target_low': np.array([-5.5, -5.5, 1.5], dtype=np.float32),  'target_high': np.array([ 5.5,  5.5, 4.0], dtype=np.float32),  'boundary_xy': 6.5,  'arrive_dist': 0.90, 'hover_steps': 16, 'hover_max_spd': 0.31},
        8:  {'target_low': np.array([-6.25, -6.25, 1.5], dtype=np.float32),'target_high': np.array([ 6.25,  6.25, 4.25], dtype=np.float32),'boundary_xy': 7.25, 'arrive_dist': 0.85, 'hover_steps': 18, 'hover_max_spd': 0.29},
        9:  {'target_low': np.array([-7.0, -7.0, 1.5], dtype=np.float32),  'target_high': np.array([ 7.0,  7.0, 4.5], dtype=np.float32),  'boundary_xy': 8.0,  'arrive_dist': 0.80, 'hover_steps': 20, 'hover_max_spd': 0.27},
        10: {'target_low': np.array([-7.75, -7.75, 1.5], dtype=np.float32),'target_high': np.array([ 7.75,  7.75, 4.75], dtype=np.float32),'boundary_xy': 8.75, 'arrive_dist': 0.75, 'hover_steps': 22, 'hover_max_spd': 0.25},
        11: {'target_low': np.array([-8.5, -8.5, 1.5], dtype=np.float32),  'target_high': np.array([ 8.5,  8.5, 5.0], dtype=np.float32),  'boundary_xy': 9.5,  'arrive_dist': 0.70, 'hover_steps': 24, 'hover_max_spd': 0.23},
        12: {'target_low': np.array([-9.25, -9.25, 1.5], dtype=np.float32),'target_high': np.array([ 9.25,  9.25, 5.25], dtype=np.float32),'boundary_xy': 10.25,'arrive_dist': 0.65, 'hover_steps': 26, 'hover_max_spd': 0.21},
        13: {'target_low': np.array([-10.0, -10.0, 1.5], dtype=np.float32),'target_high': np.array([ 10.0, 10.0, 5.5], dtype=np.float32), 'boundary_xy': 11.0, 'arrive_dist': 0.60, 'hover_steps': 28, 'hover_max_spd': 0.20},
    }
    
    MAX_CURRICULUM_LEVEL = 13
    TARGET_LOW  = CURRICULUM_LEVELS[1]['target_low']
    TARGET_HIGH = CURRICULUM_LEVELS[1]['target_high']
    MIN_HOVER_Z = 0.8

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # 動作空間：連續控制，表示無人機的三軸速度 (x, y, z)
        self.action_space = spaces.Box(
            low  = -self.MAX_SPEED,
            high =  self.MAX_SPEED,
            shape=(3,),
            dtype=np.float32
        )

        # 狀態空間：包含歸一化後的相對位置誤差與無人機當前速度
        self.observation_space = spaces.Box(
            low  = -5.0,
            high =  5.0,
            shape=(6,),
            dtype=np.float32
        )

        self.HOVER_STEPS     = 3
        self.HOVER_MAX_SPEED = 0.8
        self.ARRIVE_DIST     = 1.5

        self.curriculum_level = 1
        self._apply_curriculum(1)

        self.target        = np.zeros(3, dtype=np.float32)
        self.step_count    = 0
        self.prev_dist     = None
        self.hover_steps   = 0       

        # 用於 TensorBoard 紀錄獎勵組成的字典
        self.ep_components = {
            'progress': 0.0, 'arrive': 0.0, 'action': 0.0,
            'time': 0.0, 'boundary': 0.0, 'proximity': 0.0, 'decel': 0.0
        }

    def _apply_curriculum(self, level: int):
        """根據指定的難度層級，載入環境與邊界參數"""
        cfg = self.CURRICULUM_LEVELS[level]
        self.TARGET_LOW      = cfg['target_low'].copy()
        self.TARGET_HIGH     = cfg['target_high'].copy()
        self.BOUNDARY_XY     = cfg['boundary_xy']
        self.ARRIVE_DIST     = cfg['arrive_dist']
        self.HOVER_STEPS     = cfg['hover_steps']
        self.HOVER_MAX_SPEED = cfg['hover_max_spd']

    def set_curriculum_level(self, level: int):
        """外部呼叫以提升課程難度，並輸出紀錄"""
        level = int(np.clip(level, 1, self.MAX_CURRICULUM_LEVEL))
        if level != self.curriculum_level:
            self.curriculum_level = level
            self._apply_curriculum(level)
            self.ros.get_logger().info(
                f'[Curriculum] Level -> {level} | '
                f'Target x/y: ±{self.TARGET_HIGH[0]:.1f}m, '
                f'z: {self.TARGET_LOW[2]:.1f}~{self.TARGET_HIGH[2]:.1f}m | '
                f'ARRIVE: {self.ARRIVE_DIST:.1f}m, '
                f'HOVER: {self.HOVER_STEPS}steps @ <{self.HOVER_MAX_SPEED:.2f}m/s'
            )

    def _get_tilt_angle(self) -> float:
        """計算無人機當前傾角，用於判定是否墜機失控"""
        x, y, z, w = self.ros.current_quat
        z_z = 1.0 - 2.0 * (x**2 + y**2)
        z_z = np.clip(z_z, -1.0, 1.0)
        return float(np.arccos(z_z))

    def reset(self, seed=None, options=None):
        """重置環境：包含停止馬達、硬重置模擬器、重新起飛與姿態穩定"""
        super().reset(seed=seed)
        self.step_count = 0

        self.ros.send_velocity(0.0, 0.0, 0.0)
        self.ros.land_pub.publish(Empty())
        time.sleep(0.5)

        self.ros.reset_world()

        self.ros.land_pub.publish(Empty())
        time.sleep(1.5)

        # 確保成功起飛至指定高度
        for attempt in range(5):
            self.ros.takeoff_pub.publish(Empty())
            start_ns = self.ros.get_clock().now().nanoseconds
            while (self.ros.get_clock().now().nanoseconds - start_ns) < 5e8:
                rclpy.spin_once(self.ros, timeout_sec=0.01)
            if self.ros.current_pose[2] > 0.2:
                break

        vel_mode_msg = Bool()
        vel_mode_msg.data = True
        self.ros.vel_mode_pub.publish(vel_mode_msg)
        self.ros.soft_reset_pub.publish(Empty())

        # 等待無人機姿態與速度穩定
        stable_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            pos = self.ros.current_pose
            if abs(pos[0]) < 0.5 and abs(pos[1]) < 0.5 and pos[2] > 0.4:
                break
            if time.time() - stable_start > 3.0:
                self.ros.get_logger().warn('[WARN] 重置穩定階段超時!')
                break
        self.ros.soft_reset_pub.publish(Empty())
        time.sleep(0.3)

        for _ in range(10):
            rclpy.spin_once(self.ros, timeout_sec=0.02)
            vel_norm = float(np.linalg.norm(self.ros.current_vel))
            if vel_norm < 0.5 and self.ros.current_pose[2] > 0.4:
                break
            self.ros.send_velocity(0.0, 0.0, 0.0)
            self.ros.soft_reset_pub.publish(Empty())

        # 隨機生成目標座標
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        self.ros.send_velocity(0.0, 0.0, 0.0)
        rclpy.spin_once(self.ros, timeout_sec=0.1)

        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0
        
        self.initial_dist = self.prev_dist
        self.hover_steps = 0
        for key in self.ep_components.keys():
            self.ep_components[key] = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """
        推進環境步數，執行時間同步控制迴圈。
        確保動作能穩定執行指定的時間步長。
        """
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        real_velocity = action * 0.8

        # 強制時間阻塞：確保物理模擬器執行滿 0.1 秒
        start_ns = self.ros.get_clock().now().nanoseconds
        while (self.ros.get_clock().now().nanoseconds - start_ns) < 1e8:
            self.ros.send_velocity(*real_velocity)
            rclpy.spin_once(self.ros, timeout_sec=0.01)
        self.step_count += 1

        obs = self._get_obs()
        pos = self.ros.current_pose.copy()

        reward, terminated = self._compute_reward(action, pos)
        truncated = (self.step_count >= self.MAX_STEPS)

        # 超時懲罰設計，限制每個回合的最大步數
        if truncated and not terminated:
            r_timeout = -50.0  
            reward += r_timeout
            self.ep_components['time'] += r_timeout

        info = {}
        if terminated or truncated:
            info['ep_components'] = self.ep_components.copy()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        """
        計算獎勵函數 (Reward Function)。
        使用絕對距離誤差作為進度指標，並結合時間懲罰與邊界懲罰機制。
        參考 CRM/RCM 目標導航獎勵設計邏輯。
        """
        terminated = False
        
        # 計算絕對距離誤差
        curr_dist  = float(np.linalg.norm(pos - self.target))
        if np.isnan(curr_dist):
            curr_dist = 10.0
            
        curr_speed = float(np.linalg.norm(self.ros.current_vel))

        # 時間耗損懲罰 (Time Deduction)
        r_time = -0.2 
        
        # 進度獎勵：基於距離差異 (Difference in distance) 給予獎勵
        delta = self.prev_dist - curr_dist
        r_progress = 40.0 * delta
        self.prev_dist = curr_dist

        r_proximity = 0.0
        r_decel = 0.0
        r_action = 0.0
        r_boundary = 0.0
        r_arrive   = 0.0

        r_action = -0.01 * float(np.sum(np.square(action)))

        # 減速帶機制：確保無人機在靠近目標時降低速度以符合懸停條件
        decel_zone = self.ARRIVE_DIST * 2.0
        speed_thresh = self.HOVER_MAX_SPEED * 1.2
        if self.ARRIVE_DIST <= curr_dist < decel_zone:
            excess_speed = max(0.0, curr_speed - speed_thresh)
            proximity_ratio = (decel_zone - curr_dist) / decel_zone
            r_decel = -5.0 * excess_speed * proximity_ratio

        tilt_angle    = self._get_tilt_angle()
        is_crashed    = tilt_angle > (np.pi / 3.0)
        
        # 邊界懲罰判定
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )

        if out_of_bounds or is_crashed:
            r_boundary = -100.0
            terminated = True
            self.hover_steps = 0
        else:
            if curr_dist < self.ARRIVE_DIST:
                if curr_speed < self.HOVER_MAX_SPEED:
                    self.hover_steps += 1
                    hover_bonus = 5.0 * self.hover_steps
                    r_proximity = hover_bonus + 10.0 * (self.ARRIVE_DIST - curr_dist) - 8.0 * curr_speed
                    
                    if self.hover_steps >= self.HOVER_STEPS:
                        # 成功抵達/穿過目標獎勵 (Cross target)
                        r_arrive = 200.0
                        terminated = True
                else:
                    self.hover_steps = 0
                    r_proximity = -8.0 * curr_speed
            else:
                self.hover_steps = 0

        self.ep_components['time']      += r_time
        self.ep_components['progress']  += r_progress
        self.ep_components['proximity'] += r_proximity
        self.ep_components['action']    += r_action
        self.ep_components['decel']     += r_decel 
        self.ep_components['arrive']    += r_arrive
        self.ep_components['boundary']  += r_boundary

        reward = r_time + r_progress + r_proximity + r_action + r_decel + r_boundary + r_arrive
        return reward, terminated

    def _get_obs(self) -> np.ndarray:
        """獲取並歸一化觀察狀態 (Observation)"""
        pos    = self.ros.current_pose
        vel    = self.ros.current_vel
        target = self.target
        yaw    = self.ros.current_yaw

        dx_w = float(target[0] - pos[0])
        dy_w = float(target[1] - pos[1])
        dz_w = float(target[2] - pos[2])

        # 座標轉換：從世界座標系轉至機體座標系
        dx_b =  dx_w * np.cos(yaw) + dy_w * np.sin(yaw)
        dy_b = -dx_w * np.sin(yaw) + dy_w * np.cos(yaw)

        # 動態範圍歸一化
        scale_xy = self.BOUNDARY_XY
        scale_z  = self.TARGET_HIGH[2]
        rel_pos_scaled = np.array([dx_b / scale_xy, dy_b / scale_xy, dz_w / scale_z], dtype=np.float32)
        vels_scaled    = (vel / self.MAX_SPEED).astype(np.float32)

        obs = np.concatenate([rel_pos_scaled, vels_scaled])
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -2.0, 2.0)

        return obs

    def get_logger(self):
        return self.ros.get_logger()