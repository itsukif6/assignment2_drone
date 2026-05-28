#!/usr/bin/env python3
"""
drone_env.py
------------
無人機隨機目標導航 (Random Target Navigation) 環境。
將 ROS 2 模擬環境封裝為相容於 Gymnasium 的強化學習環境。
支援 Level 1 ~ 7 漸進式課程學習 (Curriculum Learning)。

[Reference 1]
Tan, Z., & Karaköse, M. (2023). "A new approach for drone tracking with drone
using Proximal Policy Optimization based distributed deep reinforcement learning."
SoftwareX.
- 應用部分：神經網路架構設計 (Policy Network Architecture)。
- 具體實作：採用 [256, 256] 之隱藏層大小，並搭配 Tanh 啟動函數，以適應三維空間之連續控制。

[Reference 2]
Zhang, J., Nguyen, S., Rivera, C. E. O., & Tyni, K. "AirPilot: Interpretable
PPO-based DRL Auto-Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights."
- 應用部分：
  1. 觀測空間設計：同時感知自身位置、目標位置、相對位置與速度。
  2. EffectiveSpeed 成功判定：Distance / Timestep，結合導航速度、精準度、
     過衝量與安定時間於單一指標，無人機須在目標圈內穩定停留 HOVER_STEPS 步才算成功。
  3. 傾角懲罰：連續漸進式懲罰大傾角動作，防止危險飛行。
  4. 減速帶機制：靠近目標時懲罰超速，降低過衝。

[Reference 3]
Shen, S.-E., & Huang, Y.-C. (2024). "Application of Reinforcement Learning
in Controlling Quadrotor UAV Flight Actions." Drones.
- 應用部分：PPO 模型超參數、CRM 獎勵機制。
- 具體實作：
  1. 採用文獻測試驗證之最佳 PPO 參數：learning_rate=3e-4, n_steps=2048,
     batch_size=64, gamma=0.99, gae_lambda=0.95, vf_coef=0.5。
  2. 獎勵機制設計：距離差值進度獎勵、每步時間耗損懲罰、邊界懲罰 -100。

MDP 定義：
    State  (13 維): [pos_x, pos_y, pos_z,
                     target_x, target_y, target_z,
                     rel_x, rel_y, rel_z,
                     vel_x, vel_y, vel_z,
                     distance]
    Action  (3 維): [vx, vy, vz]，範圍 [-MAX_SPEED, MAX_SPEED]
    Reward        : 見 _compute_reward()
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


# ================================================================
# Part 1: ROS 2 介面層
# ================================================================
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
        self.pose_received = False

        # Publishers
        self.cmd_vel_pub    = self.create_publisher(Twist,  '/simple_drone/cmd_vel',       10)
        self.takeoff_pub    = self.create_publisher(Empty,  '/simple_drone/takeoff',        10)
        self.land_pub       = self.create_publisher(Empty,  '/simple_drone/land',           10)
        self.vel_mode_pub   = self.create_publisher(Bool,   '/simple_drone/dronevel_mode',  10)
        self.soft_reset_pub = self.create_publisher(Empty,  '/simple_drone/reset',          10)

        # Service client：硬重置物理世界
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose,  '/simple_drone/gt_pose', self._pose_cb, 10)
        self.vel_sub  = self.create_subscription(
            Twist, '/simple_drone/gt_vel',  self._vel_cb,  10)

        self.get_logger().info('DroneROSInterface initialized.')

    def _pose_cb(self, msg: Pose):
        """收到位置訊息，更新 current_pose 與 current_quat。"""
        self.current_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        q = msg.orientation
        self.current_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        """收到速度訊息，更新 current_vel。"""
        self.current_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)

    def send_velocity(self, vx: float, vy: float, vz: float):
        """發布速度指令到 /simple_drone/cmd_vel。"""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        msg.angular.x = msg.angular.y = msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def reset_world(self) -> bool:
        """呼叫 /reset_world Service，強制 Gazebo 將無人機傳送回初始位置。"""
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_world service not available, skipping.')
            return False
        req    = EmptySrv.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result() is not None

    def takeoff(self):
        """發送起飛指令。"""
        self.takeoff_pub.publish(Empty())


# ================================================================
# Part 2: Gym 環境
# ================================================================
class DroneGymEnv(gym.Env):
    """
    Task B：隨機目標導航環境，支援 Level 1~7 課程學習。

    成功判定 (AirPilot EffectiveSpeed 概念)：
        無人機須在目標圈內 (dist < ARRIVE_DIST) 且速度低於 HOVER_MAX_SPEED，
        連續累積 HOVER_STEPS 步後才判定成功。
        - 若中途離開目標圈或速度過快，hover_count 不重置（Timestep 繼續增加，
          EffectiveSpeed = initial_dist / step_count 自動下降，懲罰過衝行為）。
        - HOVER_STEPS 隨 Level 從 5 步（L1）爬升至 20 步（L7），對應論文
          50 timesteps × 0.04s ≈ 2s，換算本環境 0.1s/step 即 20 步。
    """

    MAX_SPEED      = 1.0
    MAX_STEPS      = 600
    BOUNDARY_Z_MAX = 6.5   # 配合 Level 7 最高目標 5.5m
    BOUNDARY_Z_MIN = 0.0
    MIN_HOVER_Z    = 0.8
    MAX_CURRICULUM_LEVEL = 7

    # ------------------------------------------------------------------
    # 課程學習難度表
    # hover_steps：對應論文 50 ts×0.04s=2s，換算 0.1s/step
    #   L1: 5步(0.5s，初學緩和)  →  L7: 20步(2s，對應論文原值)
    # ------------------------------------------------------------------
    CURRICULUM_LEVELS = {
        1: {'target_low': np.array([-1.0,  -1.0,  1.5], dtype=np.float32),
            'target_high': np.array([ 1.0,   1.0,  2.5], dtype=np.float32),
            'boundary_xy': 3.0,  'arrive_dist': 1.50,
            'hover_steps': 5,    'hover_max_spd': 0.80},

        2: {'target_low': np.array([-2.5,  -2.5,  1.5], dtype=np.float32),
            'target_high': np.array([ 2.5,   2.5,  3.0], dtype=np.float32),
            'boundary_xy': 4.0,  'arrive_dist': 1.20,
            'hover_steps': 8,    'hover_max_spd': 0.60},

        3: {'target_low': np.array([-4.0,  -4.0,  1.5], dtype=np.float32),
            'target_high': np.array([ 4.0,   4.0,  3.5], dtype=np.float32),
            'boundary_xy': 5.0,  'arrive_dist': 1.00,
            'hover_steps': 11,   'hover_max_spd': 0.45},

        4: {'target_low': np.array([-5.5,  -5.5,  1.5], dtype=np.float32),
            'target_high': np.array([ 5.5,   5.5,  4.0], dtype=np.float32),
            'boundary_xy': 6.5,  'arrive_dist': 0.90,
            'hover_steps': 14,   'hover_max_spd': 0.35},

        5: {'target_low': np.array([-7.0,  -7.0,  1.5], dtype=np.float32),
            'target_high': np.array([ 7.0,   7.0,  4.5], dtype=np.float32),
            'boundary_xy': 8.0,  'arrive_dist': 0.80,
            'hover_steps': 16,   'hover_max_spd': 0.28},

        6: {'target_low': np.array([-8.5,  -8.5,  1.5], dtype=np.float32),
            'target_high': np.array([ 8.5,   8.5,  5.0], dtype=np.float32),
            'boundary_xy': 9.5,  'arrive_dist': 0.70,
            'hover_steps': 18,   'hover_max_spd': 0.24},

        7: {'target_low': np.array([-10.0, -10.0, 1.5], dtype=np.float32),
            'target_high': np.array([10.0,  10.0,  5.5], dtype=np.float32),
            'boundary_xy': 11.0, 'arrive_dist': 0.60,
            'hover_steps': 20,   'hover_max_spd': 0.20},
    }

    # 類別層級預設值（供 set_curriculum_level 前的參照）
    TARGET_LOW  = CURRICULUM_LEVELS[1]['target_low']
    TARGET_HIGH = CURRICULUM_LEVELS[1]['target_high']

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # ── 動作空間：三軸速度 [vx, vy, vz] ──────────────────────────
        self.action_space = spaces.Box(
            low=-self.MAX_SPEED, high=self.MAX_SPEED,
            shape=(3,), dtype=np.float32
        )

        # ── 觀測空間：13 維 ───────────────────────────────────────────
        # [pos(3), target(3), rel_pos(3), vel(3), distance(1)]
        # 設計依據 AirPilot Section IV：
        #   agent 同時感知自身絕對位置、目標位置與速度；
        #   加入相對位置向量與純量距離讓網路具備清晰方向感，
        #   降低神經網路隱式學習座標差的負擔。
        obs_limit = np.array(
            [15, 15,  8,    # 無人機絕對位置 (配合 L7 邊界)
             15, 15,  6,    # 目標絕對位置
             22, 22, 14,    # 相對位置 (最大對角距離)
              2,  2,  2,    # 速度
             30],           # 純量距離
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit,
            shape=(13,), dtype=np.float32
        )

        # ── 內部狀態 ────────────────────────────────────────────────
        self.curriculum_level = 1
        self._apply_curriculum(1)

        self.target       = np.zeros(3, dtype=np.float32)
        self.step_count   = 0
        self.previous_dist    = None
        self.initial_dist = None   # 回合起點距離，用於 EffectiveSpeed 計算
        self.hover_count  = 0      # 在目標圈內且低速的累積步數

        # 各項獎勵細項（供 train callback 分析）
        self.ep_components = {
            'progress': 0.0, 'arrive':   0.0, 'action': 0.0,
            'time':     0.0, 'boundary': 0.0, 'proximity': 0.0,
            'decel':    0.0, 'align':    0.0, 'tilt':   0.0,
        }

    # ──────────────────────────────────────────────────────────────
    # 課程學習
    # ──────────────────────────────────────────────────────────────
    def _apply_curriculum(self, level: int):
        """載入指定難度層級的環境參數。"""
        cfg = self.CURRICULUM_LEVELS[level]
        self.TARGET_LOW      = cfg['target_low'].copy()
        self.TARGET_HIGH     = cfg['target_high'].copy()
        self.BOUNDARY_XY     = cfg['boundary_xy']
        self.ARRIVE_DIST     = cfg['arrive_dist']
        self.HOVER_STEPS     = cfg['hover_steps']
        self.HOVER_MAX_SPEED = cfg['hover_max_spd']

    def set_curriculum_level(self, level: int):
        """外部呼叫以切換課程難度，並輸出紀錄。"""
        level = int(np.clip(level, 1, self.MAX_CURRICULUM_LEVEL))
        if level != self.curriculum_level:
            self.curriculum_level = level
            self._apply_curriculum(level)
            self.ros.get_logger().info(
                f'[Curriculum] Level -> {level} | '
                f'Target x/y: ±{self.TARGET_HIGH[0]:.1f}m, '
                f'z: {self.TARGET_LOW[2]:.1f}~{self.TARGET_HIGH[2]:.1f}m | '
                f'ARRIVE: {self.ARRIVE_DIST:.2f}m, '
                f'HOVER: {self.HOVER_STEPS} steps @ <{self.HOVER_MAX_SPEED:.2f} m/s'
            )

    # ──────────────────────────────────────────────────────────────
    # 輔助計算
    # ──────────────────────────────────────────────────────────────
    def _get_tilt_angle(self) -> float:
        """計算機身 Z 軸與世界 Z 軸的夾角 (弧度)。"""
        x, y, z, w = self.ros.current_quat
        z_z = 1.0 - 2.0 * (x ** 2 + y ** 2)
        return float(np.arccos(np.clip(z_z, -1.0, 1.0)))

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        五階段強健重置：
          1. 安全降落 + 硬重置物理世界
          2. 強健式起飛（最多 5 次重試）
          3. 切換速度控制模式 + 軟重置 PID
          4. 空間位置穩定（等待回到原點附近）
          5. 速度穩定（確認完全靜止後再生成目標）
        """
        super().reset(seed=seed)
        self.step_count  = 0
        self.hover_count = 0

        # ── 階段一：降落 + 硬重置 ────────────────────────────────
        self.ros.send_velocity(0.0, 0.0, 0.0)
        self.ros.land_pub.publish(Empty())
        time.sleep(0.5)
        self.ros.reset_world()
        self.ros.land_pub.publish(Empty())
        time.sleep(1.5)

        # ── 階段二：強健式起飛 ───────────────────────────────────
        for _ in range(5):
            self.ros.takeoff_pub.publish(Empty())
            start_ns = self.ros.get_clock().now().nanoseconds
            while (self.ros.get_clock().now().nanoseconds - start_ns) < 5e8:
                rclpy.spin_once(self.ros, timeout_sec=0.01)
            if self.ros.current_pose[2] > 0.2:
                break

        # ── 階段三：速度控制模式 + PID 清零 ─────────────────────
        vel_mode_msg      = Bool()
        vel_mode_msg.data = True
        self.ros.vel_mode_pub.publish(vel_mode_msg)
        self.ros.soft_reset_pub.publish(Empty())

        # ── 階段四：空間穩定 ─────────────────────────────────────
        stable_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            pos = self.ros.current_pose
            if abs(pos[0]) < 0.5 and abs(pos[1]) < 0.5 and pos[2] > 0.4:
                break
            if time.time() - stable_start > 3.0:
                self.ros.get_logger().warn('[Reset] Position stabilization timeout.')
                break

        self.ros.soft_reset_pub.publish(Empty())
        time.sleep(0.3)

        # ── 階段五：速度穩定 ─────────────────────────────────────
        for _ in range(10):
            rclpy.spin_once(self.ros, timeout_sec=0.02)
            if (float(np.linalg.norm(self.ros.current_vel)) < 0.5
                    and self.ros.current_pose[2] > 0.4):
                break
            self.ros.send_velocity(0.0, 0.0, 0.0)
            self.ros.soft_reset_pub.publish(Empty())

        # ── 生成目標點 ───────────────────────────────────────────
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        self.ros.send_velocity(0.0, 0.0, 0.0)
        rclpy.spin_once(self.ros, timeout_sec=0.1)

        dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        self.previous_dist = dist if not np.isnan(dist) else 5.0
        self.initial_dist  = self.previous_dist

        for key in self.ep_components:
            self.ep_components[key] = 0.0

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        action       = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        real_velocity = action * 0.8

        # 時間同步阻塞：確保每步物理時間為 0.1s
        start_time_nanosecond = self.ros.get_clock().now().nanoseconds
        while (self.ros.get_clock().now().nanoseconds - start_time_nanosecond) < 1e8:
            self.ros.send_velocity(*real_velocity)
            rclpy.spin_once(self.ros, timeout_sec=0.01)
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
            # 回傳 EffectiveSpeed 供 test.py 評估品質
            info['ep_components']  = self.ep_components.copy()
            info['effective_speed'] = (
                self.initial_dist / self.step_count
                if self.step_count > 0 else 0.0
            )

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    # 獎勵函數
    # ──────────────────────────────────────────────────────────────
    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        """
        獎勵函數由八項組成，設計目標為 L1~L7 全程通用：

        ┌──────────────┬────────────────────────────────────────────────┐
        │ 項目          │ 說明                                           │
        ├──────────────┼────────────────────────────────────────────────┤
        │ r_progress   │ 距離縮短獎勵（Shen 2024 CRM +d）                │
        │ r_align      │ 速度方向與目標方向 cosine 對齊獎勵（Tan 2023）   │
        │ r_decel      │ 靠近目標時超速的減速帶懲罰（AirPilot）           │
        │ r_proximity  │ 圈內每步形狀獎勵，越靠中心 + 越慢越高            │
        │ r_arrive     │ 連續懸停 HOVER_STEPS 步後的品質到達獎勵          │
        │ r_action     │ 動作 L2 懲罰，促進平滑飛行（AirPilot）           │
        │ r_tilt       │ 傾角漸進懲罰 30°~60°（AirPilot）                │
        │ r_boundary   │ 出界 / 墜機 -100（Shen 2024）                   │
        │ r_time       │ 每步時間耗損 -0.2（Shen 2024 -Tm）              │
        └──────────────┴────────────────────────────────────────────────┘

        L1~L7 通用性設計原則：
        - r_progress 係數固定（25.0），不受距離稀釋，L7 長距離仍有強動機。
        - r_arrive 以 EffectiveSpeed 縮放（initial_dist / step_count × K），
          飛越近、停越穩則分越高，L1 短距離快到 vs L7 長距離耗時，自動縮放公平。
        - r_align 只在圈外且速度 > 0.05 才啟動，圈內由 r_proximity 接管。
        - HOVER_STEPS / ARRIVE_DIST / HOVER_MAX_SPEED 皆由課程層級決定，
          獎勵函數本體不含魔術數字。
        """
        terminated = False

        current_dist  = float(np.linalg.norm(pos - self.target))
        if np.isnan(current_dist):
            current_dist = 10.0
        current_speed = float(np.linalg.norm(self.ros.current_vel))

        # ── 1. 時間懲罰 (-Tm) ────────────────────────────────────
        # 固定每步 -0.2，逼迫快速完成任務。
        # L1~L7 通用：數值固定，不受距離影響。
        r_time = -0.2

        # ── 2. 距離縮短進度獎勵 ──────────────────────────────────
        # [Shen 2024] CRM 獎勵設計：+d = previous_dist - current_dist。
        # 係數 25.0 確保全速靠近時（delta ≈ 0.1m/step）
        # 單步淨利潤 = 25×0.1 - 0.2(time) - 0.03(action) ≈ +2.27 > 0，
        # L7 長距離航行仍有足夠前進動機。
        # 遠離懲罰加倍（×1.4），不對稱設計讓「靠近有利、遠離很貴」。
        delta = self.previous_dist - current_dist
        if delta >= 0:
            r_progress = 25.0 * delta
        else:
            r_progress = 35.0 * delta   # 遠離加重懲罰
        self.previous_dist = current_dist

        # ── 3. 速度方向對齊獎勵 ──────────────────────────────────
        # [Tan 2023] 延伸概念：獎勵速度向量與「無人機→目標」方向的 cosine 相似度。
        # 讓無人機不只「靠近」，而是「朝正確方向加速」，避免繞圈蝸行。
        # 僅在圈外 (dist >= ARRIVE_DIST) 且有明顯速度時啟動；
        # 圈內改由 r_proximity 接管，避免雙重計算。
        r_align = 0.0
        if current_dist >= self.ARRIVE_DIST and current_speed > 0.05:
            error_vec = self.target - pos
            err_norm  = float(np.linalg.norm(error_vec))
            if err_norm > 1e-6:
                cos_sim = float(np.dot(self.ros.current_vel, error_vec)
                                / (current_speed * err_norm))
                # 範圍 [-1.5, +1.5]，對齊飛行最高 +1.5
                r_align = 1.5 * cos_sim

        # ── 4. 減速帶懲罰 ────────────────────────────────────────
        # [AirPilot] 靠近目標但速度超過閾值時，給予漸進式懲罰，降低過衝。
        # 減速帶寬度 = 2×ARRIVE_DIST，越靠近圈邊懲罰越重（proximity_ratio）。
        r_decel = 0.0
        decel_zone   = self.ARRIVE_DIST * 2.0
        speed_thresh = self.HOVER_MAX_SPEED * 1.2
        if self.ARRIVE_DIST <= current_dist < decel_zone:
            excess_speed    = max(0.0, current_speed - speed_thresh)
            proximity_ratio = (decel_zone - current_dist) / decel_zone
            r_decel = -2.0 * excess_speed * proximity_ratio

        # ── 5. 傾角漸進懲罰 ──────────────────────────────────────
        # [AirPilot] 大傾角=危險動作，給予連續懲罰而非只靠硬截止。
        # 軟門檻 30°(π/6)：超過後線性增加懲罰，至 60°(π/3) 觸發墜機判定。
        # 最大連續懲罰約 -2.5（tilt=π/3 時），不至於蓋過前進動機。
        r_tilt      = 0.0
        tilt_angle  = self._get_tilt_angle()
        soft_limit  = np.pi / 6.0    # 30°
        hard_limit  = np.pi / 3.0    # 60°
        if soft_limit < tilt_angle <= hard_limit:
            excess_tilt = (tilt_angle - soft_limit) / soft_limit   # 0~1
            r_tilt = -2.5 * excess_tilt

        # ── 6. 動作 L2 懲罰 ──────────────────────────────────────
        # [AirPilot] 抑制過大或高頻切換的控制指令，促進平滑節能飛行。
        # 最大值（action 全滿）= -0.01×3 = -0.03，影響微小不干擾主要動機。
        r_action = -0.01 * float(np.sum(np.square(action)))

        # ── 初始化其餘項目 ───────────────────────────────────────
        r_proximity = 0.0
        r_boundary  = 0.0
        r_arrive    = 0.0

        # ── 7. 邊界 / 墜機判定 ───────────────────────────────────
        # [Shen 2024] 出界或大傾角墜機 → -100 並終止。
        is_crashed    = tilt_angle > hard_limit
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )

        if out_of_bounds or is_crashed:
            r_boundary   = -100.0
            terminated   = True
            self.hover_count = 0

        else:
            if current_dist < self.ARRIVE_DIST:
                # ── 圈內邏輯 ─────────────────────────────────────
                if current_speed < self.HOVER_MAX_SPEED:
                    # 速度合格：累積懸停計數
                    # hover_count 不因離圈而重置（EffectiveSpeed 概念）
                    self.hover_count += 1

                    # 形狀獎勵：越靠中心 (+dist_ratio) 且越慢 (-speed_ratio) 越高
                    # 最差合法狀態（dist_ratio≈0, speed_ratio≈1）：
                    #   r_proximity = 1.5 + 0 - 1.0 = +0.5，淨利 ≈ +0.5 - 0.2 = +0.3 > 0
                    # 防止「靠近但不進圈」的邊緣徘徊策略。
                    dist_ratio  = (self.ARRIVE_DIST - current_dist) / self.ARRIVE_DIST
                    speed_ratio = current_speed / self.HOVER_MAX_SPEED
                    r_proximity = 1.5 + 2.0 * dist_ratio - 1.0 * speed_ratio

                    # ── 8. 到達獎勵（EffectiveSpeed 品質縮放）────────────
                    # [AirPilot] EffectiveSpeed = initial_dist / step_count，
                    # 結合導航速度、精準度、過衝量與安定時間。
                    # 到達獎勵 = 100（基礎）+ 100×EffectiveSpeed/參考速度（品質加成）
                    # 參考速度 = initial_dist / (HOVER_STEPS×2)，意即「剛好夠快就滿分」。
                    # 快到且穩 → 接近 +200；慢慢到或多次繞圈 → 接近 +100。
                    # L1~L7 通用：initial_dist 與 step_count 自然反映距離與耗時差異。
                    if self.hover_count >= self.HOVER_STEPS:
                        ref_speed  = self.initial_dist / max(self.HOVER_STEPS * 2, 1)
                        eff_speed  = self.initial_dist / max(self.step_count, 1)
                        quality    = float(np.clip(eff_speed / max(ref_speed, 1e-6), 0.0, 1.0))
                        r_arrive   = 100.0 + 100.0 * quality
                        terminated = True

                else:
                    # 速度超標：hover_count 不重置（繼續累計時間成本）
                    # 給予溫和摩擦力懲罰，防止圈內高速繞圈刷分。
                    excess_speed = current_speed - self.HOVER_MAX_SPEED
                    r_proximity  = -3.8 * excess_speed

            else:
                # 圈外：hover_count 不重置（EffectiveSpeed 概念，懲罰反覆離開）
                pass

        # ── 加總並記錄細項 ───────────────────────────────────────
        total_reward = (r_time + r_progress + r_align + r_decel +
                        r_proximity + r_arrive + r_action + r_tilt + r_boundary)

        self.ep_components['time']      += r_time
        self.ep_components['progress']  += r_progress
        self.ep_components['align']     += r_align
        self.ep_components['decel']     += r_decel
        self.ep_components['proximity'] += r_proximity
        self.ep_components['arrive']    += r_arrive
        self.ep_components['action']    += r_action
        self.ep_components['tilt']      += r_tilt
        self.ep_components['boundary']  += r_boundary

        return total_reward, terminated

    # ──────────────────────────────────────────────────────────────
    # 觀測
    # ──────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        回傳 13 維觀測向量（世界座標系，不做 Yaw 旋轉）：
        [pos_x, pos_y, pos_z,
         target_x, target_y, target_z,
         rel_x, rel_y, rel_z,
         vel_x, vel_y, vel_z,
         distance]

        設計依據 AirPilot Section IV：
        - 絕對位置 + 目標位置：讓網路知道自己在地圖哪裡。
        - 相對位置向量：直接給方向，降低網路隱式學習差值的負擔。
        - 純量距離：讓網路對「還有多遠」有明確感知。
        - 速度：對應 dPE/dt，幫助網路判斷動量方向，決策減速時機。
        """
        pos    = self.ros.current_pose
        vel    = self.ros.current_vel
        target = self.target

        rel_pos = target - pos
        dist    = float(np.linalg.norm(rel_pos))

        obs = np.concatenate([pos, target, rel_pos, vel, [dist]]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=30.0, neginf=-30.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        # 正規化至 [-1, 1]
        obs = obs / self.observation_space.high
        return obs

    def get_logger(self):
        return self.ros.get_logger()