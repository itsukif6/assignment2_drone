#!/usr/bin/env python3
"""
drone_env.py  (修正版)
----------------------
修正清單:
  問題3: reset() 後 prev_dist 使用舊 pose → 等待 pose 確實更新後才計算
  問題4: gt_vel 是 body frame，被當 world frame 用 → 移除 velocity 觀測，
         改用 (pos, target, rel_pos, dist) 共 7 維，更乾淨且無座標系混淆
  問題5: spin_once 時序不穩，step 後不保證收到新 pose → 改為等待 pose 真的變化
  問題6: obs 和 pos 讀取時機略有差異 → 統一在 spin 穩定後一次讀取

其餘邏輯（reward 函數設計、episode 結構、ROS 介面）維持原本。
"""

import numpy as np
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv

import gymnasium as gym
from gymnasium import spaces


# ================================================================
# Part 1: ROS 2 介面層
# ================================================================
class DroneROSInterface(Node):
    """
    把 ROS 2 的 publisher/subscriber/service client 包裝成簡單的介面。

    修正點（問題5）:
        新增 pose_stamp 欄位，每次收到新 pose 時遞增，
        讓 step() 可以確認 pose 真的更新了，而不只是靠 timeout。
    """

    def __init__(self):
        super().__init__('rl_drone_interface')

        self.current_pose  = np.zeros(3, dtype=np.float32)
        self.pose_received = False
        self.pose_stamp    = 0          # 每收到一筆新 pose 就 +1，用於確認更新

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )
        self.takeoff_pub = self.create_publisher(
            Empty, '/simple_drone/takeoff', 10
        )
        self.soft_reset_pub = self.create_publisher(
            Empty, '/simple_drone/reset', 10
        )

        # /reset_world Service Client（硬重置物理世界）
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self._pose_cb, 10
        )
        # 問題4：不再訂閱 gt_vel，velocity 從 observation 移除
        # （gt_vel 是 body frame，和 pos 的 world frame 不一致）

        self.get_logger().info('DroneROSInterface initialized (velocity observation disabled)')

    def _pose_cb(self, msg: Pose):
        """收到位置訊息，更新 current_pose 並遞增 pose_stamp。"""
        self.current_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z],
            dtype=np.float32
        )
        self.pose_received = True
        self.pose_stamp   += 1          # 問題5 修正用

    def send_velocity(self, vx: float, vy: float, vz: float):
        """發布速度命令到 /simple_drone/cmd_vel。"""
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.linear.z  = float(vz)
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def reset_world(self) -> bool:
        """
        硬重置：呼叫 /reset_world Service，強制 Gazebo 將無人機傳送回原點。
        """
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_world service not available, skipping hard reset')
            return False

        req    = EmptySrv.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        return future.result() is not None

    def takeoff(self):
        """發送起飛指令。"""
        self.takeoff_pub.publish(Empty())

    def spin_until_pose_updated(self, old_stamp: int, timeout_sec: float = 1.0) -> bool:
        """
        問題5 修正：持續 spin 直到收到至少一筆比 old_stamp 新的 pose，
        或超時。回傳 True 代表成功收到新 pose，False 代表超時。

        這是解決「spin_once 不保證收到新 pose」的核心修正。
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.pose_stamp > old_stamp:
                return True
        return False


# ================================================================
# Part 2: Gym 環境
# ================================================================
class DroneGymEnv(gym.Env):
    """
    Task B：隨機目標導航環境（修正版）。

    主要修正：
        問題3: reset() 結束前，等待 pose 從「reset 後的原點附近」確實穩定
        問題4: observation 維度從 13 → 7，移除 body-frame velocity
               新 obs = [pos_x, pos_y, pos_z, rel_x, rel_y, rel_z, dist]
               （target 可由 pos + rel_pos 推算，不需重複）
        問題5: step() 改為等待 pose_stamp 遞增，確保讀到 action 執行後的真實狀態
        問題6: obs 和 reward 計算統一使用同一次 pose 讀取
    """

    MAX_SPEED    = 1.0
    ARRIVE_DIST  = 0.4
    MAX_STEPS    = 200
    BOUNDARY_XY  = 3.0
    BOUNDARY_Z_MAX = 5.0
    BOUNDARY_Z_MIN = -1.0

    TARGET_LOW  = np.array([-0.5, -0.5, 1.0], dtype=np.float32)
    TARGET_HIGH = np.array([ 0.5,  0.5, 3.0], dtype=np.float32)

    MIN_HOVER_Z = 0.8

    # 問題5：step() 等待新 pose 的超時設定
    # Gazebo 物理步長 1ms，gt_pose 約 1kHz 發布，0.3 秒理論上足夠
    POSE_WAIT_TIMEOUT = 0.3

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # Action: [vx, vy, vz]
        self.action_space = spaces.Box(
            low  = -self.MAX_SPEED,
            high =  self.MAX_SPEED,
            shape=(3,),
            dtype=np.float32
        )

        # 問題4 修正：Observation 降為 7 維，移除 body-frame velocity
        # [pos_x, pos_y, pos_z, rel_x, rel_y, rel_z, dist]
        # rel = target - pos，dist = ||rel||
        obs_limit = np.array(
            [20, 20, 10,   # 無人機位置 (world frame)
             40, 40, 20,   # 相對位置 target - pos (world frame)
             50],          # 絕對距離
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low  = -obs_limit,
            high =  obs_limit,
            shape=(7,),
            dtype=np.float32
        )

        self.target     = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_dist  = None

        self.ep_components = {
            'progress': 0.0, 'proximity': 0.0, 'arrive': 0.0,
            'time': 0.0, 'boundary': 0.0,
        }

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Episode 重置流程（修正版）：

        Step 1: 硬重置 Gazebo 世界
        Step 2: 等待 pose 回到接近原點（問題3 修正）
        Step 3: 起飛
        Step 4: 等待達到 MIN_HOVER_Z
        Step 5: 生成目標點
        Step 6: 確認 prev_dist 用的是真實起飛後的 pose（問題3 修正）
        """
        super().reset(seed=seed)
        self.step_count = 0

        # --- Step 1: 硬重置 ---
        self.ros.reset_world()

        # --- Step 2: 等待 pose 確實回到原點附近（問題3 修正）---
        # reset_world 後 Gazebo 把無人機傳送回 (0, 0, 0)，
        # 但 current_pose 可能還保留上一個 episode 的舊值。
        # 必須等到 pose callback 收到「重置後」的新 pose 才能繼續，
        # 否則 prev_dist 會用錯誤的起始位置計算。
        for _ in range(30):
            rclpy.spin_once(self.ros, timeout_sec=0.1)

        # 額外確認：等待 pose 接近原點（z < 0.1 代表在地面上）
        timeout = 0
        while self.ros.current_pose[2] > 0.5 and timeout < 50:
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            timeout += 1

        # --- Step 3: 起飛 ---
        self.ros.takeoff()

        # --- Step 4: 等待達到懸停高度 ---
        for _ in range(100):
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            if self.ros.current_pose[2] > self.MIN_HOVER_Z:
                break

        # --- Step 5: 生成目標點 ---
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        # --- Step 6: 確認 prev_dist 用真實起飛後的 pose（問題3 修正）---
        # 再多 spin 幾次讓 pose 穩定
        for _ in range(10):
            rclpy.spin_once(self.ros, timeout_sec=0.05)

        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0

        for key in self.ep_components:
            self.ep_components[key] = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        """
        問題5 修正：
            發出速度指令後，等待 pose_stamp 遞增（即確認收到新 pose），
            而不是盲目 spin_once 一次就讀取狀態。
            這確保 (s, a, r, s') 裡的 s' 和 r 是基於 action 執行後的真實狀態。

        問題6 修正：
            obs 和 reward 都使用同一次 pose 讀取（step 結束時的 current_pose）。
        """
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        real_velocity = action * 0.8

        # 記錄發送前的 pose_stamp
        stamp_before = self.ros.pose_stamp

        # 發送速度指令
        self.ros.send_velocity(*real_velocity)

        # 問題5 修正：等待收到「指令執行後」的新 pose
        # spin_until_pose_updated 會持續 spin 直到 pose_stamp > stamp_before
        got_new_pose = self.ros.spin_until_pose_updated(
            stamp_before, timeout_sec=self.POSE_WAIT_TIMEOUT
        )
        if not got_new_pose:
            # 超時警告（正常訓練時應很少發生）
            self.ros.get_logger().warn(
                f'Step {self.step_count}: pose update timeout after {self.POSE_WAIT_TIMEOUT}s',
                throttle_duration_sec=5.0
            )

        self.step_count += 1

        # 問題6 修正：統一讀取一次 pose，obs 和 reward 都用這個值
        pos = self.ros.current_pose.copy()

        reward, terminated = self._compute_reward(action, pos)
        obs      = self._get_obs_from_pose(pos)
        truncated = (self.step_count >= self.MAX_STEPS)

        info = {}
        if terminated or truncated:
            info['ep_components'] = self.ep_components.copy()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        """
        Reward 函數（邏輯與原版相同，未動）：
            r_progress  : 距離縮短獎勵
            r_arrive    : 到達獎勵
            r_time      : 時間懲罰
            r_boundary  : 邊界懲罰
            r_proximity : 蘿蔔引導（常駐微小正回饋）
        """
        terminated = False
        curr_dist  = float(np.linalg.norm(pos - self.target))
        if np.isnan(curr_dist):
            curr_dist = 10.0

        # 1. 距離縮短獎勵
        r_progress = 5.0 * (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        # 2. 到達獎勵
        r_arrive = 0.0
        if curr_dist < self.ARRIVE_DIST:
            r_arrive   = 100.0
            terminated = True

        # 3. 時間懲罰
        r_time = -0.05

        # 4. 邊界懲罰
        r_boundary = 0.0
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )
        if out_of_bounds:
            r_boundary = -50.0
            terminated = True

        # 5. 蘿蔔引導
        r_proximity = max(0.0, 0.05 * (1.0 - curr_dist / 2.0))

        self.ep_components['progress']  += r_progress
        self.ep_components['proximity'] += r_proximity
        self.ep_components['arrive']    += r_arrive
        self.ep_components['time']      += r_time
        self.ep_components['boundary']  += r_boundary

        reward = r_progress + r_arrive + r_time + r_boundary + r_proximity
        return reward, terminated

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """從當前 current_pose 建構 observation（供 reset 使用）。"""
        return self._get_obs_from_pose(self.ros.current_pose.copy())

    def _get_obs_from_pose(self, pos: np.ndarray) -> np.ndarray:
        """
        問題4 修正：觀測向量從 13 維降為 7 維，移除有座標系問題的 velocity。

        新格式（7 維）：
            [pos_x, pos_y, pos_z,       ← 無人機位置 (world frame)
             rel_x, rel_y, rel_z,       ← 相對位置 = target - pos (world frame)
             dist]                       ← 絕對距離

        不再包含：
            - target 的絕對座標（可由 pos + rel_pos 推算，冗餘）
            - velocity（gt_vel 是 body frame，與 pos 的 world frame 不一致）
        """
        rel_pos = self.target - pos
        dist    = float(np.linalg.norm(rel_pos))

        obs = np.concatenate([
            pos,
            rel_pos,
            [dist]
        ]).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        obs = obs / self.observation_space.high   # 正規化到 [-1, 1]
        return obs

    def get_logger(self):
        return self.ros.get_logger()