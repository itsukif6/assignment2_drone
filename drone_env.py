#!/usr/bin/env python3
"""
drone_env.py
------------
Task B: 隨機目標導航 (Random Target Navigation)
將 NSYSU Drone 模擬環境包裝成 Gymnasium 相容的 Gym 環境.

前置安裝: 
    pip install stable-baselines3 gymnasium numpy

使用方式: 
    此檔案不直接執行, 由 train.py 和 test.py 匯入使用.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, TwistStamped
from std_msgs.msg import Empty

import gymnasium as gym
from gymnasium import spaces


# ================================================================
# Part 1: ROS 2 介面層
# 負責和 Gazebo 溝通, 把 ROS topic 包成簡單的 get/set
# ================================================================
class DroneROSInterface(Node):
    """
    把 ROS 2 的 publisher/subscriber 包裝成簡單的介面.
    Gym 環境透過這個物件和模擬器溝通, 不直接碰 ROS API.
    """

    def __init__(self):
        super().__init__('rl_drone_interface')

        # --- 儲存最新的感測值 ---
        self.current_pose = np.zeros(3, dtype=np.float32)   # [x, y, z]
        self.current_vel  = np.zeros(3, dtype=np.float32)   # [vx, vy, vz]
        self.pose_received = False   # 用來判斷是否已收到第一筆位置資料

        # --- Publishers: 發指令給無人機 ---
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )
        self.takeoff_pub = self.create_publisher(
            Empty, '/simple_drone/takeoff', 10
        )
        self.reset_pub = self.create_publisher(
            Empty, '/simple_drone/reset', 10
        )

        # --- Subscribers: 接收無人機狀態 ---
        # gt_pose: ground truth 位置 (最準確, 直接來自 Gazebo) 
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self._pose_cb, 10
        )
        # gt_vel: ground truth 速度 (比差分估計穩定很多) 
        self.vel_sub = self.create_subscription(
            Twist, '/simple_drone/gt_vel', self._vel_cb, 10
        )

        self.get_logger().info('DroneROSInterface 初始化完成')

    # ---- Callbacks: 每次收到訊息就自動更新 ----

    def _pose_cb(self, msg: Pose):
        """收到位置訊息, 更新 current_pose."""
        self.current_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z],
            dtype=np.float32
        )
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        """收到速度訊息, 更新 current_vel."""
        self.current_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z],
            dtype=np.float32
        )

    # ---- 發指令的方法 ----

    def send_velocity(self, vx: float, vy: float, vz: float):
        """發布速度命令到 /simple_drone/cmd_vel."""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def reset_drone(self):
        """
        重置無人機: 
        1. 發 /reset 讓無人機回到原點
        2. 等一下讓模擬器穩定
        3. 發 /takeoff 讓無人機起飛
        """
        self.reset_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=0.5)   # 等重置生效
        self.takeoff_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=1.5)   # 等起飛穩定


# ================================================================
# Part 2: Gym 環境
# 標準 Gymnasium 介面, PPO/SAC 等演算法都能直接使用
# ================================================================
class DroneGymEnv(gym.Env):
    """
    Task B: 隨機目標導航環境.

    每個 episode: 
        - 無人機重置回原點並起飛
        - 隨機生成一個目標點
        - Agent 輸出速度指令, 嘗試飛到目標點
        - 到達 or 超時 or 飛出邊界 → 結束

    MDP 定義: 
        State  (9維): [pos_x, pos_y, pos_z, target_x, target_y, target_z, vel_x, vel_y, vel_z]
        Action (3維): [vx, vy, vz], 範圍 [-MAX_SPEED, MAX_SPEED]
        Reward: 見 _compute_reward()
        Gamma : 在 train.py 中設定 (預設 0.99) 
    """

    # ---- 環境常數 (修改這裡來調整任務難度) ----
    MAX_SPEED      = 1.5    # 動作空間上下限, 單位 m/s
    ARRIVE_DIST    = 0.3    # 距離目標多近算「到達」, 單位 m
    MAX_STEPS      = 300    # 每個 episode 最多幾步 (1步 ≈ 0.1秒 → 30秒上限) 
    BOUNDARY_XY    = 10.0   # x/y 方向的邊界, 超過就終止
    BOUNDARY_Z_MAX = 6.0    # 最高飛多高
    BOUNDARY_Z_MIN = 0.2    # 最低 (快碰地板就終止) 

    # 目標點隨機範圍
    TARGET_LOW  = np.array([-5.0, -5.0, 1.0], dtype=np.float32)
    TARGET_HIGH = np.array([ 5.0,  5.0, 4.0], dtype=np.float32)

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # ---- 定義 Action Space ----
        # Agent 每步輸出 [vx, vy, vz], 都在 [-MAX_SPEED, MAX_SPEED]
        self.action_space = spaces.Box(
            low  = -self.MAX_SPEED,
            high =  self.MAX_SPEED,
            shape=(3,),
            dtype=np.float32
        )

        # ---- 定義 Observation Space ----
        # [drone_x, drone_y, drone_z, target_x, target_y, target_z, vel_x, vel_y, vel_z]
        # 範圍設寬一點, 避免觀測值超出邊界時報錯
        obs_limit = np.array(
            [20, 20, 10,   # 無人機位置上限
             20, 20, 10,   # 目標位置上限
             5,  5,  5],   # 速度上限
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low  = -obs_limit,
            high =  obs_limit,
            shape=(9,),
            dtype=np.float32
        )

        # ---- 內部狀態 ----
        self.target      = np.zeros(3, dtype=np.float32)
        self.step_count  = 0
        self.prev_dist   = None   # 上一步的距離, 用來算「距離縮短量」

    # ----------------------------------------------------------
    # reset(): 每個 episode 開始時呼叫
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置無人機
        self.ros.reset_drone()
        self.step_count = 0

        # 隨機生成目標點
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        # 等一下讓無人機穩定, 再讀初始距離
        rclpy.spin_once(self.ros, timeout_sec=0.3)
        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))

        self.get_logger().info(
            f'新 episode 開始, 目標點: {self.target}, 初始距離: {self.prev_dist:.2f}m'
        ) if hasattr(self, 'get_logger') else None

        return self._get_obs(), {}

    # ----------------------------------------------------------
    # step(): 每步 Agent 決策後呼叫
    # ----------------------------------------------------------
    def step(self, action):
        # 1. 把 action clip 到安全範圍 (保險用) 
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)

        # 2. 發指令給無人機, 等模擬器更新
        self.ros.send_velocity(*action)
        rclpy.spin_once(self.ros, timeout_sec=0.1)
        self.step_count += 1

        # 3. 讀取新狀態
        obs = self._get_obs()
        pos = self.ros.current_pose.copy()

        # 4. 計算 reward
        reward, terminated = self._compute_reward(action, pos)

        # 5. 超時終止
        truncated = (self.step_count >= self.MAX_STEPS)

        return obs, reward, terminated, truncated, {}

    # ----------------------------------------------------------
    # _compute_reward(): 獎勵函數 (核心設計) 
    # ----------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        """
        獎勵由五項組成: 

        1. 距離縮短獎勵 (r_progress)
           這一步比上一步靠近目標就給正分, 遠離就給負分.
           係數 5.0 讓它成為最主要的學習訊號.

        2. 到達獎勵 (r_arrive)
           距離 < ARRIVE_DIST 時給一次性大獎勵 +100.
           讓 agent 明確知道「到達」是最終目標.

        3. 時間懲罰 (r_time)
           每步固定 -0.1, 鼓勵 agent 盡快到達, 不要磨蹭.

        4. 邊界懲罰 (r_boundary)
           飛出設定範圍就 -50 並終止 episode.

        5. 動作平滑懲罰 (r_smooth)
           對速度指令的大小給小負分, 避免 agent 學出
           「瘋狂油門 + 急煞」的抖動飛法.
        """
        terminated = False
        curr_dist = float(np.linalg.norm(pos - self.target))

        # --- 1. 距離縮短獎勵 ---
        r_progress = 5.0 * (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        # --- 2. 到達獎勵 ---
        r_arrive = 0.0
        if curr_dist < self.ARRIVE_DIST:
            r_arrive = 100.0
            terminated = True

        # --- 3. 時間懲罰 ---
        r_time = -0.1

        # --- 4. 邊界懲罰 ---
        r_boundary = 0.0
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX or
            pos[2] < self.BOUNDARY_Z_MIN
        )
        if out_of_bounds:
            r_boundary = -50.0
            terminated = True

        # --- 5. 動作平滑懲罰 ---
        r_smooth = -0.05 * float(np.linalg.norm(action))

        # --- 加總 ---
        reward = r_progress + r_arrive + r_time + r_boundary + r_smooth
        return reward, terminated

    # ----------------------------------------------------------
    # _get_obs(): 把當前狀態打包成 observation vector
    # ----------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """
        回傳 9 維的觀測向量: 
        [pos_x, pos_y, pos_z, target_x, target_y, target_z, vel_x, vel_y, vel_z]
        """
        obs = np.concatenate([
            self.ros.current_pose,   # 3 維
            self.target,             # 3 維
            self.ros.current_vel,    # 3 維
        ]).astype(np.float32)

        # clip 到 observation_space 範圍內, 避免 SB3 報警告
        obs = np.clip(obs,
                      self.observation_space.low,
                      self.observation_space.high)
        return obs

    def get_logger(self):
        """讓 DroneGymEnv 也能用 ROS logger."""
        return self.ros.get_logger()
