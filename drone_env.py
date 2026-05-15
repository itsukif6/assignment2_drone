#!/usr/bin/env python3
"""
drone_env.py
------------
Task B: 隨機目標導航 ( Random Target Navigation )
將 NSYSU Drone 模擬環境包裝成 Gymnasium 相容的 Gym 環境. 

前置安裝: 
    pip install stable-baselines3 gymnasium numpy

使用方式: 
    此檔案不直接執行, 由 train.py 和 test.py 匯入使用. 

參考論文: 
    Paper 1: A new approach for drone tracking with drone using Proximal Policy Optimization based distributed deep reinforcement learning
    Paper 2: AirPilot Interpretable PPO-based DRL Auto Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights
    Paper 3: Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, TwistStamped
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv  # 呼叫原生服務

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
        # self.reset_pub = self.create_publisher(
        #     Empty, '/simple_drone/reset', 10
        # )
        self.reset_client = self.create_client(EmptySrv, '/reset_simulation')

        # --- Subscribers: 接收無人機狀態 ---
        # gt_pose: ground truth 位置 ( 最準確, 直接來自 Gazebo ) 
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self._pose_cb, 10
        )
        # gt_vel: ground truth 速度 ( 比差分估計穩定很多 ) 
        self.vel_sub = self.create_subscription(
            Twist, '/simple_drone/gt_vel', self._vel_cb, 10
        )

        self.get_logger().info('DroneROSInterface initialized. ')

    # ---- Callbacks: 每次收到訊息就自動更新 ----

    def _pose_cb(self, msg: Pose):
        """收到位置訊息, 更新 current_pose. """
        self.current_pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z],
            dtype=np.float32
        )
        self.pose_received = True

    def _vel_cb(self, msg: Twist):
        """收到速度訊息, 更新 current_vel. """
        self.current_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z],
            dtype=np.float32
        )

    # ---- 發指令的方法 ----

    def send_velocity(self, vx: float, vy: float, vz: float):
        """發布速度命令到 /simple_drone/cmd_vel. """
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
        # self.reset_pub.publish(Empty())
        self.reset_client = self.create_client(EmptySrv, '/reset_simulation')
        rclpy.spin_once(self, timeout_sec=2.0)   # 等重置生效
        self.takeoff_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=3.0)   # 等起飛穩定


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
        - 到達 or 超時 or 飛出邊界 -> 結束

    MDP 定義: 
        State  ( 9維 ): [pos_x, pos_y, pos_z, target_x, target_y, target_z, vel_x, vel_y, vel_z]
        Action ( 3維 ): [vx, vy, vz], 範圍 [-MAX_SPEED, MAX_SPEED]
        Reward: 見 _compute_reward( )
        Gamma : 在 train.py 中設定 ( 預設 0.99 ) 
    """

    # ---- 環境常數 ( 修改這裡來調整任務難度 ) ----
    MAX_SPEED      = 1.5    # 動作空間上下限, 單位 m/s
    ARRIVE_DIST    = 0.3    # 距離目標多近算到達, 單位 m
    MAX_STEPS      = 300    # 每個 episode 最多幾步 ( 1步 = 0.1秒 -> 30秒上限 ) 
    BOUNDARY_XY    = 15.0   # x/y 方向的邊界, 超過就終止
    BOUNDARY_Z_MAX = 8.0    # 最高飛多高
    BOUNDARY_Z_MIN = -1.0   # 最低

    # 目標點隨機範圍
    TARGET_LOW  = np.array([-3.0, -3.0, 1.0], dtype=np.float32)
    TARGET_HIGH = np.array([ 3.0,  3.0, 3.0], dtype=np.float32)

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
        self.prev_dist   = None   # 上一步的距離, 用來算距離縮短量

    # ----------------------------------------------------------
    # reset( ): 每個 episode 開始時呼叫
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_retries = 10
        retry_count = 0
        reset_success = False

        while retry_count < max_retries and not reset_success:
            
            # --- 呼叫 Gazebo 原生重置服務 ---
            if self.ros.reset_client.wait_for_service(timeout_sec=1.0):
                req = EmptySrv.Request()
                future = self.ros.reset_client.call_async(req)
                rclpy.spin_until_future_complete(self.ros, future, timeout_sec=2.0)
            else:
                self.get_logger().error('Gazebo reset service not available!')
            
            # 給物理引擎一點時間更新座標
            for _ in range(50):
                rclpy.spin_once(self.ros, timeout_sec=0.1)
                pos = self.ros.current_pose
                
                # 檢查 X, Y, Z 是否都回到原點附近
                if abs(pos[0]) < 0.5 and abs(pos[1]) < 0.5 and pos[2] < 0.5:
                    reset_success = True
                    break
            
            if not reset_success:
                retry_count += 1
                self.get_logger().warn(f'Reset timeout, drone is stuck at {pos}. Retrying... ({retry_count}/{max_retries})')

        if not reset_success:
            raise RuntimeError("Fatal Error: Gazebo failed to reset the drone. Please restart the simulator.")
        
        # --- 發送起飛指令 ---
        self.ros.takeoff_pub.publish(Empty())
        
        # 確保起飛達到安全高度 (Z 超過 0.8m)
        takeoff_success = False
        for _ in range(100):
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            if self.ros.current_pose[2] > 0.8:
                takeoff_success = True
                break
                
        if not takeoff_success:
            self.get_logger().warn('Warning: Drone might not have reached safe takeoff height.')

        # 初始化內部狀態
        self.step_count = 0
        self.target = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        rclpy.spin_once(self.ros, timeout_sec=0.3)
        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))

        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0
            
        return self._get_obs(), {}

    # ----------------------------------------------------------
    # step( ): 每步 Agent 決策後呼叫
    # ----------------------------------------------------------
    def step(self, action):
        # 1. 把 action clip 到安全範圍 ( 保險用 ) 
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
    # _compute_reward( ): 獎勵函數 ( 核心設計 ) 
    # ----------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        terminated = False
        curr_dist = float(np.linalg.norm(pos - self.target))

        # --- 基礎變數與進步獎勵 ---
        r_progress = 5.0 * (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        # --- 策略 2: 胡蘿蔔引導 ( Dense Positive Reward ) ---
        # 如果無人機進入目標的引力圈 ( 例如 2 公尺內 ) , 給予微小的常駐正回饋. 
        # 讓它覺得待在目標附近是一件值得的事, 誘使它不小心撞上目標拿大獎. 
        r_proximity = 0.2 if curr_dist < 2.0 else 0.0

        # --- 到達獎勵 ( 最大蘿蔔 ) ---
        r_arrive = 0.0
        if curr_dist < self.ARRIVE_DIST:
            r_arrive = 100.0
            terminated = True

        # --- 策略 1: 生存與死亡的數學題 ( 打碎自殺的誘因 ) ---
        r_time = -0.1
        # 一回合最多 300 步, 光陰耗盡最多被扣 30 分. 
        # 撞牆懲罰必須大於 -30, 設定為 -50 分. 
        # 這樣 Agent 就會發現: 活到超時 ( 最慘 -30 ) 也比 直接撞天花板 ( -50 ) 好. 
        
        r_boundary = 0.0
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX or
            pos[2] < self.BOUNDARY_Z_MIN
        )
        if out_of_bounds:
            r_boundary = -50.0  # 修改死亡懲罰, 超越時間懲罰的極限
            terminated = True
        
        # 軟性高度預警 ( 超過 6 公尺就開始給予輕微壓力, 提早產生向下飛的梯度 ) 
        elif pos[2] > 6.0:
            r_boundary = -0.5 * (pos[2] - 6.0)

        # --- 策略 3: 動作意圖的直接懲罰 ( Action Penalty ) ---
        r_action = 0.0
        # 慣性煞車: 當高度大於 6.5 公尺, 且神經網路還輸出向上的速度指令時, 直接重罰動作
        if pos[2] > 6.5 and action[2] > 0:
            r_action = -5.0 * action[2]  # 油門踩越深, 扣分越重

        # --- 動作平滑懲罰 ---
        r_smooth = -0.05 * float(np.linalg.norm(action))

        # --- 總和 ---
        reward = r_progress + r_proximity + r_arrive + r_time + r_boundary + r_action + r_smooth
        return reward, terminated

    # ----------------------------------------------------------
    # _get_obs( ): 把當前狀態打包成 observation vector
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

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def get_logger(self):
        """讓 DroneGymEnv 也能用 ROS logger. """
        return self.ros.get_logger()