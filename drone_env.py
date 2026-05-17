#!/usr/bin/env python3
"""
drone_env.py
------------
Task B: 隨機目標導航(Random Target Navigation)
將 NSYSU Drone 模擬環境包裝成 Gymnasium 相容的 Gym 環境. 

重置機制說明: 
    /simple_drone/reset(Topic): 控制器邏輯層, 只清除 PID 積分誤差與馬達指令, 
                                  無法改變 Gazebo 物理引擎中的絕對座標. 
    /reset_world(Service):      物理引擎層, 由 gazebo_ros 官方外掛提供, 
                                  呼叫後 Gazebo 核心會強制將所有模型傳送回
                                  URDF/SDF 定義的初始生成座標(Spawn Pose). 
    結論: 強化學習每個 Episode 的完整重置必須呼叫 /reset_world Service. 

參數調整依據: 
    Paper 1: Tan & Karakose(2023), SoftwareX, PPO-based distributed deep RL for drone tracking. 
    Paper 2: Zhang et al.(2024), AirPilot, PPO-based DRL auto-tuned nonlinear PID drone controller. 
    Paper 3: Shen & Huang(2024), Drones, RL in controlling quadrotor UAV flight actions. 

前置安裝: 
    pip install stable-baselines3 gymnasium numpy

使用方式: 
    此檔案不直接執行, 由 train.py 和 test.py 匯入使用. 
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv

import gymnasium as gym
from gymnasium import spaces


# ================================================================
# Part 1: ROS 2 介面層
# 負責和 Gazebo 溝通, 把 ROS topic/service 包成簡單的 get/set
# ================================================================
class DroneROSInterface(Node):
    """
    把 ROS 2 的 publisher/subscriber/service client 包裝成簡單的介面. 
    Gym 環境透過這個物件和模擬器溝通, 不直接碰 ROS API. 

    重置架構: 
        軟重置(Soft Reset): send_velocity(0, 0, 0) -> 停止動作, 維持位置. 
        硬重置(Hard Reset): reset_world() -> 呼叫 /reset_world Service, 
                             強制 Gazebo 將無人機傳送回原點. 
    """

    def __init__(self):
        super().__init__('rl_drone_interface')

        # --- 儲存最新的感測值 ---
        self.current_pose  = np.zeros(3, dtype=np.float32)
        self.current_vel   = np.zeros(3, dtype=np.float32)
        self.pose_received = False

        # --- Publishers: 發指令給無人機 ---
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )
        self.takeoff_pub = self.create_publisher(
            Empty, '/simple_drone/takeoff', 10
        )

        # /simple_drone/reset 只用於軟重置(清除控制器狀態), 
        # 不能改變物理位置, 因此在硬重置流程中不使用. 
        self.soft_reset_pub = self.create_publisher(
            Empty, '/simple_drone/reset', 10
        )

        # --- Service Client: 硬重置物理世界 ---
        # /reset_world 是 gazebo_ros 官方外掛提供的原生 Service. 
        # 呼叫後 Gazebo 核心會強制將所有模型傳送回初始 Spawn Pose(通常是原點). 
        # 這是強化學習 Episode 完整重置唯一可靠的方式. 
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

        # --- Subscribers: 接收無人機狀態 ---
        self.pose_sub = self.create_subscription(
            Pose,  '/simple_drone/gt_pose', self._pose_cb, 10
        )
        self.vel_sub  = self.create_subscription(
            Twist, '/simple_drone/gt_vel',  self._vel_cb,  10
        )

        self.get_logger().info('DroneROSInterface initialized')

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

    def send_velocity(self, vx: float, vy: float, vz: float):
        """發布速度命令到 /simple_drone/cmd_vel. """
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
        硬重置: 呼叫 /reset_world Service, 強制 Gazebo 將無人機傳送回原點. 

        流程: 
            1. 等待 /reset_world Service 可用(最多 5 秒). 
            2. 非同步呼叫 Service, 並 spin 等待回應. 
            3. 回傳是否成功. 

        回傳值: 
            True  -> 重置成功. 
            False -> Service 不可用或呼叫超時. 
        """
        # 等待 Service 上線, 避免在模擬器尚未就緒時呼叫失敗
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_world service not available, skipping hard reset')
            return False

        req    = EmptySrv.Request()
        future = self.reset_world_client.call_async(req)

        # 等待 Service 回應, 最多等 5 秒
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            return True
        else:
            return False

    def takeoff(self):
        """發送起飛指令. """
        self.takeoff_pub.publish(Empty())


# ================================================================
# Part 2: Gym 環境
# 標準 Gymnasium 介面, PPO/SAC 等演算法都能直接使用
# ================================================================
class DroneGymEnv(gym.Env):
    """
    Task B: 隨機目標導航環境. 

    每個 Episode 的流程: 
        1. 呼叫 /reset_world Service -> 無人機傳送回原點(硬重置). 
        2. 等待物理引擎穩定 -> 確認無人機確實在原點. 
        3. 發送 /takeoff -> 無人機起飛. 
        4. 等待無人機達到穩定懸停高度. 
        5. 隨機生成目標點. 
        6. Agent 輸出速度指令, 嘗試飛到目標點. 
        7. 到達 or 超時 or 飛出邊界 -> Episode 結束. 

    MDP 定義: 
        State  (13維): [pos_x, pos_y, pos_z, target_x, target_y, target_z, rel_x, rel_y, rel_z, vel_x, vel_y, vel_z, dist]
        Action (3維): [vx, vy, vz], 範圍 [-MAX_SPEED, MAX_SPEED]
        Reward: 見 _compute_reward() 
        Gamma : 0.99(依據 Paper 2 Table 1 及 Paper 3 Table 1) 
    """

    # 最大速度: 1.5 m/s, 與 Paper 2 NormalizedV 範圍 [-1, 1] 相符. 
    # 測試 0.8:
    MAX_SPEED = 0.8

    # 到達距離閾值 0.4m: 依據 Paper 3 Section 4.2.1, 
    # "when the distance value is less than 40 cm, grant a reward of +100. "
    # 測試 0.25:
    ARRIVE_DIST = 0.4

    # 每個 Episode 最多步數 200 步(20 秒): 
    # Paper 2 Section IV 使用 1000 timesteps per episode(40ms/step = 40s). 
    # 本環境每步 0.1 秒, 200 步 = 20 秒, 足夠完成短距離目標導航. 
    MAX_STEPS = 200

    # 邊界: x/y 最大 15m, z 最大 8m. 
    # 放寬至 15m 以避免訓練初期因探索而頻繁出界. 
    # 測試: 3, 5m
    BOUNDARY_XY    = 3.0
    BOUNDARY_Z_MAX = 5.0

    # 地板邊界設為 -1.0(實際不會觸發): 
    # /reset_world 後無人機在地面, 需等待起飛才開始計算 reward, 
    # 設為負值避免起飛過渡期誤判出界. 
    BOUNDARY_Z_MIN = -1.0

    # 目標點隨機範圍: 依據 Paper 3 Section 4.2 的場景設計, 
    # 目標在 x/y [-5, 5], z [1, 3] 的安全空間內隨機生成. 
    # 修改: 把原本的 +-5.0 改小, 原本的太大很難在 300000 內收斂
    TARGET_LOW  = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    TARGET_HIGH = np.array([ 1.0,  1.0, 3.0], dtype=np.float32)      

    # 等待起飛的最小安全高度: 確認無人機已真正離地才開始 Episode. 
    MIN_HOVER_Z = 0.8

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

        # Observation: [pos, target, rel_pos, vel, dist] 共 13 維. 
        # 狀態設計依據 Paper 2 Section IV: 
        # agent 同時感知自身位置、目標位置與當前速度, 
        # 加入相對位置與距離, 大幅降低神經網路的學習難度
        obs_limit = np.array(
            [20, 20, 10,   # 無人機位置
             20, 20, 10,   # 目標位置
             40, 40, 20,   # 相對位置
             5,  5,  5,    # 速度
             50],          # 絕對距離
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low  = -obs_limit,
            high =  obs_limit,
            shape=(13,),
            dtype=np.float32
        )

        # 內部狀態
        self.target     = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_dist  = None

        # 新增: 用來記錄每回合各項獎勵的累計值
        self.ep_components = {
            'progress': 0.0, 'proximity': 0.0, 'arrive': 0.0, 
            'time': 0.0, 'boundary': 0.0, 'action': 0.0, 'smooth': 0.0
        }

    def reset(self, seed=None, options=None):
        """
        Episode 重置流程: 
            Step 1: 呼叫 /reset_world Service -> 無人機傳送回原點(硬重置). 
            Step 2: 等待物理引擎穩定. 
            Step 3: 發送 /takeoff -> 無人機起飛. 
            Step 4: 等待無人機達到 MIN_HOVER_Z 高度(最多等 10 秒). 
            Step 5: 隨機生成目標點. 
        """
        super().reset(seed=seed)
        self.step_count = 0

        # --- Step 1: 硬重置 ---
        # 呼叫 /reset_world Service, 強制 Gazebo 將無人機傳送回初始 Spawn Pose. 
        # 這是唯一能真正改變物理位置的方式, /simple_drone/reset Topic 只能
        # 清除控制器狀態, 無法移動無人機. 
        self.ros.reset_world()

        # --- Step 2: 等待物理引擎穩定 ---
        # reset_world 後 Gazebo 需要幾個 tick 才能完成世界重置, 
        # 在此期間持續 spin 更新感測器資料. 
        for _ in range(20):
            rclpy.spin_once(self.ros, timeout_sec=0.1)

        # --- Step 3: 發送起飛指令 ---
        self.ros.takeoff()

        # --- Step 4: 等待起飛穩定 ---
        # 持續 spin 直到 z 超過 MIN_HOVER_Z, 或等待超時(10 秒). 
        # 確保無人機真正飛起來才開始 Episode, 避免第一步就觸發地板邊界. 
        for _ in range(100):
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            if self.ros.current_pose[2] > self.MIN_HOVER_Z:
                break

        # --- Step 5: 隨機生成目標點 ---
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        rclpy.spin_once(self.ros, timeout_sec=0.1)

        # 計算初始距離, nan 時給預設值
        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0

        # 新增: 回合重置時, 清空各項獎勵的累計值
        for key in self.ep_components.keys():
            self.ep_components[key] = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """每步執行動作並回傳新狀態. """
        # 1. 把 action clip 到安全範圍
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)

        # 2. 發指令給無人機, 等模擬器更新
        self.ros.send_velocity(*action)
        rclpy.spin_once(self.ros, timeout_sec=0.1)
        self.step_count += 1

        # 3. 讀取新狀態
        obs = self._get_obs()
        pos = self.ros.current_pose.copy()

        # 4. 計算 reward 並更新細項
        reward, terminated = self._compute_reward(action, pos)

        # 5. 超時終止
        truncated = (self.step_count >= self.MAX_STEPS)

        # 新增: 如果回合結束, 把這回合的細項打包進 info 傳出去
        info = {}
        if terminated or truncated:
            info['ep_components'] = self.ep_components.copy()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray, pos: np.ndarray):
        """
        獎勵函數, 由五項組成: 

        1. 距離縮短獎勵(r_progress): 
           依據 Paper 3(Shen & Huang, 2024) Section 4.2.1: 
           "d = Dabs_before - Dabs, the environment uses d as the basis for reward. "
           係數 10.0: 依據 Paper 1(Tan & Karakose, 2023) 實驗, 
           更強的距離訊號有助於加速初期收斂, 防止 agent 陷入局部最優. 

        2. 到達獎勵(r_arrive): 
           依據 Paper 3 Section 4.2.1: "grant a reward of +100. "
           閾值 0.4m 來自 Paper 3 的 40cm 判定標準. 

        3. 時間懲罰(r_time): 
           依據 Paper 3 Section 4.2.1 的 time deduction item. 
           固定值參考 Paper 2(AirPilot) Section IV 的 episode 終止設計, 
           有效防止 agent 在原地磨蹭. 

        4. 邊界懲罰(r_boundary): 
           依據 Paper 3 Section 4.2.1: 碰撞懲罰 -100. 
           本實作使用 -50 作為邊界懲罰(出界但尚未實際碰撞). 
        """
        terminated = False
        curr_dist  = float(np.linalg.norm(pos - self.target))
        if np.isnan(curr_dist):
            curr_dist = 10.0

        # --- 1. 距離縮短獎勵 ---
        # r_progress = 10.0 * (self.prev_dist - curr_dist)
        # 修改: 負距離, 距離越近 reward 越高, 訊號穩定不被抵消
        r_progress = 0.1*-curr_dist
        self.prev_dist = curr_dist

        # --- 2. 到達獎勵 ---
        r_arrive = 0.0
        if curr_dist < self.ARRIVE_DIST:
            r_arrive   = 100.0
            terminated = True

        # --- 3. 時間懲罰 ---
        # 時間懲罰從 -0.5 降到 -0.05: 
        # 目前 200步 * (-0.05) = -10 剛好等於整個 episode 的 reward, 
        # 導致 r_progress 的訊號完全被時間懲罰蓋過, agent 分不清楚飛近目標有沒有用. 
        # 降低時間懲罰讓距離縮短獎勵成為主要學習訊號. 
        r_time = -0.05

        # --- 4. 邊界懲罰 ---
        r_boundary = 0.0
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )
        if out_of_bounds:
            # 活著 200 步大約會被扣 40(距離) + 10(時間) = 50 分。
            # 邊界懲罰必須設為 -200，讓模型知道撞牆自殺的下場比活著找目標慘非常多。
            r_boundary = -200.0
            terminated = True

        # 新增: 
        # --- 5. 蘿蔔引導(常駐正回饋) ---
        # 只要待在目標半徑 2 公尺內, 每步都給微小加分, 抵銷時間懲罰
        r_proximity = 0.1 if curr_dist < 2.0 else 0.0

        # 新增: 將各項獎勵累加到內部記錄器中
        self.ep_components['progress']  += r_progress
        self.ep_components['proximity'] += r_proximity
        self.ep_components['arrive']    += r_arrive
        self.ep_components['time']      += r_time
        self.ep_components['boundary']  += r_boundary

        reward = r_progress + r_arrive + r_time + r_boundary + r_proximity
        return reward, terminated

    def _get_obs(self) -> np.ndarray:
        """
        回傳 13 維觀測向量: 
        [pos_x, pos_y, pos_z, target_x, target_y, target_z, 
         rel_x, rel_y, rel_z, vel_x, vel_y, vel_z, distance]

        狀態設計依據 Paper 2(AirPilot) Section IV: 
        agent 同時感知自身絕對位置、目標位置與當前速度. 
        速度資訊對應 Paper 2 中的 dPE/dt(位置誤差微分), 有助於 agent 判斷動量方向. 
        加入相對位置與距離, 讓網路具備方向感. 
        """
        pos = self.ros.current_pose
        vel = self.ros.current_vel
        target = self.target
        
        rel_pos = target - pos
        dist = float(np.linalg.norm(rel_pos))

        obs = np.concatenate([
            pos,
            target,
            rel_pos,
            vel,
            [dist]
        ]).astype(np.float32)

        # nan_to_num 防止感測器資料異常造成神經網路崩潰
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def get_logger(self):
        """讓 DroneGymEnv 也能使用 ROS logger. """
        return self.ros.get_logger()