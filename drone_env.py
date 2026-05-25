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
- 應用部分：觀測空間 (Observation Space)、懸停判定 (Hovering/Settling) 與超時中止條件。
- 具體實作：
  1. 狀態輸入採用相對位置誤差 (Position Error) 與速度 (Velocity)。
  2. 導入「減速帶機制」與「穩定懸停步數 (HOVER_STEPS)」，以減少超調量 (Overshoot)。
  3. 設定最大步數強制終止與嚴重機身傾角懲罰，以鼓勵能源效率並避免危險飛行。

[Reference 3] 
Shen, S.-E., & Huang, Y.-C. (2024). "Application of Reinforcement Learning 
in Controlling Quadrotor UAV Flight Actions." Drones.
- 應用部分：PPO 模型超參數、連續回合方法 (Continuous Round Method, CRM) 之獎勵機制。
- 具體實作：
  1. 採用文獻測試驗證之最佳 PPO 參數：learning_rate=3e-4, n_steps=2048, 
     batch_size=64, gamma=0.99, gae_lambda=0.95, vf_coef=0.5。
  2. 獎勵機制設計：導入與目標中心之距離差值 (+d) 作為進度獎勵、
     每步微小時間耗損懲罰 (-Tm)，以及碰撞邊界/障礙物之極端懲罰 (-100)。
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

        self.get_logger().info('DroneROSInterface initialize finish.')

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
            self.get_logger().warn('/reset_world failed, skipping...')
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
    BOUNDARY_Z_MAX = 6.5 # 放寬最大高度限制，配合 Level 7 的 5.5m 目標
    BOUNDARY_Z_MIN = 0.0

    # 定義 7 個 Curriculum Levels (課程學習難度階層)
    CURRICULUM_LEVELS = {
        1: {'target_low': np.array([-1.0,  -1.0,  1.5], dtype=np.float32), 'target_high': np.array([ 1.0,  1.0,  2.5], dtype=np.float32), 'boundary_xy':  3.0, 'arrive_dist': 1.50, 'hover_steps':  3, 'hover_max_spd': 0.80},
        2: {'target_low': np.array([-2.5,  -2.5,  1.5], dtype=np.float32), 'target_high': np.array([ 2.5,  2.5,  3.0], dtype=np.float32), 'boundary_xy':  4.0, 'arrive_dist': 1.20, 'hover_steps':  7, 'hover_max_spd': 0.60},
        3: {'target_low': np.array([-4.0,  -4.0,  1.5], dtype=np.float32), 'target_high': np.array([ 4.0,  4.0,  3.5], dtype=np.float32), 'boundary_xy':  5.0, 'arrive_dist': 1.00, 'hover_steps': 11, 'hover_max_spd': 0.35},
        4: {'target_low': np.array([-5.5,  -5.5,  1.5], dtype=np.float32), 'target_high': np.array([ 5.5,  5.5,  4.0], dtype=np.float32), 'boundary_xy':  6.5, 'arrive_dist': 0.90, 'hover_steps': 15, 'hover_max_spd': 0.31},
        5: {'target_low': np.array([-7.0,  -7.0,  1.5], dtype=np.float32), 'target_high': np.array([ 7.0,  7.0,  4.5], dtype=np.float32), 'boundary_xy':  8.0, 'arrive_dist': 0.80, 'hover_steps': 19, 'hover_max_spd': 0.27},
        6: {'target_low': np.array([-8.5,  -8.5,  1.5], dtype=np.float32), 'target_high': np.array([ 8.5,  8.5,  5.0], dtype=np.float32), 'boundary_xy':  9.5, 'arrive_dist': 0.70, 'hover_steps': 23, 'hover_max_spd': 0.23},
        7: {'target_low': np.array([-10.0, -10.0, 1.5], dtype=np.float32), 'target_high': np.array([10.0, 10.0,  5.5], dtype=np.float32), 'boundary_xy': 11.0, 'arrive_dist': 0.60, 'hover_steps': 27, 'hover_max_spd': 0.20},
    }

    MAX_CURRICULUM_LEVEL = 7
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

        # 狀態空間：包含正規化後的相對位置誤差與無人機當前速度
        # [參考: AirPilot 論文] 使用 PE (Position Error, 位置誤差) 與 dPE/dt (速度) 作為神經網路的輸入狀態。
        self.observation_space = spaces.Box(
            low  = -5.0,
            high =  5.0,
            shape=(6,),
            dtype=np.float32
        )

        # [參考: AirPilot 論文] 要求無人機在目標位置穩定停留一段時間 (論文中為 50 timesteps)，
        # 這裡實作了 hover_steps 來確保抵達的穩定性，而非只是擦邊掠過。
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
        """
        重置環境：包含停止馬達、硬重置模擬器、重新起飛與姿態穩定。
        目的：確保每一次強化學習回合 (Episode) 開始時，無人機都處於相同的初始狀態，
              消除上一回合殘留的動量 (Momentum) 或物理模擬器重置時的突波干擾。
        """
        # 初始化 Gymnasium 內部亂數種子與環境步數計數器
        super().reset(seed=seed)
        self.step_count = 0

        # ==========================================
        # 階段一：安全降落與物理環境重置
        # ==========================================
        # 強制清空速度指令，並發布降落指令。
        # 避免無人機在帶著高速移動的狀態下直接重置世界，這在 Gazebo 中容易導致物理碰撞或模型噴飛
        self.ros.send_velocity(0.0, 0.0, 0.0)
        self.ros.land_pub.publish(Empty())
        time.sleep(0.5)

        # 呼叫 ROS 服務，硬重置 Gazebo 世界 (包含無人機位置、物理狀態全數歸零)
        self.ros.reset_world()

        # 重置後再送一次降落指令，確保飛行控制器狀態機 (State Machine) 切換至 Grounded (著陸)
        self.ros.land_pub.publish(Empty())
        time.sleep(1.5)

        # ==========================================
        # 階段二：強健式起飛 (Robust Takeoff)
        # ==========================================
        # 模擬器中的 ROS 訊息偶爾會遺失，因此設定最多 5 次的起飛嘗試迴圈
        for attempt in range(5):
            self.ros.takeoff_pub.publish(Empty())
            start_ns = self.ros.get_clock().now().nanoseconds
            
            # 使用 ROS 時鐘阻塞等待 0.5 秒 (5e8 奈秒)，並持續處理回調函數更新感測器資料
            while (self.ros.get_clock().now().nanoseconds - start_ns) < 5e8:
                rclpy.spin_once(self.ros, timeout_sec=0.01)
                
            # 檢查 Z 軸高度是否大於 0.2 公尺，若超過則判定起飛成功並跳出嘗試迴圈
            if self.ros.current_pose[2] > 0.2:
                break

        # 確保起飛後，將無人機底層控制模式切換為「速度控制模式 (Velocity Mode)」
        vel_mode_msg = Bool()
        vel_mode_msg.data = True
        self.ros.vel_mode_pub.publish(vel_mode_msg)
        
        # 發布軟重置 (Soft Reset) 清除底層 PID 控制器的累積誤差 (Integral Windup)
        self.ros.soft_reset_pub.publish(Empty())

        # ==========================================
        # 階段三：空間位置與姿態穩定 (Position Stabilization)
        # ==========================================
        stable_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            pos = self.ros.current_pose
            
            # 確保無人機回到世界中心原點附近 (X, Y 誤差小於 0.5m) 且高度達到安全工作高度 (Z > 0.4m)
            if abs(pos[0]) < 0.5 and abs(pos[1]) < 0.5 and pos[2] > 0.4:
                break
                
            # 設定 3 秒超時機制 (Timeout)，防止無人機因底層錯誤卡在無窮迴圈
            if time.time() - stable_start > 3.0:
                self.ros.get_logger().warn('[WARN] Reset stable stage timeout!')
                break
                
        # 再次清除 PID 積分器誤差，準備進入靜止狀態
        self.ros.soft_reset_pub.publish(Empty())
        time.sleep(0.3)

        # ==========================================
        # 階段四：動態速度穩定 (Velocity Stabilization)
        # ==========================================
        # 確認無人機不僅位置正確，且物理上「完全靜止」，避免帶著初速度開始新回合
        for _ in range(10):
            rclpy.spin_once(self.ros, timeout_sec=0.02)
            vel_norm = float(np.linalg.norm(self.ros.current_vel)) # 計算三軸總速度向量大小
            
            # 速度夠慢 (< 0.5 m/s) 且維持在空中，則判定完全穩定
            if vel_norm < 0.5 and self.ros.current_pose[2] > 0.4:
                break
            # 若尚未靜止，持續發送零速度指令強制煞車
            self.ros.send_velocity(0.0, 0.0, 0.0)
            self.ros.soft_reset_pub.publish(Empty())

        # ==========================================
        # 階段五：新回合任務初始化
        # ==========================================
        # 依據當前的課程學習難度 (Curriculum Level)，在指定範圍內隨機生成新目標座標
        self.target = self.np_random.uniform(
            low=self.TARGET_LOW, high=self.TARGET_HIGH
        ).astype(np.float32)

        # 確認送出靜止指令並更新最後一次感測器狀態
        self.ros.send_velocity(0.0, 0.0, 0.0)
        rclpy.spin_once(self.ros, timeout_sec=0.1)

        # 計算初始時刻無人機與目標之間的歐幾里得距離 (Euclidean Distance)
        self.prev_dist = float(np.linalg.norm(self.ros.current_pose - self.target))
        
        # 防呆機制：若物理引擎剛重置時回傳 NaN (Not a Number)，給予預設值以防程式崩潰
        if np.isnan(self.prev_dist):
            self.prev_dist = 5.0
        
        # 紀錄回合初始距離 (initial_dist)，後續可用於獎勵函數的進度歸一化 (Normalization)
        self.initial_dist = self.prev_dist
        
        # 重置環境內部計數器與 TensorBoard 訓練紀錄變數
        self.hover_steps = 0
        for key in self.ep_components.keys():
            self.ep_components[key] = 0.0

        # 回傳環境初始狀態矩陣 (Observation) 與空的 info 字典 (符合 Gymnasium API 規範)
        return self._get_obs(), {}

    def step(self, action):
        """
        推進環境步數，執行時間同步控制迴圈。
        確保動作能穩定執行指定的時間步長。
        """
        action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        real_velocity = action * 0.8

        # [時間同步與底層致動保證機制 (Time-Synchronization & Actuation Guarantee)]
        # 1. 解決異步通訊與看門狗陷阱：
        #    在 ROS 2 異步架構下，若僅單次發布指令並依賴 `spin_once`，會因感測器回調提早觸發，使時間步長縮水；且單一指令極易觸發底層飛控的看門狗機制 (Watchdog)，導致無人機中途被強制煞停。
        # 2. 維護馬可夫決策過程 (MDP) 假設：
        #    上述的執行時間與動作結果隨機性，會徹底破壞強化學習的 MDP 核心假設（相同動作導致無法預測的位移與獎勵），使 Critic 網路無法評估價值，PPO 梯度互相抵消而無法收斂。
        # 3. 強制時間阻塞與高頻串流 (Time-Blocking & High-Frequency Streaming)：
        #    改用 while 迴圈嚴格比對模擬器時鐘，強制將時間步長鎖定為物理時間 0.1 秒。
        #    同時在 0.1 秒內以 100Hz 高頻持續發布指令來「餵飽」看門狗，確保無人機不間斷致動。
        #    此舉成功將環境轉換為具備「確定性轉移 (Deterministic Transition)」的標準 MDP 模型，是模型成功收斂的絕對關鍵。
        start_ns = self.ros.get_clock().now().nanoseconds
        while (self.ros.get_clock().now().nanoseconds - start_ns) < 1e8: # 強制時間阻塞：確保物理模擬器執行滿 0.1 秒
            self.ros.send_velocity(*real_velocity)
            rclpy.spin_once(self.ros, timeout_sec=0.01)
        self.step_count += 1

        obs = self._get_obs()
        pos = self.ros.current_pose.copy()

        reward, terminated = self._compute_reward(action, pos)
        truncated = (self.step_count >= self.MAX_STEPS)

        # 中止條件 1：超時懲罰設計
        # [參考: Shen 等人 (2024) 論文] CRM 機制中的時間扣分項目 (Time deduction item) Tm，
        # 避免無人機在環境中無意義地遊蕩。
        # [參考: AirPilot 論文] 設定最大步數 (論文為 1000 timesteps) 強制終止以鼓勵尋找最短路徑與節能。
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
        # [參考: Shen 等人 (2024) 論文] 每個 step 給予微小的時間耗損懲罰 (-Tm)，鼓勵快速完成。
        r_time = -0.2 
        
        # 進度獎勵：用 initial_dist 正規化，讓各 level 的總進度上限一致（約 40）
        # [參考: Shen 等人 (2024) 論文] CRM 獎勵設計中的「與目標中心距離差值 (+d)」。
        # 如果無人機靠近目標 (delta > 0) 就給予正獎勵，遠離則給予負獎勵。
        delta = self.prev_dist - curr_dist
        r_progress = 40.0 * delta / max(self.initial_dist, 1.0)
        self.prev_dist = curr_dist

        # 後面用到的參數初始化
        r_proximity = 0.0
        r_decel = 0.0
        r_action = 0.0
        r_boundary = 0.0
        r_arrive   = 0.0

        # 動作懲罰 (Action Penalty)：計算動作向量的平方和 (Squared L2 Norm) 並給予負回饋權重 (-0.01)
        # 目的：抑制神經網路輸出過大或劇烈震盪的控制指令，引導無人機學習平滑且節省推力 (節能) 的飛行策略
        # [參考: AirPilot 論文] 參考其核心目標後衍生之實務工程技術：
        # 1. 控制平滑度與節能目標：
        #    - AirPilot 論文明確指出，無人機最佳化的目標包含了「能源效率 (energy 
        #      efficiency)」以及「平滑收斂 (smoother convergence)」。
        #    - 強調必須「最小化震盪 (minimize oscillations)」以防止無人機出現不穩定
        #      的控制行為。此 L2 正規化懲罰能引導神經網路避免輸出過大或劇烈震盪的控制
        #      指令，學習平滑且節省推力的飛行策略。
        # 2. 縮小模擬與現實的差距 (Sim-to-Real Gap)：
        #    - AirPilot 論文在實際飛行測試中，觀察到無人機接近目標時容易出現「抖動與
        #      過衝 (jittering and overshoot)」現象。
        #    - 在強化學習的實務工程中，對神經網路的動作輸出取平方和進行負回饋懲罰，是
        #      為了有效抑制極端或高頻切換的控制指令。這能促使模型學習給出平滑的推力，
        #      從而降低將模擬環境中訓練出的策略部署至真實無人機時發生失控的風險。
        r_action = -0.01 * float(np.sum(np.square(action)))

        # 減速帶機制：確保無人機在靠近目標時降低速度以符合懸停條件
        # [參考: AirPilot 論文] 鼓勵減少超調量 (overshoot) 與安定時間 (settling time)，
        # 這裡設計了減速帶機制，懲罰在靠近目標時速度過快的行為。
        decel_zone = self.ARRIVE_DIST * 2.0
        speed_thresh = self.HOVER_MAX_SPEED * 1.2
        if self.ARRIVE_DIST <= curr_dist < decel_zone:
            excess_speed = max(0.0, curr_speed - speed_thresh)
            proximity_ratio = (decel_zone - curr_dist) / decel_zone
            r_decel = -10.0 * excess_speed * proximity_ratio

        # 中止條件 2：墜機判定
        # [參考: AirPilot 論文] 對於無人機出現過大機身角度 (危險動作) 給予嚴厲懲罰並終止。
        tilt_angle    = self._get_tilt_angle()
        is_crashed    = tilt_angle > (np.pi / 3.0)
        
        # 中止條件 3：邊界判定
        out_of_bounds = (
            abs(pos[0]) > self.BOUNDARY_XY or
            abs(pos[1]) > self.BOUNDARY_XY or
            pos[2] > self.BOUNDARY_Z_MAX   or
            pos[2] < self.BOUNDARY_Z_MIN
        )

        if out_of_bounds or is_crashed:
            # [參考: Shen 等人 (2024) 論文] CRM 邊界條件：撞擊環境障礙物給予 -100 的懲罰並終止回合。
            r_boundary = -100.0
            terminated = True
            self.hover_steps = 0
        else:
            if curr_dist < self.ARRIVE_DIST:
                if curr_speed < self.HOVER_MAX_SPEED:
                    self.hover_steps += 1
                    # 穩定的每步獎勵：距離越近、速度越慢獎勵越高
                    # 純 shape reward，無固定項，避免累積值超過 r_arrive
                    # 每步上限 ~6，level7 最多累積 27×6=162 < r_arrive=200
                    dist_ratio  = (self.ARRIVE_DIST - curr_dist) / self.ARRIVE_DIST  # 0~1
                    speed_ratio = curr_speed / self.HOVER_MAX_SPEED                  # 0~1
                    r_proximity = 6.0 * dist_ratio - 4.0 * speed_ratio
                    
                    # 中止條件 4：成功抵達目標
                    # [參考: AirPilot 論文] 必須在目標區域內穩定停留指定的 timesteps (此處為 HOVER_STEPS)。
                    if self.hover_steps >= self.HOVER_STEPS:
                        # 成功抵達/懸停目標獎勵
                        # [參考: Shen 等人 (2024) 論文] 成功穿越目標 (Cross target)，給予單次大額獎勵 (+100，此處放大為 +200)。
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
        """
        觀測狀態獲取與正規化 (Observation Extraction & Normalization)
        """
        # 1. 狀態特徵選取 (位置誤差與速度)：
        #    - [參考: AirPilot 論文] 該研究的 PPO 策略網路 (Policy Network) 明確定義了
        #      輸入狀態應包含：當前的位置誤差 (PositionError，即 target - pos) 以及位置
        #      誤差的導數 (即當前速度 Velocity) [cite: 643]。這對應了程式碼中提取 
        #      dx_w, dy_w, dz_w 以及 vel 的設計。
        # 2. 世界坐標轉機體坐標 (Ego-centric Transformation based on Yaw)：
        #    - [參考: Tan & Karaköse (2023) 論文] 該論文在無人機追蹤控制中指出，主要調整
        #      的變數為無人機的 x, y, z 位置與「偏航角 (yaw angle)」 [cite: 257]。
        #    - 在實務上，將相對位置誤差透過 yaw 角旋轉至「機體坐標系 (Body Frame)」，能
        #      讓神經網路學習到與當前朝向無關的相對控制策略 (例如：目標永遠在「我的右前方」)，
        #      這大幅提升了強化學習模型的泛化能力與收斂速度。
        # 3. 動態範圍正規化與數值裁剪 (Normalization & Clipping)：
        #    - [參考: AirPilot 論文] 為了避免極端數值導致神經網路產生不穩定的行為 (erratic 
        #      behaviors)，該論文特別強調了對數值（如速度）進行正規化處理 (Normalization) 
        #      的重要性，將數值限制在特定區間 (如 [-1, 1]) [cite: 645]。
        #    - 程式碼中將相對距離除以最大邊界 (scale_xy, scale_z)，將速度除以最大速度
        #      (MAX_SPEED)，並最終使用 np.clip 限制在 [-2.0, 2.0] 之間，完全符合此
        #      維持神經網路輸入穩定性的學理要求。
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

        # 動態範圍正規化
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