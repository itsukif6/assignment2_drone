#!/usr/bin/env python3
"""
rl_fly_to_target.py
-------------------
用 PPO 訓練無人機飛到目標點的最小可行範例。

前置安裝:
    pip install stable-baselines3 gymnasium numpy

前置條件:
    1. Gazebo 模擬已經啟動 (launch_drone)
    2. 無人機已起飛

使用方式:
    python3 rl_fly_to_target.py --mode train   # 訓練模式
    python3 rl_fly_to_target.py --mode test    # 測試訓練好的模型
"""

import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


# ================================================================
# ROS 2 介面: 負責和 Gazebo 溝通
# ================================================================
class DroneROSInterface(Node):
    """把 ROS 2 的 pub/sub 包裝成簡單的 get/set 介面。"""

    def __init__(self):
        super().__init__('rl_drone_interface')

        self.current_pose = np.zeros(3)  # [x, y, z]
        self.current_vel = np.zeros(3)   # [vx, vy, vz]

        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )
        self.takeoff_pub = self.create_publisher(
            Empty, '/simple_drone/takeoff', 10
        )
        self.reset_pub = self.create_publisher(
            Empty, '/simple_drone/reset', 10
        )
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self._pose_cb, 10
        )

    def _pose_cb(self, msg: Pose):
        new_pose = np.array([msg.position.x, msg.position.y, msg.position.z])
        # 簡易速度估計 (差分法)
        self.current_vel = (new_pose - self.current_pose) * 10  # 約 10Hz
        self.current_pose = new_pose

    def send_velocity(self, vx, vy, vz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        self.cmd_vel_pub.publish(msg)

    def reset_drone(self):
        self.reset_pub.publish(Empty())
        # 稍等一下後起飛
        rclpy.spin_once(self, timeout_sec=0.5)
        self.takeoff_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=1.0)


# ================================================================
# Gym Environment: 把 RL 標準介面包起來
# ================================================================
class DroneGymEnv(gym.Env):
    """讓無人機模擬變成 Gym 相容的環境。"""

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        # ---------- 定義 Action Space ----------
        # 速度 vx, vy, vz 都在 [-1, 1] m/s 範圍
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ---------- 定義 Observation Space ----------
        # [drone_x, drone_y, drone_z,
        #  target_x, target_y, target_z,
        #  vx, vy, vz]
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(9,), dtype=np.float32
        )

        self.target = np.array([5.0, 3.0, 2.0])  # 目標點
        self.max_steps = 200
        self.step_count = 0

    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.reset_drone()
        self.step_count = 0

        # 可以讓目標點隨機生成,提升泛化能力
        self.target = np.random.uniform(
            low=[-3, -3, 1], high=[5, 5, 3]
        ).astype(np.float32)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------
    def step(self, action):
        # 1. 執行動作
        self.ros.send_velocity(*action)
        rclpy.spin_once(self.ros, timeout_sec=0.1)

        self.step_count += 1

        # 2. 取得新狀態
        obs = self._get_obs()

        # 3. 計算 reward
        distance = np.linalg.norm(self.ros.current_pose - self.target)
        reward = -distance                          # 越接近目標越好
        reward -= 0.01 * np.linalg.norm(action)     # 鼓勵動作小 (省電)

        # 4. 判斷結束條件
        terminated = False
        if distance < 0.3:
            reward += 100.0                         # 到達大獎
            terminated = True
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------
    def _get_obs(self):
        """把當前狀態包成 observation vector。"""
        return np.concatenate([
            self.ros.current_pose,
            self.target,
            self.ros.current_vel,
        ]).astype(np.float32)


# ================================================================
# 訓練 & 測試
# ================================================================
def train(env):
    """用 PPO 演算法訓練。"""
    print('🎓 開始訓練...')
    model = PPO(
        'MlpPolicy', env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        tensorboard_log='./ppo_drone_logs/',
    )
    model.learn(total_timesteps=50_000)
    model.save('ppo_drone')
    print('✅ 訓練完成,模型已存至 ppo_drone.zip')


def test(env):
    """載入訓練好的模型並測試。"""
    print('🚀 載入模型測試...')
    model = PPO.load('ppo_drone')
    obs, _ = env.reset()
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f'Step {step}: pos={obs[:3]}, reward={reward:.2f}')
        if terminated or truncated:
            break
    print(f'✅ 測試結束,總 reward = {total_reward:.2f}')


# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    try:
        if args.mode == 'train':
            train(env)
        else:
            test(env)
    finally:
        ros_interface.send_velocity(0, 0, 0)
        ros_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
