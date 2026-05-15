#!/usr/bin/env python3
"""
train.py
--------
用 PPO 演算法訓練無人機隨機目標導航 (Task B) .

使用方式: 
    python3 train.py

訓練完成後會產生: 
    ppo_drone.zip          → 訓練好的模型
    logs/training_curve.png → reward 曲線圖
    logs/rewards.csv        → 原始數據

參考論文:
    Paper 1: A new approach for drone tracking with drone using Proximal Policy Optimization based distributed deep reinforcement learning
    Paper 2: AirPilot Interpretable PPO-based DRL Auto Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights
    Paper 3: Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 無頭模式, 不需要螢幕
import matplotlib.pyplot as plt

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from drone_env import DroneROSInterface, DroneGymEnv


# ================================================================
# 自訂 Callback: 每個 episode 結束時記錄 reward
# ================================================================
class RewardLoggerCallback(BaseCallback):
    """
    繼承 SB3 的 BaseCallback, 在訓練過程中收集每個 episode 的 reward.
    訓練結束後呼叫 save_curve() 存成 CSV 和圖片.
    """

    def __init__(self, save_dir: str = 'logs', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.episode_rewards = []       # 每個 episode 的累計 reward
        self._current_ep_reward = 0.0   # 當前 episode 的累計 reward

    def _on_step(self) -> bool:
        """每步都會被呼叫."""
        # 累計這一步的 reward
        self._current_ep_reward += self.locals['rewards'][0]

        # 如果這個 episode 結束了 (done = terminated or truncated) 
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            # 每 10 個 episode 印一次進度
            ep = len(self.episode_rewards)
            if ep % 10 == 0:
                recent_mean = np.mean(self.episode_rewards[-10:])
                print(f'  Episode {ep:4d} | 10 round mean reward: {recent_mean:.2f}')

        return True   # 回傳 True 代表繼續訓練

    def save_curve(self):
        """訓練結束後呼叫, 存 CSV 和訓練曲線圖."""
        # --- 存 CSV ---
        csv_path = os.path.join(self.save_dir, 'rewards.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])
        print(f'raw data saved: {csv_path}')

        # --- 畫訓練曲線 ---
        episodes = list(range(1, len(self.episode_rewards) + 1))
        rewards  = self.episode_rewards

        # 移動平均 (每 20 回合) 讓曲線更平滑好看
        window = 20
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards,   color='lightblue', alpha=0.5, label='round reward')
        ax.plot(episodes, smoothed,  color='steelblue', linewidth=2, label=f'move mean: ({window} rounds) ')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve — Task B Random Target Navigation', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f'training cruve saved: {img_path}')


# ================================================================
# 主訓練流程
# ================================================================
def main():
    print('=' * 55)
    print('  PPO train: Task B - Random Target Navigation')
    print('=' * 55)

    # --- 初始化 ROS 2 ---
    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    # --- 等待第一筆位置資料 (確保模擬器已啟動) ---
    print('waiting Gazebo place data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Data received, start training')

    # 網路架構設定
    # 參考: Paper 2 (Table II) 與 Paper 3 (Table 1)
    # 兩篇論文皆建議 actor 與 critic 網路使用 2 個隱藏層, 每層 128 個節點
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # 根據三篇論文的最佳超參數來初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003, # Paper 1 (Section V), Paper 2 (Table II) 與 Paper 3 (Table 1) 皆使用 0.0003 作為最佳學習率
        n_steps=2048,         # Paper 3 (Table 1) 指定 2048 步來穩定梯度更新
        batch_size=64,        # Paper 2 (Table II) 與 Paper 3 (Table 1) 建議 batch_size 為 64
        gamma=0.99,           # Paper 1 , 2 , 3 一致使用 0.99 作為折扣因子
        gae_lambda=0.95,      # Paper 1 , 2 , 3 一致使用 0.95 作為 GAE 參數
        clip_range=0.2,       # Paper 1 , 2 , 3 一致將 clip_range 參數設為 0.2
        ent_coef=0.0,         # Paper 3 (Table 1) 將熵係數設為 0.0 以加速收斂
        vf_coef=0.5,          # Paper 3 (Table 1) 將價值函數係數設為 0.5
        target_kl=0.01,       # Paper 3 (Table 1) 使用 0.01 作為 target KL 以提早停止更新
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_drone_logs/"
    )

    print("Starting training with optimized parameters...")

    # 載入現有模型繼續訓練
    # model = PPO.load('ppo_drone', env=env)

    # --- 設定 Callback ---
    callback = RewardLoggerCallback(save_dir='logs')

    # --- 開始訓練 ---
    # total_timesteps: 總共執行幾步
    # 先用 50_000 測試能不能跑通, 確認沒問題再改成 200_000。再測試 300_000。
    TOTAL_TIMESTEPS = 300_000
    print(f'\nStart trainting, total: {TOTAL_TIMESTEPS:,} steps...\n')

    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callback,
            progress_bar    = False,
        )
    except KeyboardInterrupt:
        print('\nTrain interrupted, save current...')

    # --- 儲存模型 ---
    model.save('ppo_drone')
    print('\nModel saved: ppo_drone.zip')

    # --- 儲存訓練曲線 ---
    callback.save_curve()

    # --- 清理 ---
    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()

    print('\nTrain fin.')
    print('  model: ppo_drone.zip')
    print('  cruve: logs/training_curve.png')
    print('  data : logs/rewards.csv')


if __name__ == '__main__':
    main()
