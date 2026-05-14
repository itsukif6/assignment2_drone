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
    logs/rewards.csv        → 原始數據 (方便自己再畫圖) 
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
        self._current_ep_reward = 0.0  # 當前 episode 的累計 reward

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
                print(f'  Episode {ep:4d} | 近10回合平均 reward: {recent_mean:.2f}')

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
        print(f'原始數據已存至 {csv_path}')

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
        ax.plot(episodes, rewards,   color='lightblue', alpha=0.5, label='每回合 reward')
        ax.plot(episodes, smoothed,  color='steelblue', linewidth=2, label=f'移動平均 ({window} 回合) ')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve — Task B Random Target Navigation', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f'訓練曲線已存至 {img_path}')


# ================================================================
# 主訓練流程
# ================================================================
def main():
    print('=' * 55)
    print('  PPO 訓練: Task B 隨機目標導航')
    print('=' * 55)

    # --- 初始化 ROS 2 ---
    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    # --- 等待第一筆位置資料 (確保模擬器已啟動) ---
    print('等待 Gazebo 位置資料...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('已收到位置資料, 開始訓練')

    # --- 建立 PPO 模型 ---
    # 超參數說明: 
    #   learning_rate : 學習率, 控制每次更新的幅度
    #   n_steps       : 每次更新前收集幾步資料 (越大越穩定但越慢) 
    #   batch_size    : 每次梯度更新用多少資料
    #   gamma         : 折扣因子, 0.99 代表重視長期回報
    #   ent_coef      : 熵獎勵係數, 鼓勵 agent 探索

    model = PPO(
        policy        = 'MlpPolicy',   # 全連接網路 (適合向量輸入) 
        env           = env,
        verbose       = 0,             # 關掉 SB3 自帶的 verbose, 改用我們自己的 callback
        learning_rate = 3e-4,
        n_steps       = 512,
        batch_size    = 64,
        gamma         = 0.99,
        ent_coef      = 0.01,
        tensorboard_log = './logs/tensorboard/',
    )

    # 載入現有模型繼續訓練
    # model = PPO.load('ppo_drone', env=env)

    # --- 設定 Callback ---
    callback = RewardLoggerCallback(save_dir='logs')

    # --- 開始訓練 ---
    # total_timesteps: 總共執行幾步
    # 先用 50_000 測試能不能跑通, 確認沒問題再改成 200_000。再測試 300_000。
    TOTAL_TIMESTEPS = 300_000
    print(f'\n開始訓練, 共 {TOTAL_TIMESTEPS:,} 步...\n')

    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callback,
            progress_bar    = False,
        )
    except KeyboardInterrupt:
        print('\n訓練被中斷, 儲存目前進度...')

    # --- 儲存模型 ---
    model.save('ppo_drone')
    print('\n模型已存至 ppo_drone.zip')

    # --- 儲存訓練曲線 ---
    callback.save_curve()

    # --- 清理 ---
    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()

    print('\n訓練完成')
    print('  模型: ppo_drone.zip')
    print('  曲線: logs/training_curve.png')
    print('  數據: logs/rewards.csv')


if __name__ == '__main__':
    main()
