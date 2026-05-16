#!/usr/bin/env python3
"""
train.py
--------
用 PPO 演算法訓練無人機隨機目標導航 (Task B).

超參數來源:
    Paper 1: Tan & Karakose (2023) Table 2 & 3 -> hidden_layer=256, activation=tanh, timesteps=300000.
    Paper 2: Zhang et al. (2024) Section IV     -> learning_rate=3e-4, batch_size=64, gamma=0.99.
    Paper 3: Shen & Huang (2024) Table 1        -> n_steps=2048, gae_lambda=0.95, vf_coef=0.5.

使用方式:
    python3 train.py

訓練完成後會產生:
    ppo_drone.zip           -> 訓練好的模型
    logs/training_curve.png -> reward 曲線圖
    logs/rewards.csv        -> 原始數據
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

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
        self.episode_rewards    = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        """每步都會被呼叫."""
        self._current_ep_reward += self.locals['rewards'][0]

        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            ep = len(self.episode_rewards)
            if ep % 10 == 0:
                recent_mean = np.mean(self.episode_rewards[-10:])
                print(f'  Episode {ep:4d} | last 10 mean reward: {recent_mean:.2f}')

        return True

    def save_curve(self):
        """訓練結束後呼叫, 存 CSV 和訓練曲線圖."""
        # --- 存 CSV ---
        csv_path = os.path.join(self.save_dir, 'rewards.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])
        print(f'CSV saved to {csv_path}')

        # --- 畫訓練曲線 ---
        episodes = list(range(1, len(self.episode_rewards) + 1))
        rewards  = self.episode_rewards

        # 移動平均 (每 20 回合) 讓曲線更平滑好看
        window   = 20
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards,  color='lightblue', alpha=0.5, label='per episode reward')
        ax.plot(episodes, smoothed, color='steelblue',  linewidth=2,
                label=f'moving avg ({window} episodes)')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve - Task B Random Target Navigation', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f'Training curve saved to {img_path}')


# ================================================================
# 主訓練流程
# ================================================================
def main():
    print('=' * 60)
    print('  PPO Training: Task B Random Target Navigation')
    print('=' * 60)

    # --- 初始化 ROS 2 ---
    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    # --- 等待第一筆位置資料 ---
    print('Waiting for Gazebo pose data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Pose data received. Starting training.')

    # --- 建立 PPO 模型 ---
    # 超參數說明 (每項均附論文來源):
    #
    # learning_rate = 3e-4:
    #   Paper 2 (AirPilot) Section IV: "PPO is imported from stable_baselines3
    #   with a default 3e-4 learning rate."
    #   Paper 3 Table 1 亦使用相同值.
    # learning_rate 調高: 加速初期學習.
    #   目前 -100 的平坦曲線顯示梯度訊號太弱, 提高學習率能讓有效的更新更明顯.
    #
    # n_steps = 2048:
    #   Paper 3 (Shen & Huang, 2024) Table 1: PPO n_steps = 2048.
    #   較大的 n_steps 讓每次策略更新前收集更多軌跡, 提升梯度估計的穩定性.
    # n_steps 改小: 讓每個 episode 結束後更快更新策略.
    #   200 步 = 1 個 episode, 512 步約 2-3 個 episode 就更新一次,
    #   讓 agent 更快從每次飛行中學習.
    #
    # batch_size = 64:
    #   Paper 2 Section IV & Paper 3 Table 1: batch_size = 64.
    #
    # gamma = 0.99:
    #   Paper 2 Section IV: "0.99 discount factor."
    #   Paper 3 Table 1: gamma = 0.99.
    #   高折扣因子讓 agent 重視長期回報, 適合需要多步導航的任務.
    #
    # gae_lambda = 0.95:
    #   Paper 3 Table 1: GAE Lambda = 0.95.
    #   GAE (Generalized Advantage Estimation) 平衡 bias 與 variance,
    #   0.95 是 PPO 論文原始建議值.
    #
    # ent_coef = 0.01:
    #   熵係數鼓勵探索, 防止策略過早收斂到局部最優.
    #   Paper 1 (Tan & Karakose, 2023) 的分散式實驗顯示適度探索能加速收斂.
    # ent_coef 調高: 增加探索.
    #   agent 目前停在局部最優 (原地漂移), 更高的熵係數迫使它嘗試更多樣的動作.
    #
    # vf_coef = 0.5:
    #   Paper 3 Table 1: VF Coefficient = 0.5.
    #   控制 Critic (Value Function) 損失在總損失中的權重.
    #
    # policy_kwargs -> net_arch=[256, 256], activation_fn=tanh:
    #   Paper 1 Table 2: "Hidden layer = 256, Activation function = tanh."
    #   雙層 256 神經元搭配 tanh 激活函數, 在無人機控制任務中表現穩定.
    #   Paper 2 使用 64 神經元, 但 Paper 1 的 256 對更複雜的連續控制任務更有表達力.
    #
    # total_timesteps = 300000:
    #   Paper 1 Table 3: "Stop condition: Time-steps = 300000."
    #   Paper 3 Table 1 亦使用 150000-300000 步, 300000 是合理的起始值.

    model = PPO(
        policy        = 'MlpPolicy',
        env           = env,
        verbose       = 0,
        learning_rate = 1e-3,
        n_steps       = 512,
        batch_size    = 64,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.05,
        vf_coef       = 0.5,
        policy_kwargs = dict(
            net_arch       = [256, 256],
            activation_fn  = torch.nn.Tanh,
        ),
        tensorboard_log = './logs/tensorboard/',
    )

    # --- 設定 Callback ---
    callback = RewardLoggerCallback(save_dir='logs')

    # --- 開始訓練 ---
    TOTAL_TIMESTEPS = 300_000
    print(f'\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...\n')

    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callback,
            progress_bar    = False,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted. Saving current progress...')

    # --- 儲存模型 ---
    model.save('ppo_drone')
    print('\nModel saved to ppo_drone.zip')

    # --- 儲存訓練曲線 ---
    callback.save_curve()

    # --- 清理 ---
    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()

    print('\nTraining complete.')
    print('  Model  : ppo_drone.zip')
    print('  Curve  : logs/training_curve.png')
    print('  Data   : logs/rewards.csv')


if __name__ == '__main__':
    main()