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
# 自訂 Callback: 每個 episode 結束時記錄 reward 與細項並支援自動停止
# ================================================================
class RewardLoggerCallback(BaseCallback):
    """
    繼承 SB3 的 BaseCallback, 在訓練過程中收集每個 episode 的 reward 與細項. 
    具備接續紀錄功能：若發現既有 csv，會自動讀取歷史數據並接續繪圖。
    新增：當連續 5 次日誌（共 50 回合）滿足 Arrive >= 60.00 且 Bound == 0.00 時，會自動終止訓練。
    """

    def __init__(self, save_dir: str = 'logs', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.csv_path = os.path.join(self.save_dir, 'rewards.csv')
        
        self.episode_rewards = []       # 存放【所有】episode 的累計總 reward (包含歷史)
        self._new_episodes_start_idx = 0 # 記錄這次訓練是從陣列的哪個 index 開始的
        self._current_ep_reward = 0.0   # 當前 episode 的累計總 reward
        
        # 記錄過去 10 回合的細項, 用來算平均
        self.recent_components = []

        # --- 自動停止核心計數器 ---
        self.consecutive_success_checks = 0
        self.should_stop_training = False

        # --- 關鍵修改 1：初始化時嘗試讀取歷史紀錄 ---
        self._load_history()

    def _load_history(self):
        """讀取既有的 rewards.csv，將歷史分數載入以接續畫圖"""
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # 跳過標題行
                    for row in reader:
                        if len(row) == 2:
                            # row[1] 是 reward
                            self.episode_rewards.append(float(row[1]))
                print(f"Loaded {len(self.episode_rewards)} historical episodes from {self.csv_path}")
            except Exception as e:
                print(f"Error loading history csv: {e}")
        
        # 標記這次新訓練產生的數據起點
        self._new_episodes_start_idx = len(self.episode_rewards)

    def _on_step(self) -> bool:
        """每步都會被呼叫. """
        # 累計這一步的總 reward
        self._current_ep_reward += self.locals['rewards'][0]

        # 檢查是否有傳出 infos ( 包含回合結束時的細項 ) 
        for info in self.locals.get('infos', []):
            if 'ep_components' in info:
                self.recent_components.append(info['ep_components'])

        # 如果這個 episode 結束了 ( done = terminated or truncated ) 
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            # 每 10 個 episode 印一次進度與細項平均
            ep = len(self.episode_rewards)
            
            # 確保我們有新的 components 可以算平均
            if len(self.recent_components) >= 10:
                if (ep - self._new_episodes_start_idx) % 10 == 0:
                    recent_mean = np.mean(self.episode_rewards[-10:])
                    
                    # 計算過去 10 回合各細項的平均值
                    avg_comp = {k: 0.0 for k in self.recent_components[0].keys()}
                    for comp in self.recent_components[-10:]:
                        for k, v in comp.items():
                            avg_comp[k] += v
                    for k in avg_comp.keys():
                        avg_comp[k] /= 10.0

                    print(f'Episode {ep:4d} | Total Mean: {recent_mean:7.2f} | '
                          f'Prog: {avg_comp["progress"]:6.2f} | '
                          f'Arrive: {avg_comp["arrive"]:5.2f} | '
                          f'Time: {avg_comp["time"]:6.2f} | '
                          f'Bound: {avg_comp["boundary"]:6.2f} | ', end='')
                    
                    # --- 關鍵修改：自動判定畢業邏輯門 ---
                    # 採用原版大尺度標準：Arrive 必須大於等於 60.00 且零墜機 Boundary 必須等於 0.00
                    if avg_comp["arrive"] >= 60.00 and avg_comp["boundary"] == 0.00:
                        self.consecutive_success_checks += 1
                        print(f'-> [Success: {self.consecutive_success_checks}/5]')
                        
                        if self.consecutive_success_checks >= 5:
                            print('\n\n[Auto-Stop] Successful training.')
                            print('Stopping and saving...\n')
                            self.should_stop_training = True
                    else:
                        if self.consecutive_success_checks > 0:
                            print('-> [Stop.]')
                        else:
                            print('') # 純換行
                        self.consecutive_success_checks = 0
                    
                    # 保持清單不要太長, 只留最近 10 筆
                    self.recent_components = self.recent_components[-10:]

        # 當 should_stop_training 為 True 時，回傳 False 會讓 Stable Baselines3 安全地跳出訓練迴圈
        return not self.should_stop_training

    def save_curve(self):
        """訓練結束後呼叫, 存 CSV 和訓練曲線圖. """
        # --- 存 CSV (附加模式或建立新檔) ---
        new_rewards = self.episode_rewards[self._new_episodes_start_idx:]
        
        file_exists = os.path.exists(self.csv_path)
        mode = 'a' if file_exists else 'w'
        
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode', 'reward'])
            
            for i, r in enumerate(new_rewards):
                ep_num = self._new_episodes_start_idx + i + 1
                writer.writerow([ep_num, r])
                
        print(f'Raw data appended/saved: {self.csv_path}')

        # --- 畫訓練曲線 ---
        if len(self.episode_rewards) == 0:
            return

        episodes = list(range(1, len(self.episode_rewards) + 1))
        rewards  = self.episode_rewards

        window = 20
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start:i+1]))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards,  color='lightblue', alpha=0.5, label='Round reward')
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2, label=f'Move mean:  ( {window} rounds ) ')
        
        if self._new_episodes_start_idx > 0:
            ax.axvline(x=self._new_episodes_start_idx, color='red', linestyle='--', alpha=0.5, label='Resumed Training')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve - Task B Random Target Navigation', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f'Training curve saved: {img_path}')


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

    MODEL_PATH = "ppo_drone"

    # 檢查是否有之前訓練好的模型檔 (.zip)
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model found: {MODEL_PATH}.zip, continue training...")
        # 載入舊模型，並綁定當前的環境
        model = PPO.load(MODEL_PATH, env=env, learning_rate=1e-4)

        # --- 設定 Callback ---
        callback = RewardLoggerCallback(save_dir='logs')

        # --- 開始訓練 ---
        ADDITIONAL_TIMESTEPS = 700000
        print(f'\nStarting additional training for {ADDITIONAL_TIMESTEPS:,} timesteps...\n')

        try:
            model.learn(
                total_timesteps = ADDITIONAL_TIMESTEPS,
                callback        = callback,
                progress_bar    = False,
                reset_num_timesteps = False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted via keyboard. Saving current progress...')

        # --- 儲存模型 ---
        NEW_MODEL_NAME = "ppo_drone_1.0_2.zip"
        model.save(NEW_MODEL_NAME)
        print(f'\nModel saved to {NEW_MODEL_NAME}.zip')
    else:
        print("Using new model...")
        model = PPO(
            policy        = 'MlpPolicy',
            env           = env,
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = 1024,
            batch_size    = 64,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            ent_coef      = 0.0,
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
        TOTAL_TIMESTEPS = 500_000
        print(f'\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...\n')

        try:
            model.learn(
                total_timesteps = TOTAL_TIMESTEPS,
                callback        = callback,
                progress_bar    = False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted via keyboard. Saving current progress...')

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


if __name__ == '__main__':
    main()