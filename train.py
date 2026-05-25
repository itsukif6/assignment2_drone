#!/usr/bin/env python3
"""
train.py
--------
基於 PPO 演算法的無人機導航訓練腳本。
實作課程學習 (Curriculum Learning) 機制，支援自動難度提升與動態存檔。
支援參數化指定起始等級，且同步存檔至 logs/ 與 models/ 資料夾。

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

import os
import csv
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from drone_env import DroneROSInterface, DroneGymEnv

class DualLogger:
    """
    將標準輸出 (stdout) 同步寫入終端機與日誌檔。
    確保所有 print()、SB3 訓練進度與報錯都能被完整儲存下來。
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 確保即時寫入硬碟，避免程式崩潰時遺失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def _save_level_snapshot(save_dir: str, models_dir: str, level: int,
                          episode_rewards: list,
                          start_idx: int, end_idx: int,
                          model: PPO):
    """
    處理升級或訓練結束時的模型與數據快照儲存。
    將紀錄儲存於對應的 level 資料夾，並同步模型權重至全域 models 資料夾。
    """
    level_dir = os.path.join(save_dir, f'level{level}')
    os.makedirs(level_dir, exist_ok=True)

    level_rewards = episode_rewards[start_idx:end_idx]
    if level_rewards:
        # 儲存該層級的歷史回報數據
        csv_path = os.path.join(level_dir, f'rewards_level{level}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_in_level', 'episode_global', 'reward'])
            for i, r in enumerate(level_rewards):
                writer.writerow([i + 1, start_idx + i + 1, r])

        # 繪製並儲存該層級的學習曲線圖
        window = min(20, len(level_rewards))
        smoothed = []
        for i in range(len(level_rewards)):
            s = max(0, i - window + 1)
            smoothed.append(np.mean(level_rewards[s:i + 1]))

        episodes_x = list(range(1, len(level_rewards) + 1))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes_x, level_rewards, color='lightblue', alpha=0.5, label='Episode reward')
        ax.plot(episodes_x, smoothed, color='steelblue', linewidth=2, label=f'Moving mean ({window} ep)')
        ax.set_xlabel('Episode (in this level)', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'PPO Training Curve - Level {level}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = os.path.join(level_dir, f'training_curve_level{level}.png')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

    # 同步儲存模型權重至兩個路徑
    model_path_log = os.path.join(level_dir, f'model_level{level}')
    model.save(model_path_log)
    
    model_path_global = os.path.join(models_dir, f'model_level{level}')
    model.save(model_path_global)


class CurriculumCallback(BaseCallback):
    """
    自定義回調函數，負責監控訓練進度、執行定期評估，
    並根據評估結果決定是否提升課程難度 (Level Up)。
    """
    PROMOTE_THRESHOLD = 0.65   # 觸發升級所需的評估成功率門檻 (65%)
    PROMOTE_CHECKS    = 1      # 連續達到門檻的次數要求 (單次通過即升級)
    EVAL_INTERVAL     = 50     # 每經過多少訓練回合進行一次確定性評估
    EVAL_EPISODES     = 30     # 每次評估執行的測試回合數

    FINAL_THRESHOLD   = 0.80   # 最高層級(Level 7)的通關結訓門檻
    FINAL_CHECKS      = 1      # 最高層級連續達到門檻的次數要求

    def __init__(self, save_dir='logs', models_dir='models', model_save_path='best_model', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        self.save_dir = save_dir
        self.models_dir = models_dir
        self.csv_path = os.path.join(save_dir, 'rewards.csv')
        self.model_save_path = model_save_path

        self.episode_rewards = []
        self._new_episodes_start_idx = 0   
        self._current_ep_reward = 0.0
        self.recent_components = []

        self.current_level = 1
        self._level_start_idx = 0

        self.consecutive_success_checks = 0
        self.best_det_rate = 0.0
        self.should_stop_training = False

        self._load_history()

    def _load_history(self):
        # 讀取先前的訓練紀錄，避免訓練中斷重啟後圖表斷層
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            self.episode_rewards.append(float(row[1]))
            except Exception as e:
                print(f'[Error] Load history data failed: {e}')
        self._new_episodes_start_idx = len(self.episode_rewards)
        self._level_start_idx = len(self.episode_rewards)

    def _get_env(self) -> DroneGymEnv:
        return self.locals['env'].envs[0].env

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals['rewards'][0]

        # 收集環境回傳的各項獎勵組成細節
        for info in self.locals.get('infos', []):
            if 'ep_components' in info:
                self.recent_components.append(info['ep_components'])

        dones = self.locals.get('dones', [False])
        if not dones[0]:
            return not self.should_stop_training

        # 當回合結束時，結算總回報
        self.episode_rewards.append(self._current_ep_reward)
        self._current_ep_reward = 0.0
        ep = len(self.episode_rewards)

        # 每 10 個回合輸出一次平均獎勵與各項獎勵組成分析
        if (len(self.recent_components) >= 10 and (ep - self._new_episodes_start_idx) % 10 == 0):
            recent_mean = np.mean(self.episode_rewards[-10:])
            avg_comp = {k: 0.0 for k in self.recent_components[0].keys()}
            for comp in self.recent_components[-10:]:
                for k, v in comp.items():
                    avg_comp[k] += v
            for k in avg_comp:
                avg_comp[k] /= 10.0

            print(f'[L{self.current_level}] Ep {ep:4d} | Mean: {recent_mean:7.2f} | '
                  f'Prog: {avg_comp["progress"]:5.2f} | Prox: {avg_comp["proximity"]:5.2f} | '
                  f'Acti: {avg_comp["action"]:5.2f} | Decel: {avg_comp["decel"]:5.2f} | '
                  f'Arrive: {avg_comp["arrive"]:5.2f} | Time: {avg_comp["time"]:6.2f} | '
                  f'Bound: {avg_comp["boundary"]:5.2f}')
            self.recent_components = self.recent_components[-10:]

        # 達到評估間隔，執行確定性策略評估
        if ((ep - self._new_episodes_start_idx) % self.EVAL_INTERVAL == 0 and ep > self._new_episodes_start_idx):
            det_rate = self._eval_deterministic(self.EVAL_EPISODES)
            print(f'\n[Det Eval] L{self.current_level} Ep {ep} | Det Success: {det_rate*100:.0f}%', end='')

            # 更新最佳模型
            if det_rate > self.best_det_rate:
                self.best_det_rate = det_rate
                self.model.save(self.model_save_path)
                print(f' -> [Best saved: {det_rate*100:.0f}%]', end='')

            threshold = (self.FINAL_THRESHOLD if self.current_level >= DroneGymEnv.MAX_CURRICULUM_LEVEL else self.PROMOTE_THRESHOLD)
            checks_needed = (self.FINAL_CHECKS if self.current_level >= DroneGymEnv.MAX_CURRICULUM_LEVEL else self.PROMOTE_CHECKS)

            # 判斷是否滿足單次通過的升級條件
            if det_rate >= threshold:
                self.consecutive_success_checks += 1
                print(f' [{self.consecutive_success_checks}/{checks_needed}]')

                if self.consecutive_success_checks >= checks_needed:
                    if self.current_level < DroneGymEnv.MAX_CURRICULUM_LEVEL:
                        self._promote(ep)
                    else:
                        print(f'\n[Auto-Stop] Level {self.current_level} had reach the promote. Train stop.')
                        _save_level_snapshot(self.save_dir, self.models_dir, self.current_level, self.episode_rewards, self._level_start_idx, ep, self.model)
                        self.should_stop_training = True
            else:
                if self.consecutive_success_checks > 0:
                    print(' -> [Reset]')
                else:
                    print()
                self.consecutive_success_checks = 0

        return not self.should_stop_training

    def _promote(self, current_ep: int):
        # 處理等級提升邏輯，更新環境參數並重置評估計數器
        old_level = self.current_level
        new_level = old_level + 1

        print(f'\n{"="*60}')
        print(f'  [Curriculum] LEVEL UP!  {old_level} -> {new_level}')
        print(f'{"="*60}')

        _save_level_snapshot(self.save_dir, self.models_dir, old_level, self.episode_rewards, self._level_start_idx, current_ep, self.model)

        env = self._get_env()
        env.set_curriculum_level(new_level)
        self.current_level = new_level
        self.consecutive_success_checks = 0
        self._level_start_idx = current_ep

    def _eval_deterministic(self, n_episodes: int = 10) -> float:
        # 關閉隨機探索 (deterministic=True)，測試模型真實控制能力
        ros = self._get_env().ros
        env = DroneGymEnv(ros)
        env.set_curriculum_level(self.current_level)

        successes = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            success = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated and info.get('ep_components', {}).get('arrive', 0.0) > 0:
                    success = True
            if success:
                successes += 1

        del env
        return successes / n_episodes

    def save_curve(self):
        # 繪製並儲存橫跨所有等級的總體訓練趨勢圖
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

        if not self.episode_rewards:
            return
            
        rewards = self.episode_rewards
        episodes = list(range(1, len(rewards) + 1))
        window = 20
        smoothed = []
        for i in range(len(rewards)):
            s = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[s:i + 1]))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(episodes, rewards, color='lightblue', alpha=0.5, label='Episode reward')
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2, label=f'Moving mean ({window} ep)')

        # 在圖表上標記等級切換點
        for lv in range(1, DroneGymEnv.MAX_CURRICULUM_LEVEL + 1):
            lv_csv = os.path.join(self.save_dir, f'level{lv}', f'rewards_level{lv}.csv')
            if os.path.exists(lv_csv):
                with open(lv_csv, 'r') as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    last_global_ep = int(rows[-1][1])
                    if lv < DroneGymEnv.MAX_CURRICULUM_LEVEL:
                        ax.axvline(x=last_global_ep, color='orange', linestyle='--', alpha=0.7)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve - Curriculum Learning (Level 1 to 7)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()


def main():
    os.makedirs('logs', exist_ok=True)
    log_file_path = os.path.join('logs', 'training_console.log')
    
    # 綁定標準輸出與標準錯誤到自定義的 Logger
    sys.stdout = DualLogger(log_file_path)
    sys.stderr = sys.stdout  # 讓潛在的報錯訊息也寫入同一個檔案

    parser = argparse.ArgumentParser(description="訓練無人機 PPO 模型 (支援課程學習)")
    parser.add_argument('--level', type=int, default=1, help='指定起始的課程等級 (1~7)')
    args = parser.parse_args()
    start_level = args.level

    print('=' * 60)
    print(f'  PPO train start (From level: {start_level})')
    print('=' * 60)

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 決定載入權重的邏輯，優先尋找當前設定等級之模型，次之尋找前一等級
    model_resume_path = os.path.join(models_dir, f'model_level{start_level}')
    model_prev_path   = os.path.join(models_dir, f'model_level{start_level - 1}')

    if os.path.exists(model_resume_path + '.zip'):
        model = PPO.load(model_resume_path, env=env)
    elif start_level > 1 and os.path.exists(model_prev_path + '.zip'):
        model = PPO.load(model_prev_path, env=env)
    else:
        # PPO 模型超參數設定
        # 參數值設計參照文獻設定以確保高穩定性與高樣本效率
        model = PPO(
            policy          = 'MlpPolicy',
            env             = env,
            verbose         = 1,

            # [參考: Shen 等人 (2024) 論文] PPO 訓練參數表 (Table 1)
            # 這些超參數與 Shen 論文中 PPO 最佳化參數完全一致：
            learning_rate   = 3e-4,  # 對應 Shen 論文的 0.0003 (AirPilot 也使用 3e-4)
            n_steps         = 2048,  # 對應 Shen 論文的 N steps = 2048
            batch_size      = 64,    # 對應 Shen 論文的 Batch Size = 64 (AirPilot 也使用 64)
            gamma           = 0.99,  # 對應 Shen 論文的 Gamma = 0.99 (AirPilot 也是 0.99)
            gae_lambda      = 0.95,  # 對應 Shen 論文的 GAE Lambda = 0.95
            vf_coef         = 0.5,   # 對應 Shen 論文的 VF Coefficient = 0.5
            
            n_epochs        = 10,
            ent_coef        = 0.01,

            # [參考: Tan & Karaköse (2023) 論文] 分散式 PPO 無人機追蹤
            # 架構設計參考了其處理三維狀態的設定 (Table 2)。
            policy_kwargs   = dict(
                net_arch      = [256, 256],    # 對應 Tan 論文的 Hidden layer = 256
                activation_fn = torch.nn.Tanh, # 對應 Tan 論文的 Activation function = tanh
            ),
            tensorboard_log = './logs/tensorboard/',
        )

    callback = CurriculumCallback(save_dir='logs', models_dir=models_dir, model_save_path='best_model')
    callback.current_level = start_level
    env.set_curriculum_level(start_level)

    TOTAL_TIMESTEPS = 2_000_000
    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = callback,
            progress_bar        = False,
            reset_num_timesteps = False,
        )
    except KeyboardInterrupt:
        pass
    
    # 手動中斷或訓練結束時，儲存為中斷備份檔，避免覆蓋標準升級檔
    NEW_MODEL_NAME = 'ppo_drone_curriculum_resume'
    model.save(NEW_MODEL_NAME)

    callback.save_curve()

    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()