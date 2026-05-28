#!/usr/bin/env python3
"""
train.py
--------
基於 PPO 演算法的無人機導航訓練腳本。
實作課程學習 (Curriculum Learning) 機制，支援自動難度提升與動態存檔。

[Reference 1]
Tan, Z., & Karaköse, M. (2023). "A new approach for drone tracking with drone
using Proximal Policy Optimization based distributed deep reinforcement learning."
SoftwareX.
- 神經網路架構：[256, 256] 隱藏層，Tanh 啟動函數。

[Reference 2]
Zhang, J., Nguyen, S., Rivera, C. E. O., & Tyni, K. "AirPilot: Interpretable
PPO-based DRL Auto-Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights."
- EffectiveSpeed 成功判定：Distance / Timestep，連續懸停 HOVER_STEPS 步後升級。
- 確定性評估 (deterministic=True) 衡量真實控制能力。

[Reference 3]
Shen, S.-E., & Huang, Y.-C. (2024). "Application of Reinforcement Learning
in Controlling Quadrotor UAV Flight Actions." Drones.
- PPO 超參數：learning_rate=3e-4, n_steps=2048, batch_size=64,
              gamma=0.99, gae_lambda=0.95, vf_coef=0.5。

使用方式：
    python3 train.py             # 從 Level 1 開始訓練
    python3 train.py --level 3   # 從 Level 3 繼續訓練
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


# ================================================================
# 雙重 Logger：同步輸出至終端機與日誌檔
# ================================================================
class DualLogger:
    """將 stdout 同步寫入終端機與日誌檔，確保訓練紀錄完整保存。"""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ================================================================
# 快照儲存（升級 / 結束時呼叫）
# ================================================================
def _save_level_snapshot(save_dir: str, models_dir: str, level: int,
                          episode_rewards: list,
                          start_idx: int, end_idx: int,
                          model: PPO):
    """儲存當前 Level 的模型權重、CSV 數據與學習曲線圖。"""
    level_dir = os.path.join(save_dir, f'level{level}')
    os.makedirs(level_dir, exist_ok=True)

    level_rewards = episode_rewards[start_idx:end_idx]
    if level_rewards:
        csv_path = os.path.join(level_dir, f'rewards_level{level}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_in_level', 'episode_global', 'reward'])
            for i, r in enumerate(level_rewards):
                writer.writerow([i + 1, start_idx + i + 1, r])

        window   = min(20, len(level_rewards))
        smoothed = [np.mean(level_rewards[max(0, i - window + 1):i + 1])
                    for i in range(len(level_rewards))]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(level_rewards) + 1), level_rewards,
                color='lightblue', alpha=0.5, label='Episode reward')
        ax.plot(range(1, len(level_rewards) + 1), smoothed,
                color='steelblue', linewidth=2, label=f'Moving mean ({window} ep)')
        ax.set_xlabel('Episode (in this level)', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'PPO Training Curve - Level {level}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(level_dir, f'training_curve_level{level}.png'), dpi=150)
        plt.close()

    model.save(os.path.join(level_dir,   f'model_level{level}'))
    model.save(os.path.join(models_dir,  f'model_level{level}'))


# ================================================================
# Curriculum Callback
# ================================================================
class CurriculumCallback(BaseCallback):
    """
    監控訓練進度、執行定期確定性評估，
    並根據 EffectiveSpeed 成功率自動提升課程難度。

    升級判定：
        每 EVAL_INTERVAL 個回合執行 EVAL_EPISODES 次確定性評估，
        成功率 >= PROMOTE_THRESHOLD 即升一級；
        Level 7 達到 FINAL_THRESHOLD 則結訓。
    """

    PROMOTE_THRESHOLD = 0.65   # 一般升級門檻
    FINAL_THRESHOLD   = 0.80   # Level 7 結訓門檻
    PROMOTE_CHECKS    = 1      # 連續達標次數（單次即升）
    FINAL_CHECKS      = 1
    EVAL_INTERVAL     = 50     # 每幾個回合評估一次
    EVAL_EPISODES     = 30     # 每次評估的測試回合數

    def __init__(self, save_dir='logs', models_dir='models',
                 model_save_path='best_model', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir,   exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        self.save_dir        = save_dir
        self.models_dir      = models_dir
        self.csv_path        = os.path.join(save_dir, 'rewards.csv')
        self.model_save_path = model_save_path

        self.episode_rewards         = []
        self._new_episodes_start_idx = 0
        self._current_ep_reward      = 0.0
        self.recent_components       = []

        self.current_level            = 1
        self._level_start_idx         = 0
        self.consecutive_success_checks = 0
        self.best_det_rate            = 0.0
        self.should_stop_training     = False

        self._load_history()

    def _load_history(self):
        """讀取既有 CSV，接續歷史數據繪圖。"""
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            self.episode_rewards.append(float(row[1]))
                print(f'[History] Loaded {len(self.episode_rewards)} episodes.')
            except Exception as e:
                print(f'[Error] Load history failed: {e}')
        self._new_episodes_start_idx = len(self.episode_rewards)
        self._level_start_idx        = len(self.episode_rewards)

    def _get_env(self) -> DroneGymEnv:
        return self.locals['env'].envs[0].env

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals['rewards'][0]

        for info in self.locals.get('infos', []):
            if 'ep_components' in info:
                self.recent_components.append(info['ep_components'])

        dones = self.locals.get('dones', [False])
        if not dones[0]:
            return not self.should_stop_training

        # ── 回合結束 ─────────────────────────────────────────────
        self.episode_rewards.append(self._current_ep_reward)
        self._current_ep_reward = 0.0
        ep = len(self.episode_rewards)

        # 每 10 回合印一次訓練細項
        if (len(self.recent_components) >= 10
                and (ep - self._new_episodes_start_idx) % 10 == 0):

            recent_mean = np.mean(self.episode_rewards[-10:])
            avg_comp    = {k: 0.0 for k in self.recent_components[0].keys()}
            for comp in self.recent_components[-10:]:
                for k, v in comp.items():
                    avg_comp[k] += v
            for k in avg_comp:
                avg_comp[k] /= 10.0

            # 印出所有細項（含新增的 align / tilt）
            print(f'[L{self.current_level}] Ep {ep:4d} | Mean: {recent_mean:7.2f} | '
                  f'Prog: {avg_comp["progress"]:6.2f} | '
                  f'Align: {avg_comp["align"]:5.2f} | '
                  f'Prox: {avg_comp["proximity"]:5.2f} | '
                  f'Decel: {avg_comp["decel"]:5.2f} | '
                  f'Tilt: {avg_comp["tilt"]:5.2f} | '
                  f'Acti: {avg_comp["action"]:5.2f} | '
                  f'Arrive: {avg_comp["arrive"]:6.2f} | '
                  f'Time: {avg_comp["time"]:6.2f} | '
                  f'Bound: {avg_comp["boundary"]:5.2f}')

            self.recent_components = self.recent_components[-10:]

        # ── 定期確定性評估 ────────────────────────────────────────
        if ((ep - self._new_episodes_start_idx) % self.EVAL_INTERVAL == 0
                and ep > self._new_episodes_start_idx):

            det_rate = self._eval_deterministic(self.EVAL_EPISODES)
            print(f'\n[Det Eval] L{self.current_level} Ep {ep} | '
                  f'Success: {det_rate * 100:.0f}%', end='')

            if det_rate > self.best_det_rate:
                self.best_det_rate = det_rate
                self.model.save(self.model_save_path)
                print(f' -> [Best saved: {det_rate * 100:.0f}%]', end='')

            threshold    = (self.FINAL_THRESHOLD
                            if self.current_level >= DroneGymEnv.MAX_CURRICULUM_LEVEL
                            else self.PROMOTE_THRESHOLD)
            checks_needed = (self.FINAL_CHECKS
                             if self.current_level >= DroneGymEnv.MAX_CURRICULUM_LEVEL
                             else self.PROMOTE_CHECKS)

            if det_rate >= threshold:
                self.consecutive_success_checks += 1
                print(f' [{self.consecutive_success_checks}/{checks_needed}]')

                if self.consecutive_success_checks >= checks_needed:
                    if self.current_level < DroneGymEnv.MAX_CURRICULUM_LEVEL:
                        self._promote(ep)
                    else:
                        print(f'\n[Auto-Stop] Level {self.current_level} completed.')
                        _save_level_snapshot(
                            self.save_dir, self.models_dir, self.current_level,
                            self.episode_rewards, self._level_start_idx, ep, self.model)
                        self.should_stop_training = True
            else:
                if self.consecutive_success_checks > 0:
                    print(' -> [Reset]')
                else:
                    print()
                self.consecutive_success_checks = 0

        return not self.should_stop_training

    def _promote(self, current_ep: int):
        """處理等級提升：儲存快照、切換環境難度、重置計數器。"""
        old_level = self.current_level
        new_level = old_level + 1

        print(f'\n{"=" * 60}')
        print(f'  [Curriculum] LEVEL UP!  {old_level} -> {new_level}')
        print(f'{"=" * 60}')

        _save_level_snapshot(
            self.save_dir, self.models_dir, old_level,
            self.episode_rewards, self._level_start_idx, current_ep, self.model)

        self._get_env().set_curriculum_level(new_level)
        self.current_level              = new_level
        self.consecutive_success_checks = 0
        self._level_start_idx           = current_ep

    def _eval_deterministic(self, n_episodes: int = 10) -> float:
        """
        關閉隨機探索（deterministic=True），測試模型真實控制能力。
        成功判定：ep_components['arrive'] > 0（代表完成了 HOVER_STEPS 步懸停）。
        """
        env = DroneGymEnv(self._get_env().ros)
        env.set_curriculum_level(self.current_level)

        successes = 0
        for _ in range(n_episodes):
            obs, _              = env.reset()
            terminated = truncated = False
            success             = False
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
        """訓練結束後：寫入 CSV 並繪製全局訓練曲線。"""
        new_rewards = self.episode_rewards[self._new_episodes_start_idx:]
        file_exists = os.path.exists(self.csv_path)
        mode        = 'a' if file_exists else 'w'

        with open(self.csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['episode', 'reward'])
            for i, r in enumerate(new_rewards):
                writer.writerow([self._new_episodes_start_idx + i + 1, r])

        if not self.episode_rewards:
            return

        rewards  = self.episode_rewards
        episodes = list(range(1, len(rewards) + 1))
        window   = 20
        smoothed = [np.mean(rewards[max(0, i - window + 1):i + 1])
                    for i in range(len(rewards))]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(episodes, rewards,  color='lightblue', alpha=0.5, label='Episode reward')
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2,
                label=f'Moving mean ({window} ep)')

        # 標記各 Level 切換點
        for lv in range(1, DroneGymEnv.MAX_CURRICULUM_LEVEL + 1):
            lv_csv = os.path.join(self.save_dir, f'level{lv}', f'rewards_level{lv}.csv')
            if os.path.exists(lv_csv):
                with open(lv_csv, 'r') as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    last_ep = int(rows[-1][1])
                    if lv < DroneGymEnv.MAX_CURRICULUM_LEVEL:
                        ax.axvline(x=last_ep, color='orange',
                                   linestyle='--', alpha=0.7,
                                   label=f'L{lv}→L{lv+1}')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve - Curriculum Learning (L1~L7)', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curve.png'), dpi=150)
        plt.close()
        print(f'[Curve] Saved to {self.save_dir}/training_curve.png')


# ================================================================
# 主訓練流程
# ================================================================
def main():
    os.makedirs('logs', exist_ok=True)
    sys.stdout = DualLogger(os.path.join('logs', 'training_console.log'))
    sys.stderr = sys.stdout

    parser = argparse.ArgumentParser(description='Train drone PPO with curriculum learning')
    parser.add_argument('--level', type=int, default=1,
                        help='Starting curriculum level (1~7)')
    args        = parser.parse_args()
    start_level = int(np.clip(args.level, 1, DroneGymEnv.MAX_CURRICULUM_LEVEL))

    print('=' * 60)
    print(f'  PPO Training Start (From Level: {start_level})')
    print(f'  Observation: 13-dim | Curriculum: L1~L7')
    print('=' * 60)

    rclpy.init()
    ros_interface = DroneROSInterface()
    env           = DroneGymEnv(ros_interface)

    print('Waiting for Gazebo pose data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Pose received. Starting training.\n')

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # 決定載入哪個模型（當前 Level → 前一 Level → 全新）
    model_resume = os.path.join(models_dir, f'model_level{start_level}')
    model_prev   = os.path.join(models_dir, f'model_level{start_level - 1}')

    if os.path.exists(model_resume + '.zip'):
        print(f'[Resume] Loading {model_resume}.zip')
        model = PPO.load(model_resume, env=env)
    elif start_level > 1 and os.path.exists(model_prev + '.zip'):
        print(f'[Resume] Loading previous level: {model_prev}.zip')
        model = PPO.load(model_prev, env=env)
    else:
        print('[New] Creating new PPO model...')
        model = PPO(
            policy        = 'MlpPolicy',
            env           = env,
            verbose       = 1,
            # [Shen 2024] Table 1 最佳參數
            learning_rate = 3e-4,
            n_steps       = 2048,
            batch_size    = 64,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            vf_coef       = 0.5,
            n_epochs      = 10,
            ent_coef      = 0.01,
            # [Tan 2023] Table 2 網路架構
            policy_kwargs = dict(
                net_arch      = [256, 256],
                activation_fn = torch.nn.Tanh,
            ),
            tensorboard_log = './logs/tensorboard/',
        )

    callback = CurriculumCallback(
        save_dir        = 'logs',
        models_dir      = models_dir,
        model_save_path = 'best_model',
    )
    callback.current_level = start_level
    env.set_curriculum_level(start_level)

    try:
        model.learn(
            total_timesteps     = 2_000_000,
            callback            = callback,
            progress_bar        = False,
            reset_num_timesteps = False,
        )
    except KeyboardInterrupt:
        print('\n[Interrupted] Saving checkpoint...')

    model.save('ppo_drone_curriculum_resume')
    print('Checkpoint saved: ppo_drone_curriculum_resume.zip')

    callback.save_curve()

    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()
    print('\nTraining complete.')


if __name__ == '__main__':
    main()