#!/usr/bin/env python3
"""
train.py
--------
PPO 訓練：隨機目標導航 + 懸停確認
Curriculum: ARRIVE_DIST 1.0 -> 0.8 -> 0.6 -> 0.4
每次 Det Eval 達標或創新高，自動存模型 / CSV / PNG
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneROSInterface, DroneGymEnv


# ================================================================
# Callback
# ================================================================
class TrainingCallback(BaseCallback):
    """
    每 10 episode 印 stochastic 統計
    每 100 episode 跑 deterministic eval
    達標自動切 curriculum / 存檔
    """

    CURRICULUM = [1.0, 0.8, 0.6, 0.4]   # ARRIVE_DIST 分段
    PASS_RATE  = 0.65                     # 達標門檻
    PASS_COUNT = 3                        # 連續 N 次達標才升級

    LOG_DIR        = 'logs'
    MODEL_DIR      = 'models'
    CSV_PATH       = 'logs/rewards.csv'
    CURVE_PATH     = 'logs/training_curve.png'

    def __init__(self, verbose=0):
        super().__init__(verbose)
        os.makedirs(self.LOG_DIR,   exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # episode 紀錄
        self._ep_reward      = 0.0
        self.ep_rewards      = []         # 所有 episode reward
        self.ep_successes    = []         # 0/1
        self._hover_success  = False      # 本回合是否懸停成功

        # curriculum 狀態
        self.stage           = 0          # 當前階段索引
        self.pass_streak     = 0          # 連續達標次數
        self.best_det_rate   = 0.0

        # 每回合細項追蹤
        self._recent_components = []   # 最近 10 回合的 ep_components
        self._recent_dists      = []
        self._recent_vels       = []
        self._recent_steps      = []

        # CSV header
        if not os.path.exists(self.CSV_PATH):
            with open(self.CSV_PATH, 'w', newline='') as f:
                csv.writer(f).writerow(['episode', 'reward', 'success', 'arrive_dist'])

    # ---- 內部工具 ----
    def _current_arrive(self):
        return self.CURRICULUM[self.stage]

    def _train_env(self):
        """取得 DroneGymEnv 實例（穿透 Monitor wrapper）"""
        return self.locals['env'].envs[0].env

    def _model_tag(self):
        """模型檔名包含當前 curriculum 階段"""
        return os.path.join(self.MODEL_DIR, f'best_stage{self.stage}')

    # ---- SB3 hooks ----
    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals['rewards'][0])

        # 從 info 取懸停成功旗標
        for info in self.locals.get('infos', []):
            if info.get('hover_success', False):
                self._hover_success = True

        # episode 結束
        dones = self.locals.get('dones', [False])
        if dones[0]:
            ep   = len(self.ep_rewards) + 1
            rew  = self._ep_reward
            succ = int(self._hover_success)

            self.ep_rewards.append(rew)
            self.ep_successes.append(succ)
            self._ep_reward     = 0.0
            self._hover_success = False

            # 收集細項
            for info in self.locals.get('infos', []):
                if 'ep_components' in info:
                    self._recent_components.append(info['ep_components'])
                    self._recent_dists.append(info.get('ep_dist', 0.0))
                    self._recent_vels.append(info.get('ep_vel', 0.0))
                    self._recent_steps.append(info.get('ep_steps', 0))

            # 寫 CSV（附加）
            with open(self.CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([ep, f'{rew:.2f}', succ, self._current_arrive()])

            # 每 10 episode 印 stochastic 摘要
            if ep % 10 == 0:
                recent_r = np.mean(self.ep_rewards[-10:])
                recent_s = np.mean(self.ep_successes[-10:]) * 100

                if len(self._recent_components) >= 10:
                    last10 = self._recent_components[-10:]
                    keys   = last10[0].keys()
                    avg_c  = {k: np.mean([c[k] for c in last10]) for k in keys}
                    avg_dist  = np.mean(self._recent_dists[-10:])
                    avg_vel   = np.mean(self._recent_vels[-10:])
                    avg_steps = np.mean(self._recent_steps[-10:])
                    print(f'Ep {ep:5d} | MeanR: {recent_r:8.2f} | '
                          f'Arrive(S): {recent_s:5.1f}% | '
                          f'Prog: {avg_c["progress"]:6.1f} | '
                          f'Hover: {avg_c["hover"]:6.1f} | '
                          f'Time: {avg_c["time"]:6.1f} | '
                          f'Bound: {avg_c["boundary"]:6.1f} | '
                          f'Dist: {avg_dist:.2f} | '
                          f'Vel: {avg_vel:.2f} | '
                          f'Steps: {avg_steps:.0f} | '
                          f'ARRIVE: {self._current_arrive():.1f}')
                else:
                    print(f'Ep {ep:5d} | MeanR: {recent_r:8.2f} | '
                          f'Arrive(S): {recent_s:5.1f}% | '
                          f'ARRIVE: {self._current_arrive():.1f}')

            # 每 100 episode 做 deterministic eval
            if ep % 100 == 0:
                det_rate = self._eval_deterministic(n_ep=30)
                d    = det_rate   # dict: rate / mean_steps / mean_dist
                rate = d['rate']
                print(f'\n[Det Eval] Ep {ep} | Det: {rate*100:.0f}% | '
                      f'Steps: {d["mean_steps"]:.1f} | FinalDist: {d["mean_dist"]:.2f} | '
                      f'Stage {self.stage} (ARRIVE={self._current_arrive():.1f})', end=' ')
                det_rate = rate

                # 創新高 -> 存檔
                if det_rate > self.best_det_rate:
                    self.best_det_rate = det_rate
                    self.model.save(self._model_tag())
                    self._save_curve()
                    print(f'-> [Saved] {self._model_tag()}', end=' ')

                # 達標判斷
                if det_rate >= self.PASS_RATE:
                    self.pass_streak += 1
                    print(f'-> [Pass {self.pass_streak}/{self.PASS_COUNT}]')
                    if self.pass_streak >= self.PASS_COUNT:
                        self._advance_curriculum()
                else:
                    if self.pass_streak > 0:
                        print('-> [Streak reset]')
                    else:
                        print()
                    self.pass_streak = 0

        return True

    # ---- Curriculum ----
    def _advance_curriculum(self):
        if self.stage < len(self.CURRICULUM) - 1:
            self.stage       += 1
            self.pass_streak  = 0
            self.best_det_rate = 0.0
            new_dist = self.CURRICULUM[self.stage]
            self._train_env().ARRIVE_DIST = new_dist
            # 存一個 milestone 模型
            tag = os.path.join(self.MODEL_DIR, f'milestone_stage{self.stage}')
            self.model.save(tag)
            self._save_curve()
            print(f'\n[Curriculum] Advanced to stage {self.stage} '
                  f'ARRIVE_DIST={new_dist:.1f} | saved {tag}')
        else:
            print('\n[Curriculum] All stages complete!')

    # ---- Deterministic Eval ----
    def _eval_deterministic(self, n_ep=30) -> dict:
        """回傳 {'rate', 'mean_steps', 'mean_dist'} """
        inner     = self._train_env()
        successes = 0
        all_steps = []
        all_dists = []

        for _ in range(n_ep):
            obs, _ = inner.reset()
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = inner.step(action)
                if info.get('hover_success', False):
                    successes += 1

            all_steps.append(info.get('ep_steps', inner.step_count))
            all_dists.append(info.get('ep_dist',
                float(np.linalg.norm(inner.ros.current_pose - inner.target_pos))))

        return {
            'rate':       successes / n_ep,
            'mean_steps': float(np.mean(all_steps)),
            'mean_dist':  float(np.mean(all_dists)),
        }

    # ---- 存圖 ----
    def _save_curve(self):
        if len(self.ep_rewards) < 2:
            return
        episodes = list(range(1, len(self.ep_rewards) + 1))
        window   = 20
        smoothed = [
            np.mean(self.ep_rewards[max(0, i - window + 1): i + 1])
            for i in range(len(self.ep_rewards))
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, self.ep_rewards, color='lightblue', alpha=0.4, label='Episode reward')
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
        ax.set_title('PPO Training Curve — Random Target Navigation')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.CURVE_PATH, dpi=150)
        plt.close()

    def on_training_end(self):
        self._save_curve()
        print(f'\n[Done] Curve saved: {self.CURVE_PATH}')


# ================================================================
# 主訓練流程
# ================================================================
def main():
    print('=' * 60)
    print('  PPO Training: Random Target Navigation + Hover')
    print('=' * 60)

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])

    ros = DroneROSInterface()
    env = DroneGymEnv(ros)
    env = Monitor(env, 'logs')

    # 等第一筆位姿資料
    print('Waiting for Gazebo pose...')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)
    print('Pose received. Building model...')

    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        verbose         = 0,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(
            net_arch      = [256, 256],
            activation_fn = torch.nn.Tanh,
        ),
        device          = 'cuda',
        tensorboard_log = 'logs/tensorboard',
    )

    callback = TrainingCallback()

    print('\nStarting training...\n')
    try:
        model.learn(
            total_timesteps     = 1_000_000,
            callback            = callback,
            progress_bar        = False,
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        print('\nInterrupted. Saving...')

    model.save('models/ppo_drone_final')
    callback.on_training_end()
    print('Model saved: models/ppo_drone_final.zip')

    ros.send_velocity(0, 0, 0)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()