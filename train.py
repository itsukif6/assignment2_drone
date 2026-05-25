#!/usr/bin/env python3
"""
train.py
--------
用 PPO 演算法訓練無人機隨機目標導航 (Task B).
加入課程學習 (Curriculum Learning)：Level 1 → 2 → 3 自動升級。
"""

import os
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from drone_env import DroneROSInterface, DroneGymEnv

def _save_level_snapshot(save_dir: str, level: int,
                          episode_rewards: list,
                          start_idx: int, end_idx: int,
                          model: PPO):
    level_dir = os.path.join(save_dir, f'level{level}')
    os.makedirs(level_dir, exist_ok=True)

    level_rewards = episode_rewards[start_idx:end_idx]
    if not level_rewards:
        return

    csv_path = os.path.join(level_dir, f'rewards_level{level}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode_in_level', 'episode_global', 'reward'])
        for i, r in enumerate(level_rewards):
            writer.writerow([i + 1, start_idx + i + 1, r])
    print(f'  [Snapshot] CSV  -> {csv_path}')

    window = min(20, len(level_rewards))
    smoothed = []
    for i in range(len(level_rewards)):
        s = max(0, i - window + 1)
        smoothed.append(np.mean(level_rewards[s:i + 1]))

    episodes_x = list(range(1, len(level_rewards) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes_x, level_rewards, color='lightblue', alpha=0.5, label='Episode reward')
    ax.plot(episodes_x, smoothed,      color='steelblue', linewidth=2,
            label=f'Moving mean ({window} ep)')
    ax.set_xlabel('Episode (in this level)', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(f'PPO Training Curve - Level {level}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path = os.path.join(level_dir, f'training_curve_level{level}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'  [Snapshot] PNG  -> {png_path}')

    model_path = os.path.join(level_dir, f'model_level{level}')
    model.save(model_path)
    print(f'  [Snapshot] Model-> {model_path}.zip')


class CurriculumCallback(BaseCallback):
    PROMOTE_THRESHOLD = 0.65   
    PROMOTE_CHECKS    = 3      
    EVAL_INTERVAL     = 50    
    EVAL_EPISODES     = 30     

    FINAL_THRESHOLD   = 0.80   
    FINAL_CHECKS      = 3      

    def __init__(self, save_dir='logs', model_save_path='best_model', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir        = save_dir
        self.csv_path        = os.path.join(save_dir, 'rewards.csv')
        self.model_save_path = model_save_path

        self.episode_rewards         = []
        self._new_episodes_start_idx = 0   
        self._current_ep_reward      = 0.0
        self.recent_components       = []

        self.current_level           = 1
        self._level_start_idx = len(self.episode_rewards)

        self.consecutive_success_checks = 0
        self.best_det_rate              = 0.0
        self.should_stop_training       = False

        self._load_history()

    def _load_history(self):
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            self.episode_rewards.append(float(row[1]))
                print(f'Loaded {len(self.episode_rewards)} historical episodes from {self.csv_path}')
            except Exception as e:
                print(f'Error loading history csv: {e}')
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

        self.episode_rewards.append(self._current_ep_reward)
        self._current_ep_reward = 0.0
        ep = len(self.episode_rewards)

        if (len(self.recent_components) >= 10 and
                (ep - self._new_episodes_start_idx) % 10 == 0):
            recent_mean = np.mean(self.episode_rewards[-10:])
            avg_comp = {k: 0.0 for k in self.recent_components[0].keys()}
            for comp in self.recent_components[-10:]:
                for k, v in comp.items():
                    avg_comp[k] += v
            for k in avg_comp:
                avg_comp[k] /= 10.0

            print(f'[L{self.current_level}] Ep {ep:4d} | Mean: {recent_mean:7.2f} | '
                  f'Prog: {avg_comp["progress"]:5.2f} | '
                  f'Prox: {avg_comp["proximity"]:5.2f} | '
                  f'Acti: {avg_comp["action"]:5.2f} | '
                  f'Decel: {avg_comp["decel"]:5.2f} | '
                  f'Arrive: {avg_comp["arrive"]:5.2f} | '
                  f'Time: {avg_comp["time"]:6.2f} | '
                  f'Bound: {avg_comp["boundary"]:5.2f}')
            self.recent_components = self.recent_components[-10:]

        if ((ep - self._new_episodes_start_idx) % self.EVAL_INTERVAL == 0 and
                ep > self._new_episodes_start_idx):
            det_rate = self._eval_deterministic(self.EVAL_EPISODES)
            print(f'\n[Det Eval] L{self.current_level} Ep {ep} | '
                  f'Det Success: {det_rate*100:.0f}%', end='')

            if det_rate > self.best_det_rate:
                self.best_det_rate = det_rate
                self.model.save(self.model_save_path)
                print(f' -> [Best saved: {det_rate*100:.0f}%]', end='')

            threshold = (self.FINAL_THRESHOLD
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
                        print(f'\n[Auto-Stop] Level {self.current_level} det >= '
                              f'{threshold*100:.0f}% × {checks_needed}. Training complete!')
                        _save_level_snapshot(
                            self.save_dir, self.current_level,
                            self.episode_rewards,
                            self._level_start_idx, ep,
                            self.model
                        )
                        self.should_stop_training = True
            else:
                if self.consecutive_success_checks > 0:
                    print(' -> [Reset]')
                else:
                    print()
                self.consecutive_success_checks = 0

        return not self.should_stop_training

    def _promote(self, current_ep: int):
        old_level = self.current_level
        new_level = old_level + 1

        print(f'\n{"="*60}')
        print(f'  [Curriculum] LEVEL UP!  {old_level} -> {new_level}')
        print(f'{"="*60}')

        _save_level_snapshot(
            self.save_dir, old_level,
            self.episode_rewards,
            self._level_start_idx, current_ep,
            self.model
        )

        env = self._get_env()
        env.set_curriculum_level(new_level)
        self.current_level = new_level

        self.consecutive_success_checks = 0
        self._level_start_idx = current_ep

    def _eval_deterministic(self, n_episodes: int = 10) -> float:
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
        print(f'Full rewards CSV -> {self.csv_path}')

        if not self.episode_rewards:
            return
        rewards  = self.episode_rewards
        episodes = list(range(1, len(rewards) + 1))
        window   = 20
        smoothed = []
        for i in range(len(rewards)):
            s = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[s:i + 1]))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(episodes, rewards,  color='lightblue', alpha=0.5, label='Episode reward')
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2,
                label=f'Moving mean ({window} ep)')

        for lv in range(1, DroneGymEnv.MAX_CURRICULUM_LEVEL + 1):
            lv_csv = os.path.join(self.save_dir, f'level{lv}', f'rewards_level{lv}.csv')
            if os.path.exists(lv_csv):
                with open(lv_csv, 'r') as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    last_global_ep = int(rows[-1][1])
                    if lv < DroneGymEnv.MAX_CURRICULUM_LEVEL:
                        ax.axvline(x=last_global_ep, color='orange', linestyle='--',
                                   alpha=0.7, label=f'Level {lv}→{lv+1}' if lv == 1 else '')

        if self._new_episodes_start_idx > 0:
            ax.axvline(x=self._new_episodes_start_idx, color='red', linestyle='--',
                       alpha=0.5, label='Resumed Training')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('PPO Training Curve - Curriculum Learning (Level 1→2→3→4→5)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = os.path.join(self.save_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f'Full training curve -> {png_path}')

RewardLoggerCallback = CurriculumCallback

def main():
    print('=' * 60)
    print('  PPO Training: Task B  (Curriculum Learning)')
    print('=' * 60)

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    print('Waiting for Gazebo pose data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Pose data received. Starting training.')

    MODEL_PATH = 'model_level2'

    if os.path.exists(MODEL_PATH + '.zip'):
        print(f'Model found: {MODEL_PATH}.zip, continue training...')
        model = PPO.load(
            MODEL_PATH,
            env=env,
            # custom_objects={
            #     'learning_rate': 3e-4,
            #     'batch_size': 64,
            #     'ent_coef': 0.01,
            #     'n_steps': 2048,
            # }
        )
        callback = CurriculumCallback(
            save_dir='logs',
            model_save_path='best_model'
        )
        
        callback.current_level = 1
        env.set_curriculum_level(1)

        ADDITIONAL_TIMESTEPS = 1_000_000
        print(f'\nStarting additional training for {ADDITIONAL_TIMESTEPS:,} timesteps...\n')
        try:
            model.learn(
                total_timesteps     = ADDITIONAL_TIMESTEPS,
                callback            = callback,
                progress_bar        = False,
                reset_num_timesteps = False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted. Saving...')

        NEW_MODEL_NAME = 'ppo_drone_curriculum_resume'
        model.save(NEW_MODEL_NAME)
        print(f'Model saved to {NEW_MODEL_NAME}.zip')

    else:
        print('No existing model. Training from scratch with Curriculum Learning...')
        model = PPO(
            policy          = 'MlpPolicy',
            env             = env,
            verbose         = 1,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            policy_kwargs   = dict(
                net_arch      = [256, 256],
                activation_fn = torch.nn.Tanh,
            ),
            tensorboard_log = './logs/tensorboard/',
        )
        callback = CurriculumCallback(save_dir='logs', model_save_path='best_model')

        TOTAL_TIMESTEPS = 1_000_000
        print(f'\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...\n')
        try:
            model.learn(
                total_timesteps = TOTAL_TIMESTEPS,
                callback        = callback,
                progress_bar    = False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted. Saving...')

        model.save('ppo_drone_curriculum_final')
        print('\nModel saved to ppo_drone_curriculum_final.zip')

    callback.save_curve()

    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()
    print('\nTraining complete.')


if __name__ == '__main__':
    main()