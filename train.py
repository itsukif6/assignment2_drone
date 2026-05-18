#!/usr/bin/env python3
"""
train.py  (修正版)
------------------
配合 drone_env.py 的修正，主要差異：
  - observation_space 已從 13 維改為 7 維，不需要手動調整，PPO 會自動對齊
  - 其餘超參數邏輯不變
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
# Callback
# ================================================================
class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_dir: str = 'logs', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.episode_rewards  = []
        self._current_ep_reward = 0.0
        self.recent_components  = []

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals['rewards'][0]

        for info in self.locals.get('infos', []):
            if 'ep_components' in info:
                self.recent_components.append(info['ep_components'])

        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            ep = len(self.episode_rewards)
            if ep % 10 == 0:
                recent_mean = np.mean(self.episode_rewards[-10:])

                if self.recent_components:
                    avg_comp = {k: 0.0 for k in self.recent_components[0].keys()}
                    recent_10 = self.recent_components[-10:]
                    for comp in recent_10:
                        for k, v in comp.items():
                            avg_comp[k] += v
                    for k in avg_comp:
                        avg_comp[k] /= len(recent_10)

                    print(
                        f'Episode {ep:4d} | Mean: {recent_mean:7.2f} | '
                        f'Prog: {avg_comp["progress"]:6.2f} | '
                        f'Prox: {avg_comp["proximity"]:5.2f} | '
                        f'Arrive: {avg_comp["arrive"]:5.2f} | '
                        f'Time: {avg_comp["time"]:6.2f} | '
                        f'Bound: {avg_comp["boundary"]:6.2f}'
                    )
                else:
                    print(f'Episode {ep:4d} | Mean: {recent_mean:7.2f}')

                self.recent_components = self.recent_components[-10:]

        return True

    def save_curve(self):
        csv_path = os.path.join(self.save_dir, 'rewards.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])
        print(f'Raw data saved: {csv_path}')

        episodes = list(range(1, len(self.episode_rewards) + 1))
        rewards  = self.episode_rewards
        window   = 20
        smoothed = [
            np.mean(rewards[max(0, i - window + 1): i + 1])
            for i in range(len(rewards))
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards,  color='lightblue', alpha=0.5, label='Round reward')
        ax.plot(episodes, smoothed, color='steelblue',  linewidth=2,
                label=f'Moving mean ({window} rounds)')
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
    print('  PPO Training: Task B Random Target Navigation (Fixed)')
    print('=' * 60)

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    print('Waiting for Gazebo pose data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Pose data received. Starting training.')

    MODEL_PATH = 'ppo_drone'

    if os.path.exists(MODEL_PATH + '.zip'):
        print(f'Model found: {MODEL_PATH}.zip, continue training...')
        model = PPO.load(MODEL_PATH, env=env)
        callback = RewardLoggerCallback(save_dir='logs')
        ADDITIONAL_TIMESTEPS = 300_000
        print(f'\nStarting additional training for {ADDITIONAL_TIMESTEPS:,} timesteps...\n')
        try:
            model.learn(
                total_timesteps=ADDITIONAL_TIMESTEPS,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted. Saving current progress...')
        NEW_MODEL_NAME = 'ppo_drone_continued'
        model.save(NEW_MODEL_NAME)
        print(f'\nModel saved to {NEW_MODEL_NAME}.zip')

    else:
        print('Starting fresh training...')
        model = PPO(
            policy        = 'MlpPolicy',
            env           = env,
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = 1024,
            batch_size    = 64,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            policy_kwargs = dict(
                net_arch      = [256, 256],
                activation_fn = torch.nn.Tanh,
            ),
            tensorboard_log='./logs/tensorboard/',
        )

        callback = RewardLoggerCallback(save_dir='logs')
        TOTAL_TIMESTEPS = 600_000
        print(f'\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...\n')
        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted. Saving current progress...')

        model.save(MODEL_PATH)
        print(f'\nModel saved to {MODEL_PATH}.zip')

    callback.save_curve()

    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()

    print('\nTraining complete.')
    print(f'  Model  : {MODEL_PATH}.zip')
    print('  Curve  : logs/training_curve.png')
    print('  Data   : logs/rewards.csv')


if __name__ == '__main__':
    main()