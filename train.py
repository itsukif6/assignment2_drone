#!/usr/bin/env python3
"""
train.py
--------
PPO Algorithm for Drone Random Target Navigation (Task B).

Hyperparameters references:
    Paper 1: Tan & Karakose (2023)
    Paper 2: Zhang et al. (2024)
    Paper 3: Shen & Huang (2024)
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


class RewardLoggerCallback(BaseCallback):
    """
    Custom Callback to log rewards and handle automatic termination.
    """
    def __init__(self, save_dir: str = 'logs', verbose=0):
        super().__init__(verbose)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.csv_path = os.path.join(self.save_dir, 'rewards.csv')
        
        self.episode_rewards = []
        self._new_episodes_start_idx = 0
        self._current_ep_reward = 0.0
        self.recent_components = []

        self.consecutive_success_checks = 0
        self.should_stop_training = False

        self._load_history()

    def _load_history(self):
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) == 2:
                            self.episode_rewards.append(float(row[1]))
                print(f"Loaded {len(self.episode_rewards)} historical episodes from {self.csv_path}")
            except Exception as e:
                print(f"Error loading history csv: {e}")
        
        self._new_episodes_start_idx = len(self.episode_rewards)

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
            
            if len(self.recent_components) >= 10:
                if (ep - self._new_episodes_start_idx) % 10 == 0:
                    recent_mean = np.mean(self.episode_rewards[-10:])
                    
                    avg_comp = {k: 0.0 for k in self.recent_components[0].keys()}
                    for comp in self.recent_components[-10:]:
                        for k, v in comp.items():
                            avg_comp[k] += v
                    for k in avg_comp.keys():
                        avg_comp[k] /= 10.0

                    print(f'Ep {ep:4d} | Total: {recent_mean:7.2f} | '
                          f'Prog: {avg_comp["progress"]:6.2f} | '
                          f'Decel: {avg_comp["decel"]:6.2f} | '
                          f'Hover: {avg_comp["hover"]:6.2f} | '
                          f'Arrive: {avg_comp["arrive"]:5.2f} | '
                          f'Time: {avg_comp["time"]:6.2f} | '
                          f'Bound: {avg_comp["boundary"]:6.2f} | ', end='')
                    
                    # 畢業條件：到達率 >= 60% 且無墜機
                    if avg_comp["arrive"] >= 60.00 and avg_comp["boundary"] == 0.00:
                        self.consecutive_success_checks += 1
                        print(f'-> [Success: {self.consecutive_success_checks}/5]')
                        
                        if self.consecutive_success_checks >= 5:
                            print('\n\n[Auto-Stop] Successful training.')
                            print('Stopping and saving...\n')
                            self.should_stop_training = True
                    else:
                        if self.consecutive_success_checks > 0:
                            print('-> [Stop]')
                        else:
                            print('') 
                        self.consecutive_success_checks = 0
                    
                    self.recent_components = self.recent_components[-10:]

        return not self.should_stop_training

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
                
        print(f'Raw data appended/saved: {self.csv_path}')

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
        ax.plot(episodes, smoothed, color='steelblue', linewidth=2, label=f'Move mean: ({window} rounds)')
        
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


def main():
    print('=' * 60)
    print('  PPO Training: Task B Random Target Navigation')
    print('=' * 60)

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    print('Waiting for Gazebo pose data...')
    while not ros_interface.pose_received:
        rclpy.spin_once(ros_interface, timeout_sec=0.5)
    print('Pose data received. Starting training.')

    MODEL_PATH = "ppo_drone_x"

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model found: {MODEL_PATH}.zip, continue training...")
        
        # 載入模型並套用強勢破局的參數 (動能恢復與探索重啟)
        model = PPO.load(
            MODEL_PATH, 
            env=env, 
            custom_objects={
            'learning_rate': 1e-4, 
            'ent_coef': 0.005,      
            'max_grad_norm': 0.5  
        })

        callback = RewardLoggerCallback(save_dir='logs')
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

        NEW_MODEL_NAME = "ppo_drone_0.75.zip"
        model.save(NEW_MODEL_NAME)
        print(f'\nModel saved to {NEW_MODEL_NAME}')
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
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            policy_kwargs = dict(
                net_arch       = [256, 256],
                activation_fn  = torch.nn.Tanh,
            ),
            tensorboard_log = './logs/tensorboard/',
        )

        callback = RewardLoggerCallback(save_dir='logs')
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

        model.save('ppo_drone_new')
        print('\nModel saved to ppo_drone_new.zip')

    callback.save_curve()

    ros_interface.send_velocity(0, 0, 0)
    ros_interface.destroy_node()
    rclpy.shutdown()
    print('\nTraining complete.')


if __name__ == '__main__':
    main()