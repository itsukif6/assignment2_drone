#!/usr/bin/env python3
"""
test.py
-------
載入訓練好的模型，在 Gazebo 跑 N 個 episode 並統計懸停成功率。
同時支援 P 控制器 baseline 比較。

使用方式:
    python3 test.py                        # 預設跑 10 episode，載入 models/ppo_drone_final
    python3 test.py --episodes 30
    python3 test.py --model models/best_stage2
    python3 test.py --baseline             # 改跑 P 控制器
    python3 test.py --stochastic           # 用 stochastic policy（預設 deterministic）
"""

import argparse
import math
import time

import numpy as np
import rclpy

from stable_baselines3 import PPO
from drone_env import DroneROSInterface, DroneGymEnv


# ================================================================
# P 控制器 Baseline
# ================================================================
def run_baseline_episode(env: DroneGymEnv, max_steps: int = 150) -> dict:
    """用比例控制器飛一個 episode，回傳結果。"""
    KP        = 0.4
    MAX_SPEED = 2.0
    ros       = env.ros

    obs, _ = env.reset()
    target = env.target_pos.copy()

    total_reward = 0.0
    success      = False

    for step in range(max_steps):
        pos   = ros.current_pose.copy()
        error = target - pos
        dist  = float(np.linalg.norm(error))

        if dist < env.ARRIVE_DIST:
            # 嘗試懸停確認（靜止 1.5 s = 15 步）
            hover = 0
            while hover < env.HOVER_STEPS_REQUIRED:
                ros.send_velocity(0.0, 0.0, 0.0)
                rclpy.spin_once(ros, timeout_sec=0.1)
                hover += 1
                if float(np.linalg.norm(ros.current_pose - target)) > env.ARRIVE_DIST:
                    break
            if hover >= env.HOVER_STEPS_REQUIRED:
                success = True
                break

        # P 控制
        vel   = KP * error
        speed = float(np.linalg.norm(vel))
        if speed > MAX_SPEED:
            vel = vel * (MAX_SPEED / speed)

        ros.send_velocity(*vel)
        rclpy.spin_once(ros, timeout_sec=0.1)

        # 簡易 reward（僅供比較用）
        curr_dist    = float(np.linalg.norm(ros.current_pose - target))
        total_reward += 35.0 * (dist - curr_dist) - 1.5

    ros.send_velocity(0.0, 0.0, 0.0)
    return {'success': success, 'steps': step + 1, 'reward': total_reward}


# ================================================================
# 主測試流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Test PPO drone model')
    parser.add_argument('--episodes',   type=int,  default=10,
                        help='Number of test episodes')
    parser.add_argument('--model',      type=str,  default='models/ppo_drone_final',
                        help='Model path (without .zip)')
    parser.add_argument('--baseline',   action='store_true',
                        help='Run P-controller baseline instead of RL')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    args = parser.parse_args()

    # --- 初始化 ---
    rclpy.init()
    ros = DroneROSInterface()
    env = DroneGymEnv(ros)

    print('Waiting for Gazebo pose...')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)
    print('Pose received.\n')

    deterministic = not args.stochastic

    if args.baseline:
        print(f'Mode: P-controller Baseline | Episodes: {args.episodes}')
        model = None
    else:
        model = PPO.load(args.model, env=env)
        policy_str = 'deterministic' if deterministic else 'stochastic'
        print(f'Mode: PPO RL ({policy_str}) | Model: {args.model} | Episodes: {args.episodes}')

    print('-' * 65)

    results = []

    for ep in range(1, args.episodes + 1):
        if args.baseline:
            result = run_baseline_episode(env)
            success = result['success']
            steps   = result['steps']
            reward  = result['reward']
        else:
            obs, _ = env.reset()
            target  = env.target_pos.copy()
            reward  = 0.0
            steps   = 0
            success = False

            for step in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(action)
                reward += r
                steps   = step + 1

                if info.get('hover_success', False):
                    success = True

                if terminated or truncated:
                    break

        results.append({'success': success, 'steps': steps, 'reward': reward})

        status = 'Success' if success else 'Failure'
        tgt    = env.target_pos
        print(f'Ep {ep:3d}/{args.episodes} | {status} | '
              f'Steps: {steps:4d} | Reward: {reward:8.2f} | '
              f'Target: ({tgt[0]:.1f}, {tgt[1]:.1f}, {tgt[2]:.1f}) | '
              f'ARRIVE_DIST: {env.ARRIVE_DIST:.1f}')

    # --- 統計 ---
    n_success    = sum(r['success'] for r in results)
    success_rate = n_success / args.episodes * 100
    mean_reward  = np.mean([r['reward'] for r in results])
    mean_steps   = np.mean([r['steps']  for r in results])

    print('\n' + '=' * 65)
    mode_str = 'P-controller Baseline' if args.baseline else f'PPO RL ({policy_str})'
    print(f'  Mode         : {mode_str}')
    print(f'  Success rate : {n_success}/{args.episodes} = {success_rate:.1f}%')
    print(f'  Mean reward  : {mean_reward:.2f}')
    print(f'  Mean steps   : {mean_steps:.1f}')
    print('=' * 65)

    ros.send_velocity(0.0, 0.0, 0.0)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()