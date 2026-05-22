#!/usr/bin/env python3
"""
test.py
-------
Load trained PPO model, run N episodes in Gazebo and calculate success rate.
Includes P-controller baseline comparison.
"""

import argparse
import numpy as np
import rclpy

from stable_baselines3 import PPO
from drone_env import DroneROSInterface, DroneGymEnv


def run_baseline_episode(ros: DroneROSInterface, target: np.ndarray,
                          max_steps: int = 400) -> dict:
    """
    P-Controller baseline.
    Requires 15 consecutive steps (1.5s) within target radius to succeed.
    """
    KP        = 0.5
    MAX_SPEED = 1.0
    ARRIVE    = 0.5

    total_reward = 0.0
    success = False
    prev_dist = float(np.linalg.norm(ros.current_pose - target))
    success_steps = 0

    for step in range(max_steps):
        pos = ros.current_pose.copy()
        error = target - pos
        dist  = float(np.linalg.norm(error))

        # 懸停判定邏輯
        if dist < ARRIVE:
            success_steps += 1
            if success_steps >= 15:
                success = True
                break
        else:
            success_steps = 0

        vel = KP * error
        speed = float(np.linalg.norm(vel))
        if speed > MAX_SPEED:
            vel = vel * (MAX_SPEED / speed)

        ros.send_velocity(*vel)
        rclpy.spin_once(ros, timeout_sec=0.1)

        curr_dist = float(np.linalg.norm(ros.current_pose - target))
        total_reward += 5.0 * (prev_dist - curr_dist) - 0.1
        prev_dist = curr_dist

    ros.send_velocity(0, 0, 0)
    return {'success': success, 'steps': step + 1, 'reward': total_reward}


def main():
    parser = argparse.ArgumentParser(description='Test PPO drone model')
    parser.add_argument('--episodes', type=int,   default=10,        help='Test rounds')
    parser.add_argument('--model',    type=str,   default='ppo_drone', help='Model name without zip')
    parser.add_argument('--baseline', action='store_true',            help='Run P controller baseline')
    args = parser.parse_args()

    rclpy.init()
    ros = DroneROSInterface()
    env = DroneGymEnv(ros)

    print('Wait for Gazebo pose data...')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)

    if args.baseline:
        print(f'\nRun P-Controller Baseline, total: {args.episodes} episodes')
        mode = 'baseline'
        model = None
    else:
        print(f'\nLoad model: {args.model}.zip')
        model = PPO.load(args.model)
        mode  = 'rl'
        print(f'Model load success, start testing {args.episodes} episodes\n')

    results = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        target = env.target.copy()
        ep_reward = 0.0
        ep_steps  = 0
        success   = False

        if mode == 'baseline':
            result = run_baseline_episode(ros, target, env.MAX_STEPS)
            success   = result['success']
            ep_steps  = result['steps']
            ep_reward = result['reward']
        else:
            for step in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps   = step + 1

                if terminated:
                    # 確認是否為懸停成功
                    dist = float(np.linalg.norm(ros.current_pose - target))
                    success = (dist < env.ARRIVE_DIST)
                    break

                if truncated:
                    break

        results.append({
            'episode': ep,
            'success': success,
            'steps':   ep_steps,
            'reward':  ep_reward,
        })

        status = 'Success' if success else 'Failure'
        print(f'Episode {ep:3d}/{args.episodes} | {status} | '
              f'Step: {ep_steps:4d} | Reward: {ep_reward:8.2f} | '
              f'Target: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})')

    n_success = sum(r['success'] for r in results)
    success_rate = n_success / args.episodes * 100
    mean_reward  = np.mean([r['reward']  for r in results])
    mean_steps   = np.mean([r['steps']   for r in results])

    print('\n' + '=' * 55)
    print(f'  Test mode: {"P-Controller Baseline" if mode == "baseline" else "PPO RL Agent"}')
    print(f'  Success rate : {n_success}/{args.episodes} = {success_rate:.1f}%')
    print(f'  Mean reward  : {mean_reward:.2f}')
    print(f'  Mean steps   : {mean_steps:.1f}')
    print('=' * 55)

    ros.send_velocity(0, 0, 0)
    ros.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()