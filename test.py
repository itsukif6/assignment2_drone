#!/usr/bin/env python3
"""
test.py
-------
載入訓練好的 PPO 模型, 在 Gazebo 裡跑 N 個 episode 並統計成功率. 
同時和 fly_straight.py 的 P 控制器做比較. 

使用方式: 
    python3 test.py                    # 預設跑 10 個 episode
    python3 test.py --episodes 20      # 指定跑 20 個 episode
    python3 test.py --model my_model   # 指定模型檔名  ( 不含 .zip ) 
    python3 test.py --baseline         # 改跑 P 控制器  ( 作為 baseline 比較 ) 

參考論文: 
    Paper 1: A new approach for drone tracking with drone using Proximal Policy Optimization based distributed deep reinforcement learning
    Paper 2: AirPilot Interpretable PPO-based DRL Auto Tuned Nonlinear PID Drone Controller for Robust Autonomous Flights
    Paper 3: Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions
"""

import argparse
import math
import numpy as np
import rclpy

from stable_baselines3 import PPO
from drone_env import DroneROSInterface, DroneGymEnv


# ================================================================
# P 控制器 Baseline  ( 和 fly_straight.py 相同邏輯 ) 
# 用來和 RL agent 比較
# ================================================================
def run_baseline_episode(ros: DroneROSInterface, target: np.ndarray,
                          max_steps: int = 300) -> dict:
    """
    用 P 控制器飛一個 episode, 回傳結果統計. 
    Kp 和 max_speed 與 fly_straight.py 預設值相同. 
    """
    KP        = 0.5
    MAX_SPEED = 1.0
    ARRIVE    = 0.3

    total_reward = 0.0
    success = False
    prev_dist = float(np.linalg.norm(ros.current_pose - target))

    for step in range(max_steps):
        pos = ros.current_pose.copy()
        error = target - pos
        dist  = float(np.linalg.norm(error))

        # 到達判斷
        if dist < ARRIVE:
            success = True
            break

        # P 控制計算速度
        vel = KP * error
        speed = float(np.linalg.norm(vel))
        if speed > MAX_SPEED:
            vel = vel * (MAX_SPEED / speed)

        ros.send_velocity(*vel)
        rclpy.spin_once(ros, timeout_sec=0.1)

        # 簡易 reward  ( 方便和 RL 比較 ) 
        curr_dist = float(np.linalg.norm(ros.current_pose - target))
        total_reward += 5.0 * (prev_dist - curr_dist) - 0.1
        prev_dist = curr_dist

    ros.send_velocity(0, 0, 0)
    return {'success': success, 'steps': step + 1, 'reward': total_reward}


# ================================================================
# 主測試流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Test PPO drone model')
    parser.add_argument('--episodes', type=int,   default=10,        help='Test rounds')
    parser.add_argument('--model',    type=str,   default='ppo_drone', help='Model name without zip')
    parser.add_argument('--baseline', action='store_true',            help='Run P controller baseline')
    args = parser.parse_args()

    # --- 初始化 ---
    rclpy.init()
    ros = DroneROSInterface()
    env = DroneGymEnv(ros)

    # 等待位置資料
    print('Wait for Gazebo place data... ')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)

    # --- 決定跑 RL 還是 baseline ---
    if args.baseline:
        print(f'\nRun P controller Baseline, total: {args.episodes} episodes')
        mode = 'baseline'
        model = None
    else:
        print(f'\nLoad model: {args.model}.zip')
        model = PPO.load(args.model)
        mode  = 'rl'
        print(f'Model load success, start testing {args.episodes} episodes\n')

    # --- 跑測試 ---
    results = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        target = env.target.copy()
        ep_reward = 0.0
        ep_steps  = 0
        success   = False

        if mode == 'baseline':
            # P 控制器模式
            result = run_baseline_episode(ros, target)
            success   = result['success']
            ep_steps  = result['steps']
            ep_reward = result['reward']

        else:
            # RL agent 模式
            for step in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps   = step + 1

                if terminated:
                    # 到達 or 出界都會 terminated
                    # 用距離判斷是真的到達還是出界
                    dist = float(np.linalg.norm(ros.current_pose - target))
                    success = dist < env.ARRIVE_DIST
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
              f'Step: {ep_steps:4d} | reward: {ep_reward:8.2f} | '
              f'Target:  ( {target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f} ) ')

    # --- 統計結果 ---
    n_success = sum(r['success'] for r in results)
    success_rate = n_success / args.episodes * 100
    mean_reward  = np.mean([r['reward']  for r in results])
    mean_steps   = np.mean([r['steps']   for r in results])

    print('\n' + '=' * 55)
    print(f'  Test mode: {"P controller Baseline" if mode == "baseline" else "PPO RL Agent"}')
    print(f'  Success rate  : {n_success}/{args.episodes} = {success_rate:.1f}%')
    print(f'  Mean reward: {mean_reward:.2f}')
    print(f'  Mean steps: {mean_steps:.1f}')
    print('=' * 55)

    # --- 清理 ---
    ros.send_velocity(0, 0, 0)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()