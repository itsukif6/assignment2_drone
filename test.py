#!/usr/bin/env python3
import argparse
import numpy as np
import rclpy
from stable_baselines3 import PPO
from drone_env import DroneROSInterface, DroneGymEnv


def run_baseline_episode(ros: DroneROSInterface, target: np.ndarray,
                          arrive_dist: float, hover_steps_needed: int,
                          max_steps: int = 600) -> dict:
    KP        = 0.5
    MAX_SPEED = 1.0

    total_reward = 0.0
    success      = False
    hover_count  = 0
    prev_dist    = float(np.linalg.norm(ros.current_pose - target))

    for step in range(max_steps):
        pos   = ros.current_pose.copy()
        error = target - pos
        dist  = float(np.linalg.norm(error))

        # 連續送速度指令
        if dist < arrive_dist:
            vel = np.zeros(3)
            hover_count += 1
            if hover_count >= hover_steps_needed:
                success = True
                break
        else:
            hover_count = 0
            vel   = KP * error
            speed = float(np.linalg.norm(vel))
            if speed > MAX_SPEED:
                vel = vel * (MAX_SPEED / speed)

        start_ns = ros.get_clock().now().nanoseconds
        while (ros.get_clock().now().nanoseconds - start_ns) < 1e8:
            ros.send_velocity(*vel)
            rclpy.spin_once(ros, timeout_sec=0.01)

        curr_dist     = float(np.linalg.norm(ros.current_pose - target))
        total_reward += 5.0 * (prev_dist - curr_dist) - 0.1
        prev_dist     = curr_dist

    ros.send_velocity(0, 0, 0)
    return {'success': success, 'steps': step + 1, 'reward': total_reward}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',   type=int,  default=20)
    parser.add_argument('--model',      type=str,  default='best_model')
    parser.add_argument('--level',      type=int,  default=1,
                        help='Curriculum level (1-5)')
    parser.add_argument('--baseline',   action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    ros = DroneROSInterface()
    env = DroneGymEnv(ros)
    env.set_curriculum_level(args.level)

    print('Waiting for Gazebo pose data...')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)

    if args.baseline:
        mode  = 'baseline'
        model = None
        print(f'\nBaseline P-controller | Level {args.level} | '
              f'{args.episodes} episodes')
    else:
        model = PPO.load(args.model, env=env)
        mode  = 'rl'
        deterministic = not args.stochastic
        print(f'Model: {args.model} | Level {args.level} | '
              f'{"Stochastic" if args.stochastic else "Deterministic"}')

    results = []

    for ep in range(1, args.episodes + 1):
        obs, _    = env.reset()
        target    = env.target.copy()
        ep_reward = 0.0
        ep_steps  = 0
        success   = False

        if mode == 'baseline':
            result    = run_baseline_episode(
                ros, target, env.ARRIVE_DIST, env.HOVER_STEPS)
            success   = result['success']
            ep_steps  = result['steps']
            ep_reward = result['reward']
        else:
            for step in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps   = step + 1

                if terminated:
                    # 用 ep_components 判斷是到達還是墜機
                    success = info.get('ep_components', {}).get('arrive', 0.0) > 0
                    break
                if truncated:
                    break

        results.append({
            'episode': ep, 'success': success,
            'steps': ep_steps, 'reward': ep_reward,
        })

        status = 'Success' if success else 'Failure'
        print(f'Ep {ep:3d}/{args.episodes} | {status} | '
              f'Steps: {ep_steps:4d} | Reward: {ep_reward:8.2f} | '
              f'Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})')

    n_success    = sum(r['success'] for r in results)
    success_rate = n_success / args.episodes * 100
    print('\n' + '=' * 55)
    print(f'  Level: {args.level} | '
          f'Mode: {"Baseline" if mode == "baseline" else ("Stochastic" if args.stochastic else "Deterministic")}')
    print(f'  Success: {n_success}/{args.episodes} = {success_rate:.1f}%')
    print(f'  Mean reward: {np.mean([r["reward"] for r in results]):.2f}')
    print(f'  Mean steps:  {np.mean([r["steps"]  for r in results]):.1f}')
    print('=' * 55)

    ros.send_velocity(0, 0, 0)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()