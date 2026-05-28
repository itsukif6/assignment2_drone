#!/usr/bin/env python3
"""
test.py
-------
載入訓練好的 PPO 模型，在 Gazebo 裡跑 N 個 episode 並統計：
  - 成功率（EffectiveSpeed 判定：連續懸停 HOVER_STEPS 步）
  - 平均 EffectiveSpeed = initial_dist / ep_steps（越高越好）
  - P 控制器 baseline 比較

使用方式：
    python3 test.py                         # RL，Level 1，10 episodes
    python3 test.py --episodes 20           # 指定 episode 數
    python3 test.py --model best_model      # 指定模型（不含 .zip）
    python3 test.py --level 3               # 指定課程難度
    python3 test.py --baseline              # 改跑 P 控制器
    python3 test.py --stochastic            # 使用隨機策略（預設 deterministic）
"""

import argparse
import numpy as np
import rclpy
from stable_baselines3 import PPO
from drone_env import DroneROSInterface, DroneGymEnv


# ================================================================
# P 控制器 Baseline
# ================================================================
def run_baseline_episode(env: DroneGymEnv, max_steps: int = 600) -> dict:
    """
    用 P 控制器飛一個 episode，但改為透過 env.step() 互動，
    確保 Baseline 與 PPO 享有 100% 相同的 Reward 計算標準與成功判定。
    """
    KP = 0.5
    
    ep_reward = 0.0
    ep_steps = 0
    success = False
    eff_speed = 0.0

    for step in range(max_steps):
        # 取得當前真實座標與目標
        pos = env.ros.current_pose
        target = env.target
        error = target - pos
        
        dist = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(env.ros.current_vel))

        # P 控制邏輯 (保留原本的硬編碼強制煞車策略)
        if dist < env.ARRIVE_DIST and speed < env.HOVER_MAX_SPEED:
            vel = np.zeros(3)
        else:
            vel = KP * error
            vel_norm = float(np.linalg.norm(vel))
            if vel_norm > env.MAX_SPEED:
                vel = vel * (env.MAX_SPEED / vel_norm)

        # 動作轉換：環境中 real_velocity = action * 0.8，因此 action = vel / 0.8
        action = vel / 0.8
        
        # 透過 Gym 環境步進，讓環境統一計算 Reward 與碰撞/超時判定
        obs, reward, terminated, truncated, info = env.step(action)
        
        ep_reward += reward
        ep_steps = step + 1

        if terminated:
            # 判定是否成功完成 HOVER_STEPS 懸停
            success = info.get('ep_components', {}).get('arrive', 0.0) > 0
            eff_speed = info.get('effective_speed', 0.0)
            break
            
        if truncated:
            eff_speed = info.get('effective_speed', 0.0)
            break

    return {
        'success': success,
        'steps': ep_steps,
        'reward': ep_reward,
        'effective_speed': eff_speed,
        'init_dist': env.initial_dist,
    }

# ================================================================
# 主測試流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Test PPO drone model')
    parser.add_argument('--episodes',   type=int,  default=10,
                        help='Number of test episodes')
    parser.add_argument('--model',      type=str,  default='best_model',
                        help='Model filename without .zip')
    parser.add_argument('--level',      type=int,  default=1,
                        help='Curriculum level (1~7)')
    parser.add_argument('--baseline',   action='store_true',
                        help='Run P-controller baseline instead of RL')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    args = parser.parse_args()

    # ── 初始化 ────────────────────────────────────────────────────
    rclpy.init()
    ros = DroneROSInterface()
    env = DroneGymEnv(ros)
    env.set_curriculum_level(args.level)

    print('Waiting for Gazebo pose data...')
    while not ros.pose_received:
        rclpy.spin_once(ros, timeout_sec=0.5)

    # ── 決定跑 RL 還是 baseline ───────────────────────────────────
    if args.baseline:
        mode  = 'baseline'
        model = None
        print(f'\nBaseline P-controller | Level {args.level} | '
              f'{args.episodes} episodes')
    else:
        model         = PPO.load(args.model, env=env)
        mode          = 'rl'
        deterministic = not args.stochastic
        print(f'\nModel: {args.model} | Level {args.level} | '
              f'{"Stochastic" if args.stochastic else "Deterministic"} | '
              f'{args.episodes} episodes')

    print(f'Success criteria: dist < ARRIVE_DIST({env.ARRIVE_DIST:.2f}m) + '
          f'speed < HOVER_MAX({env.HOVER_MAX_SPEED:.2f}m/s) '
          f'for {env.HOVER_STEPS} steps\n')

    # ── 跑測試 ────────────────────────────────────────────────────
    results = []

    for ep in range(1, args.episodes + 1):
        obs, _    = env.reset()
        target    = env.target.copy()
        init_dist = env.initial_dist   # 回合起點距離
        ep_reward = 0.0
        ep_steps  = 0
        success   = False
        eff_speed = 0.0

        if mode == 'baseline':
            result    = run_baseline_episode(
                env,
                max_steps = env.MAX_STEPS
            )
            success   = result['success']
            ep_steps  = result['steps']
            ep_reward = result['reward']
            eff_speed = result['effective_speed']

        else:
            for step in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps   = step + 1

                if terminated:
                    # 成功判定：ep_components['arrive'] > 0
                    # 代表完成了 HOVER_STEPS 步連續懸停
                    success   = info.get('ep_components', {}).get('arrive', 0.0) > 0
                    eff_speed = info.get('effective_speed', 0.0)
                    break
                if truncated:
                    eff_speed = info.get('effective_speed', 0.0)
                    break

        results.append({
            'episode':         ep,
            'success':         success,
            'steps':           ep_steps,
            'reward':          ep_reward,
            'effective_speed': eff_speed,
            'init_dist':       init_dist,
        })

        status = 'Success' if success else 'Failure'
        print(f'Ep {ep:3d}/{args.episodes} | {status} | '
              f'Steps: {ep_steps:4d} | Reward: {ep_reward:8.2f} | '
              f'EffSpd: {eff_speed:.3f} m/s | '
              f'Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})')

    # ── 統計結果 ──────────────────────────────────────────────────
    n_success    = sum(r['success'] for r in results)
    success_rate = n_success / args.episodes * 100
    mean_reward  = np.mean([r['reward']          for r in results])
    mean_steps   = np.mean([r['steps']           for r in results])
    mean_effspd  = np.mean([r['effective_speed'] for r in results])

    # 只對成功回合計算 EffectiveSpeed（更有意義）
    succ_effspd  = [r['effective_speed'] for r in results if r['success']]
    mean_succ_es = float(np.mean(succ_effspd)) if succ_effspd else 0.0

    print('\n' + '=' * 65)
    print(f'  Level : {args.level} | '
          f'Mode: {"Baseline" if mode == "baseline" else ("Stochastic RL" if args.stochastic else "Deterministic RL")}')
    print(f'  Success     : {n_success}/{args.episodes} = {success_rate:.1f}%')
    print(f'  Mean reward : {mean_reward:.2f}')
    print(f'  Mean steps  : {mean_steps:.1f}')
    print(f'  EffSpd (all)    : {mean_effspd:.4f} m/s')
    print(f'  EffSpd (success): {mean_succ_es:.4f} m/s')
    print('=' * 65)

    # ── 清理 ──────────────────────────────────────────────────────
    ros.send_velocity(0.0, 0.0, 0.0)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()