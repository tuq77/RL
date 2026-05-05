"""Watch the trained line-following car in MuJoCo viewer.

Automatically loads the best (or final) model from logs/track_follow.

Run:
    D:\\anaconda2025\\envs\\tutorial_for_mujoco\\python.exe watch_track.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from track_env import TrackFollowEnv


def find_model(log_dir: str = 'logs/track_follow') -> str | None:
    log_path = Path(log_dir)
    candidates = [
        log_path / 'best' / 'best_model.zip',
        log_path / 'track_follow_final.zip',
    ]
    ckpt_dir = log_path / 'checkpoints'
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob('track_ppo_*_steps.zip'))
        if ckpts:
            candidates.append(ckpts[-1])
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Watch line-following car')
    parser.add_argument('--model', type=str, default=None,
                        help='model path (auto-detect if omitted)')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    model_path = args.model or find_model()
    if model_path is None:
        print('[ERROR] No model found. Run train_track.py first.')
        return

    print(f'[INFO] Loading model: {model_path}')
    env = TrackFollowEnv(max_episode_steps=args.steps)
    model = PPO.load(model_path, env=env, device='cpu')

    obs, _ = env.reset()
    env.start_viewer()

    step = 0
    total_reward = 0.0
    episode = 0

    while env.viewer is not None and env.viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if hasattr(env.viewer, 'cam'):
            env.viewer.cam.lookat = env.data.xpos[env.car_body_id].copy()
        env.viewer.sync()
        time.sleep(env.ctrl_dt)

        if terminated or truncated:
            episode += 1
            print(f'  [ep={episode}  steps={step}  '
                  f'reward={total_reward:.1f}  '
                  f'track_q={info.get("track_quality", 0):.2f}  '
                  f'speed={info.get("speed", 0):.2f}]')
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0

    env.close()
    print('[INFO] Viewer closed.')


if __name__ == '__main__':
    main()
