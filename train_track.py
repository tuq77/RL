"""Train the line-following car using PPO (Stable-Baselines3).

This is a beginner-friendly training script.  It trains a small neural
network to read two binary IR sensors and output steering + speed commands.

Run:
    D:\\anaconda2025\\envs\\tutorial_for_mujoco\\python.exe train_track.py

What to watch for:
  - ``track_quality`` should rise toward 1.0 (car stays on the line)
  - ``speed`` should increase as the car gains confidence
  - ``ep_len_mean`` (episode length) approaching 2000 means the car survives the full episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from track_env import TrackFollowEnv


def make_env(rank: int = 0, seed: int = 0) -> callable:
    def _init():
        env = TrackFollowEnv(max_episode_steps=2000)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description='Train line-following car with PPO')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='total training timesteps (default: 500K)')
    parser.add_argument('--log-dir', type=str, default='logs/track_follow',
                        help='directory for logs & checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to a saved model .zip to resume from')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── environments ──
    train_env = DummyVecEnv([make_env(rank=0, seed=42)])
    eval_env = DummyVecEnv([make_env(rank=0, seed=999)])

    # ── PPO hyper-parameters ──
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir / 'tb'),
        device='cpu',
        policy_kwargs=policy_kwargs,
    )

    if args.resume:
        print(f'[INFO] Resuming from {args.resume}')
        model = PPO.load(args.resume, env=train_env, **ppo_kwargs)
    else:
        model = PPO('MlpPolicy', train_env, **ppo_kwargs)

    # ── callbacks ──
    ckpt_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(log_dir / 'checkpoints'),
        name_prefix='track_ppo',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / 'best'),
        log_path=str(log_dir / 'eval'),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # ── train ──
    print(f'[INFO] Training PPO for {args.timesteps:,} timesteps')
    print(f'[INFO] Logs → {log_dir}')
    print(f'[INFO] Start TensorBoard:  tensorboard --logdir {log_dir / "tb"}')
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=False,
    )

    # ── save final ──
    final_path = str(log_dir / 'track_follow_final')
    model.save(final_path)
    print(f'[INFO] Final model saved to {final_path}.zip')

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
