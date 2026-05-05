"""Check training progress by reading eval logs (no TensorBoard needed).

Run:
    D:\\anaconda2025\\envs\\tutorial_for_mujoco\\python.exe check_progress.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    eval_path = Path('logs/track_follow/eval/evaluations.npz')
    if not eval_path.exists():
        print('[INFO] No eval results yet. Training just started.')
        return

    data = np.load(str(eval_path))
    timesteps = data['timesteps']
    results = data['results']       # (n_evals, n_episodes)
    ep_lengths = data['ep_lengths']

    print(f'{"Step":>10s} | {"Mean Reward":>12s} | {"Mean Length":>12s} | {"Survival":>9s}')
    print('-' * 52)

    for i in range(len(timesteps)):
        mean_r = np.mean(results[i])
        mean_len = np.mean(ep_lengths[i])
        survival = np.mean(ep_lengths[i] >= 2000) * 100

        bar = ''
        if survival >= 80:
            bar = '  ' + '█' * 10 + '  mastered!'
        elif survival >= 50:
            bar = '  ' + '█' * 5 + '  getting there'

        print(f'{timesteps[i]:>10,} | {mean_r:>+12.1f} | {mean_len:>12.0f} | {survival:>8.0f}%{bar}')

    # best
    best_idx = np.argmax(np.mean(results, axis=1))
    print(f'\nBest: step={timesteps[best_idx]:,}  reward={np.mean(results[best_idx]):.1f}  length={np.mean(ep_lengths[best_idx]):.0f}')


if __name__ == '__main__':
    main()
