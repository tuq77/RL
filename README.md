# RL — 小车循线（Line Follower）

基于 MuJoCo 物理引擎的强化学习入门项目。一辆装有 2 个红外传感器的小车，在黑色椭圆轨道上通过 PPO 算法学会沿线行驶。

## 环境

| 要素 | 说明 |
|------|------|
| 观测 | 5 维向量 `[左传感器, 右传感器, 速度vx, 速度vy, 角速度wz]` |
| 动作 | 2 维向量 `[前进速度(0~1), 转向(-1~1)]` |
| 奖励 | 压线且速度越快奖励越高，离线无奖励 |

## 项目结构

| 文件 | 说明 |
|------|------|
| `track_env.py` | Gymnasium 强化学习环境定义 |
| `train_track.py` | PPO 训练脚本（Stable-Baselines3） |
| `track_follow.py` | MuJoCo 仿真演示脚本（支持手动/PD控制器） |
| `controller.py` | PD 控制器实现 |
| `watch_track.py` | 加载训练好的模型进行可视化评估 |
| `check_progress.py` | 查看训练进度 |
| `car_track.xml` | 赛道模型文件 |
| `logs/track_follow/best/best_model.zip` | 训练完成的最佳模型 |

## 依赖

- Python 3.x
- [MuJoCo](https://mujoco.org/) 3.6+
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- NumPy

## 使用

```bash
# 训练
python train_track.py

# 查看训练进度
python check_progress.py

# 用最佳模型演示
python watch_track.py

# 手动控制 / PD控制器演示
python track_follow.py --mode pd
```
