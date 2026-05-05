"""Gymnasium environment for line-following car with MuJoCo.
==============================================================
强化学习入门 —— 小车循线环境

一个装有 2 个红外传感器的小车，需要在黑色椭圆轨道上学会沿线行驶。
智能体接收传感器的二值读数（0=压线 / 1=离线），输出前进速度和转向指令。

RL 三要素：
  观测 (Observation) → 5 维向量: [左传感器, 右传感器, 速度vx, 速度vy, 角速度wz]
  动作 (Action)      → 2 维向量: [前进速度, 转向]  forward ∈ [0, 1], steering ∈ [-1, 1]
  奖励 (Reward)      → 压线 + 速度越快奖励越高, 离线无奖励

这是为 RL 初学者设计的最小化环境，覆盖完整流程：
环境 → 训练 → 部署
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym       # OpenAI 定义的标准 RL 环境接口
import mujoco                   # MuJoCo 物理引擎
import numpy as np
from gymnasium import spaces   # 用于定义观测空间和动作空间

# MuJoCo 小车基础模型路径（包含车身、轮子、执行器定义）
BASE_MODEL_PATH = Path(r'D:\mujoco-3.6.0-windows-x86_64\model\car\car.xml')
# 生成带轨道的模型文件路径（在基础模型上叠加黑色轨道）
TRACK_MODEL_PATH = Path(__file__).resolve().parent / 'car_track.xml'


# =========================================================================
# 轨道生成
# =========================================================================

def _make_track_points(num_points: int = 120) -> np.ndarray:
    """生成不规则椭圆轨道的采样点（120 个点围成闭合曲线）。

    使用极坐标参数方程，通过叠加正弦扰动让轨道不再是完美椭圆，
    这样小车需要持续微调转向，而不是简单转圈。

    Returns:
        np.ndarray: shape (120, 2), 每行是轨道中心线上一个点的 (x, y) 坐标
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    a, b = 1.2, 0.7  # 椭圆半长轴和半短轴
    # sin(3θ) 和 sin(5θ) 叠加不规则扰动，形成"拧巴"的椭圆
    x = (a + 0.15 * np.sin(3.0 * angles)) * np.cos(angles)
    y = (b + 0.1 * np.sin(5.0 * angles)) * np.sin(angles)
    return np.stack([x, y], axis=1)


def _create_track_xml(out_xml: Path) -> None:
    """基于小车基础模型，生成带有黑色轨道的 MuJoCo XML 文件。

    轨道由 120 个细长矩形 (box geom) 依次拼接而成，每个矩形的朝向
    与轨道切线方向对齐。轨道仅用于视觉显示和红外传感器检测，
    不与车轮发生物理碰撞。

    Args:
        out_xml: 输出 XML 文件路径
    """
    tree = ET.parse(BASE_MODEL_PATH)
    root = tree.getroot()

    # 确保 <asset> 中存在黑色材质定义
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    if root.find("./asset/material[@name='track']") is None:
        ET.SubElement(asset, 'material', name='track', rgba='0 0 0 1')

    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('Base model is missing <worldbody>')

    # 清除旧的轨道线段（避免重复添加）
    for geom in worldbody.findall('geom'):
        if geom.get('name', '').startswith('track-seg'):
            worldbody.remove(geom)

    points = _make_track_points()
    half_w = 0.02  # 轨道半宽 (2cm)

    for i, p0 in enumerate(points):
        p1 = points[(i + 1) % len(points)]  # 下一采样点（闭合环）
        mid = (p0 + p1) / 2.0                # 线段中点
        direction = p1 - p0                  # 线段方向向量
        length = float(np.linalg.norm(direction))  # 线段长度
        theta = float(np.arctan2(direction[1], direction[0]))  # 线段朝向角

        # 添加一个细长 box 作为轨道线段
        # conaffinity='0' 确保不与车轮产生物理碰撞（纯视觉+传感器用）
        worldbody.append(ET.Element(
            'geom', name=f'track-seg-{i}', type='box',
            pos=f'{mid[0]:.4f} {mid[1]:.4f} 0.001',
            size=f'{length / 2.0:.4f} {half_w:.4f} 0.001',
            material='track', euler=f'0 0 {theta:.4f}',
            contype='0', conaffinity='0',
        ))

    tree.write(out_xml, xml_declaration=True, encoding='utf-8')


# =========================================================================
# 工具函数
# =========================================================================

def _shortest_distance_to_track(world_pos: np.ndarray, track_pts: np.ndarray) -> float:
    """计算世界坐标系中某点到轨道中心线的最短距离。

    遍历所有 120 条轨道线段，用点到线段的投影公式计算距离，
    取最小值。这个距离决定了红外传感器是否"看到"黑色轨道。

    Args:
        world_pos: 世界坐标系中的查询点 (x, y, z)
        track_pts: 轨道采样点数组 (120, 2)

    Returns:
        最短距离（米）
    """
    best = float('inf')
    n = len(track_pts)
    for i in range(n):
        p0 = track_pts[i]
        p1 = track_pts[(i + 1) % n]
        seg = p1 - p0  # 线段向量
        if np.allclose(seg, 0.0):
            # 退化情况：两点重合
            best = min(best, float(np.linalg.norm(world_pos[:2] - p0)))
            continue
        # 点到线段投影参数 t ∈ [0, 1]
        t = np.dot(world_pos[:2] - p0, seg) / np.dot(seg, seg)
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + t * seg  # 投影点
        best = min(best, float(np.linalg.norm(world_pos[:2] - proj)))
    return best


# =========================================================================
# Gymnasium 环境类
# =========================================================================

class TrackFollowEnv(gym.Env):
    """小车循线 RL 环境。

    符合 Gymnasium (OpenAI Gym 继任者) 标准接口，
    可直接用于 Stable-Baselines3 等 RL 库。

    ── 观测空间 (5 维连续) ──
      索引  名称          含义                      范围
      0     left_sensor   左红外传感器 (0=压黑线)    [0, 1]
      1     right_sensor  右红外传感器 (0=压黑线)    [0, 1]
      2     vx            世界坐标系 x 方向速度      (-∞, +∞)
      3     vy            世界坐标系 y 方向速度      (-∞, +∞)
      4     wz            绕 z 轴角速度 (偏航率)     (-∞, +∞)

    ── 动作空间 (2 维连续) ──
      索引  名称        范围     含义
      0     forward     [-1, 1]  前进速度指令 (1=全速前进)
      1     steering    [-1, 1]  转向指令 (正=右转, 负=左转)

    ── 奖励函数 ──
      track_quality: 1.0 双传感器压线, 0.5 单传感器压线, 0.0 离线
      speed_bonus:   速度越快奖励越高, 但仅在压线时生效
      总奖励 = track_quality × 0.5 + speed_bonus + 存活奖励

    ── 终止条件 ──
      1. 连续 150 步离线 → terminated (失败)
      2. 达到 max_episode_steps (默认 2000) → truncated (成功存活)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 2000,
        ctrl_dt: float = 0.02,
    ) -> None:
        """初始化环境。

        Args:
            render_mode: 渲染模式，'human' 表示启用 MuJoCo viewer
            max_episode_steps: 单轮最大步数（超过则截断）
            ctrl_dt: 控制周期（秒），每 0.02 秒执行一次动作
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_episode_steps
        self.ctrl_dt = ctrl_dt

        # 生成轨道 XML 并加载 MuJoCo 模型
        _create_track_xml(TRACK_MODEL_PATH)

        # ── MuJoCo 模型与数据 ──
        # MjModel: 模型定义（几何、关节、执行器等），只读
        # MjData:  仿真状态（位置、速度、传感器等），每步更新
        self.model = mujoco.MjModel.from_xml_path(str(TRACK_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # 物理仿真步长 2ms
        # 控制周期 / 物理步长 = 每步动作需要执行的物理子步数
        self.n_substeps = max(1, int(ctrl_dt / self.model.opt.timestep))

        # ── 执行器索引（缓存 ID 避免每次字符串查找）──
        self.act_forward_id = self.model.actuator('forward').id   # 前进电机
        self.act_turn_id = self.model.actuator('turn').id         # 转向电机
        self.car_body_id = self.model.body('car').id              # 车身（用于坐标变换）

        # ── 红外传感器配置 ──
        # 两个传感器安装在小车前方，左右各偏移 1.8cm
        # 传感器位置在车身局部坐标系中定义
        self.sensor_offsets = np.array([
            [0.09, 0.018, 0.0],   # 左传感器：前方 9cm，左偏 1.8cm
            [0.09, -0.018, 0.0],  # 右传感器：前方 9cm，右偏 1.8cm
        ])
        self.track_points = _make_track_points()  # 轨道中心线采样点
        self.track_half_width = 0.024             # 轨道半宽 2.4cm（略大于视觉宽度）

        # ── 定义 RL 空间 ──
        # 动作空间: 2 维连续向量
        # forward ∈ [0, 1] 禁止倒车，steering ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,), dtype=np.float32,
        )
        # 观测空间: 5 维连续向量（传感器×2 + 速度×2 + 角速度×1）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32,
        )

        # ── 内部状态 ──
        self.step_count = 0          # 当前回合的步数计数
        self.off_track_counter = 0   # 连续离线步数计数
        self.max_off_track = 150     # 连续离线超过此值 → 终止

    # =====================================================================
    # 传感器计算（核心逻辑）
    # =====================================================================

    def _local_to_world(self, local_pos: np.ndarray) -> np.ndarray:
        """将车身局部坐标转换为世界坐标。

        用于计算传感器在车身局部空间中的位置在世界坐标系中的实际位置，
        从而判断传感器是否压在黑色轨道上。

        Args:
            local_pos: 车身局部坐标系中的位置 (3,)

        Returns:
            世界坐标系中的位置 (3,)
        """
        xpos = self.data.xpos[self.car_body_id]            # 车身世界坐标位置
        xmat = self.data.xmat[self.car_body_id].reshape(3, 3)  # 车身旋转矩阵
        return xpos + xmat @ local_pos

    def _read_sensor(self, offset: np.ndarray) -> float:
        """读取单个红外传感器的值。

        将传感器局部偏移转换到世界坐标，然后计算该点到轨道中心线的
        最短距离。如果距离 ≤ 轨道半宽，说明传感器压在黑色轨道上。

        Args:
            offset: 传感器在车身局部坐标系中的偏移 (3,)

        Returns:
            0.0 = 压在黑线上 (on track)
            1.0 = 离线 (off track)
        """
        world_pos = self._local_to_world(offset)
        dist = _shortest_distance_to_track(world_pos, self.track_points)
        return 0.0 if dist <= self.track_half_width else 1.0

    def _get_obs(self) -> np.ndarray:
        """构建观测向量。

        每次 step() 后调用，返回当前时刻的完整状态信息供智能体决策。
        包含 2 个传感器读数 + 3 个速度分量 = 5 维观测。
        """
        left = self._read_sensor(self.sensor_offsets[0])
        right = self._read_sensor(self.sensor_offsets[1])
        # qvel: 广义速度 [vx, vy, vz, wx, wy, wz, ...]
        vx = float(self.data.qvel[0])   # 世界系 x 方向线速度
        vy = float(self.data.qvel[1])   # 世界系 y 方向线速度
        wz = float(self.data.qvel[5])   # 绕 z 轴角速度（偏航率，rad/s）
        return np.array([left, right, vx, vy, wz], dtype=np.float32)

    def _is_on_track(self) -> bool:
        """判断小车是否在轨道上（至少一个传感器压线即为在线）。"""
        return self._read_sensor(self.sensor_offsets[0]) == 0.0 or \
               self._read_sensor(self.sensor_offsets[1]) == 0.0

    # =====================================================================
    # Gymnasium 标准 API
    # =====================================================================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置环境到初始状态，开始新的回合 (episode)。

        RL 训练中每个回合结束时调用此方法。小车会被随机放置在轨道上的
        某个位置，面朝轨道方向，速度归零。

        Args:
            seed: 随机种子（保证结果可复现）
            options: 额外选项（Gymnasium 标准参数，此处未使用）

        Returns:
            (obs, info): 初始观测和空信息字典
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 随机选择轨道上的一个起点
        if self.np_random is not None:
            start_idx = self.np_random.integers(0, len(self.track_points))
        else:
            start_idx = 0

        # 计算起点处的轨道切线方向，让小车初始朝向与轨道一致
        p0 = self.track_points[start_idx]
        p1 = self.track_points[(start_idx + 1) % len(self.track_points)]
        direction = p1 - p0
        yaw = float(np.arctan2(direction[1], direction[0]))
        # 欧拉角 (yaw) → 四元数 [w, x, y, z]，绕 z 轴旋转
        quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])

        self.data.qpos[:3] = np.array([p0[0], p0[1], 0.03])   # 位置
        self.data.qpos[3:7] = quat                             # 朝向
        self.data.qpos[7:] = 0.0                               # 关节角度归零
        self.data.qvel[:] = 0.0                                # 速度归零

        # 执行一次正向运动学计算，更新所有派生量 (xpos, xmat 等)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.off_track_counter = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """执行一步仿真。

        这是 RL 循环的核心：
        1. 接收智能体的动作
        2. 驱动 MuJoCo 物理仿真
        3. 计算奖励和终止条件
        4. 返回 (新观测, 奖励, 是否终止, 是否截断, 信息)

        Args:
            action: 2 维动作向量 [forward, steering]，范围 [-1, 1]

        Returns:
            obs:        新的观测向量 (5,)
            reward:     这一步获得的奖励
            terminated: 是否因失败而终止（掉线）
            truncated:  是否因超时而截断（存活到最大步数）
            info:       额外信息字典
        """
        # ── 1. 将动作写入 MuJoCo 执行器 ──
        forward_cmd = float(np.clip(action[0], 0.0, 1.0))      # 前进指令（禁止倒车）
        steering_cmd = float(np.clip(action[1], -1.0, 1.0))    # 转向指令

        self.data.ctrl[self.act_forward_id] = forward_cmd
        self.data.ctrl[self.act_turn_id] = steering_cmd

        # ── 2. 推进物理仿真 ──
        # 每步动作持续 ctrl_dt 秒，物理引擎以更小的 timestep (2ms) 运行
        # n_substeps = ctrl_dt / timestep = 0.02 / 0.002 = 10 个子步
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # ── 3. 获取新观测 ──
        obs = self._get_obs()
        left_on, right_on = obs[0], obs[1]

        # ── 4. 计算奖励 ──
        # track_quality: 轨道跟随质量
        #   1.0 = 两个传感器都压在黑线上（居中行驶，最佳状态）
        #   0.5 = 只有一个传感器压线（偏离中线，需要纠偏）
        #   0.0 = 两个传感器都离线（完全脱轨）
        track_quality = (2.0 - left_on - right_on) / 2.0

        # speed: 世界坐标系中的合速度（标量）
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # speed_bonus: 速度奖励，只有压线时才能获得
        # 系数 2.0 让速度奖励在总奖励中占主导，驱动小车"快而稳"
        speed_bonus = track_quality * speed * 2.0

        # alive: 存活奖励，只要还在线上就每步给 0.05
        alive = 0.05 if track_quality > 0.0 else 0.0

        # 总奖励 = 轨道质量 + 速度奖励 + 存活奖励
        reward = track_quality * 0.5 + speed_bonus + alive

        # ── 5. 判断终止 ──
        # 连续离线计数：一旦离线就累加，回到线上则清零
        if track_quality == 0.0:
            self.off_track_counter += 1
        else:
            self.off_track_counter = 0

        # terminated: 连续离线太久 → 判定为"失败"
        terminated = self.off_track_counter >= self.max_off_track
        # truncated: 步数用完 → 判定为"成功存活"（但不是自然终止）
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 2.0  # 掉线惩罚，让智能体学会避免脱轨

        info = {
            'track_quality': track_quality,
            'speed': speed,
            'off_track_steps': self.off_track_counter,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """渲染一帧（仅在 viewer 模式下有效）。"""
        if self.render_mode == 'human' and self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        """关闭环境，释放 MuJoCo viewer 资源。"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # =====================================================================
    # MuJoCo Viewer（可视化）
    # =====================================================================

    viewer = None  # 类属性，持有 MuJoCo viewer 句柄

    def start_viewer(self) -> None:
        """启动 MuJoCo 被动渲染窗口。

        必须在 reset() 之后、step() 循环之前调用。
        设置摄像机俯视角度以便清楚看到小车在轨道上的行驶情况。
        """
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if hasattr(self.viewer, 'cam'):
            # 调整摄像机位置和角度：从上方俯视轨道
            self.viewer.cam.lookat = np.array([0.0, 0.0, 0.0])  # 看向原点
            self.viewer.cam.distance = 3.5    # 摄像机距离
            self.viewer.cam.elevation = -30.0  # 俯视角度
            self.viewer.cam.azimuth = 90.0     # 水平旋转角
        self.viewer.sync()
