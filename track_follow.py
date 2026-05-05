"""MuJoCo line-following car simulation.

The black track geometry is only used by the simulated IR sensors.  The
controller itself only receives the two sensor states, just like a physical
two-sensor line-following car.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from controller import PDController

BASE_MODEL_PATH = Path(r'D:\mujoco-3.6.0-windows-x86_64\model\car\car.xml')
TRACK_MODEL_PATH = Path(__file__).resolve().parent / 'car_track.xml'


def make_irregular_ellipse_points(num_points: int = 120) -> np.ndarray:
    """Generate sample points for an irregular elliptical black line track."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    a = 1.2
    b = 0.7
    x = (a + 0.15 * np.sin(3.0 * angles)) * np.cos(angles)
    y = (b + 0.1 * np.sin(5.0 * angles)) * np.sin(angles)
    return np.stack([x, y], axis=1)


def create_track_model(base_xml: Path, out_xml: Path) -> None:
    """Create a copy of the car model with a visible irregular ellipse track."""
    tree = ET.parse(base_xml)
    root = tree.getroot()

    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')

    if root.find("./asset/material[@name='track']") is None:
        ET.SubElement(asset, 'material', name='track', rgba='0 0 0 1')

    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('Base model is missing <worldbody>')

    for geom in worldbody.findall('geom'):
        if geom.get('name', '').startswith('track-seg'):
            worldbody.remove(geom)

    points = make_irregular_ellipse_points()
    track_half_width = 0.02
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        mid = (p0 + p1) / 2.0
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        theta = float(np.arctan2(direction[1], direction[0]))

        worldbody.append(ET.Element(
            'geom',
            name=f'track-seg-{i}',
            type='box',
            pos=f'{mid[0]:.4f} {mid[1]:.4f} 0.001',
            size=f'{length / 2.0:.4f} {track_half_width:.4f} 0.001',
            material='track',
            euler=f'0 0 {theta:.4f}',
            contype='0',
            conaffinity='0',
        ))

    tree.write(out_xml, xml_declaration=True, encoding='utf-8')


class LineFollower:
    def __init__(self, model: mujoco.MjModel, turn_sign: float = 1.0):
        self.model = model
        self.data = mujoco.MjData(model)

        self.controller = PDController(kp=0.65, kd=0.0, max_output=0.7)
        self.forward_speed = 6.68
        self.lost_forward_speed = 3.38
        self.track_half_width = 0.024
        self.last_error = 0.0
        self.turn_sign = turn_sign

        self.sensor_offsets = np.array([
            [0.09, 0.018, 0.0],   # left sensor
            [0.09, -0.018, 0.0],  # right sensor
        ])
        self.track_points = make_irregular_ellipse_points()

        self.car_body_id = model.body('car').id
        self.act_forward_id = model.actuator('forward').id
        self.act_turn_id = model.actuator('turn').id

        self._set_initial_pose()

    def _set_initial_pose(self) -> None:
        """Place the car on the track start point and orient it along the track."""
        start_point = self.track_points[0]
        next_point = self.track_points[1]
        direction = next_point - start_point
        yaw = float(np.arctan2(direction[1], direction[0]))
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

        self.data.qpos[:3] = np.array([start_point[0], start_point[1], 0.03])
        self.data.qpos[3:7] = quat
        self.data.qpos[7:] = 0.0
        self.data.qvel[:] = 0.0
        self.last_error = 0.0
        self.controller.reset()

    def _local_to_world(self, local_pos: np.ndarray) -> np.ndarray:
        body_xpos = self.data.xpos[self.car_body_id]
        body_xmat = self.data.xmat[self.car_body_id].reshape(3, 3)
        return body_xpos + body_xmat @ local_pos

    def _distance_to_track(self, world_pos: np.ndarray) -> float:
        min_dist = float('inf')
        for i in range(len(self.track_points)):
            p0 = self.track_points[i]
            p1 = self.track_points[(i + 1) % len(self.track_points)]
            segment = p1 - p0
            if np.allclose(segment, 0.0):
                dist = np.linalg.norm(world_pos[:2] - p0)
            else:
                t = np.dot(world_pos[:2] - p0, segment) / np.dot(segment, segment)
                t = np.clip(t, 0.0, 1.0)
                proj = p0 + t * segment
                dist = np.linalg.norm(world_pos[:2] - proj)
            min_dist = min(min_dist, float(dist))
        return min_dist

    def _sensor_value(self, local_offset: np.ndarray) -> float:
        world_pos = self._local_to_world(local_offset)
        return 0.0 if self._distance_to_track(world_pos) <= self.track_half_width else 1.0

    def _read_line_sensors(self) -> tuple[bool, bool]:
        left_black = self._sensor_value(self.sensor_offsets[0]) == 0.0
        right_black = self._sensor_value(self.sensor_offsets[1]) == 0.0
        return left_black, right_black

    def _compute_error(self, current_time: float) -> float:
        left_black, right_black = self._read_line_sensors()

        if left_black and not right_black:
            self.last_error = 1.0
        elif right_black and not left_black:
            self.last_error = -1.0
        elif left_black and right_black:
            self.last_error = 0.0
        elif self.last_error == 0.0:
            self.last_error = 0.35

        return self.last_error

    def step(self, current_time: float) -> None:
        error = self._compute_error(current_time)
        steering = self.controller.update(error, current_time)
        forward = self.lost_forward_speed if abs(error) > 0.9 else self.forward_speed

        self.data.ctrl[self.act_forward_id] = forward
        self.data.ctrl[self.act_turn_id] = self.turn_sign * float(np.clip(steering, -1.0, 1.0))

    def run(self, headless: bool = False) -> None:
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)

        if headless:
            self._run_headless()
        else:
            self._run_with_viewer()

    def _run_headless(self) -> None:
        print('Running headless line-following simulation...')
        timestep = float(self.model.opt.timestep)
        while True:
            current_time = time.time()
            self.step(current_time)
            mujoco.mj_step(self.model, self.data)
            time.sleep(timestep)

    def _run_with_viewer(self) -> None:
        handle = mujoco.viewer.launch_passive(self.model, self.data)
        if hasattr(handle, 'cam'):
            handle.cam.lookat = np.array([0.0, 0.0, 0.0])
            handle.cam.distance = 3.5
            handle.cam.elevation = -30.0
            handle.cam.azimuth = 90.0
        handle.sync()

        try:
            timestep = float(self.model.opt.timestep)
            while handle.is_running():
                current_time = time.time()
                self.step(current_time)
                mujoco.mj_step(self.model, self.data)
                handle.sync()
                time.sleep(timestep)
        except KeyboardInterrupt:
            print('\nSimulation interrupted by user.')
        finally:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MuJoCo car line following demo')
    parser.add_argument('--model', type=Path, default=TRACK_MODEL_PATH, help='track model XML path')
    parser.add_argument('--headless', action='store_true', help='run without rendering')
    parser.add_argument('--turn-sign', type=float, choices=(-1.0, 1.0), default=1.0, help='flip steering direction')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f'Creating track model at {args.model}')
    create_track_model(BASE_MODEL_PATH, args.model)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    follower = LineFollower(model, turn_sign=args.turn_sign)
    follower.run(headless=args.headless)


if __name__ == '__main__':
    main()
