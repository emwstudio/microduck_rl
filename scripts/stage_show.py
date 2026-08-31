"""stage_show.py — N 只 MicroDuck 同台齐舞（本地 CPU MuJoCo）。

同一首歌曲 timeline 同步驱动 N 个机器人（同一 ONNX 策略，逐只推理），
编队站位 + 电影机位（正面/跟踪/环绕），1080p 高清录制。

用法：
    uv run python scripts/stage_show.py \
        --policy ../../artifacts/dance_v5.onnx \
        --timeline ../../dance/songs/牛来.timeline.json \
        --ducks 6 --formation row --camera orbit --record show.mp4

物理/观测契约与 dance_to_timeline.py 完全一致（timestep 0.005 × 4 = 50Hz，
61D obs，BAM 裸 XML 舵机 kp）。第 0 只鸭子是 scene_walk.xml 自带的无前缀
机器人，其余通过 MjSpec.attach 加 d{i}_ 前缀挂载。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 复用单曲 harness 的全部公共件（timeline 加载、命令映射、常量、四元数工具）
_spec = importlib.util.spec_from_file_location(
    "d2t", REPO_ROOT / "scripts" / "dance_to_timeline.py"
)
d2t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d2t)

SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
ROBOT_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/robot_walk.xml"

SPACING = 0.35  # m — 编队间距（鸭子宽 ~20cm）


def formation_positions(n: int, kind: str) -> list[tuple[float, float]]:
    """编队 (x, y) 列表。row=横排，arc=弧形，grid=两排。"""
    if kind == "row":
        return [(0.0, (i - (n - 1) / 2) * SPACING) for i in range(n)]
    if kind == "arc":
        radius = max(0.6, SPACING * n / math.pi)
        return [
            (
                -radius * (1 - math.cos(math.pi * (i + 0.5) / n - math.pi / 2)),
                radius * math.sin(math.pi * (i + 0.5) / n - math.pi / 2),
            )
            for i in range(n)
        ]
    if kind == "grid":
        front = (n + 1) // 2
        pos = [(0.0, (i - (front - 1) / 2) * SPACING) for i in range(front)]
        back = n - front
        pos += [(-SPACING, (i - (back - 1) / 2) * SPACING) for i in range(back)]
        return pos
    raise ValueError(f"unknown formation: {kind}")


class Duck:
    """单只鸭子的索引/状态集合（与 DanceHarness 的 obs 组装同构）。"""

    def __init__(self, model, data, prefix: str, spawn_xy, all_prefixes=()):
        import mujoco

        self.model = model
        self.data = data
        self.prefix = prefix

        def nid(obj, name):
            i = mujoco.mj_name2id(model, obj, f"{prefix}{name}")
            if i < 0:
                raise ValueError(f"{prefix}{name} not found")
            return i

        self.imu_id = nid(mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_id = nid(mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        free_id = nid(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.free_qpos_adr = int(model.jnt_qposadr[free_id])

        # 该鸭子的 14 个执行器：无前缀的第 0 只要排除所有 d{k}_ 前缀的执行器
        if prefix:
            match = lambda name: name.startswith(prefix)
        else:
            match = lambda name: not any(name.startswith(p) for p in all_prefixes)
        self.ctrl_ids = [
            i
            for i in range(model.nu)
            if match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        ]
        assert len(self.ctrl_ids) == 14, f"{prefix}: expected 14 actuators, got {len(self.ctrl_ids)}"
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in self.ctrl_ids
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in self.ctrl_ids
        ]
        self.ctrl_adr = list(self.ctrl_ids)  # ctrl 按执行器序号索引
        self.default_pose = d2t.DEFAULT_POSE[:14]
        self.last_action = np.zeros(14, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)

        # 初始状态：HOME 姿态 + 编队位置
        data.qpos[self.free_qpos_adr + 0] = spawn_xy[0]
        data.qpos[self.free_qpos_adr + 1] = spawn_xy[1]
        data.qpos[self.free_qpos_adr + 2] = d2t.SPAWN_Z
        data.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7] = [1, 0, 0, 0]
        for i, idx in enumerate(self.joint_qpos_indices):
            data.qpos[idx] = self.default_pose[i]

    def get_obs(self):
        sensor_adr = self.model.sensor_adr[self.imu_id]
        ang_vel = self.data.sensordata[sensor_adr : sensor_adr + 3].astype(np.float32)
        quat = self.data.xquat[self.trunk_id].astype(np.float32)
        grav = d2t._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        return np.concatenate(
            [
                ang_vel,
                grav,
                self.data.qpos[self.joint_qpos_indices].astype(np.float32) - self.default_pose,
                self.data.qvel[self.joint_qvel_indices].astype(np.float32),
                self.last_action,
                self.command,
            ]
        ).astype(np.float32)

    def apply(self, action):
        self.last_action = action.astype(np.float32).copy()
        self.data.ctrl[self.ctrl_adr] = self.default_pose + action * ACTION_SCALE

    def trunk_z(self):
        return float(self.data.xpos[self.trunk_id][2])


ACTION_SCALE = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--ducks", type=int, default=6)
    ap.add_argument("--formation", choices=["row", "arc", "grid"], default="row")
    ap.add_argument("--camera", choices=["front", "tracking", "orbit"], default="front")
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--orbit-deg-per-s", type=float, default=6.0)
    args = ap.parse_args()

    import mujoco

    timeline = d2t.load_timeline(args.timeline)
    positions = formation_positions(args.ducks, args.formation)
    print(f"Stage: {args.ducks} ducks, formation={args.formation}, camera={args.camera}")

    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for key in list(spec.keys):  # keyframes 按单机器人尺寸定义，attach 后失效
        spec.delete(key)
    for i in range(1, args.ducks):
        frame = spec.worldbody.add_frame(pos=[positions[i][0], positions[i][1], 0])
        robot = mujoco.MjSpec.from_file(str(ROBOT_XML))  # attach 会改写 child，每次重载
        spec.attach(robot, prefix=f"d{i}_", frame=frame)
    # 高分辨率离屏渲染需要加大 framebuffer（默认对 1080p 不够）
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    model = spec.compile()
    model.opt.timestep = d2t.SIM_TIMESTEP
    data = mujoco.MjData(model)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.policy))
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    all_prefixes = [f"d{i}_" for i in range(1, args.ducks)]
    ducks = [
        Duck(model, data, "" if i == 0 else f"d{i}_", positions[i], all_prefixes)
        for i in range(args.ducks)
    ]
    mujoco.mj_forward(model, data)

    import imageio.v2 as imageio

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    center_y = float(np.mean([p[1] for p in positions]))
    span = max(abs(p[1] - center_y) for p in positions) + SPACING
    camera.distance = max(0.9, span * 1.4)
    fps = int(round(1.0 / d2t.CONTROL_DT))
    writer = imageio.get_writer(str(args.record), fps=fps)
    print(f"Recording {args.width}x{args.height} → {args.record}")

    duration = float(timeline["duration"])
    n_steps = int(round(duration / d2t.CONTROL_DT))
    print(f"Running {duration:.1f}s @ {fps} Hz × {args.ducks} ducks ({n_steps} steps)")

    for step in range(n_steps):
        t = step * d2t.CONTROL_DT
        cmd6 = d2t.dance_command(t, timeline)
        obs_batch = []
        for d in ducks:
            d.command[7:13] = cmd6
            obs_batch.append(d.get_obs())
        obs = np.stack(obs_batch)
        try:
            actions = session.run([out_name], {in_name: obs})[0]
        except Exception:
            actions = np.concatenate(
                [session.run([out_name], {in_name: o.reshape(1, -1)})[0] for o in obs]
            )
        for d, a in zip(ducks, actions):
            d.apply(a)
        for _ in range(d2t.DECIMATION):
            mujoco.mj_step(model, data)

        # 机位
        if args.camera == "front":
            camera.lookat[:] = [0.0, center_y, 0.12]
            camera.azimuth = 90.0
            camera.elevation = -12.0
        elif args.camera == "tracking":
            cx = float(np.mean([d.data.xpos[d.trunk_id][0] for d in ducks]))
            cy = float(np.mean([d.data.xpos[d.trunk_id][1] for d in ducks]))
            cz = float(np.mean([d.trunk_z() for d in ducks]))
            camera.lookat[:] = [cx, cy, cz]
            camera.azimuth = 90.0
            camera.elevation = -12.0
        else:  # orbit
            camera.lookat[:] = [0.0, center_y, 0.12]
            camera.azimuth = 90.0 + args.orbit_deg_per_s * t
            camera.elevation = -15.0
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())

        if step % (fps * 4) == 0:
            zs = ", ".join(f"{d.trunk_z():.3f}" for d in ducks)
            print(f"  t={t:5.1f}s  trunk z: [{zs}]")

    writer.close()
    renderer.close()
    fallen = sum(1 for d in ducks if d.trunk_z() < 0.07)
    print(f"Done. ducks fallen at end: {fallen}/{args.ducks}")


if __name__ == "__main__":
    main()
