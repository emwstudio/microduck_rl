"""sprint_show.py — 徒脚极速出片：全速冲刺 → 直接撞上软垫墙（慢动作收尾）。

BAM 执行器 + 61D obs 契约（与训练/部署一致）。流程：
  1. 原地站立 1s
  2. cmd=2.0 全速冲刺（第三人称侧面跟拍，记录峰值速度）
  3. 撞上 x=8m 软垫墙 → 进入 1/4 速慢动作（撞前 0.6s 起），撞后 cmd=0 自然瘫倒
  4. 终端报告峰值速度（撞墙前 0.5s 滑窗）

用法（工作目录 = third_party/microduck_rl）：
    uv run --no-sync python scripts/sprint_show.py --policy ../../artifacts/sprint.onnx
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DUCKEMW_ROOT = REPO_ROOT.parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d2t = _load_module("d2t", REPO_ROOT / "scripts" / "dance_to_timeline.py")
infer = _load_module("infer", REPO_ROOT / "scripts" / "infer_policy.py")

SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_sprint_wall.xml"
WALL_FACE_X = 7.8          # 软垫墙前表面
IMPACT_X = 7.45            # 躯干过此线判撞击（留出躯干厚度）
SPRINT_CMD = 2.0
SLOWMO_FACTOR = 4          # 慢动作倍率（1/4 速）
SLOWMO_PRE_S = 0.6         # 慢动作起点（撞击前）
SLOWMO_POST_S = 2.5        # 慢动作持续（撞击后）
VIN = 7.4
VIN_DROP_GAIN = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=DUCKEMW_ROOT / "artifacts" / "sprint_show")
    ap.add_argument("--cmd", type=float, default=SPRINT_CMD)
    ap.add_argument("--max-sprint-s", type=float, default=20.0)
    args = ap.parse_args()

    import mujoco
    import onnxruntime as ort
    import imageio.v2 as imageio

    args.outdir.mkdir(parents=True, exist_ok=True)

    bam_model = infer.load_bam_model(
        kp_fw=infer.BAM_KP_FW, vin=VIN, max_current=infer.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _names = infer.load_mujoco_with_bam(
        str(SCENE_XML), bam_model, d2t.SIM_TIMESTEP, VIN_DROP_GAIN, infer.BAM_VIN_MIN
    )
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720

    session = ort.InferenceSession(str(args.policy))
    assert session.get_inputs()[0].shape[-1] == 61
    h = d2t.DanceHarness(model, data, session, joint_vel_delay=1)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])
    pad_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "crash_pad")

    data.qpos[qa + 2] = d2t.SPAWN_Z
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(h.joint_qpos_indices):
        data.qpos[idx] = h.default_pose[i]
    bam_ctrl.reset(data.qpos)
    bam_ctrl.q_target[:] = h.default_pose
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fps = int(round(1.0 / d2t.CONTROL_DT))
    mp4 = args.outdir / "sprint_show.mp4"
    writer = imageio.get_writer(str(mp4), fps=fps, macro_block_size=1)

    csvf = open(args.outdir / "sprint_show.csv", "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["t", "phase", "cmd_x", "x", "vx", "trunk_z", "pitch_deg"])

    t = 0.0
    impact_t = None
    pos_hist = []
    peak_speed = 0.0

    def pad_contact():
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == pad_geom or c.geom2 == pad_geom:
                return True
        return False

    def render_frame(slowmo):
        x = float(data.qpos[qa])
        if x < 6.2:
            cam.lookat[:] = [x, 0.0, 0.15]
            cam.distance = 1.6
        else:
            cam.lookat[:] = [min(x, 7.0), 0.0, 0.3]
            cam.distance = 2.2
        cam.azimuth = 90.0
        cam.elevation = -8.0
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        for _ in range(SLOWMO_FACTOR if slowmo else 1):
            writer.append_data(frame)

    def step_once(cmd_x, phase):
        nonlocal t, peak_speed, impact_t
        h.command[:] = 0.0
        h.command[0] = cmd_x
        action = h.infer()
        bam_ctrl.q_target[:] = h.default_pose + action * h.action_scale
        for _ in range(d2t.DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        t += d2t.CONTROL_DT

        x = float(data.qpos[qa])
        vx = float(data.cvel[h.trunk_base_id][3])
        pos_hist.append((t, x))
        while pos_hist and pos_hist[0][0] < t - 0.5:
            pos_hist.pop(0)
        v = 0.0
        if len(pos_hist) > 2:
            t0_, x0_ = pos_hist[0]
            v = (x - x0_) / max(t - t0_, 1e-6)

        z, roll, pitch, _, _ = h.trunk_state()
        csvw.writerow([f"{t:.3f}", phase, cmd_x, f"{x:.4f}", f"{vx:.4f}",
                       f"{z:.4f}", f"{math.degrees(pitch):.2f}"])

        if phase == "sprint":
            peak_speed = max(peak_speed, v)
            if impact_t is None and (x >= IMPACT_X or (pad_contact() and x > 6.0)):
                impact_t = t
                print(f"  ** 撞击 t={t:.2f}s x={x:.2f}m 瞬时vx={vx:.2f} "
                      f"滑窗v={v:.2f} 峰值={peak_speed:.2f} m/s")
        return v

    print("=== 站立 1s ===")
    for _ in range(int(1.0 / d2t.CONTROL_DT)):
        step_once(0.0, "settle")
        render_frame(False)

    print(f"=== 冲刺 cmd={args.cmd}（目标墙 x={WALL_FACE_X}m）===")
    n_max = int(args.max_sprint_s / d2t.CONTROL_DT)
    for _ in range(n_max):
        step_once(args.cmd, "sprint")
        in_slowmo = impact_t is not None or (
            impact_t is None and pos_hist and t > 0 and float(data.qpos[qa]) > IMPACT_X - 1.0
        )
        # 撞击前 SLOWMO_PRE_S 开始进入慢动作窗口
        if impact_t is None:
            v_now = 0.0
            if len(pos_hist) > 2:
                t0_, x0_ = pos_hist[0]
                v_now = (float(data.qpos[qa]) - x0_) / max(t - t0_, 1e-6)
            x_now = float(data.qpos[qa])
            eta = (IMPACT_X - x_now) / max(v_now, 0.3)
            render_frame(eta < SLOWMO_PRE_S + 0.4)
        else:
            render_frame(True)
        if impact_t is not None:
            break

    print("=== 撞后（cmd=0，自然瘫倒）===")
    for _ in range(int(SLOWMO_POST_S / d2t.CONTROL_DT)):
        step_once(0.0, "aftermath")
        render_frame(True)

    writer.close()
    csvf.close()

    print("\n=== 出片结论 ===")
    print(f"峰值速度（撞墙前 0.5s 滑窗）: {peak_speed:.3f} m/s")
    print(f"撞击时刻: {impact_t if impact_t is not None else '未撞墙（加速不足）'}")
    print(f"MP4 → {mp4}")
    print(f"CSV → {args.outdir / 'sprint_show.csv'}")


if __name__ == "__main__":
    main()
