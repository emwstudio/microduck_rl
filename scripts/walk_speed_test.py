"""walk_speed_test.py — 官方行走策略 alpha_walking.onnx 的平地速度基线（Phase 0）。

徒脚（无轮）行走，61D obs 契约与 infer_policy.py 完全一致，执行器走 BAM M6
（训练同款；--no-bam 的 XML PD 只是降级参考）。命令语义（velocity 任务）：
  - cmd[0] = lin_vel_x 目标速度 m/s（训练范围见 microduck_velocity_env_cfg.py）
  - cmd[1] = lin_vel_y，cmd[2] = ang_vel_z
  - head/body 命令槽 zero-pad（61D 契约一致）

输出 artifacts/walk_probe/：probe.csv（每控制步）+ probe.mp4（第三人称跟拍）
+ 终端结论（各档位实测速度、是否摔倒）。

用法（工作目录 = third_party/microduck_rl）：
    uv run python scripts/walk_speed_test.py
"""

from __future__ import annotations

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

SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
POLICY = DUCKEMW_ROOT / "artifacts" / "alpha_walking.onnx"
OUTDIR = DUCKEMW_ROOT / "artifacts" / "walk_probe"

SPAWN_Z = d2t.SPAWN_Z               # 0.125，与训练 spawn 一致
TILT_FALL = d2t.FALL_TILT_THRESHOLD  # 45° 倾斜判摔
VIN = 7.4                           # infer_policy.py 默认标称电压
VIN_DROP_GAIN = 0.1                 # infer_policy.py 默认负载压降


def main():
    import mujoco
    import onnxruntime as ort
    import imageio.v2 as imageio

    OUTDIR.mkdir(parents=True, exist_ok=True)

    bam_model = infer.load_bam_model(
        kp_fw=infer.BAM_KP_FW, vin=VIN, max_current=infer.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _names = infer.load_mujoco_with_bam(
        str(SCENE_XML), bam_model, d2t.SIM_TIMESTEP, VIN_DROP_GAIN, infer.BAM_VIN_MIN
    )
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480

    session = ort.InferenceSession(str(POLICY))
    assert session.get_inputs()[0].shape[-1] == 61
    h = d2t.DanceHarness(model, data, session)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])

    def spawn():
        data.qpos[qa] = 0.0
        data.qpos[qa + 1] = 0.0
        data.qpos[qa + 2] = SPAWN_Z
        data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
        data.qvel[:] = 0.0
        for i, idx in enumerate(h.joint_qpos_indices):
            data.qpos[idx] = h.default_pose[i]
        h.last_action[:] = 0.0
        bam_ctrl.reset(data.qpos)
        bam_ctrl.q_target[:] = h.default_pose
        mujoco.mj_forward(model, data)

    spawn()

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.0
    cam.azimuth = 100.0
    cam.elevation = -14.0
    fps = int(round(1.0 / d2t.CONTROL_DT))
    writer = imageio.get_writer(str(OUTDIR / "probe.mp4"), fps=fps, macro_block_size=1)

    csvf = open(OUTDIR / "probe.csv", "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["t", "phase", "cmd_x", "v_actual", "x", "y",
                   "yaw_deg", "trunk_z", "roll_deg", "pitch_deg", "fallen"])

    t = 0.0
    fallen = False
    pos_hist = []

    def step_once(cmd_x, phase):
        nonlocal t, fallen
        h.command[:] = 0.0
        h.command[0] = cmd_x
        action = h.infer()
        bam_ctrl.q_target[:] = h.default_pose + action * h.action_scale
        for _ in range(d2t.DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        t += d2t.CONTROL_DT

        x, y = float(data.qpos[qa]), float(data.qpos[qa + 1])
        pos_hist.append((t, x, y))
        while pos_hist and pos_hist[0][0] < t - 5.0:
            pos_hist.pop(0)
        v = 0.0
        if len(pos_hist) > 2:
            t0_, x0_, y0_ = pos_hist[0]
            v = math.hypot(x - x0_, y - y0_) / max(t - t0_, 1e-6)

        z, roll, pitch, _, _ = h.trunk_state()
        q = data.qpos[qa + 3 : qa + 7]
        yaw = math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))
        is_fall = (abs(roll) > TILT_FALL or abs(pitch) > TILT_FALL or z < 0.075)
        if is_fall and not fallen:
            print(f"  !! 摔倒 t={t:.1f}s phase={phase} z={z:.3f} "
                  f"roll={math.degrees(roll):.0f}° pitch={math.degrees(pitch):.0f}°")
        fallen = fallen or is_fall

        csvw.writerow([f"{t:.3f}", phase, cmd_x, f"{v:.4f}",
                       f"{x:.4f}", f"{y:.4f}", f"{yaw:.2f}", f"{z:.4f}",
                       f"{math.degrees(roll):.2f}", f"{math.degrees(pitch):.2f}",
                       int(is_fall)])

        cam.lookat[:] = data.xpos[h.trunk_base_id]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    def run_phase(phase, cmd_x, seconds):
        n = int(seconds / d2t.CONTROL_DT)
        for _ in range(n):
            step_once(cmd_x, phase)
            if fallen:
                return False
        return True

    def mean_speed_window(seconds):
        """最近 seconds 滑窗的平均速度（位移/时间）。"""
        if len(pos_hist) < 2:
            return 0.0
        t1, x1, y1 = pos_hist[-1]
        for t0, x0, y0 in pos_hist:
            if t1 - t0 >= seconds - 1e-9:
                return math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-6)
        return 0.0

    print("\n=== 原地站立 2s（零命令姿态检查）===")
    run_phase("settle", 0.0, 2.0)

    print("\n=== 速度阶梯（每档 5s，0.2 → 1.0；基线标称 0.4）===")
    ladder = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    for cmd in ladder:
        if fallen:
            print(f"  cmd={cmd}: 跳过（已摔倒）")
            results.append((cmd, None, True))
            break
        ok = run_phase(f"cmd_{cmd}", cmd, 5.0)
        # 取每档最后 3s 的位移速度（跳过加速段）
        v = mean_speed_window(3.0)
        results.append((cmd, v, fallen))
        print(f"  cmd={cmd:.1f} m/s → 实测 {v:.3f} m/s"
              + ("" if ok else "（摔倒中止）"))

    writer.close()
    csvf.close()

    print("\n=== 基线结论 ===")
    for cmd, v, fell in results:
        if v is None:
            print(f"  cmd={cmd:.1f}: 未测（摔倒）")
        else:
            print(f"  cmd={cmd:.1f}: 实测 {v:.3f} m/s" + (" [摔倒]" if fell else ""))
    v04 = next((v for cmd, v, _ in results if abs(cmd - 0.4) < 1e-9 and v is not None), None)
    if v04 is not None:
        print(f"\n基线 0.4 档位实测: {v04:.3f} m/s（目标：追上并超越 1.6 m/s）")
    print(f"CSV → {OUTDIR / 'probe.csv'}\nMP4 → {OUTDIR / 'probe.mp4'}")


if __name__ == "__main__":
    main()
