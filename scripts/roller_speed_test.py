"""roller_speed_test.py — 官方轮滑策略 roller.onnx 的平地极速摸底（Phase 0）。

rollers 任务的命令语义与走路不同（microduck_velocity_rollers_env_cfg.py:540-553
实测核实）：
  - cmd[0] = lin_vel_x：不是目标速度，是「推/滑/刹」：>0 加速推，0=滑行，
    <0=刹车，训练范围 (-0.5, 0.6)
  - cmd[2] = 朝向误差（弧度）：策略消差转向，训练时锁 0（直线专精），
    infer_policy.py 的 roller 模式用 ±1.0 rad
  - head/body 命令槽 zero-pad（61D 契约一致）

场景 scene_rollers.xml → robot_groundcontact_rollers.xml（合并后新名，4 被动轮）。
轮子轴承摩擦必须程序化设置（XML 里写非零会破坏训练 —— infer_policy.py 注释）：
passive_* 关节 dof_frictionloss = 0.003。

输出 artifacts/roller_probe/：probe.csv（每控制步）+ probe.mp4（第三人称跟拍）
+ 终端结论。

用法（工作目录 = third_party/microduck_rl）：
    uv run python scripts/roller_speed_test.py
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

SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
POLICY = DUCKEMW_ROOT / "artifacts" / "roller.onnx"
OUTDIR = DUCKEMW_ROOT / "artifacts" / "roller_probe"

WHEEL_FRICTIONLOSS = 0.003  # infer_policy.py 的 roller 模式实测值
SPAWN_Z = 0.1385            # 轮滑比站立高 13.5mm（infer_policy.py:976）
TILT_FALL = d2t.FALL_TILT_THRESHOLD  # 45° 倾斜判摔
Z_FALL_MARGIN = 0.75        # z < 滚动期最低 z × 此系数 → 判摔（脚本会打印实测范围）


def main():
    import mujoco
    import onnxruntime as ort
    import imageio.v2 as imageio

    OUTDIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    model.opt.timestep = d2t.SIM_TIMESTEP
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    data = mujoco.MjData(model)

    # 轮子轴承摩擦（必须程序化设置）
    import re
    n_wheel = 0
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and re.match(r"^passive_.*", name):
            model.dof_frictionloss[model.jnt_dofadr[j]] = WHEEL_FRICTIONLOSS
            n_wheel += 1
    print(f"轮子摩擦设置: {n_wheel} 个 passive 关节 frictionloss={WHEEL_FRICTIONLOSS}")

    session = ort.InferenceSession(str(POLICY))
    assert session.get_inputs()[0].shape[-1] == 61
    h = d2t.DanceHarness(model, data, session)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])
    data.qpos[qa + 2] = SPAWN_Z
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(h.joint_qpos_indices):
        data.qpos[idx] = h.default_pose[i]
    data.ctrl[:] = h.default_pose
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.2
    cam.azimuth = 100.0
    cam.elevation = -14.0
    fps = int(round(1.0 / d2t.CONTROL_DT))
    writer = imageio.get_writer(str(OUTDIR / "probe.mp4"), fps=fps, macro_block_size=1)

    csvf = open(OUTDIR / "probe.csv", "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["t", "phase", "cmd_x", "cmd_h", "v_actual", "x", "y",
                   "yaw_deg", "trunk_z", "roll_deg", "pitch_deg", "fallen"])

    t = 0.0
    fallen = False
    z_healthy_min = 1.0
    fallen_logged = False
    falls = []  # [(phase, t)] 全程摔倒记录

    # 里程计：每步用位移算实际速度（0.25s 滑窗平滑）
    pos_hist = []

    def reset():
        """摔倒后重新出生（继续后续测量）。"""
        nonlocal fallen, fallen_logged
        data.qpos[qa] = 0.0
        data.qpos[qa + 1] = 0.0
        data.qpos[qa + 2] = SPAWN_Z
        data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
        data.qvel[:] = 0.0
        for i, idx in enumerate(h.joint_qpos_indices):
            data.qpos[idx] = h.default_pose[i]
        data.ctrl[:] = h.default_pose
        h.last_action = np.zeros(14, dtype=np.float32)
        mujoco.mj_forward(model, data)
        fallen = False
        fallen_logged = False
        pos_hist.clear()

    def step_once(cmd_x, cmd_h, phase):
        nonlocal t, fallen, z_healthy_min, fallen_logged
        h.command[:] = 0.0
        h.command[0] = cmd_x
        h.command[2] = cmd_h
        h.infer()
        h.apply_action(h.last_action)
        for _ in range(d2t.DECIMATION):
            mujoco.mj_step(model, data)
        t += d2t.CONTROL_DT

        x, y = float(data.qpos[qa]), float(data.qpos[qa + 1])
        pos_hist.append((t, x, y))
        while pos_hist and pos_hist[0][0] < t - 0.25:
            pos_hist.pop(0)
        v = 0.0
        if len(pos_hist) > 2:
            t0_, x0_, y0_ = pos_hist[0]
            v = math.hypot(x - x0_, y - y0_) / max(t - t0_, 1e-6)

        z, roll, pitch, _, _ = h.trunk_state()
        q = data.qpos[qa + 3 : qa + 7]
        yaw = math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))
        if not fallen and abs(roll) < math.radians(20):
            z_healthy_min = min(z_healthy_min, z)
        is_fall = (abs(roll) > TILT_FALL or abs(pitch) > TILT_FALL
                   or z < max(0.075, z_healthy_min * Z_FALL_MARGIN))
        if is_fall and not fallen_logged:
            print(f"  !! 摔倒 t={t:.1f}s phase={phase} z={z:.3f} "
                  f"roll={math.degrees(roll):.0f}° pitch={math.degrees(pitch):.0f}°")
            falls.append((phase, round(t, 1)))
            fallen_logged = True
        fallen = fallen or is_fall

        csvw.writerow([f"{t:.3f}", phase, cmd_x, cmd_h, f"{v:.4f}",
                       f"{x:.4f}", f"{y:.4f}", f"{yaw:.2f}", f"{z:.4f}",
                       f"{math.degrees(roll):.2f}", f"{math.degrees(pitch):.2f}",
                       int(is_fall)])

        cam.lookat[:] = data.xpos[h.trunk_base_id]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    def run_phase(phase, cmd_x, cmd_h, seconds):
        n = int(seconds / d2t.CONTROL_DT)
        for _ in range(n):
            step_once(cmd_x, cmd_h, phase)
            if fallen:
                return False
        return True

    def speed_now():
        """最近 0.25s 滑窗的速度。"""
        if len(pos_hist) < 2:
            return 0.0
        t0_, x0_, y0_ = pos_hist[0]
        _, x1_, y1_ = pos_hist[-1]
        return math.hypot(x1_ - x0_, y1_ - y0_) / max(t - t0_, 1e-6)

    print("\n=== Phase 1: 极速阶梯（每档 4s，0.0 → 0.8 超过训练上限 0.6）===")
    ladder = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ladder_results = []
    for cmd in ladder:
        if fallen:
            print(f"  cmd={cmd}: 跳过（已摔倒）")
            ladder_results.append((cmd, None, True))
            break
        ok = run_phase(f"ladder_{cmd}", cmd, 0.0, 4.0)
        v = speed_now()
        ladder_results.append((cmd, v, fallen))
        print(f"  cmd={cmd:+.1f}: 末速 {v:.3f} m/s{'  ← 摔倒!' if fallen else ''}")
        if fallen:
            break

    stable = [(c, v) for c, v, f in ladder_results if not f and v is not None]
    if stable:
        vmax_cmd, vmax = max(stable, key=lambda p: p[1])
        print(f"稳定极速: {vmax:.3f} m/s（cmd={vmax_cmd} 档）")

    print("\n=== Phase 2: 转向（巡航 cmd=0.3，朝向误差 0.3/0.6/1.0 rad 各 3s）===")
    turn_results = []
    reset()
    if run_phase("reaccel_turn", 0.4, 0.0, 4.0):
        for h_err in (0.3, 0.6, 1.0):
            if fallen:
                reset()
                run_phase("reaccel_turn", 0.4, 0.0, 4.0)
            q = data.qpos[qa + 3 : qa + 7]
            yaw0 = math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                           1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
            x0, y0 = float(data.qpos[qa]), float(data.qpos[qa + 1])
            ok = run_phase(f"turn_{h_err}", 0.3, h_err, 3.0)
            q = data.qpos[qa + 3 : qa + 7]
            yaw1 = math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                           1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
            dyaw = (yaw1 - yaw0 + 180) % 360 - 180
            chord = math.hypot(float(data.qpos[qa]) - x0,
                               float(data.qpos[qa + 1]) - y0)
            radius = (chord / (2 * math.sin(math.radians(abs(dyaw)) / 2))
                      if abs(dyaw) > 5 else float("inf"))
            turn_results.append((h_err, dyaw / 3.0, radius, not ok))
            print(f"  h_err={h_err}: 实测角速度 {dyaw / 3.0:+.1f}°/s，"
                  f"转弯半径 ≈{radius:.2f}m{'  ← 摔倒!' if not ok else ''}")

    print("\n=== Phase 3: 制动（coast 0.0 / 缓刹 -0.2 / 急刹 -0.5）===")
    brake_results = {}
    for label, brake_cmd in (("coast", 0.0), ("gentle", -0.2), ("brake", -0.5)):
        reset()
        # 先重新加速到巡航
        if not run_phase(f"reaccel_{label}", 0.4, 0.0, 4.0):
            print(f"  {label}: 加速阶段就摔了，跳过")
            continue
        v0 = speed_now()
        x0, y0 = float(data.qpos[qa]), float(data.qpos[qa + 1])
        t0_ = t
        # 滑/刹到停
        while True:
            step_once(brake_cmd, 0.0, f"{label}_stop")
            if fallen or speed_now() < 0.02 or t - t0_ > 12.0:
                break
        dist = math.hypot(float(data.qpos[qa]) - x0, float(data.qpos[qa + 1]) - y0)
        brake_results[label] = (v0, t - t0_, dist, fallen)
        print(f"  {label}（cmd={brake_cmd}）: 初速 {v0:.3f} m/s → "
              f"停，用时 {t - t0_:.1f}s，距离 {dist:.2f}m{'  （停下时倾倒！）' if fallen else ''}")

    writer.close()
    renderer.close()
    csvf.close()

    print("\n=== 结论 ===")
    print(f"摔倒记录: {falls if falls else '无'}，健康滚动最低 trunk z: {z_healthy_min:.3f}")
    print(f"数据: {OUTDIR / 'probe.csv'}\n视频: {OUTDIR / 'probe.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
