"""roller_gp.py — 鸭子轮滑大奖赛：计时赛道 + 纯追踪控制 + 分段计时 + DNF 判定。

赛道 scene_roller_gp.xml（直道 → S 绕桩 → 180° 发卡 → 冲坡 → 独木桥 → 迷宫
直角弯×2 → 停车入位）。控制为 pure pursuit：
  - cmd[0] 推/滑/刹语义（Phase 0 实测：>0 推、0 滑、<0 刹；死区 ≤0.2 不动，
    甜区 0.3-0.5，缓刹 -0.2 预留 0.25m/1s）
  - cmd[2] 朝向误差（rad，clip ±0.6；Phase 0：0.3 → 30°/s 平滑档）
  - 每个 waypoint 带速度档（fast/cruise/slow/crawl）和 sector 标签；
    接近低速段提前缓刹；停车入位：进框且速度 <0.03 m/s 才算完赛，
    停框外罚时 +3s，冲出/摔倒/掉桥/超时 = DNF

产物 artifacts/roller_gp/：run.mp4（一镜到底跟拍）+ sectors.json + probe.csv

用法（工作目录 = third_party/microduck_rl）：
    uv run python scripts/roller_gp.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
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

SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_roller_gp.xml"
POLICY = DUCKEMW_ROOT / "artifacts" / "roller.onnx"
OUTDIR = DUCKEMW_ROOT / "artifacts" / "roller_gp"

# 速度档 → cmd_x（推档位；Phase 0 实测映射 0.3→0.42m/s 0.4→0.49 0.5→0.59）
PUSH = {"fast": 0.5, "cruise": 0.4, "slow": 0.3}
BRAKE = -0.2          # 缓刹（Phase 0：0.19m/0.9s 自 0.5m/s）
BRAKE_DIST = 0.45     # 距低速点这么近且快就缓刹
LOOKAHEAD = 0.4       # 纯追踪前视距离（m）
HEADING_CLIP = 0.6    # 朝向误差指令上限（rad）
PARK_CENTER = np.array([0.5, 0.15])
PARK_HALF = 0.25      # 0.5×0.5m 框
PARK_V_MAX = 0.03     # 停稳判定速度
TIMEOUT_S = 150.0

# 路线：(x, y, 速度档, sector)。sector 切换瞬间记一次分段时间。
ROUTE = [
    # S1 发车直道（全速加速）
    (0.8, 0.0, "fast", "S1"), (1.6, 0.0, "fast", "S1"), (2.4, 0.0, "fast", "S1"),
    (3.1, 0.0, "fast", "S1"),
    # S2 S 弯绕桩（桩 y=±0.3 交替，走对侧 ±0.12；慢速）
    (3.5, -0.12, "slow", "S2"), (4.0, 0.12, "slow", "S2"), (4.5, -0.12, "slow", "S2"),
    (5.0, 0.12, "slow", "S2"), (5.5, -0.12, "slow", "S2"), (6.0, 0.0, "slow", "S2"),
    # S3 发卡弯绕桶掉头（绕到桶右侧 → 回程车道 y=0.9）
    # 桶中心 (6.8,0.3) r=0.15；waypoint 全部保持 ≥0.45m 离桶心
    #（首跑把 (7.05,0.3) 放在离桶心 0.25m 处 = 桶面+鸭身半径，鸭子顶着桶卡死
    #  两分多钟 —— 放宽）
    (6.35, -0.30, "slow", "S3"), (7.05, -0.15, "crawl", "S3"),
    (7.25, 0.35, "crawl", "S3"), (6.9, 0.75, "crawl", "S3"), (6.3, 0.9, "slow", "S3"),
    # S4 冲坡（返程车道 -x 上行；进坡保持 fast 全速冲 —— 实测 0.4 推力在坡底失速）
    (5.6, 0.9, "fast", "S4"), (5.0, 0.9, "fast", "S4"), (4.4, 0.9, "fast", "S4"),
    (3.8, 0.9, "slow", "S4"),
    # S5 独木桥（全程最慢最直）
    (3.0, 0.9, "crawl", "S5"), (2.6, 0.9, "crawl", "S5"), (2.2, 0.9, "crawl", "S5"),
    (1.8, 0.9, "crawl", "S5"), (1.4, 0.9, "crawl", "S5"),
    # 下坡
    (1.0, 0.9, "slow", "S5"), (0.6, 0.9, "slow", "S5"),
    # S6 迷宫直角弯（W1 顶部左 90°，绕 W1 底部右 90°）
    (0.0, 0.9, "slow", "S6"), (-0.55, 0.9, "slow", "S6"), (-0.85, 0.45, "crawl", "S6"),
    (-0.85, -0.05, "crawl", "S6"), (-0.45, -0.15, "slow", "S6"), (-0.05, 0.0, "slow", "S6"),
    # S7 停车入位
    (0.3, 0.1, "crawl", "S7"), (0.5, 0.15, "crawl", "S7"),
]
PUSH["crawl"] = 0.3  # crawl 与 slow 同推力，但转弯 clip 更紧由 heading clip 控制

BRIDGE_X = (1.1, 3.3)   # 桥段 x 范围
DECK_Z_MIN = 0.15       # 桥上 z 低于此 = 掉桥 DNF（台面 0.08 + 滚动躯干 ~0.10）


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

    import re
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and re.match(r"^passive_.*", name):
            model.dof_frictionloss[model.jnt_dofadr[j]] = 0.003

    session = ort.InferenceSession(str(POLICY))
    assert session.get_inputs()[0].shape[-1] == 61
    h = d2t.DanceHarness(model, data, session)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])
    data.qpos[qa + 2] = 0.1385
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(h.joint_qpos_indices):
        data.qpos[idx] = h.default_pose[i]
    data.ctrl[:] = h.default_pose
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.7
    cam.azimuth = 100.0
    cam.elevation = -16.0
    fps = int(round(1.0 / d2t.CONTROL_DT))
    writer = imageio.get_writer(str(OUTDIR / "run.mp4"), fps=fps, macro_block_size=1)

    csvf = open(OUTDIR / "probe.csv", "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["t", "sector", "cmd_x", "cmd_h", "v", "x", "y", "yaw_deg",
                   "trunk_z", "wp_idx"])

    path = [(x, y) for x, y, _, _ in ROUTE]
    tags = [s for _, _, s, _ in ROUTE]
    sectors = [sec for _, _, _, sec in ROUTE]
    sector_start_idx = {}
    for i, sec in enumerate(sectors):
        sector_start_idx.setdefault(sec, i)

    def xy():
        return np.array([float(data.qpos[qa]), float(data.qpos[qa + 1])])

    def yaw():
        q = data.qpos[qa + 3 : qa + 7]
        return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))

    def nearest_idx(p, from_idx):
        """前向窗口内找最近路径点（允许回退 1 个点防卡死）。
        赛道会折返经过起点附近，全程最近点会把索引进度直接跳到末段
        （首跑 3.5s「完赛」的 bug）—— 必须只看前向窗口。"""
        best, bestd = from_idx, float("inf")
        for i in range(max(0, from_idx - 1), min(len(path), from_idx + 5)):
            d = math.hypot(path[i][0] - p[0], path[i][1] - p[1])
            if d < bestd:
                best, bestd = i, d
        return best, bestd

    def target_idx(p, ni):
        """从最近点沿路径前视 LOOKAHEAD 米（按折线弧长）。"""
        acc = 0.0
        prev = np.array(path[ni])
        for i in range(ni + 1, len(path)):
            seg = math.hypot(path[i][0] - prev[0], path[i][1] - prev[1])
            if acc + seg >= LOOKAHEAD:
                return i
            acc += seg
            prev = np.array(path[i])
        return len(path) - 1

    def speed_now(p_hist):
        if len(p_hist) < 2:
            return 0.0
        (t0_, x0_, y0_), (t1_, x1_, y1_) = p_hist[0], p_hist[-1]
        return math.hypot(x1_ - x0_, y1_ - y0_) / max(t1_ - t0_, 1e-6)

    # 找下一个比当前更慢的 waypoint 的距离（提前缓刹用）
    def dist_to_slower(ni):
        cur_rank = PUSH[tags[ni]]
        acc = 0.0
        prev = np.array(path[ni])
        for i in range(ni + 1, len(path)):
            acc += math.hypot(path[i][0] - prev[0], path[i][1] - prev[1])
            if PUSH[tags[i]] < cur_rank:
                return acc
            prev = np.array(path[i])
        return float("inf")

    sector_times = {}
    sector_order = []
    t = 0.0
    ni = 0
    pos_hist = []
    result = "TIMEOUT"
    v_max = 0.0
    parked = False
    penalty = 0.0

    print(f"Roller GP 发车：{len(path)} 个 waypoint，{len(sector_start_idx)} 个赛段")

    n_steps = int(TIMEOUT_S / d2t.CONTROL_DT)
    for step in range(n_steps):
        p = xy()
        pos_hist.append((t, p[0], p[1]))
        while pos_hist and pos_hist[0][0] < t - 0.25:
            pos_hist.pop(0)
        v = speed_now(pos_hist)
        v_max = max(v_max, v)

        # 路径跟踪
        ni, nd = nearest_idx(p, ni)
        ti = target_idx(p, ni)
        tgt = np.array(path[ti])
        d_err = math.atan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw()
        d_err = math.atan2(math.sin(d_err), math.cos(d_err))
        cmd_h = float(np.clip(d_err, -HEADING_CLIP, HEADING_CLIP))

        cmd_x = PUSH[tags[ni]]
        # 大转角减速（Phase 0：大角度边转边滑才稳）
        if abs(d_err) > 0.35:
            cmd_x = min(cmd_x, 0.3)
        # 接近更慢的路段提前缓刹
        if v > 0.35 and dist_to_slower(ni) < BRAKE_DIST:
            cmd_x = BRAKE

        # 停车入位接管最后阶段
        d_park = math.hypot(p[0] - PARK_CENTER[0], p[1] - PARK_CENTER[1])
        if sectors[ni] == "S7":
            if d_park < PARK_HALF and v < PARK_V_MAX:
                cmd_x = BRAKE
                parked = True
                result = "FINISH"
            elif d_park < 0.35 and v > 0.15:
                cmd_x = BRAKE
            else:
                cmd_x = min(cmd_x, 0.3)

        # 终点半径内持续刹停
        if parked:
            cmd_x = BRAKE if v > PARK_V_MAX else 0.0
            cmd_h = 0.0

        h.command[:] = 0.0
        h.command[0] = cmd_x
        h.command[2] = cmd_h
        h.infer()
        h.apply_action(h.last_action)
        for _ in range(d2t.DECIMATION):
            mujoco.mj_step(model, data)
        t += d2t.CONTROL_DT

        # 赛段计时（进入新 sector 的瞬间）
        sec = sectors[ni]
        if not sector_order or sec != sector_order[-1]:
            sector_order.append(sec)
            sector_times[sec] = t
            if len(sector_order) > 1:
                print(f"  [{sec}] 进入 t={t:.1f}s  v={v:.2f}m/s")

        # 摔倒 DNF
        z, roll, pitch, _, _ = h.trunk_state()
        if abs(roll) > d2t.FALL_TILT_THRESHOLD or abs(pitch) > d2t.FALL_TILT_THRESHOLD \
                or z < 0.075:
            result = "DNF_FALL"
            print(f"  !! 摔倒 DNF t={t:.1f}s sector={sec} z={z:.3f}")
            break
        # 掉桥 DNF
        if sec == "S5" and BRIDGE_X[0] < p[0] < BRIDGE_X[1] and z < DECK_Z_MIN:
            result = "DNF_BRIDGE"
            print(f"  !! 掉桥 DNF t={t:.1f}s x={p[0]:.2f} z={z:.3f}")
            break

        csvw.writerow([f"{t:.3f}", sec, f"{cmd_x:+.2f}", f"{cmd_h:+.2f}",
                       f"{v:.3f}", f"{p[0]:.3f}", f"{p[1]:.3f}",
                       f"{math.degrees(yaw()):.1f}", f"{z:.4f}", ni])

        cam.lookat[:] = data.xpos[h.trunk_base_id]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

        if result == "FINISH":
            # 多录 1s 停稳画面
            for _ in range(fps):
                cam.lookat[:] = data.xpos[h.trunk_base_id]
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
            break

    writer.close()
    renderer.close()
    csvf.close()

    # 停车判定：FINISH 时必然在框内（进入条件）；停歪了给罚时
    p = xy()
    in_box = (abs(p[0] - PARK_CENTER[0]) < PARK_HALF
              and abs(p[1] - PARK_CENTER[1]) < PARK_HALF)
    if result == "FINISH" and not in_box:
        penalty = 3.0

    # 分段成绩
    sector_report = {}
    prev_t = 0.0
    for sec in sector_order:
        sector_report[sec] = round(sector_times[sec] - prev_t, 2)
        prev_t = sector_times[sec]

    summary = {
        "result": result, "total_s": round(t, 2), "penalty_s": penalty,
        "total_with_penalty_s": round(t + penalty, 2),
        "v_max_ms": round(v_max, 3),
        "sector_split_s": sector_report,
        "sector_entry_t_s": {k: round(v, 2) for k, v in sector_times.items()},
        "final_xy": [round(float(v_), 3) for v_ in p],
        "parked_in_box": bool(in_box) if result == "FINISH" else False,
    }
    (OUTDIR / "sectors.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n=== 成绩: {result}  总时间 {t:.1f}s（含罚时 {t + penalty:.1f}s）"
          f"  极速 {v_max:.2f}m/s ===")
    for sec, dt_ in sector_report.items():
        print(f"  {sec}: 进入 t={sector_times[sec]:.1f}s")
    print(f"数据: {OUTDIR / 'probe.csv'}\n视频: {OUTDIR / 'run.mp4'}")
    return 0 if result == "FINISH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
