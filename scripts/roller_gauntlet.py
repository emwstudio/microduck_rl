"""roller_gauntlet.py — TRON 霓虹夜赛：轮滑闯关跑道一镜到底出片（纯仿真，不训练）。

以 roller_gp.py 为骨架（pure pursuit / DanceHarness / frictionloss / DNF 判定），
赛道几何与 waypoint 由 roller_gauntlet_assets.py 的 build_route() 同源生成。

赛道（A1-A7）：螺旋塔(cyan) → 10° 俯冲滑槽(冰蓝) → kicker + 熔岩天堑波浪谷
(白，贴地飞坡——0.5m 真跳在 ~1m/s 起飞速度下飞行时间 0.2s+、落地落差 0.2m+，
超平地策略落地冲击极限，按预案降级） → 8字左环(magenta) → 右环螺旋爬升
立交桥(orange) → 天空平桥 → 桥顶金色冲线台（下桥俯冲段五轮实证必摔，取消）。

新增（相对 roller_gp）：
  - 3D waypoint（x,y,z,gear,sector）：纯追踪仍用 xy，z 用于赛段判定/掉台 DNF
  - 倒计时起步：0.9s 站定（HUD 3-2-1-GO）后满推
  - 赛段驱动运镜：开场高空环绕→塔顶海报机位 / 螺旋塔环绕 / 俯冲贴地追尾 /
    飞跃侧前定机位 / 8字环内低位追拍 / 终点 punch-in→拉回高空海报收尾，
    切换 0.5s smoothstep 混合（插值思路抄 stage_show.py）
  - 4 倍慢动作：飞跃区（kicker 前 0.3m → 落地后 0.5m）逐物理子步渲染
    （0.005s/帧 → 统一 50fps 写出 = 天然 4x 慢镜）；冲线急停瞬间 0.5s 慢镜
  - FPV 画中画：480x270 贴右上角（2px 发光边框），仅俯冲+飞跃段显示
  - HUD（PIL 烧录，--hud/--no-hud）：速度表 / 赛段色块 / 倒计时 / NEW TOP SPEED
  - --poster：高空正俯 1920x1080 海报静帧（duck 在塔顶起跑位）

产物 artifacts/roller_gauntlet/：run.mp4 + sectors.json + probe.csv + poster.png

用法（工作目录 = third_party/microduck_rl）：
    uv run python scripts/roller_gauntlet.py                 # 640x480 快调
    uv run python scripts/roller_gauntlet.py --width 1920 --height 1080
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DUCKEMW_ROOT = REPO_ROOT.parents[1]


def _load_module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 装饰器要按 __module__ 反查 sys.modules
    spec.loader.exec_module(mod)
    return mod


d2t = _load_module("d2t", REPO_ROOT / "scripts" / "dance_to_timeline.py")
rga = _load_module("rga", REPO_ROOT / "scripts" / "roller_gauntlet_assets.py")

POLICY = DUCKEMW_ROOT / "artifacts" / "roller.onnx"
OUTDIR = DUCKEMW_ROOT / "artifacts" / "roller_gauntlet"

# 赛车线外偏：实测 duck 在 r≈1.5 环上稳态内切 0.13-0.18m（爬坡弯更深，
# R1g 在右环 φ≈100° 处内切 0.34m 掉台）——控制 waypoint 外偏 0.1m 补偿
RACING_BIAS = {"A1": 0.10, "A4": 0.10, "A5": 0.10}

# 速度档 → cmd_x（同 roller_gp Phase 0 实测映射）
PUSH = {"fast": 0.5, "cruise": 0.4, "slow": 0.3, "crawl": 0.3}
BRAKE = -0.2          # 缓刹（Phase 0：0.19m/0.9s 自 0.5m/s）
BRAKE_DIST = 0.45     # 距低速点这么近且快就缓刹
LOOKAHEAD = 0.4       # 纯追踪前视距离（m）
HEADING_CLIP = 0.6    # 朝向误差指令上限（rad）
COUNTDOWN_S = 0.9     # 倒计时站定
TIMEOUT_S = 120.0
INTRO_S = 2.0         # 开场运镜时长
CAM_BLEND_S = 0.5     # 运镜切换混合
SLOWMO_PAD_IN = 0.3   # 慢镜进入余量（kicker 前）
SLOWMO_PAD_OUT = 0.5  # 慢镜退出余量（落地后）
STAND_Z_REL = 0.115   # 站姿 trunk z 相对 deck 面高度
FALL_Z_REL = 0.075    # 相对 deck 面低于此 = 摔倒（roller_gp 绝对值 0.075 的相对化）
OFFDECK_DROP = 0.12   # 高空赛段 trunk z 低于 route z+STAND_Z_REL-OFFDECK_DROP = 掉台


def _smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def _ang_lerp(a0, a1, u):
    """角度插值（unwrap 到最近圈，防 azimuth 跳变）。"""
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    return a0 + d * u


class CamState:
    """相机当前状态 + 目标模式平滑混合。"""

    def __init__(self):
        self.lookat = np.array([0.4, 0.5, 0.0])
        self.az, self.el, self.dist = 120.0, -88.0, 12.5
        self.mode = "intro"
        self.switch_t = 0.0
        self.start = (self.lookat.copy(), self.az, self.el, self.dist)

    def set_mode(self, mode, t):
        if mode != self.mode:
            self.mode = mode
            self.switch_t = t
            self.start = (self.lookat.copy(), self.az, self.el, self.dist)

    def update(self, target, t):
        tl, taz, tel, tdist = target
        u = _smoothstep(min((t - self.switch_t) / CAM_BLEND_S, 1.0))
        sl, saz, sel, sdist = self.start
        self.lookat = sl + (np.array(tl, dtype=float) - sl) * u
        self.az = _ang_lerp(saz, taz, u)
        self.el = sel + (tel - sel) * u
        self.dist = sdist + (tdist - sdist) * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hud", dest="hud", action="store_true", default=True)
    ap.add_argument("--no-hud", dest="hud", action="store_false")
    ap.add_argument("--poster", type=Path, default=OUTDIR / "poster.png")
    ap.add_argument("--no-poster", dest="poster", action="store_const", const=None)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args()

    import mujoco
    import onnxruntime as ort
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    P = rga.P
    ROUTE = rga.build_route(P, bias=RACING_BIAS)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(rga.SCENE_XML))
    model.opt.timestep = d2t.SIM_TIMESTEP
    # 高分辨率离屏渲染需要加大 framebuffer（stage_show.py 同款写法）
    model.vis.global_.offwidth = max(args.width, 1920 if args.poster else args.width)
    model.vis.global_.offheight = max(args.height, 1080 if args.poster else args.height)
    data = mujoco.MjData(model)

    import re
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and re.match(r"^passive_.*", name):
            model.dof_frictionloss[model.jnt_dofadr[j]] = 0.003

    session = ort.InferenceSession(str(POLICY))
    assert session.get_inputs()[0].shape[-1] == 61
    h = d2t.DanceHarness(model, data, session)

    # 出生：塔顶发车平台（route[0] 沿 pad 前移 0.2m，防尾轮悬空），朝向 route[1]
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])
    spawn_yaw = math.atan2(ROUTE[1][1] - ROUTE[0][1], ROUTE[1][0] - ROUTE[0][0])
    data.qpos[qa] = ROUTE[0][0] + 0.2 * math.cos(spawn_yaw)
    data.qpos[qa + 1] = ROUTE[0][1] + 0.2 * math.sin(spawn_yaw)
    data.qpos[qa + 2] = ROUTE[0][2] + 0.1385
    data.qpos[qa + 3 : qa + 7] = [math.cos(spawn_yaw / 2), 0, 0, math.sin(spawn_yaw / 2)]
    for i, idx in enumerate(h.joint_qpos_indices):
        data.qpos[idx] = h.default_pose[i]
    data.ctrl[:] = h.default_pose
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    fpv_renderer = mujoco.Renderer(model, height=270, width=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fpv_cam = mujoco.MjvCamera()
    fpv_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fps = int(round(1.0 / d2t.CONTROL_DT))
    writer = imageio.get_writer(str(OUTDIR / "run.mp4"), fps=fps, macro_block_size=1)

    # ---- 海报静帧（高空正俯，duck 在塔顶起跑位） ----
    if args.poster:
        prenderer = mujoco.Renderer(model, height=1080, width=1920)
        pcam = mujoco.MjvCamera()
        pcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        pcam.lookat[:] = [0.4, 0.6, 0.0]
        pcam.azimuth = 90.0
        pcam.elevation = -89.0
        pcam.distance = 13.0
        prenderer.update_scene(data, camera=pcam)
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(args.poster), prenderer.render())
        prenderer.close()
        print(f"海报 → {args.poster}")

    csvf = open(OUTDIR / "probe.csv", "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["t", "sector", "cmd_x", "cmd_h", "v", "x", "y", "yaw_deg",
                   "trunk_z", "route_z", "wp_idx", "az"])

    path = [(x, y) for x, y, _, _, _ in ROUTE]
    zs = [z for _, _, z, _, _ in ROUTE]
    tags = [g for _, _, _, g, _ in ROUTE]
    sectors = [s for _, _, _, _, s in ROUTE]

    def xy():
        return np.array([float(data.qpos[qa]), float(data.qpos[qa + 1])])

    def yaw():
        q = data.qpos[qa + 3 : qa + 7]
        return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))

    def nearest_idx(p, from_idx):
        """前向窗口最近点（roller_gp 同款，防赛道折返跳索引）。"""
        best, bestd = from_idx, float("inf")
        for i in range(max(0, from_idx - 1), min(len(path), from_idx + 5)):
            d = math.hypot(path[i][0] - p[0], path[i][1] - p[1])
            if d < bestd:
                best, bestd = i, d
        return best, bestd

    def target_idx(ni):
        acc = 0.0
        prev = np.array(path[ni])
        for i in range(ni + 1, len(path)):
            seg = math.hypot(path[i][0] - prev[0], path[i][1] - prev[1])
            if acc + seg >= LOOKAHEAD:
                return i
            acc += seg
            prev = np.array(path[i])
        return len(path) - 1

    def dist_to_slower(ni):
        cur = PUSH[tags[ni]]
        acc = 0.0
        prev = np.array(path[ni])
        for i in range(ni + 1, len(path)):
            acc += math.hypot(path[i][0] - prev[0], path[i][1] - prev[1])
            if PUSH[tags[i]] < cur:
                return acc
            prev = np.array(path[i])
        return float("inf")

    # ---- HUD 资源 ----
    font_big = ImageFont.load_default(size=max(18, int(args.height * 0.055)))
    font_mid = ImageFont.load_default(size=max(12, int(args.height * 0.035)))
    font_huge = ImageFont.load_default(size=max(40, int(args.height * 0.22)))

    def draw_hud(img, t, v, sec, flash_on):
        d = ImageDraw.Draw(img)
        # 左上速度表
        d.text((14, 8), f"{v:.2f} m/s", font=font_big, fill=(240, 250, 255))
        d.text((16, 8 + font_big.size + 4), f"{v * 3.6:.1f} km/h",
               font=font_mid, fill=(140, 190, 220))
        # 右上赛段色块 + 赛段名
        name = f"{sec} {rga.SECTOR_NAME[sec]}"
        bb = d.textbbox((0, 0), name, font=font_mid)
        tw = bb[2] - bb[0]
        rgb = rga.SECTOR_RGB[sec]
        chip = (args.width - tw - 34, 8, args.width - 10, 8 + font_mid.size + 12)
        d.rectangle(chip, fill=tuple(int(c * 255) for c in rgb))
        d.text((args.width - tw - 22, 13), name, font=font_mid, fill=(5, 8, 12))
        # 中央倒计时
        if t < COUNTDOWN_S:
            n = 3 - int(t / (COUNTDOWN_S / 3))
            txt = str(n)
        elif t < COUNTDOWN_S + 0.7:
            txt = "GO!"
        else:
            txt = None
        if txt:
            bb = d.textbbox((0, 0), txt, font=font_huge)
            color = (255, 216, 60) if txt == "GO!" else (220, 240, 255)
            d.text(((args.width - (bb[2] - bb[0])) / 2,
                    (args.height - (bb[3] - bb[1])) / 2 - 20), txt,
                   font=font_huge, fill=color)
        # 破纪录爆闪
        if flash_on and int(t * 12) % 2 == 0:
            txt = "NEW TOP SPEED!"
            bb = d.textbbox((0, 0), txt, font=font_big)
            d.text(((args.width - (bb[2] - bb[0])) / 2, args.height * 0.16), txt,
                   font=font_big, fill=(255, 216, 60))
        return img

    def fpv_frame(duck_pos, duck_yaw):
        """车头视角小图：眼点 trunk 前方 0.14m/高 0.09m，看前方 0.6m。"""
        fx, fy = math.cos(duck_yaw), math.sin(duck_yaw)
        eye = np.array([duck_pos[0] + 0.14 * fx, duck_pos[1] + 0.14 * fy,
                        duck_pos[2] + 0.09])
        look = np.array([duck_pos[0] + 0.6 * fx, duck_pos[1] + 0.6 * fy,
                         duck_pos[2] - 0.02])
        dvec = eye - look
        dist = float(np.linalg.norm(dvec))
        fpv_cam.lookat[:] = look
        fpv_cam.distance = dist
        fpv_cam.azimuth = math.degrees(math.atan2(dvec[1], dvec[0]))
        fpv_cam.elevation = -math.degrees(math.asin(dvec[2] / dist))
        fpv_renderer.update_scene(data, camera=fpv_cam)
        return Image.fromarray(fpv_renderer.render())

    # ---- 运镜 ----
    camst = CamState()
    tower_top = np.array([ROUTE[0][0], ROUTE[0][1], ROUTE[0][2] + 0.2])
    poster_view = (np.array([0.4, 0.6, 0.0]), 90.0, -88.0, 12.5)

    def camera_target(mode, t, dp, dyaw):
        """各赛段镜头模式 → (lookat, azimuth, elevation, distance)。"""
        if mode == "intro":
            u = _smoothstep(min(t / INTRO_S, 1.0))
            lookat = (1 - u) * np.array([0.4, 0.5, 0.0]) + u * tower_top
            return (lookat, _ang_lerp(120.0, 300.0, u),
                    -88.0 + 56.0 * u, 12.5 - 10.1 * u)
        if mode == "A1":  # 螺旋塔外环绕跟拍
            ang = math.degrees(math.atan2(dp[1] - P.tower_c[1],
                                          dp[0] - P.tower_c[0]))
            lookat = 0.7 * dp + 0.3 * np.array([P.tower_c[0], P.tower_c[1], dp[2]])
            return (lookat, ang + 100.0, -20.0, 2.8)
        if mode == "A2":  # 俯冲贴地追尾
            return (dp + np.array([0.3 * math.cos(dyaw), 0.3 * math.sin(dyaw), 0.05]),
                    math.degrees(dyaw) + 180.0, -9.0, 1.2)
        if mode == "A3":  # 飞跃侧前方定机位
            return (np.array([(P.gap_x0 + P.gap_x1) / 2, P.dive_y, -0.05]),
                    -115.0, -10.0, 1.7)
        if mode == "A4":  # 左环环内低位追拍
            return (dp, math.degrees(dyaw) + 180.0, -16.0, 1.5)
        if mode == "A5":  # 右环立交桥：稍高追拍看爬升
            return (dp, math.degrees(dyaw) + 160.0, -22.0, 1.8)
        if mode == "A6":  # 下桥俯冲追尾
            return (dp, math.degrees(dyaw) + 180.0, -18.0, 1.6)
        if mode == "A7":  # 终点正面 punch-in
            return (dp, math.degrees(dyaw), -12.0, 1.2)
        if mode == "pullback":  # 收尾拉回高空海报机位
            u = _smoothstep(min((t - camst.switch_t) / 2.5, 1.0))
            sl, saz, sel, sdist = camst.start
            tl, taz, tel, tdist = poster_view
            return (sl + (tl - sl) * u, _ang_lerp(saz, taz, u),
                    sel + (tel - sel) * u, sdist + (tdist - sdist) * u)
        return (dp, math.degrees(dyaw) + 180.0, -15.0, 1.6)

    def render_frame(t, v, sec, flash_on, dp, dyaw, show_fpv):
        camst.update(camera_target(camst.mode, t, dp, dyaw), t)
        cam.lookat[:] = camst.lookat
        cam.azimuth = camst.az
        cam.elevation = camst.el
        cam.distance = camst.dist
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        if show_fpv:
            fpv = fpv_frame(dp, dyaw)
            px, py = args.width - 492, 46
            d = ImageDraw.Draw(img)
            d.rectangle([px - 3, py - 3, px + 482, py + 272],
                        outline=(0, 229, 255), width=2)
            img.paste(fpv, (px, py))
        if args.hud:
            img = draw_hud(img, t, v, sec, flash_on)
        writer.append_data(np.asarray(img))

    # ---- 主循环 ----
    sector_times = {}
    sector_order = []
    t = 0.0
    ni = 0
    pos_hist = []
    result = "TIMEOUT"
    v_max = 0.0
    flash_until = -1.0
    prev_vz = 0.0
    crossed = False
    cross_t = None
    finished = False

    print(f"Roller Gauntlet 发车：{len(path)} 个 waypoint，"
          f"{len(set(sectors))} 个赛段，{args.width}x{args.height}"
          f"（缺口 {'便桥' if P.gap_bridged else '真跳'}）")

    n_steps = int(args.timeout / d2t.CONTROL_DT)
    for step in range(n_steps):
        p = xy()
        pos_hist.append((t, p[0], p[1]))
        while pos_hist and pos_hist[0][0] < t - 0.25:
            pos_hist.pop(0)
        if len(pos_hist) >= 2:
            (t0_, x0_, y0_) = pos_hist[0]
            v = math.hypot(p[0] - x0_, p[1] - y0_) / max(t - t0_, 1e-6)
        else:
            v = 0.0

        # 路径跟踪
        ni, _ = nearest_idx(p, ni)
        ti = target_idx(ni)
        tgt = np.array(path[ti])
        dyaw = yaw()
        d_err = math.atan2(tgt[1] - p[1], tgt[0] - p[0]) - dyaw
        d_err = math.atan2(math.sin(d_err), math.cos(d_err))
        cmd_h = float(np.clip(d_err, -HEADING_CLIP, HEADING_CLIP))
        sec = sectors[ni]

        if t < COUNTDOWN_S:
            cmd_x, cmd_h = 0.0, 0.0  # 倒计时站定
        elif crossed:
            cmd_x, cmd_h = BRAKE, 0.0  # 冲线急停
        else:
            cmd_x = PUSH[tags[ni]]
            if abs(d_err) > 0.35:  # 大转角减速（Phase 0 实测）
                cmd_x = min(cmd_x, 0.3)
            # 提前缓刹：低速段前 BRAKE_DIST 米；高速按实测减速度 ~0.4m/s² 动态
            # 放大（A3 波浪谷出谷 ~1.1m/s 冲 A4 入环，0.45m 固定提前量不够）
            ds = dist_to_slower(ni)
            bd = max(BRAKE_DIST, (v * v - 0.3 * 0.3) / 0.8 + 0.3)
            if v > 0.35 and ds < bd:
                cmd_x = BRAKE
            # 注：R1p 撤掉了 A6 坡上限速缓刹——长平桥已把航向振荡阻尼掉，
            # 坡上缓刹的减速前倾反而把 duck 掀翻（t=104.5 v=0.56 前栽）

        h.command[:] = 0.0
        h.command[0] = cmd_x
        h.command[2] = cmd_h
        h.infer()
        h.apply_action(h.last_action)

        # 慢动作判定：飞跃区（kicker 前 0.3m → 波浪谷底爬出峡谷，~2s 实时
        # ≈8s 慢镜）或冲线后 0.5s
        in_jump = sec == "A3" and \
            P.kicker_x - SLOWMO_PAD_IN < p[0] < P.gap_x1 + SLOWMO_PAD_OUT
        finish_slowmo = crossed and (t - cross_t) < 0.5
        slowmo = in_jump or finish_slowmo

        for sub in range(d2t.DECIMATION):
            mujoco.mj_step(model, data)
            if slowmo and sub < d2t.DECIMATION - 1:
                dp = np.array([*xy(), float(data.qpos[qa + 2])])
                render_frame(t, v, sec, t < flash_until, dp, dyaw,
                             sec in ("A2", "A3"))
        t += d2t.CONTROL_DT

        # 赛段计时
        if not sector_order or sec != sector_order[-1]:
            sector_order.append(sec)
            sector_times[sec] = t
            if len(sector_order) > 1:
                print(f"  [{sec}] {rga.SECTOR_NAME[sec]} t={t:.1f}s v={v:.2f}m/s")

        # 冲线判定（赛段 A7 且过终点线 y，天空冲线台上）
        if not crossed and sec == "A7" and p[1] > P.finish_y:
            crossed = True
            cross_t = t
            result = "FINISH"
            camst.set_mode("pullback", t)
            print(f"  *** 冲线 t={t:.1f}s 总速 {v_max:.2f}m/s ***")

        # 摔倒 / 掉台 / 熔岩 DNF
        z, roll, pitch, vz, _ = h.trunk_state()
        az = (vz - prev_vz) / d2t.CONTROL_DT
        prev_vz = vz
        z_rel = z - zs[ni]
        settled = t > COUNTDOWN_S + 0.5  # 落地/站定宽限期（spawn 沉降会瞬时下蹲）
        if settled and (abs(roll) > d2t.FALL_TILT_THRESHOLD
                        or abs(pitch) > d2t.FALL_TILT_THRESHOLD
                        or z_rel < FALL_Z_REL):
            result = "DNF_FALL"
            print(f"  !! 摔倒 DNF t={t:.1f}s sector={sec} z_rel={z_rel:.3f}")
            break
        if P.gap_x0 < p[0] < P.gap_x1 and z < P.lava_z + 0.2 and not P.gap_bridged:
            result = "DNF_LAVA"
            print(f"  !! 坠熔岩 DNF t={t:.1f}s x={p[0]:.2f} z={z:.3f}")
            break
        if sec in ("A1", "A5", "A6", "A7") and z_rel < STAND_Z_REL - OFFDECK_DROP:
            result = "DNF_OFFDECK"
            print(f"  !! 掉台 DNF t={t:.1f}s sector={sec} z={z:.3f} "
                  f"route_z={zs[ni]:.3f}")
            break

        if t > COUNTDOWN_S and not crossed:
            if v > v_max:
                v_max = v
                if v > 0.72:  # 破纪录爆闪阈值（本赛道物理极速 ~0.8）
                    flash_until = t + 0.5
        v_max = max(v_max, v)

        csvw.writerow([f"{t:.3f}", sec, f"{cmd_x:+.2f}", f"{cmd_h:+.2f}",
                       f"{v:.3f}", f"{p[0]:.3f}", f"{p[1]:.3f}",
                       f"{math.degrees(dyaw):.1f}", f"{z:.4f}", f"{zs[ni]:.3f}",
                       ni, f"{az:.1f}"])

        # 运镜模式：开场 intro → 赛段模式 → 冲线 pullback
        if crossed:
            pass  # pullback 已在冲线时设置
        elif t < INTRO_S:
            camst.set_mode("intro", t)
        else:
            camst.set_mode(sec, t)

        dp = np.array([p[0], p[1], float(data.qpos[qa + 2])])
        render_frame(t, v, sec, t < flash_until, dp, dyaw, sec in ("A2", "A3"))

        if crossed and t - cross_t > 3.6:  # 急停 1s + 拉回海报 2.5s
            finished = True
            break

    writer.close()
    renderer.close()
    fpv_renderer.close()
    csvf.close()

    sector_report = {}
    for i, s in enumerate(sector_order):
        t_end = sector_times[sector_order[i + 1]] if i + 1 < len(sector_order) else t
        sector_report[s] = round(t_end - sector_times[s], 2)

    p = xy()
    summary = {
        "result": result, "total_s": round(t, 2),
        "v_max_ms": round(v_max, 3),
        "sector_split_s": sector_report,
        "sector_entry_t_s": {k: round(v_, 2) for k, v_ in sector_times.items()},
        "final_xy": [round(float(v_), 3) for v_ in p],
    }
    (OUTDIR / "sectors.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                    indent=2))
    print(f"\n=== 成绩: {result}  总时间 {t:.1f}s  极速 {v_max:.2f}m/s ===")
    for sec, dt_ in sector_report.items():
        print(f"  {sec} {rga.SECTOR_NAME[sec]}: {dt_}s")
    print(f"数据: {OUTDIR / 'probe.csv'}\n视频: {OUTDIR / 'run.mp4'}")
    return 0 if result == "FINISH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
