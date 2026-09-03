"""roller_gauntlet_assets.py — TRON 霓虹夜赛跑道资产生成器。

两件事：
  1. PIL 程序化纹理 → artifacts/roller_gauntlet/textures/*.png
     （asphalt / gantry / arrow / lava / checker / pylon）
  2. 赛道几何 → 生成 scene_roller_gauntlet.xml（deck 段 + 发光路缘条全部由
     build_route() 的数学驱动，保证 XML 赛道与 roller_gauntlet.py 的
     waypoint 严格同源：改参数 → 重跑本脚本 → XML/waypoint 一起变）

几何关键参数全在 PARAMS（塔高/螺旋坡度/缺口宽/kicker 高/岸墙角度…），
生成的 XML 顶部注释也会抄一份，方便阶梯调试对照。

用法（工作目录 = third_party/microduck_rl）：
    uv run python scripts/roller_gauntlet_assets.py
"""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DUCKEMW_ROOT = REPO_ROOT.parents[1]
OUTDIR = DUCKEMW_ROOT / "artifacts" / "roller_gauntlet"
TEXDIR = OUTDIR / "textures"
SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_roller_gauntlet.xml"
# scene xml 目录 → DuckEMW 根：microduck→robot→mjlab_microduck→src→microduck_rl→third_party→DuckEMW
TEX_REL = "../../../../../../artifacts/roller_gauntlet/textures"


# ---------------------------------------------------------------------------
# 赛道参数（调试阶梯集中改这里；实测约束见各项注释，出处 = 物理红线实测）
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # 螺旋塔（下坡可以陡；但 2 圈×周长≈12m 内降 0.66m 物理上限 ~3.1°，末段滑槽补速度）
    tower_c: tuple = (-5.0, 1.0)     # 塔心 xy
    spiral_r: float = 1.1            # 螺旋中线半径
    spiral_turns: float = 1.75       # 圈数（630°）
    z_top: float = 0.66              # 塔顶 deck 面高度
    z_spiral_end: float = 0.11       # 螺旋出口 deck 面。双约束实测：出口自跨
                                     # z(0.43L)-0.016 > z_end+0.27 → ≤0.16；
                                     # 二圈 deck 过发车平台下 → ≤0.11
    chute_deg: float = 10.0          # 俯冲滑槽坡度（R2c：8° 稳但只到 0.77；
                                     # R2b：12° 无缓冲前栽；10°+出口缓冲 折中）
    spiral_step_deg: float = 15.0    # waypoint/geom 角距
    pad_len: float = 0.6             # 塔顶发车平台长（平地，倒计时防熘车后溜掉台）
    spiral_deck_w: float = 0.46      # 螺旋段再加宽（下坡+弯速度最快）
    helix_deck_w: float = 0.50       # 右环螺旋段（R1e 实测 CW 右弯内切 0.2m，加宽吸纳）
    deck_w: float = 0.40             # 行车道宽（≥0.32 红线；R1 实测弯中内切 0.12-0.23m，加宽）
    deck_half_th: float = 0.008      # deck 半厚
    # 俯冲直道（冰蓝）：y = dive_y，从螺旋出口到 kicker
    dive_y: float = -0.1
    kicker_x: float = -1.55          # kicker 唇口 x（起飞点）
    kicker_len: float = 0.20         # kicker 坡长
    kicker_h: float = 0.015          # kicker 唇高（R2=0.015；上坡唇 4.3° 短促冲过）
    # 飞跃缺口（x 方向贯穿全场的天堑，熔岩底）。
    # 降级记录：真跳 0.5m 缺口在 ~1.2m/s 起飞速度下飞行 0.2s+ 落 0.2m+，
    # 落地冲击超平地策略极限（物理红线 0.08-0.15m 台面教训）——采用贴地波浪
    # 飞坡（gap_bridged=False 生成 dip deck：8° 俯入 / 3.3° 缓爬出，熔岩两侧）
    gap_x0: float = -1.5             # 缺口西沿
    gap_x1: float = -1.0             # 缺口东沿
    gap_bridged: bool = False        # False = 波浪飞坡 dip deck
    ground2_z: float = -0.08         # 右半场地地面高度：波浪谷只降不爬
                                     # （谷底即右半场基准面，爬出 stall 风险归零）
    dip_down_deg: float = 8.0        # 俯入坡（谷底 = ground2_z）
    lava_z: float = -0.30            # 熔岩面高度
    # 8 字双环（左环 magenta CCW 450°，右环 orange CW 360° 螺旋升桥）
    loop_r: float = 1.6
    loop_step_deg: float = 15.0
    left_c: tuple = (2.9, 0.7)
    right_c: tuple = (5.9, 0.7)
    helix_top: float = 0.30          # 右环立交顶高（>鸭高 0.25+deck 厚+0.03 余量；
                                     # R1l/m 实测 0.42 配 12° 直坡下滑=0.9m/s 侧滑失控，
                                     # 降顶高缩短俯冲）
    bank_deg: float = 0.0            # 岸式弯墙外倾（R1=0；阶梯 10→15°）
    # 终点：右环顶 heading +y，下坡冲线
    finish_y: float = 2.80           # 终点线 y（在 z=helix_top 的天空冲线台上）
    # 场地
    ground1_x: tuple = (-7.5, None)  # 西 slab x 范围（None=gap_x0）
    ground2_x: tuple = (None, 8.5)   # 东 slab（None=gap_x1）
    ground_y: tuple = (-4.0, 5.0)


P = Params()

# 分区配色（emission 发光路缘条）
SECTOR_RGB = {
    "A1": (0.0, 0.9, 1.0),    # 螺旋塔 cyan
    "A2": (0.4, 0.7, 1.0),    # 俯冲冰蓝
    "A3": (1.0, 1.0, 1.0),    # 飞跃白
    "A4": (1.0, 0.2, 0.8),    # 左环 magenta
    "A5": (1.0, 0.55, 0.1),   # 右环橙
    "A6": (1.0, 0.85, 0.2),   # 下桥冲线 金黄
    "A7": (1.0, 0.85, 0.2),   # 终点区 金黄
}
SECTOR_NAME = {
    "A1": "SPIRAL TOWER", "A2": "NEON DIVE", "A3": "GAP JUMP",
    "A4": "LOOP-8 LEFT", "A5": "LOOP-8 SKYBRIDGE", "A6": "FINAL DROP",
    "A7": "FINISH",
}


# ---------------------------------------------------------------------------
# 路线生成（waypoint = (x, y, z, gear, sector)；z 为 deck 面高度）
# ---------------------------------------------------------------------------
def spiral_frame(p: Params):
    """螺旋起止相位：出口在塔南点 (φ=-90°) heading +x（衔接俯冲直道），
    起点 = 出口倒退 turns 圈（CCW 下降）。"""
    phi_end = -90.0
    phi0 = phi_end - p.spiral_turns * 360.0
    start = (p.tower_c[0] + p.spiral_r * math.cos(math.radians(phi0)),
             p.tower_c[1] + p.spiral_r * math.sin(math.radians(phi0)))
    tang = (-math.sin(math.radians(phi0)), math.cos(math.radians(phi0)))  # CCW 切向
    return phi0, phi_end, start, tang


def _spiral_points(p: Params, rb: float = 0.0):
    """螺旋塔：φ0 CCW 下降 630° 到塔南点，出口 heading +x。实测出处：螺旋全
    下坡，平地策略下坡可以陡，但 12m 弧长降 0.65m 物理上只有 ~3.1°，速度靠
    末段俯冲直道/滑槽补。rb = 赛车线外偏（控制 waypoint 用，几何用 0）。"""
    pts = []
    n = int(round(p.spiral_turns * 360.0 / p.spiral_step_deg))  # 42
    phi0, _, _, _ = spiral_frame(p)
    r = p.spiral_r + rb
    # 均匀螺距：多圈叠层净空 = 每圈降高 - deck厚 ≥ 鸭高 0.25（R2 实测变螺距
    # 「前平后陡」让二圈只比一圈低 0.12m，duck 撞头卡在发车平台沿）
    for i in range(n + 1):
        phi = math.radians(phi0 + i * p.spiral_step_deg)
        z = p.z_top + (p.z_spiral_end - p.z_top) * i / n
        pts.append((p.tower_c[0] + r * math.cos(phi),
                    p.tower_c[1] + r * math.sin(phi), z, "cruise", "A1"))
    # R2f 试 cruise：slow 档均衡速度只有 0.65 压低极速上限；
    # 加宽 deck(0.46)+赛车线外偏后重试（R1b 时代曾摔，现已加固）
    return pts


def _hermite(p0, p1, m0, m1, n):
    """三次 Hermite 平滑过渡（落地→左环入环用）。"""
    out = []
    for i in range(1, n + 1):
        u = i / n
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        x = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]
        y = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]
        out.append((x, y))
    return out


def build_route(p: Params = P, bias: dict | None = None):
    """整条赛道 waypoint 列表 [(x, y, z, gear, sector), ...]。
    roller_gauntlet.py 与本生成器共用此函数，保证路线同源。
    bias = {sector: 径向外偏}：控制用赛车线（实测 duck 在 r≈1.5 环上稳态
    内切 0.13-0.18m、爬坡弯更深 —— 控制线外偏 0.1m 后实际轮迹落在 deck 中央）；
    几何/XML 生成必须 bias=None（deck 按无偏几何铺）。"""
    bias = bias or {}
    b1 = bias.get("A1", 0.0)
    b4 = bias.get("A4", 0.0)
    b5 = bias.get("A5", 0.0)
    route = []
    # A0→A1 塔顶发车平台（平地直段，倒计时站定防熘车）+ 螺旋塔
    _, _, sp_start, sp_tang = spiral_frame(p)
    sp_start = (p.tower_c[0] + (p.spiral_r + b1) * (sp_start[0] - p.tower_c[0]) / p.spiral_r,
                p.tower_c[1] + (p.spiral_r + b1) * (sp_start[1] - p.tower_c[1]) / p.spiral_r)
    pad_start = (sp_start[0] - sp_tang[0] * p.pad_len,
                 sp_start[1] - sp_tang[1] * p.pad_len)
    route.append((pad_start[0], pad_start[1], p.z_top, "cruise", "A1"))
    route.append((sp_start[0], sp_start[1], p.z_top, "cruise", "A1"))
    route += _spiral_points(p, b1)[1:]
    # A2 俯冲直道：螺旋出口先 0.6m 平地缓冲（R2b 实测出弯残余偏航直接上 12°
    # 坡 = 弯转坡叠加前栽，先摆直再下坡）→ chute_deg 滑槽 → 平地到 kicker
    x = p.tower_c[0] + 0.3
    z = p.z_spiral_end
    chute_tan = math.tan(math.radians(p.chute_deg))
    chute_start_x = p.tower_c[0] + 0.6
    while x < p.kicker_x - p.kicker_len:
        route.append((x, p.dive_y, max(z, 0.0), "fast", "A2"))
        if x >= chute_start_x:
            z -= chute_tan * 0.25
        x += 0.25
    # A3 波浪飞坡过熔岩天堑（kicker → 8° 俯入 → 谷底即右半场基准面，只降不爬）
    route.append((p.kicker_x, p.dive_y, p.kicker_h, "fast", "A3"))
    if p.gap_bridged:
        route.append(((p.gap_x0 + p.gap_x1) / 2, p.dive_y, 0.0, "fast", "A3"))
        route.append((p.gap_x1 + 0.5, p.dive_y, p.ground2_z, "fast", "A3"))
        route.append((p.gap_x1 + 1.2, p.dive_y, p.ground2_z, "cruise", "A3"))
        a3_end_x = p.gap_x1 + 1.2
    else:
        x_d, z_d = p.gap_x0, 0.0
        tan_dn = math.tan(math.radians(p.dip_down_deg))
        while z_d > p.ground2_z + 1e-9:    # 俯入（谷底 = 右半场基准面）
            x_d += 0.125
            z_d = max(p.ground2_z, z_d - tan_dn * 0.125)
            route.append((x_d, p.dive_y, z_d, "fast", "A3"))
        x_d += 0.25                        # 谷底平地衔接
        route.append((x_d, p.dive_y, p.ground2_z, "fast", "A3"))
        x_d += 0.5
        route.append((x_d, p.dive_y, p.ground2_z, "fast", "A3"))
        a3_end_x = x_d
    # A4 入环 Hermite → 左环 CCW 450°（南点入 → 绕整圈 → 切点 T 穿出，过 T 两次）
    # 入环全程 fast：R2c 实测 A3 尾 cruise 触发缓刹把 duck 刹到 0.15m/s，
    # 低速晃倒在 Hermite 中段——保持速度冲过去反而稳
    r4 = p.loop_r + b4
    south = (p.left_c[0], p.left_c[1] - r4)
    for x, y in _hermite((a3_end_x, p.dive_y), south, (1.4, 0.0), (1.4, 0.0), 5):
        route.append((x, y, p.ground2_z, "fast", "A4"))
    n_entry = int(round(450.0 / p.loop_step_deg))  # 30 步：-90° → 360°(≡0°=T)
    for k in range(1, n_entry + 1):
        phi = math.radians(-90.0 + k * p.loop_step_deg)
        # 末 90° 降 slow：提前缓刹进 T 点立交，防刹车+右转叠加大内切（R1e 教训）
        # 其余 fast 抢时间（总时长预算 120s）
        gear = "fast" if k <= n_entry - 6 else "slow"
        route.append((p.left_c[0] + r4 * math.cos(phi),
                      p.left_c[1] + r4 * math.sin(phi),
                      p.ground2_z, gear, "A4"))
    # A5 右环 CW 360° 螺旋爬升（φ: 180° → -180°）。前 2 步（30°）平地跑道：
    # 左环 CCW 过 T 后必经的 NE 象限与右环起点 NW 象限物理重叠（R1i 实测
    # 17mm 板沿把 duck 顶起卡死）——起点抹平与 A4 共面，爬坡从 φ=150° 开始
    n_helix = int(round(360.0 / p.loop_step_deg))  # 24
    r5 = p.loop_r + b5
    climb_end = n_helix - 2  # 末 30° 平顶：R1j 实测 A5→A6 转角触发大转角限速
    for k in range(1, n_helix + 1):                # (cmd_x≤0.3) 在 2.6° 坡上失速晃倒
        phi = math.radians(180.0 - k * p.loop_step_deg)
        z = p.ground2_z if k <= 2 else \
            p.ground2_z + (p.helix_top - p.ground2_z) * min((k - 2) / (climb_end - 2), 1.0)
        # 爬环要 cruise(0.4)：R1e/f 实测 slow(0.3) 在 2.4° 爬坡近失速晃倒
        # （roller_gp 同款教训：0.4 推力在 5.7° 坡底失速，坡上要顶格推）
        route.append((p.right_c[0] + r5 * math.cos(phi),
                      p.right_c[1] + r5 * math.sin(phi), z, "cruise", "A5"))
                      # 全程 cruise：R1n 实测末段 slow 的刹车外推（understeer）
                      # 把 duck 甩到平台外侧 r=1.84；制动放到 A6 平桥直道上
    # A6 下桥冲线 = 1.5m 平桥@helix_top（跨低弧净空 + 航向振荡阻尼段）。
    # 降级说明（R1l/m/p/q/r 五轮实证）：任何下桥方案都摔——10-12° 直坡重力
    # 加速到 0.9m/s 航向修正饱和侧滑；坡上缓刹减速前倾掀翻；7.2° 弧降弯中
    # 内切晃倒。故终点抬到桥顶「天空冲线台」，取消下桥段。
    t_xy = (p.left_c[0] + p.loop_r, p.left_c[1])
    t_x = t_xy[0] - b5  # 控制线：A5 控制末端 (r5=loop_r+b5) 正北切出
    bridge_len = 1.5
    n_b = int(round(bridge_len / 0.25))
    for k in range(1, n_b + 1):
        route.append((t_x, t_xy[1] + 0.25 * k, p.helix_top, "slow", "A6"))
    # A7 天空冲线台：z=helix_top 平地直道，finish_y 冲线，留 0.8m 刹停余量
    y = t_xy[1] + bridge_len + 0.25
    while y < p.finish_y + 0.8:
        route.append((t_x, y, p.helix_top, "cruise", "A7"))
        y += 0.25
    return route


# ---------------------------------------------------------------------------
# XML 生成
# ---------------------------------------------------------------------------
def _q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_yaw_pitch(yaw, pitch):
    """先 yaw(z) 后 pitch(局部 y)。pitch>0 = 行进方向下坡。"""
    qy = (0.0, math.sin(yaw / 2) * 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qy = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qp = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
    return _q_mul(qy, qp)


def _fmt(v, nd=4):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else str(v)


class XmlBuilder:
    def __init__(self):
        self.root = ET.Element("mujoco", model="scene_roller_gauntlet")
        self.world = None
        self.n_geom = 0

    def geom(self, parent, **kw):
        e = ET.SubElement(parent, "geom")
        for k, v in kw.items():
            e.set(k, str(v))
        self.n_geom += 1
        return e

    def light(self, parent, **kw):
        e = ET.SubElement(parent, "light")
        for k, v in kw.items():
            e.set(k, str(v))
        return e


def _deck_and_curbs(xb, parent, p0, p1, p: Params, sector, bank=0.0, deck_w=None):
    """沿 p0→p1 铺一段 deck（暗色反光）+ 两侧发光路缘条（视觉无碰撞）。
    p0/p1 = (x, y, z, ...) waypoint（z = deck 面）。返回 None。"""
    x0, y0, z0 = p0[0], p0[1], p0[2]
    x1, y1, z1 = p1[0], p1[1], p1[2]
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, L)  # 行进方向下坡为正
    q = _quat_yaw_pitch(yaw, pitch)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    quat_s = f"{_fmt(qw,5)} {_fmt(qx,5)} {_fmt(qy,5)} {_fmt(qz,5)}"
    half_w = (deck_w or p.deck_w) / 2
    # deck 面在 waypoint z，box 中心下沉 half_th（沿法线近似）
    cz = mz - p.deck_half_th
    xb.geom(parent, name=f"dk{xb.n_geom}", type="box",
            pos=f"{_fmt(mx)} {_fmt(my)} {_fmt(cz)}",
            size=f"{_fmt(L/2 + 0.02)} {_fmt(half_w)} {_fmt(p.deck_half_th)}",
            quat=quat_s, material="deck")
    # 路缘条（发光，contype=0 防卡轮；实测 duck 贴线滑行不该被 1cm 条绊倒）
    nx, ny = -math.sin(yaw), math.cos(yaw)  # 行进方向左侧法向
    rgb = SECTOR_RGB[sector]
    mat = f"curb_{sector}"
    for side in (-1, 1):
        off = side * (half_w - 0.015)
        xb.geom(parent, name=f"cb{xb.n_geom}", type="box",
                pos=f"{_fmt(mx + nx*off)} {_fmt(my + ny*off)} {_fmt(mz + 0.006)}",
                size=f"{_fmt(L/2 + 0.02)} 0.015 0.006",
                quat=quat_s, material=mat, contype="0", conaffinity="0")


def _bank_wall(xb, parent, p0, p1, p: Params, sector):
    """岸式倾斜外沿（右环外侧，bank_deg>0 时生效；发光矮墙，有碰撞防飞出）。"""
    if p.bank_deg <= 0:
        return
    x0, y0, z0, x1, y1, z1 = p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    yaw = math.atan2(dy, dx)
    # 外侧 = 远离环心一侧（右环 CW：外侧在行进右侧）
    nx, ny = math.sin(yaw), -math.cos(yaw)
    mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    bx, by = mx + nx * (p.deck_w / 2 + 0.02), my + ny * (p.deck_w / 2 + 0.02)
    bank = math.radians(p.bank_deg)
    # 绕行进轴(x 轴)外倾：先 yaw 后绕局部 x 转 bank
    qy_ = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qb = (math.sin(bank / 2), 0.0, 0.0, math.cos(bank / 2))
    q = _q_mul(qy_, qb)
    xb.geom(parent, name=f"bw{xb.n_geom}", type="box",
            pos=f"{_fmt(bx)} {_fmt(by)} {_fmt(mz + 0.03)}",
            size=f"{_fmt(L/2 + 0.02)} 0.02 0.05",
            quat=f"{_fmt(q[3],5)} {_fmt(q[0],5)} {_fmt(q[1],5)} {_fmt(q[2],5)}",
            material=f"curb_{sector}")


def _gantry(xb, parent, cx, cy, z_deck, yaw, p: Params, mat, tex_mat,
            name, half_span=None):
    """发光门架：两立柱 + 门楣横幅（贴纹理）。yaw = 行车方向。"""
    half_span = half_span or (p.deck_w / 2 + 0.06)
    nx, ny = -math.sin(yaw), math.cos(yaw)
    h = 0.42
    for side in (-1, 1):
        px, py = cx + nx * half_span * side, cy + ny * half_span * side
        xb.geom(parent, name=f"{name}_post{side}", type="box",
                pos=f"{_fmt(px)} {_fmt(py)} {_fmt(z_deck + h/2)}",
                size=f"0.025 0.025 {_fmt(h/2)}", material=mat)
    # 门楣：跨两柱，顶高 z_deck+h+0.06
    top = z_deck + h + 0.05
    xb.geom(parent, name=f"{name}_beam", type="box",
            pos=f"{_fmt(cx)} {_fmt(cy)} {_fmt(top)}",
            size=f"0.03 {_fmt(half_span + 0.04)} 0.07",
            quat=f"{_fmt(math.cos(yaw/2),5)} 0 0 {_fmt(math.sin(yaw/2),5)}",
            material=tex_mat, contype="0", conaffinity="0")
    # 顶帽发光条：俯拍时门楣读作发光线而非悬空灰板（美术反馈）
    xb.geom(parent, name=f"{name}_cap", type="box",
            pos=f"{_fmt(cx)} {_fmt(cy)} {_fmt(top + 0.082)}",
            size=f"0.032 {_fmt(half_span + 0.05)} 0.012",
            quat=f"{_fmt(math.cos(yaw/2),5)} 0 0 {_fmt(math.sin(yaw/2),5)}",
            material=mat, contype="0", conaffinity="0")


def build_scene_xml(p: Params = P) -> str:
    route = build_route(p)
    xb = XmlBuilder()
    r = xb.root

    comment = (
        f" TRON 霓虹夜赛跑道（roller_gauntlet_assets.py 生成，勿手改——改 PARAMS 重跑生成器）\n"
        f" 几何关键参数（调试阶梯对照）：\n"
        f"   塔高 z_top={p.z_top}  螺旋 {p.spiral_turns} 圈 r={p.spiral_r}"
        f"  螺旋坡度≈{math.degrees(math.atan2(p.z_top - p.z_spiral_end, p.spiral_turns * 2 * math.pi * p.spiral_r)):.2f}°\n"
        f"   kicker 高={p.kicker_h} 长={p.kicker_len} @x={p.kicker_x}"
        f"  天堑={p.gap_x1 - p.gap_x0:.2f}（{'便桥连通' if p.gap_bridged else '波浪谷 8° 俯入'}） 滑槽={p.chute_deg}°\n"
        f"   右环立交顶高={p.helix_top}（>鸭高 0.25+deck 厚） 岸墙={p.bank_deg}°"
        f"  终点=天空冲线台（无下坡段，R1l-r 实证下坡必摔）  地面2 z={p.ground2_z}\n"
        f" 物理红线遵守：螺旋/爬环全下坡或 ≤3.3° 缓坡；落差全靠缓坡/重力俯冲，不靠跳高台 ")
    r.insert(0, ET.Comment(comment))

    ET.SubElement(r, "include", file="robot_groundcontact_rollers.xml")

    vis = ET.SubElement(r, "visual")
    ET.SubElement(vis, "headlight", diffuse="0.35 0.35 0.4", ambient="0.10 0.10 0.14",
                  specular="0 0 0")
    ET.SubElement(vis, "rgba", haze="0.02 0.03 0.05 1")
    ET.SubElement(vis, "global", azimuth="90", elevation="-20",
                  offwidth="1920", offheight="1080")

    asset = ET.SubElement(r, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.015 0.025 0.05", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", name="asphalt", type="2d",
                  file=f"{TEX_REL}/asphalt.png")
    ET.SubElement(asset, "texture", name="lava", type="2d", file=f"{TEX_REL}/lava.png")
    ET.SubElement(asset, "texture", name="gantry", type="2d",
                  file=f"{TEX_REL}/gantry.png")
    ET.SubElement(asset, "texture", name="arrow", type="2d", file=f"{TEX_REL}/arrow.png")
    ET.SubElement(asset, "texture", name="checker", type="2d",
                  file=f"{TEX_REL}/checker.png")
    ET.SubElement(asset, "texture", name="pylon", type="2d", file=f"{TEX_REL}/pylon.png")
    ET.SubElement(asset, "material", name="ground", texture="asphalt",
                  texuniform="false", texrepeat="24 24", reflectance="0.25")
    ET.SubElement(asset, "material", name="deck", rgba="0.045 0.06 0.09 1",
                  reflectance="0.4", specular="0.3")
    ET.SubElement(asset, "material", name="lava", texture="lava", emission="0.45")
    ET.SubElement(asset, "material", name="gantry_tex", texture="gantry",
                  emission="0.8")
    ET.SubElement(asset, "material", name="arrow", texture="arrow", emission="0.85")
    ET.SubElement(asset, "material", name="checker", texture="checker",
                  emission="0.55")
    ET.SubElement(asset, "material", name="pylon", texture="pylon", emission="0.75")
    for sec, (cr, cg, cb) in SECTOR_RGB.items():
        ET.SubElement(asset, "material", name=f"curb_{sec}",
                      rgba=f"{cr} {cg} {cb} 1", emission="0.9")

    world = ET.SubElement(r, "worldbody")
    xb.world = world
    xb.light(world, pos="0 0 4.5", dir="0 0 -1", directional="true",
             diffuse="0.25 0.28 0.35")
    xb.light(world, name="lava_light", pos=f"{(p.gap_x0+p.gap_x1)/2} 0.5 0.7",
             dir="0 0 -1", diffuse="0.9 0.25 0.1", attenuation="1 0.4 0.2")
    xb.light(world, name="tower_light", pos=f"{p.tower_c[0]} {p.tower_c[1]} {p.z_top + 0.9}",
             dir="0 0 -1", diffuse="0.2 0.7 0.9", attenuation="1 0.5 0.25")
    fin_y = p.finish_y  # 天空冲线台（z=helix_top）
    xb.light(world, name="finish_light", pos=f"{p.left_c[0] + p.loop_r} {_fmt(fin_y)} {_fmt(p.helix_top + 0.7)}",
             dir="0 0 -1", diffuse="1.0 0.8 0.3", attenuation="1 0.5 0.25")

    gy0, gy1 = p.ground_y
    gcy, ghy = (gy0 + gy1) / 2, (gy1 - gy0) / 2
    # 西 slab（左半场，顶面 z=0）—— 厚度 0.25 让天堑有崖壁
    gx0 = p.ground1_x[0]
    xb.geom(world, name="ground_w", type="box",
            pos=f"{_fmt((gx0 + p.gap_x0)/2)} {_fmt(gcy)} -0.125",
            size=f"{_fmt((p.gap_x0 - gx0)/2)} {_fmt(ghy)} 0.125",
            material="ground")
    # 东 slab（右半场，顶面 z=ground2_z）
    gx1 = p.ground2_x[1]
    xb.geom(world, name="ground_e", type="box",
            pos=f"{_fmt((p.gap_x1 + gx1)/2)} {_fmt(gcy)} {_fmt(p.ground2_z - 0.125)}",
            size=f"{_fmt((gx1 - p.gap_x1)/2)} {_fmt(ghy)} 0.125",
            material="ground")
    # 天堑底：全场暗色沟底板（有碰撞，坠入=z 过低 DNF_FALL）+ 仅飞跃口下方的
    # 发光熔岩坑（美术反馈： orange 带贯穿全场太霸道 → 收成缺口下的发光坑底）
    xb.geom(world, name="canyon_floor", type="box",
            pos=f"{_fmt((p.gap_x0 + p.gap_x1)/2)} {_fmt(gcy)} {_fmt(p.lava_z - 0.08)}",
            size=f"{_fmt((p.gap_x1 - p.gap_x0)/2)} {_fmt(ghy)} 0.03",
            rgba="0.03 0.04 0.06 1")
    xb.geom(world, name="lava", type="box",
            pos=f"{_fmt((p.gap_x0 + p.gap_x1)/2)} {_fmt(p.dive_y + 0.1)} {_fmt(p.lava_z - 0.03)}",
            size=f"{_fmt((p.gap_x1 - p.gap_x0)/2 + 0.25)} 0.65 0.03",
            material="lava", contype="0", conaffinity="0")
    # R1 便桥（调试阶梯：gap_bridged=True 时连通缺口）
    if p.gap_bridged:
        xb.geom(world, name="gap_bridge", type="box",
                pos=f"{_fmt((p.gap_x0 + p.gap_x1)/2)} {p.dive_y} -0.008",
                size=f"{_fmt((p.gap_x1 - p.gap_x0)/2 + 0.06)} {_fmt(p.deck_w/2)} 0.008",
                material="deck")
        for side in (-1, 1):
            xb.geom(world, name=f"gap_bridge_curb{side}", type="box",
                    pos=f"{_fmt((p.gap_x0 + p.gap_x1)/2)} {_fmt(p.dive_y + side*(p.deck_w/2 - 0.015))} 0.006",
                    size=f"{_fmt((p.gap_x1 - p.gap_x0)/2 + 0.06)} 0.015 0.006",
                    material="curb_A3", contype="0", conaffinity="0")

    # kicker（R1 kicker_h=0 不生成；阶梯 0.01→0.02，坡角 10-15°）
    if p.kicker_h > 0:
        ang = math.atan2(p.kicker_h, p.kicker_len)
        q = _quat_yaw_pitch(0.0, -ang)  # 行进 +x 上坡
        cx = p.kicker_x - p.kicker_len / 2
        xb.geom(world, name="kicker", type="box",
                pos=f"{_fmt(cx)} {p.dive_y} {_fmt(p.kicker_h/2 - 0.006)}",
                size=f"{_fmt(math.hypot(p.kicker_len, p.kicker_h)/2 + 0.02)} {_fmt(p.deck_w/2)} 0.008",
                quat=f"{_fmt(q[3],5)} {_fmt(q[0],5)} {_fmt(q[1],5)} {_fmt(q[2],5)}",
                material="deck")
        for side in (-1, 1):
            xb.geom(world, name=f"kicker_curb{side}", type="box",
                    pos=f"{_fmt(cx)} {_fmt(p.dive_y + side*(p.deck_w/2 - 0.015))} {_fmt(p.kicker_h/2 + 0.004)}",
                    size=f"{_fmt(math.hypot(p.kicker_len, p.kicker_h)/2 + 0.02)} 0.015 0.006",
                    quat=f"{_fmt(q[3],5)} {_fmt(q[0],5)} {_fmt(q[1],5)} {_fmt(q[2],5)}",
                    material="curb_A3", contype="0", conaffinity="0")

    # 赛道 deck + 发光路缘条：A1 螺旋 / A2 滑槽 / A3 波浪谷 / A4 左环 /
    # A5 右环螺旋 / A6 平桥 / A7 冲线台（A2 平地末段 deck 与地面 slab 共面无碍）
    for i in range(len(route) - 1):
        sec = route[i][4]
        sec_next = route[i + 1][4]
        if sec in ("A1", "A2", "A5", "A6", "A7") or (sec == "A3" and not p.gap_bridged):
            w = {"A1": p.spiral_deck_w, "A5": p.helix_deck_w}.get(sec)
            # 螺旋顶平台 + 平桥段加宽（A5→A6 转角动量外甩余量，R1k）
            if sec == "A5" and route[i + 1][2] >= p.helix_top - 0.005:
                w = 0.62
            if sec == "A6":
                w = 0.50  # 平桥段全宽（侧滑余量，R1l）
            if sec == "A7":
                w = 0.50  # 天空冲线台全宽
            if sec == "A3":
                w = 0.50  # 波浪谷全宽
            _deck_and_curbs(xb, world, route[i], route[i + 1], p, sec, deck_w=w)
        elif sec == "A4":
            _deck_and_curbs(xb, world, route[i], route[i + 1], p, sec)
        if sec == "A5" or (sec == "A4" and sec_next == "A5"):
            _bank_wall(xb, world, route[i], route[i + 1], p, "A5")
    # 桥顶 helipad 整板（A5→A6 转角兜底：R1o 实测 duck 带出弯动量冲向西侧，
    # 左轮卡进末段螺旋 deck 与平桥间的楔形缝摔倒——整板接住任何出弯轨迹；
    # 尺寸收窄到转角区，不盖 A5 爬坡初段（其上方平桥净高只有 0.284））
    t_xy = (p.left_c[0] + p.loop_r, p.left_c[1])
    xb.geom(world, name="helipad", type="box",
            pos=f"{_fmt(t_xy[0])} {_fmt(t_xy[1] + 0.13)} {_fmt(p.helix_top - 0.008)}",
            size="0.45 0.30 0.008", material="deck")
    for ex, ey, hx, hy, mat in (
            (-0.45, 0.0, 0.008, 0.30, "curb_A5"), (0.45, 0.0, 0.008, 0.30, "curb_A6"),
            (0.0, -0.30, 0.45, 0.008, "curb_A5"), (0.0, 0.30, 0.45, 0.008, "curb_A6")):
        xb.geom(world, name=f"helipad_edge{xb.n_geom}", type="box",
                pos=f"{_fmt(t_xy[0] + ex)} {_fmt(t_xy[1] + 0.13 + ey)} {_fmt(p.helix_top + 0.004)}",
                size=f"{_fmt(hx)} {_fmt(hy)} 0.004", material=mat,
                contype="0", conaffinity="0")

    # 塔心柱 + 支撑柱（暗色 + pylon 发光纹）
    xb.geom(world, name="tower_core", type="cylinder",
            pos=f"{p.tower_c[0]} {p.tower_c[1]} {_fmt(p.z_top/2)}",
            size=f"0.09 {_fmt(p.z_top/2)}", material="pylon")
    phi0_deg, _, sp_start, sp_tang = spiral_frame(p)
    pad_start = (sp_start[0] - sp_tang[0] * p.pad_len,
                 sp_start[1] - sp_tang[1] * p.pad_len)
    for k in range(4):
        phi_deg = -135.0 - k * 90.0
        phi = math.radians(phi_deg)
        sx = p.tower_c[0] + (p.spiral_r - 0.1) * math.cos(phi)
        sy = p.tower_c[1] + (p.spiral_r - 0.1) * math.sin(phi)
        # 支撑柱顶到该处螺旋最低一圈 deck 底
        frac = min(max((phi_deg - phi0_deg) / (p.spiral_turns * 360.0), 0.0), 1.0)
        z_deck = p.z_top + (p.z_spiral_end - p.z_top) * frac
        h = max(z_deck - 0.02, 0.05)
        xb.geom(world, name=f"tower_leg{k}", type="cylinder",
                pos=f"{_fmt(sx)} {_fmt(sy)} {_fmt(h/2)}",
                size=f"0.03 {_fmt(h/2)}", rgba="0.06 0.08 0.12 1")
    # 发车平台不设立柱：塔下任何立柱都会戳穿下方螺旋 deck（R1h 实测 duck
    # 在 wp24 撞上 pad 立柱）——TRON 浮空即可
    pad_mid = ((pad_start[0] + sp_start[0]) / 2, (pad_start[1] + sp_start[1]) / 2)

    # 塔顶起跑门架（发车平台中央，heading = 螺旋 CCW 切向）
    pad_yaw = math.atan2(sp_tang[1], sp_tang[0])
    _gantry(xb, world, pad_mid[0], pad_mid[1], p.z_top, pad_yaw, p,
            "curb_A1", "gantry_tex", "start_gantry")
    # 终点门架（金黄 + checker 终点线贴花）：天空冲线台上，跨车道
    fin_y = p.finish_y
    _gantry(xb, world, t_xy[0], fin_y, p.helix_top, math.pi / 2, p,
            "curb_A6", "gantry_tex", "finish_gantry")
    xb.geom(world, name="finish_line", type="box",
            pos=f"{_fmt(t_xy[0])} {_fmt(fin_y)} {_fmt(p.helix_top + 0.003)}",
            size=f"{_fmt(p.deck_w/2 + 0.05)} 0.05 0.003",
            material="checker", contype="0", conaffinity="0")

    # 灯柱阵（俯冲直道平地段两侧，间距 0.5m，速度参照物；滑槽高架段不插）
    x = -3.7
    k = 0
    while x < p.kicker_x - 0.3:
        for side in (-1, 1):
            xb.geom(world, name=f"lamp{k}_{side}", type="cylinder",
                    pos=f"{_fmt(x)} {_fmt(p.dive_y + side * (p.deck_w/2 + 0.12))} 0.09",
                    size="0.012 0.09", material="curb_A2",
                    contype="0", conaffinity="0")
        x += 0.5
        k += 1

    # 全息广告牌 ×3（美术反馈：原侧面朝赛道看不见的灰板 → 面板正对赛道 +
    # 双立柱 + 顶部发光压条，无碰撞）
    billboards = [
        ("bb_arrow1", (p.gap_x0 - 0.7, p.dive_y - 1.1, 0.35), math.pi / 2,
         "arrow", "curb_A3"),     # 缺口南侧，面向 +y 赛道
        ("bb_arrow2", (p.gap_x1 + 0.9, p.dive_y + 1.2, 0.35), math.pi / 2,
         "arrow", "curb_A3"),     # 缺口北侧，面向 -y 赛道
        ("bb_checker", (t_xy[0] + 2.4, p.finish_y + 0.2, 0.4), 0.0,
         "checker", "curb_A6"),   # 冲线台东侧，面向 -x；R2d 教训：立柱必须
                                  # 离右环北弧 ≥0.5m（之前在 (5.7,2.44) 撞柱卡死）
    ]
    for name, (bx, by, bz), byaw, mat, frame_mat in billboards:
        gz = bz + (p.ground2_z if bx > p.gap_x1 else 0.0)
        # 面板后仰 35°（体育场大屏式：俯拍能读到纹理，不再是一条侧立的板）
        lean = math.radians(35.0)
        qyaw = (0.0, 0.0, math.sin(byaw / 2), math.cos(byaw / 2))
        qlean = (0.0, math.sin(lean / 2), 0.0, math.cos(lean / 2))
        q = _q_mul(qyaw, qlean)
        qs = f"{_fmt(q[3],5)} {_fmt(q[0],5)} {_fmt(q[1],5)} {_fmt(q[2],5)}"
        nx, ny = -math.sin(byaw), math.cos(byaw)  # 面板横向
        for side in (-1, 1):
            xb.geom(world, name=f"{name}_post{side}", type="cylinder",
                    pos=f"{_fmt(bx + nx*0.36*side)} {_fmt(by + ny*0.36*side)} {_fmt(gz - 0.22 + 0.14)}",
                    size=f"0.015 {_fmt(gz - 0.08)}", rgba="0.06 0.08 0.12 1",
                    contype="0", conaffinity="0")
        xb.geom(world, name=name, type="box",
                pos=f"{_fmt(bx)} {_fmt(by)} {_fmt(gz + 0.06)}",
                size="0.03 0.4 0.22", quat=qs,
                material=mat, contype="0", conaffinity="0")
        # 底部发光基座线
        xb.geom(world, name=f"{name}_cap", type="box",
                pos=f"{_fmt(bx)} {_fmt(by)} {_fmt(0.02 + (p.ground2_z if bx > p.gap_x1 else 0.0))}",
                size=f"0.04 0.42 0.015",
                quat=f"{_fmt(math.cos(byaw/2),5)} 0 0 {_fmt(math.sin(byaw/2),5)}",
                material=frame_mat, contype="0", conaffinity="0")

    # keyframe STAND（照抄 scene_roller_gp.xml；出生 xy/z 由脚本覆写到塔顶门内）
    keyframe = ET.SubElement(r, "keyframe")
    ET.SubElement(keyframe, "key", name="STAND", qpos=(
        f"{_fmt(route[0][0])} {_fmt(route[0][1])} {_fmt(route[0][2] + 0.1385)} 1 0 0 0 "
        "0 -0.08726646259971647 -0.457924 -0.004940 0.452984  0 0 "
        "0.3490658503988659 0.3490658503988659 0 0 "
        "0 0.08726646259971647 0.457924 0.004940 -0.452984 0 0"), ctrl=(
        "0 -0.08726646259971647 -0.457924 -0.004940 0.452984 "
        "0.3490658503988659 0.3490658503988659 0 0 "
        "0 0.08726646259971647 0.457924 0.004940 -0.452984"))

    xml = minidom.parseString(ET.tostring(r)).toprettyxml(indent="    ")
    return xml, len(route), xb.n_geom


# ---------------------------------------------------------------------------
# 纹理生成
# ---------------------------------------------------------------------------
def tex_asphalt(path):
    """深色沥青：#0a0e14 底 + 细密噪点 + 微弱网格线。"""
    rng = random.Random(7)
    img = Image.new("RGB", (512, 512), (10, 14, 20))
    px = img.load()
    for yy in range(512):
        for xx in range(512):
            n = rng.randint(-6, 8)
            r_, g_, b_ = px[xx, yy]
            px[xx, yy] = (max(0, r_ + n), max(0, g_ + n), max(0, b_ + n + 2))
    d = ImageDraw.Draw(img)
    for k in range(0, 512, 64):
        d.line([(k, 0), (k, 512)], fill=(16, 24, 34), width=1)
        d.line([(0, k), (512, k)], fill=(16, 24, 34), width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(path)


def tex_gantry(path):
    """门楣横幅：深色半透明底 + 大字 DUCK GP，亮青描边。"""
    img = Image.new("RGBA", (1024, 256), (4, 8, 14, 210))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1023, 255], outline=(0, 229, 255, 255), width=6)
    d.rectangle([10, 10, 1013, 245], outline=(0, 120, 160, 255), width=2)
    font = ImageFont.load_default(size=150)
    text = "DUCK GP"
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx, ty = (1024 - tw) / 2 - bb[0], (256 - th) / 2 - bb[1]
    # 描边后填白
    for ox in range(-4, 5, 2):
        for oy in range(-4, 5, 2):
            d.text((tx + ox, ty + oy), text, font=font, fill=(0, 229, 255, 255))
    d.text((tx, ty), text, font=font, fill=(240, 255, 255, 255))
    img.save(path)


def tex_arrow(path):
    """发光方向箭头贴花（透明底，三连 chevron，冰蓝）。"""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(3):
        x0 = 30 + i * 70
        pts = [(x0, 40), (x0 + 55, 128), (x0, 216), (x0 + 28, 216),
               (x0 + 83, 128), (x0 + 28, 40)]
        d.polygon(pts, fill=(120, 200, 255, 235))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    img.save(path)


def tex_lava(path):
    """熔岩：红橙噪点 + 暗色脉纹。"""
    rng = random.Random(11)
    img = Image.new("RGB", (512, 512), (40, 4, 2))
    px = img.load()
    for yy in range(512):
        for xx in range(512):
            n = rng.random()
            if n > 0.86:
                px[xx, yy] = (255, 120 + rng.randint(0, 100), 10)
            elif n > 0.6:
                px[xx, yy] = (160 + rng.randint(0, 60), 30 + rng.randint(0, 30), 5)
            else:
                px[xx, yy] = (30 + rng.randint(0, 25), 3, 2)
    img = img.filter(ImageFilter.GaussianBlur(2.0))
    d = ImageDraw.Draw(img)
    for _ in range(26):  # 暗脉
        x0, y0 = rng.randint(0, 512), rng.randint(0, 512)
        x1, y1 = x0 + rng.randint(-120, 120), y0 + rng.randint(-120, 120)
        d.line([(x0, y0), (x1, y1)], fill=(12, 1, 1), width=rng.randint(2, 6))
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img.save(path)


def tex_checker(path):
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for yy in range(4):
        for xx in range(4):
            if (xx + yy) % 2 == 0:
                d.rectangle([xx * 64, yy * 64, xx * 64 + 63, yy * 64 + 63],
                            fill=(8, 8, 10))
    img.save(path)


def tex_pylon(path):
    """发光柱条纹（横向渐变条带）。"""
    img = Image.new("RGB", (128, 512), (5, 10, 16))
    d = ImageDraw.Draw(img)
    for yy in range(0, 512, 42):
        d.rectangle([0, yy, 127, yy + 16], fill=(0, 190, 230))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img.save(path)


def main():
    TEXDIR.mkdir(parents=True, exist_ok=True)
    tex_asphalt(TEXDIR / "asphalt.png")
    tex_gantry(TEXDIR / "gantry.png")
    tex_arrow(TEXDIR / "arrow.png")
    tex_lava(TEXDIR / "lava.png")
    tex_checker(TEXDIR / "checker.png")
    tex_pylon(TEXDIR / "pylon.png")
    print(f"纹理 → {TEXDIR} (6 张)")

    xml, n_wp, n_geom = build_scene_xml(P)
    SCENE_XML.write_text(xml)
    print(f"场景 → {SCENE_XML}（{n_wp} waypoints, {n_geom} geoms）")

    # 自验：能被 MuJoCo 加载
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print(f"MuJoCo 加载 OK: ngeom={m.ngeom} nbody={m.nbody}")


if __name__ == "__main__":
    main()
