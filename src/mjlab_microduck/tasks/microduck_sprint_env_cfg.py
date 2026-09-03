"""Microduck SPRINT task — 徒脚平地极速（追上并超越 1.6 m/s 标杆）。

Straight-line barefoot speed specialisation of the velocity (walking) recipe.
Built verbatim on make_microduck_velocity_env_cfg (the proven DR / obs / noise /
BAM stack) with sprint-focused deltas:

  - lin_vel_x widened (-0.4, 0.4) → (-0.2, 2.0): the target band is 1.6+ m/s.
    A small negative floor keeps a braking/backup skill alive without spending
    half the data on backwards walking.
  - 侧向/转向收窄: lin_vel_y ±0.3 → ±0.1, ang_vel_z ±1.0 → ±0.5, and
    turn-in-place OFF (0.15 → 0.0). Ranges stay non-zero so the command input
    neurons never die (61D contract; deployment still writes these slots).
  - rel_forward_envs = 0.5: half the envs get forward-only commands
    (|vx| clamped ≥ 0.3, vy/wz zeroed) — straight-line sprint is the objective,
    so half the experience is exactly that.
  - air_time window shifted UP [0.125, 0.300] → [0.20, 0.45] s: feet only score
    while airborne 0.2-0.45 s per swing, i.e. long ballistic strides / a real
    flight phase instead of quick walking steps. Beyond 0.45 s the foot falls
    out of the window, so it still pays to land and push again.
  - track_linear_velocity std loosened sqrt(0.1) → sqrt(0.25) (the mjlab
    default): reward = exp(-err²/std²), so at the old std a 1 m/s tracking
    error scores exp(-10) ≈ 0 — no gradient anywhere near sprint speeds during
    bootstrap. sqrt(0.25) keeps exp(-1) ≈ 0.37 at 0.5 m/s error and a visible
    0.018 at 1 m/s error. Tightening (fixed or std-curriculum) is a Phase-2
    recipe knob.

Anti-violence regularisation is inherited from the velocity recipe
(body_ang_vel, angular_momentum, self_collisions, foot_slip, dof_pos_limits,
upright, action_rate_l2) — with the v2 adjustments below.

Recipe v2 (2026-09-03) — v1 lesson: the policy PARKED. Deploy-side measurement
(warp env eval + CPU BAM harness, cmd up to 2.0) showed 0.006 m/s: it stood
still and farmed the standing/zero-command reward mass. Four fixes, all aimed
at making motion the argmax again:

  1. standing_envs curriculum capped at 0.05 (was 0.25): a quarter of envs
     paying full track+upright+pose reward for standing still taught
     "ignore the command and park". Zero-command behavior stays trained.
  2. air_time window starts REACHABLE and ratchets up via the
     air_time_window curriculum [0.05 → 0.20] s min: feet_air_time is a step
     function, so a fixed 0.20 s floor paid exactly zero from a 0.05 s-swing
     bootstrap gait — no gradient path to a flight phase (v1 logged
     air_time ≈ 0.0001 all run).
  3. action_rate_l2 ramp capped at -0.3 (was -1.0): sprint needs fast leg
     oscillation; the full walking-recipe tax is a motion-blocker at sprint
     cadence. Smoothness still damps jitter.
  4. upright std loosened sqrt(0.05) → sqrt(0.1): acceleration needs forward
     lean; the walking-tight tolerance priced lean as heavily as falling.

Recipe v3 (2026-09-03) — v2 lesson: the policy farmed air_time by stepping
high IN PLACE (air_time_mean 0.17 s, error_vel_xy 2.1 m/s, eval speed
0.04-0.11 m/s). Additive stacks get hacked on their under-specified terms:

  1. air_time is now FORWARD-GATED (feet_air_time_forward): in-window air
     steps pay × clamp(vx_actual / vx_commanded, 0, 1) — in-place flight
     pays exactly 0. Weight 3.0 → 1.5 (shaping, not a second objective).
  2. track_linear_velocity weight 2.0 → 4.0 — the dominant objective.
  3. init_velocity_prob = 0.3: a third of envs spawn ALREADY at their
     commanded velocity (reverse-curriculum spawns — the 2 m/s frontier
     otherwise gets no on-policy data, so propulsion at speed is
     unlearnable).

Recipe v4 (2026-09-04) — 短距离爆发极速（用户：不看长距离稳定，摔了有
垫子接，追求起步后短距离内最高速度）。v3 续训到 6000 迭代实测全部
checkpoint  plateau/回退（峰值仍是 2000 轮的 1.43 m/s）：不是步数问题，
是 track 高斯在 2 m/s 误差处梯度为零——70% 从静止起步的 env 在 1 m/s
以上没有加速梯度。

  1. 命令速度课程（commands_vel）：lin_vel_x 上限 0.8 → 1.2 → 1.6 → 2.0
     分四段 —— 每段的 tracking 误差都落在高斯有梯度的范围内，前沿始终
     在能力边缘（「课程与策略已学会的东西相位对齐」）。
  2. 辅助超宽松 tracking 项（std=1.0, weight 1.0）：全速度域保底梯度
     （2 m/s 误差处 exp(-4)≈0.018 而非 exp(-16)≈0）。
  3. init_velocity_prob 0.3 → 0.5：一半 env 出生在命令速度，高速在策略
     数据翻倍（爆发导向）。
  4. episode_length_s 20 → 10：reset 频率翻倍，单位时间的起步/加速
     练习翻倍 —— 短距离极速要的是发射段，不是巡航。

Flat terrain only. 61D obs contract preserved (twist(3) + head_pose(4) +
body_pose(6)) → the exported ONNX hot-swaps into the runtime walking slot.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Sprint command ranges (see module docstring). v4: lin_vel_x 上限走课程，
# 初始档必须落在当前能力内有梯度的范围；(-0.2, 2.0) 是课程终点。
LIN_VEL_X_START = (-0.1, 0.8)
LIN_VEL_X_FINAL = (-0.2, 2.0)
LIN_VEL_X_STAGES = [
    {"step": 0,          "lin_vel_x": (-0.1, 0.8)},
    {"step": 750 * 24,   "lin_vel_x": (-0.15, 1.2)},
    {"step": 1500 * 24,  "lin_vel_x": (-0.2, 1.6)},
    {"step": 2500 * 24,  "lin_vel_x": (-0.2, 2.0)},
]
LIN_VEL_Y_RANGE = (-0.1, 0.1)   # was ±0.3 — 收窄
ANG_VEL_Z_RANGE = (-0.5, 0.5)   # was ±1.0 — 收窄
REL_FORWARD_ENVS = 0.5          # half the envs run forward-only commands
TURN_IN_PLACE_FRACTION = 0.0    # sprint is straight-line; ang range stays non-zero

# Air-time window curriculum: start at a reachable walking-grade window and
# ratchet up to the flight-phase window (see docstring, v2 fix 2).
AIR_TIME_MIN_START_S = 0.05
AIR_TIME_MIN_FINAL_S = 0.20
AIR_TIME_MAX_S = 0.45
AIR_TIME_WINDOW_STAGES = [
    {"step": 0,          "threshold_min": 0.05, "threshold_max": AIR_TIME_MAX_S},
    {"step": 500 * 24,   "threshold_min": 0.10, "threshold_max": AIR_TIME_MAX_S},
    {"step": 1000 * 24,  "threshold_min": 0.15, "threshold_max": AIR_TIME_MAX_S},
    {"step": 1500 * 24,  "threshold_min": 0.20, "threshold_max": AIR_TIME_MAX_S},
]

# v2 fix 1: standing envs capped at 5% (was 25%) — enough to train the
# zero-command idle state, too little to farm.
STANDING_ENVS_FINAL = 0.05
# v2 fix 3: action_rate tax capped — sprint cadence must stay affordable.
ACTION_RATE_FINAL = -0.3
# v2 fix 4: sprint needs forward lean; walking-tight upright prices it out.
UPRIGHT_STD = math.sqrt(0.1)

# v3: velocity tracking is THE objective; air time only shapes the gait.
TRACK_LIN_VEL_WEIGHT = 4.0   # was 2.0
AIR_TIME_WEIGHT = 1.5        # was 3.0
# v3/v4: reverse-curriculum spawns — half of envs start AT the commanded
# velocity (burst-focused: double the at-speed on-policy data).
INIT_VELOCITY_PROB = 0.5
# v4: 短距离爆发 —— 10s episodes double the launch practice per GPU-hour.
EPISODE_LENGTH_S = 10.0
# v4: auxiliary loose tracking term — gradient floor over the whole speed range.
TRACK_LIN_VEL_LOOSE_WEIGHT = 1.0
TRACK_LIN_VEL_LOOSE_STD = 1.0

# Velocity-tracking std: loose enough that sprint-speed errors still see gradient.
TRACK_LIN_VEL_STD = math.sqrt(0.25)

# Symmetry mirror-loss: a sprint mirrors fine mechanically, but keep the project
# default (OFF; never enable for asymmetric tasks — this one is symmetric-safe
# but unproven, leave the knob for a Phase-2 recipe experiment).
ENABLE_SYMMETRY = False


def make_microduck_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Sprint-specialised velocity env (flat). See module docstring."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # --- Commands: sprint ranges + straight-line bias + speed curriculum ---
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X_START
    command.ranges.lin_vel_y = LIN_VEL_Y_RANGE
    command.ranges.ang_vel_z = ANG_VEL_Z_RANGE
    command.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    command.rel_forward_envs = REL_FORWARD_ENVS
    command.init_velocity_prob = INIT_VELOCITY_PROB

    # --- Flight-phase incentive: forward-gated air time + staged window ---
    cfg.rewards["air_time"].func = microduck_mdp.feet_air_time_forward
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_MIN_START_S
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_MAX_S
    cfg.rewards["air_time"].params["command_threshold"] = 0.1
    cfg.curriculum["air_time_window"] = CurriculumTermCfg(
        func=microduck_mdp.air_time_window_curriculum,
        params={
            "reward_name": "air_time",
            "window_stages": AIR_TIME_WINDOW_STAGES,
        },
    )

    # --- Tracking gradient at sprint speeds + dominant weight ---
    cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_VEL_STD
    cfg.rewards["track_linear_velocity"].weight = TRACK_LIN_VEL_WEIGHT

    # --- v2 fix 1: cap the stand-and-farm reward mass ---
    cfg.curriculum["standing_envs"].params["standing_stages"] = [
        {"step": 0,         "rel_standing_envs": 0.02},
        {"step": 500 * 24,  "rel_standing_envs": 0.03},
        {"step": 1000 * 24, "rel_standing_envs": STANDING_ENVS_FINAL},
    ]

    # --- v2 fix 3: cap the action-rate tax at sprint cadence ---
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0,          "weight": -0.1},
        {"step": 500 * 24,   "weight": -0.15},
        {"step": 1000 * 24,  "weight": -0.2},
        {"step": 1500 * 24,  "weight": ACTION_RATE_FINAL},
    ]

    # --- v2 fix 4: allow acceleration lean ---
    cfg.rewards["upright"].params["std"] = UPRIGHT_STD

    # --- v4: command-speed curriculum (gradient alive at every stage) ---
    from mjlab.tasks.velocity.mdp import commands_vel

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": LIN_VEL_X_STAGES,
        },
    )

    # --- v4: loose tracking floor term ---
    cfg.rewards["track_linear_velocity_loose"] = RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=TRACK_LIN_VEL_LOOSE_WEIGHT,
        params={"command_name": "twist", "std": TRACK_LIN_VEL_LOOSE_STD},
    )

    # --- v4: short episodes = burst training ---
    cfg.episode_length_s = EPISODE_LENGTH_S

    return cfg


MicroduckSprintRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="sprint",
    run_name="sprint",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
)

# deepcopy guard: the env cfg factory shares nothing with the module-level RL
# cfg, but keep the import used (matches other task modules' style).
_UNUSED = deepcopy  # noqa: F841
