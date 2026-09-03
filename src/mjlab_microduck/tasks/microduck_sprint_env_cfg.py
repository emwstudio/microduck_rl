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

Flat terrain only. 61D obs contract preserved (twist(3) + head_pose(4) +
body_pose(6)) → the exported ONNX hot-swaps into the runtime walking slot.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Sprint command ranges (see module docstring).
LIN_VEL_X_RANGE = (-0.2, 2.0)
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

# Velocity-tracking std: loose enough that sprint-speed errors still see gradient.
TRACK_LIN_VEL_STD = math.sqrt(0.25)

# Symmetry mirror-loss: a sprint mirrors fine mechanically, but keep the project
# default (OFF; never enable for asymmetric tasks — this one is symmetric-safe
# but unproven, leave the knob for a Phase-2 recipe experiment).
ENABLE_SYMMETRY = False


def make_microduck_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Sprint-specialised velocity env (flat). See module docstring."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # --- Commands: sprint ranges + straight-line bias ---
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X_RANGE
    command.ranges.lin_vel_y = LIN_VEL_Y_RANGE
    command.ranges.ang_vel_z = ANG_VEL_Z_RANGE
    command.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    command.rel_forward_envs = REL_FORWARD_ENVS

    # --- Flight-phase incentive: reachable window, ratcheted up by curriculum ---
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_MIN_START_S
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_MAX_S
    cfg.curriculum["air_time_window"] = CurriculumTermCfg(
        func=microduck_mdp.air_time_window_curriculum,
        params={
            "reward_name": "air_time",
            "window_stages": AIR_TIME_WINDOW_STAGES,
        },
    )

    # --- Tracking gradient at sprint speeds ---
    cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_VEL_STD

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
