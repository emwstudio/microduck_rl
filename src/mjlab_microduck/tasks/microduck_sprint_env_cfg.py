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

Anti-violence regularisation is INHERITED UNCHANGED from the velocity recipe
(action_rate_l2 curriculum ramping to -1.0, body_ang_vel, angular_momentum,
self_collisions, foot_slip, dof_pos_limits, upright): sprint must be won with
stride length and cadence, not with trunk thrash — and the sim2real invariants
(no human-scale rotation caps, impact pressure via |a_z|/action_rate) still
hold at 2 m/s.

Flat terrain only. 61D obs contract preserved (twist(3) + head_pose(4) +
body_pose(6)) → the exported ONNX hot-swaps into the runtime walking slot.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

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

# Air-time window: longer flight per swing than walking (was 0.125–0.300 s).
AIR_TIME_MIN_S = 0.20
AIR_TIME_MAX_S = 0.45

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

    # --- Flight-phase incentive: air-time window shifted up ---
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_MIN_S
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_MAX_S

    # --- Tracking gradient at sprint speeds ---
    cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_VEL_STD

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
