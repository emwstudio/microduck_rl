"""Sprint env cfg invariants — locks the sprint recipe against regressions."""

import math

from mjlab_microduck.tasks.microduck_sprint_env_cfg import (
    AIR_TIME_MAX_S,
    AIR_TIME_MIN_S,
    make_microduck_sprint_env_cfg,
    MicroduckSprintRlCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_sprint_command_ranges():
    cfg = make_microduck_sprint_env_cfg()
    cmd = cfg.commands["twist"]
    # 主速度轴：覆盖并超越 1.6 m/s 标杆
    assert cmd.ranges.lin_vel_x == (-0.2, 2.0)
    # 侧向/转向收窄（比 walking 的 ±0.3 / ±1.0 窄，但保持非零防死权重）
    assert cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert cmd.ranges.ang_vel_z == (-0.5, 0.5)
    # 直线专精：关闭原地转向桶，半数 env 强制纯前向命令
    assert cmd.rel_turn_in_place_envs == 0.0
    assert cmd.rel_forward_envs == 0.5


def test_air_time_window_shifted_up_for_flight_phase():
    sprint = make_microduck_sprint_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    s_min = sprint.rewards["air_time"].params["threshold_min"]
    s_max = sprint.rewards["air_time"].params["threshold_max"]
    w_min = walk.rewards["air_time"].params["threshold_min"]
    w_max = walk.rewards["air_time"].params["threshold_max"]
    # 窗口整体上移：只有更长的腾空（跑步飞行相）才得分
    assert (s_min, s_max) == (AIR_TIME_MIN_S, AIR_TIME_MAX_S)
    assert s_min > w_min
    assert s_max > w_max
    # 权重与命令门限沿用 walking 配方
    assert sprint.rewards["air_time"].weight == walk.rewards["air_time"].weight == 3.0
    assert sprint.rewards["air_time"].params["command_threshold"] == 0.01


def test_tracking_std_keeps_gradient_at_sprint_speed():
    # reward = exp(-err²/std²)：std 太紧时 1 m/s 误差 → exp(-10) ≈ 0（无梯度）
    cfg = make_microduck_sprint_env_cfg()
    std = cfg.rewards["track_linear_velocity"].params["std"]
    assert std == math.sqrt(0.25)
    assert math.exp(-(1.0 / std) ** 2) > 0.01  # 1 m/s 误差处仍有可见梯度


def test_anti_violence_regularizers_inherited():
    cfg = make_microduck_sprint_env_cfg()
    # 动作平滑课程：起点 -0.1，终点 -1.0
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == -1.0
    # 躯干晃动 / 角动量 / 自碰撞 / 打滑 / 关节限位：全部为负权重惩罚
    assert cfg.rewards["body_ang_vel"].weight < 0
    assert cfg.rewards["angular_momentum"].weight < 0
    assert cfg.rewards["self_collisions"].weight < 0
    assert cfg.rewards["foot_slip"].weight < 0
    assert cfg.rewards["dof_pos_limits"].weight < 0


def test_bam_and_nan_guard_invariants_kept():
    cfg = make_microduck_sprint_env_cfg()
    # BAM 执行器下的摩擦字段展开（standalone env 必须注册）
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    # NaN 终止守卫
    assert "nan_state" in cfg.terminations


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_sprint_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    # 48 proprio + twist(3) + head_pose(4) + body_pose(6) = 61D
    assert list(terms.keys()) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]


def test_obs_parity_with_velocity_env():
    # 与 walking 完全同布局 → ONNX 可在 runtime 的 walking slot 热插拔
    sprint = make_microduck_sprint_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(sprint.observations[grp].terms.keys()) == list(
            walk.observations[grp].terms.keys()
        ), f"obs layout diverges on group {grp}"


def test_zero_command_behavior_still_trained():
    # 部署怠速 = 全零命令：standing envs 课程必须保留
    cfg = make_microduck_sprint_env_cfg()
    stages = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert stages[0]["rel_standing_envs"] > 0.0
    assert stages[-1]["rel_standing_envs"] == 0.25


def test_runner_cfg_is_sprint_specific():
    assert MicroduckSprintRlCfg.experiment_name == "sprint"
    assert MicroduckSprintRlCfg.run_name == "sprint"
    assert MicroduckSprintRlCfg.max_iterations == 4_000
    assert MicroduckSprintRlCfg.actor.obs_normalization is True
