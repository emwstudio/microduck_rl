"""Sprint env cfg invariants — locks the sprint recipe against regressions."""

import math

from mjlab_microduck.tasks.microduck_sprint_env_cfg import (
    AIR_TIME_MAX_S,
    AIR_TIME_MIN_FINAL_S,
    AIR_TIME_MIN_START_S,
    AIR_TIME_WINDOW_STAGES,
    make_microduck_sprint_env_cfg,
    MicroduckSprintRlCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_sprint_command_ranges():
    cfg = make_microduck_sprint_env_cfg()
    cmd = cfg.commands["twist"]
    # v4：初始档在能力内有梯度，(-0.2, 2.0) 由课程到达（覆盖并超越 1.6 标杆）
    assert cmd.ranges.lin_vel_x == (-0.1, 0.8)
    # 侧向/转向收窄（比 walking 的 ±0.3 / ±1.0 窄，但保持非零防死权重）
    assert cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert cmd.ranges.ang_vel_z == (-0.5, 0.5)
    # 直线专精：关闭原地转向桶，半数 env 强制纯前向命令
    assert cmd.rel_turn_in_place_envs == 0.0
    assert cmd.rel_forward_envs == 0.5


def test_air_time_window_curriculum_ratchets_up_to_flight_phase():
    # v2 教训：feet_air_time 是阶跃函数，固定 0.20s 下限从行走步态永远够不到
    # （v1 全程 air_time 奖励 ≈0）。初始窗口必须可达，再由课程逐段上移。
    sprint = make_microduck_sprint_env_cfg()
    params = sprint.rewards["air_time"].params
    assert (params["threshold_min"], params["threshold_max"]) == (
        AIR_TIME_MIN_START_S,
        AIR_TIME_MAX_S,
    )
    # 课程存在且终点 = 跑步飞行相窗口
    stages = sprint.curriculum["air_time_window"].params["window_stages"]
    assert stages == AIR_TIME_WINDOW_STAGES
    assert stages[-1]["threshold_min"] == AIR_TIME_MIN_FINAL_S
    assert stages[-1]["threshold_max"] == AIR_TIME_MAX_S
    mins = [st["threshold_min"] for st in stages]
    assert mins == sorted(mins)  # 单调上移
    # v3：权重降为 1.5（只塑形），前向门控阈值 0.1
    assert sprint.rewards["air_time"].weight == 1.5
    assert params["command_threshold"] == 0.1


def test_tracking_std_keeps_gradient_at_sprint_speed():
    # reward = exp(-err²/std²)：std 太紧时 1 m/s 误差 → exp(-10) ≈ 0（无梯度）
    cfg = make_microduck_sprint_env_cfg()
    std = cfg.rewards["track_linear_velocity"].params["std"]
    assert std == math.sqrt(0.25)
    assert math.exp(-(1.0 / std) ** 2) > 0.01  # 1 m/s 误差处仍有可见梯度
    # v3：tracking 是主导目标
    assert cfg.rewards["track_linear_velocity"].weight == 4.0


def test_anti_violence_regularizers_inherited():
    cfg = make_microduck_sprint_env_cfg()
    # 动作平滑课程：起点 -0.1，v2 封顶 -0.3（-1.0 是 sprint 摆腿的 motion-blocker）
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == -0.3
    weights = [st["weight"] for st in stages]
    assert weights == sorted(weights, reverse=True)  # 单调加税
    # 躯干晃动 / 角动量 / 自碰撞 / 打滑 / 关节限位：全部为负权重惩罚
    assert cfg.rewards["body_ang_vel"].weight < 0
    assert cfg.rewards["angular_momentum"].weight < 0
    assert cfg.rewards["self_collisions"].weight < 0
    assert cfg.rewards["foot_slip"].weight < 0
    assert cfg.rewards["dof_pos_limits"].weight < 0


def test_command_speed_curriculum_reaches_2ms():
    # v4 核心：lin_vel_x 上限 0.8→1.2→1.6→2.0 分四段，单调加宽、终点达标
    cfg = make_microduck_sprint_env_cfg()
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    xmax = [st["lin_vel_x"][1] for st in stages]
    assert xmax == [0.8, 1.2, 1.6, 2.0]
    assert stages[-1]["lin_vel_x"] == (-0.2, 2.0)


def test_loose_tracking_floor_term():
    # v4：超宽松辅助项保底全速度域梯度（主项 std=0.5 在 2 m/s 误差处梯度为 0）
    cfg = make_microduck_sprint_env_cfg()
    term = cfg.rewards["track_linear_velocity_loose"]
    assert term.weight == 1.0
    assert term.params["std"] == 1.0
    assert term.params["command_name"] == "twist"
    import math as m
    assert m.exp(-((2.0 / 1.0) ** 2)) > 0.01  # 2 m/s 误差处仍有梯度


def test_burst_training_setup():
    # v4：一半 env 出生在命令速度 + 10s 短 episode（起步/加速练习密度翻倍）
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.commands["twist"].init_velocity_prob == 0.5
    assert cfg.episode_length_s == 10.0


def test_air_time_is_forward_gated():
    # v3 核心：原地腾空刷分必须支付 0 —— 门控函数替换 + 逆向课程出生
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_sprint_env_cfg()
    assert cfg.rewards["air_time"].func is microduck_mdp.feet_air_time_forward
    # v4：50% env 出生在命令速度状态（reverse-curriculum：高速前沿的在策略数据）
    assert cfg.commands["twist"].init_velocity_prob == 0.5


def test_upright_std_allows_sprint_lean():
    # v2：加速前倾必须买得起（walking 的 sqrt(0.05) 把前倾当摔倒定价）
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.rewards["upright"].params["std"] == math.sqrt(0.1)
    assert cfg.rewards["upright"].weight == 2.0  # 权重不动，只放宽容差


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
    # v2：站立奖励质量封顶 5%（v1 的 25% 教会了策略无视命令站桩）
    assert stages[-1]["rel_standing_envs"] == 0.05


def test_runner_cfg_is_sprint_specific():
    assert MicroduckSprintRlCfg.experiment_name == "sprint"
    assert MicroduckSprintRlCfg.run_name == "sprint"
    assert MicroduckSprintRlCfg.max_iterations == 4_000
    assert MicroduckSprintRlCfg.actor.obs_normalization is True
