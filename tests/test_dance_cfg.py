"""Dance task cfg invariant tests (CPU, no GPU).

Locks in: the 61D obs contract (layout parity with the velocity base), the
dance command mapping on the body_pose slot, reward sign conventions, joint
pattern resolution on the real models, and the analytic reference generator.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_NOMINAL_HEIGHT,
    make_microduck_dance_env_cfg,
    MicroduckDanceRlCfg,
)


# --------------------------------------------------------------------------- #
# Obs layout / registration-level invariants                                    #
# --------------------------------------------------------------------------- #


def test_actor_observation_layout_matches_velocity_base():
    # Exact term-order parity with the velocity recipe, group by group: this is
    # what keeps the obs at 61D with the [twist(3), head_pose(4), body_pose(6)]
    # command block intact, so the exported ONNX hot-swaps in the runtime.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    dance = make_microduck_dance_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(dance.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), f"obs layout diverges on group {grp}"


def test_command_slots_keep_the_13d_layout():
    cfg = make_microduck_dance_env_cfg()
    # twist keeps velocity semantics (tiny ranges, dead-weight guard)
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (-0.02, 0.02)
    assert twist.ranges.ang_vel_z == (-0.05, 0.05)
    # head_pose keeps its 4D pose-delta semantics
    assert isinstance(cfg.commands["head_pose"], microduck_mdp.UniformPoseCommandCfg)
    assert len(cfg.commands["head_pose"].ranges) == 4
    # body_pose carries the 6D dance command
    dance_cmd = cfg.commands["body_pose"]
    assert isinstance(dance_cmd, microduck_mdp.DanceCommandCfg)
    assert dance_cmd.bpm_range == (90.0, 140.0)
    lo, hi = dance_cmd.move_len_beats
    assert 1.0 <= lo < hi


def test_obs_command_terms_read_the_right_command_names():
    cfg = make_microduck_dance_env_cfg()
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        assert terms["head_command"].params["command_name"] == "head_pose"
        assert terms["body_command"].params["command_name"] == "body_pose"


# --------------------------------------------------------------------------- #
# Reward structure                                                              #
# --------------------------------------------------------------------------- #


def test_dance_rewards_present_and_gait_terms_removed():
    cfg = make_microduck_dance_env_cfg()
    for name in ("dance_body_tracking", "dance_joint_tracking", "dance_beat_sync"):
        assert name in cfg.rewards, name
        assert cfg.rewards[name].weight > 0.0, name  # positive task rewards
    # gait terms make no sense with planted feet
    for gone in ("air_time", "foot_clearance", "foot_swing_height"):
        assert gone not in cfg.rewards, gone
    # the base's body_pose_tracking_6d reads the slot with pose-delta
    # semantics — must not coexist with the dance mapping
    assert "body_pose_tracking" not in cfg.rewards


def test_penalty_weights_have_the_cost_sign():
    # mjlab-base cost functions return >= 0 → NEGATIVE weight (AGENTS.md).
    cfg = make_microduck_dance_env_cfg()
    for name in (
        "no_stepping",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "foot_slip",
        "self_collisions",
    ):
        assert cfg.rewards[name].weight < 0.0, name
    # motion-blockers must be lower than the walking recipe (dynamic task)
    assert cfg.rewards["body_ang_vel"].weight > -0.05
    assert cfg.rewards["angular_momentum"].weight > -0.02


def test_pose_regularizer_is_loosened_not_removed():
    cfg = make_microduck_dance_env_cfg()
    pose = cfg.rewards["pose"]
    assert 0.0 < pose.weight < 1.0
    # standing std must be the loosened dance std, not the tight walking one
    assert pose.params["std_standing"][r".*hip_roll.*"] >= 0.15


def test_fall_termination_and_nan_guard_inherited():
    cfg = make_microduck_dance_env_cfg()
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations
    # BAM plumbing comes from the velocity factory
    assert "expand_bam_friction_fields" in cfg.events


def test_walking_semantic_curricula_removed():
    cfg = make_microduck_dance_env_cfg()
    for gone in ("standing_envs", "head_pose_range", "body_pose_range"):
        assert gone not in cfg.curriculum, gone
    # smoothness ramps but caps below the walking recipe's -1.0
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights == sorted(weights, reverse=True)  # ramps more negative
    assert weights[-1] == -0.6


def test_symmetry_augmentation_is_disabled():
    # move one-hots / phase encoding are not left-right symmetric in obs
    assert MicroduckDanceRlCfg.algorithm.symmetry_cfg is None


# --------------------------------------------------------------------------- #
# Reference generator (pure function)                                           #
# --------------------------------------------------------------------------- #


def _ref(move, phase_beats, bpm=120.0):
    n = len(phase_beats)
    return microduck_mdp.dance_reference(
        phase_beats,
        torch.full((n,), bpm),
        torch.full((n,), move, dtype=torch.long),
    )


def test_dance_reference_is_deterministic_and_finite():
    phase = torch.linspace(0.0, 8.0, 801)  # 8 beats
    for move in range(microduck_mdp.DANCE_NUM_MOVES):
        r1 = _ref(move, phase, bpm=123.0)
        r2 = _ref(move, phase, bpm=123.0)
        for key, v1 in r1.items():
            assert torch.isfinite(v1).all(), (move, key)
            assert torch.equal(v1, r2[key]), (move, key)


def test_squat_bounce_lowest_point_on_the_beat():
    phase = torch.linspace(0.0, 4.0, 2001)
    dz = _ref(microduck_mdp.DANCE_MOVE_SQUAT_BOUNCE, phase)["dz"]
    # lowest at integer beats (φ = 0 mod 2π), standing height at half beats
    amp = microduck_mdp.DANCE_SQUAT_AMPLITUDE
    assert 0.02 <= amp <= 0.03  # design: 2–3 cm
    assert torch.isclose(dz[0], torch.tensor(-amp), atol=1e-6)
    assert dz.min() >= -amp - 1e-6 and dz.max() <= 1e-6
    at_half_beat = _ref(
        microduck_mdp.DANCE_MOVE_SQUAT_BOUNCE, torch.tensor([0.5])
    )["dz"][0]
    assert torch.isclose(at_half_beat, torch.tensor(0.0), atol=1e-6)


def test_weight_shift_is_two_beat_periodic():
    phase = torch.linspace(0.0, 2.0, 1001)
    r = _ref(microduck_mdp.DANCE_MOVE_WEIGHT_SHIFT, phase)
    # one beat later the roll is exactly opposite (period = 2 beats)
    later = _ref(microduck_mdp.DANCE_MOVE_WEIGHT_SHIFT, phase + 1.0)
    assert torch.allclose(r["droll"], -later["droll"], atol=1e-5)
    amp = microduck_mdp.DANCE_ROLL_AMPLITUDE
    assert r["droll"].abs().max() <= amp + 1e-6
    # hip reference carries the same half-beat wave, ankles untouched
    assert torch.allclose(
        r["dhip_roll"] / microduck_mdp.DANCE_HIP_ROLL_AMPLITUDE,
        r["droll"] / amp,
        atol=1e-5,
    )


def test_head_bob_is_double_beat_frequency():
    phase = torch.linspace(0.0, 1.0, 1001)
    dhead = _ref(microduck_mdp.DANCE_MOVE_HEAD_BOB, phase)["dhead_pitch"]
    amp = microduck_mdp.DANCE_HEAD_BOB_AMPLITUDE
    # D·sin(2φ): two full nods per beat, bounded by ±15°
    expected = amp * torch.sin(2.0 * (2.0 * math.pi * phase))
    assert torch.allclose(dhead, expected, atol=1e-5)
    assert amp <= math.radians(15.0) + 1e-9


def test_reference_is_zero_outside_its_move():
    phase = torch.linspace(0.0, 4.0, 401)
    # squat move: no roll / head reference
    r = _ref(microduck_mdp.DANCE_MOVE_SQUAT_BOUNCE, phase)
    for key in ("droll", "roll_rate_ref", "dhip_roll", "dhead_pitch"):
        assert torch.all(r[key] == 0.0), key
    # head_bob move: body stays standing
    r = _ref(microduck_mdp.DANCE_MOVE_HEAD_BOB, phase)
    for key in ("dz", "vz_ref", "droll", "dhip_roll"):
        assert torch.all(r[key] == 0.0), key


# --------------------------------------------------------------------------- #
# DanceCommand semantics (instantiated on a minimal fake env)                  #
# --------------------------------------------------------------------------- #


class _FakeEnv:
    num_envs = 8
    device = "cpu"


def _make_command():
    cfg = microduck_mdp.DanceCommandCfg()
    return microduck_mdp.DanceCommand(cfg, _FakeEnv())


def test_dance_command_slots_carry_valid_values():
    cmd = _make_command()
    env_ids = torch.arange(_FakeEnv.num_envs)
    cmd.reset(env_ids)
    c = cmd.command
    assert c.shape == (_FakeEnv.num_envs, 6)
    # phase encoding: sin/cos on the unit circle
    assert (c[:, 0].abs() <= 1.0).all() and (c[:, 1].abs() <= 1.0).all()
    assert torch.allclose(c[:, 0] ** 2 + c[:, 1] ** 2, torch.ones(8), atol=1e-5)
    # tempo_norm = BPM/120 within the sampled range
    lo, hi = microduck_mdp.DANCE_BPM_RANGE
    assert (c[:, 2] >= lo / 120.0).all() and (c[:, 2] <= hi / 120.0).all()
    # move one-hot: exactly one of the last three slots is 1
    assert torch.allclose(c[:, 3:6].sum(dim=1), torch.ones(8))
    assert set(cmd.move_id.tolist()) <= {0, 1, 2}


def test_dance_command_phase_advances_with_tempo():
    cmd = _make_command()
    env_ids = torch.arange(_FakeEnv.num_envs)
    cmd.reset(env_ids)
    phase0 = cmd.phase_beats.clone()
    dt = 0.02
    cmd.compute(dt)
    expected = phase0 + dt * cmd.bpm / 60.0
    assert torch.allclose(cmd.phase_beats, expected, atol=1e-6)


def test_dance_command_move_switches_and_never_repeats():
    cmd = _make_command()
    env_ids = torch.arange(_FakeEnv.num_envs)
    cmd.reset(env_ids)
    # run 100 s at 50 Hz — every env must switch move at least once, always to
    # a DIFFERENT move
    for _ in range(5000):
        prev = cmd.move_id.clone()
        cmd.compute(0.02)
        changed = cmd.move_id != prev
        assert torch.all(cmd.move_id[changed] != prev[changed])
    one_hot_sum = cmd.command[:, 3:6].sum(dim=1)
    assert torch.allclose(one_hot_sum, torch.ones(8))


# --------------------------------------------------------------------------- #
# Joint patterns resolve on the real models                                     #
# --------------------------------------------------------------------------- #


def _joint_and_actuator_names(xml):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    actuators = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    return joints, actuators


def test_dance_joint_patterns_resolve_on_walk_model():
    import re

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

    joints, actuators = _joint_and_actuator_names(MICRODUCK_WALK_XML)
    actuated = [j for j in joints if j in set(actuators)]
    matched = [
        j
        for j in actuated
        if any(re.fullmatch(p, j) for p in microduck_mdp._DANCE_JOINT_PATTERNS)
    ]
    # exactly the hip_roll pair + head_pitch, all actuated servo joints
    assert sorted(matched) == ["head_pitch", "left_hip_roll", "right_hip_roll"]


def test_dance_joint_patterns_resolve_on_backlash_model():
    import re

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_XML

    joints, actuators = _joint_and_actuator_names(MICRODUCK_WALK_BACKLASH_XML)
    actuated = [j for j in joints if j in set(actuators)]
    matched = [
        j
        for j in actuated
        if any(re.fullmatch(p, j) for p in microduck_mdp._DANCE_JOINT_PATTERNS)
    ]
    # backlash hinges are passive_* and must NOT be matched as actuated targets
    assert sorted(matched) == ["head_pitch", "left_hip_roll", "right_hip_roll"]
    # ... but each matched joint must HAVE a passive_*_backlash companion so the
    # through-backlash measurement in dance_joint_tracking finds it
    for name in matched:
        assert f"passive_{name}_backlash" in joints, name


def test_nominal_height_is_the_measured_stand_z():
    # STAND_Z, shared with the standup/ball_kick envs — not the keyframe FK
    # height (0.12), which ignores contact compression.
    assert DANCE_NOMINAL_HEIGHT == 0.115


# --------------------------------------------------------------------------- #
# Reward functions against REAL mjlab data shapes (mock env, mock state)        #
# --------------------------------------------------------------------------- #
# Regression: dance_beat_sync once indexed root_link_ang_vel_w[:, 3] on a
# (N, 3) tensor (IndexError at first reward computation on GPU). The mocks
# below use the exact mjlab Entity.data shapes — root pose (N,3)/(N,4),
# lin/ang vel (N,3), joint arrays (N, n_joints) — so any out-of-bounds index
# in these three reward functions fails here on CPU.

_SERVO_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
    "right_ankle",
]


class _MockAssetData:
    """Entity.data stand-in with the REAL mjlab tensor shapes."""

    def __init__(self, n: int):
        nj = len(_SERVO_JOINT_NAMES)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = DANCE_NOMINAL_HEIGHT
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_lin_vel_w = torch.zeros(n, 3)
        self.root_link_ang_vel_w = torch.zeros(n, 3)  # (N, 3) — NOT (N, 6)
        self.joint_pos = torch.zeros(n, nj)
        self.joint_vel = torch.zeros(n, nj)
        self.default_joint_pos = torch.zeros(n, nj)


class _MockAsset:
    def __init__(self, n: int):
        self.data = _MockAssetData(n)
        self.joint_names = list(_SERVO_JOINT_NAMES)

    def find_joints_by_actuator_names(self, patterns):
        import re

        ids, names = [], []
        for i, name in enumerate(self.joint_names):
            if any(re.fullmatch(p, name) for p in patterns):
                ids.append(i)
                names.append(name)
        return ids, names


class _MockTerrain:
    def __init__(self, n: int):
        self.env_origins = torch.zeros(n, 3)


class _MockScene:
    def __init__(self, n: int):
        self._robot = _MockAsset(n)
        self.terrain = _MockTerrain(n)

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


class _MockCommandManager:
    def __init__(self, term):
        self._term = term

    def get_term(self, name):
        assert name == "body_pose"
        return self._term


class _MockRewardEnv:
    """Minimal env exposing exactly what the dance rewards touch."""

    def __init__(self, n: int = 6):
        self.num_envs = n
        self.device = "cpu"
        self.scene = _MockScene(n)
        self.command_manager = _MockCommandManager(
            microduck_mdp.DanceCommand(microduck_mdp.DanceCommandCfg(), self)
        )
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.command_manager._term.reset(torch.arange(n))


def test_dance_rewards_run_on_real_data_shapes():
    env = _MockRewardEnv()
    n = env.num_envs
    for func in (
        microduck_mdp.dance_body_tracking,
        microduck_mdp.dance_joint_tracking,
        microduck_mdp.dance_beat_sync,
    ):
        r = func(env, command_name="body_pose")
        assert r.shape == (n,), (func.__name__, r.shape)
        assert torch.isfinite(r).all(), func.__name__
    # body/joint tracking are Gaussians in [0, 1]; beat_sync is a bounded
    # potential difference in [-1, 1]
    assert (microduck_mdp.dance_body_tracking(env) >= 0).all()
    assert (microduck_mdp.dance_body_tracking(env) <= 1).all()
    assert (microduck_mdp.dance_joint_tracking(env) >= 0).all()
    assert (microduck_mdp.dance_joint_tracking(env) <= 1).all()
    r = microduck_mdp.dance_beat_sync(env)
    assert (r >= -1).all() and (r <= 1).all()


def test_dance_beat_sync_first_step_after_reset_pays_zero():
    env = _MockRewardEnv()
    # episode_length_buf == 0 → prev := current → reward exactly 0
    r = microduck_mdp.dance_beat_sync(env)
    assert torch.all(r == 0.0)
    # subsequent steps pay γΦ − Φ_prev (finite, bounded)
    env.episode_length_buf += 1
    r = microduck_mdp.dance_beat_sync(env)
    assert torch.isfinite(r).all() and (r.abs() <= 1.0).all()


def test_dance_rewards_track_their_reference_channels():
    # A mock state exactly ON the reference must score higher than an
    # off-reference state — proves the indices read the intended channels.
    n = 4
    env = _MockRewardEnv(n)
    term = env.command_manager._term
    asset = env.scene["robot"]

    # Force every env to squat_bounce at the beat low point (φ = 0 mod 2π).
    term._move_id[:] = microduck_mdp.DANCE_MOVE_SQUAT_BOUNCE
    term._phase_beats[:] = 0.0
    term._write_command()
    ref = microduck_mdp.dance_reference(
        term.phase_beats, term.bpm, term.move_id
    )
    # On-reference: trunk at STAND_Z + dz_ref. Off-reference: standing height.
    on = microduck_mdp.dance_body_tracking(env)
    asset.data.root_link_pos_w[:, 2] = DANCE_NOMINAL_HEIGHT + ref["dz"]
    on_ref = microduck_mdp.dance_body_tracking(env)
    assert torch.all(on_ref > on)

    # weight_shift at quarter phase: roll reference at +amplitude.
    term._move_id[:] = microduck_mdp.DANCE_MOVE_WEIGHT_SHIFT
    term._phase_beats[:] = 0.5  # φ = π → sin(φ/2) = 1
    term._write_command()
    ref = microduck_mdp.dance_reference(
        term.phase_beats, term.bpm, term.move_id
    )
    amp = microduck_mdp.DANCE_ROLL_AMPLITUDE
    assert torch.allclose(ref["droll"], torch.full((n,), amp), atol=1e-5)
    # roll the mock trunk onto the reference: quat for pure roll about x
    half = float(amp) / 2.0
    asset.data.root_link_quat_w[:, 0] = math.cos(half)
    asset.data.root_link_quat_w[:, 1] = math.sin(half)
    on_roll = microduck_mdp.dance_body_tracking(env)
    asset.data.root_link_quat_w[:, 1] = 0.0
    asset.data.root_link_quat_w[:, 0] = 1.0
    off_roll = microduck_mdp.dance_body_tracking(env)
    assert torch.all(on_roll > off_roll)

    # joint tracking: hip_roll joints at HOME + reference delta score highest.
    ref_hip = ref["dhip_roll"]
    ids, names = asset.find_joints_by_actuator_names([r".*hip_roll.*"])
    off = microduck_mdp.dance_joint_tracking(env)
    asset.data.joint_pos[:, ids] = (
        asset.data.default_joint_pos[:, ids] + ref_hip.unsqueeze(-1)
    )
    on_joint = microduck_mdp.dance_joint_tracking(env)
    assert torch.all(on_joint > off)

    # beat_sync: velocity exactly matching the reference raises the potential.
    term._move_id[:] = microduck_mdp.DANCE_MOVE_SQUAT_BOUNCE
    term._phase_beats[:] = 0.25  # φ = π/2 → max |vz_ref|
    term._write_command()
    ref = microduck_mdp.dance_reference(
        term.phase_beats, term.bpm, term.move_id
    )
    env.episode_length_buf += 1  # avoid the first-step zero
    asset.data.root_link_lin_vel_w[:, 2] = 0.0
    microduck_mdp.dance_beat_sync(env)  # sets prev from the off-reference state
    asset.data.root_link_lin_vel_w[:, 2] = ref["vz_ref"]
    r_on = microduck_mdp.dance_beat_sync(env)
    assert torch.all(r_on > 0.0)  # moving onto the reference pays Δprogress
