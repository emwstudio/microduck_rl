#!/usr/bin/env python3
"""Drive a dance ONNX policy with a song beat timeline (CPU MuJoCo, no keyboard).

Validates that the trained dance policy (Mjlab-Dance-Flat-MicroDuck) actually
dances ON the beat: each 50 Hz control step, the sim clock is mapped through
the timeline to the dance command written into the body_pose obs slot.

Command mapping — MUST match DanceCommand in src/mjlab_microduck/tasks/mdp.py
(the training-side generator). The 13D command block is
[twist(3), head_pose(4), body_pose(6)]; the dance policy reads body_pose as:

    body_pose[0] = sin(φ/2)        φ = 2π·((t − t0)·BPM/60), phase over 2 beats
    body_pose[1] = cos(φ/2)
    body_pose[2] = BPM / 120       (tempo_norm)
    body_pose[3] = 1 if move == 0 (squat_bounce)
    body_pose[4] = 1 if move == 1 (weight_shift)
    body_pose[5] = 1 if move == 2 (head_bob)

twist and head_pose slots are written as zeros (in-distribution: both were
sampled in small ranges around 0 during training).

The rest of the 61D observation assembly is identical to infer_policy.py
(--new-cmd-obs layout): base_ang_vel(3) + projected_gravity(3) +
joint_pos_rel(14) + joint_vel(14) + last_action(14) + command(13).

Physics matches training: mjlab runs the velocity/dance envs at
timestep=0.005 s with decimation 4 (50 Hz control). The scene XMLs ship the
MuJoCo default 0.002 s, so this script OVERRIDES model.opt.timestep to 0.005
(infer_policy.py does not — it effectively runs its control loop at 125 Hz on
these scenes, a known discrepancy we don't replicate here).

Usage:
    uv run python scripts/dance_to_timeline.py --policy dance.onnx \
        --timeline tests/fixtures/click120.timeline.json \
        --record out.mp4 --save-csv out.csv
    # no trained ONNX yet → plumbing smoke test with a zero-action mock policy:
    uv run python scripts/dance_to_timeline.py --mock-policy \
        --timeline tests/fixtures/click120.timeline.json --save-csv out.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"

# STAND2 pose — MUST match HOME_FRAME in robot/microduck_constants.py and
# DEFAULT_POSE in scripts/infer_policy.py (same joint order: actuator order).
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)

MOVE_NAMES = {0: "squat_bounce", 1: "weight_shift", 2: "head_bob"}
NUM_MOVES = 3  # the trained policy knows moves 0-2 only

# Physics / control (match training — see module docstring)
SIM_TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = SIM_TIMESTEP * DECIMATION  # 0.02 s = 50 Hz

# Fall detection (standing trunk z is STAND_Z = 0.115)
FALL_Z_THRESHOLD = 0.07
FALL_TILT_THRESHOLD = math.radians(45.0)

# Spawn height: midpoint of the training reset_base z range (0.12, 0.13)
SPAWN_Z = 0.125


# ---------------------------------------------------------------------------
# Pure timeline → command mapping (unit-tested in tests/test_dance_timeline.py)
# ---------------------------------------------------------------------------

def load_timeline(path):
    """Load and validate a timeline JSON produced by DuckEMW dance/timeline.py."""
    data = json.loads(Path(path).read_text())
    for key in ("bpm", "t0", "duration", "beat_times", "segments"):
        if key not in data:
            raise ValueError(f"timeline missing required key '{key}'")
    if not data["beat_times"]:
        raise ValueError("timeline has no beat_times")
    if not data["segments"]:
        raise ValueError("timeline has no segments")
    segments = sorted(data["segments"], key=lambda s: s["t_start"])
    for seg in segments:
        for key in ("move", "t_start", "t_end"):
            if key not in seg:
                raise ValueError(f"segment missing required key '{key}': {seg}")
        if seg["move"] not in MOVE_NAMES:
            raise ValueError(
                f"segment uses move {seg['move']} ({seg.get('move_name', '?')}) — "
                f"the trained dance policy only knows moves 0-{NUM_MOVES - 1} "
                f"({sorted(MOVE_NAMES)}); regenerate the timeline with --moves 0,1,2"
            )
    data["segments"] = segments
    return data


def beat_phase(t, t0, bpm):
    """Beat phase φ ∈ [0, 2π) at song time t. φ = 0 at every beat_time."""
    beats = (t - t0) * bpm / 60.0
    return (2.0 * math.pi * beats) % (2.0 * math.pi)


def segment_at(segments, t):
    """Segment active at time t.

    Adjacent segments overlap slightly (each segment's t_end is the beat AFTER
    its last beat, i.e. inside the next segment's range): the LATER segment
    wins the overlap. Before the first segment's t_start, the first segment's
    move is used (the dance starts anyway).
    """
    active = segments[0]
    for seg in segments:
        if seg["t_start"] <= t:
            active = seg
        else:
            break
    return active


def dance_command(t, timeline):
    """6D dance command for the body_pose obs slot at song time t.

    Returns np.float32 (6,): [sin(φ/2), cos(φ/2), bpm/120, one-hot(move)]. Mirrors
    DanceCommand._write_command in src/mjlab_microduck/tasks/mdp.py exactly —
    if that function changes, change this one (the equivalence is locked by
    tests/test_dance_timeline.py::test_dance_command_matches_training_semantics).
    """
    # UNWRAPPED beats × π = phase over a 2-beat cycle (mirrors
    # DanceCommand._write_command; do NOT use the per-beat wrapped beat_phase).
    phi_half = math.pi * (t - timeline["t0"]) * timeline["bpm"] / 60.0
    move = segment_at(timeline["segments"], t)["move"]
    cmd = np.zeros(6, dtype=np.float32)
    cmd[0] = math.sin(phi_half)
    cmd[1] = math.cos(phi_half)
    cmd[2] = timeline["bpm"] / 120.0
    cmd[3 + move] = 1.0
    return cmd


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _quat_rotate_inverse(quat, vec):
    """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def _roll_pitch_from_quat(quat):
    """ZYX roll/pitch from [w, x, y, z] (same formulas as mdp.body_pose_tracking_6d)."""
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


class DanceHarness:
    """61D obs assembly + action application, mirroring infer_policy.py."""

    def __init__(self, model, data, ort_session=None, action_scale=1.0):
        self.model = model
        self.data = data
        self.ort_session = ort_session
        self.action_scale = action_scale
        if ort_session is not None:
            self.input_name = ort_session.get_inputs()[0].name
            self.output_name = ort_session.get_outputs()[0].name

        self.imu_ang_vel_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )
        if self.imu_ang_vel_id < 0:
            raise ValueError("Sensor 'imu_ang_vel' not found in model")
        self.trunk_base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )

        # Actuated-joint qpos/qvel indices via actuator transmissions (the same
        # mechanism infer_policy uses; correct for any joint ordering).
        self.n_joints = model.nu
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]
        self.default_pose = DEFAULT_POSE[: self.n_joints]
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)

    def get_base_ang_vel(self):
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr : sensor_adr + 3].copy().astype(np.float32)

    def get_projected_gravity(self):
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        return _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    def get_observations(self):
        """61D: ang_vel(3) proj_grav(3) joint_pos_rel(14) joint_vel(14)
        last_action(14) command(13) — identical layout to infer_policy.py."""
        obs = [
            self.get_base_ang_vel(),
            self.get_projected_gravity(),
            self.data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
            - self.default_pose,
            self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32),
            self.last_action,
            self.command,
        ]
        return np.concatenate(obs).astype(np.float32)

    def infer(self):
        obs = self.get_observations().reshape(1, -1)
        action = self.ort_session.run([self.output_name], {self.input_name: obs})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def apply_action(self, action):
        self.data.ctrl[:] = self.default_pose + action * self.action_scale

    def trunk_state(self):
        """(z, roll, pitch, vz, roll_rate) of the trunk in the world frame.

        cvel[body] = (ωx, ωy, ωz, vx, vy, vz) at the body CoM in world frame
        (this is what mjlab exposes as root_link_ang_vel_w / lin_vel_w).
        """
        pos = self.data.xpos[self.trunk_base_id]
        quat = self.data.xquat[self.trunk_base_id]
        cvel = self.data.cvel[self.trunk_base_id]
        roll, pitch = _roll_pitch_from_quat(quat)
        return float(pos[2]), roll, pitch, float(cvel[5]), float(cvel[0])


def run(args):
    timeline = load_timeline(args.timeline)
    print(f"Timeline: {timeline.get('name', '?')}  bpm={timeline['bpm']:.2f}  "
          f"t0={timeline['t0']:.3f}s  duration={timeline['duration']:.1f}s  "
          f"segments={len(timeline['segments'])}")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    # Match the training physics rate (see module docstring).
    model.opt.timestep = SIM_TIMESTEP
    data = mujoco.MjData(model)

    session = None
    if args.mock_policy or not args.policy:
        print("MOCK POLICY: zero actions (holds HOME pose). Validates the "
              "timeline→command mapping and logging plumbing, not dancing.")
    else:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.policy))
        in_shape = session.get_inputs()[0].shape
        print(f"Policy: {args.policy}  input shape {in_shape}")
        if in_shape[-1] != 61:
            print(f"WARNING: policy expects {in_shape[-1]}D obs, the dance "
                  f"contract is 61D", file=sys.stderr)

    harness = DanceHarness(model, data, session, action_scale=args.action_scale)

    # Initial state = HOME pose, same as infer_policy.py / training spawn.
    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = SPAWN_Z
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(harness.joint_qpos_indices):
        data.qpos[idx] = harness.default_pose[i]
    data.ctrl[:] = harness.default_pose
    mujoco.mj_forward(model, data)

    # Optional video recording (offscreen render at the control rate).
    writer = None
    renderer = None
    camera = None
    if args.record:
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.6
        camera.azimuth = 135.0
        camera.elevation = -15.0
        writer = imageio.get_writer(str(args.record), fps=int(round(1.0 / CONTROL_DT)))
        print(f"Recording video → {args.record}")

    csv_rows = []
    duration = float(timeline["duration"])
    n_steps = int(round(duration / CONTROL_DT))
    falls = 0
    fallen = False

    print(f"Running {duration:.1f}s at {1.0 / CONTROL_DT:.0f} Hz control "
          f"({n_steps} steps, dt={SIM_TIMESTEP}s × {DECIMATION})")

    for step in range(n_steps):
        t = step * CONTROL_DT

        # Timeline → dance command (body_pose slot); twist/head_pose stay 0.
        harness.command[:] = 0.0
        cmd6 = dance_command(t, timeline)
        harness.command[7:13] = cmd6
        move = int(np.argmax(cmd6[3:6]))

        if session is not None:
            action = harness.infer()
        else:
            action = np.zeros(harness.n_joints, dtype=np.float32)
            harness.last_action = action.copy()
        harness.apply_action(action)

        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        z, roll, pitch, vz, roll_rate = harness.trunk_state()
        is_fallen = (
            z < FALL_Z_THRESHOLD
            or abs(roll) > FALL_TILT_THRESHOLD
            or abs(pitch) > FALL_TILT_THRESHOLD
        )
        if is_fallen and not fallen:
            falls += 1
            print(f"  FALL #{falls} at t={t:.2f}s "
                  f"(z={z:.3f} roll={math.degrees(roll):.0f}° "
                  f"pitch={math.degrees(pitch):.0f}°)")
        fallen = is_fallen

        csv_rows.append({
            "step": step,
            "t": f"{t:.4f}",
            "phase": f"{math.atan2(cmd6[0], cmd6[1]):.6f}",
            "move": move,
            "move_name": MOVE_NAMES[move],
            "trunk_z": f"{z:.6f}",
            "trunk_roll": f"{roll:.6f}",
            "trunk_pitch": f"{pitch:.6f}",
            "trunk_vz": f"{vz:.6f}",
            "trunk_roll_rate": f"{roll_rate:.6f}",
        })

        if writer is not None:
            camera.lookat[:] = data.xpos[harness.trunk_base_id]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"Video saved: {args.record}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"CSV saved: {args.save_csv} ({len(csv_rows)} rows)")

    print(f"\nDone: {n_steps} steps ({duration:.1f}s), falls: {falls}, "
          f"final trunk z: {z:.3f} m")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Drive a dance ONNX policy with a beat timeline (CPU MuJoCo)"
    )
    parser.add_argument("--policy", type=Path, default=None,
                        help="Path to the dance policy ONNX (normalizer baked in)")
    parser.add_argument("--timeline", type=Path, required=True,
                        help="Beat timeline JSON (DuckEMW dance/timeline.py)")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML,
                        help="Scene XML (default: scene_walk.xml, the walk model "
                             "the dance task trains on)")
    parser.add_argument("--mock-policy", action="store_true",
                        help="Skip ONNX inference, output zero actions "
                             "(plumbing smoke test without a trained policy)")
    parser.add_argument("--record", type=Path, default=None,
                        help="Record the run to this mp4 path (offscreen render)")
    parser.add_argument("--save-csv", type=Path, default=None,
                        help="Per-step log: t, phase, move, trunk z/roll/pitch, "
                             "vz, roll rate")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Action scale (training uses 1.0)")
    args = parser.parse_args()
    if not args.mock_policy and args.policy is None:
        print("No --policy given; falling back to --mock-policy", file=sys.stderr)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
