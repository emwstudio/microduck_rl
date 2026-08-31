"""Timeline → dance command mapping tests (CPU, no GPU, no ONNX).

Covers the pure mapping functions in scripts/dance_to_timeline.py and their
equivalence with the training-side DanceCommand semantics in mdp.py, plus the
extrema/alignment analysis in scripts/check_beat_align.py.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "click120.timeline.json"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


d2t = _load_script("dance_to_timeline")
cba = _load_script("check_beat_align")


# --------------------------------------------------------------------------- #
# Timeline loading / validation                                                 #
# --------------------------------------------------------------------------- #


def test_load_timeline_accepts_fixture():
    tl = d2t.load_timeline(FIXTURE)
    assert tl["bpm"] == pytest.approx(119.96)  # re-derived from beat_times, see
    # test_fixture_bpm_is_consistent_with_its_beat_times
    assert len(tl["beat_times"]) == 62
    assert len(tl["segments"]) == 8
    # segments sorted by t_start
    starts = [s["t_start"] for s in tl["segments"]]
    assert starts == sorted(starts)


def test_load_timeline_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bpm": 120.0}))
    with pytest.raises(ValueError, match="missing required key"):
        d2t.load_timeline(bad)


def test_load_timeline_rejects_untrained_moves(tmp_path):
    # The trained policy knows moves 0-4; anything beyond is rejected.
    bad = tmp_path / "bad_move.json"
    bad.write_text(json.dumps({
        "bpm": 120.0, "t0": 0.0, "duration": 10.0,
        "beat_times": [0.0, 0.5, 1.0],
        "segments": [{"move": 5, "move_name": "unknown",
                      "start_beat": 0, "end_beat": 1, "t_start": 0.0, "t_end": 1.0}],
    }))
    with pytest.raises(ValueError, match="move 5"):
        d2t.load_timeline(bad)


# --------------------------------------------------------------------------- #
# Phase / segment mapping                                                       #
# --------------------------------------------------------------------------- #


def test_beat_phase_is_zero_on_every_beat():
    # Exact synthetic grid: beat k at t0 + k·60/bpm → φ ≡ 0 mod 2π.
    bpm, t0 = 120.0, 0.25
    for k in range(20):
        phi = d2t.beat_phase(t0 + k * 60.0 / bpm, t0, bpm)
        assert math.sin(phi) == pytest.approx(0.0, abs=1e-6)
        assert math.cos(phi) == pytest.approx(1.0, abs=1e-6)
    # half a beat after t0 → φ = π
    assert d2t.beat_phase(t0 + 0.5 * 60.0 / bpm, t0, bpm) == pytest.approx(math.pi, abs=1e-9)
    # phase always in [0, 2π), including before t0
    for t in np.linspace(-1.0, 30.0, 200):
        phi = d2t.beat_phase(float(t), t0, bpm)
        assert 0.0 <= phi < 2.0 * math.pi


def test_fixture_bpm_is_consistent_with_its_beat_times():
    # DATA-QUALITY GUARD: the harness anchors phase to (t0, bpm), so a timeline
    # whose bpm disagrees with its own beat_times drifts off the real beats.
    # NOTE: the click120 fixture's bpm was RE-DERIVED from its beat_times
    # (119.96); the original librosa estimate in dance/beats.py output was
    # 117.45, which drifts ~0.6 s over the 32 s song. If you regenerate this
    # fixture from DuckEMW, fix the tempo estimate in dance/beats.py first.
    tl = d2t.load_timeline(FIXTURE)
    bt = tl["beat_times"]
    effective_bpm = 60.0 * (len(bt) - 1) / (bt[-1] - bt[0])
    assert abs(effective_bpm - tl["bpm"]) < 1.0, (
        f"timeline bpm={tl['bpm']} but beat_times imply {effective_bpm:.2f} — "
        "phase will drift off the actual beats over the song"
    )


def test_segment_at_boundaries_and_overlap():
    segs = [
        {"move": 0, "t_start": 0.5, "t_end": 5.0},
        {"move": 1, "t_start": 4.5, "t_end": 9.0},   # overlaps seg 0
        {"move": 2, "t_start": 8.5, "t_end": 12.0},
    ]
    # before the first segment: first move (dance starts anyway)
    assert d2t.segment_at(segs, 0.0)["move"] == 0
    # inside overlap: the LATER segment wins
    assert d2t.segment_at(segs, 4.6)["move"] == 1
    # exact boundary
    assert d2t.segment_at(segs, 4.5)["move"] == 1
    assert d2t.segment_at(segs, 8.4)["move"] == 1
    assert d2t.segment_at(segs, 8.5)["move"] == 2
    # past the end: keeps the last segment
    assert d2t.segment_at(segs, 999.0)["move"] == 2


def test_fixture_segment_sequence_matches_generation_order():
    # click120 was generated with --moves 0,1,2 rotating every 8 beats
    tl = d2t.load_timeline(FIXTURE)
    moves = [s["move"] for s in tl["segments"]]
    assert moves == [0, 1, 2, 0, 1, 2, 0, 1]


# --------------------------------------------------------------------------- #
# Equivalence with the training-side DanceCommand (mdp.py)                      #
# --------------------------------------------------------------------------- #


class _FakeEnv:
    num_envs = 1
    device = "cpu"


def test_dance_command_matches_training_semantics():
    """The harness mapping must equal DanceCommand._write_command exactly."""
    from mjlab_microduck.tasks import mdp as microduck_mdp

    tl = d2t.load_timeline(FIXTURE)
    term = microduck_mdp.DanceCommand(microduck_mdp.DanceCommandCfg(), _FakeEnv())

    for t in np.linspace(0.0, tl["duration"], 50):
        move = d2t.segment_at(tl["segments"], float(t))["move"]
        beats = (float(t) - tl["t0"]) * tl["bpm"] / 60.0
        term._phase_beats[:] = beats
        term._bpm[:] = tl["bpm"]
        term._move_id[:] = move
        term._write_command()
        training_cmd = term.command[0].numpy()

        harness_cmd = d2t.dance_command(float(t), tl)
        np.testing.assert_allclose(
            harness_cmd, training_cmd, atol=5e-4,  # float32: at large unwrapped
            # phases (t=30 s → φ≈170 rad) torch's float32 sin and
            # math.sin(double)→float32 differ by ~1e-5; semantic equivalence
            # (slot order, one-hot, tempo scaling) differs by ≫1e-3.
            err_msg=f"harness/training command mismatch at t={t:.3f}",
        )


def test_dance_command_layout_and_ranges():
    tl = d2t.load_timeline(FIXTURE)
    for t in np.linspace(0.0, tl["duration"], 100):
        cmd = d2t.dance_command(float(t), tl)
        assert cmd.shape == (6,)
        # sin/cos on the unit circle
        assert cmd[0] ** 2 + cmd[1] ** 2 == pytest.approx(1.0, abs=1e-5)
        # tempo_norm = bpm/120
        assert cmd[2] == pytest.approx(tl["bpm"] / 120.0, abs=1e-6)
        # 3-bit move id in slots 3-5
        assert set(np.round(cmd[3:6], 6).tolist()) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Beat-alignment analysis                                                       #
# --------------------------------------------------------------------------- #


def _write_csv(path, times, trunk_z, trunk_roll):
    import csv as _csv

    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["t", "trunk_z", "trunk_roll"])
        w.writeheader()
        for ti, zi, ri in zip(times, trunk_z, trunk_roll):
            w.writerow({"t": f"{ti:.4f}", "trunk_z": f"{zi:.6f}", "trunk_roll": f"{ri:.6f}"})


def _synthetic_timeline(tmp_path, bpm=120.0, n_beats=17):
    beat_times = [0.5 + k * 60.0 / bpm for k in range(n_beats)]
    tl = {
        "name": "synthetic", "bpm": bpm, "t0": beat_times[0],
        "duration": beat_times[-1] + 0.3, "beat_times": beat_times,
        "segments": [
            {"move": 0, "move_name": "squat_bounce", "start_beat": 0,
             "end_beat": 8, "t_start": beat_times[0], "t_end": beat_times[9]},
            {"move": 1, "move_name": "weight_shift", "start_beat": 8,
             "end_beat": 16, "t_start": beat_times[8], "t_end": beat_times[-1] + 0.3},
        ],
    }
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(tl))
    return path, beat_times


def _synth_signals(times, beat_times, z_lag_s=0.0, roll_lag_s=0.0):
    """Perfect dancer: z minima on beats (+lag), roll extrema on odd beats (+lag)."""
    t0 = beat_times[0]
    bpm = 120.0
    z = -0.0125 * (1.0 + np.cos(2 * np.pi * (times - t0 - z_lag_s) * bpm / 60.0))
    roll = 0.14 * np.sin(np.pi * (times - t0 - roll_lag_s) * bpm / 60.0)
    return z, roll


def test_check_beat_align_perfect_dancer(tmp_path):
    tl_path, beat_times = _synthetic_timeline(tmp_path)
    times = np.arange(0.0, beat_times[-1] + 0.3, 0.02)
    z, roll = _synth_signals(times, beat_times)
    csv_path = tmp_path / "log.csv"
    _write_csv(csv_path, times, z, roll)

    results = cba.analyze(csv_path, tl_path)
    assert len(results) == 2
    for r in results:
        assert r["n_extrema"] > 0
        assert r["median_abs_offset_ms"] < 20.0  # sampling grid is 20 ms
        assert r["frac_within_100ms"] == pytest.approx(1.0)


def test_check_beat_align_detects_lag(tmp_path):
    tl_path, beat_times = _synthetic_timeline(tmp_path)
    times = np.arange(0.0, beat_times[-1] + 0.3, 0.02)
    # 150 ms behind the beat everywhere → must NOT read as aligned
    z, roll = _synth_signals(times, beat_times, z_lag_s=0.15, roll_lag_s=0.15)
    csv_path = tmp_path / "log.csv"
    _write_csv(csv_path, times, z, roll)

    results = cba.analyze(csv_path, tl_path)
    for r in results:
        assert r["median_abs_offset_ms"] == pytest.approx(150.0, abs=25.0)
        assert r["frac_within_100ms"] == pytest.approx(0.0)


def test_check_beat_align_flat_signal_reports_no_extrema(tmp_path):
    tl_path, beat_times = _synthetic_timeline(tmp_path)
    times = np.arange(0.0, beat_times[-1] + 0.3, 0.02)
    z = np.full_like(times, 0.115)  # robot stands still — not dancing
    roll = np.zeros_like(times)
    csv_path = tmp_path / "log.csv"
    _write_csv(csv_path, times, z, roll)

    results = cba.analyze(csv_path, tl_path)
    assert all(r["n_extrema"] == 0 for r in results)
    assert all(r["median_abs_offset_ms"] is None for r in results)
