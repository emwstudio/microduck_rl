#!/usr/bin/env python3
"""Beat-alignment analysis for dance_to_timeline.py CSV logs.

Pure numpy + csv + json (no librosa). For each timeline segment, finds the
local extrema of the move's tracking channel and measures their distance to
the nearest beat time:

  squat_bounce (move 0): MINIMA of trunk_z vs the BEAT times (the dip's lowest
      point lands on the beat — the reference is dz = -A(1+cos φ)/2).
  weight_shift (move 1): extrema (max AND min) of trunk_roll vs the HALF-BEAT
      midpoints between consecutive beats. The training reference is
      roll = B·sin(φ/2) with φ advancing 2π per beat, so the roll crosses zero
      ON each beat and reaches +B/-B exactly halfway between beats (each side
      occupies one beat-long half-wave). Comparing to beat_times would report
      a constant ~half-period false "lag".
  head_bob (move 2): skipped — it oscillates in head_pitch, not the trunk
      channels recorded in the CSV.

Usage:
    uv run python scripts/check_beat_align.py out.csv \
        --timeline tests/fixtures/click120.timeline.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# Extrema closer than this fraction of the beat period are merged (the
# stronger one is kept) — debounces derivative-sign flicker near a plateau.
MIN_SEPARATION_BEATS = 0.4
# Skip this much time at each segment edge (move transitions are transients).
EDGE_MARGIN_S = 0.3
# An extremum must stick out from the segment mean by at least this fraction
# of the segment's signal std to count (filters flat-line noise when the robot
# isn't dancing at all).
PROMINENCE_STD_FRAC = 0.3


def find_extrema(times, values, kind, min_separation):
    """Local extrema of `values` sampled at `times`.

    kind: "min" | "max" | "both". min_separation: seconds; closer extrema of
    the same kind are merged keeping the stronger (lower min / higher max).
    Returns the extrema times as a sorted np.ndarray.
    """
    s = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    if len(s) < 3:
        return np.empty(0)
    d = np.diff(s)
    idx_min, idx_max = [], []
    for i in range(len(d) - 1):
        if d[i] < 0 <= d[i + 1]:
            idx_min.append(i + 1)
        elif d[i] > 0 >= d[i + 1]:
            idx_max.append(i + 1)

    def merge(indices, key):
        if not indices:
            return []
        kept = [indices[0]]
        for idx in indices[1:]:
            if t[idx] - t[kept[-1]] < min_separation:
                if key(idx) < key(kept[-1]):  # key: smaller = stronger
                    kept[-1] = idx
            else:
                kept.append(idx)
        return kept

    out = []
    if kind in ("min", "both"):
        out += [t[i] for i in merge(idx_min, lambda i: s[i])]
    if kind in ("max", "both"):
        out += [t[i] for i in merge(idx_max, lambda i: -s[i])]
    return np.array(sorted(out))


def nearest_beat_offset(t_ext, beat_times):
    """Signed offset (s) from t_ext to the nearest beat time."""
    diffs = np.asarray(beat_times) - t_ext
    i = int(np.argmin(np.abs(diffs)))
    return float(diffs[i])


def analyze(csv_path, timeline_path, sync_threshold_s=0.1):
    """Per-segment beat-alignment stats. Returns a list of dicts (also printed
    by main). Segments whose channel is flat (std ~ 0, robot not dancing)
    report n_extrema=0 rather than garbage timings."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")
    t = np.array([float(r["t"]) for r in rows])
    trunk_z = np.array([float(r["trunk_z"]) for r in rows])
    trunk_roll = np.array([float(r["trunk_roll"]) for r in rows])

    timeline = json.loads(Path(timeline_path).read_text())
    beat_times = timeline["beat_times"]
    beat_period = 60.0 / timeline["bpm"]
    min_sep = MIN_SEPARATION_BEATS * beat_period

    results = []
    # weight_shift roll extremes land halfway between beats (see docstring).
    half_beat_times = [
        (a + b) / 2.0 for a, b in zip(beat_times[:-1], beat_times[1:])
    ]
    for seg_idx, seg in enumerate(timeline["segments"]):
        move = seg["move"]
        if move == 0:
            signal, kind, channel = trunk_z, "min", "trunk_z minima"
            ref_times = beat_times
        elif move == 1:
            signal, kind, channel = trunk_roll, "both", "trunk_roll extrema"
            ref_times = half_beat_times
        else:
            continue  # head_bob: no trunk-channel expectation

        mask = (
            (t >= seg["t_start"] + EDGE_MARGIN_S)
            & (t <= seg["t_end"] - EDGE_MARGIN_S)
        )
        if mask.sum() < 3:
            results.append({
                "segment": seg_idx, "move": move,
                "move_name": seg.get("move_name", str(move)),
                "channel": channel, "n_extrema": 0,
                "median_abs_offset_ms": None, "frac_within_100ms": None,
            })
            continue

        ts, ss = t[mask], signal[mask]
        # Flat signal → not dancing → no meaningful extrema.
        if np.std(ss) < 1e-4:
            extrema_times = np.empty(0)
        else:
            extrema_times = find_extrema(ts, ss, kind, min_sep)
            # Prominence filter: extremum must deviate from the segment mean.
            thresh = PROMINENCE_STD_FRAC * np.std(ss)
            mean = np.mean(ss)
            keep = [
                te for te in extrema_times
                if abs(float(np.interp(te, ts, ss)) - mean) > thresh
            ]
            extrema_times = np.array(keep)

        offsets = np.array([
            abs(nearest_beat_offset(te, ref_times)) for te in extrema_times
        ])
        results.append({
            "segment": seg_idx,
            "move": move,
            "move_name": seg.get("move_name", str(move)),
            "channel": channel,
            "n_extrema": len(offsets),
            "median_abs_offset_ms": (
                float(np.median(offsets) * 1000.0) if len(offsets) else None
            ),
            "frac_within_100ms": (
                float(np.mean(offsets < sync_threshold_s)) if len(offsets) else None
            ),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", type=Path, help="CSV written by dance_to_timeline.py --save-csv")
    parser.add_argument("--timeline", type=Path, required=True, help="timeline JSON")
    parser.add_argument("--threshold-ms", type=float, default=100.0,
                        help="sync threshold for the on-beat fraction (default 100)")
    args = parser.parse_args()

    results = analyze(args.csv, args.timeline, args.threshold_ms / 1000.0)

    print(f"{'seg':>3}  {'move':<13} {'channel':<20} {'n':>3}  "
          f"{'median |dt|':>12}  {'<100ms':>7}")
    for r in results:
        med = f"{r['median_abs_offset_ms']:8.1f} ms" if r["median_abs_offset_ms"] is not None else "       -"
        frac = f"{r['frac_within_100ms']:6.0%}" if r["frac_within_100ms"] is not None else "     -"
        print(f"{r['segment']:>3}  {r['move_name']:<13} {r['channel']:<20} "
              f"{r['n_extrema']:>3}  {med}  {frac}")

    measured = [r for r in results if r["n_extrema"] > 0]
    if measured:
        overall_med = float(np.median([r["median_abs_offset_ms"] for r in measured]))
        overall_frac = float(np.mean([r["frac_within_100ms"] for r in measured]))
        print(f"\nOverall: median |dt| = {overall_med:.1f} ms, "
              f"fraction within 100 ms = {overall_frac:.0%} "
              f"({len(measured)} measured segments)")
    else:
        print("\nNo extrema detected in any segment — is the robot dancing?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
