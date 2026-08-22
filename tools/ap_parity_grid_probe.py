"""
ap_parity_grid_probe.py
Persists the max|delta Ap| (trust_region=True vs False) figure behind the
model-parity invariant and quoted in docs/SNN_MPC_TECHNICAL_REPORT.md -- previously
console-only prose ("~1e-4 at a benign operating point; ~1851 at a
gelation-like point"), with no file behind it.

Two complementary views, because they answer different questions:

  1. GRID sweep over (T, alpha) operating points. This is what the "benign vs
     gelation-like point" prose refers to: delta Ap is a property of WHERE the
     Jacobian is evaluated, not of any particular trajectory. The asymmetric
     clamp (trust_region=True) bounds the exotherm Jacobian terms; the gap
     between clamped and unclamped grows sharply with temperature at mid-cure.

  2. TRAJECTORY sweep along a live CVXPY closed-loop rollout, reported by
     tools/qp_parity_probe.py -> results/qp_parity_diagnostics.json. This is
     strictly SMALLER (max ~76) because the controller brakes the exotherm and
     never visits the most extreme (T, alpha) region the grid reaches. Both
     numbers are correct; they are not interchangeable, and a doc quoting the
     grid figure must not cite the trajectory artifact as its source.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/ap_parity_grid_probe.py
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics import linearize

# Benign point quoted in the technical report (early heat-up, negligible cure).
BENIGN = (40.0, 0.01)
# Grid spans the physically reachable cure envelope: room temp to well past the
# exotherm peak, and the full cure range where the Arrhenius term is active.
T_GRID = np.arange(20.0, 221.0, 5.0)
ALPHA_GRID = np.arange(0.01, 1.00, 0.01)


def delta_ap(T, alpha):
    Ap_f, _ = linearize(T, alpha, trust_region=False)
    Ap_t, _ = linearize(T, alpha, trust_region=True)
    return float(np.max(np.abs(Ap_f - Ap_t)))


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    benign = delta_ap(*BENIGN)
    print(f"benign point T={BENIGN[0]} C, alpha={BENIGN[1]}: "
          f"max|dAp| = {benign:.6e}")

    grid = np.zeros((len(T_GRID), len(ALPHA_GRID)))
    for i, T in enumerate(T_GRID):
        for j, a in enumerate(ALPHA_GRID):
            grid[i, j] = delta_ap(float(T), float(a))

    peak_idx = np.unravel_index(int(np.argmax(grid)), grid.shape)
    peak_T = float(T_GRID[peak_idx[0]])
    peak_alpha = float(ALPHA_GRID[peak_idx[1]])
    peak_val = float(grid[peak_idx])
    print(f"grid peak: max|dAp| = {peak_val:.4f} at T={peak_T:.1f} C, "
          f"alpha={peak_alpha:.2f}")

    # Where does the grid cross the ~1851 figure quoted in the docs?
    target = 1851.0
    near = np.argwhere(np.abs(grid - target) < 0.05 * target)
    near_pts = [{"T_degC": float(T_GRID[i]), "alpha": float(ALPHA_GRID[j]),
                 "max_abs_delta_Ap": float(grid[i, j])} for i, j in near[:12]]
    if near_pts:
        print(f"points within 5% of the quoted {target:.0f}: {len(near)} "
              f"(showing up to 12)")
        for p in near_pts[:6]:
            print(f"   T={p['T_degC']:.1f} alpha={p['alpha']:.2f} -> "
                  f"{p['max_abs_delta_Ap']:.2f}")
    else:
        print(f"NO grid point within 5% of the quoted {target:.0f} "
              f"-- the docs figure is NOT reproducible on this grid.")

    out = {
        "git_commit": commit,
        "quantity": "max|Ap(trust_region=False) - Ap(trust_region=True)|",
        "why": ("Persists the A-matrix difference behind the model-parity invariant "
                "and quoted in docs/SNN_MPC_TECHNICAL_REPORT.md, which previously existed "
                "only as prose. delta Ap is a property of the (T, alpha) point "
                "where the Jacobian is evaluated -- NOT of a trajectory."),
        "benign_point": {"T_degC": BENIGN[0], "alpha": BENIGN[1],
                         "max_abs_delta_Ap": benign},
        "grid": {
            "T_degC_range": [float(T_GRID[0]), float(T_GRID[-1])],
            "alpha_range": [float(ALPHA_GRID[0]), float(ALPHA_GRID[-1])],
            "peak": {"T_degC": peak_T, "alpha": peak_alpha,
                     "max_abs_delta_Ap": peak_val},
            "points_near_quoted_1851": near_pts,
        },
        "trajectory_comparison": {
            "source": "results/qp_parity_diagnostics.json",
            "note": ("A live CVXPY closed-loop rollout reaches only ~76 because "
                     "the controller brakes the exotherm and never visits the "
                     "grid's most extreme (T, alpha) region. Not a contradiction "
                     "with the grid peak; a different question."),
        },
    }
    dest = PROJECT_ROOT / "results" / "ap_parity_grid.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
