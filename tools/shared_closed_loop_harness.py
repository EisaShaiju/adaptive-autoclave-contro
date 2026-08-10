"""
shared_closed_loop_harness.py
Diagnostic-only harness for closed-loop equivalence: one
loop, two plants, the SAME disturbance-injection point (disturbance-before-
compute) for both controllers, so the two
controllers react to identical information at every step. The two existing
scripts (tests/test_closed_loop.py, tests/test_snn_closed_loop.py) do NOT do
this -- they inject at different points relative to control-compute -- so
neither this harness nor its output is a replacement for those scripts; it is
the only valid source for a step-aligned head-to-head number.

Does NOT modify src/*.py or tests/*.py.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/shared_closed_loop_harness.py
"""
from pathlib import Path
import sys
import json
import hashlib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
import src.constants as const

DIVERGENCE_THRESHOLD_C = 0.5


def fingerprint(*arrays, decimals=8):
    blob = b"".join(np.round(a, decimals).tobytes() for a in arrays)
    return hashlib.sha256(blob).hexdigest()[:16]


def main():
    plant_cvx = AutoclavePlant(initial_temp=28.0)
    plant_snn = AutoclavePlant(initial_temp=28.0)
    ctrl_cvx = MPCSolver(horizon=20, target_temp=120.0)
    ctrl_snn = SNNMPCSolver(horizon=20, target_temp=120.0)

    u_cvx = u_snn = 28.0
    time_steps = 160
    log = []
    first_divergence_k = None

    print("Running shared closed-loop harness (disturbance-before-compute, both branches)...")
    for k in range(time_steps):
        if k == 60:
            for p in (plant_cvx, plant_snn):
                p.T_comp -= 15.0
                p.T_tool -= 15.0

        x_cvx = plant_cvx.get_state()
        x_snn = plant_snn.get_state()

        avg_T_cvx, avg_a_cvx = np.mean(x_cvx[0:3]), np.mean(x_cvx[7:10])
        Ap_cvx, Bp_cvx = linearize(avg_T_cvx, avg_a_cvx, trust_region=False)
        rho_cvx = float(np.max(np.abs(np.linalg.eigvals(Ap_cvx))))
        fp_cvx = fingerprint(Ap_cvx, Bp_cvx, x_cvx, np.array([u_cvx]))

        avg_T_snn, avg_a_snn = np.mean(x_snn[0:3]), np.mean(x_snn[7:10])
        Ap_snn, Bp_snn = linearize(avg_T_snn, avg_a_snn, trust_region=True)
        H_raw, g_raw, C_raw, d_raw = ctrl_snn.build_dense_qp(Ap_snn, Bp_snn, x_snn, u_snn)
        fp_snn = fingerprint(H_raw, g_raw, C_raw, d_raw)

        u_cvx, t_cvx = ctrl_cvx.compute_control_action(x_cvx, u_cvx)
        u_snn, t_snn = ctrl_snn.compute_control_action(x_snn, u_snn)

        plant_cvx.step(u_cvx)
        plant_snn.step(u_snn)

        gap = abs(u_cvx - u_snn)
        if first_divergence_k is None and gap > DIVERGENCE_THRESHOLD_C:
            first_divergence_k = k

        log.append({
            "k": k, "Ta_cvx": float(u_cvx), "Ta_snn": float(u_snn), "Ta_gap": float(gap),
            "Tc1_cvx": float(x_cvx[0]), "Tc3_cvx": float(x_cvx[2]),
            "Tc1_snn": float(x_snn[0]), "Tc3_snn": float(x_snn[2]),
            "alpha1_cvx": float(x_cvx[7]), "alpha3_cvx": float(x_cvx[9]),
            "alpha1_snn": float(x_snn[7]), "alpha3_snn": float(x_snn[9]),
            "rho_Ap_cvx": rho_cvx, "solve_ms_cvx": float(t_cvx), "solve_ms_snn": float(t_snn),
            "qp_fingerprint_cvx": fp_cvx, "qp_fingerprint_snn": fp_snn,
        })

    # ---- Aggregate metrics (same set as README/parity report table) ----
    Tc1_cvx = np.array([r["Tc1_cvx"] for r in log]); Tc3_cvx = np.array([r["Tc3_cvx"] for r in log])
    Tc1_snn = np.array([r["Tc1_snn"] for r in log]); Tc3_snn = np.array([r["Tc3_snn"] for r in log])
    overshoot_cvx = max(0.0, max(Tc1_cvx.max(), Tc3_cvx.max()) - 120.0)
    overshoot_snn = max(0.0, max(Tc1_snn.max(), Tc3_snn.max()) - 120.0)
    max_delta_alpha_cvx = max(abs(np.array([r["alpha1_cvx"] for r in log]) - np.array([r["alpha3_cvx"] for r in log])))
    max_delta_alpha_snn = max(abs(np.array([r["alpha1_snn"] for r in log]) - np.array([r["alpha3_snn"] for r in log])))
    viol_cvx = sum(1 for r in log if abs(r["Tc1_cvx"] - r["Tc3_cvx"]) > const.GRADIENT_MAX + 0.1)
    viol_snn = sum(1 for r in log if abs(r["Tc1_snn"] - r["Tc3_snn"]) > const.GRADIENT_MAX + 0.1)
    avg_solve_cvx = float(np.mean([r["solve_ms_cvx"] for r in log]))
    avg_solve_snn = float(np.mean([r["solve_ms_snn"] for r in log]))
    max_gap = max(r["Ta_gap"] for r in log)

    print("\n" + "=" * 90)
    print("SHARED CLOSED-LOOP HARNESS METRICS (step-aligned, disturbance-before-compute)")
    print("=" * 90)
    print(f"{'Metric':<30}{'CVXPY/OSQP':>18}{'SNN-QP':>18}")
    print(f"{'Max Temp Overshoot (C)':<30}{overshoot_cvx:>18.2f}{overshoot_snn:>18.2f}")
    print(f"{'Max Cure Delta':<30}{max_delta_alpha_cvx:>18.4f}{max_delta_alpha_snn:>18.4f}")
    print(f"{'Constraint Violations':<30}{viol_cvx:>18d}{viol_snn:>18d}")
    print(f"{'Avg Solve Time (ms)':<30}{avg_solve_cvx:>18.2f}{avg_solve_snn:>18.2f}")
    print("-" * 90)
    print(f"Max |Ta_cvxpy - Ta_snn| over the run: {max_gap:.4f} C")
    if first_divergence_k is None:
        print(f"No step exceeded the {DIVERGENCE_THRESHOLD_C} C divergence threshold.")
    else:
        r = log[first_divergence_k]
        print(f"First divergence step (|Ta gap| > {DIVERGENCE_THRESHOLD_C} C): k={first_divergence_k}, "
              f"gap={r['Ta_gap']:.4f} C, rho(Ap_cvx)={r['rho_Ap_cvx']:.4f}")
        if first_divergence_k == 60:
            print("  -> coincides with the disturbance step.")
        if r["rho_Ap_cvx"] > 1.0:
            print("  -> coincides with rho(Ap) > 1 (gelation/exotherm regime); consistent with the")
            print("     window-mismatch finding from tools/qp_parity_probe.py (window error grows")
            print("     with prediction-horizon amplification, i.e. with rho(Ap)).")
    print("=" * 90)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "shared_closed_loop_run.json"
    manifest = {
        "convention": "disturbance-before-compute",
        "note_on_fingerprints": (
            "qp_fingerprint_cvx hashes (Ap,Bp,x0,u_prev) [fully determines the CVXPY "
            "symbolic QP]; qp_fingerprint_snn hashes build_dense_qp's raw (H,g,C,d). "
            "These are each controller's own natural representation, not a cross-"
            "controller canonical form (see the canonical-form contract for "
            "what a full canonical extractor would additionally need)."
        ),
        "horizon": 20, "target_temp": 120.0, "time_steps": time_steps,
        "divergence_threshold_C": DIVERGENCE_THRESHOLD_C,
        "first_divergence_k": first_divergence_k,
        "aggregate_metrics": {
            "overshoot_cvx": overshoot_cvx, "overshoot_snn": overshoot_snn,
            "max_delta_alpha_cvx": max_delta_alpha_cvx, "max_delta_alpha_snn": max_delta_alpha_snn,
            "violations_cvx": viol_cvx, "violations_snn": viol_snn,
            "avg_solve_ms_cvx": avg_solve_cvx, "avg_solve_ms_snn": avg_solve_snn,
            "max_Ta_gap": max_gap,
        },
        "trajectory": log,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nRun manifest + per-step trajectory written to {out_path}")


if __name__ == "__main__":
    main()
