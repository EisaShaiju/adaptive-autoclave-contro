"""
shared_state_ablation.py
Tests whether independent state divergence between the two separately-
simulated closed loops -- not the already-refuted LTV nominal-sequence
memory (see results/final_comparison/*_ltv_constnom_*/summary.json) -- is
what grows the RMS applied-control disagreement under LTV re-linearization
(README_LTV.md, "What is still open," item ii).

Method: ONE shared AutoclavePlant (not two independent ones, unlike
tools/final_controlled_comparison.py's standard harness). At every step, both
controllers call their real compute_control_action() on the IDENTICAL current
state and IDENTICAL previous applied control -- but only the CVXPY branch's
move is ever applied to advance the one shared plant. The SNN's move is
recorded, never applied, same non-interference principle every other probe
script in this repo follows. There is therefore no independent-plant state
divergence possible by construction; if RMS(u0_cvx - u0_snn) still grows under
LTV the way it does in the standard (independent-plants) harness, state
divergence is NOT the mechanism either.

Run under both linearization_mode='lti' and 'ltv' with this SAME shared-state
design, so LTI-vs-LTV is the only variable changing -- comparing against the
existing independent-plants LTI numbers would confound two different harness
designs at once.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/shared_state_ablation.py
"""
from pathlib import Path
import sys
import json
import subprocess
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
import src.constants as const

HORIZON = 10
TARGET_TEMP = 120.0
INITIAL_TEMP = 28.0
TIME_STEPS = 160
K0_SCALE = 0.1


def run_shared_state(linearization_mode):
    """One shared plant, both controllers see identical (state, u_prev) every
    step, only the CVXPY move is ever applied."""
    shared = dict(
        horizon=HORIZON, target_temp=TARGET_TEMP, trust_region=False,
        soft_state_constraints=True, linearization_mode=linearization_mode,
    )
    ctrl_cvx = MPCSolver(**shared)
    ctrl_snn = SNNMPCSolver(k0_scale=K0_SCALE, **shared)
    plant = AutoclavePlant(initial_temp=INITIAL_TEMP)

    u_ref = INITIAL_TEMP
    rows = []
    for k in range(TIME_STEPS):
        x = plant.get_state()
        # Capture Ap BEFORE compute_control_action mutates ctrl_cvx._u_nominal,
        # so rho(Ap) reflects the QP actually solved this step, not next step's.
        qp = ctrl_cvx.build_qp(x, u_ref)
        rho_ap = float(np.max(np.abs(np.linalg.eigvals(qp.linearization["Ap"]))))
        u0_cvx, _ = ctrl_cvx.compute_control_action(x, u_ref)
        u0_snn, _ = ctrl_snn.compute_control_action(x, u_ref)  # computed, never applied
        rows.append({
            "step": k, "u0_cvx": u0_cvx, "u0_snn": u0_snn,
            "diff": u0_snn - u0_cvx, "rho_Ap": rho_ap,
            "Tc1": float(x[0]), "Tc3": float(x[2]), "alpha1": float(x[7]),
        })
        plant.step(Ta_input=u0_cvx)
        u_ref = u0_cvx  # single shared trajectory's true previous control

    return rows


def rms(rows, keys=None):
    diffs = np.array([r["diff"] for r in rows if keys is None or r["step"] in keys])
    return float(np.sqrt(np.mean(diffs ** 2)))


def extract_stiff_window(rows, margin=10):
    peak_k = max(range(len(rows)), key=lambda i: rows[i]["rho_Ap"])
    lo, hi = max(0, peak_k - margin), min(len(rows), peak_k + margin + 1)
    return peak_k, lo, hi, [r["step"] for r in rows[lo:hi]]


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT).decode().strip()
    except Exception as exc:
        commit, dirty = f"unavailable: {exc}", None

    results = {}
    rows_by_mode = {}
    for mode in ("lti", "ltv"):
        print(f"Running shared-state harness, linearization_mode={mode}...")
        rows = run_shared_state(mode)
        rows_by_mode[mode] = rows
        peak_k, lo, hi, stiff_steps = extract_stiff_window(rows)
        overall_rms = rms(rows)
        stiff_rms = rms(rows, keys=set(stiff_steps))
        final_alpha = rows[-1]["alpha1"]
        print(f"  mode={mode}: overall RMS={overall_rms:.4f}  "
              f"stiff-window RMS={stiff_rms:.4f}  (window [{lo},{hi}), peak k={peak_k}, "
              f"rho(Ap)={rows[peak_k]['rho_Ap']:.4f})  final alpha1={final_alpha:.4f}")
        results[mode] = {
            "overall_rms_control_difference": overall_rms,
            "stiff_window_rms_control_difference": stiff_rms,
            "stiff_window": {"lo": lo, "hi": hi, "peak_step": peak_k,
                              "peak_rho_Ap": rows[peak_k]["rho_Ap"]},
            "final_alpha1": final_alpha,
            "cured": bool(final_alpha >= 0.99),
        }

    summary = {
        "method": "ONE shared AutoclavePlant; both controllers compute from the "
                   "identical (state, u_prev) every step; only the CVXPY move is "
                   "ever applied; the SNN move is recorded but never reaches the "
                   "plant. Isolates independent-state-divergence as a candidate "
                   "mechanism for the RMS-disagreement growth under LTV, "
                   "independently of the already-refuted nominal-sequence-memory "
                   "hypothesis (ltv_nominal_source ablation).",
        "config": {"horizon": HORIZON, "k0_scale": K0_SCALE, "soft_state_constraints": True,
                   "trust_region": False, "target_temp": TARGET_TEMP, "time_steps": TIME_STEPS},
        "git_commit": commit,
        "git_working_tree_dirty_files": dirty.splitlines() if dirty else [],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "reading": None,  # filled in below once both modes are in
    }

    lti_stiff = results["lti"]["stiff_window_rms_control_difference"]
    ltv_stiff = results["ltv"]["stiff_window_rms_control_difference"]
    # Compare against the independent-plants harness's own numbers for context
    # (different harness design -- stated explicitly, not silently conflated).
    independent_plants_lti_stiff = 0.5654   # results/final_comparison/f45e2273_clean_lti_*
    independent_plants_ltv_stiff = 2.1187   # results/final_comparison/f45e2273_clean_ltv_*
    moved_toward_lti = abs(ltv_stiff - lti_stiff) < abs(independent_plants_ltv_stiff - independent_plants_lti_stiff) / 2
    summary["reading"] = (
        f"Shared-state stiff-window RMS: LTI={lti_stiff:.4f}, LTV={ltv_stiff:.4f} "
        f"(independent-plants harness for context, NOT a like-for-like baseline -- "
        f"different harness design: LTI={independent_plants_lti_stiff:.4f}, "
        f"LTV={independent_plants_ltv_stiff:.4f}). "
        + ("State-divergence amplification is CONFIRMED as (part of) the mechanism: "
           "removing it by construction pulls LTV's RMS back toward the LTI level."
           if moved_toward_lti else
           "State-divergence amplification is REFUTED as the primary mechanism: "
           "removing it by construction did not pull LTV's RMS back toward the LTI level; "
           "the disagreement persists even when both controllers see identical inputs "
           "every step, so the source is intrinsic to how the two solvers handle the "
           "same LTV-conditioned QP, not accumulated state drift between two independent "
           "closed loops.")
    )
    print("\n" + summary["reading"])

    out_path = PROJECT_ROOT / "results" / "shared_state_ablation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Plot: applied Ta (reference/CVXPY) and the recorded-but-discarded SNN move,
    # for both modes, same visual convention as final_controlled_comparison.py.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, mode in zip(axes, ("lti", "ltv")):
        rows = rows_by_mode[mode]
        steps = [r["step"] for r in rows]
        ax.plot(steps, [r["diff"] for r in rows], 'b-', label=f"e_u(k), {mode}")
        w = results[mode]["stiff_window"]
        ax.axvspan(w["lo"], w["hi"], color='orange', alpha=0.2, label="stiff window")
        ax.set_title(f"Shared-state harness, {mode.upper()}: "
                      f"u0_snn - u0_cvx (computed, SNN move never applied)")
        ax.set_ylabel("degC")
        ax.legend(loc="upper right")
        ax.grid(True)
    axes[1].set_xlabel("step (minutes)")
    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "shared_state_ablation_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
