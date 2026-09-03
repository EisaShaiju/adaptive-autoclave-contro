"""
ltv_feasibility_probe.py
Bounded feasibility check requested by the research advisor: does an ACCURATE
(true non-linear) rollout, in place of the frozen-Jacobian LTI prediction,
shrink the predicted-infeasible region at stiff exotherm states?

This is explicitly NOT the full LTV (re-linearise-along-the-horizon) rewrite.
It answers only whether that rewrite is worth doing, by comparing -- at a
handful of stiff-window states, driven by the SAME realised control sequence
-- what the frozen-Jacobian prediction says about each gradient-constraint
row against what the true non-linear plant actually does. See
docs/PHASE4_VALIDATION_REPORT.md section 19 for the result and the advisor's
proposed mechanism (the frozen model's over-prediction of the exotherm is
what empties the feasible set at stiff steps, distinct from the separately
diagnosed convergence defect).

Method
------
1. Replay one fixed CVXPY-driven trajectory (N=10, soft, trust_region=False --
   the recommended configuration) and record every (state, applied_control)
   pair, exactly as tools/constraint_set_experiment.py does.
2. Select the stiff-window (STIFF_LO..STIFF_HI) states with the highest
   rho(Ap), always including ANCHOR_STEP=88, the state already cited in
   docs/PHASE4_VALIDATION_REPORT.md section 14.
3. At each selected state k, holding the horizon and the REALISED control
   sequence us[k:k+N] fixed:
     - linear side: propagate the frozen (Ap, Bp) from src/dynamics.linearize
       exactly as src/qp_builder.py's Phi/Gamma do (row h = state h steps
       ahead, row 0 = x0 itself), reading off (Tc1 - Tc3) at each row.
     - nonlinear side: deep-copy AutoclavePlant, seed it from state k, step
       it forward with the same control sequence, record actual (Tc1 - Tc3)
       at each row.
   Classify each row against GRADIENT_MAX (10 degC, two-sided):
     genuine  : both linear and nonlinear predict a breach (LTV would not help)
     artifact : linear predicts a breach the nonlinear rollout does not (LTV
                would remove this row from the constraint set)
     missed   : nonlinear breaches but linear does not (flag; should be rare)
     clear    : neither breaches
4. Aggregate and report what fraction of currently-flagged rows are
   artifacts of the frozen prediction versus genuine plant violations.

Usage
-----
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/ltv_feasibility_probe.py
"""
import os
import sys
import json

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.constants as const                       # noqa: E402
from src.plant_simulator import AutoclavePlant       # noqa: E402
from src.dynamics import linearize                   # noqa: E402
from src.mpc_cvxpy_controller import MPCSolver       # noqa: E402

TARGET_TEMP = 120.0
TIME_STEPS = 160
HORIZON = 10
STIFF_LO, STIFF_HI = 77, 108        # matches the constraint-set/budget probes
NUM_STATES = 6
ANCHOR_STEP = 88                    # already cited in the validation report


def record_trajectory(horizon=HORIZON):
    """One fixed CVXPY-driven trajectory, matching the recommended
    configuration (N=10, soft, trust_region=False)."""
    plant = AutoclavePlant()
    ctrl = MPCSolver(horizon=horizon, target_temp=TARGET_TEMP,
                      trust_region=False, soft_state_constraints=True)
    u_prev = const.TA_MIN
    states, us = [], []
    for _ in range(TIME_STEPS):
        x = plant.get_state()
        states.append(x.copy())
        u, _ = ctrl.compute_control_action(x, u_prev)
        us.append(u)
        plant.step(u)
        u_prev = u
    return states, us


def select_stiff_states(states):
    scored = []
    for k in range(STIFF_LO, STIFF_HI):
        x = states[k]
        avg_T, avg_a = float(np.mean(x[0:3])), float(np.mean(x[7:10]))
        Ap, _ = linearize(avg_T, avg_a, trust_region=False)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))
        scored.append((k, rho))
    scored.sort(key=lambda t: -t[1])
    chosen = {k for k, _ in scored[:NUM_STATES]}
    chosen.add(ANCHOR_STEP)
    return sorted(chosen)


def linear_prediction(x0, u_seq, horizon):
    """Predicted (Tc1 - Tc3) at each horizon row h=0..horizon-1 under the
    frozen-Jacobian LTI model, driven by the REALISED control sequence
    u_seq rather than a zero-input free response -- the same quantity the
    gradient-constraint rows bound in build_canonical_qp, evaluated at the
    trajectory's own controls instead of the solver's decision variable."""
    avg_T, avg_a = float(np.mean(x0[0:3])), float(np.mean(x0[7:10]))
    Ap, Bp = linearize(avg_T, avg_a, trust_region=False)
    x = x0.copy()
    preds = []
    for h in range(horizon):
        preds.append(float(x[0] - x[2]))
        x = Ap @ x + Bp * u_seq[h]
    return preds


def nonlinear_rollout(x0, u_seq, horizon):
    """True plant response to the SAME control sequence, on a fresh plant
    instance seeded from the recorded state -- the live trajectory used to
    select states is never mutated."""
    plant = AutoclavePlant()
    plant.T_comp = x0[0:3].copy()
    plant.T_tool = x0[3:7].copy()
    plant.alpha = x0[7:10].copy()
    actual = [float(x0[0] - x0[2])]
    for h in range(horizon - 1):
        plant.step(u_seq[h])
        s = plant.get_state()
        actual.append(float(s[0] - s[2]))
    return actual


def classify(linear_val, actual_val, limit):
    lin_flag = abs(linear_val) > limit
    act_flag = abs(actual_val) > limit
    if lin_flag and act_flag:
        return "genuine"
    if lin_flag and not act_flag:
        return "artifact"
    if not lin_flag and act_flag:
        return "missed"
    return "clear"


def main():
    print("Recording reference trajectory (CVXPY, N=10, soft, trust_region=False)...")
    states, us = record_trajectory(HORIZON)

    stiff_states = select_stiff_states(states)
    print(f"Selected stiff states: {stiff_states}")

    per_state = []
    counts = {"genuine": 0, "artifact": 0, "missed": 0, "clear": 0}
    for k in stiff_states:
        x0 = states[k]
        u_seq = [us[min(k + h, TIME_STEPS - 1)] for h in range(HORIZON)]

        linear_vals = linear_prediction(x0, u_seq, HORIZON)
        actual_vals = nonlinear_rollout(x0, u_seq, HORIZON)

        rows = []
        for h in range(HORIZON):
            label = classify(linear_vals[h], actual_vals[h], const.GRADIENT_MAX)
            counts[label] += 1
            rows.append({
                "h": h, "linear_grad_degC": round(linear_vals[h], 4),
                "actual_grad_degC": round(actual_vals[h], 4),
                "classification": label,
            })
        per_state.append({"step": k, "rows": rows})
        flagged = [r for r in rows if r["classification"] in ("genuine", "artifact")]
        n_genuine = sum(1 for r in flagged if r["classification"] == "genuine")
        n_artifact = sum(1 for r in flagged if r["classification"] == "artifact")
        print(f"  step {k:3d}: {len(flagged)} row(s) flagged by the linear model "
              f"({n_genuine} genuine, {n_artifact} artifact)")

    total_flagged = counts["genuine"] + counts["artifact"]
    artifact_frac = (counts["artifact"] / total_flagged) if total_flagged else float("nan")

    out = {
        "method": (
            "Advisor-directed quick check (not the full LTV rewrite): at each "
            "selected stiff-window state, the frozen-Jacobian LTI prediction "
            "and a true non-linear plant rollout are driven by the IDENTICAL "
            "realised control sequence from one fixed CVXPY trajectory "
            "(N=10, soft, trust_region=False), and each horizon row's "
            "predicted (Tc1-Tc3) is compared against the actual value at the "
            "GRADIENT_MAX=10 degC two-sided limit. This tests whether the "
            "frozen model's over-prediction of the exotherm is what empties "
            "the feasible set at stiff steps."
        ),
        "config": {"horizon": HORIZON, "stiff_window": [STIFF_LO, STIFF_HI],
                   "num_states": len(stiff_states), "anchor_step": ANCHOR_STEP,
                   "gradient_max_degC": const.GRADIENT_MAX},
        "states": per_state,
        "row_counts": counts,
        "artifact_fraction_of_flagged": artifact_frac,
    }

    out_path = os.path.join(PROJECT_ROOT, "results", "ltv_feasibility_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nRow classification across {len(stiff_states)} stiff states x {HORIZON} rows:")
    print(f"  genuine (LTV would NOT help):  {counts['genuine']}")
    print(f"  artifact (LTV WOULD help):     {counts['artifact']}")
    print(f"  missed (linear under-warns):   {counts['missed']}")
    print(f"  clear:                         {counts['clear']}")
    if total_flagged:
        print(f"\n{artifact_frac * 100:.1f}% of currently-flagged rows are artifacts of the "
              f"frozen-Jacobian prediction, not genuine plant violations.")
    else:
        print("\nNo rows were flagged by the linear model at these states.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
