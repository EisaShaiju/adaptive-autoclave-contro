"""
exact_penalty_audit.py
Is the soft-constrained QP returning the HARD solution where the hard problem
is feasible? That is not a matter of taste -- it is the exactness condition of
an l1 penalty, and it is measurable.

Kerrigan & Maciejowski (Control 2000): with a linear (l1) penalty of weight rho
on the constraint violation, the soft optimum equals the hard optimum provided

    rho > || lambda* ||_inf

where lambda* are the Lagrange multipliers of the HARD problem on the softened
rows. Below that threshold the penalty is not exact and the solver will trade
away constraint satisfaction that was achievable. A QUADRATIC penalty is never
exact at any finite weight.

This repository uses slack_weight_lin = 1e3 (l1) and slack_weight_quad = 1e2
(l2). Neither was derived. This script measures, at every recorded state:

  * whether the hard problem is feasible at all (slack LP certificate);
  * if feasible, ||lambda*||_inf on the gradient rows;
  * whether slack_weight_lin exceeds it;
  * the realised gap between the soft and hard solutions (u0 and full vector);
  * the slack actually used.

The l2 term is reported separately because it breaks exactness by construction:
any non-zero multiplier produces a non-zero slack under a quadratic penalty, no
matter how large the weight. Its presence is a deliberate conditioning choice
and its cost is quantified here rather than assumed negligible.

Reference: E. C. Kerrigan and J. M. Maciejowski, "Soft Constraints and Exact Penalty
Functions in Model Predictive Control", UKACC International Conference (Control), 2000.
See also docs/PHASE4_VALIDATION_REPORT.md section 16.2.
"""
import os
import sys
import json
import warnings

import numpy as np
import cvxpy as cp

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.constants as const                                   # noqa: E402
from src.plant_simulator import AutoclavePlant                  # noqa: E402
from src.dynamics import linearize                              # noqa: E402
from src.qp_builder import build_canonical_qp                   # noqa: E402
from src.mpc_cvxpy_controller import MPCSolver                  # noqa: E402

TARGET_TEMP = 120.0
TIME_STEPS = 160
STIFF_LO, STIFF_HI = 77, 108
SLACK_WEIGHT_LIN = 1.0e3
SLACK_WEIGHT_QUAD = 1.0e2


def record_states(horizon=10):
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


def solve_hard_with_duals(qp, n_actuator_rows):
    """Solve the hard QP and return (z*, ||lambda*||_inf on gradient rows).

    ECOS/Clarabel give reliable duals; OSQP's are noisier at this conditioning.
    Returns (None, None, status) when the hard set is empty.
    """
    z = cp.Variable(qp.H.shape[0])
    con = qp.A_ineq @ z <= qp.b_ineq
    pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(qp.H)) + qp.f @ z),
                    [con])
    for solver in (cp.CLARABEL, cp.ECOS, cp.OSQP):
        try:
            pr.solve(solver=solver)
            if z.value is not None and pr.status in ("optimal", "optimal_inaccurate"):
                lam = np.asarray(con.dual_value).ravel()
                lam_grad = lam[n_actuator_rows:]
                return (np.asarray(z.value),
                        float(np.max(np.abs(lam_grad))) if lam_grad.size else 0.0,
                        pr.status)
        except Exception:
            continue
    return None, None, pr.status


def solve_soft(qp):
    z = cp.Variable(qp.H.shape[0])
    pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(qp.H)) + qp.f @ z),
                    [qp.A_ineq @ z <= qp.b_ineq])
    for solver in (cp.CLARABEL, cp.ECOS, cp.OSQP):
        try:
            pr.solve(solver=solver)
            if z.value is not None:
                return np.asarray(z.value), pr.status
        except Exception:
            continue
    return None, pr.status


def hard_is_feasible(qp):
    z = cp.Variable(qp.H.shape[0])
    s = cp.Variable(qp.A_ineq.shape[0], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum(s)),
               [qp.A_ineq @ z <= qp.b_ineq + s]).solve(solver=cp.OSQP,
                                                       eps_abs=1e-9, eps_rel=1e-9)
    return (float(np.max(s.value)) if s.value is not None else float("nan"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="results/exact_penalty_audit.json")
    args = ap.parse_args()
    N = args.horizon

    states, us = record_states(10)
    cvx = MPCSolver(horizon=N)
    rows = []

    for k in range(1, TIME_STEPS):
        x, up = states[k], us[k - 1]
        Ap, Bp = linearize(float(np.mean(x[0:3])), float(np.mean(x[7:10])), False)
        common = dict(Q_diag=cvx.Q_diag, R_val=cvx.R_val, S_val=cvx.S_val,
                      target_temp=TARGET_TEMP, trust_region=False,
                      drop_uncontrollable_rows=True)
        qp_h = build_canonical_qp(Ap, Bp, x, up, N, soft_state_constraints=False, **common)
        qp_s = build_canonical_qp(Ap, Bp, x, up, N, soft_state_constraints=True, **common)

        n_act = 4 * N                       # box (2N) + slew (2N); gradient rows follow
        max_slack = hard_is_feasible(qp_h)
        feasible = max_slack <= 1e-6

        rec = {"step": k, "hard_feasible": bool(feasible),
               "hard_min_max_slack": max_slack,
               "n_grad_rows_kept": qp_h.gradient_rows["n_kept"]}

        z_soft, st_s = solve_soft(qp_s)
        if z_soft is not None:
            s_used = z_soft[N:] if qp_s.n_slacks else np.zeros(0)
            rec["soft_status"] = st_s
            rec["max_slack_used_degC"] = float(np.max(s_used)) if s_used.size else 0.0
            rec["sum_slack_degC"] = float(np.sum(s_used)) if s_used.size else 0.0

        if feasible:
            z_hard, lam_inf, st_h = solve_hard_with_duals(qp_h, n_act)
            rec["hard_status"] = st_h
            if lam_inf is not None:
                rec["lambda_inf_on_gradient_rows"] = lam_inf
                rec["exactness_satisfied"] = bool(SLACK_WEIGHT_LIN > lam_inf)
                rec["exactness_margin"] = float(SLACK_WEIGHT_LIN - lam_inf)
                if z_soft is not None and z_hard is not None:
                    rec["u0_soft_minus_hard_degC"] = float(z_soft[0] - z_hard[0])
                    rec["max_abs_u_diff_degC"] = float(
                        np.max(np.abs(z_soft[:N] - z_hard[:N])))
        rows.append(rec)

    graded = [r for r in rows if "exactness_satisfied" in r]
    feas = [r for r in rows if r["hard_feasible"]]
    stiff = [r for r in graded if STIFF_LO <= r["step"] < STIFF_HI]
    lam = [r["lambda_inf_on_gradient_rows"] for r in graded]
    u0d = [abs(r["u0_soft_minus_hard_degC"]) for r in graded
           if "u0_soft_minus_hard_degC" in r]
    slack_used = [r.get("max_slack_used_degC", 0.0) for r in rows]

    summary = {
        "slack_weight_lin": SLACK_WEIGHT_LIN,
        "slack_weight_quad": SLACK_WEIGHT_QUAD,
        "horizon": N,
        "n_steps": len(rows),
        "n_hard_feasible": len(feas),
        "pct_hard_feasible": 100.0 * len(feas) / max(1, len(rows)),
        "n_graded_for_exactness": len(graded),
        "max_lambda_inf": float(max(lam)) if lam else None,
        "median_lambda_inf": float(np.median(lam)) if lam else None,
        "pct_exactness_satisfied":
            100.0 * float(np.mean([r["exactness_satisfied"] for r in graded]))
            if graded else None,
        "pct_exactness_satisfied_stiff":
            100.0 * float(np.mean([r["exactness_satisfied"] for r in stiff]))
            if stiff else None,
        "max_u0_soft_minus_hard_degC": float(max(u0d)) if u0d else None,
        "median_u0_soft_minus_hard_degC": float(np.median(u0d)) if u0d else None,
        "max_slack_used_degC": float(max(slack_used)) if slack_used else None,
        "interpretation": (
            "pct_exactness_satisfied is the fraction of hard-FEASIBLE steps on "
            "which slack_weight_lin exceeds ||lambda*||_inf, i.e. on which the "
            "l1 penalty is exact and the soft QP is returning the hard solution. "
            "max_u0_soft_minus_hard_degC is the realised discrepancy on the "
            "APPLIED move. A non-zero value on an exact step is attributable to "
            "the quadratic slack term, which is never exact at finite weight."),
    }

    out = {"summary": summary, "per_step": rows}
    p = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
