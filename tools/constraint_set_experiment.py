"""
constraint_set_experiment.py
One-variable-at-a-time sweep over the GRADIENT CONSTRAINT SET, holding the
plant, the horizon-cost weights and the solver configuration fixed.

Question this answers
---------------------
Phase 4 closed with the stiff exotherm window at 22.6 % formal convergence and
12.9 % clipping. Three candidate causes were on the table:

  (a) gradient rows inside the plant's input-to-output dead time, which carry
      no decision variable at all (relative degree r = 5);
  (b) gradient rows far out in the horizon, where the frozen-Jacobian
      prediction has been amplified by rho(Ap)^k and is no longer meaningful;
  (c) the slack penalty being mis-scaled.

This script isolates (a) and (b). (c) is tools/exact_penalty_audit.py.

Method
------
Replay ONE fixed CVXPY-driven trajectory, freeze the resulting plant states, and
rebuild the QP at each state under every constraint-set configuration. Because
the states are frozen, every configuration sees exactly the same physics; only
the constraint set differs. Closed-loop runs cannot separate these effects,
because a configuration that steers differently visits different states.

Reported per configuration:
  * n_rows / n_slacks               -- problem size
  * row_norm_spread                 -- max/min non-zero constraint row norm,
                                       the quantity a projection solver is
                                       most sensitive to
  * cond_H                          -- conditioning of the Hessian
  * hard_infeasible_frac            -- slack LP on the HARD form (OSQP), i.e.
                                       genuine feasibility, not solver status
  * snn_converged_frac              -- the scale-invariant KKT certificate
  * snn_residual / n_projections    -- solver-side diagnostics
  * u0_gap_vs_osqp                  -- accuracy of the APPLIED move

Nothing here is a closed-loop result. Convergence measured on frozen states is
an upper bound on what the closed loop will show, because the closed loop also
has to survive the states its own errors take it to.
"""
import os
import sys
import json
import time
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
from src.snn_mpc_controller import SNNMPCSolver                 # noqa: E402
from snn_opt import OptimizationProblem, SNNSolver              # noqa: E402

TARGET_TEMP = 120.0
TIME_STEPS = 160
STIFF_LO, STIFF_HI = 77, 108        # matches the harness's stiff window
BENIGN = [20, 40, 60, 130, 150]


def record_states(horizon=10):
    """One fixed CVXPY-driven trajectory. Every configuration is scored against
    THESE states, so the physics is held constant across the sweep."""
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


def hard_slack_lp(A, b):
    """min sum(s) s.t. Az <= b + s, s >= 0. Zero iff the set is non-empty.
    This is a feasibility CERTIFICATE, independent of the solver under test --
    never use SNN convergence as evidence of feasibility."""
    z = cp.Variable(A.shape[1])
    s = cp.Variable(A.shape[0], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum(s)), [A @ z <= b + s]).solve(
        solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
    return float(np.max(s.value)) if s.value is not None else float("nan")


def osqp_solve(qp):
    z = cp.Variable(qp.H.shape[0])
    pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(qp.H)) + qp.f @ z),
                    [qp.A_ineq @ z <= qp.b_ineq])
    pr.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)
    return (None, pr.status) if z.value is None else (np.asarray(z.value), pr.status)


def snn_solve(snn, qp):
    H, g, C, d = qp.H, qp.f, qp.A_ineq, -qp.b_ineq
    H_s, g_s, C_s, d_s, D = snn._condition(H, g, C, d)
    if not (np.isfinite(H_s).all() and np.isfinite(g_s).all()):
        return None
    warm = np.zeros(H.shape[0])
    warm[:snn.N] = float(qp.reference["u_prev"])
    try:
        solver = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s),
                           snn.solver_config)
    except ValueError as exc:                    # certifiably infeasible
        return {"rejected": True, "reason": str(exc)}
    t0 = time.perf_counter()
    res = solver.solve(warm * D, verbose=False)
    ms = (time.perf_counter() - t0) * 1000.0
    z = res.final_x / D
    viol = float(np.max(qp.A_ineq @ z - qp.b_ineq)) if qp.A_ineq.size else 0.0
    return {"rejected": False, "converged": bool(res.converged), "z": z,
            "n_projections": int(getattr(res, "n_projections", -1)),
            "iterations": int(getattr(res, "iterations", -1)),
            "residual": max(0.0, viol), "ms": ms}


def evaluate(states, us, horizon, drop, nc, idxs, snn, cvx):
    rows = []
    for k in idxs:
        x, up = states[k], us[k - 1]
        Ap, Bp = linearize(float(np.mean(x[0:3])), float(np.mean(x[7:10])), False)
        common = dict(Q_diag=cvx.Q_diag, R_val=cvx.R_val, S_val=cvx.S_val,
                      target_temp=TARGET_TEMP, trust_region=False,
                      drop_uncontrollable_rows=drop, constraint_horizon=nc)
        qp_soft = build_canonical_qp(Ap, Bp, x, up, horizon,
                                     soft_state_constraints=True, **common)
        qp_hard = build_canonical_qp(Ap, Bp, x, up, horizon,
                                     soft_state_constraints=False, **common)

        nz = np.linalg.norm(qp_soft.A_ineq, axis=1)
        nz = nz[nz > 1e-12]
        spread = float(nz.max() / nz.min()) if nz.size else float("nan")

        hard_slack = hard_slack_lp(qp_hard.A_ineq, qp_hard.b_ineq)
        z_ref, status = osqp_solve(qp_soft)
        s = snn_solve(snn, qp_soft)

        rec = {
            "step": int(k),
            "n_rows": int(qp_soft.A_ineq.shape[0]),
            "n_slacks": int(qp_soft.n_slacks),
            "n_grad_kept": int(qp_soft.gradient_rows["n_kept"]),
            "row_norm_spread": spread,
            "cond_H": float(np.linalg.cond(qp_soft.H)),
            "hard_max_slack": hard_slack,
            "hard_infeasible": bool(hard_slack > 1e-6),
            "unactionable_violation_degC":
                qp_soft.gradient_rows["unactionable_predicted_violation_degC"],
            "osqp_status": status,
        }
        if s is None or s.get("rejected"):
            rec.update(snn_rejected=True, snn_converged=False)
        else:
            rec.update(snn_rejected=False, snn_converged=s["converged"],
                       snn_residual=s["residual"], snn_projections=s["n_projections"],
                       snn_iterations=s["iterations"], snn_ms=s["ms"],
                       u0_gap=(abs(float(s["z"][0]) - float(z_ref[0]))
                               if z_ref is not None else None))
        rows.append(rec)
    return rows


def agg(rows):
    def m(key, default=np.nan):
        v = [r[key] for r in rows if r.get(key) is not None and not isinstance(r.get(key), bool)]
        return float(np.median(v)) if v else default
    conv = [r for r in rows if not r.get("snn_rejected")]
    return {
        "n_states": len(rows),
        "n_grad_kept": rows[0]["n_grad_kept"],
        "n_rows": rows[0]["n_rows"],
        "n_slacks": rows[0]["n_slacks"],
        "median_row_norm_spread": m("row_norm_spread"),
        "max_row_norm_spread": float(max(r["row_norm_spread"] for r in rows)),
        "median_cond_H": m("cond_H"),
        "hard_infeasible_frac": float(np.mean([r["hard_infeasible"] for r in rows])),
        "max_unactionable_violation_degC":
            float(max(r["unactionable_violation_degC"] for r in rows)),
        "snn_converged_frac": float(np.mean([bool(r.get("snn_converged")) for r in rows])),
        "snn_rejected_frac": float(np.mean([bool(r.get("snn_rejected")) for r in rows])),
        "median_snn_residual": m("snn_residual"),
        "median_snn_projections": m("snn_projections"),
        "median_snn_ms": m("snn_ms"),
        "median_u0_gap_degC": m("u0_gap"),
        "max_u0_gap_degC": float(max([r["u0_gap"] for r in conv
                                      if r.get("u0_gap") is not None] or [np.nan])),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--out", default="results/constraint_set_experiment.json")
    args = ap.parse_args()

    print("Recording the reference trajectory (CVXPY, N=10, soft)...")
    states, us = record_states(10)
    stiff = list(range(STIFF_LO, STIFF_HI))
    print(f"  stiff window steps {STIFF_LO}..{STIFF_HI - 1} ({len(stiff)} states), "
          f"benign {BENIGN}")

    out = {
        "method": ("One frozen CVXPY trajectory; the QP is rebuilt at each recorded "
                   "state under every constraint-set configuration. Physics is "
                   "identical across configurations by construction. These are "
                   "OPEN-LOOP per-state measurements, not closed-loop results."),
        "reference_trajectory": {"controller": "MPCSolver", "horizon": 10,
                                 "soft_state_constraints": True,
                                 "time_steps": TIME_STEPS},
        "stiff_window": [STIFF_LO, STIFF_HI],
        "benign_steps": BENIGN,
        "configurations": [],
    }

    cvx = MPCSolver(horizon=10)
    for N in args.horizons:
        snn = SNNMPCSolver(horizon=N, target_temp=TARGET_TEMP, trust_region=False,
                           soft_state_constraints=True, k0_scale=0.1)
        # variable 1: keep vs drop the dead rows, everything else fixed
        # variable 2: constraint horizon, dead rows always dropped
        configs = [("keep_dead_rows", False, None), ("drop_dead_rows", True, None)]
        configs += [(f"Nc={nc}", True, nc) for nc in range(6, N + 1)]
        for name, drop, nc in configs:
            for label, idxs in (("stiff", stiff), ("benign", BENIGN)):
                t0 = time.time()
                rows = evaluate(states, us, N, drop, nc, idxs, snn, cvx)
                a = agg(rows)
                a.update(horizon=N, config=name, drop_uncontrollable_rows=drop,
                         constraint_horizon=nc, window=label,
                         wall_s=round(time.time() - t0, 1))
                out["configurations"].append(a)
                print(f"  N={N:2d} {name:16s} {label:6s}: "
                      f"grad_rows={a['n_grad_kept']:2d} "
                      f"conv={a['snn_converged_frac']*100:5.1f}% "
                      f"hard_infeas={a['hard_infeasible_frac']*100:5.1f}% "
                      f"spread={a['median_row_norm_spread']:.2e} "
                      f"u0gap={a['median_u0_gap_degC']:.2e}")

    os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, args.out)), exist_ok=True)
    p = os.path.join(PROJECT_ROOT, args.out)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
