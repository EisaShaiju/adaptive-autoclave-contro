"""
feasibility_certificate_probe.py
Settles feasible-vs-infeasible for the stiff-step MPC-QP INDEPENDENTLY of the
SNN solver, via the standard slack LP

    min_{z,s} 1^T s   s.t.  C z + d <= s,  s >= 0

    max(s*) ~ 0  -> feasible set non-empty
    max(s*) > 0  -> provably infeasible; s* IS the certificate

Motivation: three signals in this repo have each been mistaken for
infeasibility -- `converged=False`, a large cold-start violation, and a
diverging iterate. None of them is evidence about the feasible set: only a
solver-independent feasibility certificate is.

Also records, for the binding row, the split of the cold-start violation into
its decision-dependent (c_j^T z) and decision-independent (d_j) parts, and the
solver's own n_projections on the identical conditioned instance.

Reuses tools/convergence_blocker_probe.find_stiff_state so the probed state
matches results/convergence_blocker_probe.json (k=87, rho=2.4928).

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/feasibility_certificate_probe.py
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from src.snn_mpc_controller import SNNMPCSolver
from convergence_blocker_probe import find_stiff_state, TRUST_REGION, SOFT
from snn_opt import OptimizationProblem, SNNSolver

HORIZONS = (20, 10, 5)
FEAS_EPS = 1e-6   # max(s*) below this counts as numerically feasible


def slack_lp(C, d):
    """min 1^T s s.t. Cz + d <= s, s >= 0. Returns (status, s*, z*)."""
    m, n = C.shape
    z = cp.Variable(n)
    s = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(s)), [C @ z + d <= s])
    prob.solve(solver=cp.OSQP, max_iter=200000)
    s_val = None if s.value is None else np.asarray(s.value)
    z_val = None if z.value is None else np.asarray(z.value)
    return prob.status, s_val, z_val


def probe(state, N):
    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0, trust_region=TRUST_REGION,
                        soft_state_constraints=SOFT)
    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H, g, C, d = qp.H, qp.f, qp.A_ineq, -qp.b_ineq
    H_s, g_s, C_s, d_s, D = ctrl._condition(H, g, C, d)
    C_s, d_s = np.asarray(C_s), np.asarray(d_s)

    row = {"N": N, "n_constraints": int(C_s.shape[0]),
           "n_variables": int(C_s.shape[1]),
           "qp_fingerprint": qp.fingerprint()}

    # Feasibility on BOTH raw and conditioned arrays. Conditioning is a change
    # of variables and must not change feasibility; disagreement = bug in
    # _condition, not in the QP.
    for label, Cm, dv in (("raw", np.asarray(C), np.asarray(d)),
                          ("conditioned", C_s, d_s)):
        status, s, _ = slack_lp(Cm, dv)
        if s is None:
            row[label] = {"lp_status": status, "certificate": None}
            continue
        row[label] = {
            "lp_status": status,
            "sum_slack": float(s.sum()),
            "max_slack": float(s.max()),
            "binding_row": int(np.argmax(s)),
            "feasible": bool(s.max() <= FEAS_EPS),
        }

    # Row-norm census: rules out snn_opt's degenerate-row `continue` branch
    norms_sq = np.sum(C_s ** 2, axis=1)
    row["conditioned_row_norm_sq"] = {
        "min": float(norms_sq.min()), "max": float(norms_sq.max()),
        "n_below_1e-12": int(np.sum(norms_sq < 1e-12)),
    }

    # Cold-start violation decomposition on the binding row
    U_cold = np.asarray(ctrl._warm_hold(state["u_prev"], H_s.shape[0]), dtype=float)
    x_scaled = U_cold * D
    gv = C_s @ x_scaled + d_s
    j = int(np.argmax(gv))
    dep = float(C_s[j] @ x_scaled)
    row["cold_start"] = {
        "most_violated_row": j,
        "total_violation": float(gv[j]),
        "decision_dependent": dep,
        "decision_independent_offset": float(d_s[j]),
        "offset_share_pct": float(100.0 * d_s[j] / gv[j]) if gv[j] != 0 else None,
    }

    # Solver behaviour on the identical conditioned instance
    res = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s),
                    ctrl.solver_config).solve(x_scaled, verbose=False)
    final_x = np.asarray(res.final_x)
    row["snn"] = {
        "converged": bool(res.converged),
        "convergence_reason": str(res.convergence_reason),
        "iterations_used": int(res.iterations_used),
        "n_projections": int(res.n_projections),
        "max_projection_iters": int(ctrl.solver_config.max_projection_iters),
        "projections_per_iteration": float(res.n_projections / max(1, res.iterations_used)),
        "final_max_violation_recomputed": float(np.max(C_s @ final_x + d_s)),
        "final_x_norm": float(np.linalg.norm(final_x)),
    }
    return row


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    state = find_stiff_state()
    print(f"stiff state: k={state['k']} rho(Ap)={state['rho']:.4f} "
          f"trust_region={TRUST_REGION} soft_state_constraints={SOFT}")

    rows = []
    for N in HORIZONS:
        r = probe(state, N)
        rows.append(r)
        cond = r["conditioned"]
        print(f"\nN={N}: conditioned LP {cond['lp_status']} "
              f"max(s*)={cond['max_slack']:.3e} -> "
              f"{'FEASIBLE' if cond['feasible'] else 'INFEASIBLE'}")
        print(f"  cold-start offset share: {r['cold_start']['offset_share_pct']:.2f}%")
        print(f"  snn: converged={r['snn']['converged']} "
              f"n_projections={r['snn']['n_projections']} "
              f"({r['snn']['projections_per_iteration']:.1f}/iter, "
              f"cap {r['snn']['max_projection_iters']})")

    out = {
        "git_commit": commit,
        "feasibility_epsilon": FEAS_EPS,
        "config": {"trust_region": TRUST_REGION,
                   "soft_state_constraints": SOFT,
                   "target_temp": 120.0},
        "stiff_state": {"k": state["k"], "rho_Ap": state["rho"]},
        "note": ("Slack LP is solver-independent. It answers ONLY whether the "
                 "feasible set is non-empty; it says nothing about whether the "
                 "SNN converges. The infeasibility recorded in "
                 "docs/PHASE4_VALIDATION_REPORT.md invariant 6 concerns the "
                 "HARD constraint form; this probe covers the SOFT form."),
        "rows": rows,
    }
    dest = PROJECT_ROOT / "results" / "feasibility_certificate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
