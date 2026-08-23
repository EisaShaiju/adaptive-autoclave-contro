"""
gradient_row_infeasibility_probe.py
Exact, algebraic infeasibility proof for the HARD-constrained MPC-QP at the
stiff exotherm state used by tools/snn_solve_instrumentation.py -- and the
root cause of that instrumentation's n_projections=0 anomaly
(results/stiff_divergence_trace_summary.json).

Mechanism (src/qp_builder.py:100-105, :142-150):
  Gamma[i*NX:(i+1)*NX, :] is built by `for j in range(i-1, -1, -1): ...`. For
  i=0 that range is empty, so Gamma's block-row 0 is never written and stays
  exactly zero -- x_0 is the pinned current state, so BY CONSTRUCTION no future
  control can affect it. GGamma row k=0 (+/-) is therefore the exact zero
  vector for every state, at every MPC step; this is a structural property of
  the QP, not specific to any run.

  At the specific gelation-onset state probed here (rho(Ap)=1.5525), rows
  k=1..4 are ALSO exactly zero (measured, not assumed -- see `exact_zero`
  column below); this appears to be a short-horizon transport-delay property
  of Bp (composite nodes 0 and 2 have not yet differentiated within a few
  sample steps) rather than the k=0 structural guarantee, and is reported as a
  measured fact pending further study, not re-derived from first principles.

  A row that is EXACTLY zero-columned turns `c_j^T z + d_j <= 0` into the
  constant `d_j <= 0`. If d_j > 0 there, NO z -- none, regardless of
  reformulation, rescaling, warm start, or solver -- can satisfy it: this is
  the textbook zero-row LP infeasibility case, proved algebraically here
  rather than inferred from a solver's failure to converge.

  snn_opt's `_project_adaptive` guards degenerate rows with
  `if self._c_norms_sq[j] < 1e-12: continue` (skips WITHOUT incrementing the
  projection counter). Because such a row's residual g[j] never changes
  (nothing is subtracted from it), `argmax(g)` returns the SAME dead row on
  every one of the (up to) max_projection_iters attempts per outer iteration,
  forever -- explaining n_projections=0 while the recomputed violation stays
  frozen and the raw iterate diverges.

Contrast: the SOFT reformulation (src/qp_builder.py "Soft (slack)
reformulation of the GRADIENT rows only") adds a slack column to every
gradient row, including these structurally-zero ones -- giving them a nonzero
coefficient on a variable that CAN move. That is precisely why the soft QP is
feasible at this same state (see results/feasibility_certificate.json and the
soft-form check in this script) while the hard QP is not: softening does not
"paper over" degeneracy, it is the one reformulation that can fix a row whose
c_j is exactly zero, because rescaling or preconditioning provably cannot
(they multiply c_j, and 0 times anything is 0).

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/gradient_row_infeasibility_probe.py
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

import src.constants as const
from src.snn_mpc_controller import SNNMPCSolver
from snn_solve_instrumentation import find_test_states
from snn_opt import OptimizationProblem, SNNSolver


def slack_lp(C, d):
    m, n = C.shape
    z = cp.Variable(n)
    s = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(s)), [C @ z + d <= s])
    prob.solve(solver=cp.OSQP, max_iter=200000)
    s_val = None if s.value is None else np.asarray(s.value)
    return prob.status, s_val


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    easy, stiff = find_test_states()
    N = 20
    print(f"stiff state (tools/snn_solve_instrumentation.find_test_states): "
          f"k={stiff['k']} rho(Ap)={stiff['rho']:.4f}")

    # ---- HARD form: row-level algebraic infeasibility proof ----
    # `drop_uncontrollable_rows=False` is REQUIRED here and is not a default
    # left over by accident: this probe exists to demonstrate the dead rows, so
    # it must build the pre-Revision-5 constraint set that still contains them.
    # With the shipped default those rows are absent and the 4*N+k / 5*N+k
    # indexing below would silently address the wrong rows. The fix this probe
    # motivated is in src/qp_builder.py; the probe stays pinned to the old form
    # so its published finding remains reproducible.
    ctrl_hard = SNNMPCSolver(horizon=N, target_temp=120.0, soft_state_constraints=False,
                             drop_uncontrollable_rows=False)
    qp_hard = ctrl_hard.build_qp(stiff["x0"], stiff["u_prev"])
    C_hard = np.asarray(qp_hard.A_ineq)
    b_hard = np.asarray(qp_hard.b_ineq)

    grad_rows = []
    for k in range(N):
        for sign_label, row_idx in (("+", 4 * N + k), ("-", 5 * N + k)):
            c_row = C_hard[row_idx]
            exact_zero = bool(np.all(c_row == 0.0))
            b_val = float(b_hard[row_idx])
            unconditionally_infeasible = exact_zero and (b_val < 0.0)
            grad_rows.append({
                "k": k, "sign": sign_label, "row_index": row_idx,
                "exact_zero_row": exact_zero, "b_ineq": b_val,
                "unconditionally_infeasible": unconditionally_infeasible,
            })
    bad = [r for r in grad_rows if r["unconditionally_infeasible"]]
    print(f"\nHARD form: {len(bad)} rows are algebraically unconditionally "
          f"infeasible (0^T z <= negative constant):")
    for r in bad:
        print(f"  k={r['k']} sign={r['sign']} row={r['row_index']} "
              f"b_ineq={r['b_ineq']:.6e}")

    status, s = slack_lp(C_hard, -b_hard)
    hard_lp = {"lp_status": status,
              "max_slack": None if s is None else float(s.max())}
    print(f"HARD form slack LP: status={status} "
          f"max(s*)={'n/a' if s is None else f'{s.max():.6e}'}")

    # ---- SOFT form: same state, confirm feasible ----
    ctrl_soft = SNNMPCSolver(horizon=N, target_temp=120.0, soft_state_constraints=True)
    qp_soft = ctrl_soft.build_qp(stiff["x0"], stiff["u_prev"])
    C_soft, d_soft = np.asarray(qp_soft.A_ineq), -np.asarray(qp_soft.b_ineq)
    status_s, s_s = slack_lp(C_soft, d_soft)
    soft_lp = {"lp_status": status_s,
              "max_slack": None if s_s is None else float(s_s.max())}
    print(f"SOFT form slack LP:  status={status_s} "
          f"max(s*)={'n/a' if s_s is None else f'{s_s.max():.6e}'} "
          f"({'FEASIBLE' if (s_s is not None and s_s.max() <= 1e-6) else 'check'})")

    # ---- n_projections reproduction on the hard form, default config ----
    H_s, g_s, C_s, d_s, D = ctrl_hard._condition(qp_hard.H, qp_hard.f, C_hard, -b_hard)
    U_cold = np.asarray(ctrl_hard._warm_hold(stiff["u_prev"]), dtype=float)
    res = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=np.asarray(C_s), d=np.asarray(d_s)),
                    ctrl_hard.solver_config).solve(U_cold * D, verbose=False)
    snn_row = {
        "converged": bool(res.converged),
        "convergence_reason": str(res.convergence_reason),
        "iterations_used": int(res.iterations_used),
        "n_projections": int(res.n_projections),
        "final_max_violation_recomputed": float(
            np.max(np.asarray(C_s) @ np.asarray(res.final_x) + np.asarray(d_s))),
    }
    print(f"\nHARD form SNN solve: converged={snn_row['converged']} "
          f"n_projections={snn_row['n_projections']} "
          f"final_violation={snn_row['final_max_violation_recomputed']:.6e}")

    out = {
        "git_commit": commit,
        "gradient_max": const.GRADIENT_MAX,
        "stiff_state": {"k": stiff["k"], "rho_Ap": stiff["rho"]},
        "horizon": N,
        "mechanism": ("Gamma's block-row for k=0 is never written by the "
                      "recursive loop in src/qp_builder.py (empty range for "
                      "i=0), so GGamma row k=0 is EXACTLY zero for every "
                      "state -- x_0 is pinned, decision-independent by "
                      "construction. At this state, rows k=1..4 are also "
                      "measured exactly zero (transport-delay property of Bp, "
                      "not re-derived here). A row with c_j=0 and d_j>0 makes "
                      "0<=d_j false for every z: unconditional infeasibility, "
                      "immune to rescaling/preconditioning/solver choice."),
        "gradient_rows": grad_rows,
        "hard_form_slack_lp": hard_lp,
        "soft_form_slack_lp": soft_lp,
        "hard_form_snn_solve": snn_row,
    }
    dest = PROJECT_ROOT / "results" / "gradient_row_infeasibility.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
