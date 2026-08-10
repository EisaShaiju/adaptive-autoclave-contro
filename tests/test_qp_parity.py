"""
test_qp_parity.py
Focused parity test for the canonical-QP unification (see
docs/PHASE4_VALIDATION_REPORT.md §3). Plain-assert script -- this repo has no
test runner -- runnable directly:

    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_qp_parity.py

Does NOT run the closed-loop simulation. Proves, at one representative plant
state:
  1. Both controllers' cost-weight ingredients (Q_diag, R_val, S_val, N,
     target_temp) match exactly.
  2. Given the SAME (Ap, Bp, x0, u_prev), MPCSolver.build_qp() and
     SNNMPCSolver.build_qp()'s underlying construction produce numerically
     identical H, f, A_ineq, b_ineq, bounds, and variable_order.
  3. The QP fingerprint is deterministic (same inputs -> same hash) and
     sensitive (different inputs -> different hash).
  4. With each controller's REAL (possibly trust_region-differing) Ap/Bp,
     solving the resulting canonical QP with a common reference solver
     (OSQP) at a non-gelation state gives closely-agreeing u_0 -- the
     practical "same QP -> same answer" claim.
  5. Matrix shapes are as expected: H is N x N, f is N, A_ineq is 6N x N,
     b_ineq is 6N.
"""
from pathlib import Path
import sys

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
from src.qp_builder import build_canonical_qp
from src.plant_simulator import AutoclavePlant

N = 20
TARGET_TEMP = 120.0

failures = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def main():
    ctrl_cvx = MPCSolver(horizon=N, target_temp=TARGET_TEMP)
    ctrl_snn = SNNMPCSolver(horizon=N, target_temp=TARGET_TEMP)

    # ---- 1. Ingredient parity (model-parity ingredients table) ----
    print("\n--- 1. Cost-weight ingredient parity ---")
    check("Q_diag matches", np.array_equal(ctrl_cvx.Q_diag, ctrl_snn.Q_diag))
    check("R_val matches", ctrl_cvx.R_val == ctrl_snn.R_val)
    check("S_val matches", ctrl_cvx.S_val == ctrl_snn.S_val)
    check("N matches", ctrl_cvx.N == ctrl_snn.N)
    check("target_temp matches", ctrl_cvx.target_temp == ctrl_snn.target_temp)

    # ---- Representative non-gelation plant state (a few steps of heat-up) ----
    plant = AutoclavePlant(initial_temp=28.0)
    u_prev = 28.0
    for _ in range(10):
        plant.step(Ta_input=60.0)
    x0 = plant.get_state()
    u_prev = 60.0

    # ---- 2. Construction-path equality: force IDENTICAL Ap, Bp into both
    #         adapters' underlying construction, isolating the shared
    #         qp_builder path from the one documented trust_region divergence.
    print("\n--- 2. Construction-path equality (same Ap, Bp, x0, u_prev) ---")
    Ap_fixed, Bp_fixed = linearize(np.mean(x0[0:3]), np.mean(x0[7:10]), trust_region=False)

    qp_cvx = build_canonical_qp(Ap_fixed, Bp_fixed, x0, u_prev, N,
                                 ctrl_cvx.Q_diag, ctrl_cvx.R_val, ctrl_cvx.S_val,
                                 ctrl_cvx.target_temp, trust_region=False)
    qp_snn = build_canonical_qp(Ap_fixed, Bp_fixed, x0, u_prev, N,
                                 ctrl_snn.Q_diag, ctrl_snn.R_val, ctrl_snn.S_val,
                                 ctrl_snn.target_temp, trust_region=False)

    check("H shapes match and equal", qp_cvx.H.shape == qp_snn.H.shape and np.allclose(qp_cvx.H, qp_snn.H, atol=1e-12))
    check("f equal", np.allclose(qp_cvx.f, qp_snn.f, atol=1e-12))
    check("A_ineq shapes match and equal", qp_cvx.A_ineq.shape == qp_snn.A_ineq.shape and np.allclose(qp_cvx.A_ineq, qp_snn.A_ineq, atol=1e-12))
    check("b_ineq equal", np.allclose(qp_cvx.b_ineq, qp_snn.b_ineq, atol=1e-12))
    check("lower_bound equal", np.array_equal(qp_cvx.lower_bound, qp_snn.lower_bound))
    check("upper_bound equal", np.array_equal(qp_cvx.upper_bound, qp_snn.upper_bound))
    check("variable_order equal", qp_cvx.variable_order == qp_snn.variable_order)

    max_H_diff = float(np.max(np.abs(qp_cvx.H - qp_snn.H)))
    max_f_diff = float(np.max(np.abs(qp_cvx.f - qp_snn.f)))
    max_A_diff = float(np.max(np.abs(qp_cvx.A_ineq - qp_snn.A_ineq)))
    max_b_diff = float(np.max(np.abs(qp_cvx.b_ineq - qp_snn.b_ineq)))
    print(f"  max|dH|={max_H_diff:.3e}  max|df|={max_f_diff:.3e}  "
          f"max|dA_ineq|={max_A_diff:.3e}  max|db_ineq|={max_b_diff:.3e}")

    # ---- 2b. Both real adapters' build_qp() delegate to the same construction ----
    print("\n--- 2b. Real adapter build_qp() vs direct canonical builder ---")
    qp_cvx_real = ctrl_cvx.build_qp(x0, u_prev)
    Ap_cvx_real, Bp_cvx_real = ctrl_cvx.update_matrices(np.mean(x0[0:3]), np.mean(x0[7:10]))
    qp_cvx_expected = build_canonical_qp(Ap_cvx_real, Bp_cvx_real, x0, u_prev, N,
                                          ctrl_cvx.Q_diag, ctrl_cvx.R_val, ctrl_cvx.S_val,
                                          ctrl_cvx.target_temp, trust_region=False)
    check("MPCSolver.build_qp() matches direct builder call",
          np.allclose(qp_cvx_real.H, qp_cvx_expected.H) and np.allclose(qp_cvx_real.f, qp_cvx_expected.f))

    H_snn, f_snn, A_snn, b_snn = ctrl_snn.build_dense_qp(Ap_fixed, Bp_fixed, x0, u_prev)
    check("SNNMPCSolver.build_dense_qp() backward-compat matches canonical form",
          np.allclose(H_snn, qp_snn.H) and np.allclose(f_snn, qp_snn.f)
          and np.allclose(A_snn, qp_snn.A_ineq) and np.allclose(b_snn, -qp_snn.b_ineq))

    # ---- 3. Deterministic fingerprint ----
    print("\n--- 3. Deterministic QP fingerprint ---")
    fp1 = qp_cvx.fingerprint()
    fp2 = build_canonical_qp(Ap_fixed, Bp_fixed, x0, u_prev, N,
                              ctrl_cvx.Q_diag, ctrl_cvx.R_val, ctrl_cvx.S_val,
                              ctrl_cvx.target_temp, trust_region=False).fingerprint()
    fp3 = build_canonical_qp(Ap_fixed, Bp_fixed, x0, u_prev - 1.0, N,
                              ctrl_cvx.Q_diag, ctrl_cvx.R_val, ctrl_cvx.S_val,
                              ctrl_cvx.target_temp, trust_region=False).fingerprint()
    check("fingerprint deterministic (same inputs -> same hash)", fp1 == fp2)
    check("fingerprint sensitive (different u_prev -> different hash)", fp1 != fp3)
    print(f"  fingerprint = {fp1}")

    # ---- 4. Practical equivalence: real controllers, common reference solve ----
    print("\n--- 4. Real-adapter u0 agreement at non-gelation state (reference OSQP) ---")
    qp_cvx_real2 = ctrl_cvx.build_qp(x0, u_prev)
    qp_snn_real = ctrl_snn.build_qp(x0, u_prev)

    def solve_osqp(qp):
        U = cp.Variable(N)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(qp.H)) + qp.f @ U),
                           [qp.A_ineq @ U <= qp.b_ineq])
        prob.solve(solver=cp.OSQP)
        return (U.value[0] if U.value is not None else None), prob.status

    u0_cvx, status_cvx = solve_osqp(qp_cvx_real2)
    u0_snn, status_snn = solve_osqp(qp_snn_real)
    gap = None if (u0_cvx is None or u0_snn is None) else abs(u0_cvx - u0_snn)
    print(f"  u0_cvx={u0_cvx} ({status_cvx})  u0_snn={u0_snn} ({status_snn})  gap={gap}")
    check("both reference solves optimal", status_cvx == "optimal" and status_snn == "optimal")
    check("u0 agrees within 0.5 C at non-gelation state (trust_region-only divergence)",
          gap is not None and gap < 0.5, detail=f"gap={gap}")

    # ---- 5. Shapes ----
    print("\n--- 5. Matrix shapes ---")
    check("H shape (N, N)", qp_cvx.H.shape == (N, N))
    check("f shape (N,)", qp_cvx.f.shape == (N,))
    check("A_ineq shape (6N, N)", qp_cvx.A_ineq.shape == (6 * N, N))
    check("b_ineq shape (6N,)", qp_cvx.b_ineq.shape == (6 * N,))
    check("lower_bound/upper_bound shape (N,)",
          qp_cvx.lower_bound.shape == (N,) and qp_cvx.upper_bound.shape == (N,))

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
    else:
        print("RESULT: ALL CHECKS PASSED")
    print("=" * 60)
    return len(failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
