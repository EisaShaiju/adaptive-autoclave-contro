"""
conditioning_sweep.py
Systematic conditioning / convergence sweep on a representative stiff
exotherm QP, per the advisor's request to make "a genuine effort" at getting
the SNN-QP to converge feasibly to tolerance -- rather than the single
row-normalization attempt already tested and rejected
(results/qp_conditioning_change_report.json).

Axes swept (all applied IDENTICALLY to both controllers, so the comparison
stays fair by construction):
  * trust_region        -- False (pure first-order model) / True (bounded exotherm Jacobian)
  * soft_state_constraints -- False (hard gradient rows) / True (slacked gradient rows)
  * horizon N           -- controls the Ap^(N-1) amplification directly
  * k0_scale            -- SNN gradient step size margin

For each configuration, on the stiff state:
  1. Solve with OSQP (reference) -- is the QP even feasible?
  2. Solve with the SNN -- converged? feasible? how many iterations?
Feasibility is always evaluated on the ORIGINAL (unconditioned) canonical QP
from the mapped-back solution.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/conditioning_sweep.py
"""
from pathlib import Path
import sys
import json
import itertools

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
from snn_opt import OptimizationProblem, SNNSolver

ROLLOUT_STEPS = 105


def find_stiff_state(trust_region):
    """Deterministic rollout with a model-identical CVXPY controller; return
    the state with maximum rho(Ap) under the given trust_region setting."""
    plant = AutoclavePlant(initial_temp=28.0)
    ctrl = MPCSolver(horizon=20, target_temp=120.0, trust_region=trust_region)
    current_Ta = 28.0
    current_state = plant.get_state()
    best = None
    for k in range(ROLLOUT_STEPS):
        x0 = current_state.copy()
        u_prev = current_Ta
        avg_T, avg_a = np.mean(x0[0:3]), np.mean(x0[7:10])
        Ap, _ = linearize(avg_T, avg_a, trust_region=trust_region)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))
        if best is None or rho > best["rho"]:
            best = {"k": k, "x0": x0, "u_prev": u_prev, "rho": rho}
        current_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        current_state = plant.step(Ta_input=current_Ta)
    return best


def evaluate_config(state, trust_region, soft, N, k0_scale):
    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0, trust_region=trust_region,
                        soft_state_constraints=soft, k0_scale=k0_scale)
    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H, f, A, b = qp.H, qp.f, qp.A_ineq, qp.b_ineq
    C_raw, d_raw = A, -b
    problem_raw = OptimizationProblem(A=H, b=f, C=C_raw, d=d_raw)

    if not (np.isfinite(H).all() and np.isfinite(f).all() and np.isfinite(A).all()
            and np.isfinite(b).all()):
        return {"osqp_status": "non_finite_QP", "snn_converged": False,
                "snn_feasible": False, "snn_residual": None, "snn_iterations": None,
                "cond_H": None, "osqp_u0": None, "snn_u0": None}

    # --- reference solve: is the QP feasible at all? ---
    U = cp.Variable(H.shape[0])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H)) + f @ U),
                       [A @ U <= b])
    try:
        prob.solve(solver=cp.OSQP)
        osqp_status = prob.status
        osqp_u0 = float(U.value[0]) if U.value is not None else None
    except Exception as exc:
        osqp_status, osqp_u0 = f"error", None

    # --- SNN solve on the conditioned arrays, cold start ---
    H_s, g_s, C_s, d_s, D = ctrl._condition(H, f, C_raw, d_raw)
    U_cold = ctrl._warm_hold(state["u_prev"], H.shape[0])
    try:
        solver = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s),
                            ctrl.solver_config)
        result = solver.solve(U_cold * D, verbose=False)
        U_phys = result.final_x / D
        residual = float(problem_raw.max_violation(U_phys))
        feasible = residual <= ctrl.solver_config.convergence.feasibility_tol
        hit_cap = result.convergence_reason == "max_iterations"
        return {
            "osqp_status": osqp_status, "osqp_u0": osqp_u0,
            "snn_converged": bool(result.converged) and feasible and not hit_cap,
            "snn_raw_converged_flag": bool(result.converged),
            "snn_feasible": feasible, "snn_residual": residual,
            "snn_iterations": int(result.iterations_used),
            "snn_u0": float(U_phys[0]),
            "cond_H": float(np.linalg.cond(H_s)),
        }
    except Exception as exc:
        return {"osqp_status": osqp_status, "osqp_u0": osqp_u0, "snn_converged": False,
                "snn_feasible": False, "snn_residual": None, "snn_iterations": None,
                "cond_H": None, "snn_u0": None, "error": str(exc)}


def main():
    results = []
    print("Locating stiff states under each trust_region setting...")
    stiff_states = {tr: find_stiff_state(tr) for tr in (False, True)}
    for tr, s in stiff_states.items():
        print(f"  trust_region={tr}: k={s['k']}, rho(Ap)={s['rho']:.4f}")

    TRUST = [False, True]
    SOFT = [False, True]
    HORIZONS = [20, 10, 5]
    K0 = [0.5, 0.1]

    print(f"\nSweeping {len(TRUST)*len(SOFT)*len(HORIZONS)*len(K0)} configurations "
          f"on the stiff exotherm QP...\n")
    header = (f"{'trust_reg':<10}{'soft':<7}{'N':>4}{'k0':>6}{'osqp_status':>14}"
              f"{'snn_conv':>10}{'snn_feas':>10}{'residual':>13}{'iters':>7}{'cond(H_s)':>12}")
    print(header)
    print("-" * len(header))

    for tr, soft, N, k0 in itertools.product(TRUST, SOFT, HORIZONS, K0):
        state = stiff_states[tr]
        r = evaluate_config(state, tr, soft, N, k0)
        row = {"trust_region": tr, "soft_state_constraints": soft, "N": N,
               "k0_scale": k0, "rho_Ap": state["rho"], **r}
        results.append(row)
        res_str = "N/A" if r["snn_residual"] is None else f"{r['snn_residual']:.4e}"
        cond_str = "N/A" if r["cond_H"] is None else f"{r['cond_H']:.3e}"
        it_str = "N/A" if r["snn_iterations"] is None else str(r["snn_iterations"])
        print(f"{str(tr):<10}{str(soft):<7}{N:>4}{k0:>6}{r['osqp_status']:>14}"
              f"{str(r['snn_converged']):>10}{str(r['snn_feasible']):>10}"
              f"{res_str:>13}{it_str:>7}{cond_str:>12}")

    # ---- pick winners ----
    converged = [r for r in results if r["snn_converged"]]
    feasible = [r for r in results if r["snn_feasible"]]
    print("\n" + "=" * 100)
    print(f"Configurations where the SNN CONVERGED (feasible + own criterion + not iteration-capped): "
          f"{len(converged)}/{len(results)}")
    for r in converged:
        print(f"   trust_region={r['trust_region']} soft={r['soft_state_constraints']} "
              f"N={r['N']} k0={r['k0_scale']} -> residual={r['snn_residual']:.3e}, "
              f"iters={r['snn_iterations']}")
    print(f"Configurations where the SNN solution was FEASIBLE (regardless of flag): "
          f"{len(feasible)}/{len(results)}")
    for r in feasible:
        print(f"   trust_region={r['trust_region']} soft={r['soft_state_constraints']} "
              f"N={r['N']} k0={r['k0_scale']} -> residual={r['snn_residual']:.3e}, "
              f"iters={r['snn_iterations']}, raw_flag={r['snn_raw_converged_flag']}")
    print("=" * 100)

    out = PROJECT_ROOT / "results" / "conditioning_sweep.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"stiff_states": {str(k): {"k": v["k"], "rho": v["rho"]}
                                     for k, v in stiff_states.items()},
                    "results": results}, fh, indent=2, default=str)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
