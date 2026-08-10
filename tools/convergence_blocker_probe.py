"""
convergence_blocker_probe.py
Follow-up to tools/conditioning_sweep.py. The sweep found configurations where
the SNN reaches a FEASIBLE point on the stiff exotherm QP (residual well inside
feasibility_tol) yet `result.converged` still never fires. This probe isolates
WHICH of the solver's three convergence criteria is the blocker, and whether it
is a tolerance-calibration issue, genuine chattering, or slow-but-converging.

snn_opt requires ALL enabled criteria simultaneously, for `patience` consecutive
checks:
    1. objective plateau   : range(obj window)/|obj| < obj_rel_tol (1e-7)
    2. projected gradient  : ||proj_grad|| < proj_grad_tol (5e-2)
    3. feasibility         : max_violation <= feasibility_tol (1e-2)

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/convergence_blocker_probe.py
"""
from pathlib import Path
import sys
import json
from dataclasses import replace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
from snn_opt import OptimizationProblem, SNNSolver

ROLLOUT_STEPS = 105
TRUST_REGION = False          # model-identical default
SOFT = True                   # the setting that made the QP feasible


def find_stiff_state():
    plant = AutoclavePlant(initial_temp=28.0)
    ctrl = MPCSolver(horizon=20, target_temp=120.0, trust_region=TRUST_REGION)
    current_Ta = 28.0
    current_state = plant.get_state()
    best = None
    for k in range(ROLLOUT_STEPS):
        x0 = current_state.copy()
        u_prev = current_Ta
        Ap, _ = linearize(np.mean(x0[0:3]), np.mean(x0[7:10]), trust_region=TRUST_REGION)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))
        if best is None or rho > best["rho"]:
            best = {"k": k, "x0": x0, "u_prev": u_prev, "rho": rho}
        current_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        current_state = plant.step(Ta_input=current_Ta)
    return best


def probe(state, N, k0_scale, max_iterations, proj_grad_tol=None):
    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0, trust_region=TRUST_REGION,
                        soft_state_constraints=SOFT, k0_scale=k0_scale)
    cfg = ctrl.solver_config
    if proj_grad_tol is not None:
        cfg = replace(cfg, convergence=replace(cfg.convergence, proj_grad_tol=proj_grad_tol))
    cfg = replace(cfg, max_iterations=max_iterations)

    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H, f, A, b = qp.H, qp.f, qp.A_ineq, qp.b_ineq
    problem_raw = OptimizationProblem(A=H, b=f, C=A, d=-b)
    H_s, g_s, C_s, d_s, D = ctrl._condition(H, f, A, -b)
    U_cold = ctrl._warm_hold(state["u_prev"], H.shape[0])

    problem_s = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
    solver = SNNSolver(problem_s, cfg)
    result = solver.solve(U_cold * D, verbose=False)

    U_phys = result.final_x / D
    residual = float(problem_raw.max_violation(U_phys))
    feasible = residual <= cfg.convergence.feasibility_tol
    proj_grad = float(result.final_proj_grad_norm)

    return {
        "N": N, "k0_scale": k0_scale, "max_iterations": max_iterations,
        "proj_grad_tol": cfg.convergence.proj_grad_tol,
        "converged_flag": bool(result.converged),
        "reason": result.convergence_reason,
        "iterations": int(result.iterations_used),
        "residual_physical": residual, "feasible": feasible,
        "final_proj_grad_norm": proj_grad,
        "proj_grad_within_tol": proj_grad < cfg.convergence.proj_grad_tol,
        "u0": float(U_phys[0]),
    }


def main():
    state = find_stiff_state()
    print(f"Stiff state: k={state['k']}, rho(Ap)={state['rho']:.4f}, "
          f"trust_region={TRUST_REGION}, soft_state_constraints={SOFT}\n")

    rows = []

    print("--- A. Is feasibility now satisfied, and what is the projected-gradient norm? ---")
    hdr = (f"{'N':>4}{'k0':>7}{'max_it':>9}{'conv':>7}{'feasible':>10}{'residual':>12}"
           f"{'proj_grad':>12}{'pg<tol':>8}{'u0':>10}")
    print(hdr)
    for N in (20, 10, 5):
        for k0 in (0.1, 0.05):
            r = probe(state, N, k0, 8000)
            rows.append(r)
            print(f"{r['N']:>4}{r['k0_scale']:>7}{r['max_iterations']:>9}"
                  f"{str(r['converged_flag']):>7}{str(r['feasible']):>10}"
                  f"{r['residual_physical']:>12.4e}{r['final_proj_grad_norm']:>12.4e}"
                  f"{str(r['proj_grad_within_tol']):>8}{r['u0']:>10.3f}")

    print("\n--- B. Does a larger iteration budget close it? (N=20, k0=0.1) ---")
    print(hdr)
    for max_it in (8000, 20000, 50000):
        r = probe(state, 20, 0.1, max_it)
        rows.append(r)
        print(f"{r['N']:>4}{r['k0_scale']:>7}{r['max_iterations']:>9}"
              f"{str(r['converged_flag']):>7}{str(r['feasible']):>10}"
              f"{r['residual_physical']:>12.4e}{r['final_proj_grad_norm']:>12.4e}"
              f"{str(r['proj_grad_within_tol']):>8}{r['u0']:>10.3f}")

    print("\n--- C. Is proj_grad_tol simply mis-calibrated for this problem scale? ---")
    print("    (relaxing ONLY the projected-gradient tolerance; feasibility bar unchanged)")
    print(hdr)
    for pgt in (5e-2, 1.0, 10.0, 100.0):
        r = probe(state, 20, 0.1, 8000, proj_grad_tol=pgt)
        rows.append(r)
        print(f"{r['N']:>4}{r['k0_scale']:>7}{r['max_iterations']:>9}"
              f"{str(r['converged_flag']):>7}{str(r['feasible']):>10}"
              f"{r['residual_physical']:>12.4e}{r['final_proj_grad_norm']:>12.4e}"
              f"{str(r['proj_grad_within_tol']):>8}{r['u0']:>10.3f}")

    out = PROJECT_ROOT / "results" / "convergence_blocker_probe.json"
    with open(out, "w") as fh:
        json.dump({"stiff_state": {"k": state["k"], "rho": state["rho"]},
                    "trust_region": TRUST_REGION, "soft_state_constraints": SOFT,
                    "rows": rows}, fh, indent=2, default=str)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
