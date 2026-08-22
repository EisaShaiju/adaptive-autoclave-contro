"""
optimum_agreement_probe.py
Decisive test following tools/convergence_blocker_probe.py.

That probe found that with soft state constraints the SNN reaches a FEASIBLE
point on the stiff exotherm QP (residual ~1e-3 vs tol 1e-2) and returns a
stable u0 ~126 degC regardless of horizon or step size -- yet the formal
`converged` flag never fires because the projected-gradient norm sits at
~1.7e10, twelve orders of magnitude above the ABSOLUTE tolerance (5e-2), and
does not shrink with a 6x larger iteration budget.

That pattern has two possible explanations, which this probe separates:

  (a) The SNN is finding the correct optimum, and snn_opt's absolute
      projected-gradient stopping test is simply not scale-invariant on a
      problem whose gradient scale is ~1e10 -- i.e. a mis-specified FLAG on a
      correct ANSWER.
  (b) The SNN is stalling at a feasible but sub-optimal point -- a genuinely
      wrong answer that happens to satisfy the constraints.

Test: solve the IDENTICAL soft QP with OSQP as reference and compare the
applied move u0 and the objective value. Also report the projected-gradient
norm RELATIVE to the problem's own gradient scale, to show whether a
scale-invariant criterion would have fired.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/optimum_agreement_probe.py
"""
from pathlib import Path
import sys
import json

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
TRUST_REGION = False
SOFT = True


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


def run(state, N, k0_scale):
    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0, trust_region=TRUST_REGION,
                        soft_state_constraints=SOFT, k0_scale=k0_scale)
    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H, f, A, b = qp.H, qp.f, qp.A_ineq, qp.b_ineq
    problem_raw = OptimizationProblem(A=H, b=f, C=A, d=-b)

    # --- reference optimum on the ORIGINAL (unconditioned) soft QP ---
    U = cp.Variable(H.shape[0])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H)) + f @ U),
                       [A @ U <= b])
    prob.solve(solver=cp.OSQP, max_iter=200000, eps_abs=1e-8, eps_rel=1e-8)
    u_ref = U.value
    f_ref = float(problem_raw.objective(u_ref)) if u_ref is not None else None
    res_ref = float(problem_raw.max_violation(u_ref)) if u_ref is not None else None

    # --- SNN on the conditioned arrays, cold start ---
    H_s, g_s, C_s, d_s, D = ctrl._condition(H, f, A, -b)
    U_cold = ctrl._warm_hold(state["u_prev"], H.shape[0])
    solver = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s), ctrl.solver_config)
    result = solver.solve(U_cold * D, verbose=False)
    U_phys = result.final_x / D
    f_snn = float(problem_raw.objective(U_phys))
    res_snn = float(problem_raw.max_violation(U_phys))

    # Scale-invariant view of the projected-gradient test: compare the norm
    # against the problem's own gradient scale at the returned point.
    grad_scale = float(np.linalg.norm(H_s @ result.final_x + g_s))
    pg = float(result.final_proj_grad_norm)
    rel_pg = pg / max(grad_scale, 1e-30)

    obj_gap = None
    u0_gap = None
    if f_ref is not None:
        obj_gap = (f_snn - f_ref) / max(1.0, abs(f_ref))
        u0_gap = abs(float(U_phys[0]) - float(u_ref[0]))

    return {
        "N": N, "k0_scale": k0_scale,
        "osqp_status": prob.status,
        "u0_ref": None if u_ref is None else float(u_ref[0]),
        "u0_snn": float(U_phys[0]),
        "u0_gap": u0_gap,
        "obj_ref": f_ref, "obj_snn": f_snn, "obj_gap": obj_gap,
        "residual_ref": res_ref, "residual_snn": res_snn,
        "snn_feasible": res_snn <= ctrl.solver_config.convergence.feasibility_tol,
        "proj_grad_abs": pg, "grad_scale": grad_scale, "proj_grad_relative": rel_pg,
        "converged_flag": bool(result.converged),
        "iterations": int(result.iterations_used),
    }


def main():
    state = find_stiff_state()
    print(f"Stiff exotherm state: k={state['k']}, rho(Ap)={state['rho']:.4f}")
    print(f"trust_region={TRUST_REGION} (model-identical), soft_state_constraints={SOFT}\n")

    rows = []
    hdr = (f"{'N':>4}{'k0':>7}{'osqp':>12}{'u0_ref':>10}{'u0_snn':>10}{'u0_gap':>10}"
           f"{'obj_gap':>12}{'res_snn':>11}{'feas':>7}{'rel_pg':>11}{'flag':>7}")
    print(hdr)
    print("-" * len(hdr))
    # k0_scale=0.5 is SNNMPCSolver's constructor default; 0.1 is the value the
    # recommended configuration actually runs (docs/PHASE4_VALIDATION_REPORT.md
    # section 6). Both are swept because the relative projected-gradient norm at
    # the returned point depends on the step size -- 0.449 at k0=0.1 vs 0.670 at
    # k0=0.5 for N=20 -- so any quoted relative norm is meaningless without its
    # k0_scale. All three fail the 0.05 threshold at the working horizon.
    for N in (20, 10, 5):
        for k0 in (0.5, 0.1, 0.05):
            r = run(state, N, k0)
            rows.append(r)

            def fmt(v, s):
                return "N/A" if v is None else f"{v:{s}}"

            print(f"{r['N']:>4}{r['k0_scale']:>7}{r['osqp_status']:>12}"
                  f"{fmt(r['u0_ref'], '10.3f')}{r['u0_snn']:>10.3f}{fmt(r['u0_gap'], '10.4f')}"
                  f"{fmt(r['obj_gap'], '12.3e')}{r['residual_snn']:>11.3e}"
                  f"{str(r['snn_feasible']):>7}{r['proj_grad_relative']:>11.3e}"
                  f"{str(r['converged_flag']):>7}")

    print("\nInterpretation key:")
    print("  u0_gap small + obj_gap small + feasible  -> correct ANSWER, mis-specified FLAG (case a)")
    print("  u0_gap or obj_gap large                  -> genuinely sub-optimal stall  (case b)")
    print("  rel_pg = ||proj_grad|| / ||grad|| at the returned point; snn_opt tests the")
    print("  ABSOLUTE norm against 5e-2, which is not scale-invariant.")

    out = PROJECT_ROOT / "results" / "optimum_agreement_probe.json"
    with open(out, "w") as fh:
        json.dump({"stiff_state": {"k": state["k"], "rho": state["rho"]},
                    "trust_region": TRUST_REGION, "soft_state_constraints": SOFT,
                    "rows": rows}, fh, indent=2, default=str)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
