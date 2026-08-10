"""
snn_solve_instrumentation.py
Instrumentation harness for the SNN-QP solver (see
docs/PHASE4_VALIDATION_REPORT.md §5). Does NOT modify snn_opt's solver internals, SNNMPCSolver's _condition,
or any update/projection equation -- it wraps existing public entry points
(SNNMPCSolver.build_qp, SNNMPCSolver._condition, snn_opt.SNNSolver.solve) and
independently recomputes every diagnostic from the returned SolverResult
rather than trusting the solver's self-report alone.

Uses the canonical QP builder (src/qp_builder.py) from the previous stage, so
both the SNN's own construction and the CVXPY reference solve start from the
identical (H, f, A_ineq, b_ineq).

Two FIXED, deterministic test states (no RNG anywhere in this repo):
  - "easy_well_conditioned": step k=3 of a deterministic open-loop-ish CVXPY
    rollout (early heat-up, well before the exotherm).
  - "stiff_exotherm": the step with maximum rho(Ap) (trust_region=True) along
    that same deterministic rollout -- the gelation peak.
Each test case is solved from a COLD start (U_warm=None-equivalent), NOT the
receding-horizon warm start used in the live closed loop, so results are a
reproducible, isolated unit test rather than a mid-trajectory snapshot (see
docs/PHASE4_VALIDATION_REPORT.md §2 for why this matters).

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/snn_solve_instrumentation.py
"""
from pathlib import Path
import sys
import json
import time

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
import src.constants as const
from snn_opt import OptimizationProblem, SNNSolver

ROLLOUT_STEPS = 105
EASY_K = 3


def find_test_states():
    """Deterministic rollout (CVXPY controller, no disturbance, no RNG) used
    only to generate two FIXED, reproducible plant states. Returns (easy, stiff)."""
    plant = AutoclavePlant(initial_temp=28.0)
    ctrl = MPCSolver(horizon=20, target_temp=120.0)
    current_Ta = 28.0
    current_state = plant.get_state()
    states = []
    for k in range(ROLLOUT_STEPS):
        x0 = current_state.copy()
        u_prev = current_Ta
        avg_T, avg_a = np.mean(x0[0:3]), np.mean(x0[7:10])
        Ap, _ = linearize(avg_T, avg_a, trust_region=True)  # SNN's own model, for state selection
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))
        states.append({"k": k, "x0": x0, "u_prev": u_prev, "rho": rho})
        current_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        current_state = plant.step(Ta_input=current_Ta)
    easy = states[EASY_K]
    stiff = max(states, key=lambda s: s["rho"])
    return easy, stiff


def solve_reference_osqp(H_s, g_s, C_s, d_s):
    """Ground truth: OSQP on the IDENTICAL scaled arrays the SNN solves."""
    n = H_s.shape[0]
    U = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H_s)) + g_s @ U),
                       [C_s @ U + d_s <= 0])
    prob.solve(solver=cp.OSQP)
    return (U.value if U.value is not None else None), prob.status


def classify_convergence(result, feasibility_tol, max_violation_scaled):
    """Explicit convergence definition, independent of the solver's bookkeeping:
      1. feasibility: max_violation_scaled <= feasibility_tol (recomputed here,
         not read from result.constraint_violations, to catch any bookkeeping bug)
      2. the solver's own stopping condition fired (result.converged)
      3. iteration-limit termination is EXPLICITLY forced to non-convergence,
         even if some other signal looked fine.
    """
    hit_iter_cap = (result.convergence_reason == "max_iterations")
    feasible = max_violation_scaled <= feasibility_tol
    verified_converged = bool(result.converged) and feasible and not hit_iter_cap
    return verified_converged, hit_iter_cap, feasible


def run_instrumented_solve(controller, x0, u_prev, label, k_index, rho):
    qp = controller.build_qp(x0, u_prev)
    H_raw, f_raw, A_raw, b_raw = qp.H, qp.f, qp.A_ineq, qp.b_ineq
    C_raw, d_raw = A_raw, -b_raw  # canonical A_ineq@z<=b_ineq -> Cz+d<=0 (C=A_ineq, d=-b_ineq)

    H_s, g_s, C_s, d_s, D = controller._condition(H_raw, f_raw, C_raw, d_raw)

    cond_H_raw = float(np.linalg.cond(H_raw))
    cond_H_scaled = float(np.linalg.cond(H_s))

    U_cold = controller._warm_hold(u_prev)          # cold start -- see module docstring
    U_warm_scaled = U_cold * D

    problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
    solver = SNNSolver(problem, controller.solver_config)
    step_size_k0 = float(solver._k0)

    t0 = time.time()
    result = solver.solve(U_warm_scaled, verbose=False)
    solve_ms = (time.time() - t0) * 1000.0

    # ---- Independent residual recomputation (do not trust result.* alone) ----
    g_vals = C_s @ result.final_x + d_s
    ineq_resid_vec = np.maximum(0.0, g_vals)
    max_violation_scaled = float(ineq_resid_vec.max()) if ineq_resid_vec.size else 0.0
    active_mask = ineq_resid_vec > controller.solver_config.constraint_tol
    n_violated = int(np.sum(active_mask))
    mean_active_violation = float(ineq_resid_vec[active_mask].mean()) if n_violated else 0.0
    matches_solver_reported = bool(np.isclose(
        max_violation_scaled, float(np.max(result.constraint_violations)), atol=1e-9))

    objective_value_scaled = float(problem.objective(result.final_x))
    matches_solver_reported_obj = bool(np.isclose(objective_value_scaled, result.final_objective, atol=1e-9))

    verified_converged, hit_iter_cap, feasible = classify_convergence(
        result, controller.solver_config.convergence.feasibility_tol, max_violation_scaled)

    # ---- Reference solve on the identical scaled arrays ----
    u_ref_vec, ref_status = solve_reference_osqp(H_s, g_s, C_s, d_s)
    if u_ref_vec is not None:
        f_ref = float(0.5 * u_ref_vec @ H_s @ u_ref_vec + g_s @ u_ref_vec)
        obj_gap = (objective_value_scaled - f_ref) / max(1.0, abs(f_ref))
        u0_ref_physical = float(u_ref_vec[0] / D[0])
    else:
        f_ref, obj_gap, u0_ref_physical = None, None, None

    # ---- Unscale to physical Ta units ----
    U_sol_physical = result.final_x / D
    lb_viol = float(np.maximum(0.0, const.TA_MIN - U_sol_physical).max())
    ub_viol = float(np.maximum(0.0, U_sol_physical - const.TA_MAX).max())
    n_components_outside_box = int(np.sum(
        (U_sol_physical < const.TA_MIN - 1e-9) | (U_sol_physical > const.TA_MAX + 1e-9)))

    u0_raw_physical = float(U_sol_physical[0])
    lo = max(const.TA_MIN, u_prev - const.TA_RATE_MAX)
    hi = min(const.TA_MAX, u_prev + const.TA_RATE_MAX)
    u0_applied = float(np.clip(u0_raw_physical, lo, hi))
    n_clipped_variables = int(abs(u0_applied - u0_raw_physical) > 1e-6)
    applied_move_gap_vs_reference = (None if u0_ref_physical is None
                                      else abs(u0_raw_physical - u0_ref_physical))

    record = {
        "label": label, "k_index": k_index, "rho_Ap": rho,
        "trust_region": qp.linearization["trust_region"],
        "qp_fingerprint": qp.fingerprint(),
        # --- required field list ---
        "converged": bool(result.converged),
        "termination_reason": result.convergence_reason,
        "iteration_count": int(result.iterations_used),
        "max_iteration_count": int(controller.solver_config.max_iterations),
        "objective_value": objective_value_scaled,
        "raw_decision_vector": result.final_x.tolist(),          # solver-native (scaled) space
        "applied_decision_vector": U_sol_physical.tolist(),       # unscaled, physical Ta units (full horizon)
        "applied_u0": u0_applied,
        "inequality_residual_max": max_violation_scaled,
        "inequality_residual_mean_active": mean_active_violation,
        "inequality_residual_n_violated": n_violated,
        "equality_residual": None,  # not applicable: this QP class has no A_eq/b_eq
        "lower_bound_violation": lb_viol,   # physical degC, vs const.TA_MIN
        "upper_bound_violation": ub_viol,   # physical degC, vs const.TA_MAX
        "maximum_constraint_violation": max_violation_scaled,
        "objective_gap_vs_cvxpy": obj_gap,
        "n_clipped_variables": n_clipped_variables,
        "solver_step_size_k0": step_size_k0,
        "preconditioning_mode": "jacobi_diagonal (unchanged this stage)",
        "scaling_mode": "D=sqrt(diag(H)); H_s=H/(D D^T); g_s=g/D; C_s=(C/D) row-normalized; d_s=d row-normalized",
        # --- explicit convergence classification ---
        "feasibility_tol_scaled": controller.solver_config.convergence.feasibility_tol,
        "feasible_within_tol": feasible,
        "hit_iteration_cap": hit_iter_cap,
        "verified_converged": verified_converged,
        # --- cross-checks / extra context ---
        "n_projections": int(result.n_projections),
        "final_proj_grad_norm": float(result.final_proj_grad_norm),
        "n_components_outside_box_unclipped": n_components_outside_box,
        "reference_objective_scaled": f_ref,
        "reference_status": ref_status,
        "applied_move_gap_vs_reference_degC": applied_move_gap_vs_reference,
        "matches_solver_reported_violation": matches_solver_reported,
        "matches_solver_reported_objective": matches_solver_reported_obj,
        "cond_H_raw": cond_H_raw,
        "cond_H_scaled": cond_H_scaled,
        "solve_time_ms": solve_ms,
        "backend": controller.solver_config.backend,
        "x0": np.asarray(x0).tolist(),
        "u_prev": float(u_prev),
    }
    return record


def print_summary(r):
    print(f"\n--- {r['label']} (k={r['k_index']}, rho(Ap)={r['rho_Ap']:.4f}) ---")
    print(f"  converged={r['converged']}  verified_converged={r['verified_converged']}  "
          f"reason={r['termination_reason']}  iters={r['iteration_count']}/{r['max_iteration_count']}")
    print(f"  feasible_within_tol={r['feasible_within_tol']}  "
          f"max_constraint_violation(scaled)={r['maximum_constraint_violation']:.6e}  "
          f"n_violated_rows={r['inequality_residual_n_violated']}")
    print(f"  objective(scaled)={r['objective_value']:.6e}  ref(OSQP)={r['reference_objective_scaled']}  "
          f"obj_gap={r['objective_gap_vs_cvxpy']}  ref_status={r['reference_status']}")
    print(f"  applied_u0={r['applied_u0']:.4f} degC  u0_raw={r['applied_decision_vector'][0]:.4f} degC  "
          f"n_clipped_variables={r['n_clipped_variables']}  applied_move_gap_vs_ref={r['applied_move_gap_vs_reference_degC']}")
    print(f"  lower_bound_violation={r['lower_bound_violation']:.4f} degC  "
          f"upper_bound_violation={r['upper_bound_violation']:.4f} degC  "
          f"n_components_outside_box={r['n_components_outside_box_unclipped']}")
    print(f"  cond(H_raw)={r['cond_H_raw']:.3e}  cond(H_scaled)={r['cond_H_scaled']:.3e}  "
          f"step_size_k0={r['solver_step_size_k0']:.3e}  final_proj_grad_norm={r['final_proj_grad_norm']:.3e}")
    print(f"  n_projections={r['n_projections']}  solve_time_ms={r['solve_time_ms']:.2f}  "
          f"backend={r['backend']}  qp_fingerprint={r['qp_fingerprint']}")
    if not r["feasible_within_tol"]:
        print("  ** INFEASIBLE at tolerance -- objective_gap/applied_move_gap NOT evidence of success **")


def main():
    print("Generating fixed, deterministic test states...")
    easy, stiff = find_test_states()
    print(f"  easy_well_conditioned: k={easy['k']}, rho(Ap)={easy['rho']:.4f}")
    print(f"  stiff_exotherm:        k={stiff['k']}, rho(Ap)={stiff['rho']:.4f}")

    # Fresh SNNMPCSolver per test case: no warm-start cross-contamination.
    ctrl_easy = SNNMPCSolver(horizon=20, target_temp=120.0)
    ctrl_stiff = SNNMPCSolver(horizon=20, target_temp=120.0)

    easy_record = run_instrumented_solve(ctrl_easy, easy["x0"], easy["u_prev"],
                                          "easy_well_conditioned", easy["k"], easy["rho"])
    stiff_record = run_instrumented_solve(ctrl_stiff, stiff["x0"], stiff["u_prev"],
                                           "stiff_exotherm", stiff["k"], stiff["rho"])

    print_summary(easy_record)
    print_summary(stiff_record)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "snn_solve_diagnostics.json"
    manifest = {
        "convergence_definition": {
            "feasibility": "max(0, C_s@x+d_s).max() <= ConvergenceConfig.feasibility_tol (scaled space)",
            "solver_stopping_condition": "result.converged True (patience-gated objective-plateau AND "
                                          "projected-gradient-norm checks, gated by feasibility, per "
                                          "snn_opt.SNNSolver._check_convergence)",
            "iteration_limit_rule": "convergence_reason == 'max_iterations' forces verified_converged=False "
                                     "regardless of any other signal",
        },
        "residual_formulas": {
            "inequality_residual": "r(x) = max(0, C_s @ x + d_s), elementwise, scaled space",
            "equality_residual": "not applicable -- OptimizationProblem/CanonicalQP has no A_eq/b_eq "
                                  "(x0 elimination is implicit in the condensation, not an explicit constraint)",
            "lower_bound_violation": "max(0, TA_MIN - U_physical), elementwise over the unscaled horizon",
            "upper_bound_violation": "max(0, U_physical - TA_MAX), elementwise over the unscaled horizon",
            "maximum_constraint_violation": "max(inequality_residual) -- box/slew/gradient rows are all "
                                             "embedded in C_s, so this already subsumes box violations",
            "objective_gap_vs_cvxpy": "(f_snn - f_osqp_ref) / max(1, |f_osqp_ref|), both evaluated on the "
                                       "IDENTICAL scaled (H_s,g_s,C_s,d_s) via OptimizationProblem.objective()",
        },
        "cases": [easy_record, stiff_record],
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDiagnostics written to {out_path}")


if __name__ == "__main__":
    main()
