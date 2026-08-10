"""
qp_conditioning_change_report.py
Evaluates the approved conditioning change (SNNMPCSolver._condition's row-norm
formula) against a no-preconditioning baseline AND the pre-change Jacobi
formula, on the same easy/stiff fixed states used throughout this work.

Per snn-qp-verification: every reported number is computed by mapping the
solver's raw output back to physical units and evaluating it against the
ORIGINAL, unscaled canonical QP (src/qp_builder.py, untouched) -- never by
trusting the solver's internal (conditioned-space) self-report alone. No new
output clipping is introduced; "applied_control" uses the SAME safety-filter
clip already in SNNMPCSolver.compute_control_action, reported honestly
alongside the raw (unclipped) value, not in place of it.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/qp_conditioning_change_report.py
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
import src.constants as const
from snn_opt import OptimizationProblem, SNNSolver

ROLLOUT_STEPS = 105
EASY_K = 3


def find_states():
    plant = AutoclavePlant(initial_temp=28.0)
    ctrl = MPCSolver(horizon=20, target_temp=120.0)
    current_Ta = 28.0
    current_state = plant.get_state()
    states = []
    for k in range(ROLLOUT_STEPS):
        x0 = current_state.copy()
        u_prev = current_Ta
        avg_T, avg_a = np.mean(x0[0:3]), np.mean(x0[7:10])
        Ap, _ = linearize(avg_T, avg_a, trust_region=True)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))
        states.append({"k": k, "x0": x0, "u_prev": u_prev, "rho": rho})
        current_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        current_state = plant.step(Ta_input=current_Ta)
    return states[EASY_K], max(states, key=lambda s: s["rho"])


def old_condition(H, g, C, d):
    """Pre-change row normalization (1/||C_row|| only) -- comparison only."""
    D = np.sqrt(np.maximum(np.diag(H), 1e-12))
    H_s = (H / D[:, None]) / D[None, :]
    g_s = g / D
    C_s = C / D[None, :]
    row_norms = np.maximum(np.linalg.norm(C_s, axis=1, keepdims=True), 1e-10)
    return H_s, g_s, C_s / row_norms, d / row_norms.squeeze(), D


def solve_reference_osqp_raw(H_raw, f_raw, A_ineq, b_ineq, N):
    U = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H_raw)) + f_raw @ U),
                       [A_ineq @ U <= b_ineq])
    prob.solve(solver=cp.OSQP)
    return (U.value if U.value is not None else None), prob.status


def evaluate_against_original(problem_raw, U_physical, u_prev, N):
    """Every reported metric is computed on the RAW/original canonical QP,
    from the mapped-back physical solution -- never on the conditioned
    (scaled) problem the solver actually iterated on."""
    objective_value = float(problem_raw.objective(U_physical))
    g_vals = problem_raw.constraint_values(U_physical)          # A_i z - b_i, raw units
    primal_residual = float(np.max(np.maximum(0.0, g_vals)))
    lb_viol = float(np.maximum(0.0, const.TA_MIN - U_physical).max())
    ub_viol = float(np.maximum(0.0, U_physical - const.TA_MAX).max())
    bound_violation = max(lb_viol, ub_viol)
    max_constraint_violation = primal_residual  # box rows are embedded in A_ineq too

    raw_control = float(U_physical[0])
    lo = max(const.TA_MIN, u_prev - const.TA_RATE_MAX)
    hi = min(const.TA_MAX, u_prev + const.TA_RATE_MAX)
    applied_control = float(np.clip(raw_control, lo, hi))       # EXISTING safety filter only
    n_clipped = int(abs(applied_control - raw_control) > 1e-6)

    return {
        "objective_value": objective_value,
        "primal_residual": primal_residual,
        "bound_violation": bound_violation,
        "maximum_constraint_violation": max_constraint_violation,
        "raw_control": raw_control,
        "applied_control": applied_control,
        "n_clipped_variables": n_clipped,
    }


def run_config(config_name, H_raw, f_raw, A_ineq, b_ineq, ctrl, u_prev, N, f_ref):
    C_raw, d_raw = A_ineq, -b_ineq
    problem_raw = OptimizationProblem(A=H_raw, b=f_raw, C=C_raw, d=d_raw)
    U_cold_physical = ctrl._warm_hold(u_prev)

    if config_name == "no_preconditioning":
        problem = OptimizationProblem(A=H_raw, b=f_raw, C=C_raw, d=d_raw)
        solver = SNNSolver(problem, ctrl.solver_config)
        result = solver.solve(U_cold_physical, verbose=False)
        U_physical = result.final_x  # no D to map back through
    else:
        if config_name == "jacobi_old":
            H_s, g_s, C_s, d_s, D = old_condition(H_raw, f_raw, C_raw, d_raw)
        elif config_name == "jacobi_new_candidate":
            H_s, g_s, C_s, d_s, D = ctrl._condition(H_raw, f_raw, C_raw, d_raw)  # production code
        else:
            raise ValueError(config_name)
        U_cold_scaled = U_cold_physical * D
        problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
        solver = SNNSolver(problem, ctrl.solver_config)
        result = solver.solve(U_cold_scaled, verbose=False)
        U_physical = result.final_x / D  # mapping back to physical variables

    metrics = evaluate_against_original(problem_raw, U_physical, u_prev, N)
    obj_gap = (None if f_ref is None
               else (metrics["objective_value"] - f_ref) / max(1.0, abs(f_ref)))

    return {
        "configuration": config_name,
        "converged": bool(result.converged),
        "iterations": int(result.iterations_used),
        "max_iterations": int(ctrl.solver_config.max_iterations),
        "objective_value": metrics["objective_value"],
        "objective_gap": obj_gap,
        "primal_residual": metrics["primal_residual"],
        "bound_violation": metrics["bound_violation"],
        "maximum_constraint_violation": metrics["maximum_constraint_violation"],
        "raw_control": metrics["raw_control"],
        "applied_control": metrics["applied_control"],
        "n_clipped_variables": metrics["n_clipped_variables"],
    }


def main():
    print("Generating fixed, deterministic test states...")
    easy, stiff = find_states()
    N = 20
    rows = []

    for label, state in (("easy", easy), ("stiff", stiff)):
        print(f"\n=== {label} (k={state['k']}, rho(Ap)={state['rho']:.4f}) ===")
        ctrl = SNNMPCSolver(horizon=N, target_temp=120.0)  # fresh instance per state
        qp = ctrl.build_qp(state["x0"], state["u_prev"])
        H_raw, f_raw, A_ineq, b_ineq = qp.H, qp.f, qp.A_ineq, qp.b_ineq

        U_ref, ref_status = solve_reference_osqp_raw(H_raw, f_raw, A_ineq, b_ineq, N)
        f_ref = None
        if U_ref is not None:
            problem_raw_ref = OptimizationProblem(A=H_raw, b=f_raw, C=A_ineq, d=-b_ineq)
            f_ref = float(problem_raw_ref.objective(U_ref))
        print(f"  reference (OSQP on raw canonical QP): status={ref_status}  f_ref={f_ref}")

        for config_name in ("no_preconditioning", "jacobi_old", "jacobi_new_candidate"):
            ctrl_run = SNNMPCSolver(horizon=N, target_temp=120.0)  # fresh, no warm-start leakage
            row = run_config(config_name, H_raw, f_raw, A_ineq, b_ineq, ctrl_run,
                              state["u_prev"], N, f_ref)
            row["state"] = label
            row["k"] = state["k"]
            row["rho_Ap"] = state["rho"]
            row["reference_status"] = ref_status
            rows.append(row)

    # ---- print compact table ----
    print("\n" + "=" * 150)
    header = (f"{'state':<7}{'configuration':<22}{'converged':<11}{'iters':>8}{'objective':>14}"
              f"{'obj_gap':>12}{'primal_res':>13}{'bound_viol':>12}{'max_viol':>12}"
              f"{'raw_ctrl':>10}{'appl_ctrl':>10}{'n_clip':>7}")
    print(header)
    for r in rows:
        obj_gap_str = "N/A" if r["objective_gap"] is None else f"{r['objective_gap']:.4f}"
        print(f"{r['state']:<7}{r['configuration']:<22}{str(r['converged']):<11}"
              f"{r['iterations']:>5d}/{r['max_iterations']:<2d}{r['objective_value']:>14.4e}"
              f"{obj_gap_str:>12}{r['primal_residual']:>13.4e}{r['bound_violation']:>12.4e}"
              f"{r['maximum_constraint_violation']:>12.4e}{r['raw_control']:>10.2f}"
              f"{r['applied_control']:>10.2f}{r['n_clipped_variables']:>7d}")
    print("=" * 150)

    # ---- accept/reject decision: jacobi_old (baseline) vs jacobi_new_candidate ----
    print("\n--- Accept/reject: jacobi_new_candidate vs jacobi_old, per state ---")
    decisions = {}
    for label in ("easy", "stiff"):
        old = next(r for r in rows if r["state"] == label and r["configuration"] == "jacobi_old")
        new = next(r for r in rows if r["state"] == label and r["configuration"] == "jacobi_new_candidate")
        iter_improved = new["iterations"] < old["iterations"]
        feas_worse = new["maximum_constraint_violation"] > old["maximum_constraint_violation"] * 1.01
        feas_better = new["maximum_constraint_violation"] < old["maximum_constraint_violation"] * 0.99
        gap_worse = (new["objective_gap"] is not None and old["objective_gap"] is not None
                     and new["objective_gap"] > old["objective_gap"] + 1e-6)
        reject = iter_improved and (feas_worse or gap_worse)
        decisions[label] = {
            "iterations_old": old["iterations"], "iterations_new": new["iterations"],
            "max_violation_old": old["maximum_constraint_violation"],
            "max_violation_new": new["maximum_constraint_violation"],
            "feasibility_improved": feas_better, "feasibility_worsened": feas_worse,
            "objective_gap_old": old["objective_gap"], "objective_gap_new": new["objective_gap"],
            "reject_per_stated_rule": reject,
        }
        print(f"  {label}: iters {old['iterations']}->{new['iterations']}  "
              f"max_violation {old['maximum_constraint_violation']:.4e}->{new['maximum_constraint_violation']:.4e}  "
              f"feasibility_improved={feas_better}  reject_per_stated_rule={reject}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "qp_conditioning_change_report.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "decisions": decisions}, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
