"""
final_controlled_comparison.py
Final controlled closed-loop comparison. One shared harness drives BOTH
controllers against identical initial state, plant parameters, reference
trajectory, disturbance schedule (single, stated convention -- disturbance
BEFORE control-compute, applied identically to both branches), sampling time,
horizon, canonical per-step QP construction, physical constraints, variable
ordering, and (documented) scaling metadata.

Both controllers' per-step solves are REIMPLEMENTED here using the same
public building blocks each production method uses (build_qp, _condition for
SNN, identical OSQP/SNNSolver calls) -- not a different code path, just an
instrumented mirror that exposes the full decision vector and objective value
that compute_control_action()'s return signature doesn't. Every metric is
computed by mapping the solver's output back to physical units and evaluating
it against the ORIGINAL canonical QP -- never trusted from internal
(conditioned-space) self-reports alone. No new output clipping is added; the
existing safety-filter clip in SNNMPCSolver.compute_control_action is
reproduced exactly here and reported honestly.

Does NOT modify src/dynamics.py, src/qp_builder.py, src/mpc_cvxpy_controller.py,
src/snn_mpc_controller.py, or snn_opt.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/final_controlled_comparison.py
"""
from pathlib import Path
import sys
import csv
import json
import subprocess
from datetime import datetime

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
import src.constants as const
from snn_opt import OptimizationProblem, SNNSolver

HORIZON = 20
TARGET_TEMP = 120.0
INITIAL_TEMP = 28.0
TIME_STEPS = 160
FEASIBILITY_TOL = 1e-2  # matches SNNMPCSolver's ConvergenceConfig.feasibility_tol

# Shared problem configuration. Every field here is applied IDENTICALLY to both
# controllers -- that is the point. Overridable from the command line:
#   python tools/final_controlled_comparison.py --horizon 5 --soft --k0-scale 0.1
CONFIG = {
    "trust_region": False,           # identical prediction model on both sides
    "soft_state_constraints": False,
    "k0_scale": 0.5,
    "label": "baseline",
}


def make_controllers(horizon):
    """Both controllers built from the SAME configuration dict."""
    ctrl_cvx = MPCSolver(horizon=horizon, target_temp=TARGET_TEMP,
                         trust_region=CONFIG["trust_region"],
                         soft_state_constraints=CONFIG["soft_state_constraints"])
    ctrl_snn = SNNMPCSolver(horizon=horizon, target_temp=TARGET_TEMP,
                            trust_region=CONFIG["trust_region"],
                            soft_state_constraints=CONFIG["soft_state_constraints"],
                            k0_scale=CONFIG["k0_scale"])
    return ctrl_cvx, ctrl_snn


# ======================================================================
# Provenance
# ======================================================================
def capture_provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        dirty_files = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT).decode().strip()
    except Exception as exc:
        commit, dirty_files = f"unavailable: {exc}", None

    import cvxpy as _cp
    import numpy as _np
    import snn_opt as _snn_opt
    kernel_available = False
    has_openmp = None
    try:
        from snn_opt import _kernel
        kernel_available = True
        has_openmp = bool(getattr(_kernel, "HAS_OPENMP", False))
    except ImportError:
        pass

    ctrl_snn_probe = SNNMPCSolver(horizon=HORIZON, target_temp=TARGET_TEMP)
    ctrl_cvx_probe = MPCSolver(horizon=HORIZON, target_temp=TARGET_TEMP)

    return {
        "git_commit": commit,
        "git_working_tree_dirty_files": dirty_files.splitlines() if dirty_files else [],
        "cvxpy_version": _cp.__version__,
        "numpy_version": _np.__version__,
        "snn_opt_version": getattr(_snn_opt, "__version__", "unknown"),
        "snn_opt_compiled_kernel_available": kernel_available,
        "snn_opt_kernel_has_openmp": has_openmp,
        "random_seed": "N/A -- no RNG anywhere in plant/controllers (deterministic given config); "
                        "confirmed by inspection, not assumed",
        "sampling_time_TE_seconds": const.TE,
        "horizon_N": HORIZON,
        "target_temp_degC": TARGET_TEMP,
        "initial_temp_degC": INITIAL_TEMP,
        "physical_constraints": {
            "TA_MIN": const.TA_MIN, "TA_MAX": const.TA_MAX,
            "TA_RATE_MAX": const.TA_RATE_MAX, "GRADIENT_MAX": const.GRADIENT_MAX,
        },
        "cost_weights_shared": {
            "Q_diag_cvx": ctrl_cvx_probe.Q_diag.tolist(), "Q_diag_snn": ctrl_snn_probe.Q_diag.tolist(),
            "R_val_cvx": ctrl_cvx_probe.R_val, "R_val_snn": ctrl_snn_probe.R_val,
            "S_val_cvx": ctrl_cvx_probe.S_val, "S_val_snn": ctrl_snn_probe.S_val,
        },
        "variable_ordering": "z = [Ta_0, ..., Ta_{N-1}] -- identical for both controllers (src/qp_builder.py)",
        "canonical_qp_scaling_metadata": "None -- the shared canonical QP (src/qp_builder.py) is unscaled; "
                                          "SNNMPCSolver._condition applies a downstream, adapter-specific "
                                          "Jacobi scaling AFTER extraction, not part of the shared canonical form. "
                                          "CVXPY adapter applies no scaling at all.",
        "snn_solver_config": {
            "k0": ctrl_snn_probe.solver_config.k0, "k0_scale": ctrl_snn_probe.solver_config.k0_scale,
            "projection_method": ctrl_snn_probe.solver_config.projection_method,
            "max_iterations": ctrl_snn_probe.solver_config.max_iterations,
            "max_projection_iters": ctrl_snn_probe.solver_config.max_projection_iters,
            "backend": ctrl_snn_probe.solver_config.backend,
            "convergence": {
                "enable_early_stopping": ctrl_snn_probe.solver_config.convergence.enable_early_stopping,
                "check_every": ctrl_snn_probe.solver_config.convergence.check_every,
                "min_iterations": ctrl_snn_probe.solver_config.convergence.min_iterations,
                "patience": ctrl_snn_probe.solver_config.convergence.patience,
                "obj_rel_tol": ctrl_snn_probe.solver_config.convergence.obj_rel_tol,
                "proj_grad_tol": ctrl_snn_probe.solver_config.convergence.proj_grad_tol,
                "feasibility_tol": ctrl_snn_probe.solver_config.convergence.feasibility_tol,
            },
        },
        "disturbance_convention": "disturbance-before-compute: injected into both plants identically, "
                                   "BEFORE either controller reads state / computes control, at the same "
                                   "step index.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ======================================================================
# Instrumented step functions (mirror compute_control_action exactly,
# same public building blocks, zero src changes)
# ======================================================================
def cvxpy_step(ctrl, x0, u_prev):
    qp = ctrl.build_qp(x0, u_prev)
    U = cp.Variable(qp.H.shape[0])   # N inputs, plus N slacks when softened
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(qp.H)) + qp.f @ U),
                       [qp.A_ineq @ U <= qp.b_ineq])
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
        status = prob.status
        if U.value is None:
            u0 = float(u_prev)
            objective = None
        else:
            u0 = float(U.value[0])
            objective = float(0.5 * U.value @ qp.H @ U.value + qp.f @ U.value)
    except Exception:
        status, u0, objective = "exception", float(u_prev), None

    return {
        "raw_control": u0, "applied_control": u0,  # CVXPY has no separate output-clip stage
        "objective": objective, "status": status,
        "n_clipped_variables": 0,  # no clipping mechanism exists in this controller
        "rho_Ap": float(np.max(np.abs(np.linalg.eigvals(qp.linearization["Ap"])))),
        "qp_fingerprint": qp.fingerprint(),
    }


def snn_step(ctrl, x0, u_prev):
    """Mirrors SNNMPCSolver.compute_control_action exactly, including updating
    ctrl.U_warm and ctrl.n_projection_active, so a sequence of calls across a
    closed loop behaves identically to calling compute_control_action directly."""
    qp = ctrl.build_qp(x0, u_prev)
    H_raw, g_raw, C_raw, d_raw = qp.H, qp.f, qp.A_ineq, -qp.b_ineq
    problem_raw = OptimizationProblem(A=H_raw, b=g_raw, C=C_raw, d=d_raw)

    H_s, g_s, C_s, d_s, D = ctrl._condition(H_raw, g_raw, C_raw, d_raw)

    if not (np.isfinite(H_s).all() and np.isfinite(g_s).all()):
        ctrl.U_warm = None
        u_out = float(np.clip(u_prev, const.TA_MIN, const.TA_MAX))
        return {
            "raw_control": u_out, "applied_control": u_out, "objective": None,
            "converged": False, "verified_converged": False, "iterations": 0,
            "max_iterations": ctrl.solver_config.max_iterations,
            "constraint_residual_physical": None, "constraint_residual_scaled": None,
            "bound_violation": None, "n_clipped_variables": 0, "n_projections": 0,
            "rho_Ap": float(np.max(np.abs(np.linalg.eigvals(qp.linearization["Ap"])))),
            "qp_fingerprint": qp.fingerprint(), "non_finite_conditioned_problem": True,
        }

    n_total = H_raw.shape[0]
    U_raw = (ctrl._warm_hold(u_prev, n_total)
             if (ctrl.U_warm is None or ctrl.U_warm.shape[0] != n_total)
             else ctrl.U_warm)
    U_warm_scaled = U_raw * D

    problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
    solver = SNNSolver(problem, ctrl.solver_config)

    try:
        result = solver.solve(U_warm_scaled, verbose=False)
        U_sol = result.final_x / D  # mapping back to physical variables

        u_raw = float(U_sol[0])
        lo = max(const.TA_MIN, u_prev - const.TA_RATE_MAX)
        hi = min(const.TA_MAX, u_prev + const.TA_RATE_MAX)
        u_out = float(np.clip(u_raw, lo, hi))
        n_clipped = 0
        if abs(u_out - u_raw) > 1e-6:
            ctrl.n_projection_active += 1
            n_clipped = 1

        ctrl.U_warm = ctrl._shift(U_sol)

        # ---- Evaluate the MAPPED-BACK solution against the ORIGINAL canonical QP ----
        objective_physical = float(problem_raw.objective(U_sol))
        constraint_residual_physical = float(problem_raw.max_violation(U_sol))
        constraint_residual_scaled = float(np.max(np.maximum(0.0, C_s @ result.final_x + d_s)))
        lb_viol = float(np.maximum(0.0, const.TA_MIN - U_sol).max())
        ub_viol = float(np.maximum(0.0, U_sol - const.TA_MAX).max())
        bound_violation = max(lb_viol, ub_viol)

        hit_iter_cap = (result.convergence_reason == "max_iterations")
        feasible = constraint_residual_scaled <= ctrl.solver_config.convergence.feasibility_tol
        verified_converged = bool(result.converged) and feasible and not hit_iter_cap

        return {
            "raw_control": u_raw, "applied_control": u_out, "objective": objective_physical,
            "converged": bool(result.converged), "verified_converged": verified_converged,
            "iterations": int(result.iterations_used), "max_iterations": ctrl.solver_config.max_iterations,
            "constraint_residual_physical": constraint_residual_physical,
            "constraint_residual_scaled": constraint_residual_scaled,
            "bound_violation": bound_violation, "n_clipped_variables": n_clipped,
            "n_projections": int(result.n_projections),
            "rho_Ap": float(np.max(np.abs(np.linalg.eigvals(qp.linearization["Ap"])))),
            "qp_fingerprint": qp.fingerprint(), "non_finite_conditioned_problem": False,
            "_H_s": H_s, "_g_s": g_s, "_C_s": C_s, "_d_s": d_s, "_D": D,  # for objective-gap reference solve
        }
    except Exception:
        ctrl.U_warm = None
        u_out = float(np.clip(u_prev, const.TA_MIN, const.TA_MAX))
        return {
            "raw_control": u_out, "applied_control": u_out, "objective": None,
            "converged": False, "verified_converged": False, "iterations": 0,
            "max_iterations": ctrl.solver_config.max_iterations,
            "constraint_residual_physical": None, "constraint_residual_scaled": None,
            "bound_violation": None, "n_clipped_variables": 0, "n_projections": 0,
            "rho_Ap": float(np.max(np.abs(np.linalg.eigvals(qp.linearization["Ap"])))),
            "qp_fingerprint": qp.fingerprint(), "non_finite_conditioned_problem": True,
        }


def reference_objective_gap(snn_result):
    """OSQP on the SNN's own identical scaled arrays -- ground truth for the
    objective-gap metric. Only meaningful/reported on feasible steps."""
    if snn_result.get("non_finite_conditioned_problem", True):
        return None
    H_s, g_s, C_s, d_s = snn_result["_H_s"], snn_result["_g_s"], snn_result["_C_s"], snn_result["_d_s"]
    n = H_s.shape[0]
    U = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H_s)) + g_s @ U), [C_s @ U + d_s <= 0])
    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        return None
    if U.value is None:
        return None
    return float(0.5 * U.value @ H_s @ U.value + g_s @ U.value)


# ======================================================================
# Scenario runner
# ======================================================================
def run_scenario(name, disturbance_step, disturbance_magnitude, time_steps,
                  horizon=None):
    horizon = HORIZON if horizon is None else horizon
    plant_cvx = AutoclavePlant(initial_temp=INITIAL_TEMP)
    plant_snn = AutoclavePlant(initial_temp=INITIAL_TEMP)
    ctrl_cvx, ctrl_snn = make_controllers(horizon)

    u_cvx = u_snn = INITIAL_TEMP
    rows = []

    print(f"Running scenario '{name}' ({time_steps} steps, disturbance_step={disturbance_step})...")
    for k in range(time_steps):
        if disturbance_step is not None and k == disturbance_step:
            for p in (plant_cvx, plant_snn):
                p.T_comp -= disturbance_magnitude
                p.T_tool -= disturbance_magnitude

        x_cvx = plant_cvx.get_state()
        x_snn = plant_snn.get_state()

        r_cvx = cvxpy_step(ctrl_cvx, x_cvx, u_cvx)
        r_snn = snn_step(ctrl_snn, x_snn, u_snn)

        obj_gap = None
        if (r_snn["constraint_residual_scaled"] is not None
                and r_snn["constraint_residual_scaled"] <= FEASIBILITY_TOL):
            f_ref = reference_objective_gap(r_snn)
            if f_ref is not None and r_snn["objective"] is not None:
                obj_gap = (r_snn["objective"] - f_ref) / max(1.0, abs(f_ref))

        u_cvx_prev, u_snn_prev = u_cvx, u_snn
        u_cvx = r_cvx["applied_control"]
        u_snn = r_snn["applied_control"]

        plant_cvx.step(Ta_input=u_cvx)
        plant_snn.step(Ta_input=u_snn)

        slew_saturated_cvx = abs(u_cvx - u_cvx_prev) >= const.TA_RATE_MAX - 1e-6
        slew_saturated_snn = abs(u_snn - u_snn_prev) >= const.TA_RATE_MAX - 1e-6

        rows.append({
            "scenario": name, "step": k,
            "Ta_cvx_raw": r_cvx["raw_control"], "Ta_cvx_applied": r_cvx["applied_control"],
            "Ta_snn_raw": r_snn["raw_control"], "Ta_snn_applied": r_snn["applied_control"],
            "control_diff_e_u": u_snn - u_cvx,
            "cvxpy_objective": r_cvx["objective"], "cvxpy_status": r_cvx["status"],
            "snn_objective": r_snn["objective"], "snn_converged": r_snn["converged"],
            "snn_verified_converged": r_snn["verified_converged"],
            "snn_iterations": r_snn["iterations"], "snn_max_iterations": r_snn["max_iterations"],
            "snn_constraint_residual": r_snn["constraint_residual_physical"],
            "snn_constraint_residual_scaled": r_snn["constraint_residual_scaled"],
            "snn_bound_violation": r_snn["bound_violation"],
            "snn_clipped": r_snn["n_clipped_variables"],
            "cvxpy_clipped": r_cvx["n_clipped_variables"],
            "objective_gap_snn_vs_reference": obj_gap,
            "Tc1_cvx": float(x_cvx[0]), "Tc2_cvx": float(x_cvx[1]), "Tc3_cvx": float(x_cvx[2]),
            "alpha1_cvx": float(x_cvx[7]), "alpha2_cvx": float(x_cvx[8]), "alpha3_cvx": float(x_cvx[9]),
            "Tc1_snn": float(x_snn[0]), "Tc2_snn": float(x_snn[1]), "Tc3_snn": float(x_snn[2]),
            "alpha1_snn": float(x_snn[7]), "alpha2_snn": float(x_snn[8]), "alpha3_snn": float(x_snn[9]),
            "rho_Ap_cvx": r_cvx["rho_Ap"], "rho_Ap_snn": r_snn["rho_Ap"],
            "qp_fingerprint_cvx": r_cvx["qp_fingerprint"], "qp_fingerprint_snn": r_snn["qp_fingerprint"],
            "slew_saturated_cvx": slew_saturated_cvx, "slew_saturated_snn": slew_saturated_snn,
        })
    return rows


# ======================================================================
# Aggregate metrics
# ======================================================================
def compute_aggregate_metrics(rows, label):
    e_u = np.array([r["control_diff_e_u"] for r in rows])
    traj_diff = np.array([
        np.linalg.norm([
            r["Tc1_cvx"] - r["Tc1_snn"], r["Tc2_cvx"] - r["Tc2_snn"], r["Tc3_cvx"] - r["Tc3_snn"],
            r["alpha1_cvx"] - r["alpha1_snn"], r["alpha2_cvx"] - r["alpha2_snn"], r["alpha3_cvx"] - r["alpha3_snn"],
        ]) for r in rows
    ])
    residuals = np.array([r["snn_constraint_residual"] for r in rows if r["snn_constraint_residual"] is not None])
    n_clipped = sum(r["snn_clipped"] for r in rows)
    n_converged_raw = sum(1 for r in rows if r["snn_converged"])
    n_verified_converged = sum(1 for r in rows if r["snn_verified_converged"])
    obj_gaps = [r["objective_gap_snn_vs_reference"] for r in rows if r["objective_gap_snn_vs_reference"] is not None]

    n_heatup_slew_saturated_both = sum(
        1 for r in rows if r["slew_saturated_cvx"] and r["slew_saturated_snn"]
    )

    return {
        "label": label, "n_steps": len(rows),
        "rms_applied_control_difference": float(np.sqrt(np.mean(e_u ** 2))),
        "rms_trajectory_difference": float(np.sqrt(np.mean(traj_diff ** 2))),
        "max_abs_control_difference": float(np.max(np.abs(e_u))),
        "max_abs_trajectory_difference": float(np.max(traj_diff)),
        "snn_convergence_rate_raw": n_converged_raw / len(rows),
        "snn_convergence_rate_verified": n_verified_converged / len(rows),
        "max_constraint_residual": float(np.max(residuals)) if residuals.size else None,
        "n_clipped_outputs": int(n_clipped), "pct_clipped_outputs": 100.0 * n_clipped / len(rows),
        "cvxpy_n_clipped_outputs": 0,  # no clipping mechanism exists in the CVXPY controller
        "n_feasible_steps_for_objective_gap": len(obj_gaps),
        "pct_feasible_steps_for_objective_gap": 100.0 * len(obj_gaps) / len(rows),
        "mean_objective_gap_on_feasible_steps": (float(np.mean(obj_gaps)) if obj_gaps else None),
        "max_abs_objective_gap_on_feasible_steps": (float(np.max(np.abs(obj_gaps))) if obj_gaps else None),
        "n_steps_both_slew_saturated": n_heatup_slew_saturated_both,
        "pct_steps_both_slew_saturated": 100.0 * n_heatup_slew_saturated_both / len(rows),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--trust-region", action="store_true")
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--k0-scale", type=float, default=0.5)
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    CONFIG["trust_region"] = args.trust_region
    CONFIG["soft_state_constraints"] = args.soft
    CONFIG["k0_scale"] = args.k0_scale
    CONFIG["horizon"] = args.horizon
    CONFIG["label"] = args.label or (
        f"N{args.horizon}_{'soft' if args.soft else 'hard'}"
        f"_{'tr' if args.trust_region else 'notr'}_k{args.k0_scale}")
    print(f"CONFIG: {CONFIG}\n")

    provenance = capture_provenance()
    provenance["shared_configuration"] = dict(CONFIG)

    rows_nominal = run_scenario("nominal_heatup", disturbance_step=None,
                                 disturbance_magnitude=0.0, time_steps=TIME_STEPS,
                                 horizon=args.horizon)
    rows_disturbance = run_scenario("disturbance_step60", disturbance_step=60,
                                     disturbance_magnitude=15.0, time_steps=TIME_STEPS,
                                     horizon=args.horizon)

    # Scenario 3: stiff-exotherm window, extracted from the nominal run (NOT a
    # separate simulation -- see report.md methodology). Window = peak rho(Ap)
    # +/- margin, using the CVXPY branch's rho as the reference (documented).
    peak_k = max(range(len(rows_nominal)), key=lambda i: rows_nominal[i]["rho_Ap_cvx"])
    win_lo, win_hi = max(0, peak_k - 10), min(len(rows_nominal), peak_k + 21)
    rows_stiff_window = rows_nominal[win_lo:win_hi]
    print(f"\nStiff-exotherm window: steps [{win_lo}, {win_hi}) of nominal_heatup, "
          f"peak at k={peak_k} (rho(Ap)_cvx={rows_nominal[peak_k]['rho_Ap_cvx']:.4f})")

    metrics_nominal = compute_aggregate_metrics(rows_nominal, "nominal_heatup")
    metrics_disturbance = compute_aggregate_metrics(rows_disturbance, "disturbance_step60")
    metrics_stiff = compute_aggregate_metrics(rows_stiff_window,
                                               f"stiff_exotherm_window (steps {win_lo}-{win_hi - 1} of nominal_heatup)")

    # Heat-up-saturation caveat: fraction of the FIRST 60 steps of nominal_heatup
    # where BOTH controllers are pinned at the slew limit.
    heatup_rows = rows_nominal[:60]
    n_both_saturated_heatup = sum(1 for r in heatup_rows if r["slew_saturated_cvx"] and r["slew_saturated_snn"])
    pct_both_saturated_heatup = 100.0 * n_both_saturated_heatup / len(heatup_rows)

    # ---- output directory: never overwrite prior results ----
    short_commit = provenance["git_commit"][:8] if len(provenance["git_commit"]) >= 8 else "unknown"
    run_id = f"{short_commit}_{CONFIG['label']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = PROJECT_ROOT / "results" / "final_comparison" / run_id
    suffix = 0
    base_out_dir = out_dir
    while out_dir.exists():
        suffix += 1
        out_dir = Path(f"{base_out_dir}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"\nOutput directory: {out_dir}")

    # ---- per_step_metrics.csv ----
    all_rows = rows_nominal + rows_disturbance
    csv_fields = [k for k in all_rows[0].keys()]
    csv_path = out_dir / "per_step_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"Wrote {csv_path} ({len(all_rows)} rows)")

    # ---- summary.json ----
    summary = {
        "provenance": provenance,
        "scenarios": {
            "nominal_heatup": metrics_nominal,
            "disturbance_step60": metrics_disturbance,
            "stiff_exotherm_window": metrics_stiff,
        },
        "heatup_slew_saturation_caveat": {
            "n_steps_checked": len(heatup_rows),
            "n_steps_both_controllers_slew_saturated": n_both_saturated_heatup,
            "pct_steps_both_controllers_slew_saturated": pct_both_saturated_heatup,
            "interpretation": "If this percentage is high, applied-control agreement during heat-up is "
                               "expected from BOTH controllers being pinned at TA_RATE_MAX independent of "
                               "QP equivalence, and must NOT be read as evidence of solver equivalence.",
        },
        "definitions": {
            "e_u(k)": "u_SNN_applied(k) - u_CVXPY_applied(k)",
            "trajectory_difference(k)": "Euclidean norm of [Tc1,Tc2,Tc3,alpha1,alpha2,alpha3]_cvx minus "
                                         "the same vector from the SNN branch, at step k",
            "snn_constraint_residual": "max(0, A_ineq@U_sol - b_ineq) evaluated on the mapped-back physical "
                                        "solution against the ORIGINAL (unconditioned) canonical QP",
            "objective_gap_snn_vs_reference": "(f_snn - f_osqp_ref)/max(1,|f_osqp_ref|), both on SNN's own "
                                               "scaled arrays; computed and reported ONLY on steps where "
                                               "snn_constraint_residual_scaled <= feasibility_tol",
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    # ---- plot (matplotlib is an existing repo dependency, Agg backend, no plt.show()) ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for ax, rows, title in ((axes[0], rows_nominal, "Nominal heat-up"),
                             (axes[1], rows_disturbance, "Disturbance at step 60")):
        steps = [r["step"] for r in rows]
        ax.plot(steps, [r["Ta_cvx_applied"] for r in rows], 'k-', label="Ta CVXPY (applied)")
        ax.plot(steps, [r["Ta_snn_applied"] for r in rows], 'r--', label="Ta SNN (applied)")
        ax.set_title(title)
        ax.set_ylabel("Ta (degC)")
        ax.legend(loc="lower right")
        ax.grid(True)
    axes[2].plot([r["step"] for r in rows_nominal], [r["control_diff_e_u"] for r in rows_nominal],
                 'b-', label="e_u(k), nominal_heatup")
    axes[2].plot([r["step"] for r in rows_disturbance], [r["control_diff_e_u"] for r in rows_disturbance],
                 'g-', alpha=0.7, label="e_u(k), disturbance_step60")
    axes[2].axvspan(win_lo, win_hi, color='orange', alpha=0.2, label="stiff exotherm window")
    axes[2].set_title("Applied-control difference e_u(k) = u_SNN - u_CVXPY")
    axes[2].set_xlabel("step (minutes)")
    axes[2].set_ylabel("degC")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)
    plt.tight_layout()
    plot_path = out_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # ---- report.md ----
    write_report_md(out_dir, provenance, metrics_nominal, metrics_disturbance, metrics_stiff,
                     n_both_saturated_heatup, pct_both_saturated_heatup, len(heatup_rows),
                     win_lo, win_hi, peak_k, rows_nominal)

    print(f"\nAll outputs written under {out_dir}")


def write_report_md(out_dir, provenance, m_nom, m_dist, m_stiff,
                     n_sat, pct_sat, n_heatup, win_lo, win_hi, peak_k, rows_nominal):
    stiff_max_residual_str = ("N/A" if m_stiff["max_constraint_residual"] is None
                               else f"{m_stiff['max_constraint_residual']:.4e}")

    def table_row(m):
        gap = "N/A" if m["mean_objective_gap_on_feasible_steps"] is None else f"{m['mean_objective_gap_on_feasible_steps']:.4f}"
        maxres = "N/A" if m["max_constraint_residual"] is None else f"{m['max_constraint_residual']:.4e}"
        return (f"| {m['label']} | {m['rms_applied_control_difference']:.4f} | "
                f"{m['rms_trajectory_difference']:.4f} | {m['max_abs_control_difference']:.4f} | "
                f"{m['max_abs_trajectory_difference']:.4f} | {m['snn_convergence_rate_verified']:.2%} | "
                f"{maxres} | {m['n_clipped_outputs']} ({m['pct_clipped_outputs']:.1f}%) | {gap} |")

    lines = [
        "# Final Controlled Comparison Report",
        "",
        f"Generated {provenance['generated_at']} from git commit `{provenance['git_commit']}` "
        f"({'CLEAN' if not provenance['git_working_tree_dirty_files'] else 'DIRTY -- see provenance in summary.json'}).",
        "",
        "## Method",
        "",
        "One shared harness (`tools/final_controlled_comparison.py`) drives two independent "
        "`AutoclavePlant` instances, one per controller, from the identical initial state "
        f"({provenance['initial_temp_degC']} degC), the same target temperature "
        f"({provenance['target_temp_degC']} degC), the same horizon (N={provenance['horizon_N']}), "
        f"the same sampling time (TE={provenance['sampling_time_TE_seconds']}s), the same physical "
        "constraints, and the same canonical per-step QP construction (`src/qp_builder.py`) -- the only "
        "permitted divergence is the `trust_region` flag baked into each controller's own `Ap,Bp` "
        "(documented, see docs/PHASE4_VALIDATION_REPORT.md). Disturbance convention: "
        "**disturbance-before-compute**, applied "
        "identically to both plants at the same step index.",
        "",
        "Both controllers' per-step solves are reimplemented here using the same public building blocks "
        "(`build_qp`, `_condition` for SNN, identical OSQP/SNNSolver calls) each production method uses -- "
        "not a different code path, just an instrumented mirror exposing the full decision vector and "
        "objective value. Every SNN metric is computed by mapping the solver's raw output back to physical "
        "units (`U_sol = result.final_x / D`) and evaluating it against the ORIGINAL, unconditioned "
        "canonical QP -- never trusted from the solver's internal (scaled-space) self-report alone. No new "
        "output clipping was added; `applied_control` uses the exact safety-filter clip already in "
        "`SNNMPCSolver.compute_control_action`.",
        "",
        "**Scenario 3 (stiff exotherm) is a window, not a separate simulation**: steps "
        f"[{win_lo}, {win_hi}) of `nominal_heatup`, centered on its peak `rho(Ap)` "
        f"(k={peak_k}, rho={rows_nominal[peak_k]['rho_Ap_cvx']:.4f}) -- the gelation region both "
        "controllers pass through in the standard 160-step run. No separate plant initial condition "
        "was introduced (would have required modifying `AutoclavePlant`, out of scope).",
        "",
        "## Heat-up slew-saturation caveat",
        "",
        f"Of the first {n_heatup} steps (heat-up phase, before any disturbance), **{n_sat} steps "
        f"({pct_sat:.1f}%)** had BOTH controllers pinned at the `TA_RATE_MAX` slew limit "
        f"({provenance['physical_constraints']['TA_RATE_MAX']} degC/min) simultaneously.",
        "",
        (f"**This means {pct_sat:.1f}% of heat-up agreement is a slew-limit artifact, not evidence of "
         "solver equivalence** -- both controllers ramp at the same physically-imposed maximum rate "
         "regardless of any QP-level agreement, and would agree there even if their underlying QPs "
         "differed substantially, per the explicit instruction not to over-interpret this."
         if pct_sat > 20 else
         "This is a small fraction of the heat-up phase -- slew saturation is not a major confound for "
         "the heat-up-phase agreement reported below, but is reported for completeness and to avoid "
         "silently omitting the check."),
        "",
        "## Summary metrics",
        "",
        "| Scenario | RMS control diff (degC) | RMS trajectory diff | Max abs control diff (degC) | "
        "Max abs trajectory diff | SNN verified-convergence rate | Max constraint residual | "
        "Clipped outputs | Mean objective gap (feasible steps) |",
        "|---|---|---|---|---|---|---|---|---|",
        table_row(m_nom), table_row(m_dist), table_row(m_stiff),
        "",
        "`trajectory_difference(k)` = Euclidean norm of `[Tc1,Tc2,Tc3,alpha1,alpha2,alpha3]_cvx - "
        "[...]_snn`. `objective_gap` is computed and averaged ONLY on steps where the SNN's own scaled "
        "constraint residual is within `feasibility_tol` -- **a lower objective on an infeasible step is "
        "never counted here as evidence of anything**.",
        "",
        "## Feasibility and convergence detail",
        "",
        f"- nominal_heatup: {m_nom['n_feasible_steps_for_objective_gap']}/{m_nom['n_steps']} steps "
        f"({m_nom['pct_feasible_steps_for_objective_gap']:.1f}%) were feasible (scaled) enough to compute "
        f"an objective gap; verified SNN convergence rate {m_nom['snn_convergence_rate_verified']:.2%} "
        f"(raw self-reported `converged` rate {m_nom['snn_convergence_rate_raw']:.2%}).",
        f"- disturbance_step60: {m_dist['n_feasible_steps_for_objective_gap']}/{m_dist['n_steps']} steps "
        f"({m_dist['pct_feasible_steps_for_objective_gap']:.1f}%) feasible; verified convergence rate "
        f"{m_dist['snn_convergence_rate_verified']:.2%}.",
        f"- stiff_exotherm_window: {m_stiff['n_feasible_steps_for_objective_gap']}/{m_stiff['n_steps']} "
        f"steps ({m_stiff['pct_feasible_steps_for_objective_gap']:.1f}%) feasible; verified convergence "
        f"rate {m_stiff['snn_convergence_rate_verified']:.2%}; max constraint residual "
        f"{stiff_max_residual_str} -- consistent with the earlier finding that both solvers can struggle "
        f"on the raw scaled arrays at the gelation peak.",
        "",
        "## Files",
        "",
        "- `per_step_metrics.csv` -- every field listed in the task, for `nominal_heatup` and "
        "`disturbance_step60` (`stiff_exotherm_window` is a labeled subset of `nominal_heatup`'s rows).",
        "- `summary.json` -- full provenance (git commit, package versions, solver config, cost weights, "
        "physical constraints, disturbance convention) plus all aggregate metrics and metric definitions.",
        "- `comparison_plot.png` -- applied Ta overlay for both scenarios plus the control-difference trace, "
        "with the stiff-exotherm window shaded.",
        "",
        "## Known limitations (carried from prior stages, still true)",
        "",
        "- `trust_region` remains the one documented, permitted divergence between the two controllers' "
        "`Ap,Bp` -- the canonical QP construction itself is unified, but the two controllers do not always "
        "solve numerically identical QPs at every step (only when `trust_region`'s clamp is inactive).",
        "- The SNN's raw solver output can diverge at the gelation peak on the raw/scaled arrays "
        "(see `results/qp_conditioning_report.json`, `results/snn_solve_diagnostics.json`); the applied "
        "control there is recovered by the existing slew-rate safety clip, which is reported "
        "(`snn_clipped`), not concealed.",
        "- The row-normalization conditioning change proposed and tested in the prior stage was rejected "
        "after a full-solve comparison showed it regressed feasibility; `_condition` here is the original, "
        "unmodified formula.",
    ]
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
