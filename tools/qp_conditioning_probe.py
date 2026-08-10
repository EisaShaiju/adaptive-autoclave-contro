"""
qp_conditioning_probe.py
Conditioning analysis per the docs/PHASE4_VALIDATION_REPORT.md §4, on the SAME representative
stiff exotherm QP used in the snn-qp-verification stage (k=84, rho(Ap)=1.5525,
same deterministic state generator). Read-only: does NOT modify src/dynamics.py,
src/qp_builder.py, src/snn_mpc_controller.py, or snn_opt. Phi/Gamma and the
Arrhenius Jacobian terms are recomputed inline here (mirroring qp_builder.py /
dynamics.py exactly) purely for measurement -- the shipped construction path
is untouched.

Also runs ONE minimal, single-variable experiment: an eigen-whitening
preconditioner (H^{-1/2}) applied OFFLINE, fed into the UNMODIFIED SNNSolver
with the SAME config, to test whether the residual ill-conditioning after
Jacobi scaling is structural (off-diagonal/eigenvector) rather than diagonal-
magnitude-driven.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/qp_conditioning_probe.py
"""
from pathlib import Path
import sys
import json
from dataclasses import replace

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

NX = 10
ROLLOUT_STEPS = 105


def find_stiff_state():
    """Identical deterministic procedure to tools/snn_solve_instrumentation.py."""
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
        states.append({"k": k, "x0": x0, "u_prev": u_prev, "rho": rho, "avg_T": avg_T, "avg_a": avg_a})
        current_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        current_state = plant.step(Ta_input=current_Ta)
    return max(states, key=lambda s: s["rho"])


def arrhenius_jacobian_terms(avg_T, avg_a):
    """Read-only mirror of src/dynamics.py's Jacobian terms, trust_region=True."""
    T0_K = avg_T + 273.15
    f0 = const.AC * np.exp(-const.EA / (const.R * T0_K)) * (avg_a ** const.M_EXP) * ((1 - avg_a) ** const.N_EXP)
    dT = f0 * (const.EA / (const.R * (T0_K ** 2)))
    a_safe = max(1e-3, min(avg_a, 0.999))
    da = f0 * ((const.M_EXP / a_safe) - (const.N_EXP / (1 - a_safe)))
    exo_mult = (const.MR * const.DH / const.CPC) * const.TE
    exo_dT = min(exo_mult * dT, const.FC * 1.8)
    exo_da = float(np.clip(exo_mult * da, -const.FC * 3.0, const.FC * 3.0))
    da_self = (1.0 - 1e-4) + float(np.clip(da * const.TE, -0.90, 0.0))
    dT_cross = float(np.clip(dT * const.TE, 0.0, 0.1))
    return {"f0": float(f0), "dT": float(dT), "da": float(da), "exo_mult": float(exo_mult),
            "exo_dT": float(exo_dT), "exo_da": exo_da, "da_self": da_self, "dT_cross": dT_cross}


def build_phi_gamma(Ap, Bp, N):
    """Read-only mirror of qp_builder.build_canonical_qp's Phi/Gamma (s=i window)."""
    Phi = np.zeros((N * NX, NX))
    Gamma = np.zeros((N * NX, N))
    Ak = np.eye(NX)
    for i in range(N):
        Phi[i * NX:(i + 1) * NX, :] = Ak
        Ad = np.eye(NX)
        for j in range(i - 1, -1, -1):
            Gamma[i * NX:(i + 1) * NX, j] = Ad @ Bp
            Ad = Ad @ Ap
        Ak = Ak @ Ap
    return Phi, Gamma


def rng(v):
    v = np.asarray(v, dtype=float)
    return {"min": float(np.min(v)), "max": float(np.max(v))}


def main():
    print("Regenerating the same deterministic stiff state as the prior stage...")
    stiff = find_stiff_state()
    N = 20
    x0, u_prev, rho = stiff["x0"], stiff["u_prev"], stiff["rho"]
    print(f"  k={stiff['k']}  rho(Ap)={rho:.4f}  avg_T={stiff['avg_T']:.2f}  avg_a={stiff['avg_a']:.4f}")

    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0)
    qp = ctrl.build_qp(x0, u_prev)
    H_raw, f_raw, A_ineq, b_ineq = qp.H, qp.f, qp.A_ineq, qp.b_ineq
    C_raw, d_raw = A_ineq, -b_ineq
    Ap, Bp = qp.linearization["Ap"], qp.linearization["Bp"]

    H_s, g_s, C_s, d_s, D = ctrl._condition(H_raw, f_raw, C_raw, d_raw)

    report = {"k": stiff["k"], "rho_Ap": rho, "avg_T": stiff["avg_T"], "avg_a": stiff["avg_a"]}

    # ---- 1. Hessian symmetry error ----
    sym_err_raw = float(np.max(np.abs(H_raw - H_raw.T)))
    sym_err_s = float(np.max(np.abs(H_s - H_s.T)))
    report["hessian_symmetry_error"] = {"raw": sym_err_raw, "conditioned": sym_err_s}

    # ---- 2. Hessian eigenvalue range + 3. condition estimate ----
    eig_raw = np.linalg.eigvalsh(H_raw)
    eig_s = np.linalg.eigvalsh(H_s)
    report["hessian_eigenvalue_range"] = {
        "raw": {"min": float(eig_raw.min()), "max": float(eig_raw.max())},
        "conditioned": {"min": float(eig_s.min()), "max": float(eig_s.max())},
    }
    report["hessian_condition_estimate"] = {
        "raw_cond": float(np.linalg.cond(H_raw)),
        "conditioned_cond": float(np.linalg.cond(H_s)),
        "raw_eig_ratio": float(eig_raw.max() / max(eig_raw.min(), 1e-300)),
        "conditioned_eig_ratio": float(eig_s.max() / max(eig_s.min(), 1e-300)),
    }

    # ---- 4. Constraint-matrix singular-value range ----
    sv_raw = np.linalg.svd(C_raw, compute_uv=False)
    sv_s = np.linalg.svd(C_s, compute_uv=False)
    report["constraint_matrix_singular_values"] = {
        "raw": {"min": float(sv_raw.min()), "max": float(sv_raw.max())},
        "conditioned": {"min": float(sv_s.min()), "max": float(sv_s.max())},
    }

    # ---- 5/6. State- and input-variable magnitude ranges ----
    Phi, Gamma = build_phi_gamma(Ap, Bp, N)
    U_ref_vec, ref_status = None, None
    U = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H_raw)) + f_raw @ U),
                       [A_ineq @ U <= b_ineq])
    prob.solve(solver=cp.OSQP)
    if U.value is not None:
        U_ref_vec = U.value
    ref_status = prob.status

    report["input_variable_magnitude_range"] = {
        "x0_and_u_prev_context": {"u_prev": float(u_prev), "TA_MIN": const.TA_MIN, "TA_MAX": const.TA_MAX},
        "reference_osqp_solution_physical": (rng(U_ref_vec) if U_ref_vec is not None else None),
        "reference_status": ref_status,
    }
    if U_ref_vec is not None:
        X_pred = (Phi @ x0).reshape(N, NX) + (Gamma @ U_ref_vec).reshape(N, NX)
        report["state_variable_magnitude_range"] = {
            "x0_physical": rng(x0),
            "predicted_trajectory_at_reference_u": rng(X_pred),
        }
    else:
        report["state_variable_magnitude_range"] = {"x0_physical": rng(x0), "predicted_trajectory_at_reference_u": None}

    # ---- 7. Slack variables ----
    report["slack_variable_magnitude_range"] = "not applicable -- this QP formulation has no slack variables (hard inequalities only)"

    # ---- 8. Arrhenius Jacobian magnitude range ----
    jac = arrhenius_jacobian_terms(stiff["avg_T"], stiff["avg_a"])
    report["arrhenius_jacobian_terms"] = jac
    report["arrhenius_jacobian_magnitude_range"] = rng(list(jac.values()))

    # ---- 9. Objective-term magnitude range (H's three additive components) ----
    Q_diag, R_val, S_val = ctrl.Q_diag, ctrl.R_val, ctrl.S_val
    Q_bar = np.kron(np.eye(N), np.diag(Q_diag))
    R_bar = np.eye(N) * R_val
    Diff = np.eye(N)
    for i in range(1, N):
        Diff[i, i - 1] = -1.0
    S_bar = np.eye(N) * S_val
    term_tracking = 2.0 * (Gamma.T @ Q_bar @ Gamma)
    term_effort = 2.0 * R_bar
    term_slew = 2.0 * (Diff.T @ S_bar @ Diff)
    report["objective_term_magnitude_range"] = {
        "tracking_term_(Gamma^T Q Gamma)": {"frobenius_norm": float(np.linalg.norm(term_tracking)), "max_abs": float(np.max(np.abs(term_tracking)))},
        "effort_term_(R)": {"frobenius_norm": float(np.linalg.norm(term_effort)), "max_abs": float(np.max(np.abs(term_effort)))},
        "slew_term_(Diff^T S Diff)": {"frobenius_norm": float(np.linalg.norm(term_slew)), "max_abs": float(np.max(np.abs(term_slew)))},
    }

    # ---- 10. Constraint-row magnitude range (by block) ----
    row_norms_raw = np.linalg.norm(C_raw, axis=1)
    blocks = {"box_upper": (0, N), "box_lower": (N, 2 * N), "slew_pos": (2 * N, 3 * N),
              "slew_neg": (3 * N, 4 * N), "gradient_pos": (4 * N, 5 * N), "gradient_neg": (5 * N, 6 * N)}
    report["constraint_row_magnitude_range_by_block"] = {
        name: rng(row_norms_raw[a:b]) for name, (a, b) in blocks.items()
    }
    report["constraint_row_magnitude_range_overall"] = rng(row_norms_raw)

    # ---- 11/12. Current scaling / preconditioning factors ----
    report["current_scaling_factors_D"] = rng(D)
    C_over_D = C_raw / D[None, :]
    row_norm_C_over_D = np.linalg.norm(C_over_D, axis=1)
    report["current_preconditioning_row_norm_factors"] = rng(row_norm_C_over_D)
    report["diag_H_range_raw_vs_1_after_conditioning"] = {
        "raw_diag_ratio_max_over_min": float(np.max(np.diag(H_raw)) / max(np.min(np.diag(H_raw)), 1e-300)),
        "conditioned_diag_ratio": float(np.max(np.diag(H_s)) / max(np.min(np.diag(H_s)), 1e-300)),
    }
    reg_raw = 1e-3
    reg_scaled_range = rng(1e-3 / (D ** 2))
    report["regularization_significance"] = {
        "raw_added_to_H": reg_raw,
        "scaled_1e-3_over_D_squared_range": reg_scaled_range,
        "typical_H_scaled_entry_magnitude": float(np.median(np.abs(H_s[H_s != 0]))) if np.any(H_s != 0) else 0.0,
    }

    # ---- 13/14. SNN state magnitude range + iteration-progress behavior (Jacobi-conditioned) ----
    U_cold = ctrl._warm_hold(u_prev)
    U_warm_scaled = U_cold * D
    trace_config = replace(ctrl.solver_config, backend='python', record_trajectory=True)
    problem_s = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
    solver_s = SNNSolver(problem_s, trace_config)
    result_s = solver_s.solve(U_warm_scaled, verbose=False)
    norms_s = np.linalg.norm(result_s.X, axis=1)
    checkpoints = sorted(set([0, 1, 5, 50, 500, 2000, len(norms_s) - 1]))
    report["snn_state_magnitude_range_jacobi"] = {"min": float(norms_s.min()), "max": float(norms_s.max())}
    report["snn_iteration_progress_jacobi"] = [
        {"iter": int(i), "norm_x": float(norms_s[i]), "objective": float(result_s.objective_values[i]),
         "max_violation": float(result_s.constraint_violations[i])}
        for i in checkpoints if i < len(norms_s)
    ]
    report["snn_final_jacobi"] = {
        "converged": bool(result_s.converged), "reason": result_s.convergence_reason,
        "n_projections": int(result_s.n_projections),
        "final_violation": float(result_s.constraint_violations[-1]),
    }

    # ==================================================================
    # COMPARISON: (1) no preconditioning, (2) Jacobi, (3) OSQP's own scaling
    # ==================================================================
    print("\nRunning comparison: no-preconditioning vs Jacobi vs OSQP-internal-scaling...")
    comparison = {}

    # (1) No preconditioning: raw arrays straight into the UNMODIFIED SNNSolver.
    problem_raw = OptimizationProblem(A=H_raw, b=f_raw, C=C_raw, d=d_raw)
    solver_raw = SNNSolver(problem_raw, ctrl.solver_config)  # same config, fast backend
    result_raw = solver_raw.solve(U_cold, verbose=False)
    comparison["no_preconditioning"] = {
        "k0_step_size": float(solver_raw._k0),
        "cond_H": float(np.linalg.cond(H_raw)),
        "converged": bool(result_raw.converged), "reason": result_raw.convergence_reason,
        "final_violation": float(problem_raw.max_violation(result_raw.final_x)),
        "final_objective": float(result_raw.final_objective),
        "final_x_norm": float(np.linalg.norm(result_raw.final_x)),
    }

    # (2) Current Jacobi preconditioning (via ctrl._condition, unchanged).
    solver_jac = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s), ctrl.solver_config)
    result_jac = solver_jac.solve(U_warm_scaled, verbose=False)
    U_sol_jac_physical = result_jac.final_x / D
    comparison["jacobi_preconditioning"] = {
        "k0_step_size": float(solver_jac._k0),
        "cond_H": float(np.linalg.cond(H_s)),
        "converged": bool(result_jac.converged), "reason": result_jac.convergence_reason,
        "final_violation_scaled": float(problem_s.max_violation(result_jac.final_x)),
        "final_violation_physical": float(problem_raw.max_violation(U_sol_jac_physical)),
        "final_objective": float(result_jac.final_objective),
        "final_x_norm_physical": float(np.linalg.norm(U_sol_jac_physical)),
    }

    # (3) Existing scaling already in the repo: OSQP's own internal equilibration,
    #     via the CVXPY path this exact raw canonical QP would take (default vs disabled).
    U2 = cp.Variable(N)
    prob_scaled = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U2, cp.psd_wrap(H_raw)) + f_raw @ U2),
                              [A_ineq @ U2 <= b_ineq])
    prob_scaled.solve(solver=cp.OSQP, warm_start=True)  # default: OSQP scaling ON
    U3 = cp.Variable(N)
    prob_unscaled = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U3, cp.psd_wrap(H_raw)) + f_raw @ U3),
                                [A_ineq @ U3 <= b_ineq])
    try:
        prob_unscaled.solve(solver=cp.OSQP, warm_start=True, scaling=0)  # OSQP scaling OFF
        unscaled_status = prob_unscaled.status
        unscaled_iters = getattr(prob_unscaled.solver_stats, "num_iters", None)
    except Exception as exc:
        unscaled_status, unscaled_iters = f"error: {exc}", None
    comparison["osqp_internal_scaling_existing_in_repo"] = {
        "note": "This is the scaling OSQP already applies by default whenever the CVXPY/baseline "
                "controller solves -- it is NOT wired to the SNN's projected-gradient solver, which "
                "cannot consume it (snn_opt has no equivalent automatic equilibration).",
        "with_default_scaling": {
            "status": prob_scaled.status,
            "iterations": getattr(prob_scaled.solver_stats, "num_iters", None),
            "objective": float(prob_scaled.value) if prob_scaled.value is not None else None,
        },
        "scaling_disabled": {
            "status": unscaled_status,
            "iterations": unscaled_iters,
            "objective": float(prob_unscaled.value) if prob_unscaled.value is not None else None,
        },
    }
    report["comparison_1_2_3"] = comparison

    # ==================================================================
    # MINIMAL EXPERIMENT: eigen-whitening preconditioner (offline, solver unchanged)
    # ==================================================================
    print("Running minimal experiment: eigen-whitening preconditioner vs Jacobi...")
    eigvals, eigvecs = np.linalg.eigh(H_raw)
    eigvals_clipped = np.maximum(eigvals, 1e-8)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T          # H^{-1/2}
    W_inv = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T           # H^{+1/2}

    H_w = W @ H_raw @ W
    f_w = W @ f_raw
    C_w_pre = C_raw @ W
    row_norms_w = np.maximum(np.linalg.norm(C_w_pre, axis=1, keepdims=True), 1e-10)
    C_w = C_w_pre / row_norms_w
    d_w = d_raw / row_norms_w.squeeze()

    U_cold_whitened = W_inv @ U_cold  # z = W zhat  =>  zhat = W^{-1} z

    problem_w = OptimizationProblem(A=H_w, b=f_w, C=C_w, d=d_w)
    solver_w = SNNSolver(problem_w, ctrl.solver_config)  # UNMODIFIED solver + config
    result_w = solver_w.solve(U_cold_whitened, verbose=False)
    U_sol_w_physical = W @ result_w.final_x

    experiment = {
        "transform": "z = W zhat + 0,  W = H_raw^{-1/2} (symmetric eigen-whitening)",
        "cond_H_whitened_sanity_check": float(np.linalg.cond(H_w)),  # should be ~1.0
        "cond_H_jacobi_for_comparison": float(np.linalg.cond(H_s)),
        "converged": bool(result_w.converged), "reason": result_w.convergence_reason,
        "final_violation_physical": float(problem_raw.max_violation(U_sol_w_physical)),
        "final_objective_whitened_space": float(result_w.final_objective),
        "final_x_norm_physical": float(np.linalg.norm(U_sol_w_physical)),
        "jacobi_final_violation_physical_for_comparison": comparison["jacobi_preconditioning"]["final_violation_physical"],
        "jacobi_final_x_norm_physical_for_comparison": comparison["jacobi_preconditioning"]["final_x_norm_physical"],
    }
    report["minimal_experiment_eigen_whitening"] = experiment

    # ---- print compact summary (no large arrays to terminal) ----
    print("\n" + "=" * 90)
    print(f"QP CONDITIONING REPORT -- stiff exotherm QP (k={stiff['k']}, rho(Ap)={rho:.4f})")
    print("=" * 90)
    print(f"Hessian symmetry error: raw={sym_err_raw:.3e}  conditioned={sym_err_s:.3e}")
    print(f"Hessian eigenvalues: raw=[{eig_raw.min():.3e}, {eig_raw.max():.3e}]  "
          f"conditioned=[{eig_s.min():.3e}, {eig_s.max():.3e}]")
    print(f"cond(H): raw={report['hessian_condition_estimate']['raw_cond']:.3e}  "
          f"conditioned={report['hessian_condition_estimate']['conditioned_cond']:.3e}")
    print(f"C singular values: raw=[{sv_raw.min():.3e},{sv_raw.max():.3e}]  "
          f"conditioned=[{sv_s.min():.3e},{sv_s.max():.3e}]")
    print(f"diag(H) ratio: raw={report['diag_H_range_raw_vs_1_after_conditioning']['raw_diag_ratio_max_over_min']:.3e}  "
          f"conditioned={report['diag_H_range_raw_vs_1_after_conditioning']['conditioned_diag_ratio']:.3e}")
    print(f"D (scaling factors) range: [{report['current_scaling_factors_D']['min']:.3e}, "
          f"{report['current_scaling_factors_D']['max']:.3e}]")
    print(f"row_norm(C/D) range: [{report['current_preconditioning_row_norm_factors']['min']:.3e}, "
          f"{report['current_preconditioning_row_norm_factors']['max']:.3e}]")
    print(f"regularization 1e-3/D^2 range: [{reg_scaled_range['min']:.3e}, {reg_scaled_range['max']:.3e}]  "
          f"vs typical H_s entry ~{report['regularization_significance']['typical_H_scaled_entry_magnitude']:.3e}")
    print("\n--- Comparison (1) no-precond / (2) Jacobi / (3) OSQP-internal ---")
    print(f"  (1) no_preconditioning:  k0={comparison['no_preconditioning']['k0_step_size']:.3e}  "
          f"converged={comparison['no_preconditioning']['converged']}  "
          f"final_violation={comparison['no_preconditioning']['final_violation']:.3e}")
    print(f"  (2) jacobi:              k0={comparison['jacobi_preconditioning']['k0_step_size']:.3e}  "
          f"converged={comparison['jacobi_preconditioning']['converged']}  "
          f"final_violation_physical={comparison['jacobi_preconditioning']['final_violation_physical']:.3e}")
    print(f"  (3) osqp_default_scaling: status={comparison['osqp_internal_scaling_existing_in_repo']['with_default_scaling']['status']}  "
          f"iters={comparison['osqp_internal_scaling_existing_in_repo']['with_default_scaling']['iterations']}")
    print(f"      osqp_scaling_disabled: status={comparison['osqp_internal_scaling_existing_in_repo']['scaling_disabled']['status']}  "
          f"iters={comparison['osqp_internal_scaling_existing_in_repo']['scaling_disabled']['iterations']}")
    print("\n--- Minimal experiment: eigen-whitening vs Jacobi ---")
    print(f"  cond(H_whitened)={experiment['cond_H_whitened_sanity_check']:.3e} (sanity, should be ~1.0)  "
          f"cond(H_jacobi)={experiment['cond_H_jacobi_for_comparison']:.3e}")
    print(f"  whitened: converged={experiment['converged']}  reason={experiment['reason']}  "
          f"final_violation_physical={experiment['final_violation_physical']:.3e}  "
          f"final_x_norm={experiment['final_x_norm_physical']:.3e}")
    print(f"  jacobi:   final_violation_physical={experiment['jacobi_final_violation_physical_for_comparison']:.3e}  "
          f"final_x_norm={experiment['jacobi_final_x_norm_physical_for_comparison']:.3e}")
    print("=" * 90)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "qp_conditioning_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
