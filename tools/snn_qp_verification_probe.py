"""
snn_qp_verification_probe.py
Diagnostic-only probe implementing the SNN-QP verification
reference-solver harness. Does NOT modify src/dynamics.py, src/constants.py,
src/plant_simulator.py, or src/snn_mpc_controller.py -- it drives the live SNN
closed loop and, at representative steps, calls the SNNMPCSolver instance's
own existing public methods (build_dense_qp, _condition, _warm_hold) to
reconstruct the identical arrays that were live-solved, then solves those same
conditioned arrays with OSQP as ground truth.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/snn_qp_verification_probe.py
"""
from pathlib import Path
import sys

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize
from snn_opt import OptimizationProblem, SNNSolver

FEASIBILITY_TOL = 1e-2


def solve_reference_osqp(H_s, g_s, C_s, d_s):
    """Ground-truth solve of the SNN's own SCALED arrays (post-_condition)."""
    n = H_s.shape[0]
    U = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H_s)) + g_s @ U),
                       [C_s @ U + d_s <= 0])
    prob.solve(solver=cp.OSQP)
    return (U.value if U.value is not None else None), prob.status


def main():
    plant = AutoclavePlant(initial_temp=28.0)
    controller = SNNMPCSolver(horizon=20, target_temp=120.0)

    time_steps = 160
    current_Ta = 28.0

    trajectory = []  # snapshots taken BEFORE each compute_control_action call

    print("Running live SNN closed loop to capture representative steps...")
    for k in range(time_steps):
        current_state = plant.get_state()

        if k == 60:
            plant.T_comp -= 15.0
            plant.T_tool -= 15.0
            current_state = plant.get_state()

        avg_T = float(np.mean(current_state[0:3]))
        avg_a = float(np.mean(current_state[7:10]))
        Ap, Bp = linearize(avg_T, avg_a, trust_region=True)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ap))))

        U_warm_snapshot = None if controller.U_warm is None else controller.U_warm.copy()

        u_out, _ = controller.compute_control_action(current_state, current_Ta)

        trajectory.append({
            "k": k, "x0": current_state.copy(), "u_prev": current_Ta,
            "Ap": Ap, "Bp": Bp, "rho": rho,
            "U_warm_before": U_warm_snapshot, "u_out_live": u_out,
        })

        current_Ta = u_out
        plant.step(current_Ta)

    rhos = np.array([t["rho"] for t in trajectory])
    gelation_peak_k = int(np.argmax(rhos))
    heat_up_k = 10
    post_brake_k = min(gelation_peak_k + 15, time_steps - 1)
    steady_state_k = time_steps - 1

    probed = {
        "heat-up": heat_up_k,
        "gelation-peak": gelation_peak_k,
        "post-brake": post_brake_k,
        "steady-state": steady_state_k,
    }

    rows = []
    for label, k in probed.items():
        t = trajectory[k]

        H_raw, g_raw, C_raw, d_raw = controller.build_dense_qp(t["Ap"], t["Bp"], t["x0"], t["u_prev"])
        H_s, g_s, C_s, d_s, D = controller._condition(H_raw, g_raw, C_raw, d_raw)

        U_raw = controller._warm_hold(t["u_prev"]) if t["U_warm_before"] is None else t["U_warm_before"]
        U_warm_scaled = U_raw * D

        # Ground truth: OSQP on the identical scaled arrays
        u_ref_vec, ref_status = solve_reference_osqp(H_s, g_s, C_s, d_s)

        # SNN solve on the identical scaled arrays (same warm start as live)
        problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
        solver = SNNSolver(problem, controller.solver_config)
        result = solver.solve(U_warm_scaled, verbose=False)
        U_sol = result.final_x / D
        u0_snn = float(U_sol[0])

        primal_resid = float(np.max(np.maximum(0.0, C_s @ result.final_x + d_s)))
        f_snn = float(0.5 * result.final_x @ H_s @ result.final_x + g_s @ result.final_x)

        if u_ref_vec is not None:
            u0_ref = float(u_ref_vec[0] / D[0])
            f_ref = float(0.5 * u_ref_vec @ H_s @ u_ref_vec + g_s @ u_ref_vec)
            applied_gap = abs(u0_snn - u0_ref)
            obj_gap = (f_snn - f_ref) / max(1.0, abs(f_ref))
        else:
            u0_ref, f_ref, applied_gap, obj_gap = None, None, None, None

        rows.append({
            "label": label, "k": k, "rho": t["rho"],
            "u0_snn": u0_snn, "u0_ref": u0_ref, "ref_status": ref_status,
            "applied_gap": applied_gap, "primal_resid": primal_resid,
            "obj_gap": obj_gap, "converged": result.converged,
            "n_projections": result.n_projections,
        })

    print("\n" + "=" * 130)
    print("SNN-QP REFERENCE-SOLVER CROSS-CHECK")
    print("=" * 130)
    header = (f"{'step':<14}{'k':>5}{'rho(Ap)':>9}{'u0_snn':>10}{'u0_osqp_ref':>13}"
              f"{'applied_gap':>13}{'primal_resid':>13}{'obj_gap':>12}{'converged':>11}"
              f"{'n_proj':>8}{'raw_infeas':>11}{'ref_status':>16}")
    print(header)
    for r in rows:
        raw_infeas = r["primal_resid"] > FEASIBILITY_TOL
        u0_ref_str = "N/A" if r["u0_ref"] is None else f"{r['u0_ref']:.4f}"
        gap_str = "N/A" if r["applied_gap"] is None else f"{r['applied_gap']:.4f}"
        obj_gap_str = "N/A" if r["obj_gap"] is None else f"{r['obj_gap']:.6f}"
        row = (f"{r['label']:<14}{r['k']:>5}{r['rho']:>9.4f}{r['u0_snn']:>10.4f}"
               f"{u0_ref_str:>13}{gap_str:>13}{r['primal_resid']:>13.6f}"
               f"{obj_gap_str:>12}{str(r['converged']):>11}{r['n_projections']:>8}"
               f"{str(raw_infeas):>11}{r['ref_status']:>16}")
        print(row)
    print("=" * 130)
    print("primal_resid is on the SNN's OWN scaled arrays (max(0, Cs@x+ds)); raw_infeas")
    print("flags whether the raw solver iterate itself violates feasibility_tol (1e-2)")
    print("BEFORE the downstream output-clip safety filter. applied_gap is the number")
    print("that determines closed-loop behavior; converged/obj_gap explain why it")
    print("does or doesn't hold.")
    print("=" * 130)


if __name__ == "__main__":
    main()
