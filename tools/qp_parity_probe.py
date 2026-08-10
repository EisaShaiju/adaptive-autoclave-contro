"""
qp_parity_probe.py
Diagnostic-only probe implementing the model-parity window-derivation
check. Does NOT modify src/dynamics.py, src/constants.py, src/plant_simulator.py,
or either controller. Read-only with respect to the shipped code; it drives the
live CVXPY closed loop, then independently re-derives the condensed dense QP
two ways (the correct `s=i` window vs. `build_dense_qp`'s actual `s=i+1`
window) and compares u_0 against the live sparse-form CVXPY/OSQP solve.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/qp_parity_probe.py
"""
from pathlib import Path
import sys

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.dynamics import linearize
import src.constants as const

N = 20
Q_DIAG = np.zeros(10)
Q_DIAG[0:3] = 100.0
R_VAL = 0.1
S_VAL = 1.0
TARGET_TEMP = 120.0
NX = 10


def build_dense_qp_window(Ap, Bp, x0, u_prev, s_offset):
    """Condense to (H, g, C, d) over U in R^N. s_offset=0 reproduces the
    CVXPY window (row i predicts x_i, matching the cost/constraint summed
    over x_0..x_{N-1}). s_offset=1 reproduces build_dense_qp's actual
    behavior (row i predicts x_{i+1})."""
    Phi = np.zeros((N * NX, NX))
    Gamma = np.zeros((N * NX, N))
    Ak = np.eye(NX)
    for i in range(N):
        if s_offset == 1:
            Ak = Ak @ Ap
            Phi[i * NX:(i + 1) * NX, :] = Ak
            Ad = np.eye(NX)
            for j in range(i, -1, -1):
                Gamma[i * NX:(i + 1) * NX, j] = Ad @ Bp
                Ad = Ad @ Ap
        else:
            # s_offset == 0: store BEFORE multiplying -> Phi[0] = Ap^0 = I
            Phi[i * NX:(i + 1) * NX, :] = Ak
            Ad = np.eye(NX)
            for j in range(i - 1, -1, -1):
                Gamma[i * NX:(i + 1) * NX, j] = Ad @ Bp
                Ad = Ad @ Ap
            Ak = Ak @ Ap

    Q_bar = np.kron(np.eye(N), np.diag(Q_DIAG))
    R_bar = np.eye(N) * R_VAL

    Diff = np.eye(N)
    for i in range(1, N):
        Diff[i, i - 1] = -1.0
    S_bar = np.eye(N) * S_VAL

    d0 = np.zeros(N)
    d0[0] = u_prev

    X_ref = np.zeros(N * NX)
    for i in range(N):
        X_ref[i * NX: i * NX + 3] = TARGET_TEMP

    H = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar + Diff.T @ S_bar @ Diff)
    H = (H + H.T) / 2.0
    H += np.eye(N) * 1e-3

    E = Phi @ x0 - X_ref
    g = 2.0 * (Gamma.T @ Q_bar @ E) - 2.0 * (Diff.T @ S_bar @ d0)

    C_rows, d_rows = [], []
    C_rows.append(np.eye(N)); d_rows.append(-np.ones(N) * const.TA_MAX)
    C_rows.append(-np.eye(N)); d_rows.append(np.ones(N) * const.TA_MIN)
    C_rows.append(Diff); d_rows.append(-d0 - const.TA_RATE_MAX)
    C_rows.append(-Diff); d_rows.append(d0 - const.TA_RATE_MAX)

    G = np.zeros((N, N * NX))
    for i in range(N):
        G[i, i * NX + 0] = 1.0
        G[i, i * NX + 2] = -1.0
    GGamma = G @ Gamma
    GPhi_x = G @ Phi @ x0
    C_rows.append(GGamma); d_rows.append(GPhi_x - const.GRADIENT_MAX)
    C_rows.append(-GGamma); d_rows.append(-GPhi_x - const.GRADIENT_MAX)

    return H, g, np.vstack(C_rows), np.concatenate(d_rows)


def solve_condensed_osqp(H, g, C, d):
    U = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H)) + g @ U),
                       [C @ U + d <= 0])
    prob.solve(solver=cp.OSQP)
    return (U.value[0] if U.value is not None else None), prob.status


def main():
    plant = AutoclavePlant(initial_temp=28.0)
    controller = MPCSolver(horizon=N, target_temp=TARGET_TEMP)

    time_steps = 160
    current_Ta = 28.0
    current_state = plant.get_state()

    trajectory = []  # (k, x0, u_prev, Ap, Bp, rho, u0_live)

    print("Running live CVXPY closed loop to capture representative steps...")
    for k in range(time_steps):
        x0 = current_state.copy()
        u_prev = current_Ta

        avg_T = np.mean(x0[0:3])
        avg_a = np.mean(x0[7:10])
        Ap, Bp = linearize(avg_T, avg_a, trust_region=False)
        rho = np.max(np.abs(np.linalg.eigvals(Ap)))

        current_Ta, _ = controller.compute_control_action(x0, u_prev)
        u0_live = current_Ta

        trajectory.append({"k": k, "x0": x0, "u_prev": u_prev, "Ap": Ap,
                            "Bp": Bp, "rho": rho, "u0_live": u0_live})

        if k == 60:
            plant.T_comp -= 15.0
            plant.T_tool -= 15.0

        current_state = plant.step(Ta_input=current_Ta)

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

    print("\n" + "=" * 100)
    print("WINDOW-DERIVATION PROBE (model parity)")
    print("=" * 100)
    header = f"{'step':<14}{'k':>5}{'rho(Ap)':>10}{'u0_live_cvxpy':>16}{'u0_s=i_OSQP':>14}{'u0_s=i+1_OSQP':>16}{'|live-s=i|':>12}{'|live-s=i+1|':>14}{'status_i':>10}{'status_i+1':>12}"
    print(header)

    results = []
    for label, k in probed.items():
        t = trajectory[k]
        H0, g0, C0, d0 = build_dense_qp_window(t["Ap"], t["Bp"], t["x0"], t["u_prev"], s_offset=0)
        H1, g1, C1, d1 = build_dense_qp_window(t["Ap"], t["Bp"], t["x0"], t["u_prev"], s_offset=1)

        u0_si, status_i = solve_condensed_osqp(H0, g0, C0, d0)
        u0_si1, status_i1 = solve_condensed_osqp(H1, g1, C1, d1)

        u0_live = t["u0_live"]
        delta_i = None if u0_si is None else abs(u0_live - u0_si)
        delta_i1 = None if u0_si1 is None else abs(u0_live - u0_si1)

        row = (f"{label:<14}{k:>5}{t['rho']:>10.4f}{u0_live:>16.4f}"
               f"{('N/A' if u0_si is None else f'{u0_si:.4f}'):>14}"
               f"{('N/A' if u0_si1 is None else f'{u0_si1:.4f}'):>16}"
               f"{('N/A' if delta_i is None else f'{delta_i:.4f}'):>12}"
               f"{('N/A' if delta_i1 is None else f'{delta_i1:.4f}'):>14}"
               f"{status_i:>10}{status_i1:>12}")
        print(row)
        results.append({
            "label": label, "k": k, "rho": float(t["rho"]),
            "u0_live_cvxpy": float(u0_live),
            "u0_s_i": None if u0_si is None else float(u0_si),
            "u0_s_i1": None if u0_si1 is None else float(u0_si1),
            "delta_s_i": None if delta_i is None else float(delta_i),
            "delta_s_i1": None if delta_i1 is None else float(delta_i1),
            "status_s_i": status_i, "status_s_i1": status_i1,
        })

    print("=" * 100)
    print("Interpretation: if |live-s=i| << |live-s=i+1| at a step, the CORRECT")
    print("window is s=i and build_dense_qp's s=i+1 (its actual, shipped behavior)")
    print("is the mismatch. Both windows can look 'plausible' individually --")
    print("only this comparison against the live CVXPY solve localizes it.")
    print("=" * 100)

    return results


if __name__ == "__main__":
    main()
