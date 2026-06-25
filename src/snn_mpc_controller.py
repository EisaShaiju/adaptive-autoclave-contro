"""
snn_mpc_controller.py
Closed-loop MPC solver utilizing Spiking Neural Networks (LIF Dynamics).
Final Architecture: Preconditioned Jacobi Scaling with Matrix-Embedded Box Constraints.
"""

import numpy as np
import time
import src.constants as const
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


class SNNMPCSolver:
    def __init__(self, horizon=20, target_temp=120.0):
        self.N = horizon
        self.nx = 10
        self.nu = 1
        self.target_temp = target_temp

        conv_config = ConvergenceConfig(
            enable_early_stopping=True,
            check_every=50,
            min_iterations=100,
            patience=3,
            obj_rel_tol=1e-5,
            proj_grad_tol=1e-2, 
            feasibility_tol=1e-2,
        )
        
        self.solver_config = SolverConfig(
            k0=None,            # Auto-computes safely now that the matrix is scaled
            k0_scale=0.8,       # Confident step size for a conditioned matrix
            projection_method='adaptive',
            max_iterations=5000,
            # NATIVE SCALAR BOUNDS REMOVED - They break in scaled vector space.
            convergence=conv_config,
        )

        self.U_warm = None

        self.Q_diag = np.zeros(self.nx)
        for i in range(3):
            self.Q_diag[i] = 100.0
        self.R_val = 0.1
        self.S_val = 1.0

    def _warm_hold(self, u_prev):
        return np.full(self.N, float(u_prev))

    def _shift(self, U):
        W = np.zeros(self.N)
        W[:-1] = U[1:]
        W[-1] = U[-1]
        return W

    # ------------------------------------------------------------------
    # The Math Engine: Jacobi Preconditioning
    # ------------------------------------------------------------------
    def _condition(self, H, g, C, d):
        """Preconditions the highly stiff MPC matrix to allow rapid solver convergence."""
        D = np.sqrt(np.maximum(np.diag(H), 1e-12))
        H_s = (H / D[:, None]) / D[None, :]
        g_s = g / D
        C_s = C / D[None, :]
        
        # Normalize constraint rows to prevent adaptive projection explosions
        row_norms = np.maximum(np.linalg.norm(C_s, axis=1, keepdims=True), 1e-10)
        
        return H_s, g_s, C_s / row_norms, d / row_norms.squeeze(), D

    # ------------------------------------------------------------------
    # Plant linearisation
    # ------------------------------------------------------------------
    def update_matrices(self, T0_degC, alpha0):
        Ap = np.zeros((self.nx, self.nx))
        Bp = np.zeros(self.nx)
        T0_K = T0_degC + 273.15

        f0 = const.AC * np.exp(-const.EA / (const.R * T0_K)) * \
             (alpha0 ** const.M_EXP) * ((1 - alpha0) ** const.N_EXP)

        dT = f0 * (const.EA / (const.R * T0_K ** 2))
        a_safe = max(1e-3, min(alpha0, 0.999))
        da = f0 * (const.M_EXP / a_safe - const.N_EXP / (1 - a_safe))

        exo_mult = (const.MR * const.DH / const.CPC) * const.TE

        exo_dT = min(exo_mult * dT,   const.FC * 1.8)
        exo_da = float(np.clip(exo_mult * da, -const.FC * 3.0, const.FC * 3.0))
        da_TE  = float(np.clip(da * const.TE,  -0.90, 0.0))
        dT_TE  = float(np.clip(dT * const.TE,   0.0,  0.1))

        for i in range(const.NZ_C):
            Ap[i, i] = 1 - 2 * const.FC + exo_dT
            Ap[i, i + 7] = exo_da
            if i == 0:
                Ap[i, i + 1] = 2 * const.FC
            elif i == const.NZ_C - 1:
                Ap[i, i - 1] = const.FC
                Ap[i, const.NZ_C] = const.FC
            else:
                Ap[i, i - 1] = const.FC
                Ap[i, i + 1] = const.FC

        for i in range(const.NZ_C, const.NZ_C + const.NZ_T):
            Ap[i, i] = 1 - 2 * const.FT
            if i == const.NZ_C:
                Ap[i, i - 1] = const.FT
                Ap[i, i + 1] = const.FT
            elif i == const.NZ_C + const.NZ_T - 1:
                Ap[i, i - 1] = const.FT
                Bp[i] = const.FT
            else:
                Ap[i, i - 1] = const.FT
                Ap[i, i + 1] = const.FT

        for i in range(7, 10):
            Ap[i, i]     = (1.0 - 1e-4) + da_TE
            Ap[i, i - 7] = dT_TE

        return Ap, Bp

    def build_dense_qp(self, Ap, Bp, x0, u_prev):
        rho = np.max(np.abs(np.linalg.eigvals(Ap)))
        if rho >= 1.0:
            Ap = Ap * (0.98 / rho)

        Phi   = np.zeros((self.N * self.nx, self.nx))
        Gamma = np.zeros((self.N * self.nx, self.N))
        Ak = np.eye(self.nx)
        for i in range(self.N):
            Ak = Ak @ Ap
            Phi[i * self.nx:(i + 1) * self.nx, :] = Ak
            Ad = np.eye(self.nx)
            for j in range(i, -1, -1):
                Gamma[i * self.nx:(i + 1) * self.nx, j] = Ad @ Bp
                Ad = Ad @ Ap

        Q_bar = np.kron(np.eye(self.N), np.diag(self.Q_diag))
        R_bar = np.eye(self.N) * self.R_val

        Diff = np.eye(self.N)
        for i in range(1, self.N):
            Diff[i, i - 1] = -1.0
        S_bar = np.eye(self.N) * self.S_val

        d0 = np.zeros(self.N)
        d0[0] = u_prev

        X_ref = np.zeros(self.N * self.nx)
        for i in range(self.N):
            X_ref[i * self.nx: i * self.nx + 3] = self.target_temp

        H = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar + Diff.T @ S_bar @ Diff)
        H = (H + H.T) / 2.0
        H += np.eye(self.N) * 1e-3

        E = Phi @ x0 - X_ref
        g = 2.0 * (Gamma.T @ Q_bar @ E) - 2.0 * (Diff.T @ S_bar @ d0)

        C_rows, d_rows = [], []
        
        # 1. Box Constraints (Matrix Embedded so they scale mathematically)
        C_rows.append( np.eye(self.N));  d_rows.append(-np.ones(self.N) * const.TA_MAX)
        C_rows.append(-np.eye(self.N));  d_rows.append( np.ones(self.N) * const.TA_MIN)
        
        # 2. Slew Constraints
        C_rows.append( Diff);            d_rows.append(-d0 - const.TA_RATE_MAX)
        C_rows.append(-Diff);            d_rows.append( d0 - const.TA_RATE_MAX)

        # 3. Gradient Constraints
        G = np.zeros((self.N, self.N * self.nx))
        for i in range(self.N):
            G[i, i * self.nx + 0] =  1.0
            G[i, i * self.nx + 2] = -1.0
        GGamma = G @ Gamma
        GPhi_x = G @ Phi @ x0
        C_rows.append( GGamma);  d_rows.append( GPhi_x - const.GRADIENT_MAX)
        C_rows.append(-GGamma);  d_rows.append(-GPhi_x - const.GRADIENT_MAX)

        return H, g, np.vstack(C_rows), np.concatenate(d_rows)

    def compute_control_action(self, current_state, u_prev):
        t0 = time.time()

        avg_T = float(np.mean(current_state[0:3]))
        avg_a = float(np.mean(current_state[7:10]))
        Ap, Bp = self.update_matrices(avg_T, avg_a)
        
        H_raw, g_raw, C_raw, d_raw = self.build_dense_qp(Ap, Bp, current_state, u_prev)

        # Precondition the matrix to unlock solver speed
        H_s, g_s, C_s, d_s, D = self._condition(H_raw, g_raw, C_raw, d_raw)

        if not (np.isfinite(H_s).all() and np.isfinite(g_s).all()):
            self.U_warm = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0

        U_raw = self._warm_hold(u_prev) if self.U_warm is None else self.U_warm
        
        # Math Fix: Shift into Scaled Space (*)
        U_warm_scaled = U_raw * D

        problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)
        solver  = SNNSolver(problem, self.solver_config)

        try:
            # Disable terminal spam during normal predictions
            result = solver.solve(U_warm_scaled, verbose=False)
            solve_time = (time.time() - t0) * 1000.0
            
            # Math Fix: Shift back out of Scaled Space (/)
            U_sol = result.final_x / D

            if not result.converged and result.n_projections > 500:
                print(f"  SNN LOG: high spike traffic ({result.n_projections})")

            u_out = float(U_sol[0])
            u_out = float(np.clip(u_out,
                                  u_prev - const.TA_RATE_MAX,
                                  u_prev + const.TA_RATE_MAX))
            u_out = float(np.clip(u_out, const.TA_MIN, const.TA_MAX))

            self.U_warm = self._shift(U_sol)
            return u_out, solve_time

        except Exception as exc:
            self.U_warm = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0