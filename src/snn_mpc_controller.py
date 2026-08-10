"""
snn_mpc_controller.py
Closed-loop MPC solver utilizing Spiking Neural Networks (LIF Dynamics).
Final Architecture: Preconditioned Jacobi Scaling with Matrix-Embedded Box Constraints.
"""

import numpy as np
import time
import src.constants as const
from src.dynamics import linearize
from src.qp_builder import build_canonical_qp
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


class SNNMPCSolver:
    def __init__(self, horizon=20, target_temp=120.0, trust_region=False,
                 soft_state_constraints=False, k0_scale=0.5):
        self.N = horizon
        self.nx = 10
        self.nu = 1
        self.target_temp = target_temp
        # Prediction-model and QP-form options. These are part of the SHARED
        # problem definition: whatever is set here must be set identically on
        # MPCSolver for the two controllers to be solving the same problem.
        # Default trust_region=False (pure first-order model) so that the
        # DEFAULT configuration of the two controllers is model-identical;
        # trust_region=True is now an explicit opt-in, not a silent asymmetry.
        self.trust_region = trust_region
        self.soft_state_constraints = soft_state_constraints

        conv_config = ConvergenceConfig(
            enable_early_stopping=True,
            check_every=50,
            min_iterations=100,
            patience=3,
            obj_rel_tol=1e-7,
            proj_grad_tol=5e-2,
            feasibility_tol=1e-2,
        )

        self.solver_config = SolverConfig(
            k0=None,            # Auto-computes from the Lipschitz constant of the
            k0_scale=k0_scale,  # Jacobi-conditioned Hessian (stable margin).
            projection_method='adaptive',
            max_iterations=8000,
            max_projection_iters=200,   # Budget to resolve the coupled slew/gradient
                                        # active set (100 leaves residual violations).
            backend='c',        # Compiled kernel: numerically identical to the pure-
                                # Python reference but ~85x faster (~0.1 s/step vs ~13 s).
            # NATIVE SCALAR BOUNDS REMOVED - They break in scaled vector space.
            convergence=conv_config,
        )

        self.U_warm = None

        # Diagnostic: how many steps the input-feasibility projection (below)
        # actually had to correct the raw solver output. In a solver that fully
        # enforced its own constraints this would stay 0; it is non-zero here
        # (heat-up slew + exotherm), which we report honestly rather than hide.
        self.n_projection_active = 0

        self.Q_diag = np.zeros(self.nx)
        for i in range(3):
            self.Q_diag[i] = 100.0
        self.R_val = 0.1
        self.S_val = 1.0

    def _warm_hold(self, u_prev, n_total=None):
        """Cold start: hold u_prev across the horizon. When the QP carries
        gradient slacks, the slack block starts at 0 (assume attainable)."""
        n_total = self.N if n_total is None else n_total
        W = np.zeros(n_total)
        W[:self.N] = float(u_prev)
        return W

    def _shift(self, U):
        """Receding-horizon shift of the input block; any slack block is
        shifted the same way (it is indexed by horizon step too)."""
        W = np.zeros_like(U)
        W[:self.N - 1] = U[1:self.N]
        W[self.N - 1] = U[self.N - 1]
        if U.shape[0] > self.N:
            S = U[self.N:]
            W[self.N:self.N + S.shape[0] - 1] = S[1:]
            W[-1] = S[-1]
        return W

    # ------------------------------------------------------------------
    # The Math Engine: Jacobi Preconditioning
    # ------------------------------------------------------------------
    def _condition(self, H, g, C, d):
        """Preconditions the highly stiff MPC matrix to allow rapid solver convergence.

        NOTE: a candidate change to this row-normalization (1/max(||C_row||,
        |d_row|) instead of 1/||C_row|| alone) was proposed and tested after a
        qp-conditioning analysis found the last-horizon-step gradient-constraint
        row's offset d_j can be 5+ orders of magnitude larger than ||C_row_j||
        at high rho(Ap). It fixed the diagnosed cold-start scaled violation
        (1.7e12 -> 1.0) but a full end-to-end comparison against this baseline
        (evaluating the mapped-back solution on the ORIGINAL canonical QP, not
        just the cold-start point) showed it REGRESSES the full 8000-iteration
        solve on both a benign and a stiff test state -- feasibility got worse
        in both cases (easy: 10.1 -> 309; stiff: 1.59e5 -> 2.54e5), with no
        iteration-count benefit anywhere. Rejected; kept as the original
        formula below. See results/qp_conditioning_change_report.json.
        """
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
        """Linearised Ap, Bp for the current step, from the shared canonical
        model (src/dynamics.py), using this controller's configured
        trust_region setting.

        HISTORY: this used to be hardcoded trust_region=True while the CVXPY
        baseline used False, which meant the two controllers were predicting
        with different models (max|dAp| up to ~1851 in the gelation region)
        and therefore could not be compared head-to-head. It is now an
        explicit constructor parameter defaulting to False on BOTH
        controllers, so the model-identical configuration is the default and
        any divergence has to be asked for deliberately."""
        return linearize(T0_degC, alpha0, trust_region=self.trust_region)

    def build_qp(self, current_state, u_prev):
        """Build this step's canonical QP via the shared condensation path
        (src/qp_builder.py) -- the SAME construction the CVXPY adapter uses,
        so both controllers solve numerically identical (H, f, A_ineq, b_ineq)
        given the same (Ap, Bp, x0, u_prev). See qp-contract skill.

        The linearized Ap is used as-is (trust_region=True bounds only the
        exotherm Jacobian terms feeding into it, in src/dynamics.py -- this
        method does not itself shrink or otherwise modify Ap). During the
        exothermic gelation phase its spectral radius can exceed 1 (the local
        linear model genuinely predicts thermal runaway -- which is precisely
        what the controller must anticipate to brake in time); Jacobi
        preconditioning (_condition), applied downstream, keeps the condensed
        QP solvable without erasing that signal."""
        avg_T = float(np.mean(current_state[0:3]))
        avg_a = float(np.mean(current_state[7:10]))
        Ap, Bp = self.update_matrices(avg_T, avg_a)
        return build_canonical_qp(
            Ap, Bp, current_state, u_prev, self.N,
            self.Q_diag, self.R_val, self.S_val, self.target_temp,
            trust_region=self.trust_region,
            soft_state_constraints=self.soft_state_constraints,
        )

    def build_dense_qp(self, Ap, Bp, x0, u_prev):
        """Backward-compatible (H, g, C, d) view of the canonical QP, in the
        Cx+d<=0 convention _condition()/SNNSolver expect (C=A_ineq, d=-b_ineq).
        Kept for callers that already extract Ap/Bp themselves (diagnostic
        probes under tools/); build_qp() is the preferred entry point."""
        qp = build_canonical_qp(
            Ap, Bp, x0, u_prev, self.N,
            self.Q_diag, self.R_val, self.S_val, self.target_temp,
            trust_region=self.trust_region,
            soft_state_constraints=self.soft_state_constraints,
        )
        return qp.H, qp.f, qp.A_ineq, -qp.b_ineq

    def compute_control_action(self, current_state, u_prev):
        t0 = time.time()

        qp = self.build_qp(current_state, u_prev)
        H_raw, g_raw, C_raw, d_raw = qp.H, qp.f, qp.A_ineq, -qp.b_ineq

        # Precondition the matrix to unlock solver speed
        H_s, g_s, C_s, d_s, D = self._condition(H_raw, g_raw, C_raw, d_raw)

        if not (np.isfinite(H_s).all() and np.isfinite(g_s).all()):
            self.U_warm = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0

        n_total = H_raw.shape[0]
        U_raw = (self._warm_hold(u_prev, n_total)
                 if (self.U_warm is None or self.U_warm.shape[0] != n_total)
                 else self.U_warm)

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

            # Input-feasibility projection (safety filter). This is the exact
            # Euclidean projection of the first control onto its feasible set
            # {u : |u - u_prev| <= rate, TA_MIN <= u <= TA_MAX} -- clipping a
            # scalar onto an interval IS its projection. It is NOT an arbitrary
            # hack: it guarantees the *applied* input is admissible even when the
            # projection-based SNN solver leaves a residual constraint violation
            # in its horizon plan. Ideally it is inactive; we count when it isn't.
            u_raw = float(U_sol[0])
            lo = max(const.TA_MIN, u_prev - const.TA_RATE_MAX)
            hi = min(const.TA_MAX, u_prev + const.TA_RATE_MAX)
            u_out = float(np.clip(u_raw, lo, hi))
            if abs(u_out - u_raw) > 1e-6:
                self.n_projection_active += 1

            self.U_warm = self._shift(U_sol)
            return u_out, solve_time

        except Exception as exc:
            self.U_warm = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0