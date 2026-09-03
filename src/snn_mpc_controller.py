"""
snn_mpc_controller.py
Closed-loop MPC solver utilizing Spiking Neural Networks (LIF Dynamics).
Final Architecture: Preconditioned Jacobi Scaling with Matrix-Embedded Box Constraints.
"""

import dataclasses
import numpy as np
import time
import src.constants as const
from src.dynamics import linearize, linearize_trajectory, shift_nominal_sequence
from src.qp_builder import build_canonical_qp
import snn_opt
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig

# snn_opt >= 0.6.0 replaced the ABSOLUTE projected-gradient stopping test with a
# scale-invariant KKT-cone certificate, selected via
# ConvergenceConfig.optimality_test. `proj_grad_tol` survives only as a
# deprecated constructor-only alias -- and passing it SILENTLY forces
# optimality_test='legacy_projected_gradient', so an upgrade alone leaves the old
# scale-sensitive test in place. We therefore detect the field and configure
# explicitly rather than relying on defaults.
_HAS_KKT_CERTIFICATE = any(
    f.name == "optimality_test" for f in dataclasses.fields(ConvergenceConfig))

# 0.5.0 also turned max_projection_iters from a per-call cap into a hard
# WATCHDOG: exceeding it aborts the solve with `projection_budget_exhausted`
# instead of continuing. Under 0.6.0 the old budget of 200 aborts after ~1
# outer iteration, the solver returns essentially its cold start, and the
# closed loop STOPS CURING (measured: final alpha 0.0000, max Tc1 41.8 degC).
# 2000 restores full cure. See docs/PHASE4_VALIDATION_REPORT.md section 11.1
# and results/kkt_certificate_probe.json.
#
# REVISION 5: 2000 was still too small, and this was the real stiff-window
# defect. Measured on 31 frozen stiff-exotherm states (N=10, soft, k0=0.1),
# `convergence_reason` was `projection_budget_exhausted` on 15 of 31 -- the
# solver ABORTING after ~130 of its 8000 permitted iterations, not failing to
# converge. Raising the budget to 5000 eliminates every such abort and lifts
# formal convergence there from 16.1 % to 25.8 %. It saturates at 5000:
# 20000, 100000 and 500000 all give exactly 25.8 %, so this is a real
# threshold, not a knob to keep turning.
#
# The cost is large and must not be hidden: the median stiff-window solve goes
# from ~9.5 ms to ~295 ms. That is not a regression -- the old figure was fast
# BECAUSE the solver was giving up early. Part of the previously reported speed
# was premature abort, and the corrected number is the true cost of attempting
# the solve.
#
# The residual 74 % of stiff steps terminate on `max_iterations`, and raising
# THAT does nothing: 8000 / 30000 / 100000 all give 25.8 % at 12x the time.
# Those steps plateau without meeting the KKT test and are a genuine limit of
# the projected-gradient method on this QP, not a budget problem.
# See results/solver_budget_experiment.json.
_MAX_PROJECTION_ITERS = 5000 if _HAS_KKT_CERTIFICATE else 200


class SNNMPCSolver:
    def __init__(self, horizon=20, target_temp=120.0, trust_region=False,
                 soft_state_constraints=False, k0_scale=0.5,
                 drop_uncontrollable_rows=True, constraint_horizon=None,
                 linearization_mode='lti'):
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
        # Constraint-set options -- also part of the shared problem, and also
        # subject to the identical-on-both-controllers rule. Dropping the
        # structurally-zero gradient rows is what stops snn_opt's projection
        # selector from re-picking a dead row forever (the n_projections = 0
        # anomaly); it is a correctness fix, not a solver tweak.
        self.drop_uncontrollable_rows = drop_uncontrollable_rows
        self.constraint_horizon = constraint_horizon
        # 'lti' (default): single frozen Jacobian reused across the horizon,
        # unchanged from before LTV support existed. 'ltv': re-linearize at
        # each horizon step along a nominal trajectory (src.dynamics
        # .linearize_trajectory) -- see the branch README for why. MUST be
        # set identically on both controllers, same rule as trust_region.
        if linearization_mode not in ('lti', 'ltv'):
            raise ValueError(f"linearization_mode must be 'lti' or 'ltv', got {linearization_mode!r}")
        self.linearization_mode = linearization_mode
        # The nominal control sequence LTV mode re-linearizes along. This is
        # a SEPARATE concept from self.U_warm (this solver's own cold-start
        # point for the projected-gradient iteration, in SCALED space) -- it
        # is part of the shared problem definition and must be computed
        # identically to MPCSolver's, or the two controllers' Ap_seq/Bp_seq
        # silently diverge.
        self._u_nominal = None

        _conv_common = dict(
            enable_early_stopping=True,
            check_every=50,
            min_iterations=100,
            patience=3,
            obj_rel_tol=1e-7,
            feasibility_tol=1e-2,
        )
        if _HAS_KKT_CERTIFICATE:
            # Scale-invariant certificate. Must be requested EXPLICITLY and
            # WITHOUT proj_grad_tol, or snn_opt falls back to the legacy
            # absolute test (see module header).
            conv_config = ConvergenceConfig(
                optimality_test='kkt',
                kkt_abs_tol=1e-9,
                kkt_rel_tol=1e-4,
                **_conv_common,
            )
        else:
            # snn_opt 0.4.x: only the absolute projected-gradient test exists.
            # Documented as a scale-sensitive heuristic -- its 5e-2 threshold is
            # compared against a norm of order 1e10 at N=20, so it cannot fire.
            conv_config = ConvergenceConfig(proj_grad_tol=5e-2, **_conv_common)

        self.solver_config = SolverConfig(
            k0=None,            # Auto-computes from the Lipschitz constant of the
            k0_scale=k0_scale,  # Jacobi-conditioned Hessian (stable margin).
            projection_method='adaptive',
            max_iterations=8000,
            max_projection_iters=_MAX_PROJECTION_ITERS,  # version-dependent -- see
                                        # module header. 0.4.x: per-call cap (200).
                                        # >=0.5.0: hard watchdog, needs 2000.
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

        # Steps where snn_opt (>=0.6.0) refused the QP as certifiably infeasible
        # -- a zero-normal constraint row with d > 0. Expected to stay 0 on the
        # recommended soft form; non-zero indicates the hard form is in use.
        self.n_infeasible_qp = 0
        self.last_infeasibility_reason = None

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
        given the same (Ap, Bp, x0, u_prev). See docs/PHASE4_VALIDATION_REPORT.md §3.

        The linearized Ap is used as-is (trust_region=True bounds only the
        exotherm Jacobian terms feeding into it, in src/dynamics.py -- this
        method does not itself shrink or otherwise modify Ap). During the
        exothermic gelation phase its spectral radius can exceed 1 (the local
        linear model genuinely predicts thermal runaway -- which is precisely
        what the controller must anticipate to brake in time); Jacobi
        preconditioning (_condition), applied downstream, keeps the condensed
        QP solvable without erasing that signal."""
        if self.linearization_mode == 'ltv':
            u_nominal = shift_nominal_sequence(self._u_nominal, u_prev, self.N)
            Ap, Bp = linearize_trajectory(current_state, u_nominal, trust_region=self.trust_region)
        else:
            avg_T = float(np.mean(current_state[0:3]))
            avg_a = float(np.mean(current_state[7:10]))
            Ap, Bp = self.update_matrices(avg_T, avg_a)
        return build_canonical_qp(
            Ap, Bp, current_state, u_prev, self.N,
            self.Q_diag, self.R_val, self.S_val, self.target_temp,
            trust_region=self.trust_region,
            soft_state_constraints=self.soft_state_constraints,
            drop_uncontrollable_rows=self.drop_uncontrollable_rows,
            constraint_horizon=self.constraint_horizon,
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
            drop_uncontrollable_rows=self.drop_uncontrollable_rows,
            constraint_horizon=self.constraint_horizon,
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
            self._u_nominal = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0

        n_total = H_raw.shape[0]
        U_raw = (self._warm_hold(u_prev, n_total)
                 if (self.U_warm is None or self.U_warm.shape[0] != n_total)
                 else self.U_warm)

        # Math Fix: Shift into Scaled Space (*)
        U_warm_scaled = U_raw * D

        problem = OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s)

        try:
            solver = SNNSolver(problem, self.solver_config)
        except ValueError as exc:
            # snn_opt >= 0.6.0 refuses to construct on a CERTIFIABLY INFEASIBLE
            # problem: a constraint row with a zero normal and d > 0 reduces to
            # `0 <= negative`, unsatisfiable for every z. This is correct, and it
            # independently reproduces this project's own finding -- the hard
            # form's gradient rows k=0..4 are exactly decision-independent
            # (docs/PHASE4_VALIDATION_REPORT.md section 4.1). It fires on the
            # HARD form; the recommended soft form is feasible and unaffected.
            # Count it and hold the previous input rather than crashing the loop.
            self.n_infeasible_qp += 1
            self.last_infeasibility_reason = str(exc)
            self.U_warm = None
            self._u_nominal = None
            return (float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)),
                    (time.time() - t0) * 1000.0)

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
            self._u_nominal = np.asarray(U_sol[:self.N], dtype=float)
            return u_out, solve_time

        except Exception as exc:
            self.U_warm = None
            self._u_nominal = None
            return float(np.clip(u_prev, const.TA_MIN, const.TA_MAX)), (time.time() - t0) * 1000.0