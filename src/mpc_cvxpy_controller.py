"""
mpc_cvxpy_controller.py
Closed-loop MPC solver using Phase-Wise Linearization.
"""
import numpy as np
import cvxpy as cp
import time
import src.constants as const
from src.dynamics import linearize, linearize_trajectory, shift_nominal_sequence
from src.qp_builder import build_canonical_qp

class MPCSolver:
    def __init__(self, horizon=20, target_temp=120.0, trust_region=False,
                 soft_state_constraints=False, drop_uncontrollable_rows=True,
                 constraint_horizon=None, linearization_mode='lti',
                 ltv_nominal_source='warm_start'):
        self.N = horizon
        self.nx = 10  # 7 Temps, 3 Alphas
        self.nu = 1   # 1 Input (Ta)
        self.target_temp = target_temp
        # Prediction-model and QP-form options. These are part of the SHARED
        # problem definition: whatever is set here must be set identically on
        # SNNMPCSolver for the two controllers to be solving the same problem.
        self.trust_region = trust_region
        self.soft_state_constraints = soft_state_constraints
        # Constraint-set options -- also part of the shared problem, and also
        # subject to the identical-on-both-controllers rule.
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
        # Where the LTV nominal sequence comes from (ignored under 'lti').
        # 'warm_start' (default): shift THIS controller's own previous solve
        # by one step -- the standard successive-linearization convention,
        # and what every result in README_LTV.md before the ablation section
        # used. 'constant': always hold u_prev across the horizon, i.e. never
        # read self._u_nominal -- removes the path-dependent memory of the
        # controller's own solve history, isolating whether that memory (as
        # opposed to LTV re-linearization itself) is what grew the RMS
        # applied-control difference between the two controllers (README_LTV
        # .md Caveat 2). Must be set identically on both controllers, same
        # rule as trust_region.
        if ltv_nominal_source not in ('warm_start', 'constant'):
            raise ValueError(f"ltv_nominal_source must be 'warm_start' or 'constant', got {ltv_nominal_source!r}")
        self.ltv_nominal_source = ltv_nominal_source
        # The nominal control sequence LTV mode re-linearizes along. This is
        # NOT a solver warm start (CVXPY manages its own via warm_start=True)
        # -- it is part of the shared problem definition and must be
        # computed identically to SNNMPCSolver's, or the two controllers'
        # Ap_seq/Bp_seq (and therefore their QPs) silently diverge.
        self._u_nominal = None

        # Tuning Weights (canonical form: Q as a diagonal vector, R/S as
        # scalars -- matches SNNMPCSolver's Q_diag/R_val/S_val so both
        # controllers feed build_canonical_qp identical ingredients).
        self.Q_diag = np.zeros(self.nx)
        self.Q_diag[0:3] = 100.0  # Heavily penalize composite temps deviating from target
        self.R_val = 0.1          # Penalty for absolute effort (kept small)
        self.S_val = 1.0          # Penalty for rate of change (Delta u)

    def update_matrices(self, T0_degC, alpha0):
        """Linearised Ap, Bp for the current step, from the shared canonical
        model (src/dynamics.py), using this controller's configured
        trust_region setting."""
        return linearize(T0_degC, alpha0, trust_region=self.trust_region)

    def build_qp(self, current_state, u_prev):
        """Build this step's canonical QP via the shared condensation path
        (src/qp_builder.py) -- the SAME construction the SNN adapter uses,
        so both controllers solve numerically identical (H, f, A_ineq, b_ineq)
        given the same (Ap, Bp, x0, u_prev). See docs/PHASE4_VALIDATION_REPORT.md §3."""
        if self.linearization_mode == 'ltv':
            prior = self._u_nominal if self.ltv_nominal_source == 'warm_start' else None
            u_nominal = shift_nominal_sequence(prior, u_prev, self.N)
            Ap, Bp = linearize_trajectory(current_state, u_nominal, trust_region=self.trust_region)
        else:
            avg_T = np.mean(current_state[0:3])
            avg_a = np.mean(current_state[7:10])
            Ap, Bp = self.update_matrices(avg_T, avg_a)
        return build_canonical_qp(
            Ap, Bp, current_state, u_prev, self.N,
            self.Q_diag, self.R_val, self.S_val, self.target_temp,
            trust_region=self.trust_region,
            soft_state_constraints=self.soft_state_constraints,
            drop_uncontrollable_rows=self.drop_uncontrollable_rows,
            constraint_horizon=self.constraint_horizon,
        )

    def compute_control_action(self, current_state, u_prev):
        """Calculates optimal Ta input over prediction horizon."""
        start_time = time.time()

        qp = self.build_qp(current_state, u_prev)

        U = cp.Variable(qp.H.shape[0])   # N inputs, plus N slacks when softened
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(qp.H)) + qp.f @ U),
            [qp.A_ineq @ U <= qp.b_ineq],
        )

        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            solve_time = (time.time() - start_time) * 1000 # Convert to ms

            if prob.status in ['optimal_inaccurate', cp.OPTIMAL_INACCURATE]:
                print(f" MPC LOG: Solve status '{prob.status}'. Linear approximation is struggling with the exotherm!")

            if U.value is None:
                print(f" MPC WARNING: Solver returned None. Falling back to previous input: {u_prev:.2f}°C")
                self._u_nominal = None
                return u_prev, solve_time
            self._u_nominal = np.asarray(U.value[:self.N], dtype=float)
            return U.value[0], solve_time
        except Exception as e:
            print(f" MPC CRITICAL ERROR: Solver crashed! Exception: {e}")
            print(f" Falling back to previous input: {u_prev:.2f}°C")
            self._u_nominal = None
            return u_prev, (time.time() - start_time) * 1000