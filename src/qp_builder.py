"""
qp_builder.py
Canonical per-step MPC-QP construction. This is the SINGLE condensation path
for both the CVXPY and SNN-QP controllers -- see
docs/PHASE4_VALIDATION_REPORT.md §3 for the formal definition and §4 for why
the soft-constraint variant exists.

CONTRACT: both controllers must be constructed with the same `trust_region`
and `soft_state_constraints` settings. Given identical (Ap, Bp, x0, u_prev)
this function then produces bit-identical (H, f, A_ineq, b_ineq), bounds and
variable ordering for both -- verified over a full 160-step trajectory
(zero difference on every array at every step) and unit-tested in
tests/test_qp_parity.py.

Condenses the linearised plant (Ap, Bp) plus the MPC cost/constraints into one
dense QP over the decision variable z = [Ta_0, ..., Ta_{N-1}]:

    min_z  0.5 z^T H z + f^T z
    s.t.   A_ineq @ z <= b_ineq

Row i of the internal Phi/Gamma matrices predicts x_i (Phi[0] = Ap^0 = I),
matching the horizon the CVXPY controller's original symbolic formulation
actually costs and constrains: x_0 is pinned (not decision-dependent, so its
cost term is a constant offset), the dynamics constraint x_{k+1} = Ap@x_k +
Bp@u_k defines x_1..x_N, and the cost/gradient-constraint sums range over
k=0..N-1, i.e. x_0..x_{N-1}. Using exponent s=i+1 instead (row i predicts
x_{i+1}) is a plausible, easy-to-miss off-by-one that produces a well-posed
but DIFFERENT QP -- verified empirically (tools/qp_parity_probe.py) to diverge
from the live CVXPY solve by up to ~1.2 degC during and after the gelation
exotherm, while this s=i construction matches it to ~0.02 degC.

Box constraints (Ta_min <= u_k <= Ta_max) are embedded as explicit A_ineq rows
rather than native variable bounds, matching the existing SNN adapter's
approach: `SolverConfig`'s native scalar bounds were already found to "break
in scaled vector space" (see snn_mpc_controller.py history), so both adapters
route ALL constraints through the same A_ineq/b_ineq -- lower_bound/upper_bound
on CanonicalQP are +-inf and exist only as documented metadata.

`trust_region` is accepted as an input (it is already baked into the caller's
Ap, Bp by src/dynamics.linearize) and recorded in `linearization`; it is never
re-derived here. It defaults to False on both controllers, so the
model-identical configuration is the default.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

import src.constants as const

NX = 10  # 3 composite temps + 4 tooling temps + 3 cure states


@dataclass
class CanonicalQP:
    H: np.ndarray
    f: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    A_eq: object
    b_eq: object
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    variable_order: list
    scaling: object
    linearization: dict
    reference: dict
    n_inputs: int = 0      # first n_inputs entries of z are the Ta sequence
    n_slacks: int = 0      # remaining entries (if any) are gradient-constraint slacks
    gradient_rows: dict = None   # relative degree, kept/dropped rows, unactionable violation

    def fingerprint(self, decimals=8):
        """Deterministic identifier for this QP instance (see docs/PHASE4_VALIDATION_REPORT.md §3)."""
        blob = b"".join(
            np.round(a, decimals).tobytes()
            for a in (self.H, self.f, self.A_ineq, self.b_ineq)
        )
        return hashlib.sha256(blob).hexdigest()[:16]


def build_canonical_qp(Ap, Bp, x0, u_prev, N, Q_diag, R_val, S_val, target_temp,
                        trust_region, soft_state_constraints=False,
                        slack_weight_quad=1.0e2, slack_weight_lin=1.0e3,
                        drop_uncontrollable_rows=True, constraint_horizon=None,
                        grad_row_zero_tol=1.0e-12):
    """Build the canonical dense QP shared by both controllers.

    Parameters (the full set that determines the QP): Ap, Bp
    (from src.dynamics.linearize), x0 (current 10-vector plant state), u_prev
    (previous applied Ta), N (horizon), Q_diag/R_val/S_val (cost weights),
    target_temp, and trust_region (recorded, not re-derived -- caller already
    chose it when calling linearize()).

    soft_state_constraints : bool
        False (default): the original hard-constraint form; z = [Ta_0..Ta_{N-1}].
        True: the predicted-state (gradient) rows gain non-negative slacks with
        an exact-penalty cost; z = [Ta_0..Ta_{N-1}, s_0..s_{m-1}]. Actuator box
        and slew rows stay hard in both cases. Whatever is chosen must be
        applied IDENTICALLY to both controllers -- it is part of the shared
        canonical problem, not a per-solver adjustment.
    drop_uncontrollable_rows : bool
        True (default): omit gradient rows whose normal is exactly zero, i.e.
        the rows inside the plant's input-to-gradient dead time. They contain
        no decision variable and are therefore not constraints; keeping them
        makes the QP unconditionally infeasible whenever the free response
        breaches the limit, and starves the SNN's projection selector. What
        they would have said is reported in `gradient_rows` instead of being
        discarded. False reproduces the pre-Revision-5 QP for A/B comparison.
    constraint_horizon : int or None
        Impose gradient rows only for k < constraint_horizon. None (default)
        keeps every live row. Use to bound the constraint set to the window
        where the frozen-Jacobian prediction is trustworthy.
    grad_row_zero_tol : float
        Squared-norm-equivalent threshold below which a gradient row normal
        counts as structurally zero. Matches snn_opt's own 1e-12 skip rule, so
        the two agree on which rows are degenerate.

    The number of gradient rows is therefore state-dependent. Both controllers
    call this function, so they still receive bit-identical arrays; the parity
    claim is unaffected, but code that hard-codes `6N` rows is not.
    """
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

    Q_bar = np.kron(np.eye(N), np.diag(Q_diag))
    R_bar = np.eye(N) * R_val

    Diff = np.eye(N)
    for i in range(1, N):
        Diff[i, i - 1] = -1.0
    S_bar = np.eye(N) * S_val

    d0 = np.zeros(N)
    d0[0] = u_prev

    X_ref = np.zeros(N * NX)
    for i in range(N):
        X_ref[i * NX: i * NX + 3] = target_temp

    H = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar + Diff.T @ S_bar @ Diff)
    H = (H + H.T) / 2.0
    H += np.eye(N) * 1e-3

    E = Phi @ x0 - X_ref
    f = 2.0 * (Gamma.T @ Q_bar @ E) - 2.0 * (Diff.T @ S_bar @ d0)

    # ---- Actuator constraints: ALWAYS hard ---------------------------------
    # Box (Ta_min <= u <= Ta_max) and slew (|u_k - u_{k-1}| <= TA_RATE_MAX,
    # u_{-1} = u_prev). Embedded as A_ineq rows rather than native bounds --
    # see the module docstring.
    A_hard_u = np.vstack([np.eye(N), -np.eye(N), Diff, -Diff])
    b_hard_u = np.concatenate([
        np.ones(N) * const.TA_MAX,
        -np.ones(N) * const.TA_MIN,
        d0 + const.TA_RATE_MAX,
        const.TA_RATE_MAX - d0,
    ])

    # ---- Gradient (output) constraint: |x[0,k] - x[2,k]| <= GRADIENT_MAX ----
    G = np.zeros((N, N * NX))
    for i in range(N):
        G[i, i * NX + 0] = 1.0
        G[i, i * NX + 2] = -1.0
    GGamma = G @ Gamma
    GPhi_x = G @ Phi @ x0
    b_grad_hi = const.GRADIENT_MAX - GPhi_x     # +(Tc1-Tc3)_k <= GRADIENT_MAX
    b_grad_lo = const.GRADIENT_MAX + GPhi_x     # -(Tc1-Tc3)_k <= GRADIENT_MAX

    # RELATIVE DEGREE / DEAD ROWS.  Row k of GGamma is c^T [A^{k-1} B ... B],
    # so it is the ZERO VECTOR whenever c^T A^p B = 0 for every p <= k-1. For
    # this plant Ta enters at the outer tooling node and has to diffuse inward
    # one node per sample, so the gradient output (Tc1 - Tc3) has relative
    # degree r = 5: rows k = 0..4 are EXACTLY zero at every operating point and
    # every horizon. Such a row is not a constraint -- it reduces to
    # `0 <= GRADIENT_MAX -+ (Phi x0)_k`, a predicate on the CURRENT state that
    # no input can influence. Keeping it does two kinds of damage:
    #   * if the predicate is false the QP is UNCONDITIONALLY infeasible, at
    #     any horizon, for any solver (this is the documented `0 <= -2.25e2`);
    #   * snn_opt skips rows with squared norm < 1e-12 WITHOUT incrementing its
    #     projection counter or changing the residual, so the selector re-picks
    #     the same dead row forever -- the `n_projections = 0` anomaly.
    # This is a documented MPC failure mode: the MathWorks MPC Toolbox states
    # the same rule for output-variable constraints inside a plant's delay
    # (a plant with 5 samples of delay cannot satisfy an OV constraint before
    # the 6th prediction step). The rows are dropped from the QP and reported
    # instead, via `gradient_rows` below -- dropping them silently would turn a
    # real physical limitation (a predicted excursion the actuator cannot
    # pre-empt within the dead time) into an invisible one.
    row_norm = np.linalg.norm(GGamma, axis=1)
    dead_mask = row_norm <= grad_row_zero_tol
    live_idx = np.flatnonzero(~dead_mask)
    relative_degree = int(live_idx[0]) if live_idx.size else N

    keep_mask = ~dead_mask if drop_uncontrollable_rows else np.ones(N, bool)

    # Optional constraint horizon: impose gradient rows only for k < Nc. The
    # prediction is a frozen-Jacobian extrapolation, and during gelation
    # rho(Ap) > 1 amplifies it geometrically (measured: ~1798 degC predicted
    # against ~7 degC actual, ten steps out). Rows beyond the trustworthy
    # window are constraints on a number the model does not know. None = keep
    # every live row, i.e. Nc = N.
    if constraint_horizon is not None:
        horizon_mask = np.arange(N) < int(constraint_horizon)
        dropped_horizon = np.flatnonzero(keep_mask & ~horizon_mask)
        keep_mask = keep_mask & horizon_mask
    else:
        dropped_horizon = np.array([], dtype=int)

    kept = np.flatnonzero(keep_mask)
    n_grad = kept.size

    # What the dropped-as-uncontrollable rows WOULD have said. Positive means
    # the free response is predicted to breach GRADIENT_MAX inside the dead
    # time -- physically meaningful, and unactionable by construction.
    dropped_dead = np.flatnonzero(dead_mask) if drop_uncontrollable_rows else np.array([], dtype=int)
    uncontrollable_violation = float(max(
        [0.0] + [max(-b_grad_hi[i], -b_grad_lo[i]) for i in dropped_dead]
    ))

    A_grad_u = np.vstack([GGamma[kept], -GGamma[kept]]) if n_grad else np.zeros((0, N))
    b_grad = np.concatenate([b_grad_hi[kept], b_grad_lo[kept]]) if n_grad else np.zeros(0)

    n_slacks = 0
    if not soft_state_constraints:
        A_ineq = np.vstack([A_hard_u, A_grad_u])
        b_ineq = np.concatenate([b_hard_u, b_grad])
    else:
        # ---- Soft (slack) reformulation of the GRADIENT rows only ----------
        # Input box + slew rows stay HARD: they are actuator limits, always
        # satisfiable, and softening them would be exactly the "hide an
        # infeasibility" anti-pattern. Only the predicted-STATE constraint is
        # softened -- standard MPC practice, and necessary here because at
        # rho(Ap)>1 the horizon-end gradient rows are amplified by Ap^(N-1)
        # into a genuinely infeasible set (OSQP reports `infeasible` on the
        # hard form too, so this is not an SNN-specific rescue).
        #
        #   z = [Ta_0..Ta_{N-1}, s_0..s_{m-1}],  s >= 0,  m = # kept rows
        #   |x[0,k]-x[2,k]| <= GRADIENT_MAX + s_k   for the kept k only
        #   cost += slack_weight_quad * ||s||^2 + slack_weight_lin * sum(s)
        #
        # One slack per kept row index, shared by that row's upper and lower
        # halves (a two-sided constraint cannot be violated in both directions
        # at once, so a shared slack is exact and halves the variable count).
        #
        # The exact-penalty linear term makes s_k collapse to 0 whenever the
        # hard constraint IS attainable -- provided slack_weight_lin exceeds
        # the dual norm of the hard problem's multipliers (Kerrigan &
        # Maciejowski). That condition is measurable and is measured by
        # tools/exact_penalty_audit.py; it is NOT assumed here.
        n_slacks = n_grad
        n_total = N + n_slacks

        H_full = np.zeros((n_total, n_total))
        H_full[:N, :N] = H
        if n_slacks:
            H_full[N:, N:] = np.eye(n_slacks) * (2.0 * slack_weight_quad)

        f_full = np.zeros(n_total)
        f_full[:N] = f
        f_full[N:] = slack_weight_lin

        A_hard = np.zeros((A_hard_u.shape[0], n_total))
        A_hard[:, :N] = A_hard_u

        # Gradient rows gain -s_k on the left-hand side.
        A_grad_s = np.zeros((2 * n_grad, n_total))
        if n_grad:
            A_grad_s[:, :N] = A_grad_u
            A_grad_s[:n_grad, N:] = -np.eye(n_grad)
            A_grad_s[n_grad:, N:] = -np.eye(n_grad)

        # Slack non-negativity: -s_k <= 0
        A_pos = np.zeros((n_slacks, n_total))
        if n_slacks:
            A_pos[:, N:] = -np.eye(n_slacks)

        H = H_full
        f = f_full
        A_ineq = np.vstack([A_hard, A_grad_s, A_pos])
        b_ineq = np.concatenate([b_hard_u, b_grad, np.zeros(n_slacks)])

    avg_T = float(np.mean(x0[0:3]))
    avg_a = float(np.mean(x0[7:10]))
    n_total = N + n_slacks
    var_order = [f"Ta_{i}" for i in range(N)] + [f"slack_grad_{i}" for i in range(n_slacks)]

    return CanonicalQP(
        H=H, f=f, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=None, b_eq=None,
        lower_bound=np.full(n_total, -np.inf), upper_bound=np.full(n_total, np.inf),
        variable_order=var_order,
        scaling=None,
        linearization={
            "Ap": Ap, "Bp": Bp, "trust_region": trust_region,
            "avg_T": avg_T, "avg_a": avg_a,
        },
        reference={
            "x0": np.asarray(x0), "u_prev": float(u_prev), "N": N,
            "Q_diag": np.asarray(Q_diag), "R_val": R_val, "S_val": S_val,
            "target_temp": target_temp,
            "soft_state_constraints": soft_state_constraints,
            "slack_weight_quad": slack_weight_quad,
            "slack_weight_lin": slack_weight_lin,
            "drop_uncontrollable_rows": drop_uncontrollable_rows,
            "constraint_horizon": constraint_horizon,
        },
        n_inputs=N, n_slacks=n_slacks,
        gradient_rows={
            # Structural: the first row index whose normal is non-zero, i.e.
            # the input-to-gradient dead time in samples. Constant for this
            # plant (5) because it follows from the diffusion stencil, but
            # measured per build rather than assumed.
            "relative_degree": relative_degree,
            "n_total": N,
            "n_kept": int(n_grad),
            "kept": kept.tolist(),
            "dropped_uncontrollable": dropped_dead.tolist(),
            "dropped_beyond_constraint_horizon": dropped_horizon.tolist(),
            # Largest amount (degC) by which the free response is predicted to
            # breach GRADIENT_MAX on a row the input cannot reach. 0.0 means
            # every dropped row was satisfied anyway. This is the honest
            # residual of dropping them: a predicted excursion, not a solver
            # failure and not something the controller could have prevented.
            "unactionable_predicted_violation_degC": uncontrollable_violation,
            "row_norms": row_norm.tolist(),
        },
    )
