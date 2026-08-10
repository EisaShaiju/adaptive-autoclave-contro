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

`trust_region` remains the one documented, permitted divergence between the
two controllers (see src/dynamics.py) -- it is accepted as an input (baked
into the caller's Ap, Bp) and recorded in `linearization`, never silently
re-derived here.
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

    def fingerprint(self, decimals=8):
        """Deterministic identifier for this QP instance (qp-contract skill)."""
        blob = b"".join(
            np.round(a, decimals).tobytes()
            for a in (self.H, self.f, self.A_ineq, self.b_ineq)
        )
        return hashlib.sha256(blob).hexdigest()[:16]


def build_canonical_qp(Ap, Bp, x0, u_prev, N, Q_diag, R_val, S_val, target_temp,
                        trust_region, soft_state_constraints=False,
                        slack_weight_quad=1.0e2, slack_weight_lin=1.0e3):
    """Build the canonical dense QP shared by both controllers.

    Parameters mirror the ingredients table in the model-parity skill: Ap, Bp
    (from src.dynamics.linearize), x0 (current 10-vector plant state), u_prev
    (previous applied Ta), N (horizon), Q_diag/R_val/S_val (cost weights),
    target_temp, and trust_region (recorded, not re-derived -- caller already
    chose it when calling linearize()).

    soft_state_constraints : bool
        False (default): the original hard-constraint form; z = [Ta_0..Ta_{N-1}].
        True: the predicted-state (gradient) rows gain non-negative slacks with
        an exact-penalty cost; z = [Ta_0..Ta_{N-1}, s_0..s_{N-1}]. Actuator box
        and slew rows stay hard in both cases. Whatever is chosen must be
        applied IDENTICALLY to both controllers -- it is part of the shared
        canonical problem, not a per-solver adjustment.
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

    A_rows, b_rows = [], []

    # Box constraints (Ta_min <= u <= Ta_max), embedded as A_ineq rows.
    A_rows.append(np.eye(N));  b_rows.append(np.ones(N) * const.TA_MAX)
    A_rows.append(-np.eye(N)); b_rows.append(-np.ones(N) * const.TA_MIN)

    # Slew-rate constraints (|u_k - u_{k-1}| <= TA_RATE_MAX, u_{-1} = u_prev).
    A_rows.append(Diff);  b_rows.append(d0 + const.TA_RATE_MAX)
    A_rows.append(-Diff); b_rows.append(const.TA_RATE_MAX - d0)

    # Gradient constraint (|x[0,k] - x[2,k]| <= GRADIENT_MAX) for k=0..N-1.
    G = np.zeros((N, N * NX))
    for i in range(N):
        G[i, i * NX + 0] = 1.0
        G[i, i * NX + 2] = -1.0
    GGamma = G @ Gamma
    GPhi_x = G @ Phi @ x0
    A_rows.append(GGamma);  b_rows.append(const.GRADIENT_MAX - GPhi_x)
    A_rows.append(-GGamma); b_rows.append(const.GRADIENT_MAX + GPhi_x)

    A_ineq = np.vstack(A_rows)
    b_ineq = np.concatenate(b_rows)
    n_slacks = 0

    if soft_state_constraints:
        # ---- Soft (slack) reformulation of the GRADIENT rows only ----------
        # Input box + slew rows stay HARD: they are actuator limits, always
        # satisfiable, and softening them would be exactly the "hide an
        # infeasibility" anti-pattern. Only the predicted-STATE constraint is
        # softened -- standard MPC practice, and necessary here because at
        # rho(Ap)>1 the horizon-end gradient rows are amplified by Ap^(N-1)
        # into a genuinely infeasible set (OSQP reports `infeasible` on the
        # hard form too, so this is not an SNN-specific rescue).
        #
        #   z = [Ta_0..Ta_{N-1}, s_0..s_{N-1}],  s >= 0
        #   |x[0,k]-x[2,k]| <= GRADIENT_MAX + s_k
        #   cost += slack_weight_quad * ||s||^2 + slack_weight_lin * sum(s)
        #
        # The exact-penalty linear term makes s_k collapse to 0 whenever the
        # hard constraint IS attainable, so this does not silently relax a
        # constraint that could have been met.
        n_slacks = N
        n_total = N + n_slacks

        H_full = np.zeros((n_total, n_total))
        H_full[:N, :N] = H
        H_full[N:, N:] = np.eye(n_slacks) * (2.0 * slack_weight_quad)

        f_full = np.zeros(n_total)
        f_full[:N] = f
        f_full[N:] = slack_weight_lin

        n_hard = 4 * N          # box (2N) + slew (2N) rows, u-only
        A_hard = np.zeros((n_hard, n_total))
        A_hard[:, :N] = A_ineq[:n_hard, :]
        b_hard = b_ineq[:n_hard]

        # Gradient rows gain -s_k on the left-hand side.
        A_grad = np.zeros((2 * N, n_total))
        A_grad[:, :N] = A_ineq[n_hard:, :]
        A_grad[:N, N:] = -np.eye(N)
        A_grad[N:, N:] = -np.eye(N)
        b_grad = b_ineq[n_hard:]

        # Slack non-negativity: -s_k <= 0
        A_snn = np.zeros((n_slacks, n_total))
        A_snn[:, N:] = -np.eye(n_slacks)
        b_snn = np.zeros(n_slacks)

        H = H_full
        f = f_full
        A_ineq = np.vstack([A_hard, A_grad, A_snn])
        b_ineq = np.concatenate([b_hard, b_grad, b_snn])

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
        },
        n_inputs=N, n_slacks=n_slacks,
    )
