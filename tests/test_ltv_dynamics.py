"""
test_ltv_dynamics.py
Correctness and regression tests for LTV (time-varying) prediction support,
added in the ltv-time-varying-prediction branch -- see README_LTV.md for why
this branch exists. Plain-assert script, runnable directly:

    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_ltv_dynamics.py

Covers what tests/test_qp_parity.py does not:
  1. The LTV Phi/Gamma recursion in src/qp_builder.py is mathematically
     correct for a genuinely time-varying (non-commuting) Ap sequence --
     checked against an independent, hand-rolled forward simulation of the
     linear time-varying system, using only build_canonical_qp's PUBLIC
     output arrays (no reliance on its internal Phi/Gamma).
  2. LTV with a CONSTANT Ap/Bp sequence numerically matches (tight tolerance,
     not necessarily bit-identical -- see src/qp_builder.py's module
     docstring on why the LTI branch is a separate, untouched code path)
     the existing LTI single-array call.
  3. The plant's relative degree (5) is unchanged under LTV at both a benign
     and a stiff state -- the argument in
     docs/PHASE4_VALIDATION_REPORT.md section 14 is structural (fixed
     diffusion-stencil connectivity), not a property of which operating
     points Ap is evaluated at, so this must still hold.
  4. Both controllers' build_qp() produce identical QPs under linearization_
     mode='ltv' given the same current_state/u_prev and the same nominal
     sequence -- the parity claim must survive the LTV extension.
"""
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mpc_cvxpy_controller import MPCSolver
from src.snn_mpc_controller import SNNMPCSolver
from src.dynamics import linearize, linearize_trajectory, shift_nominal_sequence
from src.qp_builder import build_canonical_qp, NX
from src.plant_simulator import AutoclavePlant
import src.constants as const

N = 3
TARGET_TEMP = 120.0

failures = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def qp_kwargs(ctrl):
    return dict(Q_diag=ctrl.Q_diag, R_val=ctrl.R_val, S_val=ctrl.S_val,
                target_temp=ctrl.target_temp, trust_region=False)


def predicted_gradient_from_qp(qp, u_test, N):
    """Recover (Tc1-Tc3) predicted at each horizon row from ONLY the public
    QP arrays -- hard form, drop_uncontrollable_rows=False, so all N gradient
    rows are present at A_ineq[4N:5N] (the +GRADIENT_MAX-facing half)."""
    b_grad_hi = qp.b_ineq[4 * N:5 * N]
    GGamma = qp.A_ineq[4 * N:5 * N, :]
    GPhi_x = const.GRADIENT_MAX - b_grad_hi
    return GPhi_x + GGamma @ u_test


def main():
    # ---- Three genuinely different, non-commuting Jacobians -------------
    print("--- 1. LTV Phi/Gamma recursion vs. independent hand rollout ---")
    x0 = np.zeros(NX)
    x0[0:3] = 60.0
    x0[3:7] = 50.0
    x0[7:10] = 0.3
    Ap0, Bp0 = linearize(60.0, 0.3, trust_region=False)
    Ap1, Bp1 = linearize(90.0, 0.6, trust_region=False)
    Ap2, Bp2 = linearize(120.0, 0.9, trust_region=False)
    Ap_seq, Bp_seq = [Ap0, Ap1, Ap2], [Bp0, Bp1, Bp2]

    u_prev = 60.0
    u_test = np.array([70.0, 40.0, 100.0])

    common = dict(Q_diag=np.full(NX, 1.0), R_val=0.1, S_val=1.0,
                  target_temp=TARGET_TEMP, trust_region=False,
                  soft_state_constraints=False, drop_uncontrollable_rows=False)
    qp_ltv = build_canonical_qp(Ap_seq, Bp_seq, x0, u_prev, N, **common)

    pred_from_qp = predicted_gradient_from_qp(qp_ltv, u_test, N)

    # Independent hand rollout of the SAME time-varying linear system, using
    # only linearize()'s outputs directly -- no shared code with qp_builder.
    x = x0.copy()
    pred_manual = []
    for h in range(N):
        pred_manual.append(x[0] - x[2])
        x = Ap_seq[h] @ x + Bp_seq[h] * u_test[h]
    pred_manual = np.array(pred_manual)

    check("LTV predicted gradient matches independent hand rollout",
          np.allclose(pred_from_qp, pred_manual, atol=1e-9),
          f"qp={pred_from_qp}, manual={pred_manual}")

    # A wrong (append-order) recursion would NOT match on this non-commuting
    # triple; confirm the matrices actually don't commute, else the check
    # above would pass vacuously.
    noncommuting = not np.allclose(Ap0 @ Ap1, Ap1 @ Ap0, atol=1e-8)
    check("sanity: the three Jacobians do not commute (test is not vacuous)",
          noncommuting)

    # ---- 2. LTV with a constant sequence matches the LTI path -----------
    print("\n--- 2. LTV(constant sequence) matches LTI within tolerance ---")
    qp_lti = build_canonical_qp(Ap0, Bp0, x0, u_prev, N, **common)
    qp_ltv_const = build_canonical_qp([Ap0] * N, [Bp0] * N, x0, u_prev, N, **common)

    check("H matches", np.allclose(qp_lti.H, qp_ltv_const.H, atol=1e-9))
    check("f matches", np.allclose(qp_lti.f, qp_ltv_const.f, atol=1e-9))
    check("A_ineq matches", np.allclose(qp_lti.A_ineq, qp_ltv_const.A_ineq, atol=1e-9))
    check("b_ineq matches", np.allclose(qp_lti.b_ineq, qp_ltv_const.b_ineq, atol=1e-9))
    check("gradient_rows relative_degree matches",
          qp_lti.gradient_rows["relative_degree"] == qp_ltv_const.gradient_rows["relative_degree"])

    # ---- 3. Relative degree unchanged under LTV, benign and stiff states --
    print("\n--- 3. Relative degree stays 5 under LTV ---")
    plant = AutoclavePlant()
    ctrl = MPCSolver(horizon=10, target_temp=TARGET_TEMP, soft_state_constraints=True)
    u_p = const.TA_MIN
    states = []
    for _ in range(100):
        s = plant.get_state()
        states.append(s.copy())
        u_p, _ = ctrl.compute_control_action(s, u_p)
        plant.step(u_p)

    benign_state, benign_uprev = states[20], const.TA_MIN
    stiff_state, stiff_uprev = states[88], const.TA_MIN  # u_prev value irrelevant to relative degree

    for label, state in (("benign (step 20)", benign_state), ("stiff (step 88)", stiff_state)):
        nominal = shift_nominal_sequence(None, 60.0, 10)
        Ap_s, Bp_s = linearize_trajectory(state, nominal, trust_region=False)
        qp = build_canonical_qp(Ap_s, Bp_s, state, 60.0, 10,
                                 Q_diag=np.full(NX, 1.0), R_val=0.1, S_val=1.0,
                                 target_temp=TARGET_TEMP, trust_region=False,
                                 soft_state_constraints=True, drop_uncontrollable_rows=True)
        check(f"relative degree is 5 at {label}",
              qp.gradient_rows["relative_degree"] == 5,
              f"got {qp.gradient_rows['relative_degree']}")
        check(f"dropped rows are exactly {{0..4}} at {label}",
              qp.gradient_rows["dropped_uncontrollable"] == [0, 1, 2, 3, 4])

    # ---- 4. Both controllers agree under linearization_mode='ltv' -------
    print("\n--- 4. Controller parity under LTV mode ---")
    x_test = states[50]
    u_prev_test = 55.0
    nominal_shared = shift_nominal_sequence(None, u_prev_test, 10)

    ctrl_cvx = MPCSolver(horizon=10, target_temp=TARGET_TEMP,
                          soft_state_constraints=True, linearization_mode='ltv')
    ctrl_snn = SNNMPCSolver(horizon=10, target_temp=TARGET_TEMP,
                             soft_state_constraints=True, linearization_mode='ltv', k0_scale=0.1)
    # Force both onto the IDENTICAL nominal trajectory (in isolation there is
    # no shared solve history to derive it from) -- this is exactly the
    # invariant linearization_mode='ltv' depends on in closed-loop use, where
    # each controller derives it from its own previous solve instead.
    ctrl_cvx._u_nominal = nominal_shared.copy()
    ctrl_snn._u_nominal = nominal_shared.copy()

    qp_cvx = ctrl_cvx.build_qp(x_test, u_prev_test)
    qp_snn = ctrl_snn.build_qp(x_test, u_prev_test)

    check("LTV mode: H matches across controllers", np.allclose(qp_cvx.H, qp_snn.H, atol=1e-9))
    check("LTV mode: f matches across controllers", np.allclose(qp_cvx.f, qp_snn.f, atol=1e-9))
    check("LTV mode: A_ineq matches across controllers", np.allclose(qp_cvx.A_ineq, qp_snn.A_ineq, atol=1e-9))
    check("LTV mode: b_ineq matches across controllers", np.allclose(qp_cvx.b_ineq, qp_snn.b_ineq, atol=1e-9))
    check("LTV mode: gradient_rows agree",
          qp_cvx.gradient_rows["kept"] == qp_snn.gradient_rows["kept"])

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: ALL CHECKS PASSED")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
