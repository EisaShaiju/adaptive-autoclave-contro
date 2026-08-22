"""
slack_weight_sensitivity.py
Answers the obvious reviewer question about the soft-constraint reformulation:
"were the penalty weights tuned until the numbers looked good?"

src/qp_builder.py defaults to slack_weight_lin = 1e3 (exact-penalty linear term)
and slack_weight_quad = 1e2 (quadratic regulariser). Neither value was
previously justified anywhere. This sweeps them over four orders of magnitude
and reports whether the conclusions move.

Two properties are checked, because they fail in opposite directions:

  1. EXACT PENALTY / no silent relaxation. The linear term must be large enough
     that slacks collapse to ~0 wherever the hard gradient constraint IS
     attainable. If they do not, the formulation is quietly relaxing a
     constraint that could have been met. Checked at a benign (early heat-up)
     state where the hard constraint is comfortably satisfiable.

  2. FEASIBILITY AT THE STIFF STEP. The weights must not be so large that the
     QP is numerically driven back toward the infeasible hard problem. Checked
     via the solver-independent slack LP at the stiff exotherm state.

Also reports u0 at both states so any control-level sensitivity is visible.

Exact-penalty theory: for an L1 penalty on the constraint violation, slacks are
driven to exactly zero once the penalty weight exceeds the magnitude of the
optimal Lagrange multiplier of the corresponding hard constraint. Above that
threshold the soft problem's solution COINCIDES with the hard problem's
wherever the hard problem is feasible -- which is why a plateau (rather than a
best value) is the correct thing to look for below.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/slack_weight_sensitivity.py
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import cvxpy as cp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import src.constants as const
from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
from src.dynamics import linearize
from src.qp_builder import build_canonical_qp

N = 10                      # recommended horizon
TARGET_TEMP = 120.0
Q_DIAG = np.zeros(10); Q_DIAG[0:3] = 100.0
R_VAL, S_VAL = 0.1, 1.0
BENIGN_K = 10               # early heat-up: hard gradient constraint attainable
ROLLOUT_STEPS = 105

LIN_WEIGHTS = [1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5]
QUAD_WEIGHTS = [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4]
DEFAULTS = (1.0e3, 1.0e2)   # (lin, quad) currently shipped


def rollout_states():
    """Deterministic CVXPY rollout -> (benign state, stiffest state)."""
    plant = AutoclavePlant(initial_temp=28.0)
    ctrl = MPCSolver(horizon=20, target_temp=TARGET_TEMP, trust_region=False)
    cur_Ta, states = 28.0, []
    x = plant.get_state()
    for k in range(ROLLOUT_STEPS):
        x0, u_prev = x.copy(), cur_Ta
        Ap, _ = linearize(np.mean(x0[0:3]), np.mean(x0[7:10]), trust_region=False)
        states.append({"k": k, "x0": x0, "u_prev": u_prev,
                       "rho": float(np.max(np.abs(np.linalg.eigvals(Ap))))})
        cur_Ta, _ = ctrl.compute_control_action(x0, u_prev)
        x = plant.step(Ta_input=cur_Ta)
    return states[BENIGN_K], max(states, key=lambda s: s["rho"])


def build(state, lin, quad):
    Ap, Bp = linearize(np.mean(state["x0"][0:3]), np.mean(state["x0"][7:10]),
                       trust_region=False)
    return build_canonical_qp(Ap, Bp, state["x0"], state["u_prev"], N,
                              Q_DIAG, R_VAL, S_VAL, TARGET_TEMP,
                              trust_region=False, soft_state_constraints=True,
                              slack_weight_lin=lin, slack_weight_quad=quad)


def solve_osqp(qp):
    U = cp.Variable(qp.H.shape[0])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(qp.H)) + qp.f @ U),
                      [qp.A_ineq @ U <= qp.b_ineq])
    prob.solve(solver=cp.OSQP, max_iter=200000)
    return prob.status, (None if U.value is None else np.asarray(U.value))


def slack_lp(qp):
    C, d = np.asarray(qp.A_ineq), -np.asarray(qp.b_ineq)
    m, n = C.shape
    z, s = cp.Variable(n), cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(s)), [C @ z + d <= s])
    prob.solve(solver=cp.OSQP, max_iter=200000)
    return prob.status, (None if s.value is None else float(np.max(s.value)))


def probe(state, lin, quad):
    qp = build(state, lin, quad)
    status, z = solve_osqp(qp)
    lp_status, max_s = slack_lp(qp)
    row = {"slack_weight_lin": lin, "slack_weight_quad": quad,
           "osqp_status": status, "lp_status": lp_status,
           "lp_max_slack": max_s,
           "feasible_set_nonempty": (max_s is not None and max_s <= 1e-6)}
    if z is not None:
        slacks = z[N:]                       # decision vector is [Ta_0..Ta_{N-1}, s_0..s_{N-1}]
        row.update({"u0": float(z[0]),
                    "max_slack_at_optimum": float(np.max(slacks)),
                    "sum_slack_at_optimum": float(np.sum(slacks))})
    else:
        row.update({"u0": None, "max_slack_at_optimum": None,
                    "sum_slack_at_optimum": None})
    return row


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    benign, stiff = rollout_states()
    print(f"benign state k={benign['k']} rho={benign['rho']:.4f} | "
          f"stiff state k={stiff['k']} rho={stiff['rho']:.4f}")
    print(f"shipped defaults: slack_weight_lin={DEFAULTS[0]:.0e}, "
          f"slack_weight_quad={DEFAULTS[1]:.0e}\n")

    out = {"git_commit": commit, "horizon": N,
           "shipped_defaults": {"slack_weight_lin": DEFAULTS[0],
                                "slack_weight_quad": DEFAULTS[1]},
           "benign_state": {"k": benign["k"], "rho_Ap": benign["rho"]},
           "stiff_state": {"k": stiff["k"], "rho_Ap": stiff["rho"]},
           "sweeps": {}}

    for label, state in (("benign", benign), ("stiff", stiff)):
        print(f"=== {label} state: sweeping slack_weight_lin "
              f"(quad fixed at {DEFAULTS[1]:.0e}) ===")
        print(f"{'lin':>10}{'u0':>10}{'max_slack@opt':>16}{'sum_slack@opt':>16}"
              f"{'LP max s*':>12}{'feasible':>10}")
        lin_rows = []
        for lin in LIN_WEIGHTS:
            r = probe(state, lin, DEFAULTS[1])
            lin_rows.append(r)
            print(f"{lin:>10.0e}{_f(r['u0'],'10.4f')}{_f(r['max_slack_at_optimum'],'16.3e')}"
                  f"{_f(r['sum_slack_at_optimum'],'16.3e')}{_f(r['lp_max_slack'],'12.3e')}"
                  f"{str(r['feasible_set_nonempty']):>10}")

        print(f"\n=== {label} state: sweeping slack_weight_quad "
              f"(lin fixed at {DEFAULTS[0]:.0e}) ===")
        print(f"{'quad':>10}{'u0':>10}{'max_slack@opt':>16}{'sum_slack@opt':>16}"
              f"{'LP max s*':>12}{'feasible':>10}")
        quad_rows = []
        for quad in QUAD_WEIGHTS:
            r = probe(state, DEFAULTS[0], quad)
            quad_rows.append(r)
            print(f"{quad:>10.0e}{_f(r['u0'],'10.4f')}{_f(r['max_slack_at_optimum'],'16.3e')}"
                  f"{_f(r['sum_slack_at_optimum'],'16.3e')}{_f(r['lp_max_slack'],'12.3e')}"
                  f"{str(r['feasible_set_nonempty']):>10}")
        print()

        u0s = [r["u0"] for r in lin_rows + quad_rows if r["u0"] is not None]
        out["sweeps"][label] = {
            "lin_sweep": lin_rows, "quad_sweep": quad_rows,
            "u0_spread_degC": (float(max(u0s) - min(u0s)) if u0s else None),
        }
        if u0s:
            print(f"  -> u0 spread across the whole {label} sweep: "
                  f"{max(u0s) - min(u0s):.6f} °C\n")

    dest = PROJECT_ROOT / "results" / "slack_weight_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


def _f(v, spec):
    return f"{'N/A':>{spec.split('.')[0]}}" if v is None else f"{v:{spec}}"


if __name__ == "__main__":
    main()
