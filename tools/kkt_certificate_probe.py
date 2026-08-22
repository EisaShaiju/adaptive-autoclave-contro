"""
kkt_certificate_probe.py
The decisive snn_opt 0.6.0 experiment: does the NEW scale-invariant KKT
certificate fire where the OLD absolute projected-gradient test could not?

TRAP THIS PROBE EXISTS TO AVOID: snn_opt 0.6.0 keeps `proj_grad_tol` as a
deprecated constructor-only alias, and passing it silently forces
`optimality_test='legacy_projected_gradient'`. src/snn_mpc_controller.py DOES
pass it, so simply upgrading the package leaves the controller on the OLD test
and the new certificate is never exercised. Verified: the shipped controller
resolves to optimality_test='legacy_projected_gradient' under 0.6.0.

This probe therefore builds THREE configurations on identical QP arrays:
  1. legacy  -- optimality_test='legacy_projected_gradient', tol 5e-2 (shipped)
  2. kkt     -- optimality_test='kkt' at upstream default tolerances
  3. kkt_rel -- optimality_test='kkt' with a loosened kkt_rel_tol, to see how
                far the certificate is from firing rather than only that it fails

Prediction on record BEFORE running (docs/PHASE4_VALIDATION_REPORT.md 5.2):
since the relative
projected-gradient norm at N=20 is 0.449-0.670, a scale-invariant test is
expected to STILL NOT fire at the working horizon. A firing certificate would
be a surprise requiring explanation, not a success to report uncritically.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/kkt_certificate_probe.py
"""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import snn_opt
from snn_opt import OptimizationProblem, SNNSolver, ConvergenceConfig
from src.snn_mpc_controller import SNNMPCSolver
from convergence_blocker_probe import find_stiff_state, TRUST_REGION, SOFT

HORIZONS = (20, 10, 5)
K0 = 0.1          # the recommended configuration's step size


def variants(base_conv):
    """Three convergence configs on otherwise identical solver settings."""
    return {
        "legacy": replace(base_conv, optimality_test="legacy_projected_gradient",
                          legacy_proj_grad_tol=5e-2),
        "kkt_default": replace(base_conv, optimality_test="kkt",
                               legacy_proj_grad_tol=None,
                               kkt_abs_tol=1e-9, kkt_rel_tol=1e-4),
        "kkt_loose": replace(base_conv, optimality_test="kkt",
                             legacy_proj_grad_tol=None,
                             kkt_abs_tol=1e-6, kkt_rel_tol=5e-2),
    }


def getf(res, name):
    v = getattr(res, name, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def run(state, N):
    ctrl = SNNMPCSolver(horizon=N, target_temp=120.0, trust_region=TRUST_REGION,
                        soft_state_constraints=SOFT, k0_scale=K0)
    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H_s, g_s, C_s, d_s, D = ctrl._condition(qp.H, qp.f, qp.A_ineq, -qp.b_ineq)
    C_s, d_s = np.asarray(C_s), np.asarray(d_s)
    U_cold = np.asarray(ctrl._warm_hold(state["u_prev"], np.shape(H_s)[0]),
                        dtype=float)

    out = []
    for name, conv in variants(ctrl.solver_config.convergence).items():
        cfg = replace(ctrl.solver_config, convergence=conv)
        res = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s),
                        cfg).solve(U_cold * D, verbose=False)
        x = np.asarray(res.final_x)
        row = {
            "N": N, "variant": name,
            "optimality_test": getf(res, "optimality_test"),
            "converged": bool(res.converged),
            "convergence_reason": str(res.convergence_reason),
            "iterations_used": int(res.iterations_used),
            "n_projections": int(res.n_projections),
            "u0_physical": float((x / D)[0]),
            "max_violation_recomputed": float(np.max(C_s @ x + d_s)),
            "final_x_norm": float(np.linalg.norm(x)),
            # 0.6.0 result fields -- absent on older versions, hence getf()
            "kkt_residual": getf(res, "kkt_residual"),
            "kkt_scale": getf(res, "kkt_scale"),
            "kkt_tolerance": getf(res, "kkt_tolerance"),
            "kkt_fit_status": getf(res, "kkt_fit_status"),
            "kkt_stationarity_residual": getf(res, "kkt_stationarity_residual"),
            "kkt_complementarity_residual": getf(res, "kkt_complementarity_residual"),
            "stationarity_residual": getf(res, "stationarity_residual"),
            "joint_feasible": getf(res, "joint_feasible"),
            "projection_budget_exhausted": getf(res, "projection_budget_exhausted"),
        }
        # how far from firing? ratio of residual to its own tolerance
        if isinstance(row["kkt_residual"], float) and isinstance(row["kkt_tolerance"], float):
            row["kkt_residual_over_tolerance"] = (
                row["kkt_residual"] / row["kkt_tolerance"]
                if row["kkt_tolerance"] else None)
        out.append(row)
    return out


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    state = find_stiff_state()
    print(f"snn_opt {snn_opt.__version__} | stiff state k={state['k']} "
          f"rho={state['rho']:.4f} | soft form, k0_scale={K0}\n")

    rows = []
    hdr = (f"{'N':>3} {'variant':>12} {'conv':>6} {'reason':>18} {'iters':>6} "
           f"{'nproj':>7} {'u0':>11} {'viol':>11} {'kkt_res':>11} {'res/tol':>10}")
    print(hdr); print("-" * len(hdr))
    for N in HORIZONS:
        for r in run(state, N):
            rows.append(r)
            kr = r["kkt_residual"]
            rt = r.get("kkt_residual_over_tolerance")
            print(f"{r['N']:>3} {r['variant']:>12} {str(r['converged']):>6} "
                  f"{r['convergence_reason'][:18]:>18} {r['iterations_used']:>6} "
                  f"{r['n_projections']:>7} {r['u0_physical']:>11.5f} "
                  f"{r['max_violation_recomputed']:>11.3e} "
                  f"{('n/a' if kr is None else f'{kr:.3e}'):>11} "
                  f"{('n/a' if rt is None else f'{rt:.2e}'):>10}")
        print()

    out = {
        "git_commit": commit,
        "snn_opt_version": str(snn_opt.__version__),
        "k0_scale": K0, "trust_region": TRUST_REGION,
        "soft_state_constraints": SOFT,
        "stiff_state": {"k": state["k"], "rho_Ap": state["rho"]},
        "prediction_on_record": (
            "Relative projected-gradient norm at N=20 is 0.449-0.670, so a "
            "scale-invariant KKT test was predicted NOT to fire at the working "
            "horizon before this probe was run."),
        "trap_note": (
            "snn_opt 0.6.0 silently selects optimality_test="
            "'legacy_projected_gradient' when the deprecated proj_grad_tol "
            "alias is passed. src/snn_mpc_controller.py passes it, so upgrading "
            "the package alone does NOT enable the new certificate."),
        "rows": rows,
    }
    dest = PROJECT_ROOT / "results" / "kkt_certificate_probe.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
