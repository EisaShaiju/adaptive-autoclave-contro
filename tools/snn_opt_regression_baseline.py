"""
snn_opt_regression_baseline.py
Deterministic numerical fingerprint of the installed snn_opt, for use as a
BEFORE/AFTER regression gate around a dependency upgrade.

Captures, on fixed saved QPs (no RNG anywhere in this repo):
  - package version, module path, backend actually loaded
  - per-(N, k0_scale) solve results on the stiff-state soft QP
  - the hard-form n_projections=0 instance
Writes results/snn_opt_regression_<tag>.json.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe \
        tools/snn_opt_regression_baseline.py <tag>
"""
from pathlib import Path
import hashlib
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
from snn_opt import OptimizationProblem, SNNSolver
from src.snn_mpc_controller import SNNMPCSolver
from convergence_blocker_probe import find_stiff_state, TRUST_REGION, SOFT
from snn_solve_instrumentation import find_test_states


def backend_actually_loaded():
    """Confirm the compiled kernel loaded rather than a silent python fallback."""
    try:
        from snn_opt import _kernel  # noqa: F401
        return "c (compiled _kernel imported)"
    except Exception as exc:
        return f"python fallback ({type(exc).__name__}: {exc})"


def solve_fingerprint(ctrl, state, tag):
    qp = ctrl.build_qp(state["x0"], state["u_prev"])
    H_s, g_s, C_s, d_s, D = ctrl._condition(qp.H, qp.f, qp.A_ineq, -qp.b_ineq)
    C_s, d_s = np.asarray(C_s), np.asarray(d_s)
    U_cold = np.asarray(ctrl._warm_hold(state["u_prev"], np.shape(H_s)[0]), dtype=float)

    try:
        solver = SNNSolver(OptimizationProblem(A=H_s, b=g_s, C=C_s, d=d_s),
                           ctrl.solver_config)
    except ValueError as exc:
        # snn_opt >= 0.6.0 refuses certifiably infeasible problems (zero-normal
        # row with d > 0). This is the CORRECT outcome on the hard form and
        # independently reproduces this repo's own proof -- record it, do not
        # treat it as a failure of the regression run.
        return {
            "case": tag,
            "qp_fingerprint": qp.fingerprint(),
            "rejected_as_infeasible": True,
            "rejection_message": str(exc),
        }
    res = solver.solve(U_cold * D, verbose=False)
    x = np.asarray(res.final_x)
    return {
        "case": tag,
        "qp_fingerprint": qp.fingerprint(),
        "u0_physical": float((x / D)[0]),
        "converged": bool(res.converged),
        "convergence_reason": str(res.convergence_reason),
        "iterations_used": int(res.iterations_used),
        "n_projections": int(res.n_projections),
        "final_max_violation": float(np.max(C_s @ x + d_s)),
        "final_x_norm": float(np.linalg.norm(x)),
        # hash of the solution vector: catches any numerical drift the scalars miss
        "final_x_sha256_8dp": hashlib.sha256(
            np.round(x, 8).tobytes()).hexdigest()[:16],
    }


def _fmt(r):
    if r.get("rejected_as_infeasible"):
        return f"REJECTED AS INFEASIBLE -- {r['rejection_message'][:70]}"
    return (f"u0={r['u0_physical']:.6f} conv={r['converged']} "
            f"nproj={r['n_projections']} viol={r['final_max_violation']:.3e}")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "untagged"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    print(f"snn_opt version : {getattr(snn_opt, '__version__', '?')}")
    print(f"module path     : {Path(snn_opt.__file__).parent}")
    print(f"backend loaded  : {backend_actually_loaded()}")

    rows = []

    # --- soft-form stiff state, the configurations that matter ---
    stiff = find_stiff_state()
    for N in (20, 10, 5):
        for k0 in (0.5, 0.1):
            ctrl = SNNMPCSolver(horizon=N, target_temp=120.0,
                                trust_region=TRUST_REGION,
                                soft_state_constraints=SOFT, k0_scale=k0)
            r = solve_fingerprint(ctrl, stiff, f"soft_k87_N{N}_k0{k0}")
            rows.append(r)
            print(f"  {r['case']:>22}: {_fmt(r)}")

    # --- hard-form state: the n_projections=0 instance ---
    _easy, hard_stiff = find_test_states()
    ctrl_hard = SNNMPCSolver(horizon=20, target_temp=120.0,
                             soft_state_constraints=False)
    r = solve_fingerprint(ctrl_hard, hard_stiff, "hard_k84_N20_default")
    rows.append(r)
    print(f"  {r['case']:>22}: {_fmt(r)}")

    out = {
        "tag": tag,
        "git_commit": commit,
        "snn_opt_version": str(getattr(snn_opt, "__version__", "?")),
        "snn_opt_path": str(Path(snn_opt.__file__).parent),
        "backend_loaded": backend_actually_loaded(),
        "stiff_state_soft": {"k": stiff["k"], "rho_Ap": stiff["rho"]},
        "stiff_state_hard": {"k": hard_stiff["k"], "rho_Ap": hard_stiff["rho"]},
        "rows": rows,
    }
    dest = PROJECT_ROOT / "results" / f"snn_opt_regression_{tag}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
