"""
export_fpga_qp_dump.py
Per-step QP export for Ameer's Kria KV260 HLS/fixed-point snn_opt kernel.
Exports exactly what the software SNN solver consumes, at the recommended
configuration (N=10, soft, k0_scale=0.1, trust_region=False), LTI mode (the
FPGA kernel runs the same algorithm, so it cannot settle the LTV/convergence
question -- Ameer's own stated scope for this run).

Row layout (confirmed against src/qp_builder.py, not assumed -- soft-
constraint form, m = n_grad kept rows after relative-degree-5 removal,
z = [Ta_0..Ta_{N-1}, s_0..s_{m-1}]):
  rows [0, 2N)        actuator box      (exported as native per-variable bounds)
  rows [2N, 4N)       slew rate         (A_slew / b_slew -- NOT a per-variable
                                          bound, a genuine u_k/u_{k-1} coupling)
  rows [4N, 4N+2m)    output/gradient   (A_out / b_out -- the only rows that
                                          mix control and slack columns)
  rows [4N+2m, 4N+3m) slack >= 0        (exported as a second native-bound
                                          block, beyond Ameer's literal ask,
                                          since it's free and genuinely a bound)

"Exactly as the SNN receives it" = the CONDITIONED arrays (H_s, g_s, C_s,
d_s) from SNNMPCSolver._condition -- confirmed: a two-stage scaling
(column-scale by D = sqrt(diag(H)), then row-normalize), solved as
C_s x + d_s <= 0 (sign convention: d = -b_ineq). Raw canonical (H, f,
A_ineq, b_ineq) and the scaling diagonal D are included alongside so either
space can be reconstructed. Warm start is exported in the SAME scaled space
the solver actually consumes (U_warm_scaled = U_raw * D), shape N+m
(includes the slack block, zero-initialized on cold start).

Because relative degree is a structural, state-independent property
(docs/PHASE4_VALIDATION_REPORT.md section 14), m = N - 5 at every step here, so array shapes are
constant across the whole trajectory -- safe to stack into fixed-shape numpy
arrays rather than per-step ragged storage.

Usage:
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/export_fpga_qp_dump.py
"""
from pathlib import Path
import sys
import subprocess
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import final_controlled_comparison as fcc  # noqa: E402  (tools/ added to sys.path above)
from src.plant_simulator import AutoclavePlant  # noqa: E402
import src.constants as const  # noqa: E402

HORIZON = 10
TIME_STEPS = 160
OUT_DIR = PROJECT_ROOT / "results" / "fpga_export"


def row_blocks(N, m):
    """Row-index boundaries for the soft-constraint form, confirmed against
    src/qp_builder.py (see module docstring)."""
    return {
        "box": (0, 2 * N),
        "slew": (2 * N, 4 * N),
        "output": (4 * N, 4 * N + 2 * m),
        "slack_nonneg": (4 * N + 2 * m, 4 * N + 3 * m),
    }


def export_scenario(name, disturbance_step, disturbance_magnitude, time_steps, commit):
    fcc.CONFIG.update({
        "trust_region": False, "soft_state_constraints": True, "k0_scale": 0.1,
        "drop_uncontrollable_rows": True, "constraint_horizon": None,
        "linearization_mode": "lti", "ltv_nominal_source": "warm_start",
    })
    ctrl_cvx, ctrl_snn = fcc.make_controllers(HORIZON)
    plant_cvx = AutoclavePlant(initial_temp=fcc.INITIAL_TEMP)
    plant_snn = AutoclavePlant(initial_temp=fcc.INITIAL_TEMP)
    u_cvx = u_snn = fcc.INITIAL_TEMP

    N, m = None, None
    rec = {k: [] for k in (
        "H_s", "g_s", "C_s", "d_s", "D_scale", "H_raw", "f_raw",
        "A_box_lo", "A_box_hi", "A_slew", "b_slew", "A_out", "b_out",
        "slack_lo", "slack_hi",
        "warm_start_scaled", "u0_cvx", "u0_snn",
        "snn_converged", "snn_convergence_reason", "snn_iterations",
        "snn_kkt_residual", "snn_kkt_tolerance", "snn_constraint_residual",
    )}

    print(f"Exporting scenario '{name}' ({time_steps} steps)...")
    for k in range(time_steps):
        if disturbance_step is not None and k == disturbance_step:
            for p in (plant_cvx, plant_snn):
                p.T_comp -= disturbance_magnitude
                p.T_tool -= disturbance_magnitude

        x_cvx = plant_cvx.get_state()
        x_snn = plant_snn.get_state()

        # Raw canonical QP, captured independently of the instrumented mirror
        # (safe to call twice for LTI mode: build_qp depends only on (x,
        # u_prev) and static config here, not on mutable per-step state).
        qp_raw = ctrl_snn.build_qp(x_snn, u_snn)
        if N is None:
            N = ctrl_snn.N
            m = qp_raw.gradient_rows["n_kept"]
            blocks = row_blocks(N, m)
            print(f"  N={N}, m={m} (kept gradient rows), "
                  f"total rows={4*N+3*m}, total cols={N+m}")

        # Snapshot the warm start as it stands BEFORE this step's solve --
        # snn_step overwrites ctrl_snn.U_warm with the post-solve shifted
        # value, so this must be captured first. Mirrors snn_step's own
        # cold-start-vs-carried-warm-start logic exactly.
        n_total = N + m
        U_warm_before = (ctrl_snn._warm_hold(u_snn, n_total)
                          if (ctrl_snn.U_warm is None or ctrl_snn.U_warm.shape[0] != n_total)
                          else ctrl_snn.U_warm)

        r_cvx = fcc.cvxpy_step(ctrl_cvx, x_cvx, u_cvx)
        r_snn = fcc.snn_step(ctrl_snn, x_snn, u_snn)

        if r_snn.get("non_finite_conditioned_problem", True):
            print(f"  step {k}: non-finite conditioned problem, skipping "
                  f"(matches existing 0%-rate expectation at this config -- "
                  f"flag if this ever actually fires)")
            u_cvx, u_snn = r_cvx["applied_control"], r_snn["applied_control"]
            plant_cvx.step(Ta_input=u_cvx)
            plant_snn.step(Ta_input=u_snn)
            continue

        A_raw, b_raw = qp_raw.A_ineq, qp_raw.b_ineq
        H_s, g_s, C_s, d_s, D = r_snn["_H_s"], r_snn["_g_s"], r_snn["_C_s"], r_snn["_d_s"], r_snn["_D"]

        lo, hi = blocks["box"]
        box_rows_b = b_raw[lo:hi]
        # rows [0,N): Ta <= TA_MAX  (A row = +e_i);  rows [N,2N): -Ta <= -TA_MIN
        box_hi = box_rows_b[:N].copy()
        box_lo = -box_rows_b[N:2 * N].copy()

        lo, hi = blocks["slew"]
        A_slew, b_slew = A_raw[lo:hi], b_raw[lo:hi]

        lo, hi = blocks["output"]
        A_out, b_out = A_raw[lo:hi], b_raw[lo:hi]

        lo, hi = blocks["slack_nonneg"]
        # -s_k <= 0  =>  s_k >= 0 natively; exported as bounds, not a general row.
        slack_lo = np.zeros(m)
        slack_hi = np.full(m, np.inf)

        warm_start_scaled = U_warm_before * D

        for key, val in (
            ("H_s", H_s), ("g_s", g_s), ("C_s", C_s), ("d_s", d_s), ("D_scale", D),
            ("H_raw", qp_raw.H), ("f_raw", qp_raw.f),
            ("A_box_lo", box_lo), ("A_box_hi", box_hi),
            ("A_slew", A_slew), ("b_slew", b_slew),
            ("A_out", A_out), ("b_out", b_out),
            ("slack_lo", slack_lo), ("slack_hi", slack_hi),
            ("warm_start_scaled", warm_start_scaled),
            ("u0_cvx", r_cvx["applied_control"]),
            ("u0_snn", r_snn["applied_control"]),
            ("snn_converged", r_snn["verified_converged"]),
            ("snn_convergence_reason", r_snn["convergence_reason"]),
            ("snn_iterations", r_snn["iterations"]),
            ("snn_kkt_residual", r_snn.get("kkt_residual")),
            ("snn_kkt_tolerance", r_snn.get("kkt_tolerance")),
            ("snn_constraint_residual", r_snn["constraint_residual_physical"]),
        ):
            rec[key].append(val)

        u_cvx, u_snn = r_cvx["applied_control"], r_snn["applied_control"]
        plant_cvx.step(Ta_input=u_cvx)
        plant_snn.step(Ta_input=u_snn)

    # Stack into fixed-shape arrays (safe: m constant across the whole run,
    # relative degree is structural per docs/PHASE4_VALIDATION_REPORT.md section 14).
    out = {}
    for key in ("H_s", "g_s", "C_s", "d_s", "D_scale", "H_raw", "f_raw",
                "A_box_lo", "A_box_hi", "A_slew", "b_slew", "A_out", "b_out",
                "slack_lo", "slack_hi", "warm_start_scaled",
                "u0_cvx", "u0_snn", "snn_converged", "snn_iterations",
                "snn_constraint_residual"):
        out[key] = np.array(rec[key])
    out["snn_convergence_reason"] = np.array(rec["snn_convergence_reason"], dtype=object)
    out["snn_kkt_residual"] = np.array([v if v is not None else np.nan for v in rec["snn_kkt_residual"]])
    out["snn_kkt_tolerance"] = np.array([v if v is not None else np.nan for v in rec["snn_kkt_tolerance"]])
    out["n_control"] = N
    out["n_slack"] = m
    out["horizon"] = HORIZON
    out["commit_hash"] = commit
    out["scenario"] = name
    out["sign_convention"] = "solved as C_s @ z + d_s <= 0 (d = -b_ineq); box/slew/output bounds same convention"
    out["z_ordering"] = "z = [Ta_0..Ta_{N-1}, s_0..s_{m-1}]"
    out["TA_MIN"] = const.TA_MIN
    out["TA_MAX"] = const.TA_MAX
    out["TA_RATE_MAX"] = const.TA_RATE_MAX

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{commit[:8]}_{name}.npz"
    np.savez(out_path, **out)
    print(f"  Wrote {out_path} ({len(rec['u0_cvx'])} steps)")
    return out_path


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT).decode().strip()
    except Exception as exc:
        commit, dirty = f"unavailable: {exc}", None
    if dirty:
        print("WARNING: working tree is dirty -- exported commit_hash will not "
              "fully provenance this run:")
        print(dirty)

    paths = [
        export_scenario("nominal_heatup", None, 0.0, TIME_STEPS, commit),
        export_scenario("disturbance_step60", 60, 15.0, TIME_STEPS, commit),
    ]

    provenance = fcc.capture_provenance(horizon=HORIZON)
    readme_path = OUT_DIR / "README.txt"
    with open(readme_path, "w") as f:
        f.write(f"""FPGA QP export -- generated {datetime.now().isoformat(timespec='seconds')}
Commit: {commit}
Working tree dirty: {bool(dirty)}

Files: {', '.join(p.name for p in paths)}

Each .npz, per scenario, stacks one QP per closed-loop step (160 steps,
recommended config N=10, soft, k0_scale=0.1, trust_region=False, LTI mode --
see module docstring of tools/export_fpga_qp_dump.py for why LTI and not LTV).

Arrays (all stacked along axis 0 = step):
  H_s, g_s, C_s, d_s   -- the CONDITIONED QP exactly as the SNN solver reads it.
                           Solved as: 0.5 z^T H_s z + g_s^T z  s.t.  C_s z + d_s <= 0.
  D_scale              -- the column-scaling diagonal; H_raw = D*H_s*D, etc.
                           (H_s = D^-1 H_raw D^-1, g_s = D^-1 f_raw, C_s = D^-1 A_raw
                           row-normalized; see _condition() in src/snn_mpc_controller.py).
  H_raw, f_raw         -- the raw canonical QP (pre-conditioning), for cross-check.
  A_box_lo, A_box_hi   -- actuator box bounds per step, length n_control (native
                           per-variable bounds: A_box_lo <= Ta_k <= A_box_hi).
  A_slew, b_slew       -- slew-rate rows (NOT a per-variable bound -- couples
                           Ta_k and Ta_{{k-1}}): A_slew @ z <= b_slew.
  A_out, b_out         -- output/gradient constraint rows (the only rows mixing
                           control and slack columns): A_out @ z <= b_out.
  slack_lo, slack_hi   -- slack variables' native bound (lo=0, hi=+inf), length
                           n_slack. Free addition beyond the literal ask, since
                           it's a genuine per-variable bound like the actuator
                           box, not a general row.

  IMPORTANT -- two different unit spaces are exported side by side:
  A_box_lo/A_box_hi, A_slew/b_slew, A_out/b_out, slack_lo/slack_hi, H_raw,
  f_raw are all in RAW PHYSICAL units (same space as TA_MIN/TA_MAX below).
  H_s, g_s, C_s, d_s, warm_start_scaled are in the CONDITIONED space the
  solver actually iterates in. D_scale is the per-variable column-scaling
  vector relating them (scaled_z_i = raw_z_i * D_scale_i) -- use it if you
  need the box/slew/output bounds in the conditioned space instead of raw.
  u0_cvx, u0_snn       -- applied first-move solutions, CVXPY and software SNN.
  snn_converged, snn_convergence_reason, snn_iterations,
  snn_kkt_residual, snn_kkt_tolerance, snn_constraint_residual
                       -- software-SNN solve diagnostics per step, for comparing
                          the FPGA kernel's output against the software run at
                          matched accuracy.

Scalars: n_control (=N), n_slack (=m), horizon, commit_hash, scenario,
sign_convention, z_ordering, TA_MIN, TA_MAX, TA_RATE_MAX.

z ordering: z = [Ta_0..Ta_{{N-1}}, s_0..s_{{m-1}}]  (n_control={provenance['horizon_N']},
and n_slack is constant across the whole trajectory per the relative-degree-5
structural invariant -- see module docstring for why that's safe to assume).

snn_opt solver config / iteration budget (same field names as
tools/final_controlled_comparison.py's capture_provenance(), so this export
and that script's summary.json describe the same run without a translation
layer):
{provenance['snn_solver_config']}
""")
    print(f"\nWrote {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
