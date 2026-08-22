# Artifact index — every reported figure to its file

Purpose: no number in `docs/PHASE4_VALIDATION_REPORT.md` or the repository's
contributor notes should exist only in prose or console scrollback. Each row
names the claim, the file that backs it, and the script that regenerates that
file.

Regenerate everything with `PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
.venv/Scripts/python.exe tools/<script>`.

## Commit-hash translation (history rewrite, 2026-08-22)

Repository history was rewritten on 2026-08-22 to remove local working-notes
references that were never meant to be published. **No tracked file content
changed at either branch tip** — the tip trees are byte-identical before and
after (`8452cf4e` on `snn-qp-baseline-parity`, `931dc73f` on `main`) — but every
commit hash changed. Artifact directory names and the `git_commit` field inside
each `summary.json` therefore record **pre-rewrite** hashes. Translate with:

| recorded (pre-rewrite) | current | commit |
|---|---|---|
| `d80ccb05` | `139c73bb` | fix(docs): typeset equations properly in the PDF build |
| `f1b1d8da` | `98a9cb9e` | Phase-4 closure |
| `1dc59a13` | `1d24b1dd` | docs: add engineering report for the SNN-QP parity work |
| `4c06918f` | `a174496c` | Phase-4 validation |
| `8e31825f` | `aa679aa0` | docs: add PDF build of the technical report |
| `c82cf72f` | `0226d36e` | chore: restore `.gitignore` |

**`70b83cd9_*` directories name a commit that was already unreachable** *before*
the rewrite — it is a superseded pre-amend version of `1dc59a13`, present as a
dangling object in the old repository but not on any branch. Those artifacts
predate the current results and are retained only for the horizon sweep cited in
§7; read them as "generated around the engineering-report commit", not as
pointing at a resolvable hash.

| Reported quantity | Where cited | Artifact | Generator |
|---|---|---|---|
| Canonical QP bit-identity (both controllers) | §3, model-parity invariant | `qp_parity_diagnostics.json` | `qp_parity_probe.py` |
| Off-by-one window deltas (`s=i` vs `s=i+1`) | §3 | `qp_parity_diagnostics.json` → `off_by_one_deltas` | `qp_parity_probe.py` |
| `max\|ΔAp\|` ≈ 1e−4 benign / ≈ 1851 gelation | tech report, model-parity invariant | `ap_parity_grid.json` | `ap_parity_grid_probe.py` |
| `max\|ΔAp\|` along a live rollout (≈ 76) | — (context only) | `qp_parity_diagnostics.json` → `ap_parity_diff` | `qp_parity_probe.py` |
| 12/12 hard configurations infeasible under OSQP | §4 | `conditioning_sweep.json` | `conditioning_sweep.py` |
| Exact zero-row infeasibility proof (k=2,3,4) | §4.1 | `gradient_row_infeasibility.json` | `gradient_row_infeasibility_probe.py` |
| Soft-form feasibility (`max s*` ≤ 3.9e−7, N=5/10/20) | §4.1 | `feasibility_certificate.json` | `feasibility_certificate_probe.py` |
| `n_projections = 0` (hard form, k=84) | §4.1, §11.4 | `gradient_row_infeasibility.json`, `stiff_divergence_trace_summary.json` | `gradient_row_infeasibility_probe.py` |
| Projection saturation 200.0/200 (soft, k=87) | §4.1 | `feasibility_certificate.json` → `snn.projections_per_iteration` | `feasibility_certificate_probe.py` |
| Projected-gradient norm 1.66e10, budget-invariant | §5.1 | `convergence_blocker_probe.json` | `convergence_blocker_probe.py` |
| N=5 agreement: `u₀` 5e−4, obj −5.4e−8, rel-pg 0.0097 | §5.1 | `optimum_agreement_probe.json` | `optimum_agreement_probe.py` |
| **N=20 rel-pg 0.449 / 0.595 / 0.670 by `k0_scale`** | §5.2 | `optimum_agreement_probe.json` | `optimum_agreement_probe.py` |
| **N=20 `u₀` vs OSQP = 1.5e−6 °C** | §5.2 | `optimum_agreement_probe.json` | `optimum_agreement_probe.py` |
| N=10 `u₀` gap 3.7–4.1 °C at stiff step | §5.2 | `optimum_agreement_probe.json` | `optimum_agreement_probe.py` |
| Eigen-whitening: cond(H)=1.0, still `converged=False` | §4 | `qp_conditioning_report.json` → `minimal_experiment_eigen_whitening` | `qp_conditioning_probe.py` |
| Jacobi conditioning change report | §4 | `qp_conditioning_change_report.json` | `qp_conditioning_change_report.py` |
| RMS control/trajectory differences, clipping rates | §6, §8 | `final_comparison/*/per_step_metrics.csv`, `summary.json` | `final_controlled_comparison.py` |
| Horizon sweep N=5/10/20 (α, max Tc1, verdict) | §7 | `final_comparison/70b83cd9_N{5,10,20}_soft_identical_*/` | `final_controlled_comparison.py` |
| N=5 trap: 11.41 °C nominal / 1.85 °C disturbance, α ≤ 1.0e−5 | §7 | `final_comparison/70b83cd9_N5_soft_identical_*/per_step_metrics.csv` | `final_controlled_comparison.py` |
| Shared-harness closed-loop run | §2, §6 | `shared_closed_loop_run.json` | `shared_closed_loop_harness.py` |
| Slew saturation 41.7 % (first 60 steps) / 33.1 % (all 160) / 77.4 % (stiff) | §6, README | `final_comparison/*/summary.json` → `heatup_slew_saturation_caveat` and `scenarios.*.pct_steps_both_slew_saturated` | `final_controlled_comparison.py` |
| Per-step solve diagnostics (easy + stiff) | §5 | `snn_solve_diagnostics.json` | `snn_solve_instrumentation.py` |
| Slack-weight sensitivity (`u₀` spread 0.000/0.048 °C) | §3.1 | `slack_weight_sensitivity.json` | `slack_weight_sensitivity.py` |
| **Rev. 4 headline metrics (`snn_opt` 0.6.0)** | §5.3, §6, §8, README | `final_comparison/d80ccb05_N10_soft_snnopt060_*/` | `final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1` |
| KKT vs legacy certificate; N=20 ratio 113 | §5.3 | `kkt_certificate_probe.json` | `kkt_certificate_probe.py` |
| Dependency before/after fingerprint | §5.3 | `snn_opt_regression_{pre040,post060_adapted}.json` | `snn_opt_regression_baseline.py <tag>` |
| Row-82 infeasibility rejection by 0.6.0 | §5.3, §4.1 | `snn_opt_regression_post060_adapted.json` → `rejected_as_infeasible` | `snn_opt_regression_baseline.py` |
| **Compute time (current, 0.6.0): CVXPY 5.47 ms vs SNN 48.66 ms median (8.9×)** | §8.5 | `final_comparison/d80ccb05_N10_soft_snnopt060_*/summary.json` → `timing_ms` | `final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1` |
| Cure gate (final `α`, max `Tc1`, cured yes/no) per scenario | README, §5.3 | `final_comparison/*/summary.json` → `scenarios.*.cure_gate` | `final_controlled_comparison.py` |
| Resolved `optimality_test` actually used by the SNN run | §5.3 | `final_comparison/*/summary.json` → `provenance.snn_solver_config.convergence.optimality_test` | `final_controlled_comparison.py` |
| Compute time (superseded, 0.4.0): 5.76 ms vs 113.75 ms (19.8×) | §8.5 (history) | `final_comparison/d80ccb05_N10_soft_timed_*/summary.json` → `timing_ms` | same, under `snn_opt` 0.4.0 |
| `snn_opt` 0.4.0 vs upstream 0.6.0 comparison | §11.1 | `snn_opt_upstream_diff.json` | manual (WebFetch of upstream README/CHANGELOG) |

## Known provenance caveats

- **`snn_opt_upstream_diff.json`** is the one entry not generated by a script in
  this repo: it records a read-only WebFetch of the upstream README and
  CHANGELOG on 2026-08-18, taken **before** the upgrade. Upstream may move;
  re-verify before relying on it. Its "installed version" field is a snapshot of
  the pre-upgrade state (0.4.0); the installed version is now **0.6.0**.
- **`ap_parity_grid.json` vs `qp_parity_diagnostics.json`** report `max|ΔAp|`
  over *different domains* (operating-point grid ≈ 1851 vs live trajectory
  ≈ 76). Both are correct; citing one where the other is meant is the error to
  avoid. The docs' ≈1851 figure refers to the grid.
- **Relative projected-gradient norms depend on `k0_scale`** and are meaningless
  without it. `optimum_agreement_probe.json` now sweeps 0.5 (constructor
  default), 0.1 (recommended config), and 0.05.
- **`stiff_divergence_trace_summary.json`** was traced with
  `backend='python'`, `record_trajectory=True` at `k=84` **hard** form; the
  `k=87` probes use the default backend and the **soft** form. Do not compare
  across those without saying which.
- **Timing is configuration-dependent and was previously misreported.** The
  current figure is **8.9×** (CVXPY 5.47 ms vs SNN 48.66 ms median) at the
  *recommended* config (N=10, soft, `k0_scale=0.1`, `snn_opt` 0.6.0), both
  controllers in one process. Under 0.4.0 the same config gave 19.8×. The older
  `shared_closed_loop_run.json` (N=20, **hard**) shows CVXPY 116 ms vs SNN
  186 ms — a ~1.6× ratio — because CVXPY is itself slow at N=20 on a
  configuration this repo no longer recommends. Quoting that ratio as
  "comparable" was the error corrected in §8.5. Always state horizon,
  constraint form, **and `snn_opt` version** with any timing number.
- **Timing is also the only non-reproducible number here.** Two identical
  regenerations of the 0.6.0 run agreed to the last recorded digit on every
  metric *except* `timing_ms`: the SNN nominal median moved 47.0 → 48.66 ms
  (ratio 8.6× → 8.9×) and the disturbance scenario moved further under
  background load, with its ratio still holding at 8.5–8.7×. Cite the ratio;
  treat absolute milliseconds as indicative of the machine, not the method.
- **Stale provenance in the 0.4.0 `*_N10_soft_timed_*` directory.** Its
  `provenance.horizon_N` reads 20 and its `snn_solver_config` shows constructor
  defaults (`k0_scale` 0.5), because an earlier version of
  `final_controlled_comparison.py` captured module defaults instead of the run
  configuration; its generated `report.md` inherits the wrong "N=20" sentence.
  The run really used N=10 / soft / `k0_scale=0.1` — see its
  `provenance.shared_configuration`, which was always correct. The bug is fixed
  and the 0.6.0 directory is clean; the 0.4.0 directory is left as-is rather
  than regenerated, because regenerating it would require downgrading the
  dependency and would silently change the before-side of the comparison.
- **Metrics are not comparable across `snn_opt` versions without care.**
  Anything measured before 2026-08-18 is `snn_opt` 0.4.0. Objective gaps
  especially: 0.4.0 could only grade 45–59 % of steps, so its means are over an
  easier subset. Restrict to a common step set before comparing (§5.3 shows the
  like-for-like result is identical). Directories are tagged: `*_N10_soft_timed_*`
  is 0.4.0, `*_N10_soft_snnopt060_*` is 0.6.0.
- **`docs/SNN_MPC_TECHNICAL_REPORT.md` is current** as of Revision 2 — it now
  reports the 0.6.0 results throughout, and its PDF build is regenerated from
  it. The only report still carrying a superseded banner is
  `docs/SNN_QP_SOLVER_PARITY_REPORT.md`, kept as a historical record.
