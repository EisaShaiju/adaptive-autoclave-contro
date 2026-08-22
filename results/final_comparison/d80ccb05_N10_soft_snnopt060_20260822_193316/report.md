# Final Controlled Comparison Report

Generated 2026-08-22T19:32:16 from git commit `d80ccb058a93e9eaf06c87c6ef9a36ce6d5bde29` (DIRTY -- see provenance in summary.json).

## Method

One shared harness (`tools/final_controlled_comparison.py`) drives two independent `AutoclavePlant` instances, one per controller, from the identical initial state (28.0 degC), the same target temperature (120.0 degC), the same horizon (N=10), the same sampling time (TE=60.0s), the same physical constraints, and the same canonical per-step QP construction (`src/qp_builder.py`) -- the only permitted divergence is the `trust_region` flag baked into each controller's own `Ap,Bp` (documented in docs/PHASE4_VALIDATION_REPORT.md). Disturbance convention: **disturbance-before-compute**, applied identically to both plants at the same step index, so both controllers react to identical information.

Both controllers' per-step solves are reimplemented here using the same public building blocks (`build_qp`, `_condition` for SNN, identical OSQP/SNNSolver calls) each production method uses -- not a different code path, just an instrumented mirror exposing the full decision vector and objective value. Every SNN metric is computed by mapping the solver's raw output back to physical units (`U_sol = result.final_x / D`) and evaluating it against the ORIGINAL, unconditioned canonical QP -- never trusted from the solver's internal (scaled-space) self-report alone. No new output clipping was added; `applied_control` uses the exact safety-filter clip already in `SNNMPCSolver.compute_control_action`.

**Scenario 3 (stiff exotherm) is a window, not a separate simulation**: steps [77, 108) of `nominal_heatup`, centered on its peak `rho(Ap)` (k=87, rho=1.7217) -- the gelation region both controllers pass through in the standard 160-step run. No separate plant initial condition was introduced (would have required modifying `AutoclavePlant`, out of scope).

## Heat-up slew-saturation caveat

Of the first 60 steps (heat-up phase, before any disturbance), **25 steps (41.7%)** had BOTH controllers pinned at the `TA_RATE_MAX` slew limit (4.0 degC/min) simultaneously.

**This means 41.7% of heat-up agreement is a slew-limit artifact, not evidence of solver equivalence** -- both controllers ramp at the same physically-imposed maximum rate regardless of any QP-level agreement, and would agree there even if their underlying QPs differed substantially, per the explicit instruction not to over-interpret this.

## Summary metrics

| Scenario | RMS control diff (degC) | RMS trajectory diff | Max abs control diff (degC) | Max abs trajectory diff | SNN verified-convergence rate | Max constraint residual | Clipped outputs | Mean objective gap (feasible steps) |
|---|---|---|---|---|---|---|---|---|
| nominal_heatup | 0.7074 | 0.2510 | 3.5158 | 0.6675 | 51.25% | 3.4572e-05 | 6 (3.8%) | 0.0007 |
| disturbance_step60 | 0.7929 | 0.2861 | 3.9519 | 0.6730 | 46.88% | 1.9194e-05 | 2 (1.2%) | 0.0027 |
| stiff_exotherm_window (steps 77-107 of nominal_heatup) | 0.5654 | 0.4183 | 0.6551 | 0.6675 | 22.58% | 1.9200e-05 | 4 (12.9%) | 0.0026 |

`trajectory_difference(k)` = Euclidean norm of `[Tc1,Tc2,Tc3,alpha1,alpha2,alpha3]_cvx - [...]_snn`. `objective_gap` is computed and averaged ONLY on steps where the SNN's own scaled constraint residual is within `feasibility_tol` -- **a lower objective on an infeasible step is never counted here as evidence of anything**.

## Feasibility and convergence detail

- nominal_heatup: 160/160 steps (100.0%) were feasible (scaled) enough to compute an objective gap; verified SNN convergence rate 51.25% (raw self-reported `converged` rate 51.25%).
- disturbance_step60: 160/160 steps (100.0%) feasible; verified convergence rate 46.88%.
- stiff_exotherm_window: 31/31 steps (100.0%) feasible; verified convergence rate 22.58%; max constraint residual 1.9200e-05 -- consistent with the earlier finding that both solvers can struggle on the raw scaled arrays at the gelation peak.

## Files

- `per_step_metrics.csv` -- every field listed in the task, for `nominal_heatup` and `disturbance_step60` (`stiff_exotherm_window` is a labeled subset of `nominal_heatup`'s rows).
- `summary.json` -- full provenance (git commit, package versions, solver config, cost weights, physical constraints, disturbance convention) plus all aggregate metrics and metric definitions.
- `comparison_plot.png` -- applied Ta overlay for both scenarios plus the control-difference trace, with the stiff-exotherm window shaded.

## Known limitations (carried from prior stages, still true)

- `trust_region` remains the one documented, permitted divergence between the two controllers' `Ap,Bp` -- the canonical QP construction itself is unified, but the two controllers do not always solve numerically identical QPs at every step (only when `trust_region`'s clamp is inactive).
- The SNN's raw solver output can diverge at the gelation peak on the raw/scaled arrays (see `results/qp_conditioning_report.json`, `results/snn_solve_diagnostics.json`); the applied control there is recovered by the existing slew-rate safety clip, which is reported (`snn_clipped`), not concealed.
- The row-normalization conditioning change proposed and tested in the prior stage was rejected after a full-solve comparison showed it regressed feasibility; `_condition` here is the original, unmodified formula.