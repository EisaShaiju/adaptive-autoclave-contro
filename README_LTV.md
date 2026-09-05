# Why this branch exists: the LTV (time-varying) prediction rewrite

This branch implements the "principled remedy" named throughout
`docs/PHASE4_VALIDATION_REPORT.md` (sections 18-19) and
`docs/SNN_MPC_TECHNICAL_REPORT.md`: replacing the frozen-Jacobian (LTI)
prediction model with a linear time-varying (LTV) one that re-linearizes at
each step along a predicted trajectory, instead of freezing a single Jacobian
at the current state and reusing it for the entire prediction horizon.

## The chain of evidence that led here

1. **The frozen model over-predicts the exotherm.** Measured on `main`
   (`docs/PHASE4_VALIDATION_REPORT.md` section 14 / section 19): ten steps
   ahead during gelation, the frozen-Jacobian prediction says the
   through-thickness gradient reaches ~1798 degC; the actual plant reaches
   ~6.8 degC. The reaction is self-limiting as alpha approaches 1, and a
   frozen Jacobian cannot see its own extinction.
2. **This over-prediction is what manufactures the infeasibility at stiff
   steps**, per the project's research advisor: the frozen model over-states
   future constraint violations, and that over-statement -- not a genuine
   physical limitation -- is what empties the feasible set.
3. **Before committing a ~week rewrite, a bounded check was run** on `main`
   (`tools/ltv_feasibility_probe.py`, `docs/PHASE4_VALIDATION_REPORT.md`
   section 19): at six stiff states, driving the frozen prediction and a true
   nonlinear rollout with the identical control sequence, **63.3% of the
   gradient rows the frozen model flags as violated turned out to be
   artifacts**, with zero missed (dangerous-direction) rows. This settled the
   question in favor of the rewrite, by the advisor's own stated test.
4. **This branch is that rewrite.**

The advisor was explicit that this is a **feasibility fix, not a convergence
fix** -- it should shrink the infeasible region and the resulting slack
magnitudes, and should *not* be expected to move the 16.1% stiff-window
convergence rate, since `tools/constraint_set_experiment.py` already showed
convergence is insensitive to the constraint set's structure. One of the
results below does not match that expectation, and is flagged prominently
rather than smoothed over.

## What changed, technically

- **`src/dynamics.py`**: two new functions, `linearize_trajectory(x0, u_seq,
  trust_region)` and `shift_nominal_sequence(u_seq, u_prev, N)`.
  `linearize_trajectory` rolls the true non-linear plant
  (`src/plant_simulator.AutoclavePlant`) forward under a nominal control
  sequence and calls the existing `linearize()` at each visited state,
  returning one `(Ap, Bp)` pair per horizon step instead of one for the whole
  horizon. It deliberately reuses the exact non-linear model this repo
  already treats as ground truth, rather than a separate cruder model --
  consistent with the rest of the repo, which makes no plant-model-mismatch
  assumption anywhere else either.
- **`src/qp_builder.py`**: `build_canonical_qp` now accepts `Ap`/`Bp` as
  either a single array (existing LTI behaviour, byte-for-byte unchanged) or
  a list of `N` arrays (new LTV behaviour, a separate code branch with the
  correct time-varying recursion). This was deliberately kept as two
  independent branches rather than one generalized loop, specifically so the
  already-validated LTI numbers (`tests/test_qp_parity.py`) cannot drift by
  even a rounding error -- confirmed: `max|dH|=0`, `max|df|=0`,
  `max|dA_ineq|=0`, `max|db_ineq|=0` against `main`.
- **Both controllers** (`src/mpc_cvxpy_controller.py`,
  `src/snn_mpc_controller.py`): new constructor parameter
  `linearization_mode='lti'|'ltv'` (default `'lti'`, so nothing changes for
  any existing caller). In `'ltv'` mode, `build_qp()` computes a nominal
  control sequence and calls `linearize_trajectory` instead of a single
  `linearize()` call. A second new parameter, `ltv_nominal_source=
  'warm_start'|'constant'` (default `'warm_start'`, ignored under `'lti'`),
  controls where that nominal sequence comes from -- see the ablation section
  below for why this was added after the fact.
- **`tools/final_controlled_comparison.py`**: new `--linearization-mode
  {lti,ltv}` flag, threaded through `CONFIG`/`make_controllers()`/the
  instrumented step functions, so a full closed-loop run can be regenerated
  under either mode for direct comparison. Also `--ltv-nominal-source
  {warm_start,constant}` (default `warm_start`, matching every result before
  the ablation section below), added specifically to test Caveat 2's
  hypothesis by removing the per-controller warm-started nominal without
  changing anything else.
- **`tests/test_ltv_dynamics.py`** (new): the LTV Phi/Gamma recursion is
  checked against an independent, hand-rolled forward simulation using three
  genuinely non-commuting Jacobians (a wrong multiplication order would fail
  this test but pass a same-matrix sanity check, so a "the matrices don't
  commute" assertion guards against a vacuous pass); LTV with a constant
  sequence is checked to numerically match the existing LTI path; the
  relative degree (5) is checked to survive LTV at both a benign and a stiff
  state; both controllers are checked to still produce bit-identical QPs
  under LTV given an identical nominal sequence. All 15 checks pass, and
  `tests/test_qp_parity.py` passes unchanged (zero LTI regression).

### A design decision worth stating plainly: what the nominal trajectory is

Each controller's `_u_nominal` is seeded from **its own previous solve**,
shifted by one step (the standard successive-linearization / warm-started
NMPC convention). This means that once the two controllers' solutions start
to differ even slightly -- which they always eventually do, since one solves
exactly (OSQP) and the other approximately (the SNN) -- **their LTV
re-linearization points diverge too**, compounding over time. This is
different from LTI, where the Jacobian depends only on the current state, not
on solve history, so it carries no such memory.

This does **not** break the formal "same QP" parity claim: given an
*identical* nominal sequence, both controllers still produce bit-identical
QPs (`tests/test_ltv_dynamics.py` check 4). But it does mean the two
closed-loop trajectories -- which already run as two independent simulations
in `tools/final_controlled_comparison.py`, each reacting to its own prior
solve -- diverge somewhat faster under LTV. See the RMS-difference result
below; it is a real, measured consequence of this choice, not a bug, and an
alternative (a single shared nominal trajectory, e.g. always holding `u_prev`
constant, or always deriving it from one designated reference solver) was not
implemented or tested here. That is a legitimate open design question for
whoever picks this branch up next.

## Measured result: LTI vs. LTV at the recommended configuration

Both runs: `N=10`, soft state constraints, `k0_scale=0.1`, `trust_region=False`,
same commit, same harness (`tools/final_controlled_comparison.py`), one run
each (this repo has no RNG anywhere -- confirmed by inspection -- so a run is
exactly reproducible, not a stochastic sample).

```
.venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --label lti
.venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --linearization-mode ltv --label ltv
```

| metric | LTI | LTV | reading |
|---|---|---|---|
| Clipping (all 3 scenarios) | 0.0% | 0.0% | unchanged -- both stay at the Revision-5 result |
| Max constraint residual | ~7.6e-7 | ~7.3e-7 | unchanged, both tiny |
| Cure completion (all 4 sub-runs) | alpha to 1.0, `cured=True` | alpha to 1.0, `cured=True` | no N=5-style trap; checked before believing anything else |
| Max unactionable predicted violation, nominal | 391.3 degC | 191.7 degC | **halved** -- the dropped dead rows' predicted (but unenforceable) violation shrinks with a more accurate model, as expected |
| Max unactionable predicted violation, disturbance | 420.3 degC | 182.0 degC | same pattern |
| SNN convergence, nominal | 50.0% | 56.25% | up |
| SNN convergence, disturbance | 45.6% | 50.0% | up |
| **SNN convergence, stiff window** | **16.1%** | **41.9%** | **up substantially -- see caveat below** |
| Median SNN solve time, nominal | 99.1 ms | 49.6 ms | down (more steps converge and stop early) |
| Median SNN solve time, stiff window | 364.2 ms | 254.3 ms | down |
| SNN/CVXPY total-time ratio (median), nominal | 17.2x | 7.0x | improved |
| SNN/CVXPY total-time ratio (median), stiff window | 61.7x | 40.6x | improved |
| RMS applied-control difference, nominal | 0.71 degC | 1.08 degC | **up** -- see caveat below |
| RMS applied-control difference, disturbance | 0.79 degC | 2.12 degC | **up** |
| RMS applied-control difference, stiff window | 0.57 degC | 2.12 degC | **up** |

### Caveat 1: the stiff-window convergence result contradicts the advisor's stated prediction, and that is being reported, not hidden

The advisor's explicit prediction was that an accurate prediction "will not
[move convergence]... it is a feasibility fix, not a convergence fix." The
measured stiff-window rate moved from 16.1% to 41.9% -- a large change in the
opposite direction from that prediction. Two confounds must be stated before
this is read as a refutation of that prediction:

1. **The stiff window itself is defined differently in the two runs.** It is
   extracted as "peak-rho(Ap) plus/minus a margin" from each run's *own*
   closed-loop trajectory (`tools/final_controlled_comparison.py`), and the
   LTI and LTV closed loops are not the same trajectory (peak at step 87 vs.
   88; window `[77,108)` vs. `[78,109)`). This is the same class of caveat
   `docs/PHASE4_VALIDATION_REPORT.md` already states for other comparisons in
   this repo ("never compare two convergence rates without stating whether
   the constraint set matched") -- here the *states themselves* differ, not
   just the constraint set.
2. **The underlying QP genuinely differs between the two runs** -- that is
   the entire point of LTV -- so a change in `kkt_scale` (the certificate's
   problem-dependent threshold, see `docs/PHASE4_VALIDATION_REPORT.md`
   section 15.4) is expected and could move the rate in either direction
   independent of whether the *solution quality* changed.

Neither confound is a reason to discard the result. It is a reason to
present it exactly as measured, flag the disagreement with the advisor's
prediction explicitly, and ask him to weigh in before treating stiff-window
convergence-under-LTV as settled either way. **This is the single most
important open item for whoever reviews this branch.**

### Caveat 2, updated: the RMS-difference hypothesis was tested directly, and it does not hold

The original hypothesis (previous revision of this section): the RMS growth
comes from each controller's LTV re-linearization being seeded from *its
own* previous solve -- a path-dependent feedback loop absent in LTI, where
the Jacobian depends only on the current state, not on solve history.

This was named as a hypothesis, not measured. It has now been tested
directly with the natural ablation: `ltv_nominal_source='constant'` (see
"What changed, technically" above) makes both controllers always
re-linearize along a nominal that just holds `u_prev` constant across the
horizon, instead of shifting their own previous solve -- removing the
path-dependent memory entirely while changing nothing else. Same recommended
configuration (N=10, soft, k0_scale=0.1), same commit:

```bash
.venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --linearization-mode ltv --ltv-nominal-source constant --label ltv_constnom
```

| metric | LTI | LTV (warm_start) | LTV (constant) | reading |
|---|---|---|---|---|
| RMS applied-control difference, nominal | 0.71 | 1.08 | 1.06 | **unchanged by removing the memory** |
| RMS applied-control difference, disturbance | 0.79 | 2.12 | 1.89 | small drop, still far above LTI |
| RMS applied-control difference, stiff window | 0.57 | 2.12 | 2.07 | **unchanged by removing the memory** |
| SNN convergence, nominal | 50.0% | 56.25% | 56.25% | identical |
| SNN convergence, disturbance | 45.6% | 50.0% | 49.4% | ~identical |
| SNN convergence, stiff window | 16.1% | 41.9% | 41.9% | identical |
| Cure completion | cured | cured | cured | unaffected |
| Clipping | 0.0% | 0.0% | 0.0% | unaffected |

**The hypothesis is refuted.** Removing the per-controller warm-started
memory barely moves the RMS difference and does not move the convergence
rate at all. Whatever grows the disagreement between the two controllers
under LTV, it is not primarily the nominal *control* sequence's path
dependence.

The more likely mechanism, consistent with everything measured here: LTV
re-linearizes at `N` points along a rollout **from the current state**,
not just once at the current state the way LTI does. The two controllers'
plants are already two independent simulations that drift apart slightly
(one solves exactly, one approximately) -- true under LTI too. But under
LTI that drift only ever perturbs a *single* Jacobian evaluation per step;
under LTV the same drift perturbs the state at which *every one* of the
`N` re-linearization points is evaluated, and those points feed into `Phi`
(hence `f` and the constraint offsets `b_ineq`) through the current state's
own already-nonzero cure/temperature components -- confirmed directly in
`tests/test_ltv_dynamics.py` section 5, where two very different nominal
control sequences at the same stiff state left `H`/`A_ineq` exactly
unaffected but changed `f`/`b_ineq` by tens of units, because the plant's own
existing cure state, not the commanded input, is what the differing
Jacobians act on. LTV therefore appears to amplify **existing small-state
divergence**, not to introduce a new one through its nominal-sequence
bookkeeping. This is a stronger, evidence-backed account than the original
hypothesis, but it was reasoned from the ablation's negative result, not
confirmed by a second, independent positive test (e.g. deliberately forcing
identical plant states on both sides and checking the RMS difference
vanishes) -- that would be the natural next experiment.

## What is still open after this branch

- **The stiff-window convergence result needs the advisor's review** before
  it is treated as a settled number (Caveat 1). Do not quote 41.9% as a
  replacement for 16.1% in any headline table without that review. The
  ablation above adds evidence relevant to that review: convergence under
  LTV is completely insensitive to the nominal-sequence mechanism, so
  whatever moved it is tied to the re-linearization itself, not this
  particular design choice within it.
- **The RMS-difference growth's mechanism is now narrowed, not fully
  isolated.** The ablation rules out the warm-started nominal sequence as
  the primary cause and points at LTV's amplification of pre-existing
  inter-controller state divergence instead (see above), but that account
  has not itself been directly tested (e.g. by forcing both controllers onto
  one shared plant state and checking the difference disappears).
- **Only the recommended configuration (N=10, soft, k0_scale=0.1) was run.**
  The N=20/hard/trust_region sweeps that exist for the LTI baseline
  (`results/final_comparison/`) were not repeated under LTV.
- **No PDF/paper-grade writeup was produced for this branch.** The technical
  report and validation report on `main` describe the LTI results only, plus
  the pre-rewrite feasibility check; they have not been updated with this
  branch's numbers, deliberately, until the open items above are resolved.

## Reproducing this branch's results

```bash
# Regression + correctness tests (both must pass with zero failures)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_qp_parity.py
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_ltv_dynamics.py

# LTI baseline (regenerated fresh on this branch, for a same-commit comparison)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --label lti

# LTV
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --linearization-mode ltv --label ltv

# LTV, nominal-source ablation (Caveat 2)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1 --linearization-mode ltv --ltv-nominal-source constant --label ltv_constnom
```

All three write to `results/final_comparison/<commit>_<label>_<timestamp>/summary.json`.
