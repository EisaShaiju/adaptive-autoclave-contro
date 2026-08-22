# Phase-4 Validation Report — SNN-QP vs. CVXPY/OSQP

**Revision 4.** Revision 1 reported the comparison *before* the advisor's
three technical points were addressed. Revision 2 superseded it: the
prediction model became identical on both sides, the per-step QP became
feasible at stiff steps, and every number was regenerated. Revision 3 proved
the stiff-step infeasibility algebraically rather than inferring it (§4.1),
resolved the `n_projections = 0` anomaly as the same mechanism (§4.1), reported
the convergence measurement at the **working** horizon rather than generalising
from N = 5 (§5.2), reconciled the N = 5 trap temperatures against the saved
trace (§7), and gave every quoted figure a file
(`results/artifact-index.md`).

**Revision 4 upgrades the solver dependency from `snn_opt` 0.4.0 to 0.6.0 and
this materially changes the result** (§5.3). Formal convergence moves from
**0 %** to **51.3 % / 46.9 % / 22.6 %** across the three scenarios, every step
becomes feasible, the max constraint residual falls five orders of magnitude,
clipping falls from 13.1 % to 3.8 %, and the SNN gets ~2.4× faster. The
headline conclusion is upgraded from **B** to **B′** — the solver now converges
on a substantial minority-to-majority of steps, but **not reliably**, and not
at all at N = 20. Where a metric moved, both values are stated.

**Source commit:** `70b83cd9b5dd35b2caf967e936355102037f597b`, **working tree
dirty** — the numbers describe the working tree, not a tagged release.
Reproducing them requires the uncommitted `src/qp_builder.py`,
`src/dynamics.py`, and the modified controllers, not just the named commit.

**Provenance:** every figure below maps to a file and a regenerating script in
`results/artifact-index.md`, including that index's own known caveats.

---

## 0. What changed since Revision 1, and why

| Advisor's point | Status | Action taken |
|---|---|---|
| SNN uses a different prediction model (clamps Arrhenius Jacobian terms) | **Fixed** | `trust_region` was hardcoded `True` (SNN) / `False` (CVXPY). It is now an explicit constructor parameter **defaulting to `False` on both**, so the model-identical configuration is the default and any divergence must be asked for deliberately. |
| SNN "rescales A" | Already fixed | The `Ap *= 0.98/rho` line was removed in commit `21261d2`; verified absent. |
| Constraints enforced by output clipping, not the solver | **Fixed** | Box, slew, and gradient constraints are all explicit `A_ineq` rows in the shared canonical QP for *both* controllers (`src/qp_builder.py`). The scalar output clip remains only as a reported safety filter, and its activation rate is published below. |
| SNN not converging; runs to full iteration budget; returns an infeasible point | **Root cause found** | See §4. The hard-constrained QP is **infeasible at stiff steps** — OSQP reports `infeasible` on it too. No solver can converge feasibly on a problem with no feasible point. |
| Raw command wants ≈ +11 °C/step, clipped to +4 | **Confirmed** | Measured: median requested step on clipped steps was **12.3 °C**, clipped to exactly **4.0 °C**. |
| Heat-up agreement is a slew-saturation artifact | **Confirmed and quantified** | 41.7 % of the first 60 steps have *both* controllers pinned at `TA_RATE_MAX`. Reported as a caveat, not as evidence. |
| Conditioning is the problem | **Partly refuted, then superseded** | Eigen-whitening `H` to `cond = 1.0` exactly changed final feasibility by < 3 % — Hessian conditioning was never the dominant driver. The dominant driver was QP infeasibility (§4). |
| FPGA work | **Deferred** | §13. |

### Revision 3 — the advisor's follow-up points

| Advisor's point | Status | Action taken |
|---|---|---|
| The gradient-constraint offset is decision-independent, so no rescaling can fix it | **Confirmed and strengthened** | §4.1 proves it algebraically: `Γ`'s block-row 0 is never written (empty loop range at `i=0`), so gradient rows `k = 0..4` have `c_j` **exactly** zero at the probed state; rows `k = 2,3,4` pair that with a negative RHS, making them unsatisfiable for *every* `z`. Not an amplification artifact. |
| `converged=False` reflects the stopping condition, not a wrong answer | **Confirmed** | §5.1. Whitening to `cond(H) = 1.0` changes nothing (conditioning ruled out); the absolute `proj_grad_tol` is the blocker. |
| Do not overclaim from N = 5; report the relative norm at the working horizon | **Actioned** | §5.2 reports N = 20 directly: relative projected-gradient norm **0.449–0.670** depending on `k0_scale` — a scale-invariant test would still **not** fire. The claim made is "`u₀` matches OSQP to 1.5e−6 °C where it can be checked", not "a better tolerance converges everywhere". Limitations and §13 updated to match. |
| Isolate `n_projections = 0`: genuine, or a counter bug? | **Resolved — genuine** | §4.1. `_project_adaptive`'s `if ‖c_j‖² < 1e-12: continue` guard skips the zero row *without* incrementing the counter or changing the residual, so `argmax` re-selects it indefinitely. The counter is correct; the loop is stuck. Fed back as an upstream recommendation (§11.5). |
| Check the latest upstream `SNN_opt` | **Actioned, not adopted** | v0.6.0 ships a scale-invariant KKT certificate and v0.5.0 a `projection_budget_exhausted` watchdog — both target defects found here. Installed remains 0.4.0; nothing was installed. §11.1, `results/snn_opt_upstream_diff.json`. |
| The N = 5 trap cools below the reported 11.4 °C | **Reconciled** | §7. 11.41 °C is `Tc1` *final* in `nominal_heatup`; the `disturbance_step60` scenario in the same CSV reaches **1.85 °C**. Scenario, column and reduction are now named, and `α ≤ 1.0e−5` replaces "never leaves 0.0". |
| Persist console-only quantities (ΔAp, off-by-one deltas) | **Actioned** | `results/qp_parity_diagnostics.json` (off-by-one deltas, trajectory ΔAp) and `results/ap_parity_grid.json` (the ≈1e−4 benign / ≈1851 gelation figures). Full map in `results/artifact-index.md`. |
| Hold the hardware phase | **Held** | §13, with the three specific software conditions that would make it meaningful. |

### Revision 4 — solver dependency upgrade

| Change | Status | Action taken |
|---|---|---|
| Adopt `snn_opt` 0.6.0 | **Done** | §5.3. Convergence 0 % → 22.6–51.3 %, all steps feasible, residual −5 orders, clipping −3.5×, solve time −2.4×. Required two non-obvious config changes; not a drop-in. |
| Verify the §5.2 N = 20 prediction | **Confirmed** | The scale-invariant certificate still does not fire at the working horizon — KKT residual/tolerance = 113. Prediction was recorded *before* the experiment. |
| Independent check of the §4.1 infeasibility proof | **Corroborated** | 0.6.0 rejects the hard-form QP naming **row 82** — the same row this report derived analytically, found independently upstream. |
| Guard against a false accuracy regression | **Checked** | The apparent objective-gap increase is a coverage artifact; on a common step subset the two versions are statistically identical (§5.3). |

---

## 1. Definitions

No use of "converged", "feasible", "equivalent", "optimal", or "safe" below is
unqualified; each maps to one of these:

- **Formally converged** — `result.converged == True` **and** constraint
  residual ≤ `feasibility_tol` (0.01) **and** `convergence_reason !=
  "max_iterations"`.
- **Feasible** — the mapped-back decision vector's maximum constraint residual,
  evaluated against the **original, unconditioned** canonical QP, is ≤ 0.01.
- **Objective gap** — `(f_snn − f_ref)/max(1, |f_ref|)` against an OSQP solve of
  the *identical* arrays; computed **only on feasible steps**.
- **Safe** — (a) applied `Ta` never violates its box/rate bounds, and (b)
  `|Tc1 − Tc3|` stays within `GRADIENT_MAX = 10 °C` (+0.1 buffer).

## 2. Methodology

One shared harness (`tools/final_controlled_comparison.py`) drives two
independent `AutoclavePlant` instances from an identical initial state
(28 °C), identical target (120 °C), identical horizon, identical sampling time
(`TE = 60 s`), identical constraints, and the identical canonical QP
construction — with the disturbance injected **before** control-compute in
both branches. All configuration flags (`trust_region`,
`soft_state_constraints`, `k0_scale`, horizon) are applied from one shared
dict to **both** controllers; the harness cannot express an asymmetric
configuration.

Every SNN metric is computed by mapping the solver's raw output back to
physical units (`U_sol = final_x / D`) and evaluating it against the
**original, unconditioned** canonical QP — never from the solver's own
scaled-space self-report. No RNG exists anywhere in the plant or controllers;
results are deterministic.

**Reproduction:**
```
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe \
  tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/conditioning_sweep.py
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tools/optimum_agreement_probe.py
```

## 3. Canonical QP definition

Both controllers condense to the same dense QP:

```
min_z  0.5 zᵀ H z + fᵀ z     s.t.  A_ineq z ≤ b_ineq
```

Hard form: `z = [Ta_0 … Ta_{N−1}]`, 6N constraint rows (box, slew, gradient).
Soft form (new): `z = [Ta_0 … Ta_{N−1}, s_0 … s_{N−1}]`, 7N rows — the
**predicted-state (gradient) rows** gain non-negative slacks with an exact
penalty (`slack_weight_lin = 1e3`, `slack_weight_quad = 1e2`). **Actuator box
and slew rows stay hard**: they are real actuator limits, always satisfiable,
and softening them would be exactly the hide-an-infeasibility anti-pattern.
The exact-penalty term drives slacks to zero whenever the hard constraint is
attainable — verified: slack `≈ 4e−29` at a benign state.

### 3.1 The penalty weights are not tuned — sensitivity sweep

`slack_weight_lin = 1e3` and `slack_weight_quad = 1e2` are the shipped
defaults, and the obvious objection is that they were chosen to produce a
convenient answer. They were not. Sweeping each over five values (four to five
orders of magnitude) at both a benign and the stiff state, with the other held
at its default:

| | benign state (k = 10) | stiff state (k = 87, ρ = 2.49) |
|---|---|---|
| `u₀` across **all 10** configurations | 72.0000 °C, every one | 130.0000 °C (129.95 at `quad = 1e4`) |
| **`u₀` spread over the whole sweep** | **0.000000 °C** | **0.048 °C** |
| Max slack at the optimum | 1.3e−29 … 2.4e−27 | 9.14e3, unchanged by any weight |
| Feasible set non-empty (slack LP) | yes, all | yes, all |

Two conclusions, both load-bearing:

1. **No silent relaxation.** Where the hard gradient constraint *is* attainable
   (benign state), slacks collapse to ~1e−27 or smaller at **every** weight
   tested — including `lin = 1e1`, two orders below the shipped value. The
   exact-penalty property is a structural feature of the L1 term, not an
   artifact of the chosen weight. This is the property that makes softening
   safe: it cannot quietly loosen a constraint that could have been met.
2. **The conclusions are weight-insensitive.** Total `u₀` variation across the
   entire sweep is 0.048 °C at the stiff step and exactly zero at the benign
   one — against a reported RMS control difference of 0.714 °C. No result in
   this report would change under any weight tested.

This is the expected behaviour for an exact (L1) penalty: once the weight
exceeds the optimal multiplier of the corresponding hard constraint, the soft
problem's solution *coincides* with the hard problem's wherever the latter is
feasible. The correct signature is therefore a **plateau**, not an optimum —
and a plateau is what the sweep shows.

**The unflattering half.** At the stiff state the slack sits at `9.14e3` and no
weight reduces it. That is not a tuning failure — it is the infeasibility of
§4.1 restated: with `ρ(Ap) = 2.49` the linear model predicts a thermal
runaway, so the predicted gradient genuinely cannot be held under
`GRADIENT_MAX = 10`. The slack is absorbing the model's own runaway prediction.
The soft form makes the QP solvable; it does not make the underlying constraint
achievable, and §10 records the resulting weakened guarantee.

Source: `tools/slack_weight_sensitivity.py` →
`results/slack_weight_sensitivity.json`.

### 3.2 Literature basis for the soft reformulation

The formulation follows the elastic / ℓ1-penalty treatment in **"Sequential ℓ1
Quadratic Programming for Nonlinear Model Predictive Control"**
(IFAC-PapersOnLine, <https://www.sciencedirect.com/science/article/pii/S2405896319301934>),
the governing source for slack-variable handling in this repository. Soft
output/state constraints
with an exact penalty are standard MPC practice, not a device introduced for
this project.

Where this repository **agrees** with that source: slacks are explicit decision
variables carried in the QP (not post-hoc repair); the penalty is exact (ℓ1
linear term), so it is not a generic relaxation; slack values and penalty
weights are reported rather than absorbed (§3.1); and the report keeps its four
distinct operations separate, as the reference requires — explicit slack
formulation, a solver returning an infeasible point, projection onto the
feasible set, and post-solve output clipping (§8) are *not* interchangeable and
are never conflated here.

Where it **differs**, stated so the reader can judge: the cited work softens
constraints to keep *sequential QP subproblems* solvable during an SQP
iteration, expecting feasibility to be restored as the iterates converge. This
repository solves a **single** QP per control step, and at stiff steps the hard
constraint is not merely temporarily violated but **structurally unattainable**
(§4.1) — so the slack is not a transient restoration device but a permanent
admission that the predicted gradient limit cannot be held there. That is a
weaker guarantee than the reference's setting implies, and §10 records it as a
limitation rather than treating the citation as blanket justification.

Cost weights `Q_diag = [100,100,100,0…]`, `R = 0.1`, `S = 1.0` are identical on
both controllers (`tests/test_qp_parity.py`, 20/20 checks pass; given the same
`(Ap, Bp, x0, u_prev)` the two adapters produce **bit-identical** `H, f,
A_ineq, b_ineq`, `max|Δ| = 0` on every array, matching fingerprints).

## 4. Root cause: the stiff QP was infeasible, not merely ill-conditioned

`tools/conditioning_sweep.py` swept 24 configurations
(`trust_region` × `soft` × `N ∈ {20,10,5}` × `k0_scale`) on the stiff exotherm
QP. The decisive result:

**Every one of the 12 hard-constrained configurations is `infeasible` under
OSQP** — at every horizon, under both prediction models. The gradient
constraint at horizon step `N−1` has an offset `d ≈ G·Ap^{N−1}·x₀` amplified by
`ρ(Ap)^{N−1}` (≈ 4200× at `ρ = 1.55`, N = 20), placing the constraint boundary
astronomically far from any attainable point. The cold-start point is violated
by **254,376** before a single iteration runs.

This reframes the earlier diagnosis: the SNN was not failing to solve a hard
problem — it was being asked to find a feasible point of a set that is empty.
The projected-gradient iterate had nowhere to converge to. It also explains why
a variable-space rescaling could never fix it: `d`'s dominant term has no
decision-variable dependence, so no `z = Dẑ + z₀` transformation can shrink it,
which is precisely why the eigen-whitening experiment (`cond(H) = 1.0` exactly)
changed nothing.

With soft state constraints the reference solver returns `optimal` and the SNN
returns feasible points for the first time in this investigation.

### 4.1 The infeasibility is exact and structural, not an amplification artifact

The §4 account — a finite offset amplified by `ρ(Ap)^{N−1}` until it is
unreachable — is directionally right but understates the result. The first
gradient rows are not *nearly* decision-independent; they are **exactly** so,
and the infeasibility can be proved algebraically rather than inferred from a
solver's failure.

In `src/qp_builder.py`, `Γ`'s block-row `i` is built by
`for j in range(i-1, -1, -1)`, which is an **empty range at `i = 0`**. That
block-row is therefore identically zero for every state, at every MPC step —
`x₀` is the pinned current state, so by construction no future control can
influence it. The `k = 0` gradient-constraint row inherits this exactly. At the
gelation-onset state (`k = 84`, `ρ(Ap) = 1.5525`, hard form), rows `k = 1..4`
are measured exactly zero as well.

Reading off the right-hand sides:

| k | row | `c_j` | `b_ineq` | status |
|---|---|---|---|---|
| 0 | 80 | exactly **0** | +14.44 | trivially satisfied |
| 1 | 81 | exactly **0** | +13.24 | trivially satisfied |
| 2 | 82 | exactly **0** | **−17.67** | **unsatisfiable** |
| 3 | 83 | exactly **0** | **−89.97** | **unsatisfiable** |
| 4 | 84 | exactly **0** | **−225.05** | **unsatisfiable** |

A row with `c_j` exactly zero reduces `c_jᵀz + d_j ≤ 0` to the constant
`d_j ≤ 0`. Where `d_j > 0` — rows `k = 2,3,4` — **no `z` whatsoever** satisfies
it. Not a different scaling, not a different solver, not a longer run. Rows
`k ≥ 5` are also violated at cold start but are *not* zero-normed, so those are
ordinary violations. Slack LP confirms at the aggregate level:
`max(s*) = 1.60e5` hard versus `2.4e−7` soft, same state.

This also explains why softening works and rescaling cannot. Preconditioning
multiplies `c_j`; zero times anything is zero. The soft reformulation adds a
slack *column* to these rows, giving them a coefficient on a variable that can
actually move — the one class of change able to repair a structurally zero row.

**And it is the complete explanation of the `n_projections = 0` anomaly.**
`snn_opt`'s `_project_adaptive` guards degenerate rows with
`if self._c_norms_sq[j] < 1e-12: continue` — which skips the row *without*
incrementing `n_iters` and *without* subtracting anything from the residual.
So `argmax(g)` re-selects the same dead row on every one of the (up to)
`max_projection_iters` attempts, on every one of 8000 outer iterations. The
counter is not wrong; it faithfully reports a loop stuck on a row it
structurally cannot resolve, which is why the violation stays pinned at
`~2.25e12` while `‖x‖` grows a thousandfold. A separate and *unrelated*
pathology exists on the **soft** form at `k = 87`, where the projector
saturates at exactly 200.0/200 projections per outer iteration on a QP that is
independently confirmed feasible — a budget limit, not a degenerate row. The
two must not be conflated.

Source: `tools/gradient_row_infeasibility_probe.py` →
`results/gradient_row_infeasibility.json`;
`tools/feasibility_certificate_probe.py` → `results/feasibility_certificate.json`.

## 5. Convergence results

| Configuration | SNN formally converged | SNN feasible | Max residual |
|---|---|---|---|
| Hard, model-mismatched, N = 20 (Rev. 1) | 0 / 320 steps (0.0 %) | 20.6 % / 27.5 % of steps | 1.85e5 |
| Soft, model-identical, N = 20 | 0.0 % | 28.1 % / 38.8 % | 43.4 |
| Soft, model-identical, N = 10 — `snn_opt` 0.4.0 | 0.0 % | 48.1 % / 58.8 % | 1.55 |
| **Soft, model-identical, N = 10 — `snn_opt` 0.6.0** | **51.3 % / 46.9 %** | **100 % / 100 %** | **3.46e−5** |

Two separate improvements, from two separate causes. The QP reformulation
(Rev. 2–3) took the max constraint residual from 1.85e5 to 1.55 and roughly
doubled the feasible-step fraction. The dependency upgrade (Rev. 4, §5.3) took
it a further five orders to ~2e−5, made **every** step feasible, and moved the
formal flag off zero for the first time.

**§5.1 and §5.2 below analyse the 0.4.0 behaviour**, where the flag never fired
at all. They are retained unchanged because they establish *why* it never fired
— a scale-sensitive absolute criterion — which is precisely the defect
upstream's 0.6.0 certificate fixes, and because their N = 20 prediction is the
one §5.3 goes on to confirm. Read §5.3 for the current numbers.

### 5.1 Why the flag never fires — and evidence the answer is nonetheless right

`tools/convergence_blocker_probe.py` isolated the blocker: the projected-gradient
norm sits at **1.66e10** (N = 20) and is **unchanged by a 6× larger iteration
budget** (8000 → 50000). `snn_opt` tests this norm as an **absolute** quantity
against `proj_grad_tol = 5e-2`. On a problem whose gradient scale is ~1e10, that
demands twelve orders of magnitude of reduction — the criterion is not
scale-invariant and cannot fire regardless of solver quality.

`tools/optimum_agreement_probe.py` then tested whether the returned point is
actually correct, at **N = 5** where the `Ap^{N−1}` amplification is mild:

| Metric | Result |
|---|---|
| `u₀` vs. OSQP optimum | agree to **0.0005 °C** |
| Objective gap | **−5.4e−8** (relative) |
| Feasible | yes (residual 5.4e−4) |
| *Relative* projected-gradient norm | **0.0097 — inside the 0.05 tolerance** |
| `converged` flag | still `False` |

So on a well-posed stiff QP the SNN **does** reach the optimum, and a
scale-invariant form of the solver's own criterion **would** have fired. The
persistent `False` is a property of the library's absolute stopping test, not
evidence that the answer is wrong. This is a specific, reportable finding.

**Caveat, stated plainly:** this exactness holds at N = 5, which §7 shows is
useless as a controller. The N = 5 result must not be generalised — the
`Ap^{N−1}` amplification there is mild, and it is precisely the regime where a
relative test flatters the solver.

### 5.2 The same measurement at the working horizon — reported as it falls

Re-running the identical probe at N = 20, the horizon that actually cures the
part, across the step sizes in use (`k0_scale = 0.5` is the constructor
default; `0.1` is what the recommended configuration runs):

| N = 20 | `k0=0.5` | `k0=0.1` | `k0=0.05` |
|---|---|---|---|
| `u₀` vs. OSQP optimum | **0.0000 °C** | **0.0000 °C** | **0.0000 °C** |
| *Relative* projected-gradient norm | **0.670** | **0.449** | **0.595** |
| Inside the 0.05 tolerance? | no | no | no |
| Relative objective gap | 955 | 40.2 | 15.6 |
| Feasible (residual ≤ 1e−2) | no (1.56e−2) | yes (2.66e−3) | yes (1.11e−3) |
| `converged` flag | `False` | `False` | `False` |

Two things follow, and they must be stated separately.

**A scale-invariant test would still not fire at the working horizon.** The
relative projected-gradient norm is 0.449–0.670 depending on step size — an
order of magnitude above the 0.05 threshold, not marginally outside it.
Replacing the absolute criterion with a relative one is the correct fix for a
mis-specified flag, but it does **not** convert N = 20 into a converged solve,
and no claim in this report depends on it doing so. Any quoted relative norm is
meaningless without its `k0_scale`, which is why all three appear above.

**The applied move is nonetheless exactly right.** `u₀` agrees with the OSQP
optimum on the identical conditioned arrays to `1.5e−6 °C` at N = 20 — better
than the N = 5 case — while the *full-horizon* decision vector is far from
optimal (relative objective gap 40 at `k0=0.1`, iterate norm ~3e9). The
controller applies only `u₀`, so the closed-loop behaviour is governed by the
quantity that is correct; but the solver has **not** solved the whole QP, and
this report does not claim it has.

At N = 10 the same probe shows a `u₀` gap of **3.7–4.1 °C** at this stiff step
with a near-zero relative objective gap (~1e−3) — i.e. a genuinely flat optimum
where two materially different first moves cost almost the same. That is a
worst-case cold-start single step, not the closed-loop figure (§6 reports RMS
0.714 °C at N = 10 warm-started), but it is the least flattering point in the
sweep and is recorded rather than omitted.

Source: `tools/optimum_agreement_probe.py` → `results/optimum_agreement_probe.json`.

### 5.3 Upgrading `snn_opt` 0.4.0 → 0.6.0 — what changed and what did not

Upstream shipped the fix this report asked for. v0.6.0 replaces the absolute
projected-gradient test with a **scale-invariant KKT-cone certificate**
(`ConvergenceConfig.optimality_test = 'kkt'`); v0.5.0 turned
`max_projection_iters` into a hard **watchdog** and added
`projection_budget_exhausted`. Both are now adopted.

**Two traps, both hit and both documented — an upgrade alone is not enough.**

1. **The new certificate is not enabled by upgrading.** `proj_grad_tol`
   survives as a deprecated constructor-only alias, and passing it *silently*
   forces `optimality_test = 'legacy_projected_gradient'`. This controller
   passed it, so the first post-upgrade run was still on the **old** test.
   Measured and confirmed before drawing any conclusion; the controller now
   selects the certificate explicitly.
2. **The old projection budget silently broke the controller.** Under 0.5.0+
   semantics, `max_projection_iters = 200` aborts the solve after ~1 outer
   iteration. The solver returns essentially its cold start, and **the plant
   never cures**: final `α = 0.0000`, max `Tc1` 41.8 °C, 1.4 ms/step. This is
   the §7 degeneracy signature arriving through a dependency change rather than
   a horizon choice. Raising the budget to **2000** restores full cure
   (`α = 1.0000`, max `Tc1` 137.84 °C) and is ~2.4× faster than the legacy test
   at the same budget.

Both settings are version-detected in `src/snn_mpc_controller.py`, so the
repository still runs correctly under 0.4.x.

#### Measured effect, recommended configuration (N = 10, soft, `k0_scale = 0.1`)

| | 0.4.0 | **0.6.0** |
|---|---|---|
| Formal convergence — nominal | 0.0 % | **51.3 %** |
| — disturbance | 0.0 % | **46.9 %** |
| — stiff window | 0.0 % | **22.6 %** |
| Max constraint residual | 1.554 / 2.069 / 1.554 | **3.46e−5 / 1.92e−5 / 1.92e−5** |
| Steps feasible enough to grade | 48.1 % / 58.8 % / 45.2 % | **100 % / 100 % / 100 %** |
| Clipped applied moves | 13.1 % / 15.0 % / 29.0 % | **3.8 % / 1.3 % / 12.9 %** |
| Median SNN solve | 113.7 ms | **48.7 ms** |
| SNN ÷ CVXPY time | 19.8× | **8.9×** |
| RMS applied-control difference | 0.714 °C | 0.707 °C |
| Closed-loop overshoot / cure Δ | 13.24 °C / 0.3418 | 13.23 °C / 0.3417 |
| Final cure `α` (cure gate) | 1.00 / 1.00 / 0.9999 | 1.00 / 1.00 / 0.9999 |

Every non-timing number in this table is bit-reproducible: two independent
regenerations of the 0.6.0 run agreed to the last recorded digit on convergence,
residual, clipping, RMS difference and objective gap. **Timing is the one
exception** — absolute medians move with machine load (the SNN nominal median
was 47.0 ms on one run and 48.7 ms on the next), while the *ratio* is stable to
within a few percent (8.6× vs 8.9×). Quote the ratio; treat the absolute
milliseconds as indicative.

#### The objective gap did **not** get worse — a like-for-like check

The naive comparison shows the mean objective gap rising (1.05e−4 → 6.72e−4).
That is an artifact: 0.4.0 could only evaluate the gap on the 48 % of steps
where its output was feasible enough, while 0.6.0 evaluates **all** of them,
including the hard ones. Restricted to the *same* steps 0.4.0 could measure:

| scenario | 0.4.0 | 0.6.0, same steps | 0.6.0, all steps |
|---|---|---|---|
| nominal (n = 77) | 1.201e−4 | **1.215e−4** | 6.724e−4 |
| disturbance (n = 94) | 1.348e−4 | **1.347e−4** | 2.651e−3 |

Accuracy is unchanged; coverage more than doubled. Reporting the headline
numbers without this check would have claimed a regression that does not exist.

#### What did **not** change

- **N = 20 still does not converge.** The prediction recorded in §5.2 before the
  upgrade — that a scale-invariant test would still not fire at the working
  horizon, given a relative projected-gradient norm of 0.449–0.670 — is
  **confirmed**: the KKT residual/tolerance ratio at the stiff N = 20 step is
  **113**. It fails by two orders of magnitude, not marginally.
  (N = 5 now converges formally for the first time, ratio 1.9e−7; N = 10 is
  the interesting middle at 2.40.)
- **Closed-loop behaviour is essentially identical** (overshoot 13.24 → 13.23 °C,
  cure Δ 0.3418 → 0.3417, 2 gradient excursions either way). The upgrade
  improved the *solver*, not the *control outcome* — which is itself evidence
  that the applied move was already right, as §5.2 argued.
- **The SNN is still slower than OSQP** — ~8.9× rather than 19.8×.
- **Clipping has not vanished**: 12.9 % of stiff-window moves still come from
  the safety filter.

#### Independent confirmation of §4.1

Constructing the **hard**-form stiff QP under 0.6.0 raises:

```
ValueError: constraint row 82 has a zero normal and d > 0:
the problem is certifiably infeasible (0 <= -176724786245.99728)
```

**Row 82 is exactly the `k = 2` row identified analytically in §4.1**, found
independently by the upstream library. The infeasibility argument in this
report is therefore corroborated by a third party that had no knowledge of it.
`SNNMPCSolver` catches this, counts it in `n_infeasible_qp`, and holds the
previous input rather than crashing; it does not fire on the recommended soft
form.

Source: `tools/kkt_certificate_probe.py` → `results/kkt_certificate_probe.json`;
`tools/snn_opt_regression_baseline.py` →
`results/snn_opt_regression_{pre040,post060_adapted}.json`;
`results/final_comparison/d80ccb05_N10_soft_snnopt060_*/`.

## 6. Control and trajectory comparison — the three requested numbers

Recommended configuration: **N = 10, soft state constraints, model-identical
(`trust_region = False` both sides), `k0_scale = 0.1`**.

Numbers below are **`snn_opt` 0.6.0** (current). The 0.4.0 values are given in
parentheses where they differ materially — see §5.3 for the full comparison.

| | nominal heat-up | disturbance @ 60 | stiff-exotherm window |
|---|---|---|---|
| **1. RMS applied-control difference** | **0.707 °C** (0.714) | **0.793 °C** (0.791) | **0.565 °C** (0.559) |
| **2. RMS closed-loop trajectory difference** | **0.251** | **0.286** | **0.418** |
| Max abs. control difference | 3.52 °C (3.57) | 3.95 °C | 0.66 °C |
| Max abs. trajectory difference | 0.668 | 0.673 | 0.668 |
| **3a. SNN max constraint residual** | **3.46e−5** (1.554) | **1.92e−5** (2.069) | **1.92e−5** (1.554) |
| **3b. SNN formally converged** | **51.3 %** (0.0) | **46.9 %** (0.0) | **22.6 %** (0.0) |
| Feasible steps (obj-gap computable) | 100 % (48.1) | 100 % (58.8) | 100 % (45.2) |
| Mean objective gap on feasible steps | 6.72e−4 † | 2.65e−3 † | 2.56e−3 † |
| Clipped outputs | 6 (3.8 %) | 2 (1.3 %) | 4 (12.9 %) |

† **Not comparable to the 0.4.0 figures** (1.05e−4 / 1.22e−4 / 5.02e−5), which
were averaged over only the ~half of steps that version could grade. On the
*same* step subset, 0.6.0 gives 1.215e−4 and 1.347e−4 — statistically identical.
See §5.3.

Against Revision 1 (hard constraints, mismatched model, N = 20): RMS control
difference **16.005 → 0.714 °C** (−96 %), max control difference
**57.70 → 3.57 °C** (−94 %), max residual **1.85e5 → 1.55** (−5 orders),
clipping **19.4 % → 13.1 %**.

**Slew-saturation caveat.** 41.7 % of the first 60 heat-up steps have *both*
controllers pinned at `TA_RATE_MAX`; in the stiff-exotherm window it is 77.4 %.
Agreement on those steps reflects a shared actuator limit, **not** solver
agreement, and is excluded from any equivalence claim.

## 7. Horizon is a genuine constraint, not a free parameter

| N | RMS ctrl diff | Formally converged | Final `α` (cure) | Max `Tc1` | Verdict |
|---|---|---|---|---|---|
| 20 | 4.259 °C | 0.0 % | 1.000 | 138.1 °C | cures; largest disagreement |
| **10** | **0.714 °C** | 0.0 % | **1.000** | 137.8 °C | **recommended** |
| 5 | 0.000 °C | 98.8 % | **0.000** | 28.0 °C | **degenerate — rejected** |

N = 5 produces apparently perfect agreement (RMS 0.000 °C, 98.8 % formal
convergence, zero clipping, zero gradient violations) — and is **worthless**.
With a 5-minute horizon the controller cannot see past the thermal lag, so the
energy term dominates and it drives `Ta` to the 10 °C minimum: in the
`nominal_heatup` scenario the part *cools* from 28 °C to `Tc1` = 11.41 °C by the
final step (`disturbance_step60`, the same run, reaches 1.85 °C — colder still,
since the 15 °C drop compounds the same saturated-low-`Ta` behaviour); and
**`α` stays at or below 1.0e-5 for the entire run — no cure occurs at all.** Both
controllers agree perfectly because both are doing nothing, saturated against
the same lower bound. This is reported specifically because it is the exact trap
of reading agreement as equivalence, and it would have been the headline result
had the cure state not been checked. (Source:
`results/final_comparison/70b83cd9_N5_soft_identical_20260811_015200/per_step_metrics.csv`,
columns `Tc1_cvx`/`alpha1_cvx`, both scenarios.)

## 8. Clipping analysis

| Configuration | SNN clipped | CVXPY clipped |
|---|---|---|
| Rev. 1 (hard, mismatched, N = 20) | 19.4 % (54.8 % in stiff window) | 0 % |
| N = 10, soft, identical — `snn_opt` 0.4.0 | 13.1 % (29.0 % in stiff window) | 0 % |
| **N = 10, soft, identical — `snn_opt` 0.6.0** | **3.8 %** (12.9 % in stiff window) | 0 % |

CVXPY has no clipping stage — its constraints are enforced inside the QP. The
SNN's clip is the exact Euclidean projection of the first move onto its
feasible interval. It is a scalar safety filter, **not** a feasibility guarantee
for the full decision vector, and its rate is published rather than absorbed.

Clipping has fallen by a factor of ~3.5 with the dependency upgrade (§5.3),
tracking the five-order improvement in constraint residual — the solver output
now needs correcting far less often. **It has not vanished**: at 12.9 % of
stiff-window steps the applied move still originates from the filter rather
than from a certified solver output, and that remains the single largest
outstanding objection to an equivalence claim.

## 8.5 Compute time

Previously absent from this report. Both controllers are now timed in the
**same process, on the same per-step sequence**, at the recommended
configuration (N = 10, soft, `trust_region = False`, `k0_scale = 0.1`).
`build` covers the canonical QP construction plus solver-object setup;
`solve` covers only the solver call. Medians are quoted alongside means because
CVXPY's first few solves are compilation-dominated outliers.

| Median total ms/step, nominal | `snn_opt` 0.4.0 | **`snn_opt` 0.6.0** |
|---|---|---|
| CVXPY / OSQP | 5.76 | 5.47 |
| SNN-QP | 113.75 | **48.66** |
| **Ratio (SNN ÷ CVXPY)** | **19.8×** | **8.9×** |

**The SNN is ~8.9× slower than OSQP — better than before, still not
comparable.** Under 0.4.0 the solver never terminated early, exhausting its
full 8000-iteration budget every step at a near-constant ~113 ms. Under 0.6.0
the KKT certificate lets roughly half of nominal steps stop early, roughly
halving the median; in the stiff window it drops further to 27.2 ms (5.0×
CVXPY), because there the projection watchdog terminates the hardest solves
sooner.

QP *construction* is marginally cheaper on the SNN side (0.73 vs 0.74 ms) —
the entire gap is in the solve.

**Read the ratio, not the milliseconds.** Timing is the only quantity in this
report that is not bit-reproducible. Repeating the identical 0.6.0 run on the
same machine moved the SNN nominal median from 47.0 to 48.7 ms and the CVXPY
median from 5.5 to 5.5 ms, i.e. the ratio from 8.6× to 8.9×; the disturbance
scenario moved further (both sides roughly +50 %) under background load while
its *ratio* held at 8.5–8.7×. Every other metric here — convergence, residual,
clipping, RMS difference, objective gap — reproduced to the last recorded digit
across those runs. Treat absolute milliseconds as indicative and the ratio as
the finding.

**Correcting an earlier claim.** A previous statement that compute time was
"comparable to, not faster than" OSQP was drawn from the superseded N = 20 /
hard-constraint harness, where CVXPY is itself slow (116 ms/step) and the ratio
flatters the SNN to ~1.6×. At the configuration this report actually
recommends, that claim does not hold and has been corrected here and in the
repository's contributor notes. The gap has since narrowed from 19.8× to ~8.9×
via the dependency upgrade (§5.3) — but "narrower" is not "comparable", and no claim of
computational advantage is made anywhere in this report.

**What this does and does not mean.** This is a CPU comparison between a
mature, C-backed interior-point/ADMM solver and a research SNN solver running
its full iteration budget — it is not evidence about the SNN's value
proposition, which rests on neuromorphic or FPGA execution where the
iteration structure maps to hardware differently. It *is*, however, the honest
CPU number, and no claim of computational advantage is made anywhere in this
report.

Source: `tools/final_controlled_comparison.py` →
`results/final_comparison/d80ccb05_N10_soft_timed_*/summary.json` (`timing_ms`),
per-step columns `build_ms_*`, `solve_ms_*`, `total_ms_*` in the CSV.

## 9. Plant behaviour (per the §1 definition of "safe")

At N = 10, soft, model-identical, both controllers produce **the same** closed-loop
outcome: 2 gradient-limit excursions per 160 steps, max `Tc1` 137.8 °C, final
`α = 1.000` (full cure), and **zero** actuator box or rate violations. Gradient
excursions are a property of the plant/horizon, not of the solver choice —
CVXPY exhibits them equally.

## 10. Limitations

- **Convergence is partial, not achieved.** Under `snn_opt` 0.6.0 the rate is
  51.3 % / 46.9 % / **22.6 %** (nominal / disturbance / stiff window). The
  stiff window — the regime the controller exists for — is the worst of the
  three. At **N = 20 the certificate fails by a factor of 113** and the §5.2
  prediction that a scale-invariant test would not rescue that horizon is
  confirmed.
- **The full-horizon solution at N = 20 is not optimal**, even though `u₀` is:
  relative objective gap 40 at `k0=0.1`, iterate norm ~3e9. Only the first move
  is applied, but "solves the same QP" refers to the problem posed, not to the
  whole vector returned.
- **The result is dependency-sensitive in both directions.** Two `snn_opt`
  configuration details — the deprecated `proj_grad_tol` alias silently
  selecting the legacy test, and `max_projection_iters` becoming a hard
  watchdog — each independently produce a *wrong* conclusion if missed: the
  first leaves the old scale-sensitive criterion in place, the second stops the
  plant curing altogether (§5.3). Any future dependency change must re-run
  `tools/snn_opt_regression_baseline.py` and check final `α ≈ 1.0` before any
  metric is read.
- Objective gaps across solver versions are **not** comparable unless
  restricted to a common step set — 0.4.0 could grade only 45–59 % of steps
  (§5.3).
- Soft state constraints change the control problem: gradient limits become
  strongly-penalised preferences rather than hard guarantees. Both controllers
  see the identical change, so the *comparison* stays fair, but the absolute
  safety guarantee is weaker than the hard formulation nominally promised (a
  formulation which was, at stiff steps, infeasible and therefore unimplementable).
- The stiff-exotherm scenario is a labelled window of the nominal run, not an
  independently-initialised simulation.
- Reducing N from 20 to 10 was selected empirically on this plant; it has not
  been justified from a control-theoretic stability argument.
- Numbers are not comparable to `docs/SNN_QP_SOLVER_PARITY_REPORT.md` (different
  methodology: misaligned scripts, pre-fix QP window, no residual definitions).

## 11. Next steps

1. ~~Evaluate `snn_opt` ≥ 0.6.0.~~ **Done — adopted in Revision 4, see §5.3.**
   Formal convergence 0 % → 22.6–51.3 %, all steps feasible, residual down five
   orders, clipping down 3.5×, solve time down 2.4×. The §5.2 prediction that
   the new certificate would still not fire at N = 20 was **confirmed** (KKT
   residual/tolerance = 113). Adoption required two non-obvious configuration
   changes, both documented in §5.3 — it was not a drop-in.
2. Investigate CVXPY's own `infeasible`-status steps under the hard formulation
   (21/160 nominal) — the symmetric counterpart to the SNN investigation. §4.1
   now supplies the mechanism: the hard form's `k = 0..4` gradient rows are
   exactly decision-independent, so CVXPY is correct to report `infeasible`.
3. Justify the horizon choice from the plant's thermal time constants rather
   than empirically.
4. ~~Resolve the unexplained `n_projections = 0`-despite-large-residual
   anomaly.~~ **Resolved — see §4.1.** The projector locks onto an exactly
   zero-normed, unconditionally-violated gradient row; `_project_adaptive`'s
   `if ||c_j||² < 1e-12: continue` guard skips it *without* incrementing the
   counter and *without* altering the residual, so `argmax` re-selects the same
   dead row indefinitely. Not a counter bug.
5. ~~Report the degenerate-row behaviour upstream.~~ **Already fixed upstream.**
   `snn_opt` 0.6.0 raises `ValueError: constraint row 82 has a zero normal and
   d > 0: the problem is certifiably infeasible` on the hard-form stiff QP —
   naming **the same row** this report identified analytically in §4.1, found
   independently. What remains is to confirm whether the *saturating* case
   (soft form, 200/200 projections per iteration) is fully addressed by the
   0.5.0 watchdog or only made visible by it.
6. **Close the stiff-window gap.** With convergence at 22.6 % and clipping at
   12.9 % there, the stiff exotherm is now the sole remaining barrier to an
   equivalence claim. Candidate directions: a horizon or terminal-set redesign
   that makes the hard gradient constraint attainable (which would remove the
   need for slacks entirely — see §3.1), step-size adaptation through the
   exotherm, or a warm-start strategy specific to the gelation transient.
7. **Re-examine `k0_scale`.** Under 0.6.0 the relationship between step size and
   final feasibility inverted at N = 20 (`k0 = 0.5` now reaches 8.7e−7 while
   `k0 = 0.1` exhausts its projection budget at 1.2e−2). The recommended
   `k0 = 0.1` was tuned against 0.4.0 and may no longer be optimal.

## 12. Overall conclusion

**B′ — Same QP; SNN-QP is feasible everywhere and converges on a substantial
fraction of steps, but not reliably, and not at the long horizon.**

"Same QP" is a substantiated claim, not a qualified one: the prediction model
is identical by default, the canonical construction is bit-identical given
identical inputs, and all constraints live in the QP for both controllers.

What changed in Revision 4 (§5.3), on the recommended configuration:

| | Rev. 1 | Rev. 3 (0.4.0) | **Rev. 4 (0.6.0)** |
|---|---|---|---|
| Formal convergence | 0 % | 0 % | **22.6–51.3 %** |
| Steps feasible enough to grade | 21–28 % | 45–59 % | **100 %** |
| Max residual | 1.85e5 | ~1.55 | **~2e−5** |
| Clipped applied moves | 19.4 % | 13.1 % | **3.8 %** |
| RMS applied-control difference | 16.005 °C | 0.714 °C | **0.707 °C** |

**Why this is still not conclusion A (equivalence).** Formal convergence is
22.6 % in the stiff window — the regime the controller exists to handle — so a
majority of the hardest steps still terminate without a certificate. Peak
control difference is 3.52 °C. At N = 20 the certificate fails by a factor of
113. And 12.9 % of stiff-window applied moves still come from the safety clip
rather than a certified solver output. No tolerance this report would state in
advance is met by those four numbers together.

**Why it is no longer plain B.** "Does not reliably converge" understates a
solver that is now feasible on 100 % of steps, certified on about half of the
nominal ones, and accurate to 1.2e−4 relative objective gap on every step it
can be graded on. The honest position is that the *previous* conclusion was
partly an artifact of a mis-specified stopping test and an inadequate
projection budget in the pinned dependency — and partly a real limitation that
survives the fix.

C is not available: the soft-form QP is feasible everywhere (§4.1), and the
hard-form infeasibility is now proved rather than inferred. D does not apply —
the comparison *was* fully identical, which is what Revision 2 established.

## 13. FPGA / hardware deployment

**Remains deferred**, in agreement with the advisor — but the case has moved.

Revision 3 listed three software problems blocking a meaningful hardware phase.
The 0.6.0 upgrade (§5.3) has substantially addressed two of them:

| Pre-hardware condition (Rev. 3) | Status |
|---|---|
| Projection budget no longer saturates | **Largely resolved** — watchdog terminates cleanly; residual ~2e−5 |
| Applied move no longer depends on the clip at ~29 % of stiff steps | **Improved, not resolved** — now 12.9 % |
| Full-horizon N = 20 solution is optimal, not just `u₀` | **Not resolved** — certificate fails by 113× |

The prediction recorded in Revision 3 — that a scale-invariant certificate
would make the *criterion* honest without making the N = 20 solve converge —
was confirmed exactly. That horizon remains unsolved.

**Why hardware is still premature.** Formal convergence in the stiff exotherm
window is 22.6 %, and 12.9 % of applied moves there still come from the safety
clip. Committing to fixed hardware means freezing the solver's behaviour in the
one regime where it is least certified. The remaining work (§11.6–11.7) is
narrow and well-posed — close the stiff window — and it is a software problem.

**What would change the recommendation:** stiff-window convergence
comfortably above ~90 % with clipping in the low single digits, on a
configuration that still cures the part. At that point the hardware experiment
tests the SNN's actual value proposition — energy and latency on neuromorphic
substrate — rather than baking in a solver that is unreliable exactly when the
plant is hardest to control.
