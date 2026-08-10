# Phase-4 Validation Report — SNN-QP vs. CVXPY/OSQP

**Revision 2.** Revision 1 reported the comparison *before* the advisor's
three technical points were addressed. This revision supersedes it: the
prediction model is now identical on both sides, the per-step QP is now
feasible at stiff steps, and every number below has been regenerated. Where
a metric moved, both values are stated.

**Source commit:** `70b83cd9b5dd35b2caf967e936355102037f597b`, **working tree
dirty** — the numbers describe the working tree, not a tagged release.
Reproducing them requires the uncommitted `src/qp_builder.py`,
`src/dynamics.py`, and the modified controllers, not just the named commit.

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

## 5. Convergence results

| Configuration | SNN formally converged | SNN feasible | Max residual |
|---|---|---|---|
| Hard, model-mismatched, N = 20 (Rev. 1) | 0 / 320 steps (0.0 %) | 20.6 % / 27.5 % of steps | 1.85e5 |
| Soft, model-identical, N = 20 | 0.0 % | 28.1 % / 38.8 % | 43.4 |
| **Soft, model-identical, N = 10** | **0.0 %** | **48.1 % / 58.8 %** | **1.55** |

Max constraint residual improved by **five orders of magnitude** (1.85e5 → 1.55)
and the feasible-step fraction roughly doubled. **The formal convergence flag
still never fires.**

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
useless as a controller. At the horizons that actually control the plant
(N = 10, 20) the returned point is feasible but not verified optimal — mean
objective gap 1.1e−4 on feasible steps, but 41–52 % of steps are not feasible
enough to evaluate at all.

## 6. Control and trajectory comparison — the three requested numbers

Recommended configuration: **N = 10, soft state constraints, model-identical
(`trust_region = False` both sides), `k0_scale = 0.1`**.

| | nominal heat-up | disturbance @ 60 | stiff-exotherm window |
|---|---|---|---|
| **1. RMS applied-control difference** | **0.714 °C** | **0.791 °C** | **0.559 °C** |
| **2. RMS closed-loop trajectory difference** | **0.249** | **0.279** | **0.414** |
| Max abs. control difference | 3.57 °C | 3.95 °C | 0.65 °C |
| Max abs. trajectory difference | 0.660 | 0.656 | 0.660 |
| **3a. SNN max constraint residual** | **1.554** | **2.069** | **1.554** |
| **3b. SNN formally converged** | **0.0 %** | **0.0 %** | **0.0 %** |
| Feasible steps (obj-gap computable) | 48.1 % | 58.8 % | 45.2 % |
| Mean objective gap on feasible steps | 1.05e−4 | 1.22e−4 | 5.02e−5 |
| Clipped outputs | 21 (13.1 %) | 24 (15.0 %) | 9 (29.0 %) |

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
energy term dominates and it drives `Ta` to the 10 °C minimum: the part *cools*
from 28 °C to 11.4 °C and **`α` never leaves 0.0 — no cure occurs at all.** Both
controllers agree perfectly because both are doing nothing, saturated against
the same lower bound. This is reported specifically because it is the exact trap
of reading agreement as equivalence, and it would have been the headline result
had the cure state not been checked.

## 8. Clipping analysis

| Configuration | SNN clipped | CVXPY clipped |
|---|---|---|
| Rev. 1 (hard, mismatched, N = 20) | 19.4 % (54.8 % in stiff window) | 0 % |
| **N = 10, soft, identical** | **13.1 %** (29.0 % in stiff window) | 0 % |

CVXPY has no clipping stage — its constraints are enforced inside the QP. The
SNN's clip is the exact Euclidean projection of the first move onto its
feasible interval. It is a scalar safety filter, **not** a feasibility guarantee
for the full decision vector, and its rate is published rather than absorbed.
Clipping fell but did not vanish; at ~29 % of stiff-window steps the applied
move still comes from the filter rather than a fully feasible solver output.

## 9. Plant behaviour (per the §1 definition of "safe")

At N = 10, soft, model-identical, both controllers produce **the same** closed-loop
outcome: 2 gradient-limit excursions per 160 steps, max `Tc1` 137.8 °C, final
`α = 1.000` (full cure), and **zero** actuator box or rate violations. Gradient
excursions are a property of the plant/horizon, not of the solver choice —
CVXPY exhibits them equally.

## 10. Limitations

- **The formal convergence criterion is never met** at any usable horizon
  (0/320 steps). The N = 5 evidence shows this is an absolute-vs-relative
  tolerance artifact, but that has been demonstrated only where the horizon is
  too short to control the plant.
- 41–52 % of steps at N = 10 are not feasible enough to evaluate an objective
  gap; the reported 1.05e−4 mean describes only the feasible remainder.
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

1. Raise the absolute-vs-relative `proj_grad_tol` issue upstream with `snn_opt`,
   or wrap a scale-invariant convergence test in the adapter, and re-measure the
   convergence rate at N = 10.
2. Investigate CVXPY's own `infeasible`-status steps under the hard formulation
   (21/160 nominal) — the symmetric counterpart to the SNN investigation.
3. Justify the horizon choice from the plant's thermal time constants rather
   than empirically.
4. Resolve the unexplained `n_projections = 0`-despite-large-residual anomaly.

## 12. Overall conclusion

**B — Same QP, but SNN-QP does not reliably converge.**

"Same QP" is now a substantiated claim rather than a qualified one: the
prediction model is identical by default, the canonical construction is
bit-identical given identical inputs, and all constraints live in the QP for
both controllers. The remaining shortfall is convergence, and it is precise:
**0 % formal convergence at every usable horizon, with 48–59 % of steps feasible
and a residual of ~1.55**. That is a large improvement over Revision 1
(0 %, 21–28 % feasible, residual 1.85e5) and the applied controls now agree to
**0.714 °C RMS**, but it is not equivalence.

Conclusion A is not available: 3.57 °C peak control difference and a formal
convergence rate of zero cannot be called equivalence under any tolerance this
report would be willing to state in advance. C understates the now-demonstrated
QP identity. D is not applicable — the comparison *was* fully identical, which
is exactly what this revision established.

## 13. FPGA / hardware deployment

**Remains deferred**, in agreement with the advisor. The formal convergence rate
is zero at every horizon that actually cures the part, and ~29 % of stiff-window
applied moves still originate from the safety clip rather than a feasible solver
output. Committing that behaviour to fixed hardware would freeze in the
iteration-budget exhaustion and clip dependence documented here. The most
valuable pre-hardware work is now narrow and well-defined: a scale-invariant
convergence test (§11.1), which is a software change.
