# Spiking neural network solvers for model predictive control of autoclave composite curing

**Eisa Shaiju**

---

> **Revision 2 — final software-phase report.** Revision 1 was written against
> `snn_opt` 0.4.0, in which the formal convergence flag never fired at any
> horizon. Section 3.4 of that revision diagnosed the cause as an absolute,
> scale-sensitive stopping test and predicted, on the record, that a
> scale-invariant replacement would raise the convergence rate at short and
> medium horizons but would **still not fire at the working horizon**. This
> revision reports the outcome of acting on that diagnosis. Every number below
> is measured under `snn_opt` 0.6.0 unless explicitly labelled 0.4.0, and every
> figure is traceable to a file through `results/artifact-index.md`.
>
> The headline conclusion is revised from *same problem, unreliable convergence*
> to *same problem, feasible everywhere, convergence established on a
> substantial minority of steps and specifically absent where the controller
> matters most*. It is still not an equivalence claim.
>
> Step-by-step validation evidence: `docs/PHASE4_VALIDATION_REPORT.md`.

---

## Abstract

Model predictive control (MPC) requires solving a constrained quadratic program (QP) at every
control interval, which places a hard computational floor on the achievable sampling rate and
motivates interest in solvers that map onto parallel, event-driven hardware. Recent work has
shown that the time-averaged firing rates of certain inhibition-dominated spiking neural
networks (SNNs) yield the solutions of linear and quadratic programs, suggesting that an SNN
could serve as a drop-in QP solver inside an MPC loop. Here we test that proposition on a
demanding physical problem: the autoclave cure of a thick-sectioned composite laminate, whose
exothermic Arrhenius kinetics make the linearised prediction model unstable precisely where
control is most critical. We implement a non-linear finite-difference digital twin, a classical
CVXPY/OSQP MPC baseline, and an SNN-QP controller built on a projected-gradient/LIF solver, and
we subject the claim of solver equivalence to direct measurement. We report four findings.
First, establishing that two controllers solve the *same* QP is substantially harder than it
appears: we identify and correct a one-step prediction-window error and an asymmetric Jacobian
clamp, either of which silently invalidates a head-to-head comparison, and we unify both
controllers onto a single canonical QP construction verified bit-identical across a full
trajectory. Second, the per-step QP is *infeasible* at stiff exotherm steps under the natural
hard-constraint formulation — not approximately, but algebraically: the condensed prediction
matrix has an identically zero first block-row, so five uniformity-constraint rows carry a zero
normal vector, three of them against a negative right-hand side. The SNN's apparent failure to
converge was therefore, for a large part of the trajectory, the correct response to an empty
feasible set, and no rescaling or preconditioning could have repaired it. Third, once the
problem is made feasible by an exact ℓ₁ penalty on the state rows alone, the SNN reaches the
optimum to within 5 × 10⁻⁴ °C of a reference solver while its convergence flag never fires,
because the library's projected-gradient stopping test is absolute rather than scale-invariant
and cannot trigger on a problem whose gradient scale is ~10¹⁰. Fourth — the test of that
diagnosis — replacing the criterion with a scale-invariant KKT certificate raises formal
convergence from 0 % to 51 % / 47 % / 23 % across nominal, disturbance and stiff-exotherm
operation, makes every step feasible, and reduces the maximum constraint residual by five orders
of magnitude, while leaving the closed-loop trajectory essentially unchanged (overshoot 13.24 →
13.23 °C): the applied control was already right, and what improved was its certification. At
the long horizon the new certificate still fails by a factor of 113, exactly as predicted before
it was run. We conclude that the two controllers solve the same QP, that their applied controls
agree to 0.707 °C RMS, and that convergence is now established on a substantial minority of
steps but remains weakest — 23 % — in the stiff exotherm window the controller exists to handle.

---

## 1 Introduction

Thick-sectioned polymer-matrix composites are cured in an autoclave under a prescribed
temperature profile. The governing difficulty is that the resin's polymerisation is
exothermic: past the gelation point the reaction liberates heat faster than the low-conductivity
laminate can shed it, so the interior of the part can run away thermally even while its surface
is being cooled. Standard practice uses open-loop ramp-and-hold recipes, which cannot react to
this, and which consequently trade cycle time and part quality for safety margin.

Model predictive control is the natural remedy, since it can anticipate the exotherm over a
prediction horizon and brake before the interior overshoots. Its cost is computational: MPC
requires a constrained QP solution every sampling interval. This has motivated a broad search
for QP solvers suited to specialised hardware, and in particular for solvers whose structure
maps onto parallel or event-driven substrates rather than sequential CPUs.

A line of theoretical work has established that spiking neural networks are intimately connected
to convex optimisation: the connectivity, thresholds, and time constants of a network of leaky
integrate-and-fire (LIF) neurons can be placed in correspondence with the parameters of a linear
or quadratic program, such that the network's output solves that program [2]. If this
correspondence holds in practice, an SNN is not merely a biologically-inspired analogue of a
solver — it *is* a solver, one that is natively event-driven and therefore a candidate for
low-power neuromorphic or FPGA deployment.

This report asks whether that candidacy survives contact with a hard control problem. The
question is not whether an SNN can solve *some* QP; it is whether an SNN can be substituted for
a classical solver inside a closed control loop, on a plant stiff enough to be interesting, and
produce demonstrably the same control. We take the phrase "demonstrably the same" seriously,
and much of what follows concerns the surprising difficulty of establishing it.

Our contributions are:

1. A complete simulation environment — non-linear plant twin, classical MPC baseline, and SNN-QP
   controller — in which both controllers are driven from a single shared QP construction, so
   that solver equivalence is testable rather than assumed (§2, §3.1).
2. The identification of two distinct failure modes that each invalidate a naive comparison and
   which are individually invisible in aggregate closed-loop metrics: a one-step prediction-window
   offset, and an asymmetric regularisation of the Arrhenius Jacobian (§3.2).
3. An algebraic proof that the per-step QP is genuinely infeasible through the exotherm under
   hard state constraints — later corroborated independently by the solver library itself,
   which names the same constraint row — reframing the solver's behaviour and establishing that
   no amount of preconditioning could have repaired it (§3.3).
4. A precise diagnosis of the residual non-convergence as an artefact of an absolute, rather
   than scale-invariant, stopping criterion, together with evidence that the returned solution
   is nonetheless correct (§3.4).
5. A pre-registered test of that diagnosis: the predicted consequences of adopting a
   scale-invariant certificate were written down, with the number they were derived from,
   *before* the change was made; both the predicted improvement and the predicted residual
   failure at the working horizon are confirmed (§3.5). We also report two upgrade hazards that
   left every aggregate metric looking plausible while the part failed to cure (§3.6) and the
   compute cost honestly — the SNN is ~8.9× slower than the reference solver on CPU (§3.7).

---

## 2 Methods

### 2.1 Plant model

The laminate and its tooling are discretised as a one-dimensional conduction problem with three
composite nodes and four tooling nodes. The state is

$$x = [T_{c,1}, T_{c,2}, T_{c,3},\; T_{t,1..4},\; \alpha_1, \alpha_2, \alpha_3]^\top \in \mathbb{R}^{10},$$

where $T_c$ are composite temperatures, $T_t$ tooling temperatures, and $\alpha$ the local degree
of cure. The single actuator is the autoclave air temperature $T_a$, which enters at the outer
tooling node.

Cure advances by the single-term autocatalytic Arrhenius law

$$\frac{d\alpha}{dt} = A_c \exp\!\left(-\frac{E_a}{RT}\right)\alpha^m (1-\alpha)^n, \tag{1}$$

and liberates heat into the composite energy balance in proportion to $d\alpha/dt$. The plant is
integrated by explicit finite differences at $T_E = 60$ s and is never linearised: it is the
ground truth against which both controllers are judged. All physical parameters are taken from
the reference formulation [1] and were verified term-by-term against its appendix.

Open-loop simulation reproduces the phenomenon of interest: under a static ramp-and-hold profile
the part centre exceeds 140 °C, well above the surface temperature and into thermal degradation.

### 2.2 Model predictive control formulation

At each interval both controllers linearise the plant about the current operating point
$(\bar T, \bar\alpha)$, obtaining a discrete-time model $(A_p, B_p)$. Differentiating (1)
produces the Jacobian terms

$$\frac{\partial \dot\alpha}{\partial T} = f_0 \frac{E_a}{RT^2}, \qquad
\frac{\partial \dot\alpha}{\partial \alpha} = f_0\left(\frac{m}{\alpha} - \frac{n}{1-\alpha}\right), \tag{2}$$

with $f_0$ the reaction rate at the operating point. The second expression has genuine poles at
$\alpha \to 0$ and $\alpha \to 1$; we therefore evaluate it at $\alpha$ clamped to
$[10^{-3}, 0.999]$. This regularises the *point at which the derivative is evaluated*, not the
model, and is applied identically to both controllers.

The MPC problem over a horizon $N$ penalises tracking error, control effort, and control slew,
subject to actuator limits and a spatial-uniformity constraint:

$$\min_{u} \sum_{k=0}^{N-1} \|x_k - x_{\text{ref}}\|_Q^2 + R u_k^2 + S(u_k - u_{k-1})^2 \tag{3}$$

$$\text{s.t.}\quad x_{k+1} = A_p x_k + B_p u_k, \quad
T_{a,\min} \le u_k \le T_{a,\max}, \quad
|u_k - u_{k-1}| \le \Delta_{\max}, \quad
|x_{k}^{(1)} - x_{k}^{(3)}| \le G_{\max}.$$

The final constraint bounds the centre-to-surface temperature difference and is what enforces a
uniform, inside-out cure.

### 2.3 Canonical condensed form

Eliminating the state recursion yields a dense QP in the input sequence
$z = [u_0, \ldots, u_{N-1}]^\top$:

$$\min_z \tfrac{1}{2} z^\top H z + f^\top z \quad \text{s.t.} \quad A_{\text{ineq}} z \le b_{\text{ineq}}, \tag{4}$$

with $H = 2(\Gamma^\top \bar Q \Gamma + \bar R + D^\top \bar S D)$ and
$f = 2\Gamma^\top \bar Q (\Phi x_0 - x_{\text{ref}})- 2 D^\top \bar S d_0$, where $\Phi$ and
$\Gamma$ are the free- and forced-response matrices and $D$ is the first-difference operator.
Row $i$ of $\Phi$ is $A_p^{\,i}$, predicting $x_i$; this indexing is load-bearing and is
revisited in §3.2.

**Both controllers call one implementation of (4).** The CVXPY/OSQP baseline solves it directly.
The SNN controller applies a Jacobi change of variables before solving, described next.

### 2.4 SNN-QP solver and preconditioning

The SNN solver minimises $\tfrac12 x^\top A x + b^\top x$ subject to $Cx + d \le 0$ by
alternating a gradient step with a greedy projection onto violated constraint boundaries — the
LIF analogue being that a neuron's membrane potential is integrated until it crosses threshold,
at which point a spike applies a discrete corrective displacement [2]. Constraints map to
thresholds; the constraint Gram matrix $CC^\top$ plays the role of the recurrent weight matrix.
We use the `snn_opt` implementation [5] with its compiled backend, and we classify what it
actually runs as an *approximate iterative solver* rather than a realisation of the
continuous-time theory: the theory motivates the method but does not certify this
discretisation on this problem, and the distinction is load-bearing for every claim below.

Because $H$ in (4) is stiff, the SNN adapter applies Jacobi scaling with
$D_s = \operatorname{diag}(\sqrt{\operatorname{diag} H})$:

$$\hat H = D_s^{-1} H D_s^{-1}, \quad \hat g = D_s^{-1} f, \quad z = D_s^{-1}\hat z, \tag{5}$$

with constraint rows subsequently normalised. This is an exact change of variables; the solution
is mapped back by $z^\star = \hat z^\star / D_s$ before use, and all reported residuals and
objectives are evaluated on the *original* problem (4), never on the scaled surrogate. We note
that the reference solver applies its own internal equilibration, so (5) is not an asymmetry
between the two controllers so much as an explicit statement of what the reference does
implicitly.

### 2.5 Evaluation protocol

Head-to-head numbers are generated by a single harness driving two independent plant instances
from identical initial conditions, with disturbances injected at the same point in the per-step
sequence for both branches. This matters: two separately-written simulation scripts had injected
the disturbance on opposite sides of the control computation, producing a one-step
misalignment that quietly invalidated any step-wise comparison between them.

We distinguish four questions that are frequently conflated:

- **Feasibility** — does the returned point satisfy $A_{\text{ineq}} z \le b_{\text{ineq}}$ to
  tolerance, measured on the original problem?
- **Optimality** — is its objective close to a reference solver's on the identical arrays?
- **Applied-move agreement** — does $u_0$, the only quantity reaching the plant, match?
- **Formal convergence** — did the solver's own stopping criterion fire, and if not, why?

Only the third determines closed-loop behaviour; the others explain it. We report a lower
objective attained at an infeasible point as a failure, never as a success.

---

## 3 Results

### 3.1 A first implementation, and two bugs that flattered it

An initial SNN-QP controller closed the loop and produced a superficially reasonable trajectory,
but was worse than the baseline on every metric and roughly a hundred times slower
(≈13 s per solve). Two defects were responsible.

The first was mundane: the solver was running its pure-Python reference path rather than its
compiled kernel. Switching backends reduced a single solve from ≈5300 ms to ≈60 ms with
bit-identical output.

The second was substantive. The condensed builder contained a stabilisation step that rescaled
the prediction matrix whenever it became unstable,

```
rho = max(abs(eigvals(Ap)));  if rho >= 1:  Ap *= 0.98 / rho
```

The intent was conditioning; the effect was to delete the exotherm from the prediction. During
gelation $\rho(A_p)$ genuinely exceeds 1 — the linearised model is *correctly* predicting
thermal runaway, and that prediction is exactly the signal the controller needs in order to brake
in time. With it suppressed, the controller held the autoclave at maximum temperature through
the exotherm and overshot by 20 °C against the baseline's 13 °C. Removing the rescaling restored
correct braking.

The lesson generalises beyond this codebase: an unstable prediction model is not automatically a
numerical problem to be suppressed. Here instability was the physics.

### 3.2 Establishing that two controllers solve the same problem

With those fixes the two controllers produced similar aggregate metrics, and it was tempting to
declare equivalence. Direct measurement did not support it, for two independent reasons.

**A one-step prediction window offset.** The condensed builder constructed $\Phi$ with exponent
$i+1$ rather than $i$, so its cost and constraints applied to states $x_1 \ldots x_N$ while the
baseline's applied to $x_0 \ldots x_{N-1}$. Both windows yield well-posed QPs; both produce
plausible trajectories. Reconstructing each window and solving both with the *same* reference
solver, so that only the window varied, showed agreement with the live baseline to 0.018 °C for
the correct window and disagreement of 1.20 °C for the shipped one — with the discrepancy
concentrated, predictably, where the dynamics are least linear.

**An asymmetric Jacobian clamp.** The SNN controller bounded the exotherm Jacobian terms of (2)
while the baseline did not. At a benign operating point this is negligible
($\max|\Delta A_p| \approx 10^{-4}$); at a gelation-like point it reaches $\approx 1851$. The two
controllers were therefore predicting with materially different models exactly where the
comparison mattered.

Both were removed by routing both controllers through one canonical construction (§2.3) and by
making the clamp an explicit parameter defaulting to *off* on both. Verification: building both
controllers' QPs from the same state at each of 160 trajectory steps, including the disturbance
and the full exotherm, gives a maximum difference of exactly zero across $H$, $f$,
$A_{\text{ineq}}$, $b_{\text{ineq}}$, the bounds, and the variable ordering.

We stress how weak aggregate metrics are as evidence here. Before the fixes, the two controllers
matched to within a few degrees on peak overshoot while diverging by as much as 64 °C in applied
control mid-trajectory. Summary statistics can agree while the underlying trajectories do not.

### 3.3 The stiff QP is infeasible, not merely ill-conditioned

With the comparison made fair, the SNN still exhausted its iteration budget at every step and
returned points violating the constraints by large margins. The natural hypothesis, and the one
we initially pursued, was conditioning: the condensed Hessian is assembled from weights spanning
three orders of magnitude and compounded through $\Gamma$, whose entries involve $A_p^k$.

That hypothesis is false, and we can exclude it directly. Replacing the Jacobi scaling (5) with
exact eigen-whitening, so that $\operatorname{cond}(\hat H) = 1.0$ — the best-conditioned
Hessian obtainable — changed the final feasibility by less than 3 %. A perfectly conditioned
Hessian did not help.

Sweeping 24 configurations across clamp setting, constraint form, horizon, and step size located
the real cause. **All twelve hard-constrained configurations are reported infeasible by the
reference solver**, at every horizon and under both prediction models. The mechanism is visible
in the constraint offsets: the uniformity constraint at horizon step $N-1$ carries an offset
$\propto G A_p^{N-1} x_0$, amplified by $\rho(A_p)^{N-1} \approx 4200$ at the gelation peak. The
feasible set is empty. The cold-start point is violated by $2.5\times10^5$ before a single
iteration executes.

**The infeasibility is exact, not marginal.** A solver certificate is evidence but not proof, so
we established the result algebraically. Condensation writes the forced-response matrix $\Gamma$
one block-row at a time, and block-row $0$ is never written: $x_0$ is the pinned current state,
so no future input can influence it, and the corresponding loop range is empty. The row is
therefore the exact zero vector, structurally, for every state and at every step — not small,
but identically zero. Uniformity row $k=0$ inherits it directly, and at the gelation operating
point rows $k=1\ldots4$ measure zero to machine precision as well. Rows $k=2,3,4$ pair a zero
normal with a *negative* right-hand side, so those constraints read

$$0 \le -2.25\times10^{2}, \tag{6}$$

which is false for every $z$ in $\mathbb{R}^{N}$. This is a stronger statement than "the solver
reports infeasible": it holds independently of any solver, any tolerance, and any scaling. It
also settles the whitening result of the previous paragraph and disposes of the rescaling
hypothesis entirely — rescaling multiplies the constraint normal, and zero times anything is
zero.

The same mechanism resolves a separate anomaly. Through the stiff window the solver reported
performing *no projections at all* while its scaled constraint violation stayed pinned near
$2.25\times10^{12}$ and the iterate grew by three orders of magnitude — a pattern consistent
either with a genuinely idle projector or with a broken counter. It is the former, and it is a
consequence of the same dead row: the library's projection step guards degenerate constraints
with a $\|c_j\|^2 < 10^{-12}$ test and skips them *without* incrementing its counter and without
altering the residual. The greedy selector therefore re-selects the same unfixable row on every
one of 8000 iterations. One mechanism accounts for the zero counter, the frozen violation, and
the diverging iterate simultaneously.

**Independent corroboration.** A later version of the solver library, developed with no
knowledge of this analysis, refuses to construct the hard-form problem at all, raising
*"constraint row 82 has a zero normal and $d>0$: the problem is certifiably infeasible"*. Row 82
is precisely the $k=2$ row derived above.

Introducing slack variables on the predicted-state rows only — actuator limits are real physical
bounds and remain hard — with an exact $\ell_1$ penalty [6] restores feasibility. The linear penalty term
drives slacks to zero wherever the hard constraint is attainable (observed magnitude $10^{-29}$
at a benign state), so nothing is relaxed that could have been satisfied. A solver-independent
slack linear program confirms the soft form is feasible at every horizon tested, with minimum
total slack $\le 3.9\times10^{-7}$. Applied identically to both controllers, this preserves the
parity established in §3.2. We note that softening is not a workaround chosen for convenience:
against a zero constraint normal it is the *only* repair available, because adding a slack
column is the only way to give the row a coefficient on something that can move.

Throughout the remainder of this report we name which constraint form is meant. The hard form is
infeasible and the soft form is feasible; both statements are true, and conflating them reads as
a contradiction.

### 3.4 Correct answers under a mis-specified stopping test

With a feasible problem the SNN returns feasible points, and the closed-loop numbers improve
substantially (Table 3). The formal convergence flag, however, still never fires.

The blocker is identifiable. The projected-gradient norm at the returned point is
$\approx 1.66\times10^{10}$ and is unchanged by a six-fold increase in iteration budget. The
solver tests this norm as an *absolute* quantity against a tolerance of $5\times10^{-2}$. On a
problem whose gradient scale is $10^{10}$, satisfying that test demands twelve orders of
magnitude of reduction; the criterion is not scale-invariant and cannot fire regardless of
solution quality.

To determine whether the returned point is nonetheless correct, we evaluated at a horizon short
enough that the $A_p^{N-1}$ amplification is mild, and compared against a reference solve of the
identical problem:

| Quantity | Value |
|---|---|
| Applied move $u_0$ vs. reference optimum | agree to $5\times10^{-4}$ °C |
| Relative objective gap | $-5.4\times10^{-8}$ |
| Constraint residual | $5.4\times10^{-4}$ (feasible) |
| Projected-gradient norm, *relative* to gradient scale | $9.7\times10^{-3}$ — **inside** the $5\times10^{-2}$ tolerance |
| Reported `converged` flag | `False` |

On a well-posed stiff QP the SNN reaches the optimum, and a scale-invariant form of the solver's
own criterion would have fired. The persistent negative flag is a property of the stopping test,
not evidence of a wrong answer. This is, we suggest, the most transferable finding here: absolute
tolerances on gradient norms are unsafe for condensed MPC problems, whose objective scale varies
by orders of magnitude with the operating point.

**A prediction, recorded before it was tested.** The result above is at a horizon of 5, where
the $A_p^{N-1}$ amplification is mild, and it would be an overclaim to generalise it. We
therefore measured the same *relative* projected-gradient norm at the working horizon of 20,
where it is **0.449 to 0.670** depending on the gradient step size — an order of magnitude
outside the $5\times10^{-2}$ tolerance. On that basis we recorded the prediction that a
scale-invariant test, however correctly specified, would **still not fire at the working
horizon**, and that the defensible claim would remain "the applied move is correct where we can
check it against a reference solver", never "a better tolerance converges everywhere". Section
3.5 reports the test of that prediction.

We flag one methodological point that produced a spurious disagreement in our own work. The
relative norm depends on the gradient step size and is meaningless without it; two measurements
that differ (0.449 versus 0.670) can both be correct and describe different configurations. We
now report the norm swept across step sizes so it cannot be quoted without one.

### 3.5 Acting on the diagnosis: a scale-invariant certificate

Section 3.4 attributes the persistent negative flag to the stopping test rather than to the
solution. That is a falsifiable claim, and the way to test it is to replace the test and see
which quantities move. A later release of the solver library supplies exactly the required
substitute: a scale-invariant Karush–Kuhn–Tucker cone certificate, selectable in place of the
legacy absolute projected-gradient test. We adopted it under a before-and-after regression
fingerprint that records, for each horizon and step size, the applied move, the convergence
flag, the projection counts, the residuals, and a hash of the full solution vector.

The diagnosis is confirmed, in both directions.

| | legacy absolute test | **scale-invariant KKT** |
|---|---|---|
| Formal convergence — nominal | 0.0 % | **51.3 %** |
| — disturbance | 0.0 % | **46.9 %** |
| — stiff exotherm window | 0.0 % | **22.6 %** |
| Steps feasible enough to score an objective gap | 48.1 % / 58.8 % / 45.2 % | **100 % / 100 % / 100 %** |
| Max constraint residual | 1.55 / 2.07 / 1.55 | **3.5e−5 / 1.9e−5 / 1.9e−5** |
| Applied moves corrected by the safety filter | 13.1 % / 15.0 % / 29.0 % | **3.8 % / 1.3 % / 12.9 %** |
| Closed-loop overshoot | 13.24 °C | 13.23 °C |
| Peak cure gradient $\Delta\alpha$ | 0.3418 | 0.3417 |

**Table 1.** Effect of replacing the stopping criterion, at the recommended configuration
(horizon 10, soft state constraints, identical prediction model). Triples are
nominal / disturbance / stiff-exotherm-window.

The pattern in the last two rows is the point. The certificate moved a great deal; the *control
outcome* moved by 0.01 °C. That is the signature of a previously mis-specified flag rather than
a previously wrong answer, and it is the strongest available evidence for the §3.4 diagnosis:
had the solver genuinely been returning bad points, correcting the stopping test would have
changed the trajectory, not merely the label attached to it.

**And the prediction of residual failure is also confirmed.** At the long horizon the new
certificate does not fire, and does not fail marginally: the ratio of KKT residual to tolerance
at the stiff $N=20$ step is **113**. Horizon 5 now converges properly (ratio $1.9\times10^{-7}$)
and horizon 10 is the interesting middle case (2.40). We regard reporting this as more useful
than the improvement above it: the improvement was hoped for, whereas the failure was predicted
in advance, from a number recorded before the change, and finding it exactly where it was
expected is what makes the diagnosis in §3.4 credible rather than post hoc.

**A regression that was not one.** The naive reading of the same data shows the mean objective
gap worsening, from $1.05\times10^{-4}$ to $6.72\times10^{-4}$. It has not. The legacy
configuration could only evaluate a gap on the ~half of steps where its output was feasible
enough to grade, so its mean is taken over an easier subset; the new configuration grades every
step, including the hard ones. Restricted to the *same* steps, the two are statistically
identical ($1.201\times10^{-4}$ versus $1.215\times10^{-4}$ on the nominal scenario,
$1.348\times10^{-4}$ versus $1.347\times10^{-4}$ on the disturbance scenario). Accuracy did not
change; coverage doubled. Reporting the headline figures without this check would have announced
a regression that does not exist, and we note that the check is only possible because per-step
records are retained rather than summarised.

### 3.6 Two upgrade hazards, and a rule they share with §3.8

Adopting the new certificate was not a drop-in substitution, and both of the hazards we hit
share a structure worth stating explicitly, because in each case *every aggregate metric
remained plausible while the result was wrong*.

The first is silent argument deprecation. Passing the library's now-deprecated absolute-tolerance
argument does not warn; it silently forces the legacy stopping test. Our controller passed it,
so the first post-upgrade run was still using the *old* criterion, and had we not asserted the
resolved configuration after construction we would have reported that the new certificate does
not help — the exact opposite of the true result. We now record the resolved criterion into the
provenance block of every run.

The second is a change in the meaning of an existing parameter. A projection-iteration limit that
was previously a per-call cap became a hard watchdog that *aborts* the solve. At the value
carried over from the previous version, the solver terminates after approximately one outer
iteration and returns essentially its cold start. The applied controls remain within actuator
limits, the RMS differences remain small, the timing improves, and **the part never cures**:
final degree of cure $\alpha = 0.0000$ against a required $1.0$, with the composite peaking at
41.8 °C instead of ~138 °C. Nothing in the metrics table announces this.

The rule both hazards enforce, which §3.8 arrives at from an entirely different direction, is:
**check the terminal degree of cure before believing any other number**. We now emit it as an
explicit per-scenario cure gate alongside the metrics rather than leaving it to a reader's
diligence.

### 3.7 Compute time

The SNN is slower than the reference solver, and we report this plainly because an earlier
draft of this work did not. Both controllers are timed in the same process, on the same per-step
sequence, at the recommended configuration.

| Median total ms/step, nominal | legacy configuration | **scale-invariant KKT** |
|---|---|---|
| Reference (CVXPY / OSQP) | 5.76 | 5.47 |
| SNN-QP | 113.75 | **48.66** |
| **Ratio** | **19.8×** | **8.9×** |

**Table 2.** Wall-clock comparison, single process, horizon 10, soft state constraints.

Under the legacy criterion the solver never terminated early, exhausting its full 8000-iteration
budget at a near-constant ~114 ms per step. The scale-invariant certificate lets roughly half of
nominal steps stop early, approximately halving the median; inside the stiff window it falls
further, to 27.2 ms (5.0× the reference), because there the projection watchdog terminates the
hardest solves sooner. QP *construction* costs the two sides the same (0.73 versus 0.74 ms); the
entire gap is in the solve.

Two caveats. First, timing is the only quantity in this report that is not bit-reproducible:
repeating the identical run moves the absolute medians by a few percent with machine load, while
the ratio holds. Read the ratio. Second, an earlier claim in this project that compute time was
"comparable to" the reference was drawn from a superseded long-horizon, hard-constraint harness
in which the reference solver is itself slow (116 ms/step), flattering the ratio to ~1.6×. That
claim did not hold at the configuration actually recommended and has been withdrawn. No claim of
computational advantage is made anywhere in this report; the architectural case for the SNN
rests on event-driven hardware, which this work does not evaluate.

### 3.8 Closed-loop results, and a degenerate configuration

Table 3 gives the controlled comparison at the settled configuration.

**Table 3.** Controlled head-to-head, horizon 10, soft state constraints, identical prediction
model, scale-invariant certificate. Trajectory difference is the Euclidean norm over composite
temperatures and cure states.

| Metric | Nominal | Disturbance | Exotherm window |
|---|---|---|---|
| RMS applied-control difference | 0.707 °C | 0.793 °C | 0.565 °C |
| RMS trajectory difference | 0.251 | 0.286 | 0.418 |
| Max applied-control difference | 3.52 °C | 3.95 °C | 0.66 °C |
| SNN max constraint residual | $3.5\times10^{-5}$ | $1.9\times10^{-5}$ | $1.9\times10^{-5}$ |
| SNN formal convergence rate | 51.3 % | 46.9 % | 22.6 % |
| Steps feasible enough to score objective gap | 100 % | 100 % | 100 % |
| Mean objective gap on feasible steps | $6.7\times10^{-4}$ | $2.7\times10^{-3}$ | $2.6\times10^{-3}$ |
| Applied moves corrected by safety filter | 3.8 % | 1.3 % | 12.9 % |
| Final degree of cure (cure gate) | 0.9999 | 0.9998 | — |

Relative to the pre-correction configuration, RMS applied-control difference falls from 16.005 °C
to 0.707 °C and the maximum constraint residual from $1.85\times10^{5}$ to $3.5\times10^{-5}$ —
ten orders of magnitude. Both controllers achieve full uniform cure with comparable overshoot
(13.77 °C baseline, 13.23 °C SNN) and identical actuator-limit compliance.

The objective-gap row is not comparable to the corresponding row of the legacy configuration;
see the like-for-like restriction in §3.5. The convergence row is the honest weak point of the
study, and its shape matters more than its average: the stiff exotherm window, the single regime
that motivates predictive control of this process at all, is where the certificate fires least
often and where the safety filter intervenes most.

**A cautionary configuration.** At horizon 5 the comparison appears flawless: RMS applied-control
difference 0.000 °C, formal convergence 98.8 %, zero clipping, zero constraint violations. It is
also useless. A five-minute horizon cannot see past the thermal transport lag, so the effort term
dominates and the controller drives the actuator to its lower bound. In the nominal scenario the
part cools from 28 °C to 11.4 °C; in the disturbance scenario, to 1.85 °C. The degree of cure
reaches at most $1.0\times10^{-5}$ — below reporting precision, though we note that it is not
identically zero, and an earlier draft of this report said "never departs from zero", which the
saved trace does not support. Both controllers agree perfectly because both are inert, saturated
against the same bound. We report this because it is the precise failure mode that an
equivalence study is most likely to mistake for success, and because it is invisible in every
metric except the cure state itself — the same rule §3.6 reaches from the direction of a
dependency upgrade. The same caution applies more weakly during heat-up, where 41.7 % of the
first 60 steps — and 33.1 % of all 160, and 77.4 % inside the stiff window — have both
controllers pinned at the slew limit; agreement on those steps reflects a shared actuator bound
and is excluded from our claims.

---

## 4 Discussion

We set out to test whether a spiking network can be substituted for a classical QP solver inside
an MPC loop. Our answer is qualified in a specific way: the two controllers demonstrably receive
the same problem, their applied controls agree to 0.707 °C RMS, every step of the trajectory is
now feasible with a maximum constraint residual of $3.5\times10^{-5}$, and formal convergence —
previously unattainable at any horizon that cures the part — is now certified on 51 % of nominal
steps. But convergence is **22.6 %** in the stiff exotherm window, 12.9 % of applied moves there
still originate from a downstream safety filter rather than a certified solver output, and at
the long horizon the certificate fails by a factor of 113. We therefore state the result as
*same problem, feasible everywhere, convergence established on a substantial minority of steps
and weakest exactly where the controller matters*, and continue to decline to call it
equivalence. We use no comparative term — "equivalent", "on par", "comparable" — anywhere in
this report without an accompanying numerical tolerance.

It is worth being precise about what improved and what did not, because the two are easily
conflated. What improved is the solver's *certification* of its own output: the flag, the
residual, the feasibility coverage, and the frequency with which a downstream filter has to
intervene. What did not measurably change is the *control*: overshoot moved by 0.01 °C and the
peak cure gradient by 0.0001. A reader is entitled to ask whether the work therefore changed
anything. It did, in the sense that matters for deployment — before, a correct applied move
could not be distinguished from an incorrect one by any signal the solver emitted; now it can,
on about half of steps. An uncertified correct answer and an uncertified wrong answer look
identical from outside, and closing that gap is the prerequisite for trusting the loop rather
than merely observing that it happens to work.

Four observations seem worth carrying forward.

**Equivalence is an experimental claim with a high evidentiary bar.** Two of the defects we found
— the prediction-window offset and the asymmetric Jacobian clamp — produced closed-loop
trajectories that looked entirely reasonable and aggregate metrics that were close. Neither is
detectable without constructing both QPs and comparing them numerically. We would suggest that
any claim of solver substitutability be accompanied by a per-step array-level comparison, not a
metrics table.

**Infeasibility masquerades as non-convergence.** For a substantial portion of the trajectory the
problem we posed had no solution, and every diagnostic we ran on the solver was, in retrospect,
measuring the consequences of that. The signature — iteration budget exhausted, large residuals,
lower-than-optimal objective at an infeasible point — is indistinguishable from a struggling
solver until a reference solver is asked the same question. Condensed MPC formulations with
state constraints and unstable prediction models appear especially prone to this, since the
constraint offsets inherit $\rho(A_p)^{N-1}$ amplification while the constraint gradients do not.

**Stopping criteria must be scale-invariant.** The solver studied here reached the optimum and
reported failure, because its convergence test compared an absolute gradient norm against a
fixed tolerance on a problem whose natural scale is $10^{10}$. This is a small implementation
detail with a large consequence, and it will recur in any application where the objective scale
varies with the operating point — which is to say, in most non-linear MPC. Replacing it was the
single highest-value change in this project. But scale-invariance is necessary, not sufficient:
the corrected certificate still does not fire at the working horizon, so the underlying
iteration, not only its stopping rule, remains a limitation.

**A metric that looks plausible is not a metric that is correct.** Four separate failure modes
in this study — a degenerate short horizon, a deprecated argument silently selecting the old
criterion, a repurposed iteration limit aborting the solve, and an objective gap that appeared
to regress — each left a table of aggregate numbers that a reader would accept. Three of the
four were caught only by checking a quantity outside the table (the terminal degree of cure, the
resolved solver configuration, the set of steps a mean was taken over). We would suggest that
studies of this kind carry an explicit, machine-checked gate on a physical outcome, distinct
from the metrics being reported, and that per-step records be retained so that a suspicious
aggregate can be re-partitioned after the fact rather than re-measured.

**Limitations.** The horizon reduction that made the problem tractable was selected empirically
rather than from the plant's thermal time constants. Softening the uniformity constraint weakens
its guarantee, though the hard form it replaces is provably infeasible and therefore
unimplementable; we mitigate this by reporting slack magnitudes and by verifying that the
solution is insensitive to the penalty weight across two orders of magnitude. Our strongest
evidence that the returned solution is optimal comes from a horizon too short to control the
plant; at the working horizon we establish that the applied move matches the reference optimum
to $1.5\times10^{-6}$ °C but the full-horizon vector does not, and we do not claim otherwise.
Formal convergence remains absent on roughly half of nominal steps and three-quarters of stiff
ones. The residual clipping rate of 12.9 % in the stiff window is the single largest outstanding
objection to any equivalence claim. Compute time is ~8.9× the reference on CPU. Finally, all
results are in simulation against a digital twin of a single laminate geometry; no physical
autoclave was involved.

**Future work.** The remaining problem is now specific rather than diffuse: it is the stiff
exotherm window. Three candidate directions follow from the analysis above. A horizon or
terminal-set redesign that makes the hard uniformity constraint attainable would remove the need
for slacks altogether and is the most principled of the three, since it addresses the zero-normal
mechanism of §3.3 at its source. Step-size adaptation through the exotherm targets the
observation that the relative projected-gradient norm at the working horizon depends strongly on
the step size, which was tuned against the superseded criterion and is very likely no longer
optimal. A gelation-specific warm start would attack the same steps from the initialisation
side. Separately, it remains to be confirmed whether the saturating-projection pathology is
genuinely repaired by the new watchdog or merely made visible by it.

Hardware implementation should continue to wait. Committing a solver to fixed hardware while its
convergence certificate fires on under a quarter of steps in the regime the controller exists
for, and while an eighth of the applied moves in that regime come from a safety filter, would
freeze in the solver's behaviour precisely where it is least certified, in a substrate where
diagnosis is far more expensive. A reasonable gate would be stiff-window convergence comfortably
above 90 % with clipping in the low single digits, on a configuration that still cures the part.
The architectural argument for event-driven hardware is unaffected by any of this and remains
the motivation for the work — on CPU the SNN is strictly worse than a mature classical solver,
and it is only on a substrate that exploits its parallelism that the comparison becomes
interesting. That is exactly why the hardware phase should test the SNN's real value
proposition rather than inherit a known weakness.

---

## Broader impact

The nearer-term impact of work in this direction is on process efficiency: closed-loop cure
control shortens cycle times and reduces scrap in composite manufacturing, with corresponding
reductions in energy and material waste relative to conservative open-loop recipes. A
longer-term impact concerns where such controllers can run — solvers that map onto low-power
event-driven hardware make predictive control viable on embedded and edge platforms that cannot
support a conventional optimisation stack.

Two cautions attach. First, a controller that appears to track a reference correctly while its
underlying solver has not converged is a genuine safety concern in any process where constraint
satisfaction protects the product or the equipment, and the present work is largely an
illustration of how easily that state can be mistaken for success. We would argue that
convergence and feasibility should be logged and monitored in deployment, not merely checked in
simulation. Second, the same argument that makes neuromorphic solvers attractive for industrial
edge control makes them attractive for autonomous systems generally, including applications
whose desirability is less clear; that trade-off is not specific to this work but is not absent
from it either.

---

## Reproducibility

Every quantitative claim in this report is backed by a file in the repository's `results/`
directory, and `results/artifact-index.md` maps each reported figure to that file and to the
script which regenerates it. No number here exists only in prose or in console output. The
per-step records underlying every aggregate are retained as CSV, which is what made the
like-for-like objective-gap check of §3.5 possible after the fact.

Runs record their own provenance: the git commit and working-tree state, the versions of the
solver stack, whether the compiled backend was actually loaded, the resolved convergence
criterion, the full shared configuration applied to both controllers, and the terminal degree of
cure. Both controllers are constructed from a single configuration dictionary, so a
configuration cannot be applied to one side and not the other. Dependency changes are gated by a
before-and-after fingerprint capturing, per horizon and step size, the applied move, the
convergence flag, projection counts, residuals, and a hash of the solution vector.

All results are deterministic given the configuration: there is no random number generator
anywhere in the plant or either controller. The single exception is wall-clock timing, which
varies with machine load; §3.7 reports the ratio for that reason.

---

## References

[1] P. Dufour, D. J. Michaud, Y. Touré, and P. S. Dhurjati. A partial differential equation model
predictive control strategy: application to autoclave composite processing. *Computers &
Chemical Engineering*, 28(4):545–556, 2004.

[2] A. Mancoo, S. W. Keemink, and C. K. Machens. Understanding spiking networks through convex
optimization. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 2020.

[3] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd. OSQP: an operator splitting
solver for quadratic programs. *Mathematical Programming Computation*, 12:637–672, 2020.

[4] S. Diamond and S. Boyd. CVXPY: a Python-embedded modeling language for convex optimization.
*Journal of Machine Learning Research*, 17(83):1–5, 2016.

[5] A. H. Khan. `snn_opt`: a spiking-neural-network solver for quadratic programs. Software
package, versions 0.4.0 and 0.6.0. <https://github.com/ahkhan03/SNN_opt>

[6] E. C. Kerrigan and J. M. Maciejowski. Soft constraints and exact penalty functions in model
predictive control. *Proceedings of the UKACC International Conference (Control)*, 2000.
