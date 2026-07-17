# SNN-QP Solver Parity — Engineering Report

**Branch:** `fix/snn-qp-solver-parity`
**Scope:** make the neuromorphic SNN-QP controller a rigorous, fair drop-in for the CVXPY/OSQP MPC baseline, verify the plant model against the reference paper, and add tooling to read the reference PDFs.

This report is written to be read start-to-finish by someone (e.g. the project advisor) who has *not* seen the debugging session. It covers, in order:

1. [What was asked vs. what this branch delivers](#1-what-was-asked-vs-what-this-branch-delivers)
2. [How each issue was solved, and the sources cited](#2-how-each-issue-was-solved-and-the-sources-cited)
3. [Background needed to understand the work](#3-background-needed-to-understand-the-work)
4. [Every problem in the previous (`main`) implementation](#4-every-problem-in-the-previous-main-implementation)
5. [Every code change and the reasoning](#5-every-code-change-and-the-reasoning)
6. [How to reproduce](#6-how-to-reproduce)

---

## 1. What was asked vs. what this branch delivers

### 1.1 The advisor's directive
The research question fixed early in the project was:

> *"Can an SNN-based MPC controller **match or outperform** conventional controllers while offering advantages in speed and energy efficiency?"*

The agreed plan built to it in phases. The conventional baseline was specified first, with an explicit rule:

> *"First get the conventional baseline working first so the comparison is rigorous when we add SNN-QP."*
> *"Get the baseline metrics (max temp overshoot, cure uniformity, constraint violations, solver time per step)."*

The Phase-4 task was then to map each per-step MPC-QP onto the SNN solver, run it closed-loop against **the same** non-linear digital twin, and compare on those same metrics. Implicit in the research question: the SNN-QP must (a) **reproduce the baseline's control quality** — because it solves the *same* QP — and (b) be **competitive in compute**.

### 1.2 What `main` actually had
The SNN was implemented and the repo was marked "Phase 4 complete", but the comparison it produced was **not** the one the directive asked for:

| Metric | CVXPY/OSQP baseline | SNN on `main` | Verdict |
|---|---|---|---|
| Max temp overshoot | 13.12 °C | **20.04 °C** | worse |
| Peak cure gradient (Δα) | 0.3579 | 0.3852 | worse |
| Constraint violations | 1 | 2 | worse |
| Solver time / step | ~124 ms | **~13,074 ms** | ~100× slower |

The SNN was worse on every quality axis *and* two orders of magnitude slower — so it did not demonstrate "match or outperform." The README explained the slowness away as an inevitable CPU cost and framed the worse control as acceptable. That framing masked two concrete, fixable bugs.

### 1.3 What this branch delivers
After fixing those bugs, the SNN-QP reproduces the baseline's control from the *identical* QP:

| Metric | CVXPY/OSQP baseline | SNN (this branch) | Verdict |
|---|---|---|---|
| Max temp overshoot | 13.12 °C | **10.82 °C** | on par (slightly better) |
| Peak cure gradient (Δα) | 0.3579 | **0.3289** | on par (slightly better) |
| Constraint violations | 1 | 2 | ~equal (one extra transient touch) |
| Solver time / step | ~124 ms | **~186 ms** | comparable on CPU (was ~100× slower) |

Deliverables on the branch:
- **Solver fix** so the SNN-QP matches the baseline (`src/snn_mpc_controller.py`).
- **Head-to-head artifact**: `assets/snn_vs_cvxpy_overlay.png` + a comparison table in the README; regenerated `assets/snn_closedloop_test.png`.
- **Model verification**: every physical constant cross-checked against the reference paper, with fidelity + limitation notes (`ref_docs/Mathematical_formulation.md`).
- **Tooling**: `tools/pdf_to_md.py`, a pip-only OCR PDF→Markdown converter for reading the reference PDFs.

---

## 2. How each issue was solved, and the sources cited

### 2.1 Diagnosis-first method
Nothing was changed until the failure was reproduced and isolated. The chain was:

1. Read the installed `snn_opt` solver source to learn its true contract: it minimises `½ xᵀAx + bᵀx` s.t. `Cx + d ≤ 0` via **projected gradient descent + greedy boundary projection** (LIF-style dynamics), and it ships a **compiled C kernel** as well as the default pure-Python one.
2. **Isolated the solver from the loop**: solved a single representative per-step QP with both the SNN and a reference solver (OSQP on the identical `H, g, C, d`). This separated "does the SNN solve the QP correctly" from "is the QP itself right".
3. **Isolated the formulation**: compared, step by step, the control produced by the CVXPY controller, the `main` SNN, and an SNN variant with one line removed — on the same plant states. This pinpointed the exotherm-braking failure to a single line.

### 2.2 The speed fix
The controller used `snn_opt`'s **pure-Python reference backend**. The package also ships a numerically-identical **compiled kernel** (`backend='c'`). Switching to it cut a single solve from ~5,300 ms to ~60 ms (~85×) with bit-for-bit the same result, bringing the SNN to ~186 ms/step in the loop — comparable to OSQP.

### 2.3 The quality fix (the real bug)
`build_dense_qp` shrank the linearised plant matrix whenever the exotherm made it unstable:
```python
rho = np.max(np.abs(np.linalg.eigvals(Ap)))
if rho >= 1.0:
    Ap = Ap * (0.98 / rho)   # <-- erases the exotherm from the prediction
```
Tracing the control minute-by-minute showed the consequence directly: during gelation (minutes ~90–104, where `rho(Ap)` climbs to ≈1.55) the baseline **brakes** the autoclave to 126 °C, but the shrunk-model SNN stayed at **130 °C** — it never saw the exotherm coming, so it did not brake, and overshot. Removing the shrink (and keeping the existing Jacobi preconditioner to control conditioning) restores correct braking and drops overshoot from 20.0 °C to 10.8 °C.

### 2.4 Kinetics verification
Because the advisor had flagged the Arrhenius model, every constant in `src/constants.py` was cross-checked against the reference paper's Appendix. **All match exactly**, and the single-term model and the `α = 0⁺` cure seed are both faithful to the paper (details in §3.3). Conclusion: the plant is a correct reproduction; the fragility lives in the *linearisation*, not the model.

### 2.5 Sources cited
- **P. Dufour, D.J. Michaud, Y. Touré, P.S. Dhurjati, "A partial differential equation model predictive control strategy: Application to autoclave composite processing", *Computers & Chemical Engineering* 28(4):545–556, 2004.** — the source of the plant PDE, the cure kinetics, and every physical constant. (Its kinetics in turn come from Bogetti and Pillai et al., as the paper states.) DOI: 10.1016/j.compchemeng.2003.08.007.
- **M.A. Mancoo, S. Boerlin, C.K. Machens, "Understanding spiking networks through convex optimization", *NeurIPS* 2020.** — the theoretical basis of the `snn_opt` solver (its projected-gradient-with-boundary-projection dynamics). This is the reference for the *convergence* behaviour discussed in §3.4.
- **`snn_opt` package source** (v0.4.0) — solver contract, the `backend='c'` compiled kernel, and the adaptive projection method.

---

## 3. Background needed to understand the work

### 3.1 The physical system
A thick composite laminate cures inside an autoclave. The only actuator is the **autoclave air temperature `Ta`**. Heat diffuses inward through a steel tooling layer, then through the composite. The resin's polymerisation is **exothermic**: past a point it releases heat faster than the low-conductivity composite can shed it, so the **centre** can run away to 140 °C+ even though the surface is cooler. Good control must anticipate this and *brake* (drop `Ta`) before the centre overshoots, while keeping the centre-to-surface gradient small for an "inside-out" cure.

State vector (10 values): 3 composite node temperatures, 4 tooling node temperatures, 3 cure states `α`. Grid and constants are Dufour's (verified — §3.3).

### 3.2 MPC as a QP, and why the SNN needs it
At each minute the controller linearises the plant about the current operating point, forms a **condensed** QP over an `N=20` horizon (state eliminated, decision variable = the `Ta` sequence), and applies only the first move (receding horizon). The baseline solves that QP with **OSQP**; the SNN controller solves the **identical** QP with `snn_opt`. Because it is the same QP, at the optimum the two must give the same control — that equivalence is the entire point of the comparison.

### 3.3 Kinetics fidelity (why we did *not* change the model)
Dufour's cure rate is the **single-term autocatalytic** law:
```
dα/dt = A_c · exp(−E_a / R T) · α^m · (1−α)^n
```
Cross-check of `constants.py` against the paper's Appendix — all exact:

| Quantity | Paper | `constants.py` |
|---|---|---|
| `k_c` conductivity | 0.23793 | `KC=0.23793` |
| density | 1890 | `RHO_C=1890` |
| `c_pc` | 1.134×10³ | `CPC=1134` |
| `β_t` | 0.3·β_c | `0.3*BETA_C` |
| `m_r` resin mass ratio | 46 % | `MR=0.46` |
| `−ΔH_r` | 8.525×10⁴ | `DH=85250` |
| `A_c` | 1.233×10²¹ | `AC=1.233e21` |
| `E_a` | 1.674×10⁵ | `EA=167400` |
| `m`, `n` | 0.524, 1.476 | `0.524, 1.476` |
| nodes | 3 composite / 4 tooling | `NZ_C=3, NZ_T=4` |
| `Ta`, rate, gradient bounds | 10–130 °C, ±4 °C/min, ±10 °C | match |

Two things this settles:
- The model is single-term (there is **no** two-term Kamal–Sourour model in Dufour), so "upgrading" to two-term would move *away* from the reference. We kept the model.
- The `alpha = 1e-5` seed in the plant is **not a hack** — Dufour's initial condition is literally `α(z,0) = 0⁺`. The single-term rate is exactly 0 at `α=0` (the `α^m` factor), so a positive seed is intrinsic to the model.

The genuine fragility is in the **QP linearisation**: the cure Jacobian `J_α = f₀·(m/α − n/(1−α))` is singular as `α→0` and `α→1`, which is why the controllers clamp `α` to `[1e-3, 0.999]` before building `Ap`. Dufour never hits this because it optimises the *non-linear* model directly (Levenberg–Marquardt); forcing the kinetics into a QP is our adaptation, and the clamps are its cost.

### 3.4 Convergence: output vs. formal criterion
An important nuance, measured per step against OSQP:
- **Output converges:** the applied first move matches OSQP to `|Δu₀| ≈ 0.000 °C` at every well-posed step. This is why the closed-loop metrics match.
- **The solver's formal `converged` flag stays False:** it runs its full iteration budget rather than early-stopping, because the projected-gradient tolerance is not met — the iterate *chatters* along the coupled slew/gradient active set (projecting onto one constraint nudges a neighbour). This limit-cycle is intrinsic to greedy boundary projection and is exactly the kind of problem the Mancoo et al. line of work studies.
- **Peak-exotherm step:** with `rho(Ap) ≈ 1.55` over `N=20`, the condensed QP amplifies by `1.55²⁰ ≈ 4400×` and becomes numerically extreme; the raw iterate diverges there (**OSQP also fails** on the same scaled instance) and the physically-correct move is recovered by the slew-rate clamp. Shortening the horizon or adding a step-size/trust-region guard for the high-`rho` regime is the natural follow-up if a *converged* solution is required at that instant.

---

## 4. Every problem in the previous (`main`) implementation

1. **Wrong solver backend (performance).** Used `snn_opt`'s pure-Python reference path → ~13 s/step. The compiled `backend='c'` kernel (identical math) was available and unused.
2. **Model-shrinking bug (correctness).** `build_dense_qp` scaled `Ap` by `0.98/rho` when `rho ≥ 1`, deleting the exotherm from the prediction. The controller therefore failed to brake during gelation and overshot (20 °C vs 13 °C). This was the dominant quality defect and was **independent** of solver accuracy — the applied `u₀` was solved correctly for the *wrong* (shrunk) QP.
3. **Misleading documentation.** The README presented the SNN as "Phase 4 complete" while its metrics were worse than the baseline on every axis, explained the ~13 s/step as an expected CPU cost, and framed the extra overshoot/violation as fine. It did not disclose that the SNN was not, in fact, matching the baseline.
4. **Undisclosed non-convergence.** The solver never met its formal convergence criteria (it always ran to the iteration cap and saturated its projection budget — "chattering"). On `main` this was described only as evidence of "strict constraint adherence"; it was not characterised as non-convergence.
5. **Unguarded numerical blow-up at the exotherm.** At the peak-exotherm step the condensed QP is numerically extreme; the raw solve diverges and only the downstream slew clamp yields a sane move. `main` neither surfaced nor documented this.
6. **Kinetics left ambiguous.** The advisor had flagged the Arrhenius model; `main` carried the single-term model with `α=1e-5` seeding and multiple Jacobian clamps but no note on whether this matched the reference or why the clamps exist. (Verified faithful in this branch — §3.3.)
7. **Reference PDFs unreadable.** `ref_docs/dufour mpc.pdf` uses an embedded font with no Unicode map, so ordinary text extraction yields garbage — making it hard to check the model against the source. No tooling addressed this.

---

## 5. Every code change and the reasoning

### 5.1 `src/snn_mpc_controller.py` — solver configuration
**Before:**
```python
conv_config = ConvergenceConfig(
    enable_early_stopping=True, check_every=50, min_iterations=100,
    patience=3, obj_rel_tol=1e-5, proj_grad_tol=1e-2, feasibility_tol=1e-2,
)
self.solver_config = SolverConfig(
    k0=None, k0_scale=0.8, projection_method='adaptive',
    max_iterations=5000, convergence=conv_config,
)
```
**After:**
```python
conv_config = ConvergenceConfig(
    enable_early_stopping=True, check_every=50, min_iterations=100,
    patience=3, obj_rel_tol=1e-7, proj_grad_tol=5e-2, feasibility_tol=1e-2,
)
self.solver_config = SolverConfig(
    k0=None, k0_scale=0.5, projection_method='adaptive',
    max_iterations=8000, max_projection_iters=200,
    backend='c', convergence=conv_config,
)
```
Reasoning, field by field:
- **`backend='c'`** — use the compiled kernel: ~85× faster, numerically identical. This is the single biggest speed lever and the reason the SNN is now compute-comparable to OSQP.
- **`max_projection_iters=200`** (was the default 100) — the per-step QP has tightly coupled slew + gradient constraints; 100 greedy projections left visible residual infeasibility in the horizon plan. 200 halves it. (Closed-loop applied control is unchanged, but the internal solution is cleaner.)
- **`max_iterations=8000`** (was 5000) — headroom for the stiffest steps; cheap now that each iteration is compiled.
- **`k0_scale=0.5`** (was 0.8) — a more conservative gradient step. With the true (un-shrunk) dynamics the Hessian's dynamic range is larger, so a smaller step is safer.
- **`obj_rel_tol=1e-7`** (was 1e-5) and **`proj_grad_tol=5e-2`** (was 1e-2) — tuned so the plateau/gradient early-stop is neither falsely triggered nor pointlessly strict; documented honestly in §3.4, the strict gradient criterion still isn't met on the stiff QPs.

### 5.2 `src/snn_mpc_controller.py` — remove the model shrink
**Before:**
```python
def build_dense_qp(self, Ap, Bp, x0, u_prev):
    rho = np.max(np.abs(np.linalg.eigvals(Ap)))
    if rho >= 1.0:
        Ap = Ap * (0.98 / rho)
    Phi = np.zeros((self.N * self.nx, self.nx))
    ...
```
**After:**
```python
def build_dense_qp(self, Ap, Bp, x0, u_prev):
    # NOTE: the linearized Ap is intentionally used as-is. During the
    # exothermic gelation phase its spectral radius exceeds 1 (the local
    # linear model genuinely predicts thermal runaway -- which is precisely
    # what the controller must anticipate to brake in time). Shrinking Ap to
    # rho<1 erases the exotherm from the prediction and makes the SNN fail to
    # brake, so we keep the true dynamics and rely on Jacobi preconditioning
    # (_condition) to keep the condensed QP well-conditioned.
    Phi = np.zeros((self.N * self.nx, self.nx))
    ...
```
Reasoning: `rho(Ap) > 1` during gelation is **physically meaningful** — the local linear model is telling the controller "this will run away." Suppressing it removes the very signal the controller needs. Conditioning is instead handled where it belongs, in `_condition` (Jacobi preconditioning of the assembled Hessian), which is a change of variables and does **not** distort the dynamics. Removing the shrink also deletes a per-step `eigvals` call (small speed bonus).

> Everything else in `build_dense_qp` (the `Phi/Gamma` condensation, `Q/R/S` weighting, and the box/slew/gradient constraint rows) is unchanged and was already correct.

### 5.3 `README.md`
- Replaced the Phase-4 "SNN Metrics" numbers with the real ones (overshoot 10.82 °C, ~186 ms/step) and added a **head-to-head table** plus the new overlay figure.
- Rewrote the Phase-4 roadmap bullets and the Phase-4 figure caption, which previously described chattering/13 s-per-step as expected behaviour, to state the parity result and the numerically-honest framing.
- Corrected the baseline "compute time" note to reflect that it *is* now a comparison target.
- Documented the new `tools/` directory and how to use `pdf_to_md.py`.

### 5.4 `ref_docs/Mathematical_formulation.md` — new "Model Fidelity and Known Limitations" section
Adds, with no change to the maths already there:
- **5.1** the exact parameter cross-check against Dufour's Appendix;
- **5.2** why the `0⁺` cure seed is prescribed by the model, not a workaround;
- **5.3** the linearisation caveat (singular cure Jacobian → the `α` clamps);
- **5.4** the constant-vs-funnel gradient-bound difference from Dufour;
- **5.5** the output-vs-formal convergence distinction (§3.4).

### 5.5 `assets/`
- **`snn_closedloop_test.png`** regenerated from the fixed controller.
- **`snn_vs_cvxpy_overlay.png`** (new) — both controllers on an identical plant, disturbance, and QP, showing the trajectories overlap through heat-up and disturbance rejection and differ only in the post-exotherm recovery.

### 5.6 `tools/pdf_to_md.py` + `tools/requirements-pdf.txt` (new)
A small CLI that converts reference PDFs to Markdown: it extracts **native text** where the PDF allows it, and automatically falls back to **OCR** (RapidOCR / ONNX) for scanned or broken-font pages such as `dufour mpc.pdf`. Design points:
- **Pip-only, CPU-only** — no Tesseract/Poppler system binaries and no PyTorch. Rendering uses PyMuPDF; OCR uses `rapidocr-onnxruntime`.
- A `printable_ratio` heuristic decides per page whether native text is usable (≈0.9 for a clean page, ≈0.45 for the broken-font Dufour pages) and OCRs only when needed.
- `--pages`, `--dpi`, `--force-ocr`, `--ocr-threshold` flags; graceful messages if the optional deps are missing.
- Throughput ≈ 6 s/page on a laptop CPU (≈4 min for a 38-page paper). **Caveat:** OCR reconstructs body text and tables reliably but only *approximates* mathematical notation — verify equations against the source.

---

## 6. How to reproduce

```bash
# baseline + SNN closed loops (regenerate metrics and figures)
python tests/test_open_loop_baseline.py     # exotherm runaway sanity (~140 C peak)
python tests/test_closed_loop.py            # CVXPY/OSQP baseline metrics
python tests/test_snn_closed_loop.py        # SNN-QP metrics (overshoot 10.82 C, ~186 ms/step)

# read a reference PDF as Markdown
pip install -r tools/requirements-pdf.txt
python tools/pdf_to_md.py "ref_docs/dufour mpc.pdf"
```

**Environment note:** the SNN path requires the `snn_opt` package with its compiled `_kernel` extension present (it ships in the wheel used here). If only the pure-Python backend is available, change `backend='c'` back to `backend='python'` in `SNNMPCSolver.__init__` — the result is identical, just ~85× slower.
