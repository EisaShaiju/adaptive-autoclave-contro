# Adaptive Process Control for Autoclave Composite Curing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/Optimization-CVXPY-orange)
![Control](https://img.shields.io/badge/Control-MPC-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview
This repository contains the simulation environment and control architecture for optimizing the autoclave curing process of thick-sectioned composite laminates. The project replaces unmanaged open-loop curing cycles with real-time closed-loop Model Predictive Control (MPC), and asks whether an event-driven Spiking Neural Network (SNN) solver can close the *same* MPC loop.

> **Current status of that question.** The two controllers now provably receive the **same per-step QP** (bit-identical arrays from one shared builder, identical prediction model). On the closed loop their applied controls agree to **0.714 °C RMS**. But the SNN's formal convergence criterion is met on **0 %** of steps at any horizon that actually cures the part, and ~13 % of applied moves still come from a downstream safety clip. The honest verdict is therefore **"same QP, but the SNN-QP does not reliably converge"** — *not* equivalence. Full evidence, definitions, and limitations: **[`docs/PHASE4_VALIDATION_REPORT.md`](docs/PHASE4_VALIDATION_REPORT.md) (Revision 2)**.

The core physical plant is modeled based on the highly non-linear Arrhenius curing kinetics and 1D spatial heat transfer dynamics detailed in Dufour et al. (2004).

## Project Roadmap

This research is divided into four main phases:

- [x] **Phase 1: Mathematical Formulation**
  - Discrete-time linearized state-space modeling per cure phase (Heat-up, Dwell, Cool-down).
  - Derivation of Fourier numbers and linearized Arrhenius Jacobians.
  - Formulation of the MPC Quadratic Programming (QP) objective, penalizing tracking error, control effort, and overall thermal energy consumption.
- [x] **Phase 2: Open-Loop Plant Digital Twin**
  - Implementation of the Explicit Finite Difference Method in Python.
  - Solving the raw, non-linear Partial Differential Equations (PDEs) to simulate composite curing.
  - Verified exothermic thermal runaway (140°C+ spikes) when subjected to standard open-loop heating profiles.
- [x] **Phase 3: Closed-Loop CVXPY Baseline**
  - Integration of an active-set/interior-point QP solver (CVXPY) into the simulation loop.
  - Enforcement of real-time spatial constraints (10°C max gradient) to prevent residual thermal stress.
  - Verified active cooling ("Thermal Braking") during the exothermic gelation phase; disturbance rejection tested via a 15°C step perturbation injected directly into the plant state at step 60.
- [x] **Phase 4: Neuromorphic SNN Integration**
  - Mapped each per-step MPC QP to a spiking LIF dynamical system (gradient descent + discrete boundary projections), with Jacobi preconditioning of the condensed Hessian and an auto-computed $k_0$ step size.
  - Ran on the compiled `snn_opt` kernel (`backend='c'`): ~115 ms/step, comparable to the OSQP baseline on CPU and ~85× faster than the pure-Python reference path.
  - Established that the condensed QP must retain the *true* linearized dynamics through the exotherm (spectral radius > 1) — artificially stabilizing the prediction erases the exotherm and defeats the brake.
- [x] **Phase 4b: Validation pass** (see [`docs/PHASE4_VALIDATION_REPORT.md`](docs/PHASE4_VALIDATION_REPORT.md))
  - Unified both controllers onto **one canonical QP builder** (`src/qp_builder.py`) — bit-identical `H, f, A_ineq, b_ineq` given identical inputs — and made the prediction model identical by default.
  - Found the hard-constrained per-step QP is **infeasible at stiff exotherm steps** (OSQP reports `infeasible` on 12/12 hard configurations swept): the solver was being asked to find a feasible point of an empty set. Softening the predicted-state rows fixes it.
  - Diagnosed the residual non-convergence as `snn_opt`'s **absolute** projected-gradient tolerance on a problem whose gradient scale is ~1e10 — at a short horizon the SNN provably reaches the optimum (`u₀` to 0.0005 °C, relative objective −5.4e−8).

## Head-to-Head: SNN-QP vs CVXPY/OSQP

Both controllers driven by **one shared harness** — same initial state, same plant, same disturbance timing, same horizon, same canonical QP, same constraints. Configuration: `N=10`, soft state constraints, `trust_region=False` on both.

**1. Closed-loop control quality** (disturbance scenario, 15 °C step at t=60):

| Controller | Max Overshoot | Peak Cure Gradient | Gradient Violations | Compute/step |
|---|---|---|---|---|
| CVXPY / OSQP | 13.77 °C | 0.3445 Δα | 2 | ~5.6 ms |
| **SNN-QP (`snn_opt`)** | **13.24 °C** | **0.3418 Δα** | **2** | **~115 ms** |

Both reach full uniform cure (final α = 1.000) with zero actuator-limit violations.

**2. Solver agreement — the numbers that actually test the claim:**

| Metric | Nominal heat-up | Disturbance @ 60 | Stiff exotherm window |
|---|---|---|---|
| RMS applied-control difference | **0.714 °C** | 0.791 °C | 0.559 °C |
| RMS closed-loop trajectory difference | **0.249** | 0.279 | 0.414 |
| Max abs. control difference | 3.57 °C | 3.95 °C | 0.65 °C |
| SNN max constraint residual | 1.55 | 2.07 | 1.55 |
| **SNN formally converged** | **0 %** | **0 %** | **0 %** |
| Steps feasible enough to score objective gap | 48.1 % | 58.8 % | 45.2 % |
| Mean objective gap (feasible steps only) | 1.05e−4 | 1.22e−4 | 5.02e−5 |
| Applied moves corrected by the safety clip | 13.1 % | 15.0 % | 29.0 % |

Against the pre-validation configuration (mismatched model, hard constraints, N=20): RMS control difference **16.005 → 0.714 °C** (−96 %), max **57.70 → 3.57 °C** (−94 %), max constraint residual **1.85e5 → 1.55** (five orders of magnitude).

**Caveats stated up front, not buried:**
- **Formal convergence is 0 %.** The applied controls agree closely and the objective gap is small where measurable, but the solver never satisfies its own stopping criterion at a usable horizon. This is why the verdict is not "equivalent".
- **41.7 % of heat-up steps have *both* controllers pinned at the 4 °C/min slew limit** (77.4 % inside the stiff window). Agreement on those steps reflects a shared actuator limit, not solver agreement, and is excluded from the claim.
- **Horizon is load-bearing.** `N=5` gives apparently perfect agreement (0.000 °C RMS, 98.8 % formal convergence) and is **degenerate** — it drives the oven to its 10 °C floor and the part never cures (α = 0.000). Always check final α before reading agreement as success.

![SNN-QP vs CVXPY Overlay](assets/snn_vs_cvxpy_overlay.png)

> The overlay above was generated from the pre-validation configuration and is retained for historical context; the current numbers are the tables above and the plots under `results/final_comparison/`.

## Visualizing Control Performance

### Phase 2: Open-Loop Thermal Runaway
In the unmanaged open-loop baseline, the autoclave air follows a static, pre-programmed ramp-and-hold profile. Because the system is blind to the internal state of the composite, the exothermic chemical reaction during the gelation phase causes the center temperature to violently spike past 140°C, leading to thermal degradation and severe internal residual stress.

![Open-Loop Thermal Runaway](assets/openloop_test.png)

### Phase 3: Closed-Loop MPC Active Control
With the CVXPY MPC active, the solver's prediction horizon anticipates the exponential exothermic heat generation. Before the center temperature can critically overshoot, the controller dynamically drops the autoclave temperature (engaging a "Thermal Brake" around t=105 mins) to pull excess heat out of the composite surface. This contains the internal spike and ensures the center and surface cure uniformly ($\Delta \alpha = 0$).

![Closed-Loop MPC Performance](assets/closedloop_test.png)


### Phase 4: Closed-Loop SNN-MPC Active Control
In the final neuromorphic implementation, the CVXPY engine is entirely replaced by the Spiking Neural Network solver. The SNN executes the same control strategy as the baseline: it heats the autoclave along the maximum physical speed limit (4°C/min), rejects the 15°C thermal disturbance injected at t=60, and brakes to catch the exponential exotherm.

The SNN-QP is handed the **identical** per-step QP as the baseline (verified bit-identical, `tests/test_qp_parity.py`), and reaches full uniform cure with an overshoot of 13.24 °C against the baseline's 13.77 °C. That closeness is a measured agreement of the *applied control* (0.714 °C RMS), **not** a demonstration that the solver converged — its formal convergence criterion is met on 0 % of steps, and ~13 % of applied moves are corrected by a downstream safety clip. See [`docs/PHASE4_VALIDATION_REPORT.md`](docs/PHASE4_VALIDATION_REPORT.md) for what is and is not established.

![SNN-MPC Performance](assets/snn_closedloop_test.png)

## Running the code

Python 3.13 in a checked-in `.venv/` (Windows). Use the interpreter explicitly.
`MPLBACKEND=Agg` is required for a non-interactive run (each script ends in `plt.show()`);
`PYTHONIOENCODING=utf-8` because the scripts print `°C`.

```bash
# The three simulations (plain scripts, NOT pytest)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_open_loop_baseline.py  # exotherm runaway sanity (~140 C peak)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_closed_loop.py         # CVXPY/OSQP baseline metrics
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_snn_closed_loop.py     # SNN-QP metrics

# Proof that both controllers receive the identical QP (20 assertions, fast)
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe tests/test_qp_parity.py

# The controlled head-to-head that produced the tables above
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg .venv/Scripts/python.exe \
    tools/final_controlled_comparison.py --horizon 10 --soft --k0-scale 0.1
```

There is no test runner, linter, or build step. `requirements.txt` is UTF-16-encoded (re-export
as UTF-8 if pip chokes). `snn_opt==0.4.0` is not on the public index in the usual form — do not
casually reinstall the venv.

### Reproducing each claim

| Claim in this README | Script that proves it | Evidence written to |
|---|---|---|
| Both controllers receive identical QPs | `tests/test_qp_parity.py` | stdout (20 checks) |
| The three agreement numbers | `tools/final_controlled_comparison.py` | `results/final_comparison/<run>/` |
| The hard QP is infeasible at stiff steps | `tools/conditioning_sweep.py` | `results/conditioning_sweep.json` |
| Non-convergence is an absolute-tolerance artifact | `tools/optimum_agreement_probe.py` | `results/optimum_agreement_probe.json` |
| Per-solve diagnostics (residuals, bounds, clipping) | `tools/snn_solve_instrumentation.py` | `results/snn_solve_diagnostics.json` |
| Hessian conditioning analysis | `tools/qp_conditioning_probe.py` | `results/qp_conditioning_report.json` |

Every run under `results/final_comparison/` carries its own `summary.json` with the full
provenance block (git commit, package versions, solver config, cost weights, disturbance
convention) plus `per_step_metrics.csv`, `report.md`, and a plot. Runs are never overwritten.

**Configuration warning.** `trust_region` and `soft_state_constraints` must be set *identically*
on both controllers, or they stop solving the same problem and no head-to-head number is valid.
The comparison harness enforces this structurally (one config feeds both); the standalone
scripts rely on the matching constructor arguments at the top of each `run_*()`.

## Repository Structure

```text
├── ref_docs/
│   └── dufour mpc.pdf                  # Core mathematical reference literature
├── docs/
│   ├── PHASE4_VALIDATION_REPORT.md     # ** AUTHORITATIVE ** validation report (Rev. 2)
│   └── SNN_QP_SOLVER_PARITY_REPORT.md  # Superseded earlier report, kept for history
├── src/
│   ├── constants.py                    # Material properties, spatial limits, and tuning weights
│   ├── dynamics.py                     # linearize(): single source of truth for (Ap, Bp)
│   ├── qp_builder.py                   # build_canonical_qp(): single per-step QP construction
│   ├── mpc_cvxpy_controller.py         # CVXPY/OSQP baseline adapter
│   ├── snn_mpc_controller.py           # Neuromorphic SNN-QP adapter (Jacobi preconditioning)
│   └── plant_simulator.py              # Non-linear 1D PDE engine (The Digital Twin)
├── tests/
│   ├── test_open_loop_baseline.py      # Verification against unmanaged thermal runaway
│   ├── test_closed_loop.py             # CVXPY MPC execution
│   ├── test_snn_closed_loop.py         # SNN-MPC execution and metric logging
│   └── test_qp_parity.py               # Proves both adapters receive identical QPs
├── tools/
│   ├── final_controlled_comparison.py  # The controlled head-to-head harness
│   ├── conditioning_sweep.py           # trust_region x soft x horizon x step-size sweep
│   ├── optimum_agreement_probe.py      # SNN answer vs. known optimum
│   ├── convergence_blocker_probe.py    # Which convergence criterion blocks, and why
│   ├── snn_solve_instrumentation.py    # Full per-solve diagnostic record
│   ├── qp_conditioning_probe.py        # Hessian/constraint conditioning analysis
│   └── pdf_to_md.py                    # Reference-PDF -> Markdown (native text + OCR fallback)
├── results/                            # Generated evidence (never overwritten)
├── LICENSE
├── README.md
└── requirements.txt                    # Includes snn_opt, CVXPY, NumPy, SciPy, Matplotlib
```

### Converting reference papers to Markdown
Some reference PDFs (e.g. `ref_docs/dufour mpc.pdf`) use embedded fonts with no Unicode map, so plain text extraction yields garbage. `tools/pdf_to_md.py` extracts native text where possible and automatically falls back to OCR (RapidOCR/ONNX) for scanned or broken-font pages — pip-only and CPU-only (no system binaries, no GPU):

```bash
pip install -r tools/requirements-pdf.txt
python tools/pdf_to_md.py "ref_docs/dufour mpc.pdf"          # -> ref_docs/dufour mpc.md
python tools/pdf_to_md.py paper.pdf -o notes.md --pages 1-12 # subset a large paper
```

OCR runs at roughly 6 s/page on a laptop CPU (~4 min for a 38-page paper). Body text and tables are reliable; mathematical notation is only approximated — verify equations against the source.