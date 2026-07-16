# Adaptive Process Control for Autoclave Composite Curing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/Optimization-CVXPY-orange)
![Control](https://img.shields.io/badge/Control-MPC-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview
This repository contains the simulation environment and control architecture for optimizing the autoclave curing process of thick-sectioned composite laminates. The project aims to replace unmanaged open-loop curing cycles with real-time closed-loop Model Predictive Control (MPC), and ultimately to demonstrate that an event-driven Spiking Neural Network (SNN) solver can correctly close the same MPC loop — producing equivalent control quality while being architecturally suited to low-power neuromorphic and FPGA edge deployment.

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
  - Mapped each per-step MPC QP to an equivalent spiking LIF dynamical system (gradient descent + discrete boundary projections), with Jacobi preconditioning of the condensed Hessian and an auto-computed $k_0$ step size.
  - Validated that the SNN-QP closed-loop trajectory matches the OSQP baseline (overshoot 10.82°C vs 13.12°C) while governing the stiff, non-linear Arrhenius thermodynamics.
  - Ran on the compiled `snn_opt` kernel (`backend='c'`): ~186 ms/step, comparable to the OSQP baseline on CPU and ~70× faster than the pure-Python reference path.
  - Established that the condensed QP must retain the *true* linearized dynamics through the exotherm (spectral radius > 1) — artificially stabilizing the prediction erases the exotherm and defeats the brake — with preconditioning keeping the resulting QP well-conditioned.

## Baseline Metrics (Classical MPC)
Running the Phase 3 closed-loop solver establishes the following reference for SNN-QP equivalence validation:
* **Max Temp Overshoot:** ~13.12°C (Safely contained below degradation limits)
* **Peak In-Process Cure Gradient:** ~0.36 Δα (Maximum spatial non-uniformity during gelation)
* **Constraint Violations:** 1 (During exothermic spike, logged for reference)
* **Average Compute Time:** ~124 ms/step (OSQP on CPU; head-to-head against the SNN-QP in Phase 4)

## SNN Metrics (Neuromorphic MPC)
Running the Phase 4 closed-loop SNN solver against the same non-linear digital twin yields control quality on par with the OSQP baseline — the SNN-QP solves the identical per-step MPC-QP and reproduces the baseline trajectory:
* **Max Temp Overshoot:** 10.82°C (brakes the exotherm slightly earlier than the baseline)
* **Peak In-Process Cure Gradient:** 0.3289 Δα
* **Constraint Violations:** 2 (one extra transient gradient touch during the more aggressive brake)
* **Average Compute Time:** ~186 ms/step (compiled `snn_opt` kernel; comparable to the OSQP baseline on CPU). This is the sequential-CPU reference — the architectural value of the projection dynamics is O(1)-style convergence on parallel, event-driven neuromorphic / FPGA hardware, not CPU wall-clock.

### Head-to-Head: SNN-QP vs CVXPY/OSQP
Both controllers were run on an identical plant, disturbance, and MPC-QP formulation:

| Controller | Max Overshoot | Peak Cure Gradient | Constraint Violations | Compute/step |
|---|---|---|---|---|
| CVXPY / OSQP | 13.12°C | 0.3579 Δα | 1 | ~124 ms |
| **SNN-QP (`snn_opt`)** | **10.82°C** | **0.3289 Δα** | **2** | **~186 ms** |

![SNN-QP vs CVXPY Overlay](assets/snn_vs_cvxpy_overlay.png)

The two trajectories are indistinguishable through heat-up and disturbance rejection; they differ only in the post-exotherm recovery, where the SNN's slightly harder brake trades one transient gradient touch for lower peak overshoot.

## Visualizing Control Performance

### Phase 2: Open-Loop Thermal Runaway
In the unmanaged open-loop baseline, the autoclave air follows a static, pre-programmed ramp-and-hold profile. Because the system is blind to the internal state of the composite, the exothermic chemical reaction during the gelation phase causes the center temperature to violently spike past 140°C, leading to thermal degradation and severe internal residual stress.

![Open-Loop Thermal Runaway](assets/openloop_test.png)

### Phase 3: Closed-Loop MPC Active Control
With the CVXPY MPC active, the solver's prediction horizon anticipates the exponential exothermic heat generation. Before the center temperature can critically overshoot, the controller dynamically drops the autoclave temperature (engaging a "Thermal Brake" around t=105 mins) to pull excess heat out of the composite surface. This contains the internal spike and ensures the center and surface cure uniformly ($\Delta \alpha = 0$).

![Closed-Loop MPC Performance](assets/closedloop_test.png)


### Phase 4: Closed-Loop SNN-MPC Active Control
In the final neuromorphic implementation, the CVXPY engine is entirely replaced by the Spiking Neural Network solver. The SNN executes the same control strategy as the baseline: it heats the autoclave along the maximum physical speed limit (4°C/min), smoothly rejects the 15°C thermal disturbance injected at t=60, and executes a sharp thermal brake to catch the exponential exotherm near t=104.

Because the SNN-QP solves the identical per-step MPC-QP, its closed-loop trajectory reproduces the OSQP baseline (see the head-to-head overlay above), reaching full, uniform cure with an overshoot of 10.82°C.

![SNN-MPC Performance](assets/snn_closedloop_test.png)

## Repository Structure

```text
├── ref_docs/
│   └── dufour mpc.pdf                  # Core mathematical reference literature
├── src/
│   ├── __init__.py
│   ├── constants.py                    # Material properties, spatial limits, and tuning weights
│   ├── mpc_cvxpy_controller.py         # Closed-loop baseline QP solver with Jacobian linearization
│   ├── snn_mpc_controller.py           # Neuromorphic SNN-QP solver using native LIF dynamics
│   └── plant_simulator.py              # Non-linear 1D PDE engine (The Digital Twin)
├── tests/
│   ├── __init__.py
│   ├── test_closed_loop.py             # CVXPY MPC execution
│   ├── test_snn_closed_loop.py         # SNN-MPC execution, metric logging, and graph generation
│   └── test_open_loop_baseline.py      # Verification script against unmanaged thermal runaway
├── tools/
│   ├── pdf_to_md.py                    # Reference-PDF -> Markdown (native text + OCR fallback)
│   └── requirements-pdf.txt            # Pip-only, CPU-only deps for the PDF tool (no Tesseract/Poppler)
├── .gitignore
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