# Adaptive Process Control for Autoclave Composite Curing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/Optimization-CVXPY-orange)
![Control](https://img.shields.io/badge/Control-MPC-brightgreen)
![Status](https://img.shields.io/badge/Status-Phase_3_Complete-success)

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
- [ ] **Phase 4: Neuromorphic SNN Integration (In Progress)**
  - Mapping each per-step MPC QP to an equivalent spiking LIF dynamical system (Mancoo et al. equivalence framework).
  - Validating that SNN-QP produces the same per-step control actions and closed-loop trajectories as CVXPY, within tolerance.
  - Reporting convergence behaviour (iterations to tolerance) as the primary solver metric.
  - Deployment value lies in neuromorphic/FPGA hardware (event-driven, massively parallel); CPU runtime is not the comparison axis.

## Baseline Metrics (Classical MPC)
Running the Phase 3 closed-loop solver establishes the following reference for SNN-QP equivalence validation:
* **Max Temp Overshoot:** ~13.12°C (Safely contained below degradation limits)
* **Peak In-Process Cure Gradient:** ~0.36 Δα (Maximum spatial non-uniformity during gelation)
* **Constraint Violations:** 1 (During exothermic spike, logged for reference)
* **Average Compute Time:** ~120 ms/step (CPU reference; not used as a comparison target — see Phase 4)

## Visualizing Control Performance

### Phase 2: Open-Loop Thermal Runaway
In the unmanaged open-loop baseline, the autoclave air follows a static, pre-programmed ramp-and-hold profile. Because the system is blind to the internal state of the composite, the exothermic chemical reaction during the gelation phase causes the center temperature to violently spike past 140°C, leading to thermal degradation and severe internal residual stress.

![Open-Loop Thermal Runaway](assets/openloop_test.png)

### Phase 3: Closed-Loop MPC Active Control
With the CVXPY MPC active, the solver's prediction horizon anticipates the exponential exothermic heat generation. Before the center temperature can critically overshoot, the controller dynamically drops the autoclave temperature (engaging a "Thermal Brake" around t=105 mins) to pull excess heat out of the composite surface. This contains the internal spike and ensures the center and surface cure uniformly ($\Delta \alpha = 0$).

![Closed-Loop MPC Performance](assets\closedloop_test.png)

## Repository Structure

```text
├── ref_docs/
│   └── dufour mpc.pdf                  # Core mathematical reference literature
├── src/
│   ├── __init__.py
│   ├── constants.py                    # Material properties, spatial limits, and tuning weights
│   ├── mpc_cvxpy_controller.py         # Closed-loop baseline QP solver with Jacobian linearization
│   └── plant_simulator.py              # Non-linear 1D PDE engine (The Digital Twin)
├── tests/
│   ├── __init__.py
│   ├── test_closed_loop.py             # MPC execution, disturbance injection, and metric logging
│   └── test_open_loop_baseline.py      # Verification script against unmanaged thermal runaway
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                    # Environment dependencies (NumPy, SciPy, CVXPY, Matplotlib)
