# Adaptive Process Control for Autoclave Composite Curing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/Optimization-CVXPY-orange)
![Control](https://img.shields.io/badge/Control-MPC-brightgreen)

## Overview
This repository contains the simulation environment and control architecture for optimizing the autoclave curing process of thick-sectioned composite laminates. The project aims to replace mathematically heavy, open-loop curing cycles with real-time, closed-loop Model Predictive Control (MPC), and ultimately, highly efficient event-driven Spiking Neural Networks (SNNs).

The core physical plant is modeled based on the highly non-linear Arrhenius curing kinetics and 1D spatial heat transfer dynamics detailed in Dufour et al. (2004).

## Project Roadmap

This research is divided into four main phases:

- [x] **Phase 1: Mathematical Formulation**
  - Discrete-time linearized state-space modeling per cure phase (Heat-up, Dwell, Cool-down).
  - Derivation of Fourier numbers and linearized Arrhenius Jacobians.
  - Formulation of the MPC Quadratic Programming (QP) objective, penalizing tracking error, control effort, and overall thermal energy consumption.
- [ ] **Phase 2: Open-Loop Plant Digital Twin**
  - Implementation of the Explicit Finite Difference Method in Python.
  - Solving the raw, non-linear Partial Differential Equations (PDEs) to simulate composite curing.
  - Verification against the nominal reference profiles (Figure 4 & 6 from Dufour).
- [ ] **Phase 3: Closed-Loop CVXPY Baseline**
  - Integration of a conventional active-set/interior-point QP solver (CVXPY) into the simulation loop.
  - Enforcement of real-time spatial constraints (inside-out curing funnel) to prevent residual thermal stress.
  - Benchmarking of solver calculation times, trajectory overshoot, and energy efficiency.
- [ ] **Phase 4: Neuromorphic SNN Integration**
  - Mapping the QP objective landscape to an event-driven Spiking Neural Network.
  - Bypassing heavy matrix inversion to achieve microsecond, low-power adaptive control suitable for aerospace manufacturing edge devices.

## Repository Structure

```text
├── docs/
│   ├── MPC_QP_Formulation.pdf      # Mathematical derivation and explicit matrix dimensions
│   └── reference_literature/       # Associated papers (Dufour, etc.)
├── src/
│   ├── plant_simulator.py          # Non-linear PDE engine (The Digital Twin)
│   ├── mpc_cvxpy_controller.py     # Closed-loop baseline QP solver
│   └── snn_controller/             # (Upcoming) Neuromorphic control policies
├── tests/
│   └── test_open_loop_baseline.py  # Verification scripts against published data
├── requirements.txt
└── README.md