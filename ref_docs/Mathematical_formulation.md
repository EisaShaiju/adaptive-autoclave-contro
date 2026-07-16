# Mathematical Formulation for Autoclave Curing MPC

This document outlines the theoretical foundation and mathematical linearizations used to model the 1D composite curing process and construct the closed-loop Model Predictive Control (MPC) matrices. The physical constants and core kinetics are adapted from Dufour et al. (2004).

---

## 1. State-Space Representation

The physical plant is discretized into a 1D spatial grid using the Explicit Finite Difference Method. The state vector $x_k$ at any discrete time step $k$ consists of 10 variables: 3 composite nodes (temperatures), 4 tooling nodes (temperatures), and 3 cure states (fractional degree of cure for the composite).

The state vector is defined as:
$$x_k = [T_{c1}, T_{c2}, T_{c3}, T_{t1}, T_{t2}, T_{t3}, T_{t4}, \alpha_1, \alpha_2, \alpha_3]^T$$

The manipulated variable (input) $u_k$ is the temperature of the autoclave air, which interacts strictly with the outermost tooling node ($T_{t4}$):
$$u_k = [T_a]$$

---

## 2. Heat Transfer Dynamics

The thermal diffusion through the composite and the steel tooling is governed by their respective Fourier numbers, calculated as:
$$F_c = \frac{k_c \Delta t}{\rho_c C_{pc} \Delta z^2} \quad \text{and} \quad F_t = \frac{k_t \Delta t}{\rho_t C_{pt} \Delta z^2}$$

For an internal node $i$, the explicit finite difference calculation is purely linear:
$$T_i^{k+1} = T_i^k + F \left( T_{i-1}^k - 2T_i^k + T_{i+1}^k \right)$$

Boundary conditions enforce a symmetry axis at the center of the composite ($T_{c1}$) and convective heat transfer at the autoclave boundary ($T_{t4}$).

---

## 3. Cure Kinetics & Exothermic Heat Generation

The chemical cross-linking of the epoxy resin generates internal exothermic heat. The rate of this reaction is modeled by the highly non-linear Arrhenius equation:
$$f(T, \alpha) = \frac{d\alpha}{dt} = A_c e^{-\frac{E_a}{RT}} \alpha^m (1-\alpha)^n$$

The physical temperature rise due to this chemical reaction over a single time step $\Delta t$ is scaled by the material's mass ratio and enthalpy:
$$\Delta T_{exo} = \left( \frac{M_R \Delta H}{C_{pc}} \right) \cdot f(T, \alpha) \cdot \Delta t$$

### Phase-Wise Linearization (The Jacobians)
Because CVXPY requires a strictly linear Quadratic Programming (QP) formulation, the Arrhenius kinetics must be linearized at a specific operating point $(T_0, \alpha_0)$ using a first-order Taylor series expansion. 

The base kinetic rate at the operating point is:
$$f_0 = A_c e^{-\frac{E_a}{R T_0}} \alpha_0^m (1-\alpha_0)^n$$

The partial derivative with respect to Temperature ($J_T$) is:
$$J_T = \frac{\partial f}{\partial T} \bigg|_{0} = f_0 \cdot \left( \frac{E_a}{R T_0^2} \right)$$

The partial derivative with respect to the Degree of Cure ($J_\alpha$) is:
$$J_\alpha = \frac{\partial f}{\partial \alpha} \bigg|_{0} = f_0 \cdot \left( \frac{m}{\alpha_0} - \frac{n}{1-\alpha_0} \right)$$

These scalar Jacobians are injected into the discrete-time transition matrix $A_p$ to predict the exothermic spike linearly.

---

## 4. MPC Quadratic Programming Objective

The control policy minimizes a cost function $J$ over a prediction horizon $N$, balancing setpoint tracking against absolute energy consumption and aggressive actuator movements.

$$\min_{u} \sum_{k=0}^{N-1} \left( (x_k - x_{target})^T Q (x_k - x_{target}) + u_k^T R u_k + \Delta u_k^T S \Delta u_k \right)$$

**Where:**
* **$Q$:** Penalty matrix for state deviation from the target temperature (Prioritizing composite nodes).
* **$R$:** Penalty for absolute control effort (Minimizing overall energy usage).
* **$S$:** Penalty for the rate of change of the control input ($\Delta u_k$), minimizing actuator wear and thermal shocks.

**Subject to Physical Constraints:**

1. **Actuator Limits:** $T_{a,min} \le u_k \le T_{a,max}$
2. **Rate Limits:** $|\Delta u_k| \le Rate_{max}$
3. **Thermal Stress Gradient:** $|T_{c1} - T_{c3}| \le 10^\circ C$

---

## 5. Model Fidelity and Known Limitations

### 5.1 Parameters verified against Dufour (2004)
Every physical constant in `src/constants.py` reproduces the Appendix table of Dufour et al. (2004) *exactly*: $k_c = 0.23793$, $\rho_c = 1890$, $c_{pc} = 1134$, $\beta_t = 0.3\beta_c$, $m_r = 0.46$, $-\Delta H_r = 8.525\times10^4$, $A_c = 1.233\times10^{21}$, $E_a = 1.674\times10^5$, $m = 0.524$, $n = 1.476$, with a 3-node composite / 4-node tooling grid and the same actuator/gradient bounds. The plant PDE and the **single-term autocatalytic kinetics** $\dot\alpha = A_c e^{-E_a/RT}\alpha^m(1-\alpha)^n$ are the exact model Dufour adopts from Bogetti and Pillai — this project uses no alternative (e.g. two-term Kamal–Sourour) kinetics.

### 5.2 The `0^+` cure seed is prescribed by the model
The single-term rate is identically zero at $\alpha = 0$ (the $\alpha^m$ factor), so the reaction cannot self-initiate from a true zero state. This is intrinsic to the model: Dufour's initial condition is written $\alpha(z,0) = 0^+$. The plant's `alpha = 1e-5` initialization implements exactly this "zero-plus" seed; it is faithful to the reference, not a numerical workaround.

### 5.3 Linearization caveat (the real fragility)
Dufour solves the *nonlinear* model directly (Levenberg–Marquardt). This project instead linearizes the Arrhenius term into a per-step QP so it can be solved by both OSQP and the SNN-QP solver. That linearization is where care is required: the cure Jacobian
$$J_\alpha = f_0\left(\frac{m}{\alpha} - \frac{n}{1-\alpha}\right)$$
is **singular as $\alpha \to 0$ and $\alpha \to 1$** (e.g. $J_\alpha \approx +8.9$ at the $\alpha = 10^{-5}$ seed). The controllers therefore clamp $\alpha$ to $[10^{-3}, 0.999]$ and bound the exotherm Jacobian contributions before injecting them into $A_p$. This keeps the local linear model well-posed through cure onset and completion; it is a property of the QP *adaptation*, not of the plant.

### 5.4 Gradient bound
Dufour reports that a **constant** $\pm 10^\circ C$ center–surface bound is insufficient for an ideal inside-out cure (their minimum-time objective drives the degree-of-cure crossover to $\approx 0.92$ instead of $< 0.5$) and designs a *funnel-shaped*, time-varying bound instead. This project uses the constant $\pm 10^\circ C$ bound with a **setpoint-tracking** objective (target $120^\circ C$) rather than Dufour's minimum-time objective; a funnel-shaped bound is a natural future refinement if inside-out crossover is later added as an explicit quality target.

### 5.5 Solver convergence: output vs. formal criterion
A careful distinction, verified per step against OSQP on the identical preconditioned QP:

* **Output convergence (holds):** the applied control $u_0$ produced by the SNN-QP matches the OSQP optimum to $|\Delta u_0| \approx 0.000^\circ C$ at every well-posed step. This is why the closed-loop trajectory reproduces the baseline.
* **Formal convergence (does not hold):** the solver's early-stop criterion (projected-gradient norm below tolerance *and* feasibility) is **not** met — the solver runs its full iteration budget every step. On these stiff, tightly-constrained QPs the projected gradient plateaus well above tolerance because the iterate *chatters* along the coupled slew/gradient active set (one projection perturbs a neighbouring constraint). This limit-cycle behaviour is intrinsic to greedy boundary projection and is the aspect most relevant to ongoing SNN-solver research.
* **Peak-exotherm instance:** at the exotherm the condensed QP is numerically extreme — the local model's spectral radius reaches $\approx 1.55$, so $A_p^{N}$ over $N=20$ amplifies by $\approx 4400\times$. The raw iterate diverges there (and OSQP fails on the same scaled instance); the physically-correct control is recovered by the slew-rate clamp in `compute_control_action`. Shortening the horizon or adding a step-size/trust-region guard for the high-$\rho$ regime is the natural next step if a converged solution is required at that instant.