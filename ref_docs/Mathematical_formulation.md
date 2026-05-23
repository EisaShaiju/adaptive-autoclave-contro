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

The control policy minimizes a cost function $J$ over a prediction horizon $N$, balancing setpoint tracking against aggressive actuator movements.

$$\min_{u} \sum_{k=0}^{N-1} \left( (x_k - x_{target})^T Q (x_k - x_{target}) + \Delta u_k^T R \Delta u_k \right)$$

**Subject to Physical Constraints:**
1. **Actuator Limits:** $T_{a,min} \le u_k \le T_{a,max}$
2. **Rate Limits:** $|\Delta u_k| \le \text{Rate}_{max}$
3. **Thermal Stress Gradient:** $|T_{c1} - T_{c3}| \le 10^\circ\text{C}$