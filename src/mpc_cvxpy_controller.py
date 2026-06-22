"""
mpc_cvxpy_controller.py
Closed-loop MPC solver using Phase-Wise Linearization.
"""
import numpy as np
import cvxpy as cp
import time
import src.constants as const

class MPCSolver:
    def __init__(self, horizon=20, target_temp=120.0):
        self.N = horizon
        self.nx = 10  # 7 Temps, 3 Alphas
        self.nu = 1   # 1 Input (Ta)
        self.target_temp = target_temp
        
        # Tuning Weights
        self.Q = np.zeros((self.nx, self.nx))
        for i in range(3): # Heavily penalize composite temps deviating from target
            self.Q[i, i] = 100.0 
            
        self.R = np.array([[0.1]])  # Penalty for absolute effort (kept small)
        self.S = np.array([[1.0]])  # Penalty for rate of change (Delta u)
        
    def update_matrices(self, T0_degC, alpha0):
        """Builds the Linearized Ap and Bp matrices for the current step."""
        Ap = np.zeros((self.nx, self.nx))
        Bp = np.zeros(self.nx)
        
        T0_K = T0_degC + 273.15
        
        # 1. Base Kinetics at Operating Point
        f0 = const.AC * np.exp(-const.EA / (const.R * T0_K)) * \
             (alpha0**const.M_EXP) * ((1 - alpha0)**const.N_EXP)
             
        # 2. Jacobians (Partial Derivatives)
        dT = f0 * (const.EA / (const.R * (T0_K**2)))
        # Prevent division by zero if alpha is at boundaries
        a_safe = max(1e-3, min(alpha0, 0.999))
        da = f0 * ((const.M_EXP / a_safe) - (const.N_EXP / (1 - a_safe)))
        
        exo_mult = (const.MR * const.DH / const.CPC) * const.TE
        
        # 3. Fill Ap Matrix (Discrete State Space)
        # Composite Heat Transfer & Exotherm
        for i in range(const.NZ_C):
            Ap[i, i] = 1 - 2*const.FC + (exo_mult * dT) # Self temp + exothermic T-dependence
            Ap[i, i+7] = exo_mult * da                  # Exothermic alpha-dependence
            
            if i == 0:
                Ap[i, i+1] = 2 * const.FC
            elif i == const.NZ_C - 1:
                Ap[i, i-1] = const.FC
                Ap[i, const.NZ_C] = const.FC # Tooling boundary
            else:
                Ap[i, i-1] = const.FC
                Ap[i, i+1] = const.FC

        # Tooling Heat Transfer
        for i in range(const.NZ_C, const.NZ_C + const.NZ_T):
            Ap[i, i] = 1 - 2*const.FT
            if i == const.NZ_C:
                Ap[i, i-1] = const.FT # Composite boundary
                Ap[i, i+1] = const.FT
            elif i == const.NZ_C + const.NZ_T - 1:
                Ap[i, i-1] = const.FT
                Bp[i] = const.FT      # INPUT BOUNDARY (Ta) enters here
            else:
                Ap[i, i-1] = const.FT
                Ap[i, i+1] = const.FT
                
        # Alpha Dynamics
        for i in range(7, 10):
            Ap[i, i] = 1 + (da * const.TE)          # Alpha depends on alpha
            Ap[i, i-7] = dT * const.TE              # Alpha depends on Temp

        return Ap, Bp

    def compute_control_action(self, current_state, u_prev):
        """Calculates optimal Ta input over prediction horizon."""
        start_time = time.time()
        
        # Use average composite state to linearize
        avg_T = np.mean(current_state[0:3])
        avg_a = np.mean(current_state[7:10])
        Ap, Bp = self.update_matrices(avg_T, avg_a)
        
        # CVXPY Variables
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        
        cost = 0
        constraints = [x[:, 0] == current_state]
        
        x_target = np.zeros(self.nx)
        x_target[0:3] = self.target_temp
        
        for k in range(self.N):
            # Cost function: Track target + minimize actuator effort
            cost += cp.quad_form(x[:, k] - x_target, self.Q)

            # Absolute effort penalty(R)
            cost+=cp.quad_form(u[:,k],self.R)

            # Rate of change/slew rate penalty
            if k == 0:
                cost += cp.quad_form(u[:, k] - u_prev, self.S)
            else:
                cost += cp.quad_form(u[:, k] - u[:, k-1], self.S)
            
            # System Dynamics Constraint
            constraints += [x[:, k+1] == Ap @ x[:, k] + Bp * u[0, k]]
            
            # Physical Constraints
            constraints += [u[:, k] >= const.TA_MIN]
            constraints += [u[:, k] <= const.TA_MAX]
            
            # Rate of heating constraint
            if k == 0:
                constraints += [cp.abs(u[:, k] - u_prev) <= const.TA_RATE_MAX]
            else:
                constraints += [cp.abs(u[:, k] - u[:, k-1]) <= const.TA_RATE_MAX]
                
            # Gradient Constraint: |Center Temp - Surface Temp| <= 10
            constraints += [cp.abs(x[0, k] - x[2, k]) <= const.GRADIENT_MAX]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            solve_time = (time.time() - start_time) * 1000 # Convert to ms
            
            if u.value is None:
                return u_prev, solve_time
            return u.value[0, 0], solve_time
        except Exception:
            return u_prev, (time.time() - start_time) * 1000