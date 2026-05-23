"""
mpc_cvxpy_controller.py
The closed-loop MPC solver using Phase-Wise Linearization.
"""
import numpy as np
import cvxpy as cp
import src.constants as const

class MPCSolver:
    def __init__(self, horizon=25):
        self.N = horizon
        self.nx = 10 # 7 Temps, 3 Alphas
        self.nu = 1  # 1 Input (Ta)
        
        # Tuning Weights (Q, R, S)
        self.weight_Q = 100.0 # High penalty for missing target temp
        self.weight_R = 1.0   # Penalty for aggressive heater changes
        self.weight_S = 0.1   # Penalty for high energy usage
        
        # CVXPY Variable
        self.z = cp.Variable(self.N)

    def update_matrices(self, current_state, cure_phase):
        """
        Updates Ap and Bp based on the current phase (Heat-up, Dwell, Cool-down).
        This requires the calculated analytical Jacobians for the Arrhenius equation.
        """
        Ap = np.zeros((self.nx, self.nx))
        Bp = np.zeros(self.nx)
        
        # TODO: Insert linearized A and B matrices here
        
        return Ap, Bp

    def compute_control_action(self, current_state, u_prev, target_trajectory):
        """
        Builds the QP constraints and objective, then solves it.
        """
        # 1. Update linear matrices for the current operating point
        Ap, Bp = self.update_matrices(current_state, "Dwell")
        
        # 2. Build Phi and Gamma (Unroll horizon)
        # TODO: Matrix exponentiation loop for Phi and Gamma
        
        # 3. Define standard QP Cost
        # cost = cp.sum_squares(...) + ...
        
        # 4. Define Linear Constraints (Limits from constants.py)
        # constraints = [self.z >= const.TA_MIN, ...]
        
        # 5. Solve
        # prob = cp.Problem(cp.Minimize(cost), constraints)
        # prob.solve(solver=cp.OSQP)
        
        # return self.z.value[0]
        pass