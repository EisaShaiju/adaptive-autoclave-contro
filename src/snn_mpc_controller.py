"""
snn_mpc_controller.py
Closed-loop MPC solver utilizing Spiking Neural Networks (LIF Dynamics).
Translates phase-wise linearized Arrhenius physics into Dense Canonical QP form.
"""
import numpy as np
import time
import src.constants as const
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig

class SNNMPCSolver:
    def __init__(self, horizon=20, target_temp=120.0):
            self.N = horizon
            self.nx = 10  # 7 Composite/Tool Temps, 3 Alphas
            self.nu = 1   # 1 Input (Autoclave Air Ta)
            self.target_temp = target_temp

            conv_config = ConvergenceConfig(
                enable_early_stopping=True,
                check_every=50,
                min_iterations=100,
                patience=3
            )
            
            self.solver_config = SolverConfig(
                k0=None,                      
                k0_scale=0.1,                 # Back to a reasonable scale 
                projection_method='adaptive', 
                max_iterations=5000,
                convergence=conv_config
            )
            
            # WARM START: Set to None initially so we can spawn it safely inside the box
            self.U_warm = None 
            
            self.Q_diag = np.zeros(self.nx)
            for i in range(3):
                self.Q_diag[i] = 100.0  
                
            self.R_val = 0.1  
            self.S_val = 1.0

    def update_matrices(self, T0_degC, alpha0):
        """Builds the Linearized Ap and Bp matrices (Identical to CVXPY Baseline)"""
        Ap = np.zeros((self.nx, self.nx))
        Bp = np.zeros(self.nx)
        T0_K = T0_degC + 273.15
        
        # Arrhenius Kinetics at Operating Point
        f0 = const.AC * np.exp(-const.EA / (const.R * T0_K)) * \
             (alpha0**const.M_EXP) * ((1 - alpha0)**const.N_EXP)
             
        dT = f0 * (const.EA / (const.R * (T0_K**2)))
        a_safe = max(1e-3, min(alpha0, 0.999))
        da = f0 * ((const.M_EXP / a_safe) - (const.N_EXP / (1 - a_safe)))
        
        exo_mult = (const.MR * const.DH / const.CPC) * const.TE
        
        # Heat Transfer & Exotherm
        for i in range(const.NZ_C):
            Ap[i, i] = 1 - 2*const.FC + (exo_mult * dT)
            Ap[i, i+7] = exo_mult * da
            if i == 0:
                Ap[i, i+1] = 2 * const.FC
            elif i == const.NZ_C - 1:
                Ap[i, i-1] = const.FC
                Ap[i, const.NZ_C] = const.FC
            else:
                Ap[i, i-1] = const.FC
                Ap[i, i+1] = const.FC

        for i in range(const.NZ_C, const.NZ_C + const.NZ_T):
            Ap[i, i] = 1 - 2*const.FT
            if i == const.NZ_C:
                Ap[i, i-1] = const.FT
                Ap[i, i+1] = const.FT
            elif i == const.NZ_C + const.NZ_T - 1:
                Ap[i, i-1] = const.FT
                Bp[i] = const.FT 
            else:
                Ap[i, i-1] = const.FT
                Ap[i, i+1] = const.FT
                
        for i in range(7, 10):
            Ap[i, i] = 1 + (da * const.TE)
            Ap[i, i-7] = dT * const.TE

        return Ap, Bp

    def build_dense_qp(self, Ap, Bp, x0, u_prev):
        """
        THE SNN TRANSLATION LAYER:
        Condenses the sequential MPC loop into 4 strict SNN matrices: A, b, C, d
        """
        # 1. Build Prediction Matrices (Phi and Gamma)
        Phi = np.zeros((self.N * self.nx, self.nx))
        Gamma = np.zeros((self.N * self.nx, self.N))
        
        A_k = np.eye(self.nx)
        for i in range(self.N):
            A_k = A_k @ Ap
            Phi[i*self.nx:(i+1)*self.nx, :] = A_k
            
            A_diff = np.eye(self.nx)
            for j in range(i, -1, -1):
                Gamma[i*self.nx:(i+1)*self.nx, j] = A_diff @ Bp
                A_diff = A_diff @ Ap
                
        # 2. Build Block Cost Matrices
        Q_bar = np.kron(np.eye(self.N), np.diag(self.Q_diag))
        R_bar = np.eye(self.N) * self.R_val
        
        D = np.eye(self.N)
        for i in range(1, self.N):
            D[i, i-1] = -1.0
            
        S_bar = np.eye(self.N) * self.S_val
        
        d0 = np.zeros(self.N)
        d0[0] = u_prev
        
        X_ref = np.zeros(self.N * self.nx)
        for i in range(self.N):
            X_ref[i*self.nx : i*self.nx + 3] = self.target_temp
            
        # 3. Calculate canonical A_snn (The biological synaptic weights)
        A_snn = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar + D.T @ S_bar @ D)
        A_snn = (A_snn + A_snn.T) / 2.0 # Ensure absolute symmetry for SNN stability
        
        # --- THE NEW REGULARIZATION LINE ---
        # This acts as mathematical "glue" to prevent the matrix from collapsing
        A_snn += np.eye(self.N) * 1e-3
        
        # 4. Calculate canonical b_snn (The biological injection current)
        E = Phi @ x0 - X_ref
        b_snn = 2.0 * (Gamma.T @ Q_bar @ E) - 2.0 * (D.T @ S_bar @ d0)
        
        # 5. Build Physical Constraint Walls (C_snn and d_snn)
        C_list = []
        d_list = []
        
        # Upper & Lower Temp Bounds
        C_list.append(np.eye(self.N))
        d_list.append(-np.ones(self.N) * const.TA_MAX)
        C_list.append(-np.eye(self.N))
        d_list.append(np.ones(self.N) * const.TA_MIN)
        
        # Slew Rate Bounds
        C_list.append(D)
        d_list.append(-d0 - const.TA_RATE_MAX)
        C_list.append(-D)
        d_list.append(d0 - const.TA_RATE_MAX)
        
        # Gradient Constraint: |Center - Surface| <= Grad_max
        G = np.zeros((self.N, self.N * self.nx))
        for i in range(self.N):
            G[i, i*self.nx + 0] = 1.0   # Center
            G[i, i*self.nx + 2] = -1.0  # Surface
            
        C_list.append(G @ Gamma)
        d_list.append(G @ Phi @ x0 - const.GRADIENT_MAX)
        C_list.append(-G @ Gamma)
        d_list.append(-G @ Phi @ x0 - const.GRADIENT_MAX)
        
        C_snn = np.vstack(C_list)
        d_snn = np.concatenate(d_list)
        
        row_norms = np.linalg.norm(C_snn, axis=1)
        row_norms[row_norms < 1e-12] = 1.0  # Prevent divide-by-zero on empty rows
        
        C_snn = C_snn / row_norms[:, np.newaxis]
        d_snn = d_snn / row_norms
        
        return A_snn, b_snn, C_snn, d_snn

    def compute_control_action(self, current_state, u_prev):
        """Calculates optimal Ta input using the Spiking Neural Network."""
        start_time = time.time()
        
        # Safely initialize the warm start inside the legal physical boundary
        if self.U_warm is None:
            self.U_warm = np.full(self.N, u_prev)
        
        # 1. Linearize Plant
        avg_T = np.mean(current_state[0:3])
        avg_a = np.mean(current_state[7:10])
        Ap, Bp = self.update_matrices(avg_T, avg_a)
        
        # 2. Condense into Canonical Form
        A_snn, b_snn, C_snn, d_snn = self.build_dense_qp(Ap, Bp, current_state, u_prev)
        
        # 3. Map to Spiking Solver
        problem = OptimizationProblem(A=A_snn, b=b_snn, C=C_snn, d=d_snn)
        solver = SNNSolver(problem, self.solver_config)
        
        # 4. Run Physics Simulation (Fire Spikes!)
        try:
            # WARM START: Pass the previous solution in to cut convergence time
            result = solver.solve(self.U_warm, verbose=False)
            solve_time = (time.time() - start_time) * 1000
            
            # Save final voltages to jump-start the next minute
            self.U_warm = result.final_x 
            
            #  required reporting: How hard did it bounce off the walls?
            if result.n_projections > 1000:
                print(f" SNN LOG: Heavy constraint traffic! Spikes: {result.n_projections}")
                
            return self.U_warm[0], solve_time
            
        except Exception as e:
            print(f" SNN CRITICAL ERROR: {e}")
            return u_prev, (time.time() - start_time) * 1000