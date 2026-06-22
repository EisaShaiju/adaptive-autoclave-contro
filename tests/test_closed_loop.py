"""
test_closed_loop.py
Runs the Plant with the CVXPY Controller, tests disturbances, and outputs metrics.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
from src.mpc_cvxpy_controller import MPCSolver
import src.constants as const

def run_closed_loop():
    plant = AutoclavePlant(initial_temp=28.0)
    controller = MPCSolver(horizon=20, target_temp=120.0)
    
    total_minutes = 160
    time_steps = int(total_minutes * 60 / const.TE)
    
    # Storage
    history_Tc1 = [] 
    history_Tc3 = [] 
    history_ac1 = [] 
    history_ac3 = [] 
    Ta_profile = []
    solve_times = []
    gradient_violations = 0

    current_Ta = 28.0
    current_state = plant.get_state()

    print("Starting Closed-Loop MPC Simulation with Disturbance...")
    
    for k in range(time_steps):            
        # 1. Controller calculates its planned optimal move based on current sensors
        current_Ta, s_time = controller.compute_control_action(current_state, current_Ta)
        solve_times.append(s_time)

        # INJECT DISTURBANCE: Sudden drop in oven temp at t=60 mins
        # This overrides the controller's plan and forces a physical shock to the oven
        if k == 60:
            print(">> INJECTING DISTURBANCE: Part looses 15 degrees of thermal energy")
            plant.T_comp-=15.0
            plant.T_tool-=15.0
        
        Ta_profile.append(current_Ta) # Log the actual applied temperature
        
        # 2. Plant executes the actual physical move
        current_state = plant.step(Ta_input=current_Ta)
        
        # 3. Log data
        Tc1, Tc3 = current_state[0], current_state[2]
        history_Tc1.append(Tc1)
        history_Tc3.append(Tc3)
        history_ac1.append(current_state[7])
        history_ac3.append(current_state[9])
        
        # Track gradient violations
        if abs(Tc1 - Tc3) > const.GRADIENT_MAX + 0.1: # Minor floating point buffer
            gradient_violations += 1

    # ==========================================
    # CALCULATE METRICS
    # ==========================================
    target_temp = 120.0
    max_temp_reached = max(max(history_Tc1), max(history_Tc3))
    max_overshoot = max(0.0, max_temp_reached - target_temp)
    
    # Cure Uniformity: Max Delta Alpha during the critical gelation phase
    max_delta_alpha = max(np.abs(np.array(history_ac3) - np.array(history_ac1)))
    
    avg_solve_time = np.mean(solve_times)

    print("\n--- PHASE 3: MPC CONTROLLER METRICS ---")
    print(f"Max Temp Overshoot:    {max_overshoot:.2f} °C")
    print(f"Max Cure Delta:        {max_delta_alpha:.4f} (Delta Alpha during cure)")
    print(f"Constraint Violations: {gradient_violations}")
    print(f"Avg Solver Time/Step:  {avg_solve_time:.2f} ms")
    print("---------------------------------------")

    # ==========================================
    # PLOTTING
    # ==========================================
    time_axis = np.arange(time_steps)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, Ta_profile, 'k-', linewidth=2, label='Autoclave Air (Ta) - MPC Governed')
    plt.plot(time_axis, history_Tc3, 'b--', label='Surface Temp (Tc3)')
    plt.plot(time_axis, history_Tc1, 'r:', label='Center Temp (Tc1)')
    plt.axvline(x=60, color='g', linestyle='-.', alpha=0.5, label='Disturbance Injected')
    plt.ylabel('Temperature (°C)')
    plt.title('Closed-Loop MPC: Temperature Tracking & Disturbance Rejection')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, history_ac3, 'b--', label='Surface Alpha')
    plt.plot(time_axis, history_ac1, 'r:', label='Center Alpha')
    plt.axhline(y=0.5, color='k', linestyle='-.', label='Gelation Point')
    plt.ylabel('Fractional Degree of Cure')
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_closed_loop()
