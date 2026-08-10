"""
test_snn_closed_loop.py
Executes the Spiking Neural Network MPC against the exact same 
non-linear physical twin used in Phase 3 to prove mathematical equivalence.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.plant_simulator import AutoclavePlant
from src.snn_mpc_controller import SNNMPCSolver
import src.constants as const

def run_snn_closed_loop():
    print("\n" + "="*60)
    print(" Starting NEUROMORPHIC SNN-MPC Simulation")
    print("="*60)
    
    plant = AutoclavePlant(initial_temp=28.0)
    # Recommended configuration (docs/PHASE4_VALIDATION_REPORT.md Rev. 2).
    # These MUST match test_closed_loop.py exactly, or the two controllers stop
    # solving the same QP and no head-to-head number is valid.
    controller = SNNMPCSolver(horizon=10, target_temp=120.0,
                               trust_region=False, soft_state_constraints=True,
                               k0_scale=0.1)

    # Storage arrays
    history_ta = []
    history_ac1 = []
    history_ac3 = []
    history_alpha1 = []
    history_alpha3 = []
    solve_times = []
    constraint_violations = 0
    
    current_Ta = 28.0 # Starting oven temp

    # Simulate 160 minutes
    for k in range(160):
        # --- PROGRESS TRACKER ---
        print(f" Simulating Minute {k:03d}/160 | Current Oven Temp: {current_Ta:6.2f} °C")
        
        # Read sensors
        current_state = plant.get_state()
        
        # Log data
        history_ta.append(current_Ta)
        history_ac1.append(current_state[0])
        history_ac3.append(current_state[2])
        history_alpha1.append(current_state[7])
        history_alpha3.append(current_state[9])
        
        # INJECT DISTURBANCE: Direct thermal shock to physical materials
        if k == 60:
            print("\n" + "!"*60)
            print(">> INJECTING DISTURBANCE: Part loses 15°C of thermal energy!")
            print("!"*60 + "\n")
            plant.T_comp -= 15.0
            plant.T_tool -= 15.0
            current_state = plant.get_state() # Update sensor read
            
        # SNN Calculates Next Move
        optimal_Ta, t_solve = controller.compute_control_action(current_state, current_Ta)
        solve_times.append(t_solve)
        
        current_Ta = optimal_Ta
        
        # Step Physics Engine
        plant.step(current_Ta)
        
        # Track gradient safety limits for metric reporting
        if abs(current_state[0] - current_state[2]) > const.GRADIENT_MAX + 0.1:
            constraint_violations += 1

    # --- SNN METRICS ---
    max_overshoot = max(0, max(history_ac1) - 120.0)
    max_delta_alpha = max(abs(np.array(history_alpha1) - np.array(history_alpha3)))
    avg_solve_time = np.mean(solve_times)

    print("\n" + "="*40)
    print("--- PHASE 4: SNN CONTROLLER METRICS ---")
    print(f"Max Temp Overshoot:    {max_overshoot:.2f} °C")
    print(f"Max Cure Delta:        {max_delta_alpha:.4f} (Peak in-process difference)")
    print(f"Constraint Violations: {constraint_violations}")
    print(f"Avg Solver Time/Step:  {avg_solve_time:.2f} ms")
    print("="*40 + "\n")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    time_axis = np.arange(160)

    # Plot 1: Temperatures
    ax1.plot(time_axis, history_ta, 'k-', linewidth=2, label='Autoclave Air (Ta) - SNN Governed')
    ax1.plot(time_axis, history_ac3, 'b--', label='Surface Temp (Tc3)')
    ax1.plot(time_axis, history_ac1, 'r-', label='Center Temp (Tc1)')
    ax1.axhline(y=120.0, color='g', linestyle=':', linewidth=2, label='Target Temp (120°C)')
    ax1.axvline(x=60, color='orange', linestyle='-.', alpha=0.8, label='Disturbance Injected')
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Neuromorphic SNN-MPC: Temperature Tracking & Disturbance Rejection')
    ax1.grid(True)
    ax1.legend(loc='lower right')

    # Plot 2: Alpha
    ax2.plot(time_axis, history_alpha3, 'b--', label='Surface Alpha')
    ax2.plot(time_axis, history_alpha1, 'r-', label='Center Alpha')
    ax2.axhline(y=0.5, color='k', linestyle='-.', label='Gelation Point')
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_ylabel('Fractional Degree of Cure')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_snn_closed_loop()