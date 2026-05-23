"""
test_open_loop_baseline.py
Runs the Plant Simulator without a controller to verify the nominal cure profile.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plant_simulator import AutoclavePlant
import src.constants as const

def run_open_loop():
    plant = AutoclavePlant(initial_temp=28.0)
    
    total_minutes = 160
    time_steps = int(total_minutes * 60 / const.TE)
    
    # Store history for plotting
    history_Tc1 = [] # Center Temp
    history_Tc3 = [] # Surface Temp
    history_ac1 = [] # Center Alpha
    history_ac3 = [] # Surface Alpha
    Ta_profile = []  # Input Ta
    
    # Create the nominal dummy input (Ramp to 130C at 2C/min)
    current_Ta = 28.0
    ramp_rate_per_sec = 2.0 / 60.0

    print("Starting Open-Loop Simulation (Dufour Baseline)...")
    for k in range(time_steps):
        if current_Ta < const.TA_MAX:
            current_Ta += ramp_rate_per_sec * const.TE
            
        Ta_profile.append(current_Ta)
        
        # Step the physics engine forward 1 minute
        state = plant.step(Ta_input=current_Ta)
        
        # Save data: state is [Tc1, Tc2, Tc3, Tt1...Tt4, ac1, ac2, ac3]
        history_Tc1.append(state[0])
        history_Tc3.append(state[2])
        history_ac1.append(state[7])
        history_ac3.append(state[9])

    # Plotting Logic
    time_axis = np.arange(time_steps)
    
    plt.figure(figsize=(10, 8))
    
    # Temperature Plot (Should match Dufour Fig 4)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, Ta_profile, 'k-', label='Autoclave Air (Ta)')
    plt.plot(time_axis, history_Tc3, 'b--', label='Surface Temp (Tc3)')
    plt.plot(time_axis, history_Tc1, 'r:', label='Center Temp (Tc1)')
    plt.ylabel('Temperature (°C)')
    plt.title('Open-Loop Temperature Profile')
    plt.legend()
    plt.grid(True)
    
    # Alpha Plot (Should match Dufour Fig 6)
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
    run_open_loop()
