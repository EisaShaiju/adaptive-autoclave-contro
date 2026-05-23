"""
constants.py
Stores all physical parameters, spatial grids, and simulation limits based on Dufour (2004).
"""
import numpy as np

# --- 1. PHYSICAL PARAMETERS ---
KC = 0.23793          # Thermal conductivity of composite (W/m/K)
RHO_C = 1890          # Density of composite (kg/m^3)
CPC = 1134            # Heat capacity of composite (J/kg/K)
MR = 0.46             # Mass ratio of resin
DH = 85250            # Enthalpy of reaction (J/kg)
AC = 1.233e21         # Pre-exponential factor (1/s)
EA = 167400           # Activation energy (J/mol)
R = 8.314             # Gas constant (J/mol/K)
M_EXP = 0.524         # Kinetic exponent m
N_EXP = 1.476         # Kinetic exponent n

# Thermal Diffusivities
BETA_C = KC / (RHO_C * CPC)
BETA_T = 0.3 * BETA_C 

# --- 2. SPATIAL & TIME DISCRETIZATION ---
TE = 60.0             # Time step (seconds)
ZC = 0.0127           # Half thickness of composite (m)
ZT = 0.01             # Tooling thickness (m)

NZ_C = 3              # Number of composite nodes
NZ_T = 4              # Number of tooling nodes
DZ_C = ZC / NZ_C      # Spatial step (composite)
DZ_T = ZT / NZ_T      # Spatial step (tooling)

# Fourier numbers (Heat transfer multipliers)
FC = (BETA_C * TE) / (DZ_C**2)
FT = (BETA_T * TE) / (DZ_T**2)

# --- 3. OPERATING LIMITS ---
TA_MIN = 10.0         # Min autoclave temp (°C)
TA_MAX = 130.0        # Max autoclave temp (°C)
TA_RATE_MAX = 4.0     # Max ramp rate (°C/min)
GRADIENT_MAX = 10.0   # Max allowed diff between Center and Surface (°C)