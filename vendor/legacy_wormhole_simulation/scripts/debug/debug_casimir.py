#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('src')

from src.physics.exotic_matter import AdvancedCasimirExoticMatter
from src.physics.constants import HBAR, C, K_B, E_0, ELECTRON_MASS, ELEMENTARY_CHARGE

# Create the same Casimir matter as in test
casimir = AdvancedCasimirExoticMatter(
    plate_separation=1e-6,  # 1 micron
    temperature=300,        # Room temperature
    experimental_calibration='decca_2003'
)

# Test coordinates from validation test
coordinates = (0.0, 1000.0, np.pi/2, 0.0)  # t, r, theta, phi

print("Debug Casimir Energy Density Calculation")
print("=" * 50)
print(f"Plate separation: {casimir.a:.2e} m")
print(f"Base energy density: {casimir.rho_base:.6e} J/m³")

# Debug the base calculation step by step
raw_base = -np.pi**2 * HBAR * C / (240 * casimir.a**4)
print(f"Raw base calculation (-pi^2*hbar*c/240a^4): {raw_base:.6e} J/m^3")

# Check experimental data (access as module constant)
from src.physics.exotic_matter import CASIMIR_EXPERIMENTAL_DATA
exp_data = CASIMIR_EXPERIMENTAL_DATA.get(casimir.experimental_calibration, {})
correction_factor = exp_data.get('force_coefficient', 1.0)
print(f"Experimental correction factor: {correction_factor}")

# Check temperature correction
lambda_T = HBAR * C / (K_B * casimir.T)
print(f"Thermal wavelength: {lambda_T:.6e} m")
print(f"Plate separation vs thermal wavelength: {casimir.a:.6e} / {lambda_T:.6e} = {casimir.a/lambda_T:.6e}")

if casimir.a < lambda_T:
    temp_correction = 1 + 45 * K_B * casimir.T / (4 * np.pi * HBAR * C / casimir.a)
    print(f"Temperature correction (high T limit): {temp_correction:.6f}")
else:
    temp_correction = 1 - np.exp(-2 * np.pi * casimir.a / lambda_T)
    print(f"Temperature correction (low T limit): {temp_correction:.6f}")

# Debug conductivity correction
sigma_conductivity = casimir.sigma_conductivity

omega_p = np.sqrt(sigma_conductivity * ELEMENTARY_CHARGE**2 / (E_0 * ELECTRON_MASS))
skin_depth = C / omega_p
conductivity_correction = 1 - 0.5 * (skin_depth / casimir.a)**0.5

print(f"Conductivity: {sigma_conductivity:.2e} S/m")
print(f"Plasma frequency: {omega_p:.2e} rad/s") 
print(f"Skin depth: {skin_depth:.2e} m")
print(f"Skin depth / plate separation: {skin_depth / casimir.a:.2e}")
print(f"Conductivity correction: {conductivity_correction:.6f}")

total_correction = correction_factor * conductivity_correction * temp_correction
print(f"Total correction factor: {total_correction:.6f}")
print(f"Expected base after corrections: {raw_base * total_correction:.6e}")
print(f"Actual rho_base: {casimir.rho_base:.6e}")
print(f"Ratio actual/expected: {casimir.rho_base / (raw_base * total_correction):.2f}")
print(f"Test coordinates: {coordinates}")

# Debug the spatial profile calculation (using new method)
t, r, theta, phi = coordinates
throat_radius = 1000.0  # Fixed throat radius as in updated method
scale_length = throat_radius * 0.5
print(f"Throat radius used: {throat_radius:.6e} m")
print(f"Scale length: {scale_length:.6e} m")
print(f"Distance from throat (r - throat_radius): {r - throat_radius:.6e} m")

# Calculate profile components
distance_squared = (r - throat_radius)**2
denominator = 2 * scale_length**2
exponent = -distance_squared / denominator

print(f"Distance squared: {distance_squared:.6e}")
print(f"Denominator (2*throat_radius^2): {denominator:.6e}")
print(f"Exponent: {exponent:.6e}")

if exponent < -100:
    print("WARNING: Exponent is very negative, exp() will be effectively 0")

profile = np.exp(exponent)
print(f"Profile factor: {profile:.6e}")

final_density = casimir.rho_base * profile
print(f"Final energy density: {final_density:.6e} J/m³")

# Theoretical comparison
theoretical_rho = -np.pi**2 * HBAR * C / (720 * casimir.a**4)
print(f"Theoretical (720 formula): {theoretical_rho:.6e} J/m³")
print(f"Base density vs theoretical: {casimir.rho_base/theoretical_rho:.3f}")