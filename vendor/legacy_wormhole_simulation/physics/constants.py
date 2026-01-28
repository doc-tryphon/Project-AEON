"""
Physical constants and units for quantum wormhole simulation.

This module defines fundamental physical constants in SI units and 
provides unit conversion utilities for relativistic calculations.
"""

import numpy as np
from typing import Dict, Any

class PhysicsConstants:
    """Container class for physical constants used in simulations."""
    # Fundamental constants (SI units)
    C = 299792458.0                    # Speed of light [m/s]
    G = 6.67430e-11                    # Gravitational constant [m^3 kg^-1 s^-2]
    HBAR = 1.054571817e-34            # Reduced Planck constant [J⋅s]
    H = 2 * np.pi * HBAR              # Planck constant [J⋅s]
    K_B = 1.380649e-23                # Boltzmann constant [J/K]
    E_0 = 8.8541878128e-12            # Vacuum permittivity [F/m]
    MU_0 = 1.25663706212e-6           # Vacuum permeability [H/m]
    
    # Derived constants
    PLANCK_LENGTH = np.sqrt(HBAR * G / C**3)    # [m]
    PLANCK_TIME = PLANCK_LENGTH / C             # [s]
    PLANCK_MASS = np.sqrt(HBAR * C / G)         # [kg]
    PLANCK_ENERGY = PLANCK_MASS * C**2          # [J]
    PLANCK_TEMPERATURE = PLANCK_ENERGY / K_B    # [K]

# Fundamental constants (SI units)
C = 299792458.0                    # Speed of light [m/s]
G = 6.67430e-11                    # Gravitational constant [m^3 kg^-1 s^-2]
HBAR = 1.054571817e-34            # Reduced Planck constant [J⋅s]
H = 2 * np.pi * HBAR              # Planck constant [J⋅s]
K_B = 1.380649e-23                # Boltzmann constant [J/K]
E_0 = 8.8541878128e-12            # Vacuum permittivity [F/m]
MU_0 = 1.25663706212e-6           # Vacuum permeability [H/m]

# Derived constants
PLANCK_LENGTH = np.sqrt(HBAR * G / C**3)    # [m]
PLANCK_TIME = PLANCK_LENGTH / C             # [s]
PLANCK_MASS = np.sqrt(HBAR * C / G)         # [kg]
PLANCK_ENERGY = PLANCK_MASS * C**2          # [J]
PLANCK_TEMPERATURE = PLANCK_ENERGY / K_B    # [K]

# Particle physics constants
ELECTRON_MASS = 9.1093837015e-31    # [kg]
PROTON_MASS = 1.67262192369e-27     # [kg]
NEUTRON_MASS = 1.67492749804e-27    # [kg]
ELEMENTARY_CHARGE = 1.602176634e-19 # [C]

# Cosmological constants
HUBBLE_CONSTANT = 2.2e-18          # Hubble constant [s^-1] (approx 70 km/s/Mpc)
CRITICAL_DENSITY = 3 * HUBBLE_CONSTANT**2 / (8 * np.pi * G)  # [kg/m^3]

# Conversion factors
EV_TO_JOULES = ELEMENTARY_CHARGE   # [J/eV]
JOULES_TO_EV = 1.0 / EV_TO_JOULES  # [eV/J]
SOLAR_MASS = 1.98847e30            # [kg]
PARSEC = 3.0857e16                 # [m]
LIGHT_YEAR = C * 365.25 * 24 * 3600 # [m]

# Natural units (c = ℏ = G = 1)
class NaturalUnits:
    """Natural units system for general relativity calculations."""
    
    @staticmethod
    def length_to_natural(length_m: float) -> float:
        """Convert length from meters to natural units."""
        return length_m / PLANCK_LENGTH
    
    @staticmethod
    def time_to_natural(time_s: float) -> float:
        """Convert time from seconds to natural units."""
        return time_s / PLANCK_TIME
    
    @staticmethod
    def mass_to_natural(mass_kg: float) -> float:
        """Convert mass from kg to natural units."""
        return mass_kg / PLANCK_MASS
    
    @staticmethod
    def energy_to_natural(energy_j: float) -> float:
        """Convert energy from Joules to natural units."""
        return energy_j / PLANCK_ENERGY
    
    @staticmethod
    def natural_to_length(length_natural: float) -> float:
        """Convert length from natural units to meters."""
        return length_natural * PLANCK_LENGTH
    
    @staticmethod
    def natural_to_time(time_natural: float) -> float:
        """Convert time from natural units to seconds."""
        return time_natural * PLANCK_TIME
    
    @staticmethod
    def natural_to_mass(mass_natural: float) -> float:
        """Convert mass from natural units to kg."""
        return mass_natural * PLANCK_MASS
    
    @staticmethod
    def natural_to_energy(energy_natural: float) -> float:
        """Convert energy from natural units to Joules."""
        return energy_natural * PLANCK_ENERGY

# Geometrized units (G = c = 1)
class GeometrizedUnits:
    """Geometrized units for spacetime calculations."""
    
    LENGTH_UNIT = G / C**2  # [m/kg] - 1 kg of mass = G/c² meters
    TIME_UNIT = G / C**3    # [s/kg] - 1 kg of mass = G/c³ seconds
    
    @staticmethod
    def mass_to_length(mass_kg: float) -> float:
        """Convert mass to equivalent length in geometrized units."""
        return mass_kg * G / C**2
    
    @staticmethod
    def mass_to_time(mass_kg: float) -> float:
        """Convert mass to equivalent time in geometrized units."""
        return mass_kg * G / C**3
    
    @staticmethod
    def schwarzschild_radius(mass_kg: float) -> float:
        """Calculate Schwarzschild radius for given mass."""
        return 2 * G * mass_kg / C**2

# Wormhole-specific constants
class WormholeConstants:
    """Constants specific to wormhole physics."""
    
    # Minimum throat radius for traversable wormholes (order of magnitude)
    MIN_THROAT_RADIUS = 1e3  # [m] - roughly 1 km minimum
    
    # Exotic matter energy density scale
    EXOTIC_ENERGY_SCALE = -CRITICAL_DENSITY  # Negative critical density
    
    # Typical wormhole mass scale
    TYPICAL_MASS = SOLAR_MASS  # Solar mass scale
    
    # Traversal time scales
    MIN_TRAVERSAL_TIME = MIN_THROAT_RADIUS / C  # Light crossing time
    
    @staticmethod
    def throat_area(radius: float) -> float:
        """Calculate throat area for given radius."""
        return 4 * np.pi * radius**2
    
    @staticmethod
    def exotic_matter_required(throat_radius: float) -> float:
        """Estimate exotic matter energy density required."""
        # Rough estimate based on Morris-Thorne wormhole
        return -C**4 / (16 * np.pi * G**2 * throat_radius**2)

# Unit conversion utilities
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between different unit systems."""
    
    conversion_table = {
        ('m', 'planck_length'): lambda x: x / PLANCK_LENGTH,
        ('planck_length', 'm'): lambda x: x * PLANCK_LENGTH,
        ('kg', 'solar_mass'): lambda x: x / SOLAR_MASS,
        ('solar_mass', 'kg'): lambda x: x * SOLAR_MASS,
        ('j', 'ev'): lambda x: x * JOULES_TO_EV,
        ('ev', 'j'): lambda x: x * EV_TO_JOULES,
        ('m', 'ly'): lambda x: x / LIGHT_YEAR,
        ('ly', 'm'): lambda x: x * LIGHT_YEAR,
        ('m', 'pc'): lambda x: x / PARSEC,
        ('pc', 'm'): lambda x: x * PARSEC,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversion_table:
        return conversion_table[key](value)
    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

# Physical scales dictionary
PHYSICAL_SCALES = {
    'quantum': {
        'length': PLANCK_LENGTH,
        'time': PLANCK_TIME,
        'mass': PLANCK_MASS,
        'energy': PLANCK_ENERGY
    },
    'atomic': {
        'length': 1e-10,  # Angstrom
        'time': 1e-15,    # Femtosecond
        'mass': ELECTRON_MASS,
        'energy': EV_TO_JOULES
    },
    'stellar': {
        'length': 1e9,    # Gigameter
        'time': 1e6,      # Million seconds
        'mass': SOLAR_MASS,
        'energy': 1e42    # Joules (supernova scale)
    },
    'cosmological': {
        'length': PARSEC,
        'time': 1e17,     # ~3 billion years
        'mass': 1e12 * SOLAR_MASS,  # Galaxy mass
        'energy': 1e51    # Joules (gamma ray burst)
    }
}

def get_scale(scale_name: str) -> Dict[str, float]:
    """Get physical scales for different regimes."""
    if scale_name not in PHYSICAL_SCALES:
        raise ValueError(f"Scale '{scale_name}' not recognized. "
                        f"Available: {list(PHYSICAL_SCALES.keys())}")
    return PHYSICAL_SCALES[scale_name]

# Numerical precision constants
NUMERICAL_PRECISION = {
    'float_precision': np.finfo(np.float64).eps,
    'integration_tolerance': 1e-12,
    'convergence_threshold': 1e-10,
    'max_iterations': 10000
}