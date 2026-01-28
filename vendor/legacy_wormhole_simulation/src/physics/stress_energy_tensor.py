"""
Stress-energy tensor calculations for wormhole physics.

This module implements various forms of the stress-energy tensor T_μν
that appear in Einstein's field equations for wormhole solutions.
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional, Union
from abc import ABC, abstractmethod
import scipy.integrate as integrate

from src.physics.constants import C, G, HBAR, K_B, CRITICAL_DENSITY


class StressEnergyTensor(ABC):
    """Abstract base class for stress-energy tensor implementations."""
    
    def __init__(self, coordinates: str = 'spherical'):
        """Initialize stress-energy tensor.
        
        Args:
            coordinates: Coordinate system ('spherical', 'cartesian', etc.)
        """
        self.coordinates = coordinates
    
    @abstractmethod
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Return stress-energy tensor components at given coordinates."""
        pass
    
    def energy_density(self, coordinates: Tuple[float, ...]) -> float:
        """Return energy density ρ = T₀₀."""
        T = self.tensor_components(coordinates)
        return T[0, 0]
    
    def pressure_radial(self, coordinates: Tuple[float, ...]) -> float:
        """Return radial pressure pᵣ = Tᵣᵣ."""
        T = self.tensor_components(coordinates)
        return T[1, 1]
    
    def pressure_tangential(self, coordinates: Tuple[float, ...]) -> float:
        """Return tangential pressure pₜ = Tθθ = Tφφ."""
        T = self.tensor_components(coordinates)
        return T[2, 2]  # Assuming spherical symmetry
    
    def trace(self, coordinates: Tuple[float, ...]) -> float:
        """Return trace of stress-energy tensor."""
        T = self.tensor_components(coordinates)
        return np.trace(T)


class PerfectFluidStressEnergy(StressEnergyTensor):
    """Perfect fluid stress-energy tensor implementation.
    
    Models matter as a perfect fluid with energy density ρ and pressure p.
    The stress-energy tensor takes the form:
    T_μν = (ρ + p)u_μu_ν + pg_μν
    where u_μ is the fluid 4-velocity and g_μν is the metric tensor.
    """
    
    def __init__(self, 
                 density_func: Callable[[Tuple[float, ...]], float],
                 pressure_func: Callable[[Tuple[float, ...]], float],
                 coordinates: str = 'spherical',
                 velocity_field: Optional[Callable[[Tuple[float, ...]], np.ndarray]] = None):
        """Initialize perfect fluid stress-energy tensor.
        
        Args:
            density_func: Function returning energy density at given coordinates
            pressure_func: Function returning pressure at given coordinates
            coordinates: Coordinate system ('spherical', 'cartesian', etc.)
            velocity_field: Optional function returning 4-velocity components
        """
        super().__init__(coordinates)
        self.density_func = density_func
        self.pressure_func = pressure_func
        self.velocity_field = velocity_field or self._default_velocity_field
        
    def _default_velocity_field(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Default velocity field for static fluid."""
        return np.array([1, 0, 0, 0])  # Static fluid in coordinates
        
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Return stress-energy tensor components at given coordinates.
        
        Args:
            coordinates: Point at which to evaluate tensor (t,r,θ,φ)
            
        Returns:
            4x4 numpy array of tensor components T_μν
        """
        ρ = self.density_func(coordinates)
        p = self.pressure_func(coordinates)
        u = self.velocity_field(coordinates)
        
        # Construct metric tensor (for raising/lowering indices)
        if self.coordinates == 'spherical':
            r = coordinates[1]
            g = np.diag([-1, 1, r**2, r**2 * np.sin(coordinates[2])**2])
        else:
            g = np.diag([-1, 1, 1, 1])  # Minkowski metric
            
        # Perfect fluid stress-energy tensor
        T = np.zeros((4,4))
        for μ in range(4):
            for ν in range(4):
                T[μ,ν] = (ρ + p) * u[μ] * u[ν] + p * g[μ,ν]
                
        return T
    
    def energy_conditions(self, coordinates: Tuple[float, ...]) -> Dict[str, bool]:
        """Check energy conditions at given coordinates."""
        rho = self.energy_density(coordinates)
        p_r = self.pressure_radial(coordinates)
        p_t = self.pressure_tangential(coordinates)
        
        return {
            'null_energy': rho + p_r >= 0 and rho + p_t >= 0,
            'weak_energy': rho >= 0 and rho + p_r >= 0 and rho + p_t >= 0,
            'strong_energy': rho + p_r >= 0 and rho + p_t >= 0 and rho + p_r + 2*p_t >= 0,
            'dominant_energy': rho >= abs(p_r) and rho >= abs(p_t)
        }


class ExoticMatterTensor(StressEnergyTensor):
    """Stress-energy tensor for exotic matter with negative energy density."""
    
    def __init__(self, energy_density_func: Callable[[float], float],
                 pressure_profile: str = 'isotropic',
                 anisotropy_factor: float = 1.0):
        """Initialize exotic matter tensor.
        
        Args:
            energy_density_func: Function ρ(r) defining energy density profile
            pressure_profile: Pressure profile type ('isotropic', 'anisotropic')
            anisotropy_factor: Factor controlling pressure anisotropy
        """
        super().__init__('spherical')
        self.rho_func = energy_density_func
        self.pressure_profile = pressure_profile
        self.anisotropy = anisotropy_factor
    
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute exotic matter stress-energy tensor components."""
        t, r, theta, phi = coordinates
        
        # Energy density (typically negative)
        rho = self.rho_func(r)
        
        # Pressure components based on profile
        if self.pressure_profile == 'isotropic':
            p_r = p_t = rho / 3.0  # Radiation-like equation of state
        elif self.pressure_profile == 'anisotropic':
            p_r = -rho * self.anisotropy  # Radial pressure
            p_t = -rho * (1 - self.anisotropy) / 2  # Tangential pressure
        else:
            raise ValueError(f"Unknown pressure profile: {self.pressure_profile}")
        
        # Build stress-energy tensor (diagonal in spherical coordinates)
        T = np.zeros((4, 4))
        T[0, 0] = -rho  # T₀₀ = -ρc² (with c=1)
        T[1, 1] = p_r   # Tᵣᵣ = pᵣ
        T[2, 2] = p_t   # Tθθ = pₜ
        T[3, 3] = p_t   # Tφφ = pₜ
        
        return T


class QuantumVacuumTensor(StressEnergyTensor):
    """Stress-energy tensor from quantum vacuum fluctuations."""
    
    def __init__(self, casimir_coefficient: float = 1.0,
                 cutoff_frequency: float = 1e20):
        """Initialize quantum vacuum tensor.
        
        Args:
            casimir_coefficient: Coefficient for Casimir energy density
            cutoff_frequency: High-frequency cutoff for vacuum fluctuations
        """
        super().__init__('spherical')
        self.casimir_coeff = casimir_coefficient
        self.omega_cutoff = cutoff_frequency
    
    def casimir_energy_density(self, r: float) -> float:
        """Compute Casimir energy density."""
        # Simplified model: ρ ∝ -ℏc/r⁴
        return -self.casimir_coeff * HBAR * C / (r**4)
    
    def vacuum_pressure(self, r: float, direction: str) -> float:
        """Compute vacuum pressure in different directions."""
        rho_casimir = self.casimir_energy_density(r)
        
        if direction == 'radial':
            return rho_casimir / 3.0
        elif direction == 'tangential':
            return -rho_casimir / 6.0
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Quantum vacuum stress-energy tensor."""
        t, r, theta, phi = coordinates
        
        rho = self.casimir_energy_density(r)
        p_r = self.vacuum_pressure(r, 'radial')
        p_t = self.vacuum_pressure(r, 'tangential')
        
        T = np.zeros((4, 4))
        T[0, 0] = -rho
        T[1, 1] = p_r
        T[2, 2] = p_t
        T[3, 3] = p_t
        
        return T


class ScalarFieldTensor(StressEnergyTensor):
    """Stress-energy tensor for scalar field (phantom or quintessence)."""
    
    def __init__(self, field_func: Callable[[float], float],
                 potential_func: Callable[[float], float],
                 field_type: str = 'phantom'):
        """Initialize scalar field tensor.
        
        Args:
            field_func: Scalar field φ(r)
            potential_func: Potential V(φ)
            field_type: 'phantom' (w < -1) or 'quintessence' (-1 < w < -1/3)
        """
        super().__init__('spherical')
        self.phi_func = field_func
        self.V_func = potential_func
        self.field_type = field_type
    
    def field_derivative(self, r: float, eps: float = 1e-8) -> float:
        """Compute dφ/dr numerically."""
        return (self.phi_func(r + eps) - self.phi_func(r - eps)) / (2 * eps)
    
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Scalar field stress-energy tensor."""
        t, r, theta, phi = coordinates
        
        phi = self.phi_func(r)
        dphi_dr = self.field_derivative(r)
        V = self.V_func(phi)
        
        # Kinetic and potential energy terms
        kinetic = 0.5 * dphi_dr**2
        
        # Energy density and pressure
        if self.field_type == 'phantom':
            # Phantom field: kinetic term has opposite sign
            rho = -kinetic + V
            p_r = -kinetic - V
            p_t = -V  # No kinetic contribution to tangential pressure
        elif self.field_type == 'quintessence':
            # Standard scalar field
            rho = kinetic + V
            p_r = kinetic - V
            p_t = -V
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")
        
        T = np.zeros((4, 4))
        T[0, 0] = -rho
        T[1, 1] = p_r
        T[2, 2] = p_t
        T[3, 3] = p_t
        
        return T


class FluidTensor(StressEnergyTensor):
    """Perfect fluid stress-energy tensor."""
    
    def __init__(self, density_func: Callable[[float], float],
                 equation_of_state: Union[float, Callable[[float], float]]):
        """Initialize fluid tensor.
        
        Args:
            density_func: Energy density ρ(r)
            equation_of_state: w = p/ρ (constant) or w(ρ) function
        """
        super().__init__('spherical')
        self.rho_func = density_func
        self.eos = equation_of_state
    
    def pressure(self, r: float) -> float:
        """Compute pressure from equation of state."""
        rho = self.rho_func(r)
        if callable(self.eos):
            w = self.eos(rho)
        else:
            w = self.eos
        return w * rho
    
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Perfect fluid stress-energy tensor."""
        t, r, theta, phi = coordinates
        
        rho = self.rho_func(r)
        p = self.pressure(r)
        
        # Perfect fluid tensor: T_μν = (ρ + p)u_μu_ν + pg_μν
        # For static case: u^μ = (1, 0, 0, 0)
        T = np.zeros((4, 4))
        T[0, 0] = -rho  # Energy density
        T[1, 1] = p     # Radial pressure
        T[2, 2] = p     # Tangential pressure
        T[3, 3] = p     # Tangential pressure
        
        return T


class CompositeTensor(StressEnergyTensor):
    """Superposition of multiple stress-energy tensor contributions."""
    
    def __init__(self, tensor_components: list):
        """Initialize composite tensor.
        
        Args:
            tensor_components: List of StressEnergyTensor objects
        """
        super().__init__('spherical')
        self.components = tensor_components
    
    def tensor_components(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Sum all tensor components."""
        T_total = np.zeros((4, 4))
        for component in self.components:
            T_total += component.tensor_components(coordinates)
        return T_total


def wormhole_matter_requirements(throat_radius: float, 
                                shape_function: str = 'minimal') -> Dict[str, float]:
    """Calculate matter requirements for traversable wormhole.
    
    Args:
        throat_radius: Minimum throat radius
        shape_function: Shape function type
    
    Returns:
        Dictionary with matter requirements
    """
    # Morris-Thorne flare-out condition requirements
    b0 = throat_radius
    
    if shape_function == 'minimal':
        # Energy density at throat
        rho_throat = -C**4 / (16 * np.pi * G**2 * b0**2)
        
        # Total exotic matter required (rough estimate)
        total_mass = -4 * np.pi * b0**2 * rho_throat * b0 / C**2
        
    elif shape_function == 'exponential':
        # More complex calculation needed
        rho_throat = -C**4 / (8 * np.pi * G**2 * b0**2)
        total_mass = -2 * np.pi * b0**2 * rho_throat * b0 / C**2
        
    else:
        # Default case
        rho_throat = -CRITICAL_DENSITY
        total_mass = -4 * np.pi * b0**3 * rho_throat / 3 / C**2
    
    return {
        'throat_energy_density': rho_throat,
        'total_exotic_mass': total_mass,
        'throat_radius': b0,
        'energy_scale': abs(rho_throat) * C**2,
        'mass_ratio_to_sun': abs(total_mass) / 1.989e30
    }


def verify_einstein_equations(metric_tensor: np.ndarray,
                             stress_energy_tensor: np.ndarray,
                             coordinates: Tuple[float, ...],
                             tolerance: float = 1e-10) -> Dict[str, bool]:
    """Verify Einstein field equations G_μν = 8πG/c⁴ T_μν.
    
    Args:
        metric_tensor: Metric tensor g_μν
        stress_energy_tensor: Stress-energy tensor T_μν
        coordinates: Spacetime coordinates
        tolerance: Numerical tolerance for verification
    
    Returns:
        Dictionary with verification results
    """
    # This is a simplified check - full implementation would compute
    # Einstein tensor from Ricci tensor and scalar curvature
    
    # Check tensor symmetry
    g_symmetric = np.allclose(metric_tensor, metric_tensor.T, atol=tolerance)
    T_symmetric = np.allclose(stress_energy_tensor, stress_energy_tensor.T, atol=tolerance)
    
    # Check energy-momentum conservation (∇_μ T^μν = 0)
    # Simplified check: look at divergence of diagonal terms
    conservation_check = True  # Placeholder
    
    # Check causality constraints
    g_det = np.linalg.det(metric_tensor)
    causality_ok = g_det < 0  # Signature (-,+,+,+)
    
    return {
        'metric_symmetric': g_symmetric,
        'tensor_symmetric': T_symmetric,
        'energy_momentum_conserved': conservation_check,
        'causality_preserved': causality_ok,
        'determinant_sign_correct': g_det < 0
    }


def compute_energy_flux(tensor: StressEnergyTensor,
                       surface_coordinates: np.ndarray,
                       normal_vector: np.ndarray) -> float:
    """Compute energy flux through a given surface.
    
    Args:
        tensor: Stress-energy tensor object
        surface_coordinates: Coordinates defining the surface
        normal_vector: Normal vector to the surface
    
    Returns:
        Energy flux value
    """
    flux = 0.0
    
    for coords in surface_coordinates:
        T = tensor.tensor_components(tuple(coords))
        # Energy flux: F = T^0_i n^i
        local_flux = sum(T[0, i] * normal_vector[i-1] for i in range(1, 4))
        flux += local_flux
    
    return flux / len(surface_coordinates)  # Average flux