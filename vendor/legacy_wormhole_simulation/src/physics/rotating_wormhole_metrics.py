"""
Rotating Wormhole Metrics - Kerr-like Geometries.

This module implements rotating wormhole spacetime metrics, including
Kerr-like geometries and extended metrics with angular momentum effects
for more realistic traversable wormhole scenarios.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
import scipy.integrate as integrate
import scipy.optimize as optimize
from dataclasses import dataclass
import logging

from src.physics.spacetime_metrics import SpacetimeMetric
from src.physics.constants import G, C, HBAR

logger = logging.getLogger(__name__)


@dataclass
class RotationParameters:
    """Parameters for rotating wormhole geometry."""
    
    angular_momentum: float  # Total angular momentum (J⋅s)
    spin_parameter: float    # Dimensionless spin parameter a = J/(Mc)
    rotation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Rotation axis (normalized)
    frame_dragging_coefficient: float = 1.0  # Frame-dragging strength
    ergosphere_enabled: bool = True  # Whether to include ergosphere effects
    
    def __post_init__(self):
        """Validate and normalize rotation parameters."""
        # Normalize rotation axis
        axis = np.array(self.rotation_axis)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            self.rotation_axis = tuple(axis / axis_norm)
        else:
            self.rotation_axis = (0.0, 0.0, 1.0)
        
        # Validate spin parameter (should be < 1 for traversable wormholes)
        if abs(self.spin_parameter) >= 1.0:
            logger.warning(f"High spin parameter {self.spin_parameter} may cause stability issues")


class RotatingWormholeMetric(SpacetimeMetric):
    """Base class for rotating wormhole metrics."""
    
    def __init__(self, throat_radius: float, mass: float, 
                 rotation_params: RotationParameters):
        """Initialize rotating wormhole metric.
        
        Args:
            throat_radius: Minimum radius of wormhole throat
            mass: Total mass-energy of wormhole
            rotation_params: Rotation parameters
        """
        super().__init__()
        self.throat_radius = throat_radius
        self.mass = mass
        self.rotation_params = rotation_params
        
        # Derived parameters
        self.schwarzschild_radius = 2 * G * mass / C**2
        self.angular_momentum = rotation_params.angular_momentum
        self.spin_parameter = rotation_params.spin_parameter
        
        # Check physical consistency
        if self.throat_radius < self.schwarzschild_radius:
            logger.warning("Throat radius smaller than Schwarzschild radius - "
                         "may indicate unphysical configuration")
    
    @abstractmethod
    def shape_function(self, r: float) -> float:
        """Shape function b(r) for the wormhole geometry."""
        pass
    
    @abstractmethod
    def lapse_function(self, r: float, theta: float = np.pi/2) -> float:
        """Lapse function determining time dilation effects."""
        pass
    
    @abstractmethod
    def frame_dragging_function(self, r: float, theta: float = np.pi/2) -> float:
        """Frame-dragging function ω(r,θ) for rotational effects."""
        pass


class KerrLikeWormhole(RotatingWormholeMetric):
    """Kerr-like rotating wormhole with frame-dragging effects."""
    
    def __init__(self, throat_radius: float, mass: float,
                 rotation_params: RotationParameters,
                 wormhole_parameter: float = 0.1):
        """Initialize Kerr-like wormhole.
        
        Args:
            throat_radius: Throat radius
            mass: Mass parameter
            rotation_params: Rotation parameters  
            wormhole_parameter: Parameter controlling wormhole properties
        """
        super().__init__(throat_radius, mass, rotation_params)
        self.wormhole_parameter = wormhole_parameter
        
        # Compute characteristic length scales
        self.rotation_length = abs(self.angular_momentum) / (mass * C) if mass > 0 else 0
        self.effective_horizon = self._compute_effective_horizon()
        
        logger.info(f"Initialized Kerr-like wormhole: throat={throat_radius:.1f}m, "
                   f"mass={mass:.2e}kg, spin={self.spin_parameter:.3f}")
    
    def shape_function(self, r: float) -> float:
        """Shape function for Kerr-like wormhole.
        
        The shape function is modified from Morris-Thorne to include
        rotation effects and prevent horizon formation.
        """
        b0 = self.throat_radius
        
        # Handle r=0 case
        if r <= 0:
            return b0
        
        # Base Morris-Thorne shape function
        base_shape = b0 * (1 + (b0/r)**2)
        
        # Rotation modification to prevent horizon formation
        # Reduce effective shape function near rotation length scales
        rotation_correction = 1 + self.wormhole_parameter * (self.rotation_length / (r + self.rotation_length))
        
        # Ensure shape function properties: b(r) < r and b'(b0) < 1
        modified_shape = base_shape / rotation_correction
        
        # Enforce physical constraints
        if r <= b0:
            return b0  # At throat
        else:
            return min(modified_shape, 0.99 * r)  # Prevent b(r) >= r
    
    def lapse_function(self, r: float, theta: float = np.pi/2) -> float:
        """Lapse function with rotation effects.
        
        Modified to include centrifugal effects and frame-dragging.
        """
        b_r = self.shape_function(r)
        
        # Base redshift function (Morris-Thorne style)
        if r <= self.throat_radius:
            base_lapse = 1.0
        else:
            # Avoid horizon formation
            base_lapse = np.exp(-self.wormhole_parameter * b_r / r)
        
        # Rotation corrections
        cos_theta = np.cos(theta)
        
        # Centrifugal potential correction
        if r > 0:
            centrifugal_term = (self.rotation_length / r)**2 * (1 - cos_theta**2)
            rotation_lapse = 1 - 0.5 * centrifugal_term
        else:
            rotation_lapse = 1.0
        
        return base_lapse * max(rotation_lapse, 0.1)  # Prevent negative lapse
    
    def frame_dragging_function(self, r: float, theta: float = np.pi/2) -> float:
        """Frame-dragging function ω(r,θ).
        
        Determines how much the rotating wormhole drags inertial frames.
        """
        if r <= 0:
            return 0.0
        
        # Frame-dragging strength based on angular momentum
        rotation_strength = self.rotation_params.frame_dragging_coefficient
        
        # Distance-dependent falloff
        distance_factor = (self.throat_radius / r)**3
        
        # Angular dependence (stronger near equatorial plane)
        sin_theta = np.sin(theta)
        angular_factor = sin_theta**2
        
        # Combine factors
        frame_dragging = rotation_strength * distance_factor * angular_factor
        
        # Sign depends on rotation direction
        if self.angular_momentum < 0:
            frame_dragging *= -1
        
        return frame_dragging
    
    def metric_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute metric tensor for Kerr-like wormhole.
        
        Uses Boyer-Lindquist-like coordinates with wormhole modifications.
        """
        t, r, theta, phi = coordinates
        
        # Compute metric functions
        b_r = self.shape_function(r)
        lapse = self.lapse_function(r, theta)
        omega = self.frame_dragging_function(r, theta)
        
        # Metric components
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # g_tt: time-time component
        g_tt = -lapse**2 + omega**2 * (r**2 + self.rotation_length**2 * cos_theta**2) * sin_theta**2
        
        # g_rr: radial component (modified for wormhole)
        g_rr = 1 / (1 - b_r/r) if r > b_r else 1.0
        
        # g_θθ: theta component
        g_theta_theta = r**2 + self.rotation_length**2 * cos_theta**2
        
        # g_φφ: phi component  
        g_phi_phi = (r**2 + self.rotation_length**2 * cos_theta**2) * sin_theta**2
        
        # g_tφ: time-phi cross term (frame-dragging)
        g_t_phi = -omega * (r**2 + self.rotation_length**2 * cos_theta**2) * sin_theta**2
        
        # Construct metric tensor
        metric = np.zeros((4, 4))
        metric[0, 0] = g_tt
        metric[1, 1] = g_rr
        metric[2, 2] = g_theta_theta
        metric[3, 3] = g_phi_phi
        metric[0, 3] = metric[3, 0] = g_t_phi
        
        return metric
    
    def christoffel_symbols(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Christoffel symbols for the rotating metric.
        
        Uses numerical differentiation of the metric tensor.
        """
        t, r, theta, phi = coordinates
        eps = 1e-8
        
        # Get metric at point
        g = self.metric_tensor(coordinates)
        g_inv = np.linalg.inv(g)
        
        # Initialize Christoffel symbols
        gamma = np.zeros((4, 4, 4))
        
        # Compute via finite differences
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    
                    # Partial derivatives of metric
                    coords_plus = list(coordinates)
                    coords_minus = list(coordinates)
                    
                    coords_plus[mu] += eps
                    coords_minus[mu] -= eps
                    
                    g_plus = self.metric_tensor(tuple(coords_plus))
                    g_minus = self.metric_tensor(tuple(coords_minus))
                    
                    dg_dmu = (g_plus - g_minus) / (2 * eps)
                    
                    # Christoffel symbol computation
                    for nu in range(4):
                        gamma[mu, alpha, beta] += 0.5 * g_inv[mu, nu] * (
                            dg_dmu[nu, alpha] + dg_dmu[nu, beta] - 
                            self._metric_derivative(coordinates, nu, alpha, beta)
                        )
        
        return gamma
    
    def metric_tt(self, r: float, theta: float) -> float:
        """Time-time component of metric tensor."""
        lapse = self.lapse_function(r, theta)
        omega = self.frame_dragging_function(r, theta)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return -lapse**2 + omega**2 * (r**2 + self.rotation_length**2 * cos_theta**2) * sin_theta**2
    
    def metric_rr(self, r: float, theta: float) -> float:
        """Radial component of metric tensor."""
        b_r = self.shape_function(r)
        return 1 / (1 - b_r/r) if r > b_r else 1.0
    
    def metric_t_phi(self, r: float, theta: float) -> float:
        """Time-phi cross component (frame-dragging term)."""
        omega = self.frame_dragging_function(r, theta)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return -omega * (r**2 + self.rotation_length**2 * cos_theta**2) * sin_theta**2
    
    def ergosphere_radius(self, theta: float) -> float:
        """Radius of ergosphere at given theta angle."""
        return self.ergosphere_boundary(theta)
    
    def is_inside_ergosphere(self, r: float, theta: float) -> bool:
        """Check if point is inside ergosphere."""
        try:
            ergo_r = self.ergosphere_radius(theta)
            return r < ergo_r
        except:
            return False
    
    def analyze_stability(self) -> bool:
        """Analyze wormhole stability based on energy conditions."""
        try:
            # Check if effective horizon exists (bad for traversability)
            if np.isfinite(self.effective_horizon):
                return False
            
            # Check energy condition violations at throat
            violation = self.energy_condition_violation(self.throat_radius, np.pi/2)
            
            # Should have moderate exotic matter (not too extreme)
            return -1e-2 < violation < 0
        except:
            return False
    
    def energy_condition_violation(self, r: float, theta: float) -> float:
        """Compute energy condition violation at given point."""
        # Simplified estimate based on curvature
        b_r = self.shape_function(r)
        lapse = self.lapse_function(r, theta)
        
        # Approximate stress-energy components
        rho = -b_r / (8 * np.pi * G * r**3)  # Energy density
        p_r = rho * lapse**2  # Radial pressure
        
        # Null Energy Condition violation: ρ + p_r < 0
        nec_violation = rho + p_r
        
        return nec_violation
    
    def _metric_derivative(self, coordinates: Tuple[float, ...], 
                          component: int, alpha: int, beta: int) -> float:
        """Compute derivative of metric component."""
        eps = 1e-8
        coords_plus = list(coordinates)
        coords_minus = list(coordinates)
        
        coords_plus[component] += eps
        coords_minus[component] -= eps
        
        g_plus = self.metric_tensor(tuple(coords_plus))
        g_minus = self.metric_tensor(tuple(coords_minus))
        
        return (g_plus[alpha, beta] - g_minus[alpha, beta]) / (2 * eps)
    
    def _compute_effective_horizon(self) -> float:
        """Compute effective horizon radius (if any)."""
        
        # For rotating wormholes, check where g_tt = 0
        def lapse_equation(r):
            return self.lapse_function(r, np.pi/2)**2 - self.frame_dragging_function(r, np.pi/2)**2
        
        try:
            # Look for horizon between throat and 10x throat radius
            r_range = np.linspace(self.throat_radius, 10 * self.throat_radius, 1000)
            lapse_values = [lapse_equation(r) for r in r_range]
            
            # Find sign changes (horizon locations)
            sign_changes = []
            for i in range(len(lapse_values) - 1):
                if lapse_values[i] * lapse_values[i+1] < 0:
                    # Refine with root finding
                    try:
                        horizon = optimize.brentq(lapse_equation, r_range[i], r_range[i+1])
                        sign_changes.append(horizon)
                    except:
                        pass
            
            if sign_changes:
                return min(sign_changes)  # Innermost horizon
            else:
                return float('inf')  # No horizon
                
        except Exception as e:
            logger.warning(f"Could not compute effective horizon: {e}")
            return float('inf')
    
    def ergosphere_boundary(self, theta: float) -> float:
        """Compute ergosphere boundary at given theta.
        
        The ergosphere is where g_tt = 0.
        """
        if not self.rotation_params.ergosphere_enabled:
            return float('inf')
        
        def ergosphere_equation(r):
            lapse = self.lapse_function(r, theta)
            omega = self.frame_dragging_function(r, theta)
            sin_theta = np.sin(theta)
            
            # g_tt = 0 condition
            return -lapse**2 + omega**2 * (r**2 + self.rotation_length**2 * np.cos(theta)**2) * sin_theta**2
        
        try:
            # Search for ergosphere between throat and several throat radii
            r_max = 5 * self.throat_radius
            return optimize.brentq(ergosphere_equation, self.throat_radius, r_max)
        except:
            return float('inf')  # No ergosphere at this angle
    
    def angular_velocity(self, r: float, theta: float = np.pi/2) -> float:
        """Compute frame-dragging angular velocity Ω(r,θ).
        
        This is the angular velocity that a locally non-rotating observer
        appears to have relative to infinity.
        """
        omega = self.frame_dragging_function(r, theta)
        
        # Extract g_tφ and g_φφ from metric
        g = self.metric_tensor((0, r, theta, 0))
        g_t_phi = g[0, 3]
        g_phi_phi = g[3, 3]
        
        if abs(g_phi_phi) > 1e-10:
            return -g_t_phi / g_phi_phi
        else:
            return 0.0


class ExtendedRotatingWormhole(KerrLikeWormhole):
    """Extended rotating wormhole with additional exotic matter effects."""
    
    def __init__(self, throat_radius: float, mass: float,
                 rotation_params: RotationParameters,
                 exotic_matter_coupling: float = 0.1,
                 magnetic_dipole_moment: float = 0.0):
        """Initialize extended rotating wormhole.
        
        Args:
            throat_radius: Throat radius
            mass: Mass parameter
            rotation_params: Rotation parameters
            exotic_matter_coupling: Coupling strength to exotic matter
            magnetic_dipole_moment: Magnetic dipole moment
        """
        super().__init__(throat_radius, mass, rotation_params)
        self.exotic_coupling = exotic_matter_coupling
        self.magnetic_moment = magnetic_dipole_moment
        
        # Aliases for test compatibility
        self.exotic_matter_coupling = exotic_matter_coupling
        self.magnetic_dipole_moment = magnetic_dipole_moment
        
        logger.info(f"Initialized extended rotating wormhole with exotic coupling {exotic_matter_coupling}")
    
    def shape_function(self, r: float) -> float:
        """Enhanced shape function with exotic matter corrections."""
        base_shape = super().shape_function(r)
        
        # Exotic matter modification
        if r > self.throat_radius:
            exotic_correction = 1 + self.exotic_coupling * np.exp(-(r - self.throat_radius) / self.throat_radius)
            return base_shape * exotic_correction
        else:
            return base_shape
    
    def electromagnetic_field_strength(self, r: float, theta: float) -> Tuple[float, float]:
        """Compute electromagnetic field strength (E, B) for magnetic dipole."""
        
        if abs(self.magnetic_moment) < 1e-20:
            return 0.0, 0.0
        
        # Magnetic dipole field in rotating frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Radial magnetic field
        B_r = 2 * self.magnetic_moment * cos_theta / r**3
        
        # Theta magnetic field  
        B_theta = self.magnetic_moment * sin_theta / r**3
        
        # Electric field from rotation (magnetic dipole + rotation → electric field)
        omega = self.frame_dragging_function(r, theta)
        E_magnitude = abs(omega * self.magnetic_moment / r**2)
        
        return E_magnitude, np.sqrt(B_r**2 + B_theta**2)
    
    def electromagnetic_correction(self, r: float, theta: float) -> float:
        """Compute electromagnetic correction to metric."""
        if abs(self.magnetic_moment) < 1e-20:
            return 0.0
        
        # Simple electromagnetic stress contribution
        E_field, B_field = self.electromagnetic_field_strength(r, theta)
        
        # Electromagnetic stress-energy contribution (simplified)
        em_stress = (E_field**2 + B_field**2) / (8 * np.pi)
        
        return em_stress * G / C**4


def create_rotating_wormhole(wormhole_type: str = "kerr_like",
                           throat_radius: float = 1000.0,
                           mass: float = 1e30,
                           angular_momentum: float = 1e42,
                           **kwargs) -> RotatingWormholeMetric:
    """Factory function to create rotating wormhole metrics.
    
    Args:
        wormhole_type: Type of rotating wormhole ("kerr_like", "extended")
        throat_radius: Throat radius in meters
        mass: Mass in kg
        angular_momentum: Angular momentum in J⋅s
        **kwargs: Additional parameters
        
    Returns:
        Rotating wormhole metric instance
    """
    
    # Compute spin parameter
    if mass > 0:
        spin_parameter = angular_momentum / (mass * C)
    else:
        spin_parameter = 0.0
    
    # Create rotation parameters
    rotation_params = RotationParameters(
        angular_momentum=angular_momentum,
        spin_parameter=spin_parameter,
        rotation_axis=kwargs.get('rotation_axis', (0.0, 0.0, 1.0)),
        frame_dragging_coefficient=kwargs.get('frame_dragging_coefficient', 1.0),
        ergosphere_enabled=kwargs.get('ergosphere_enabled', True)
    )
    
    if wormhole_type.lower() == "kerr_like":
        return KerrLikeWormhole(
            throat_radius=throat_radius,
            mass=mass,
            rotation_params=rotation_params,
            wormhole_parameter=kwargs.get('wormhole_parameter', 0.1)
        )
    
    elif wormhole_type.lower() == "extended":
        return ExtendedRotatingWormhole(
            throat_radius=throat_radius,
            mass=mass,
            rotation_params=rotation_params,
            exotic_matter_coupling=kwargs.get('exotic_matter_coupling', 0.1),
            magnetic_dipole_moment=kwargs.get('magnetic_dipole_moment', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown wormhole type: {wormhole_type}")


def analyze_rotating_wormhole_stability(wormhole: RotatingWormholeMetric,
                                      r_range: Tuple[float, float] = None) -> Dict[str, Any]:
    """Analyze stability properties of rotating wormhole.
    
    Args:
        wormhole: Rotating wormhole metric
        r_range: Range of radii to analyze (r_min, r_max)
        
    Returns:
        Dictionary of stability analysis results
    """
    
    if r_range is None:
        r_min = wormhole.throat_radius
        r_max = 10 * wormhole.throat_radius
    else:
        r_min, r_max = r_range
    
    analysis = {
        'throat_radius': wormhole.throat_radius,
        'spin_parameter': wormhole.spin_parameter,
        'effective_horizon': wormhole.effective_horizon,
        'stability_indicators': {},
        'ergosphere_analysis': {}
    }
    
    # Sample points for analysis
    r_points = np.linspace(r_min, r_max, 50)
    theta_points = np.linspace(0.1, np.pi - 0.1, 10)
    
    try:
        # Check for coordinate singularities
        singular_points = []
        for r in r_points:
            try:
                g = wormhole.metric_tensor((0, r, np.pi/2, 0))
                det_g = np.linalg.det(g)
                if abs(det_g) < 1e-12:
                    singular_points.append(r)
            except:
                singular_points.append(r)
        
        analysis['singular_points'] = singular_points
        
        # Ergosphere analysis
        if wormhole.rotation_params.ergosphere_enabled:
            ergosphere_boundaries = []
            for theta in theta_points:
                try:
                    r_ergo = wormhole.ergosphere_boundary(theta)
                    if np.isfinite(r_ergo):
                        ergosphere_boundaries.append((theta, r_ergo))
                except:
                    pass
            
            analysis['ergosphere_analysis'] = {
                'boundaries': ergosphere_boundaries,
                'max_radius': max([r for _, r in ergosphere_boundaries]) if ergosphere_boundaries else None
            }
        
        # Frame-dragging analysis
        max_omega = 0
        for r in r_points:
            for theta in theta_points:
                omega = abs(wormhole.angular_velocity(r, theta))
                max_omega = max(max_omega, omega)
        
        analysis['stability_indicators'] = {
            'max_angular_velocity': max_omega,
            'has_singularities': len(singular_points) > 0,
            'has_ergosphere': len(analysis['ergosphere_analysis'].get('boundaries', [])) > 0,
            'rotation_strength': abs(wormhole.spin_parameter)
        }
        
        # Overall stability assessment
        stability_score = 1.0
        if len(singular_points) > 0:
            stability_score *= 0.5
        if abs(wormhole.spin_parameter) > 0.8:
            stability_score *= 0.7
        if max_omega > 0.5 * C:  # Significant fraction of light speed
            stability_score *= 0.8
        
        analysis['overall_stability_score'] = stability_score
        
    except Exception as e:
        logger.error(f"Stability analysis failed: {e}")
        analysis['error'] = str(e)
    
    return analysis