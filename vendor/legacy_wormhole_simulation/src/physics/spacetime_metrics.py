"""
Spacetime metrics for wormhole geometries.

This module implements various spacetime metrics used in wormhole physics,
including Morris-Thorne wormholes, Ellis wormholes, and other exotic geometries.
"""

import numpy as np
import sympy as sp
from typing import Tuple, Dict, Callable, Optional, Union
from abc import ABC, abstractmethod

from src.physics.constants import C, G, GeometrizedUnits


class SpacetimeMetric(ABC):
    """Abstract base class for spacetime metrics."""
    
    def __init__(self, coordinates: str = 'spherical'):
        """Initialize metric with coordinate system."""
        self.coordinates = coordinates
        self.dimension = 4  # 4D spacetime
        
    @abstractmethod
    def metric_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Return the metric tensor at given coordinates."""
        pass
    
    @abstractmethod
    def christoffel_symbols(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Christoffel symbols at given coordinates."""
        pass
    
    def inverse_metric(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute inverse metric tensor."""
        g = self.metric_tensor(coordinates)
        return np.linalg.inv(g)
    
    def metric_determinant(self, coordinates: Tuple[float, ...]) -> float:
        """Compute determinant of metric tensor."""
        g = self.metric_tensor(coordinates)
        return np.linalg.det(g)
    
    def proper_distance(self, coord1: Tuple[float, ...], 
                       coord2: Tuple[float, ...]) -> float:
        """Calculate proper distance between two points."""
        # This is a simplified calculation - full implementation would integrate
        diff = np.array(coord2) - np.array(coord1)
        g = self.metric_tensor(coord1)
        return np.sqrt(np.abs(np.dot(diff, np.dot(g, diff))))


class MorrisThorneeWormhole(SpacetimeMetric):
    """Morris-Thorne traversable wormhole metric.
    
    Line element: ds² = -dt² + dr²/(1 - b(r)/r) + r²(dθ² + sin²θ dφ²)
    where b(r) is the shape function.
    """
    
    def __init__(self, throat_radius: float, shape_function: str = 'minimal'):
        """Initialize Morris-Thorne wormhole.
        
        Args:
            throat_radius: Minimum radius of wormhole throat
            shape_function: Type of shape function ('minimal', 'exponential', 'polynomial')
        """
        super().__init__('spherical')
        self.b0 = throat_radius
        self.throat_radius = throat_radius  # For compatibility
        self.shape_function_type = shape_function
        
    def shape_function(self, r: float) -> float:
        """Shape function b(r) for the wormhole."""
        if self.shape_function_type == 'minimal':
            # Minimal case: b(r) = b0
            return self.b0
        elif self.shape_function_type == 'exponential':
            # Exponential: b(r) = b0 * exp(-α(r-b0))
            alpha = 1.0 / self.b0
            return self.b0 * np.exp(-alpha * (r - self.b0)) if r >= self.b0 else self.b0
        elif self.shape_function_type == 'polynomial':
            # Polynomial: b(r) = b0 * (b0/r)²
            return self.b0 * (self.b0 / r)**2 if r >= self.b0 else self.b0
        else:
            raise ValueError(f"Unknown shape function: {self.shape_function_type}")
    
    def shape_function_derivative(self, r: float) -> float:
        """Derivative of shape function db/dr."""
        if self.shape_function_type == 'minimal':
            return 0.0
        elif self.shape_function_type == 'exponential':
            alpha = 1.0 / self.b0
            if r >= self.b0:
                return -alpha * self.b0 * np.exp(-alpha * (r - self.b0))
            else:
                return 0.0
        elif self.shape_function_type == 'polynomial':
            if r >= self.b0:
                return -2 * self.b0**3 / r**3
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown shape function: {self.shape_function_type}")
    
    def metric_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Morris-Thorne metric tensor in (t, r, θ, φ) coordinates."""
        t, r, theta, phi = coordinates
        
        # Avoid singularity at throat
        r = max(r, self.b0 + 1e-10)
        
        b = self.shape_function(r)
        
        # Metric components
        g_tt = -1.0
        g_rr = 1.0 / (1.0 - b/r) if r > b else 1e10  # Large value near throat
        g_theta_theta = r**2
        g_phi_phi = r**2 * np.sin(theta)**2
        
        # Build metric tensor
        g = np.zeros((4, 4))
        g[0, 0] = g_tt
        g[1, 1] = g_rr
        g[2, 2] = g_theta_theta
        g[3, 3] = g_phi_phi
        
        return g
    
    def christoffel_symbols(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Christoffel symbols for Morris-Thorne metric."""
        t, r, theta, phi = coordinates
        r = max(r, self.b0 + 1e-10)
        
        b = self.shape_function(r)
        db_dr = self.shape_function_derivative(r)
        
        # Initialize Christoffel symbols
        gamma = np.zeros((4, 4, 4))
        
        # Non-zero components
        # Γʳₜₜ
        gamma[1, 0, 0] = (b - r * db_dr) / (2 * r**2 * (1 - b/r))
        
        # Γʳᵣᵣ  
        gamma[1, 1, 1] = (r * db_dr - b) / (2 * r * (r - b))
        
        # Γʳθθ
        gamma[1, 2, 2] = -(r - b)
        
        # Γʳφφ
        gamma[1, 3, 3] = -(r - b) * np.sin(theta)**2
        
        # Γθᵣθ = Γθθᵣ
        gamma[2, 1, 2] = gamma[2, 2, 1] = 1.0 / r
        
        # Γθφφ
        gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # Γφᵣφ = Γφφᵣ
        gamma[3, 1, 3] = gamma[3, 3, 1] = 1.0 / r
        
        # Γφθφ = Γφφθ
        gamma[3, 2, 3] = gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return gamma


class EllisWormhole(SpacetimeMetric):
    """Ellis wormhole (drainhole) metric.
    
    A simple wormhole solution with topology R × S³.
    """
    
    def __init__(self, scale_parameter: float = 1.0):
        """Initialize Ellis wormhole.
        
        Args:
            scale_parameter: Characteristic scale of the wormhole
        """
        super().__init__('ellis')
        self.a = scale_parameter
    
    def metric_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Ellis wormhole metric tensor."""
        t, u, theta, phi = coordinates
        
        # Metric components in (t, u, θ, φ) coordinates
        g_tt = -1.0
        g_uu = self.a**2 / (self.a**2 + u**2)
        g_theta_theta = self.a**2 + u**2
        g_phi_phi = (self.a**2 + u**2) * np.sin(theta)**2
        
        g = np.zeros((4, 4))
        g[0, 0] = g_tt
        g[1, 1] = g_uu
        g[2, 2] = g_theta_theta
        g[3, 3] = g_phi_phi
        
        return g
    
    def christoffel_symbols(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Christoffel symbols for Ellis wormhole."""
        t, u, theta, phi = coordinates
        
        gamma = np.zeros((4, 4, 4))
        
        # Non-zero components
        # Γᵘᵤᵤ
        gamma[1, 1, 1] = -u / (self.a**2 + u**2)
        
        # Γᵘθθ
        gamma[1, 2, 2] = -u
        
        # Γᵘφφ
        gamma[1, 3, 3] = -u * np.sin(theta)**2
        
        # Γθᵤθ = Γθθᵤ
        gamma[2, 1, 2] = gamma[2, 2, 1] = u / (self.a**2 + u**2)
        
        # Γθφφ
        gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # Γφᵤφ = Γφφᵤ
        gamma[3, 1, 3] = gamma[3, 3, 1] = u / (self.a**2 + u**2)
        
        # Γφθφ = Γφφθ
        gamma[3, 2, 3] = gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return gamma


class SchwarzschildWormhole(SpacetimeMetric):
    """Schwarzschild wormhole (Einstein-Rosen bridge).
    
    Note: This is not traversable and leads to a black hole.
    """
    
    def __init__(self, mass: float):
        """Initialize Schwarzschild wormhole.
        
        Args:
            mass: Mass parameter (in kg)
        """
        super().__init__('schwarzschild')
        self.M = mass
        self.rs = GeometrizedUnits.schwarzschild_radius(mass)
    
    def metric_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Schwarzschild metric tensor."""
        t, r, theta, phi = coordinates
        
        # Avoid singularity
        r = max(r, self.rs + 1e-10)
        
        g_tt = -(1 - self.rs/r)
        g_rr = 1 / (1 - self.rs/r)
        g_theta_theta = r**2
        g_phi_phi = r**2 * np.sin(theta)**2
        
        g = np.zeros((4, 4))
        g[0, 0] = g_tt
        g[1, 1] = g_rr
        g[2, 2] = g_theta_theta
        g[3, 3] = g_phi_phi
        
        return g
    
    def christoffel_symbols(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Christoffel symbols for Schwarzschild metric."""
        t, r, theta, phi = coordinates
        r = max(r, self.rs + 1e-10)
        
        gamma = np.zeros((4, 4, 4))
        
        # Non-zero components
        # Γᵗᵣₜ = Γᵗₜᵣ
        gamma[0, 1, 0] = gamma[0, 0, 1] = self.rs / (2 * r * (r - self.rs))
        
        # Γʳₜₜ
        gamma[1, 0, 0] = self.rs * (r - self.rs) / (2 * r**3)
        
        # Γʳᵣᵣ
        gamma[1, 1, 1] = -self.rs / (2 * r * (r - self.rs))
        
        # Γʳθθ
        gamma[1, 2, 2] = -(r - self.rs)
        
        # Γʳφφ
        gamma[1, 3, 3] = -(r - self.rs) * np.sin(theta)**2
        
        # Γθᵣθ = Γθθᵣ
        gamma[2, 1, 2] = gamma[2, 2, 1] = 1.0 / r
        
        # Γθφφ
        gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # Γφᵣφ = Γφφᵣ
        gamma[3, 1, 3] = gamma[3, 3, 1] = 1.0 / r
        
        # Γφθφ = Γφφθ
        gamma[3, 2, 3] = gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return gamma


class WormholeMetricFactory:
    """Factory class for creating different wormhole metrics."""
    
    @staticmethod
    def create_metric(metric_type: str, **kwargs) -> SpacetimeMetric:
        """Create a wormhole metric of specified type.
        
        Args:
            metric_type: Type of metric ('morris-thorne', 'ellis', 'schwarzschild')
            **kwargs: Parameters specific to each metric type
        """
        if metric_type.lower() == 'morris-thorne':
            throat_radius = kwargs.get('throat_radius', 1000.0)  # 1 km default
            shape_function = kwargs.get('shape_function', 'minimal')
            return MorrisThorneeWormhole(throat_radius, shape_function)
        
        elif metric_type.lower() == 'ellis':
            scale_parameter = kwargs.get('scale_parameter', 1.0)
            return EllisWormhole(scale_parameter)
        
        elif metric_type.lower() == 'schwarzschild':
            mass = kwargs.get('mass', 1.989e30)  # Solar mass default
            return SchwarzschildWormhole(mass)
        
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")


def compute_geodesics(metric: SpacetimeMetric, 
                     initial_position: np.ndarray,
                     initial_velocity: np.ndarray,
                     proper_time: float,
                     num_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Compute geodesic trajectories in given spacetime.
    
    Args:
        metric: Spacetime metric object
        initial_position: Initial 4-position
        initial_velocity: Initial 4-velocity
        proper_time: Total proper time for integration
        num_steps: Number of integration steps
    
    Returns:
        Tuple of (positions, velocities) arrays
    """
    dt = proper_time / num_steps
    
    positions = np.zeros((num_steps + 1, 4))
    velocities = np.zeros((num_steps + 1, 4))
    
    positions[0] = initial_position
    velocities[0] = initial_velocity
    
    for i in range(num_steps):
        pos = positions[i]
        vel = velocities[i]
        
        # Compute Christoffel symbols at current position
        gamma = metric.christoffel_symbols(tuple(pos))
        
        # Geodesic equation: d²x^μ/dτ² = -Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ)
        acceleration = np.zeros(4)
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    acceleration[mu] -= gamma[mu, nu, rho] * vel[nu] * vel[rho]
        
        # Simple Euler integration (could be improved with RK4)
        velocities[i + 1] = vel + acceleration * dt
        positions[i + 1] = pos + vel * dt
    
    return positions, velocities


def compute_tidal_forces(metric: SpacetimeMetric,
                        position: Tuple[float, ...],
                        separation: np.ndarray) -> np.ndarray:
    """Compute tidal forces at given position.
    
    Args:
        metric: Spacetime metric
        position: 4-position in spacetime
        separation: Spatial separation vector
    
    Returns:
        Tidal acceleration vector
    """
    # This is a simplified calculation
    # Full implementation would use Riemann tensor
    
    eps = 1e-8
    tidal_accel = np.zeros(3)
    
    for i in range(3):
        pos_plus = list(position)
        pos_minus = list(position)
        pos_plus[i + 1] += eps
        pos_minus[i + 1] -= eps
        
        g_plus = metric.metric_tensor(tuple(pos_plus))
        g_minus = metric.metric_tensor(tuple(pos_minus))
        
        # Approximate tidal acceleration
        tidal_accel[i] = (g_plus[i + 1, i + 1] - g_minus[i + 1, i + 1]) / (2 * eps)
    
    return tidal_accel * np.linalg.norm(separation)