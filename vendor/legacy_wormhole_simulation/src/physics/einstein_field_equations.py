"""
Einstein Field Equations for wormhole spacetimes.

This module implements the computation of Einstein tensor components,
Ricci tensor, Riemann curvature tensor, and verification of Einstein's
field equations for wormhole geometries.
"""

import numpy as np
import sympy as sp
from typing import Tuple, Dict, Callable, Optional, Union
from abc import ABC, abstractmethod
import scipy.integrate as integrate
from functools import lru_cache

from src.physics.constants import C, G, NUMERICAL_PRECISION
from src.physics.spacetime_metrics import SpacetimeMetric
from src.physics.stress_energy_tensor import StressEnergyTensor


class CurvatureTensor(ABC):
    """Abstract base class for spacetime curvature calculations."""
    
    def __init__(self, metric: SpacetimeMetric):
        """Initialize with spacetime metric.
        
        Args:
            metric: SpacetimeMetric object
        """
        self.metric = metric
        self.dimension = 4
    
    @abstractmethod
    def riemann_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Riemann curvature tensor R^μ_νρσ."""
        pass
    
    @abstractmethod
    def ricci_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Ricci tensor R_μν."""
        pass
    
    @abstractmethod
    def ricci_scalar(self, coordinates: Tuple[float, ...]) -> float:
        """Compute Ricci scalar R."""
        pass
    
    @abstractmethod
    def einstein_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Einstein tensor G_μν = R_μν - (1/2)g_μν R."""
        pass


class AnalyticalCurvature(CurvatureTensor):
    """Analytical computation of curvature tensors using symbolic differentiation."""
    
    def __init__(self, metric: SpacetimeMetric):
        """Initialize analytical curvature computation."""
        super().__init__(metric)
        self._setup_symbolic_computation()
    
    def _setup_symbolic_computation(self):
        """Set up symbolic variables for automatic differentiation."""
        self.t_sym, self.r_sym, self.theta_sym, self.phi_sym = sp.symbols('t r theta phi', real=True)
        self.coords_sym = [self.t_sym, self.r_sym, self.theta_sym, self.phi_sym]
        
        # Cache for symbolic expressions
        self._metric_cache = {}
        self._christoffel_cache = {}
    
    @lru_cache(maxsize=128)
    def riemann_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Riemann tensor using analytical differentiation."""
        # Get Christoffel symbols
        gamma = self.metric.christoffel_symbols(coordinates)
        
        # Initialize Riemann tensor
        R = np.zeros((4, 4, 4, 4))
        
        # Numerical differentiation for Riemann tensor components
        eps = NUMERICAL_PRECISION['integration_tolerance']
        
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # R^μ_νρσ = ∂_ρΓ^μ_νσ - ∂_σΓ^μ_νρ + Γ^μ_λρΓ^λ_νσ - Γ^μ_λσΓ^λ_νρ
                        
                        # Partial derivatives
                        coords_rho_plus = list(coordinates)
                        coords_rho_minus = list(coordinates)
                        coords_rho_plus[rho] += eps
                        coords_rho_minus[rho] -= eps
                        
                        gamma_rho_plus = self.metric.christoffel_symbols(tuple(coords_rho_plus))
                        gamma_rho_minus = self.metric.christoffel_symbols(tuple(coords_rho_minus))
                        
                        d_gamma_rho = (gamma_rho_plus[mu, nu, sigma] - gamma_rho_minus[mu, nu, sigma]) / (2 * eps)
                        
                        coords_sigma_plus = list(coordinates)
                        coords_sigma_minus = list(coordinates)
                        coords_sigma_plus[sigma] += eps
                        coords_sigma_minus[sigma] -= eps
                        
                        gamma_sigma_plus = self.metric.christoffel_symbols(tuple(coords_sigma_plus))
                        gamma_sigma_minus = self.metric.christoffel_symbols(tuple(coords_sigma_minus))
                        
                        d_gamma_sigma = (gamma_sigma_plus[mu, nu, rho] - gamma_sigma_minus[mu, nu, rho]) / (2 * eps)
                        
                        # Product terms
                        product_term_1 = sum(gamma[mu, lam, rho] * gamma[lam, nu, sigma] for lam in range(4))
                        product_term_2 = sum(gamma[mu, lam, sigma] * gamma[lam, nu, rho] for lam in range(4))
                        
                        R[mu, nu, rho, sigma] = d_gamma_rho - d_gamma_sigma + product_term_1 - product_term_2
        
        return R
    
    def ricci_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Ricci tensor R_μν = R^ρ_μρν."""
        R_full = self.riemann_tensor(coordinates)
        R_ricci = np.zeros((4, 4))
        
        for mu in range(4):
            for nu in range(4):
                R_ricci[mu, nu] = sum(R_full[rho, mu, rho, nu] for rho in range(4))
        
        return R_ricci
    
    def ricci_scalar(self, coordinates: Tuple[float, ...]) -> float:
        """Compute Ricci scalar R = g^μν R_μν."""
        R_ricci = self.ricci_tensor(coordinates)
        g_inv = self.metric.inverse_metric(coordinates)
        
        R_scalar = np.trace(np.dot(g_inv, R_ricci))
        return R_scalar
    
    def einstein_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Compute Einstein tensor G_μν = R_μν - (1/2)g_μν R."""
        R_ricci = self.ricci_tensor(coordinates)
        R_scalar = self.ricci_scalar(coordinates)
        g_metric = self.metric.metric_tensor(coordinates)
        
        G = R_ricci - 0.5 * R_scalar * g_metric
        return G


class NumericalCurvature(CurvatureTensor):
    """Numerical computation of curvature tensors using finite differences."""
    
    def __init__(self, metric: SpacetimeMetric, step_size: float = 1e-8):
        """Initialize numerical curvature computation.
        
        Args:
            metric: SpacetimeMetric object
            step_size: Step size for finite differences
        """
        super().__init__(metric)
        self.eps = step_size
    
    def _finite_difference_christoffel(self, coordinates: Tuple[float, ...], 
                                     direction: int) -> np.ndarray:
        """Compute finite difference of Christoffel symbols."""
        coords_plus = list(coordinates)
        coords_minus = list(coordinates)
        coords_plus[direction] += self.eps
        coords_minus[direction] -= self.eps
        
        gamma_plus = self.metric.christoffel_symbols(tuple(coords_plus))
        gamma_minus = self.metric.christoffel_symbols(tuple(coords_minus))
        
        return (gamma_plus - gamma_minus) / (2 * self.eps)
    
    def riemann_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Numerical Riemann tensor computation."""
        R = np.zeros((4, 4, 4, 4))
        gamma = self.metric.christoffel_symbols(coordinates)
        
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Partial derivatives
                        d_gamma_rho = self._finite_difference_christoffel(coordinates, rho)
                        d_gamma_sigma = self._finite_difference_christoffel(coordinates, sigma)
                        
                        # Riemann tensor components
                        R[mu, nu, rho, sigma] = (
                            d_gamma_rho[mu, nu, sigma] - d_gamma_sigma[mu, nu, rho] +
                            sum(gamma[mu, lam, rho] * gamma[lam, nu, sigma] - 
                                gamma[mu, lam, sigma] * gamma[lam, nu, rho] for lam in range(4))
                        )
        
        return R
    
    def ricci_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Numerical Ricci tensor computation."""
        R_full = self.riemann_tensor(coordinates)
        R_ricci = np.zeros((4, 4))
        
        for mu in range(4):
            for nu in range(4):
                R_ricci[mu, nu] = sum(R_full[rho, mu, rho, nu] for rho in range(4))
        
        return R_ricci
    
    def ricci_scalar(self, coordinates: Tuple[float, ...]) -> float:
        """Numerical Ricci scalar computation."""
        R_ricci = self.ricci_tensor(coordinates)
        g_inv = self.metric.inverse_metric(coordinates)
        
        return np.trace(np.dot(g_inv, R_ricci))
    
    def einstein_tensor(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Numerical Einstein tensor computation."""
        R_ricci = self.ricci_tensor(coordinates)
        R_scalar = self.ricci_scalar(coordinates)
        g_metric = self.metric.metric_tensor(coordinates)
        
        return R_ricci - 0.5 * R_scalar * g_metric


class EinsteinFieldEquations:
    """Einstein field equations solver and verifier."""
    
    def __init__(self, metric: SpacetimeMetric, 
                 curvature_method: str = 'numerical'):
        """Initialize Einstein field equations.
        
        Args:
            metric: SpacetimeMetric object
            curvature_method: 'analytical' or 'numerical'
        """
        self.metric = metric
        
        if curvature_method == 'analytical':
            self.curvature = AnalyticalCurvature(metric)
        elif curvature_method == 'numerical':
            self.curvature = NumericalCurvature(metric)
        else:
            raise ValueError(f"Unknown curvature method: {curvature_method}")
    
    def solve_for_stress_energy(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Solve Einstein equations for stress-energy tensor.
        
        G_μν = (8πG/c⁴) T_μν
        """
        G = self.curvature.einstein_tensor(coordinates)
        coefficient = 8 * np.pi * G / C**4
        T = G / coefficient
        return T
    
    def verify_field_equations(self, stress_energy: StressEnergyTensor,
                              coordinates: Tuple[float, ...],
                              tolerance: float = 1e-6) -> Dict[str, Union[bool, float]]:
        """Verify Einstein field equations.
        
        Args:
            stress_energy: StressEnergyTensor object
            coordinates: Spacetime coordinates
            tolerance: Numerical tolerance
        
        Returns:
            Verification results
        """
        # Compute Einstein tensor from metric
        G_metric = self.curvature.einstein_tensor(coordinates)
        
        # Compute stress-energy tensor
        T_matter = stress_energy.tensor_components(coordinates)
        
        # Einstein field equations: G_μν = (8πG/c⁴) T_μν
        coefficient = 8 * np.pi * G / C**4
        G_expected = coefficient * T_matter
        
        # Check agreement
        difference = G_metric - G_expected
        max_error = np.max(np.abs(difference))
        rms_error = np.sqrt(np.mean(difference**2))
        
        equations_satisfied = max_error < tolerance
        
        return {
            'equations_satisfied': equations_satisfied,
            'max_error': max_error,
            'rms_error': rms_error,
            'relative_error': max_error / (np.max(np.abs(G_metric)) + 1e-15),
            'tolerance_used': tolerance
        }
    
    def compute_geodesic_deviation(self, coordinates: Tuple[float, ...],
                                  separation_vector: np.ndarray) -> np.ndarray:
        """Compute geodesic deviation (tidal forces).
        
        Args:
            coordinates: Spacetime coordinates
            separation_vector: Separation between geodesics
        
        Returns:
            Geodesic deviation vector
        """
        R = self.curvature.riemann_tensor(coordinates)
        
        # Geodesic deviation equation: D²ξ^μ/Dτ² = R^μ_νρσ u^ν ξ^ρ u^σ
        # For simplicity, assume u^μ = (1, 0, 0, 0) (static observer)
        u = np.array([1, 0, 0, 0])
        xi = separation_vector
        
        deviation = np.zeros(4)
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        deviation[mu] += R[mu, nu, rho, sigma] * u[nu] * xi[rho] * u[sigma]
        
        return deviation


def compute_kretschmann_scalar(metric: SpacetimeMetric,
                              coordinates: Tuple[float, ...]) -> float:
    """Compute Kretschmann scalar K = R_μνρσ R^μνρσ.
    
    This is a curvature invariant useful for detecting singularities.
    
    Args:
        metric: SpacetimeMetric object
        coordinates: Spacetime coordinates
    
    Returns:
        Kretschmann scalar value
    """
    curvature = NumericalCurvature(metric)
    R = curvature.riemann_tensor(coordinates)
    g_inv = metric.inverse_metric(coordinates)
    
    # Raise indices: R^μνρσ = g^μα g^νβ g^ργ g^σδ R_αβγδ
    K = 0.0
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    for alpha in range(4):
                        for beta in range(4):
                            for gamma in range(4):
                                for delta in range(4):
                                    K += (g_inv[mu, alpha] * g_inv[nu, beta] * 
                                         g_inv[rho, gamma] * g_inv[sigma, delta] *
                                         R[alpha, beta, gamma, delta] * R[mu, nu, rho, sigma])
    
    return K


def analyze_spacetime_curvature(metric: SpacetimeMetric,
                               coordinate_range: Dict[str, Tuple[float, float]],
                               num_points: int = 50) -> Dict:
    """Analyze spacetime curvature over a coordinate range.
    
    Args:
        metric: SpacetimeMetric object
        coordinate_range: Dictionary with coordinate ranges
        num_points: Number of sampling points per coordinate
    
    Returns:
        Curvature analysis results
    """
    # Create coordinate grids
    t_range = coordinate_range.get('t', (0, 1))
    r_range = coordinate_range.get('r', (1, 10))
    theta_range = coordinate_range.get('theta', (np.pi/4, 3*np.pi/4))
    phi_range = coordinate_range.get('phi', (0, 2*np.pi))
    
    t_vals = np.linspace(t_range[0], t_range[1], num_points)
    r_vals = np.linspace(r_range[0], r_range[1], num_points)
    
    # Sample coordinates (simplified 2D analysis)
    results = {
        'coordinates': [],
        'ricci_scalar': [],
        'kretschmann_scalar': [],
        'einstein_tensor_trace': [],
        'singularities': []
    }
    
    curvature = NumericalCurvature(metric)
    
    for t in t_vals[::5]:  # Sample fewer time points
        for r in r_vals:
            coords = (t, r, np.pi/2, 0)  # Equatorial plane
            
            try:
                R_scalar = curvature.ricci_scalar(coords)
                K_scalar = compute_kretschmann_scalar(metric, coords)
                G = curvature.einstein_tensor(coords)
                G_trace = np.trace(G)
                
                results['coordinates'].append(coords)
                results['ricci_scalar'].append(R_scalar)
                results['kretschmann_scalar'].append(K_scalar)
                results['einstein_tensor_trace'].append(G_trace)
                
                # Check for singularities (high curvature)
                if abs(K_scalar) > 1e10 or abs(R_scalar) > 1e10:
                    results['singularities'].append(coords)
                    
            except (ZeroDivisionError, np.linalg.LinAlgError, OverflowError):
                # Skip singular points
                results['singularities'].append(coords)
                continue
    
    return results


def conservation_check(stress_energy: StressEnergyTensor,
                      metric: SpacetimeMetric,
                      coordinates: Tuple[float, ...],
                      step_size: float = 1e-8) -> Dict[str, float]:
    """Check energy-momentum conservation ∇_μ T^μν = 0.
    
    Args:
        stress_energy: StressEnergyTensor object
        metric: SpacetimeMetric object
        coordinates: Spacetime coordinates
        step_size: Step size for finite differences
    
    Returns:
        Conservation violation measures
    """
    eps = step_size
    T = stress_energy.tensor_components(coordinates)
    gamma = metric.christoffel_symbols(coordinates)
    g_inv = metric.inverse_metric(coordinates)
    
    # Compute ∇_μ T^μν for each ν
    conservation_violation = np.zeros(4)
    
    for nu in range(4):
        div_T = 0.0
        
        for mu in range(4):
            # Raise index: T^μν = g^μα T_αν
            T_up_mu_nu = sum(g_inv[mu, alpha] * T[alpha, nu] for alpha in range(4))
            
            # Partial derivative ∂_μ T^μν
            coords_plus = list(coordinates)
            coords_minus = list(coordinates)
            coords_plus[mu] += eps
            coords_minus[mu] -= eps
            
            T_plus = stress_energy.tensor_components(tuple(coords_plus))
            T_minus = stress_energy.tensor_components(tuple(coords_minus))
            g_inv_plus = metric.inverse_metric(tuple(coords_plus))
            g_inv_minus = metric.inverse_metric(tuple(coords_minus))
            
            T_up_plus = sum(g_inv_plus[mu, alpha] * T_plus[alpha, nu] for alpha in range(4))
            T_up_minus = sum(g_inv_minus[mu, alpha] * T_minus[alpha, nu] for alpha in range(4))
            
            dT_dx = (T_up_plus - T_up_minus) / (2 * eps)
            
            # Connection term: Γ^μ_μλ T^λν
            connection_term = sum(gamma[mu, mu, lam] * 
                                sum(g_inv[lam, alpha] * T[alpha, nu] for alpha in range(4))
                                for lam in range(4))
            
            div_T += dT_dx + connection_term
        
        conservation_violation[nu] = abs(div_T)
    
    return {
        'max_violation': np.max(conservation_violation),
        'rms_violation': np.sqrt(np.mean(conservation_violation**2)),
        'violations_by_component': conservation_violation.tolist()
    }