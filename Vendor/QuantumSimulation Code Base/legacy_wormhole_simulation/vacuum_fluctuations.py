"""
Quantum vacuum fluctuations in curved spacetime.

This module simulates quantum field effects including Casimir forces,
Hawking radiation, Unruh effect, and vacuum polarization in wormhole
geometries, implementing both perturbative and non-perturbative approaches.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Tuple, Optional
import scipy.constants as const

from src.physics.constants import HBAR, C, G
from src.physics.spacetime_metrics import SpacetimeMetric


class VacuumFluctuations:
    """Simulates quantum vacuum fluctuations in curved spacetime.
    
    This class implements quantum field theory in curved spacetime, specifically handling
    vacuum fluctuations near wormhole geometries. The implementation follows the 
    semi-classical approximation where quantum fields propagate on a classical curved background.

    Mathematical Framework:
    ---------------------
    1. Field Quantization:
       φ(x) = ∑ₖ (aₖfₖ(x) + aₖ†fₖ*(x))
       where:
       - φ(x) is the quantum field
       - fₖ(x) are mode functions
       - aₖ, aₖ† are annihilation/creation operators

    2. Vacuum State Definition:
       |0⟩ₘ = exp(-β/2 ∑ωₖnₖ) |0⟩
       where:
       - |0⟩ₘ is the modified vacuum
       - β is inverse temperature
       - ωₖ are mode frequencies
       - nₖ are number operators

    3. Energy-Momentum Tensor:
       ⟨Tμν⟩ = ∑ₖ(∂μfₖ∂νfₖ* - ½gμν(∂αfₖ∂ᵅfₖ* + m²|fₖ|²))
       
    4. Casimir Effect Near Wormhole:
       Ec = -π²ℏc/720a⁴ × F(r/a)
       where F(r/a) is the geometric form factor

    Features:
    ---------
    - Quantum field mode decomposition
    - Vacuum state computation in curved space
    - Energy density and pressure calculations
    - Casimir force simulation
    - Hawking/Unruh effect modeling
    - Vacuum polarization computation
    
    Example Usage:
    -------------
    >>> # Initialize with wormhole metric
    >>> vacuum = VacuumFluctuations(metric=wormhole_metric)
    >>> 
    >>> # Setup and compute vacuum effects
    >>> vacuum.initialize_modes(kmax=10)
    >>> energy = vacuum.compute_energy_density(r=2.0)
    >>> pressure = vacuum.compute_vacuum_pressure()
    >>> 
    >>> # Analyze stability impact
    >>> is_stable = vacuum.check_vacuum_stability()
    >>> fluctuations = vacuum.get_quantum_fluctuations()

    References:
    -----------
    [1] Birrell & Davies, "Quantum Fields in Curved Space" (1982)
    [2] Ford & Roman, "Averaged Energy Conditions and Quantum Inequalities" (1995)
    [3] Visser, "Quantum Vacuum Energy in General Relativity" (1996)
    """
    
    def __init__(self, 
                metric: SpacetimeMetric,
                num_modes: int = 100,
                cutoff_energy: float = 1e15):  # eV
        """Initialize vacuum fluctuations simulator.
        
        Args:
            metric: Spacetime metric for curved space calculations
            num_modes: Number of field modes to consider
            cutoff_energy: UV cutoff energy in eV
        """
        self.metric = metric
        self.num_modes = num_modes
        self.cutoff_energy = cutoff_energy
        
        # Setup frequency modes
        self.setup_modes()
        
    def setup_modes(self):
        """Initialize quantum field modes."""
        # Logarithmically spaced frequencies up to cutoff
        max_freq = self.cutoff_energy * const.e / HBAR  # Convert eV to angular frequency
        self.frequencies = np.logspace(0, np.log10(max_freq), self.num_modes)
        
        # Create mode operators
        self.a = [qt.destroy(2) for _ in range(self.num_modes)]  # Annihilation operators
        self.a_dag = [op.dag() for op in self.a]  # Creation operators
        
    def vacuum_state(self) -> qt.Qobj:
        """Generate vacuum state for all modes."""
        return qt.tensor([qt.basis([2], 0) for _ in range(self.num_modes)])
        
    def number_operator(self, mode: int) -> qt.Qobj:
        """Number operator for given mode."""
        return self.a_dag[mode] * self.a[mode]
        
    def energy_density(self, coordinates: Tuple[float, ...]) -> float:
        """Calculate vacuum energy density at given coordinates.
        
        Args:
            coordinates: Spacetime coordinates (t, r, θ, φ)
            
        Returns:
            Energy density in J/m³
        """
        # Get metric components
        g = self.metric.metric_tensor(coordinates)
        g_det = np.linalg.det(g)
        
        # Sum over mode contributions
        energy = 0.0
        for i, freq in enumerate(self.frequencies):
            # Mode energy includes metric-dependent correction
            mode_energy = HBAR * freq * np.sqrt(-g_det) / 2
            energy += mode_energy
            
        return energy
        
    def casimir_force(self, separation: float) -> float:
        """Calculate Casimir force between parallel plates.
        
        Args:
            separation: Plate separation in meters
            
        Returns:
            Force per unit area in N/m²
        """
        # Standard Casimir force with metric corrections
        g_tt = self.metric.metric_tensor((0, separation/2, 0, 0))[0,0]
        
        force_density = -np.pi**2 * HBAR * C / (240 * separation**4)
        return force_density * np.sqrt(-g_tt)
        
    def particle_spectrum(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """Calculate spectrum of particle creation from vacuum.
        
        Args:
            coordinates: Spacetime coordinates
            
        Returns:
            Array of particle numbers for each mode
        """
        spectrum = np.zeros(self.num_modes)
        
        # Calculate Bogoliubov coefficients
        for i, freq in enumerate(self.frequencies):
            beta = self._bogoliubov_coefficient(coordinates, freq)
            spectrum[i] = np.abs(beta)**2
            
        return spectrum
        
    def _bogoliubov_coefficient(self, 
                               coordinates: Tuple[float, ...],
                               frequency: float) -> complex:
        """Calculate Bogoliubov coefficient for mode mixing.
        
        This determines particle creation from vacuum fluctuations.
        """
        # Get metric data
        g = self.metric.metric_tensor(coordinates)
        christoffel = self.metric.christoffel_symbols(coordinates)
        
        # Simplified calculation for weak field approximation
        g_tt = g[0,0]
        gamma = 1/np.sqrt(1 - 2*G*self.metric.mass/(C**2 * coordinates[1]))
        
        beta = np.pi * frequency/(2*C) * (gamma - 1)
        return complex(beta)
from typing import List, Tuple, Dict, Optional, Union, Callable
import scipy.special as special
import scipy.integrate as integrate
from abc import ABC, abstractmethod

from src.physics.constants import HBAR, C, K_B, PLANCK_LENGTH, PLANCK_TIME
from src.physics.spacetime_metrics import SpacetimeMetric


class QuantumField(ABC):
    """Abstract base class for quantum fields in curved spacetime."""
    
    def __init__(self, field_type: str, mass: float = 0.0):
        """Initialize quantum field.
        
        Args:
            field_type: Type of field ('scalar', 'fermion', 'vector')
            mass: Field mass (in natural units)
        """
        self.field_type = field_type
        self.mass = mass
        self.spin = self._get_spin()
    
    def _get_spin(self) -> float:
        """Get spin of the field."""
        spin_map = {'scalar': 0, 'fermion': 0.5, 'vector': 1}
        return spin_map.get(self.field_type, 0)
    
    @abstractmethod
    def mode_functions(self, spacetime_point: Tuple[float, ...],
                      mode_index: int) -> complex:
        """Return mode functions at given spacetime point."""
        pass
    
    @abstractmethod
    def vacuum_expectation_value(self, operator: str,
                                spacetime_point: Tuple[float, ...]) -> float:
        """Compute vacuum expectation value of field operator."""
        pass


class ScalarQuantumField(QuantumField):
    """Scalar quantum field in curved spacetime."""
    
    def __init__(self, mass: float = 0.0, coupling_constant: float = 0.0):
        """Initialize scalar field.
        
        Args:
            mass: Field mass
            coupling_constant: Self-interaction coupling (λφ⁴ theory)
        """
        super().__init__('scalar', mass)
        self.lambda_coupling = coupling_constant
        self.cutoff_frequency = C / PLANCK_LENGTH  # Planck scale cutoff
    
    def mode_functions(self, spacetime_point: Tuple[float, ...],
                      mode_index: int) -> complex:
        """Scalar field mode functions."""
        t, r, theta, phi = spacetime_point
        n, l, m = self._decode_mode_index(mode_index)
        
        # Radial function (simplified)
        if self.mass == 0:
            R_nl = np.sqrt(2 / r) * special.spherical_jn(l, n * np.pi * r)
        else:
            # Massive field: modified Bessel functions
            kr = np.sqrt(n**2 - self.mass**2 * r**2)
            R_nl = special.spherical_jn(l, kr) if kr > 0 else 0
        
        # Angular functions
        Y_lm = special.sph_harm(m, l, phi, theta)
        
        # Time dependence
        omega_n = np.sqrt(n**2 + self.mass**2)
        time_factor = np.exp(-1j * omega_n * t)
        
        return R_nl * Y_lm * time_factor
    
    def _decode_mode_index(self, mode_index: int) -> Tuple[int, int, int]:
        """Decode linear mode index to (n, l, m) quantum numbers."""
        # Simplified mapping
        n = mode_index // 100 + 1
        l = (mode_index // 10) % 10
        m = mode_index % 10 - 5  # m ranges from -5 to 4
        return n, l, m
    
    def vacuum_expectation_value(self, operator: str,
                                spacetime_point: Tuple[float, ...]) -> float:
        """Compute VEV of field operators."""
        t, r, theta, phi = spacetime_point
        
        if operator == 'field_squared':
            # ⟨φ²⟩ vacuum expectation value
            return self._field_squared_vev(r)
        
        elif operator == 'energy_density':
            # ⟨T₀₀⟩ energy density
            return self._energy_density_vev(r)
        
        elif operator == 'stress_tensor':
            # Full stress-energy tensor components
            return self._stress_tensor_vev(spacetime_point)
        
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _field_squared_vev(self, r: float) -> float:
        """Vacuum expectation value of φ²."""
        # Regularized using Pauli-Villars
        # ⟨φ²⟩ ~ ∫ d³k ℏ/(2ωₖ) with UV cutoff
        
        if r <= PLANCK_LENGTH:
            return np.inf  # Divergent at origin
        
        # Dimensional analysis: [φ²] = [Energy]² in natural units
        # Include geometric factors
        result = HBAR * self.cutoff_frequency / (16 * np.pi**2 * r**2)
        
        # Mass correction
        if self.mass > 0:
            result *= np.exp(-2 * self.mass * r)  # Yukawa suppression
        
        return result
    
    def _energy_density_vev(self, r: float) -> float:
        """Vacuum expectation value of energy density T₀₀."""
        # From field squared VEV
        phi_squared = self._field_squared_vev(r)
        
        # Energy density includes kinetic and potential terms
        # Simplified: ρ ~ ∂φ²/∂r² + m²φ²
        gradient_term = HBAR * self.cutoff_frequency / (8 * np.pi**2 * r**4)
        mass_term = self.mass**2 * phi_squared
        
        return gradient_term + mass_term
    
    def _stress_tensor_vev(self, spacetime_point: Tuple[float, ...]) -> np.ndarray:
        """Full stress-energy tensor VEV."""
        t, r, theta, phi = spacetime_point
        
        # Diagonal stress tensor in spherical coordinates
        T = np.zeros((4, 4))
        
        # T₀₀ (energy density)
        T[0, 0] = -self._energy_density_vev(r)
        
        # T₁₁ (radial pressure)
        T[1, 1] = self._energy_density_vev(r) / 3  # Radiation-like
        
        # T₂₂, T₃₃ (angular pressures)
        T[2, 2] = T[3, 3] = -self._energy_density_vev(r) / 6
        
        return T


class CasimirEffect:
    """Casimir effect calculation in various geometries."""
    
    def __init__(self, geometry: str = 'parallel_plates'):
        """Initialize Casimir effect calculation.
        
        Args:
            geometry: Geometry type ('parallel_plates', 'sphere', 'cylinder')
        """
        self.geometry = geometry
    
    def casimir_energy(self, **geometry_params) -> float:
        """Calculate Casimir energy for given geometry."""
        if self.geometry == 'parallel_plates':
            return self._parallel_plates_energy(**geometry_params)
        elif self.geometry == 'sphere':
            return self._spherical_casimir_energy(**geometry_params)
        elif self.geometry == 'cylinder':
            return self._cylindrical_casimir_energy(**geometry_params)
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")
    
    def _parallel_plates_energy(self, plate_separation: float,
                               plate_area: float = 1.0) -> float:
        """Casimir energy between parallel plates.
        
        E = -π²ℏc A / (240 d³)
        """
        d = plate_separation
        A = plate_area
        
        if d <= 0:
            return -np.inf
        
        return -np.pi**2 * HBAR * C * A / (240 * d**3)
    
    def _spherical_casimir_energy(self, sphere_radius: float) -> float:
        """Casimir energy of conducting sphere.
        
        E = -0.09 ℏc / R (approximate)
        """
        R = sphere_radius
        
        if R <= 0:
            return -np.inf
        
        # Coefficient from numerical calculations
        coefficient = -0.09
        return coefficient * HBAR * C / R
    
    def _cylindrical_casimir_energy(self, radius: float, length: float) -> float:
        """Casimir energy of conducting cylinder."""
        R = radius
        L = length
        
        if R <= 0 or L <= 0:
            return -np.inf
        
        # Simplified formula
        return -np.pi * HBAR * C * L / (24 * R**2)
    
    def casimir_force(self, **geometry_params) -> float:
        """Calculate Casimir force from energy derivative."""
        if self.geometry == 'parallel_plates':
            d = geometry_params.get('plate_separation', 1.0)
            A = geometry_params.get('plate_area', 1.0)
            
            # Force: F = -dE/dd
            return -np.pi**2 * HBAR * C * A / (80 * d**4)
        
        elif self.geometry == 'sphere':
            R = geometry_params.get('sphere_radius', 1.0)
            return 0.09 * HBAR * C / R**2
        
        else:
            # General numerical derivative
            eps = 1e-8
            if 'plate_separation' in geometry_params:
                params_plus = geometry_params.copy()
                params_minus = geometry_params.copy()
                params_plus['plate_separation'] += eps
                params_minus['plate_separation'] -= eps
                
                E_plus = self.casimir_energy(**params_plus)
                E_minus = self.casimir_energy(**params_minus)
                
                return -(E_plus - E_minus) / (2 * eps)
            
            return 0.0


class UnruhEffect:
    """Unruh effect: vacuum appears thermal to accelerated observers."""
    
    def __init__(self, acceleration: float):
        """Initialize Unruh effect calculation.
        
        Args:
            acceleration: Proper acceleration of observer
        """
        self.acceleration = acceleration
        self.unruh_temperature = HBAR * acceleration / (2 * np.pi * K_B * C)
    
    def thermal_spectrum(self, frequency: float) -> float:
        """Thermal spectrum seen by accelerated observer.
        
        n(ω) = 1 / (exp(ℏω/kT) - 1)
        """
        if frequency <= 0:
            return 0.0
        
        beta = 1.0 / (K_B * self.unruh_temperature)
        exponent = HBAR * frequency * beta
        
        # Avoid overflow
        if exponent > 100:
            return 0.0
        
        return 1.0 / (np.exp(exponent) - 1)
    
    def particle_detection_rate(self, detector_coupling: float,
                               frequency_range: Tuple[float, float]) -> float:
        """Rate of particle detection by Unruh-DeWitt detector."""
        omega_min, omega_max = frequency_range
        
        def integrand(omega):
            return (detector_coupling**2 * omega * 
                   self.thermal_spectrum(omega) * 
                   np.exp(-omega / self.acceleration))  # Detector response
        
        rate, _ = integrate.quad(integrand, omega_min, omega_max)
        return rate
    
    def entanglement_degradation(self, initial_entanglement: float,
                                proper_time: float) -> float:
        """Entanglement degradation due to Unruh effect."""
        # Exponential decay with characteristic time
        tau_decoherence = HBAR / (K_B * self.unruh_temperature)
        
        return initial_entanglement * np.exp(-proper_time / tau_decoherence)


class HawkingRadiation:
    """Hawking radiation from black holes and wormholes."""
    
    def __init__(self, black_hole_mass: float):
        """Initialize Hawking radiation calculation.
        
        Args:
            black_hole_mass: Mass of black hole (in kg)
        """
        self.mass = black_hole_mass
        self.schwarzschild_radius = 2 * HBAR * self.mass / (C**3)  # In natural units
        self.hawking_temperature = HBAR * C**3 / (8 * np.pi * K_B * self.mass)
        self.surface_gravity = C**4 / (4 * self.mass)  # Surface gravity
    
    def thermal_spectrum(self, frequency: float) -> float:
        """Hawking thermal spectrum."""
        if frequency <= 0:
            return 0.0
        
        beta = 1.0 / (K_B * self.hawking_temperature)
        exponent = HBAR * frequency * beta
        
        if exponent > 100:
            return 0.0
        
        return frequency**2 / (2 * np.pi**2 * C**3) / (np.exp(exponent) - 1)
    
    def luminosity(self) -> float:
        """Total luminosity of Hawking radiation.
        
        L = σA T⁴ where σ is Stefan-Boltzmann constant
        """
        area = 4 * np.pi * self.schwarzschild_radius**2
        sigma_sb = 2 * np.pi**5 * K_B**4 / (15 * HBAR**3 * C**2)  # Stefan-Boltzmann
        
        return sigma_sb * area * self.hawking_temperature**4
    
    def evaporation_time(self) -> float:
        """Time for complete evaporation."""
        # t = 5120πG²M³/(ℏc⁴)
        return (5120 * np.pi * (HBAR * self.mass)**3) / (C**5)
    
    def information_scrambling_time(self) -> float:
        """Information scrambling time scale."""
        # t_scramble ~ M log(M) in Planck units
        return self.schwarzschild_radius * np.log(self.mass) / C
    
    def page_curve(self, evolution_times: np.ndarray) -> np.ndarray:
        """Page curve: entanglement entropy vs time."""
        t_page = self.evaporation_time() / 2  # Page time
        S_max = self.mass  # Maximum entropy ~ Bekenstein-Hawking
        
        entropies = []
        for t in evolution_times:
            if t < t_page:
                # Growing phase
                S = S_max * (t / t_page)
            else:
                # Decreasing phase (information recovery)
                S = S_max * (1 - (t - t_page) / t_page)
                S = max(0, S)  # Entropy cannot be negative
            
            entropies.append(S)
        
        return np.array(entropies)


class VacuumPolarization:
    """Vacuum polarization effects in curved spacetime."""
    
    def __init__(self, field: QuantumField, metric: SpacetimeMetric):
        """Initialize vacuum polarization calculation.
        
        Args:
            field: Quantum field
            metric: Background spacetime metric
        """
        self.field = field
        self.metric = metric
        self.regularization_scale = C / PLANCK_LENGTH
    
    def stress_tensor_polarization(self, spacetime_point: Tuple[float, ...]) -> np.ndarray:
        """Compute vacuum polarization stress-energy tensor."""
        # This is a simplified calculation
        # Full calculation requires Green's functions in curved spacetime
        
        t, r, theta, phi = spacetime_point
        
        # Vacuum expectation value from field
        if hasattr(self.field, 'vacuum_expectation_value'):
            T_vac = self.field.vacuum_expectation_value('stress_tensor', spacetime_point)
            if isinstance(T_vac, np.ndarray):
                return T_vac
        
        # Fallback: dimensional analysis estimate
        energy_scale = HBAR * self.regularization_scale
        length_scale = 1.0 / self.regularization_scale
        
        # Geometric factors from metric curvature
        g = self.metric.metric_tensor(spacetime_point)
        R_scalar = self._estimate_ricci_scalar(spacetime_point)  # Simplified
        
        # Polarization tensor ~ (energy scale)⁴ × geometric factors
        T_pol = np.zeros((4, 4))
        
        for mu in range(4):
            for nu in range(4):
                if mu == nu:
                    T_pol[mu, nu] = (energy_scale**4 / (16 * np.pi**2)) * R_scalar * g[mu, nu]
                # Off-diagonal terms are typically smaller
        
        return T_pol
    
    def _estimate_ricci_scalar(self, spacetime_point: Tuple[float, ...]) -> float:
        """Simplified Ricci scalar estimate."""
        # This would normally require computing full curvature
        # Using simplified geometric estimate
        
        t, r, theta, phi = spacetime_point
        
        # For spherically symmetric metrics
        if hasattr(self.metric, 'shape_function'):  # Morris-Thorne wormhole
            b = self.metric.shape_function(r)
            if r > b:
                return -6 * b / (r**2 * (r - b))  # Approximate curvature
        
        # Default: small curvature
        return 1.0 / r**2 if r > 0 else 0.0
    
    def beta_function(self, coupling_constant: float, 
                     energy_scale: float) -> float:
        """Beta function for running coupling constants.
        
        β(g) = μ dg/dμ (renormalization group equation)
        """
        # For scalar field with λφ⁴ interaction
        if self.field.field_type == 'scalar':
            # One-loop beta function
            beta = 3 * coupling_constant**2 / (16 * np.pi**2)
            
            # Include curved spacetime corrections
            curvature_correction = 1.0  # Simplified
            return beta * curvature_correction
        
        return 0.0  # No running for free fields
    
    def trace_anomaly(self, spacetime_point: Tuple[float, ...]) -> float:
        """Trace anomaly: ⟨Tμμ⟩ ≠ 0 in curved spacetime."""
        # Conformal anomaly for massless fields
        
        if self.field.mass == 0:  # Massless field
            R_scalar = self._estimate_ricci_scalar(spacetime_point)
            
            # Trace anomaly coefficient depends on field type
            if self.field.field_type == 'scalar':
                coefficient = 1 / (24 * np.pi**2)  # For real scalar
            elif self.field.field_type == 'fermion':
                coefficient = 11 / (120 * np.pi**2)  # For Dirac fermion
            else:
                coefficient = 0.0
            
            return coefficient * HBAR * R_scalar
        
        else:  # Massive field
            # Mass provides natural scale, no anomaly
            return self.field.mass**2 * self.field.vacuum_expectation_value(
                'field_squared', spacetime_point)


class QuantumFluctuationSimulator:
    """Comprehensive quantum vacuum fluctuation simulator."""
    
    def __init__(self, field_type: str = 'scalar', 
                 spacetime_metric: SpacetimeMetric = None):
        """Initialize quantum fluctuation simulator.
        
        Args:
            field_type: Type of quantum field
            spacetime_metric: Background spacetime geometry
        """
        self.field = ScalarQuantumField() if field_type == 'scalar' else None
        self.metric = spacetime_metric
        
        # Initialize various effects
        self.casimir = CasimirEffect()
        self.vacuum_polarization = VacuumPolarization(self.field, self.metric) if self.metric else None
    
    def simulate_vacuum_energy(self, spatial_region: Dict,
                              num_modes: int = 1000) -> Dict:
        """Simulate vacuum energy in given spatial region."""
        results = {
            'total_energy': 0.0,
            'energy_density_profile': [],
            'mode_contributions': [],
            'regularized_energy': 0.0
        }
        
        # Sample points in spatial region
        if 'radius_range' in spatial_region:
            r_min, r_max = spatial_region['radius_range']
            radii = np.linspace(r_min, r_max, 50)
            
            energy_densities = []
            
            for r in radii:
                spacetime_point = (0, r, np.pi/2, 0)  # t=0, equatorial plane
                
                if self.field:
                    rho_vac = self.field.vacuum_expectation_value('energy_density', spacetime_point)
                    energy_densities.append(rho_vac)
                else:
                    energy_densities.append(0.0)
            
            results['energy_density_profile'] = energy_densities
            
            # Integrate to get total energy (simplified)
            if energy_densities:
                total_energy = 4 * np.pi * np.trapz([r**2 * rho for r, rho in zip(radii, energy_densities)], radii)
                results['total_energy'] = total_energy
        
        return results
    
    def wormhole_vacuum_effects(self, throat_radius: float,
                               observer_trajectory: List[Tuple[float, ...]]) -> Dict:
        """Analyze vacuum effects specific to wormholes."""
        results = {
            'casimir_contribution': 0.0,
            'hawking_like_radiation': 0.0,
            'vacuum_polarization': [],
            'traversability_constraints': {}
        }
        
        # Casimir effect in wormhole throat
        if throat_radius > 0:
            # Treat throat as cylindrical cavity
            self.casimir.geometry = 'cylinder'
            casimir_energy = self.casimir.casimir_energy(radius=throat_radius, length=2*throat_radius)
            results['casimir_contribution'] = casimir_energy
        
        # Vacuum polarization along trajectory
        if self.vacuum_polarization and observer_trajectory:
            polarization_data = []
            
            for point in observer_trajectory:
                T_pol = self.vacuum_polarization.stress_tensor_polarization(point)
                trace_anomaly = self.vacuum_polarization.trace_anomaly(point)
                
                polarization_data.append({
                    'point': point,
                    'stress_tensor': T_pol.tolist() if isinstance(T_pol, np.ndarray) else T_pol,
                    'trace_anomaly': trace_anomaly
                })
            
            results['vacuum_polarization'] = polarization_data
        
        # Traversability constraints from vacuum effects
        if throat_radius > 0:
            # Estimate vacuum energy density at throat
            throat_point = (0, throat_radius, np.pi/2, 0)
            
            if self.field:
                rho_throat = self.field.vacuum_expectation_value('energy_density', throat_point)
                
                # Compare with exotic matter requirements
                exotic_matter_scale = HBAR * C / (throat_radius**4)
                
                results['traversability_constraints'] = {
                    'vacuum_energy_density': rho_throat,
                    'exotic_matter_scale': exotic_matter_scale,
                    'energy_condition_violation': abs(rho_throat) > exotic_matter_scale,
                    'vacuum_dominance': abs(rho_throat / exotic_matter_scale)
                }
        
        return results
    
    def quantum_decoherence_rate(self, entangled_system_size: float,
                                environment_temperature: float = 0.0) -> float:
        """Estimate quantum decoherence rate due to vacuum fluctuations."""
        # Decoherence rate ~ (system size / coherence length)² × interaction rate
        
        # Coherence length from vacuum fluctuations
        if environment_temperature > 0:
            # Thermal coherence length
            l_coherence = HBAR * C / (K_B * environment_temperature)
        else:
            # Zero temperature: coherence limited by vacuum fluctuations
            l_coherence = PLANCK_LENGTH * (PLANCK_LENGTH / entangled_system_size)**(1/4)
        
        # Interaction rate with vacuum
        interaction_rate = C / l_coherence
        
        # Decoherence rate
        gamma_decoherence = interaction_rate * (entangled_system_size / l_coherence)**2
        
        return gamma_decoherence


def analyze_vacuum_stability(field_configuration: Dict,
                            spacetime_region: Dict) -> Dict:
    """Analyze vacuum stability under field perturbations.
    
    Args:
        field_configuration: Field parameters and perturbations
        spacetime_region: Region of spacetime to analyze
    
    Returns:
        Stability analysis results
    """
    # Effective potential analysis
    field_value = field_configuration.get('field_value', 0.0)
    field_mass = field_configuration.get('mass', 0.0)
    coupling = field_configuration.get('coupling', 0.0)
    
    # Effective potential V_eff(φ) = ½m²φ² + ¼λφ⁴ + curved spacetime corrections
    def effective_potential(phi):
        return 0.5 * field_mass**2 * phi**2 + 0.25 * coupling * phi**4
    
    # Find minima
    phi_range = np.linspace(-2, 2, 1000)
    V_eff = [effective_potential(phi) for phi in phi_range]
    
    # Local minima
    minima_indices = []
    for i in range(1, len(V_eff) - 1):
        if V_eff[i] < V_eff[i-1] and V_eff[i] < V_eff[i+1]:
            minima_indices.append(i)
    
    vacuum_states = [phi_range[i] for i in minima_indices]
    
    # Stability criterion
    stable = len(vacuum_states) > 0 and coupling >= 0
    
    return {
        'vacuum_states': vacuum_states,
        'stable': stable,
        'potential_barrier_height': max(V_eff) - min(V_eff) if V_eff else 0.0,
        'false_vacuum': len(vacuum_states) > 1,
        'tunneling_rate': np.exp(-max(V_eff) / HBAR) if V_eff and stable else 0.0
    }