"""
Dynamic Throat Evolution - Time-Dependent Wormhole Geometries.

This module implements dynamic evolution of wormhole throat geometries,
including time-dependent metrics, stability analysis, and collapse/expansion
scenarios for traversable wormhole studies.
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
from src.physics.rotating_wormhole_metrics import KerrLikeWormhole, RotationParameters
from src.physics.constants import G, C, HBAR

logger = logging.getLogger(__name__)


@dataclass
class EvolutionParameters:
    """Parameters controlling dynamic throat evolution."""
    
    # Time evolution controls
    evolution_timescale: float = 1000.0  # Characteristic evolution time (s)
    damping_coefficient: float = 0.1     # Energy dissipation rate
    quantum_stabilization: float = 0.05  # Quantum pressure effects
    
    # Driving forces
    exotic_matter_pressure: float = -1e-3  # Exotic matter pressure (Pa)
    casimir_energy_density: float = -1e-10  # Casimir effect contribution
    vacuum_polarization: float = 1e-12     # Vacuum polarization effects
    
    # Boundary conditions
    minimum_throat_radius: float = 100.0   # Minimum stable throat (m)
    maximum_throat_radius: float = 10000.0 # Maximum allowed throat (m)
    critical_expansion_rate: float = 0.1   # Critical expansion rate (m/s)
    
    # Stability thresholds
    collapse_threshold: float = -0.5       # Negative growth rate for collapse
    stabilization_tolerance: float = 1e-6  # Convergence tolerance
    
    def __post_init__(self):
        """Validate evolution parameters."""
        if self.minimum_throat_radius <= 0:
            raise ValueError("Minimum throat radius must be positive")
        
        if self.maximum_throat_radius <= self.minimum_throat_radius:
            raise ValueError("Maximum throat radius must exceed minimum")
        
        if self.evolution_timescale <= 0:
            raise ValueError("Evolution timescale must be positive")


class DynamicThroatEvolution:
    """Dynamic evolution system for wormhole throat geometries."""
    
    def __init__(self, initial_wormhole: KerrLikeWormhole,
                 evolution_params: EvolutionParameters):
        """Initialize dynamic throat evolution.
        
        Args:
            initial_wormhole: Initial wormhole configuration
            evolution_params: Evolution control parameters
        """
        self.initial_wormhole = initial_wormhole
        self.evolution_params = evolution_params
        
        # Evolution state tracking
        self.current_time = 0.0
        self.throat_history = []
        self.stability_history = []
        self.energy_history = []
        
        # Initialize with starting conditions
        self.current_throat_radius = initial_wormhole.throat_radius
        self.current_mass = initial_wormhole.mass
        self.current_rotation_params = initial_wormhole.rotation_params
        
        logger.info(f"Initialized dynamic throat evolution: "
                   f"initial_radius={self.current_throat_radius:.1f}m, "
                   f"timescale={evolution_params.evolution_timescale:.1f}s")
    
    def throat_evolution_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Differential equation for throat radius evolution.
        
        Implements the master equation:
        db/dt = f(b, M, J, exotic_matter, quantum_effects)
        
        Args:
            t: Time coordinate
            state: [throat_radius, mass, angular_momentum]
            
        Returns:
            Time derivatives [db/dt, dM/dt, dJ/dt]
        """
        b, M, J = state
        
        # Ensure physical bounds
        b = max(self.evolution_params.minimum_throat_radius, 
                min(b, self.evolution_params.maximum_throat_radius))
        
        # Gravitational pressure (tends to collapse)
        gravitational_pressure = -G * M / (C**2 * b**2)
        
        # Exotic matter pressure (tends to expand) 
        exotic_pressure = self.evolution_params.exotic_matter_pressure
        
        # Quantum stabilization effects
        casimir_force = self.evolution_params.casimir_energy_density * HBAR * C / b**4
        vacuum_polarization = self.evolution_params.vacuum_polarization * HBAR * C / b**3
        quantum_pressure = casimir_force + vacuum_polarization
        
        # Rotation effects (centrifugal expansion)
        if M > 0 and b > 0:
            rotation_pressure = (J / (M * C))**2 / (8 * np.pi * b**4)
        else:
            rotation_pressure = 0.0
        
        # Total pressure balance
        net_pressure = (gravitational_pressure + exotic_pressure + 
                       quantum_pressure + rotation_pressure)
        
        # Throat radius evolution
        db_dt = net_pressure / self.evolution_params.damping_coefficient
        
        # Apply stabilization near quantum scale
        if b < 2 * self.evolution_params.minimum_throat_radius:
            quantum_stabilization = self.evolution_params.quantum_stabilization / b
            db_dt += quantum_stabilization
        
        # Mass evolution (exotic matter effects)
        dM_dt = -abs(exotic_pressure) * 4 * np.pi * b**2 / C**2
        
        # Angular momentum evolution (frame-dragging damping)
        if J != 0:
            dJ_dt = -self.evolution_params.damping_coefficient * J / self.evolution_params.evolution_timescale
        else:
            dJ_dt = 0.0
        
        return np.array([db_dt, dM_dt, dJ_dt])
    
    def evolve_throat(self, time_span: float, num_steps: int = 1000) -> Dict[str, Any]:
        """Evolve throat geometry over specified time span.
        
        Args:
            time_span: Total evolution time (seconds)
            num_steps: Number of integration steps
            
        Returns:
            Evolution results dictionary
        """
        # Initial conditions
        initial_state = np.array([
            self.current_throat_radius,
            self.current_mass,
            self.current_rotation_params.angular_momentum
        ])
        
        # Time points
        t_span = (self.current_time, self.current_time + time_span)
        t_eval = np.linspace(t_span[0], t_span[1], num_steps)
        
        logger.info(f"Evolving throat over {time_span:.1f}s with {num_steps} steps")
        
        try:
            # Solve evolution equations
            solution = integrate.solve_ivp(
                self.throat_evolution_equation,
                t_span,
                initial_state,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if not solution.success:
                logger.warning(f"Evolution integration failed: {solution.message}")
                return self._create_failed_result()
            
            # Extract results
            times = solution.t
            throat_radii = solution.y[0]
            masses = solution.y[1]
            angular_momenta = solution.y[2]
            
            # Analyze evolution
            results = self._analyze_evolution(times, throat_radii, masses, angular_momenta)
            
            # Update current state
            self.current_time = times[-1]
            self.current_throat_radius = throat_radii[-1]
            self.current_mass = masses[-1]
            
            # Update rotation parameters
            if masses[-1] > 0:
                new_spin_parameter = angular_momenta[-1] / (masses[-1] * C)
            else:
                new_spin_parameter = 0.0
            
            self.current_rotation_params = RotationParameters(
                angular_momentum=angular_momenta[-1],
                spin_parameter=new_spin_parameter,
                rotation_axis=self.current_rotation_params.rotation_axis,
                frame_dragging_coefficient=self.current_rotation_params.frame_dragging_coefficient,
                ergosphere_enabled=self.current_rotation_params.ergosphere_enabled
            )
            
            # Store history
            self.throat_history.extend(list(zip(times, throat_radii)))
            
            logger.info(f"Evolution completed: final_radius={throat_radii[-1]:.1f}m")
            
            return results
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return self._create_failed_result()
    
    def _analyze_evolution(self, times: np.ndarray, throat_radii: np.ndarray,
                          masses: np.ndarray, angular_momenta: np.ndarray) -> Dict[str, Any]:
        """Analyze evolution results and determine stability."""
        
        # Compute derivatives
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        throat_velocity = np.gradient(throat_radii, dt)
        throat_acceleration = np.gradient(throat_velocity, dt)
        
        # Classify evolution phases
        expansion_mask = throat_velocity > self.evolution_params.critical_expansion_rate
        contraction_mask = throat_velocity < self.evolution_params.collapse_threshold
        stable_mask = np.abs(throat_velocity) < self.evolution_params.stabilization_tolerance
        
        # Evolution statistics
        max_radius = np.max(throat_radii)
        min_radius = np.min(throat_radii)
        final_radius = throat_radii[-1]
        initial_radius = throat_radii[0]
        
        # Determine final state
        if final_radius < self.evolution_params.minimum_throat_radius:
            final_state = "collapsed"
        elif final_radius > self.evolution_params.maximum_throat_radius:
            final_state = "runaway_expansion"
        elif np.abs(throat_velocity[-1]) < self.evolution_params.stabilization_tolerance:
            final_state = "stabilized"
        elif throat_velocity[-1] > 0:
            final_state = "expanding"
        else:
            final_state = "contracting"
        
        # Stability analysis
        stability_score = self._compute_stability_score(throat_radii, throat_velocity)
        
        # Energy analysis
        energy_analysis = self._analyze_energy_evolution(times, throat_radii, masses, angular_momenta)
        
        return {
            'evolution_success': True,
            'times': times,
            'throat_radii': throat_radii,
            'masses': masses,
            'angular_momenta': angular_momenta,
            'throat_velocity': throat_velocity,
            'throat_acceleration': throat_acceleration,
            'statistics': {
                'initial_radius': initial_radius,
                'final_radius': final_radius,
                'max_radius': max_radius,
                'min_radius': min_radius,
                'radius_change': final_radius - initial_radius,
                'relative_change': (final_radius - initial_radius) / initial_radius,
                'final_state': final_state,
                'stability_score': stability_score
            },
            'phase_analysis': {
                'expansion_fraction': np.sum(expansion_mask) / len(expansion_mask),
                'contraction_fraction': np.sum(contraction_mask) / len(contraction_mask),
                'stable_fraction': np.sum(stable_mask) / len(stable_mask),
                'dominant_phase': self._determine_dominant_phase(expansion_mask, contraction_mask, stable_mask)
            },
            'energy_analysis': energy_analysis
        }
    
    def _compute_stability_score(self, throat_radii: np.ndarray, throat_velocity: np.ndarray) -> float:
        """Compute stability score based on throat evolution."""
        
        # Penalize large variations
        radius_variance = np.var(throat_radii) / np.mean(throat_radii)**2
        velocity_variance = np.var(throat_velocity) / (np.mean(np.abs(throat_velocity)) + 1e-10)
        
        # Reward steady states
        final_velocity = abs(throat_velocity[-1])
        steady_bonus = 1.0 / (1.0 + final_velocity / self.evolution_params.stabilization_tolerance)
        
        # Combined stability score (0 = unstable, 1 = perfectly stable)
        stability_score = steady_bonus / (1.0 + radius_variance + velocity_variance)
        
        return min(1.0, max(0.0, stability_score))
    
    def _determine_dominant_phase(self, expansion_mask: np.ndarray, 
                                 contraction_mask: np.ndarray, 
                                 stable_mask: np.ndarray) -> str:
        """Determine the dominant evolution phase."""
        fractions = {
            'expansion': np.sum(expansion_mask) / len(expansion_mask),
            'contraction': np.sum(contraction_mask) / len(contraction_mask),
            'stable': np.sum(stable_mask) / len(stable_mask)
        }
        
        return max(fractions, key=fractions.get)
    
    def _analyze_energy_evolution(self, times: np.ndarray, throat_radii: np.ndarray,
                                 masses: np.ndarray, angular_momenta: np.ndarray) -> Dict[str, Any]:
        """Analyze energy evolution during throat dynamics."""
        
        # Gravitational binding energy
        gravitational_energy = -G * masses**2 / (C**2 * throat_radii)
        
        # Rotational energy
        rotational_energy = angular_momenta**2 / (2 * masses * throat_radii**2)
        
        # Exotic matter energy (negative)
        exotic_energy = self.evolution_params.exotic_matter_pressure * 4 * np.pi * throat_radii**3 / 3
        
        # Total energy
        total_energy = gravitational_energy + rotational_energy + exotic_energy
        
        # Energy conservation check
        initial_energy = total_energy[0]
        final_energy = total_energy[-1]
        energy_change = final_energy - initial_energy
        energy_conservation_violation = abs(energy_change) / abs(initial_energy)
        
        return {
            'gravitational_energy': gravitational_energy,
            'rotational_energy': rotational_energy,
            'exotic_energy': exotic_energy,
            'total_energy': total_energy,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': energy_change,
            'energy_conservation_violation': energy_conservation_violation
        }
    
    def _create_failed_result(self) -> Dict[str, Any]:
        """Create result dictionary for failed evolution."""
        return {
            'evolution_success': False,
            'error_message': "Evolution integration failed",
            'final_state': "error"
        }
    
    def get_current_wormhole(self) -> KerrLikeWormhole:
        """Get current wormhole configuration after evolution."""
        return KerrLikeWormhole(
            throat_radius=self.current_throat_radius,
            mass=self.current_mass,
            rotation_params=self.current_rotation_params,
            wormhole_parameter=self.initial_wormhole.wormhole_parameter
        )
    
    def analyze_long_term_stability(self, total_time: float = 10000.0,
                                   checkpoint_interval: float = 1000.0) -> Dict[str, Any]:
        """Perform long-term stability analysis with multiple evolution phases.
        
        Args:
            total_time: Total simulation time
            checkpoint_interval: Time between stability checkpoints
            
        Returns:
            Long-term stability analysis results
        """
        logger.info(f"Starting long-term stability analysis over {total_time:.1f}s")
        
        checkpoints = []
        evolution_phases = []
        cumulative_time = 0.0
        
        while cumulative_time < total_time:
            # Evolve for checkpoint interval
            remaining_time = min(checkpoint_interval, total_time - cumulative_time)
            
            evolution_result = self.evolve_throat(remaining_time, num_steps=500)
            
            if not evolution_result['evolution_success']:
                logger.warning("Evolution failed during long-term analysis")
                break
            
            # Record checkpoint
            checkpoint = {
                'time': cumulative_time + remaining_time,
                'throat_radius': self.current_throat_radius,
                'mass': self.current_mass,
                'angular_momentum': self.current_rotation_params.angular_momentum,
                'stability_score': evolution_result['statistics']['stability_score'],
                'final_state': evolution_result['statistics']['final_state']
            }
            
            checkpoints.append(checkpoint)
            evolution_phases.append(evolution_result)
            
            cumulative_time += remaining_time
            
            # Check for termination conditions
            if (self.current_throat_radius < self.evolution_params.minimum_throat_radius or
                self.current_throat_radius > self.evolution_params.maximum_throat_radius):
                logger.info(f"Evolution terminated at t={cumulative_time:.1f}s due to boundary conditions")
                break
        
        # Analyze long-term trends
        long_term_analysis = self._analyze_long_term_trends(checkpoints, evolution_phases)
        
        return {
            'total_evolution_time': cumulative_time,
            'checkpoints': checkpoints,
            'evolution_phases': evolution_phases,
            'long_term_analysis': long_term_analysis
        }
    
    def _analyze_long_term_trends(self, checkpoints: List[Dict], 
                                 evolution_phases: List[Dict]) -> Dict[str, Any]:
        """Analyze long-term evolution trends."""
        
        if not checkpoints:
            return {'error': 'No checkpoints available'}
        
        times = np.array([cp['time'] for cp in checkpoints])
        radii = np.array([cp['throat_radius'] for cp in checkpoints])
        masses = np.array([cp['mass'] for cp in checkpoints])
        stability_scores = np.array([cp['stability_score'] for cp in checkpoints])
        
        # Trend analysis
        radius_trend = np.polyfit(times, radii, 1)[0] if len(times) > 1 else 0.0
        mass_trend = np.polyfit(times, masses, 1)[0] if len(times) > 1 else 0.0
        stability_trend = np.polyfit(times, stability_scores, 1)[0] if len(times) > 1 else 0.0
        
        # Classify long-term behavior
        if abs(radius_trend) < 1e-6:
            long_term_behavior = "stable"
        elif radius_trend > 0:
            long_term_behavior = "expanding"
        else:
            long_term_behavior = "contracting"
        
        # Overall stability assessment
        mean_stability = np.mean(stability_scores)
        stability_variance = np.var(stability_scores)
        
        if mean_stability > 0.8 and stability_variance < 0.1:
            overall_assessment = "highly_stable"
        elif mean_stability > 0.5 and stability_variance < 0.3:
            overall_assessment = "moderately_stable" 
        else:
            overall_assessment = "unstable"
        
        return {
            'radius_trend': radius_trend,
            'mass_trend': mass_trend,
            'stability_trend': stability_trend,
            'long_term_behavior': long_term_behavior,
            'mean_stability_score': mean_stability,
            'stability_variance': stability_variance,
            'overall_assessment': overall_assessment,
            'final_radius': radii[-1],
            'total_radius_change': radii[-1] - radii[0],
            'relative_radius_change': (radii[-1] - radii[0]) / radii[0]
        }


class ThroatCollapseScenario:
    """Specialized scenario for studying wormhole throat collapse."""
    
    def __init__(self, initial_radius: float = 1000.0, 
                 collapse_timescale: float = 100.0):
        """Initialize collapse scenario.
        
        Args:
            initial_radius: Starting throat radius
            collapse_timescale: Characteristic collapse time
        """
        self.initial_radius = initial_radius
        self.collapse_timescale = collapse_timescale
        
        # Create initial wormhole with minimal exotic matter
        rotation_params = RotationParameters(
            angular_momentum=1e42,
            spin_parameter=0.1
        )
        
        initial_wormhole = KerrLikeWormhole(
            throat_radius=initial_radius,
            mass=1e30,
            rotation_params=rotation_params
        )
        
        # Evolution parameters favoring collapse
        evolution_params = EvolutionParameters(
            evolution_timescale=collapse_timescale,
            damping_coefficient=0.5,  # Higher damping
            exotic_matter_pressure=-1e-2,  # Insufficient exotic matter
            quantum_stabilization=0.01,  # Weak quantum effects
            minimum_throat_radius=10.0,
            collapse_threshold=-1.0
        )
        
        self.evolution_system = DynamicThroatEvolution(initial_wormhole, evolution_params)
    
    def run_collapse_simulation(self, simulation_time: float = 500.0) -> Dict[str, Any]:
        """Run throat collapse simulation."""
        logger.info(f"Running throat collapse simulation for {simulation_time:.1f}s")
        
        results = self.evolution_system.evolve_throat(simulation_time, num_steps=2000)
        
        if results['evolution_success']:
            # Analyze collapse dynamics
            collapse_analysis = self._analyze_collapse_dynamics(results)
            results['collapse_analysis'] = collapse_analysis
        
        return results
    
    def _analyze_collapse_dynamics(self, evolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific collapse dynamics."""
        
        times = evolution_results['times']
        radii = evolution_results['throat_radii']
        velocities = evolution_results['throat_velocity']
        
        # Find collapse onset
        collapse_onset_idx = np.argmax(velocities < -0.1)
        collapse_onset_time = times[collapse_onset_idx] if collapse_onset_idx > 0 else times[0]
        
        # Find if/when collapse stops
        if evolution_results['statistics']['final_state'] == 'collapsed':
            collapse_time = times[-1]
            collapse_prevented = False
        else:
            # Find minimum radius point
            min_radius_idx = np.argmin(radii)
            collapse_time = times[min_radius_idx]
            collapse_prevented = True
        
        # Collapse rate analysis
        collapse_phase_mask = velocities < 0
        if np.any(collapse_phase_mask):
            collapse_velocities = velocities[collapse_phase_mask]
            max_collapse_rate = np.min(collapse_velocities)  # Most negative
            mean_collapse_rate = np.mean(collapse_velocities)
        else:
            max_collapse_rate = 0.0
            mean_collapse_rate = 0.0
        
        return {
            'collapse_onset_time': collapse_onset_time,
            'total_collapse_time': collapse_time - collapse_onset_time,
            'collapse_prevented': collapse_prevented,
            'max_collapse_rate': abs(max_collapse_rate),
            'mean_collapse_rate': abs(mean_collapse_rate),
            'minimum_radius_reached': np.min(radii),
            'final_radius': radii[-1]
        }


class ThroatExpansionScenario:
    """Specialized scenario for studying wormhole throat expansion."""
    
    def __init__(self, initial_radius: float = 500.0,
                 expansion_timescale: float = 1000.0):
        """Initialize expansion scenario.
        
        Args:
            initial_radius: Starting throat radius  
            expansion_timescale: Characteristic expansion time
        """
        self.initial_radius = initial_radius
        self.expansion_timescale = expansion_timescale
        
        # Create initial wormhole with strong exotic matter
        rotation_params = RotationParameters(
            angular_momentum=5e44,  # High rotation for expansion
            spin_parameter=0.7
        )
        
        initial_wormhole = KerrLikeWormhole(
            throat_radius=initial_radius,
            mass=5e29,  # Lower mass
            rotation_params=rotation_params
        )
        
        # Evolution parameters favoring expansion
        evolution_params = EvolutionParameters(
            evolution_timescale=expansion_timescale,
            damping_coefficient=0.05,  # Lower damping
            exotic_matter_pressure=1e-2,  # Strong exotic matter pressure
            quantum_stabilization=0.1,   # Strong quantum stabilization
            maximum_throat_radius=20000.0,
            critical_expansion_rate=0.05
        )
        
        self.evolution_system = DynamicThroatEvolution(initial_wormhole, evolution_params)
    
    def run_expansion_simulation(self, simulation_time: float = 2000.0) -> Dict[str, Any]:
        """Run throat expansion simulation."""
        logger.info(f"Running throat expansion simulation for {simulation_time:.1f}s")
        
        results = self.evolution_system.evolve_throat(simulation_time, num_steps=2000)
        
        if results['evolution_success']:
            # Analyze expansion dynamics
            expansion_analysis = self._analyze_expansion_dynamics(results)
            results['expansion_analysis'] = expansion_analysis
        
        return results
    
    def _analyze_expansion_dynamics(self, evolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific expansion dynamics."""
        
        times = evolution_results['times']
        radii = evolution_results['throat_radii']
        velocities = evolution_results['throat_velocity']
        
        # Find expansion phases
        expansion_mask = velocities > 0.01
        rapid_expansion_mask = velocities > 0.1
        
        # Expansion statistics
        expansion_fraction = np.sum(expansion_mask) / len(expansion_mask)
        rapid_expansion_fraction = np.sum(rapid_expansion_mask) / len(rapid_expansion_mask)
        
        # Maximum expansion rate
        max_expansion_rate = np.max(velocities) if len(velocities) > 0 else 0.0
        
        # Steady-state analysis
        final_velocity = velocities[-1]
        approaching_steady_state = abs(final_velocity) < 0.001
        
        return {
            'expansion_fraction': expansion_fraction,
            'rapid_expansion_fraction': rapid_expansion_fraction,
            'max_expansion_rate': max_expansion_rate,
            'final_expansion_rate': final_velocity,
            'approaching_steady_state': approaching_steady_state,
            'total_radius_increase': radii[-1] - radii[0],
            'expansion_efficiency': (radii[-1] - radii[0]) / self.initial_radius
        }


def create_evolution_scenario(scenario_type: str = "standard",
                            **kwargs) -> DynamicThroatEvolution:
    """Factory function to create different evolution scenarios.
    
    Args:
        scenario_type: Type of scenario ("standard", "collapse", "expansion")
        **kwargs: Scenario-specific parameters
        
    Returns:
        Configured evolution system
    """
    
    if scenario_type.lower() == "collapse":
        scenario = ThroatCollapseScenario(
            initial_radius=kwargs.get('initial_radius', 1000.0),
            collapse_timescale=kwargs.get('collapse_timescale', 100.0)
        )
        return scenario.evolution_system
    
    elif scenario_type.lower() == "expansion":
        scenario = ThroatExpansionScenario(
            initial_radius=kwargs.get('initial_radius', 500.0),
            expansion_timescale=kwargs.get('expansion_timescale', 1000.0)
        )
        return scenario.evolution_system
    
    else:  # standard scenario
        # Standard balanced evolution scenario
        rotation_params = RotationParameters(
            angular_momentum=kwargs.get('angular_momentum', 1e43),
            spin_parameter=kwargs.get('spin_parameter', 0.3)
        )
        
        initial_wormhole = KerrLikeWormhole(
            throat_radius=kwargs.get('initial_radius', 1000.0),
            mass=kwargs.get('mass', 1e30),
            rotation_params=rotation_params
        )
        
        evolution_params = EvolutionParameters(
            evolution_timescale=kwargs.get('evolution_timescale', 1000.0),
            exotic_matter_pressure=kwargs.get('exotic_matter_pressure', -1e-3),
            quantum_stabilization=kwargs.get('quantum_stabilization', 0.05)
        )
        
        return DynamicThroatEvolution(initial_wormhole, evolution_params)


def compare_evolution_scenarios(scenarios: List[str],
                              simulation_time: float = 1000.0,
                              **kwargs) -> Dict[str, Any]:
    """Compare multiple evolution scenarios side by side.
    
    Args:
        scenarios: List of scenario types to compare
        simulation_time: Evolution time for each scenario
        **kwargs: Shared parameters for scenarios
        
    Returns:
        Comparative analysis results
    """
    
    logger.info(f"Comparing {len(scenarios)} evolution scenarios")
    
    results = {}
    
    for scenario_type in scenarios:
        logger.info(f"Running {scenario_type} scenario")
        
        try:
            evolution_system = create_evolution_scenario(scenario_type, **kwargs)
            scenario_results = evolution_system.evolve_throat(simulation_time)
            scenario_results['scenario_type'] = scenario_type
            results[scenario_type] = scenario_results
            
        except Exception as e:
            logger.error(f"Failed to run {scenario_type} scenario: {e}")
            results[scenario_type] = {
                'evolution_success': False,
                'error': str(e),
                'scenario_type': scenario_type
            }
    
    # Comparative analysis
    comparative_analysis = _analyze_scenario_comparison(results)
    
    return {
        'individual_results': results,
        'comparative_analysis': comparative_analysis
    }


def _analyze_scenario_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze and compare evolution scenario results."""
    
    successful_scenarios = {k: v for k, v in results.items() 
                          if v.get('evolution_success', False)}
    
    if not successful_scenarios:
        return {'error': 'No successful scenarios to compare'}
    
    # Extract key metrics for comparison
    comparison_metrics = {}
    
    for scenario, result in successful_scenarios.items():
        stats = result['statistics']
        comparison_metrics[scenario] = {
            'final_radius': stats['final_radius'],
            'radius_change': stats['radius_change'],
            'relative_change': stats['relative_change'],
            'stability_score': stats['stability_score'],
            'final_state': stats['final_state']
        }
    
    # Find best/worst scenarios
    stability_scores = {k: v['stability_score'] for k, v in comparison_metrics.items()}
    most_stable = max(stability_scores, key=stability_scores.get)
    least_stable = min(stability_scores, key=stability_scores.get)
    
    # Summary statistics
    final_radii = [v['final_radius'] for v in comparison_metrics.values()]
    stability_scores_list = [v['stability_score'] for v in comparison_metrics.values()]
    
    return {
        'scenario_metrics': comparison_metrics,
        'most_stable_scenario': most_stable,
        'least_stable_scenario': least_stable,
        'mean_final_radius': np.mean(final_radii),
        'std_final_radius': np.std(final_radii),
        'mean_stability_score': np.mean(stability_scores_list),
        'std_stability_score': np.std(stability_scores_list),
        'successful_scenarios': len(successful_scenarios),
        'total_scenarios': len(results)
    }