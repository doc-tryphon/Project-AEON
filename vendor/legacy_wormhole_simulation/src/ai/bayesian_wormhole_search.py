"""
Bayesian Search for Traversable Wormhole Solutions.

This module implements advanced Bayesian optimization specifically designed
to find traversable wormhole configurations that satisfy physical constraints
and optimize for human traversability.
"""

import numpy as np
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import logging
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class TraversabilityConstraints:
    """Physical constraints for traversable wormholes."""
    
    # Energy condition constraints
    max_nec_violation: float = -1e-6  # Null Energy Condition violation limit
    max_wec_violation: float = -1e-6  # Weak Energy Condition violation limit
    max_sec_violation: float = -1e-5  # Strong Energy Condition violation limit
    
    # Tidal force constraints (for human survival)
    max_tidal_force: float = 1000.0  # Maximum tidal force in Newtons
    human_height: float = 1.8  # Human height for tidal force calculation
    
    # Stability constraints  
    min_stability_time: float = 100.0  # Minimum stability time (seconds)
    min_throat_stability: float = 0.8  # Minimum throat stability coefficient
    
    # Quantum constraints
    min_entanglement_persistence: float = 0.5  # Minimum entanglement preservation
    max_decoherence_rate: float = 0.1  # Maximum decoherence rate (1/s)
    
    # Engineering constraints
    max_exotic_matter_energy: float = 1e20  # Maximum exotic matter energy (J)
    min_traversal_speed: float = 0.01 * 299792458  # Minimum traversal speed (m/s)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert constraints to dictionary."""
        return {
            'max_nec_violation': self.max_nec_violation,
            'max_wec_violation': self.max_wec_violation,
            'max_sec_violation': self.max_sec_violation,
            'max_tidal_force': self.max_tidal_force,
            'min_stability_time': self.min_stability_time,
            'min_throat_stability': self.min_throat_stability,
            'min_entanglement_persistence': self.min_entanglement_persistence,
            'max_decoherence_rate': self.max_decoherence_rate,
            'max_exotic_matter_energy': self.max_exotic_matter_energy,
            'min_traversal_speed': self.min_traversal_speed
        }


@dataclass
class WormholeCandidate:
    """A candidate traversable wormhole solution."""
    
    parameters: Dict[str, float]
    traversability_score: float
    constraint_violations: Dict[str, float]
    physics_metrics: Dict[str, float]
    feasibility_score: float
    discovery_iteration: int
    evaluation_time: float
    
    def is_physically_viable(self, constraints: TraversabilityConstraints) -> bool:
        """Check if candidate satisfies physical constraints."""
        violations = self.constraint_violations
        
        checks = [
            violations.get('nec_violation', 0) >= constraints.max_nec_violation,
            violations.get('wec_violation', 0) >= constraints.max_wec_violation,
            violations.get('sec_violation', 0) >= constraints.max_sec_violation,
            violations.get('tidal_force', float('inf')) <= constraints.max_tidal_force,
            self.physics_metrics.get('stability_time', 0) >= constraints.min_stability_time,
            self.physics_metrics.get('entanglement_persistence', 0) >= constraints.min_entanglement_persistence
        ]
        
        return all(checks)
    
    def get_overall_score(self) -> float:
        """Get overall score combining traversability and feasibility."""
        return 0.7 * self.traversability_score + 0.3 * self.feasibility_score


class TraversabilityObjective:
    """Enhanced objective function for traversable wormhole search."""
    
    def __init__(self, quantum_circuit_factory: Callable,
                 constraints: TraversabilityConstraints,
                 penalty_weight: float = 10.0):
        """Initialize traversability objective.
        
        Args:
            quantum_circuit_factory: Function to create quantum circuits
            constraints: Physical constraints for traversability
            penalty_weight: Weight for constraint violation penalties
        """
        self.circuit_factory = quantum_circuit_factory
        self.constraints = constraints
        self.penalty_weight = penalty_weight
        self.evaluation_count = 0
        self.candidate_history: List[WormholeCandidate] = []
        
    def __call__(self, parameters: np.ndarray) -> float:
        """Evaluate wormhole configuration with constraint penalties.
        
        Args:
            parameters: [throat_radius, mass, exotic_matter_density, 
                        traversal_probability, quantum_coherence_time]
                        
        Returns:
            Penalized objective score
        """
        start_time = time.time()
        self.evaluation_count += 1
        
        try:
            # Convert parameters to configuration
            config = self._parameters_to_config(parameters)
            
            # Create and evaluate quantum circuit
            circuit = self.circuit_factory(config)
            
            # Comprehensive evaluation
            traversability_score = self._evaluate_traversability(circuit, config)
            constraint_violations = self._evaluate_constraints(circuit, config)
            physics_metrics = self._evaluate_physics(circuit, config)
            feasibility_score = self._evaluate_feasibility(config, constraint_violations)
            
            # Apply constraint penalties
            penalty = self._compute_constraint_penalty(constraint_violations)
            penalized_score = traversability_score - penalty
            
            # Create candidate record
            evaluation_time = time.time() - start_time
            candidate = WormholeCandidate(
                parameters=config.copy(),
                traversability_score=traversability_score,
                constraint_violations=constraint_violations,
                physics_metrics=physics_metrics,
                feasibility_score=feasibility_score,
                discovery_iteration=self.evaluation_count,
                evaluation_time=evaluation_time
            )
            
            self.candidate_history.append(candidate)
            
            # Log significant discoveries
            if candidate.is_physically_viable(self.constraints):
                logger.info(f"VIABLE CANDIDATE {self.evaluation_count}: "
                          f"traversability={traversability_score:.4f}, "
                          f"feasibility={feasibility_score:.4f}")
            
            return penalized_score
            
        except Exception as e:
            logger.warning(f"Evaluation {self.evaluation_count} failed: {e}")
            return -1000.0
    
    def _parameters_to_config(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to configuration dictionary."""
        return {
            'throat_radius': float(parameters[0]),
            'mass': float(parameters[1]),
            'exotic_matter_density': float(parameters[2]),
            'traversal_probability': float(parameters[3]),
            'quantum_coherence_time': float(parameters[4])
        }
    
    def _evaluate_traversability(self, circuit, config: Dict[str, float]) -> float:
        """Evaluate wormhole traversability."""
        try:
            # AI-optimized traversability prediction
            state = circuit.create_traversal_state(use_ai_params=True)
            base_traversability = circuit.predict_traversability(state)
            
            # Additional traversability factors
            throat_radius = config['throat_radius']
            mass = config['mass']
            
            # Size factor (larger throats easier to traverse)
            size_factor = min(1.0, throat_radius / 1000.0)  # Normalize to 1km
            
            # Mass factor (moderate mass preferred)
            optimal_mass = 1e30
            mass_factor = np.exp(-0.5 * ((np.log(mass) - np.log(optimal_mass))/2)**2)
            
            # Combined traversability
            traversability = base_traversability * size_factor * mass_factor
            
            return np.clip(traversability, 0, 1)
            
        except Exception as e:
            logger.warning(f"Traversability evaluation failed: {e}")
            return 0.0
    
    def _evaluate_constraints(self, circuit, config: Dict[str, float]) -> Dict[str, float]:
        """Evaluate physical constraint violations."""
        violations = {}
        
        try:
            # Create quantum state
            state = circuit.create_traversal_state()
            
            # Energy condition violations (from exotic matter)
            exotic_density = config['exotic_matter_density']
            pressure = exotic_density / 3.0  # Rough estimate
            
            violations['nec_violation'] = exotic_density + pressure
            violations['wec_violation'] = exotic_density
            violations['sec_violation'] = exotic_density + 3 * pressure
            
            # Tidal force estimation
            try:
                # Simplified tidal force calculation
                throat_radius = config['throat_radius']
                mass = config['mass']
                G = 6.67430e-11
                
                # Tidal acceleration at throat
                tidal_accel = 2 * G * mass / throat_radius**3
                tidal_force = tidal_accel * 70.0 * self.constraints.human_height  # 70kg human
                violations['tidal_force'] = tidal_force
                
            except:
                violations['tidal_force'] = float('inf')
            
            # Quantum decoherence
            coherence_time = config['quantum_coherence_time']
            violations['decoherence_rate'] = 1.0 / max(coherence_time, 1.0)
            
        except Exception as e:
            logger.warning(f"Constraint evaluation failed: {e}")
            # Set severe violations for failed evaluations
            violations = {
                'nec_violation': 1.0,
                'wec_violation': 1.0, 
                'sec_violation': 1.0,
                'tidal_force': 1e6,
                'decoherence_rate': 1.0
            }
        
        return violations
    
    def _evaluate_physics(self, circuit, config: Dict[str, float]) -> Dict[str, float]:
        """Evaluate physics metrics."""
        metrics = {}
        
        try:
            # Entanglement persistence
            evolution_data = circuit.time_evolve(time_steps=3, dt=0.5)
            initial_entanglement = evolution_data[0]['concurrence']
            final_entanglement = evolution_data[-1]['concurrence']
            
            metrics['entanglement_persistence'] = final_entanglement / (initial_entanglement + 1e-10)
            metrics['stability_time'] = config['quantum_coherence_time']
            
            # Quantum state quality
            state = circuit.create_traversal_state()
            metrics['entanglement_entropy'] = circuit.compute_entanglement_entropy(state)
            metrics['concurrence'] = circuit.compute_concurrence(state)
            
        except Exception as e:
            logger.warning(f"Physics evaluation failed: {e}")
            metrics = {
                'entanglement_persistence': 0.0,
                'stability_time': 0.0,
                'entanglement_entropy': 0.0,
                'concurrence': 0.0
            }
        
        return metrics
    
    def _evaluate_feasibility(self, config: Dict[str, float], 
                            violations: Dict[str, float]) -> float:
        """Evaluate engineering feasibility."""
        
        # Exotic matter energy requirement
        throat_radius = config['throat_radius']
        exotic_density = abs(config['exotic_matter_density'])
        throat_volume = 4/3 * np.pi * throat_radius**3
        exotic_energy = exotic_density * throat_volume
        
        # Feasibility factors
        energy_feasibility = np.exp(-exotic_energy / self.constraints.max_exotic_matter_energy)
        
        # Constraint satisfaction
        constraint_satisfaction = 1.0
        if violations['tidal_force'] > self.constraints.max_tidal_force:
            constraint_satisfaction *= 0.1
        if violations['decoherence_rate'] > self.constraints.max_decoherence_rate:
            constraint_satisfaction *= 0.5
        
        return energy_feasibility * constraint_satisfaction
    
    def _compute_constraint_penalty(self, violations: Dict[str, float]) -> float:
        """Compute penalty for constraint violations."""
        penalty = 0.0
        
        # Energy condition penalties
        if violations['nec_violation'] > self.constraints.max_nec_violation:
            penalty += self.penalty_weight * abs(violations['nec_violation'])
        
        if violations['wec_violation'] > self.constraints.max_wec_violation:
            penalty += self.penalty_weight * abs(violations['wec_violation'])
        
        # Tidal force penalty
        if violations['tidal_force'] > self.constraints.max_tidal_force:
            penalty += self.penalty_weight * (violations['tidal_force'] / self.constraints.max_tidal_force - 1)
        
        # Decoherence penalty
        if violations['decoherence_rate'] > self.constraints.max_decoherence_rate:
            penalty += self.penalty_weight * violations['decoherence_rate']
        
        return penalty
    
    def get_viable_candidates(self) -> List[WormholeCandidate]:
        """Get all physically viable wormhole candidates."""
        return [
            candidate for candidate in self.candidate_history
            if candidate.is_physically_viable(self.constraints)
        ]
    
    def get_best_candidate(self) -> Optional[WormholeCandidate]:
        """Get the best overall wormhole candidate."""
        if not self.candidate_history:
            return None
        
        return max(self.candidate_history, key=lambda c: c.get_overall_score())


class BayesianWormholeSearch:
    """Bayesian search for traversable wormhole solutions."""
    
    def __init__(self, 
                 quantum_circuit_factory: Callable,
                 constraints: Optional[TraversabilityConstraints] = None,
                 acquisition_function: str = 'expected_improvement'):
        """Initialize Bayesian wormhole search.
        
        Args:
            quantum_circuit_factory: Function to create quantum circuits
            constraints: Physical constraints (if None, uses defaults)
            acquisition_function: Acquisition function for Bayesian optimization
        """
        self.circuit_factory = quantum_circuit_factory
        self.constraints = constraints or TraversabilityConstraints()
        self.acquisition_function = acquisition_function
        
        # Setup specialized GP for wormhole physics
        kernel = (
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) *
            Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # Optimization state
        self.X_observed = []
        self.y_observed = []
        self.objective = None
        
    def search(self, 
               n_iterations: int = 100,
               n_initial: int = 20,
               parameter_bounds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run Bayesian search for traversable wormhole solutions.
        
        Args:
            n_iterations: Total search iterations
            n_initial: Initial random samples
            parameter_bounds: Parameter bounds array
            
        Returns:
            Search results dictionary
        """
        start_time = time.time()
        
        # Default parameter bounds for wormhole physics
        if parameter_bounds is None:
            parameter_bounds = np.array([
                [500.0, 3000.0],      # throat_radius (m)
                [1e29, 5e31],         # mass (kg)
                [-1e-2, -1e-4],       # exotic_matter_density (J/m³)
                [0.5, 0.99],          # traversal_probability
                [50.0, 500.0]         # quantum_coherence_time (s)
            ])
        
        # Initialize objective function
        self.objective = TraversabilityObjective(
            self.circuit_factory,
            self.constraints,
            penalty_weight=5.0
        )
        
        logger.info(f"Starting Bayesian wormhole search: {n_iterations} iterations")
        logger.info(f"Physical constraints: {self.constraints.to_dict()}")
        
        # Phase 1: Initial random exploration
        logger.info("Phase 1: Initial exploration...")
        for i in range(n_initial):
            x = self._random_sample(parameter_bounds)
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
            
            logger.info(f"Initial sample {i+1}/{n_initial}: score={y:.4f}")
            
            # Early termination if viable candidate found
            viable_candidates = self.objective.get_viable_candidates()
            if len(viable_candidates) >= 3:
                logger.info(f"Found {len(viable_candidates)} viable candidates, proceeding to exploitation phase")
                break
        
        # Phase 2: Bayesian optimization
        logger.info("Phase 2: Bayesian optimization...")
        viable_found = 0
        
        for iteration in range(n_initial, n_iterations):
            # Fit GP
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            self.gp.fit(X_array, y_array)
            
            # Adaptive acquisition strategy
            viable_candidates = self.objective.get_viable_candidates()
            if len(viable_candidates) < 5:
                # Exploration phase: find viable solutions
                acq_func = 'upper_confidence_bound'
                kappa = 3.0
            else:
                # Exploitation phase: optimize viable solutions
                acq_func = 'expected_improvement'
                kappa = 1.0
            
            # Find next point
            next_x = self._optimize_acquisition(parameter_bounds, acq_func, kappa)
            next_y = self.objective(next_x)
            
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            # Track viable candidates
            current_viable = len(self.objective.get_viable_candidates())
            if current_viable > viable_found:
                viable_found = current_viable
                logger.info(f"Iteration {iteration+1}: NEW VIABLE CANDIDATE found! "
                          f"Total viable: {viable_found}")
            
            if iteration % 10 == 0:
                best_score = max(self.y_observed)
                logger.info(f"Iteration {iteration+1}/{n_iterations}: "
                          f"score={next_y:.4f}, best={best_score:.4f}, "
                          f"viable={current_viable}")
        
        search_time = time.time() - start_time
        
        # Compile results
        results = self._compile_search_results(search_time)
        
        logger.info(f"Bayesian wormhole search completed in {search_time:.2f}s")
        logger.info(f"Total evaluations: {len(self.y_observed)}")
        logger.info(f"Viable candidates found: {len(results['viable_candidates'])}")
        
        return results
    
    def _random_sample(self, bounds: np.ndarray) -> np.ndarray:
        """Generate random sample within bounds."""
        return np.random.uniform(bounds[:, 0], bounds[:, 1])
    
    def _optimize_acquisition(self, bounds: np.ndarray, 
                            acq_func: str = None, kappa: float = 2.0) -> np.ndarray:
        """Optimize acquisition function."""
        
        if acq_func is None:
            acq_func = self.acquisition_function
        
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            
            if acq_func == 'expected_improvement':
                current_best = max(self.y_observed)
                improvement = mu - current_best
                z = improvement / (sigma + 1e-9)
                ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]
            
            elif acq_func == 'upper_confidence_bound':
                ucb = mu + kappa * sigma
                return -ucb[0]
            
            elif acq_func == 'probability_improvement':
                current_best = max(self.y_observed)
                z = (mu - current_best) / (sigma + 1e-9)
                pi = norm.cdf(z)
                return -pi[0]
            
            else:
                raise ValueError(f"Unknown acquisition function: {acq_func}")
        
        # Multi-start optimization
        best_x = None
        best_value = float('inf')
        
        for _ in range(15):  # More starts for critical optimization
            x0 = self._random_sample(bounds)
            
            result = minimize(
                acquisition,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_x = result.x
        
        return best_x if best_x is not None else self._random_sample(bounds)
    
    def _compile_search_results(self, search_time: float) -> Dict[str, Any]:
        """Compile comprehensive search results."""
        
        viable_candidates = self.objective.get_viable_candidates()
        best_candidate = self.objective.get_best_candidate()
        
        results = {
            'search_time': search_time,
            'total_evaluations': len(self.y_observed),
            'viable_candidates': [
                {
                    'parameters': candidate.parameters,
                    'traversability_score': candidate.traversability_score,
                    'feasibility_score': candidate.feasibility_score,
                    'overall_score': candidate.get_overall_score(),
                    'constraint_violations': candidate.constraint_violations,
                    'physics_metrics': candidate.physics_metrics,
                    'discovery_iteration': candidate.discovery_iteration
                }
                for candidate in viable_candidates
            ],
            'best_candidate': None,
            'search_statistics': {
                'best_score': max(self.y_observed) if self.y_observed else 0,
                'mean_score': np.mean(self.y_observed) if self.y_observed else 0,
                'std_score': np.std(self.y_observed) if self.y_observed else 0,
                'viable_fraction': len(viable_candidates) / len(self.y_observed) if self.y_observed else 0
            },
            'constraints_used': self.constraints.to_dict(),
            'optimization_history': [
                {
                    'iteration': i+1,
                    'score': score,
                    'parameters': param.tolist()
                }
                for i, (param, score) in enumerate(zip(self.X_observed, self.y_observed))
            ]
        }
        
        if best_candidate:
            results['best_candidate'] = {
                'parameters': best_candidate.parameters,
                'traversability_score': best_candidate.traversability_score,
                'feasibility_score': best_candidate.feasibility_score,
                'overall_score': best_candidate.get_overall_score(),
                'constraint_violations': best_candidate.constraint_violations,
                'physics_metrics': best_candidate.physics_metrics,
                'is_viable': best_candidate.is_physically_viable(self.constraints)
            }
        
        return results


def save_search_results(results: Dict[str, Any], filename: str):
    """Save Bayesian search results to file."""
    
    # Ensure all values are JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Bayesian search results saved to {filename}")


def create_wormhole_circuit_factory(num_qubits: int = 4):
    """Create quantum circuit factory for wormhole search."""
    
    def factory(config: Dict[str, float]):
        """Create quantum circuit with wormhole configuration."""
        try:
            from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
            
            geometry_params = {
                'throat_radius': config['throat_radius'],
                'mass': config['mass'],
                'exotic_matter_density': config['exotic_matter_density'],
                'traversal_probability': config['traversal_probability']
            }
            
            return HybridQuantumAICircuit(num_qubits, geometry_params)
            
        except Exception as e:
            logger.error(f"Failed to create wormhole circuit: {e}")
            raise
    
    return factory