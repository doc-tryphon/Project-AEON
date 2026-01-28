"""
ML-driven Parameter Space Exploration for Wormhole Optimization.

This module implements various machine learning optimizers to systematically
explore the wormhole parameter space and find optimal traversable configurations.
"""

import numpy as np
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.optimize import differential_evolution, minimize
from typing import Dict, List, Tuple, Callable, Optional, Any
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)


@dataclass
class ParameterBounds:
    """Parameter bounds for optimization."""
    throat_radius: Tuple[float, float] = (100.0, 5000.0)  # meters
    mass: Tuple[float, float] = (1e29, 1e32)  # kg  
    exotic_matter_density: Tuple[float, float] = (-1e-2, -1e-4)  # J/m³
    traversal_probability: Tuple[float, float] = (0.1, 0.99)
    quantum_coherence_time: Tuple[float, float] = (10.0, 1000.0)  # seconds
    
    def to_bounds_array(self) -> np.ndarray:
        """Convert to scipy-compatible bounds array."""
        return np.array([
            self.throat_radius,
            self.mass,
            self.exotic_matter_density,
            self.traversal_probability,
            self.quantum_coherence_time
        ])
    
    def get_parameter_names(self) -> List[str]:
        """Get ordered parameter names."""
        return [
            'throat_radius',
            'mass', 
            'exotic_matter_density',
            'traversal_probability',
            'quantum_coherence_time'
        ]


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_parameters: Dict[str, float]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    converged: bool
    method: str


class WormholeObjectiveFunction:
    """Objective function for wormhole parameter optimization."""
    
    def __init__(self, quantum_circuit_factory: Callable, 
                 weights: Optional[Dict[str, float]] = None):
        """Initialize objective function.
        
        Args:
            quantum_circuit_factory: Function to create quantum circuits
            weights: Weights for different optimization criteria
        """
        self.circuit_factory = quantum_circuit_factory
        self.weights = weights or {
            'traversability': 0.4,
            'stability': 0.3,
            'entanglement': 0.2,
            'efficiency': 0.1
        }
        self.evaluation_count = 0
        self.evaluation_history = []
        
    def __call__(self, parameters: np.ndarray) -> float:
        """Evaluate wormhole configuration.
        
        Args:
            parameters: [throat_radius, mass, exotic_matter_density, 
                        traversal_probability, quantum_coherence_time]
                        
        Returns:
            Objective score (higher is better)
        """
        self.evaluation_count += 1
        start_time = time.time()
        
        try:
            # Convert parameters to configuration
            config = self._parameters_to_config(parameters)
            
            # Create quantum circuit
            circuit = self.circuit_factory(config)
            
            # Evaluate multiple criteria
            scores = self._evaluate_configuration(circuit, config)
            
            # Compute weighted objective
            objective = sum(self.weights[key] * scores[key] for key in self.weights)
            
            # Add to history
            evaluation_time = time.time() - start_time
            self.evaluation_history.append({
                'parameters': parameters.copy(),
                'config': config.copy(),
                'scores': scores.copy(),
                'objective': objective,
                'evaluation_time': evaluation_time,
                'evaluation_id': self.evaluation_count
            })
            
            logger.debug(f"Evaluation {self.evaluation_count}: objective={objective:.6f}, time={evaluation_time:.3f}s")
            
            return objective
            
        except Exception as e:
            logger.warning(f"Evaluation {self.evaluation_count} failed: {e}")
            # Return poor score for failed evaluations
            return -1000.0
    
    def _parameters_to_config(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to configuration dictionary."""
        param_names = ParameterBounds().get_parameter_names()
        return {name: float(param) for name, param in zip(param_names, parameters)}
    
    def _evaluate_configuration(self, circuit, config: Dict[str, float]) -> Dict[str, float]:
        """Evaluate wormhole configuration on multiple criteria."""
        
        # 1. Traversability score
        try:
            state = circuit.create_traversal_state(use_ai_params=True)
            traversability = circuit.predict_traversability(state)
        except:
            traversability = 0.0
        
        # 2. Stability score (based on entanglement preservation)
        try:
            evolution_data = circuit.time_evolve(time_steps=5, dt=0.1)
            initial_concurrence = evolution_data[0]['concurrence']
            final_concurrence = evolution_data[-1]['concurrence']
            stability = final_concurrence / (initial_concurrence + 1e-10)
        except:
            stability = 0.0
        
        # 3. Entanglement score
        try:
            entropy = circuit.compute_entanglement_entropy()
            concurrence = circuit.compute_concurrence()
            entanglement = (entropy + concurrence) / 2.0
        except:
            entanglement = 0.0
        
        # 4. Efficiency score (inverse of required exotic matter)
        exotic_density = abs(config['exotic_matter_density'])
        efficiency = 1.0 / (1.0 + exotic_density * 1e3)  # Normalize
        
        return {
            'traversability': np.clip(traversability, 0, 1),
            'stability': np.clip(stability, 0, 1),
            'entanglement': np.clip(entanglement, 0, 1),
            'efficiency': np.clip(efficiency, 0, 1)
        }


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(self, objective_function: WormholeObjectiveFunction,
                 parameter_bounds: ParameterBounds,
                 kernel: Optional[Any] = None,
                 acquisition_function: str = 'expected_improvement'):
        """Initialize Bayesian optimizer.
        
        Args:
            objective_function: Objective function to optimize
            parameter_bounds: Parameter bounds
            kernel: GP kernel (if None, uses default)
            acquisition_function: Acquisition function type
        """
        self.objective = objective_function
        self.bounds = parameter_bounds
        self.acquisition_function = acquisition_function
        
        # Setup Gaussian Process
        if kernel is None:
            kernel = RBF(length_scale=1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
    
    def optimize(self, n_iterations: int = 50, n_initial: int = 10) -> OptimizationResult:
        """Run Bayesian optimization.
        
        Args:
            n_iterations: Total iterations
            n_initial: Initial random samples
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        bounds_array = self.bounds.to_bounds_array()
        
        logger.info(f"Starting Bayesian optimization: {n_iterations} iterations, {n_initial} initial samples")
        
        # Initial random sampling
        logger.info("Generating initial random samples...")
        for i in range(n_initial):
            x = self._random_sample(bounds_array)
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
            logger.info(f"Initial sample {i+1}/{n_initial}: score={y:.6f}")
        
        # Bayesian optimization loop
        for iteration in range(n_initial, n_iterations):
            logger.info(f"Bayesian iteration {iteration+1}/{n_iterations}")
            
            # Fit GP to current observations
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            self.gp.fit(X_array, y_array)
            
            # Find next point using acquisition function
            next_x = self._optimize_acquisition(bounds_array)
            next_y = self.objective(next_x)
            
            # Add to observations
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            current_best = max(self.y_observed)
            logger.info(f"  Next point score: {next_y:.6f}, Best so far: {current_best:.6f}")
        
        # Find best result
        best_idx = np.argmax(self.y_observed)
        best_x = self.X_observed[best_idx]
        best_y = self.y_observed[best_idx]
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=self._parameters_to_dict(best_x),
            best_score=best_y,
            optimization_history=self.objective.evaluation_history,
            total_evaluations=len(self.y_observed),
            optimization_time=optimization_time,
            converged=True,  # Bayesian optimization always "converges"
            method="Bayesian_GP"
        )
    
    def _random_sample(self, bounds: np.ndarray) -> np.ndarray:
        """Generate random sample within bounds."""
        return np.random.uniform(bounds[:, 0], bounds[:, 1])
    
    def _optimize_acquisition(self, bounds: np.ndarray) -> np.ndarray:
        """Optimize acquisition function to find next point."""
        
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            
            if self.acquisition_function == 'expected_improvement':
                # Expected Improvement
                current_best = max(self.y_observed)
                improvement = mu - current_best
                z = improvement / (sigma + 1e-9)
                ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
                return -ei[0]  # Minimize negative EI
            
            elif self.acquisition_function == 'upper_confidence_bound':
                # Upper Confidence Bound
                kappa = 2.0  # Exploration parameter
                ucb = mu + kappa * sigma
                return -ucb[0]  # Minimize negative UCB
            
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        # Multi-start optimization of acquisition function
        best_x = None
        best_value = float('inf')
        
        for _ in range(10):  # Multiple random starts
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
        
        if best_x is None:
            # Fallback to random sample
            best_x = self._random_sample(bounds)
        
        return best_x
    
    def _normal_cdf(self, x):
        """Standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names = self.bounds.get_parameter_names()
        return {name: float(param) for name, param in zip(param_names, parameters)}


class DifferentialEvolutionOptimizer:
    """Differential Evolution optimizer for global search."""
    
    def __init__(self, objective_function: WormholeObjectiveFunction,
                 parameter_bounds: ParameterBounds):
        """Initialize DE optimizer."""
        self.objective = objective_function
        self.bounds = parameter_bounds
    
    def optimize(self, max_evaluations: int = 100, population_size: int = 15) -> OptimizationResult:
        """Run differential evolution optimization."""
        start_time = time.time()
        bounds_array = self.bounds.to_bounds_array()
        
        logger.info(f"Starting Differential Evolution: {max_evaluations} evaluations, population {population_size}")
        
        # Wrapper to minimize (DE minimizes, we want to maximize)
        def minimize_objective(x):
            return -self.objective(x)
        
        result = differential_evolution(
            minimize_objective,
            bounds_array,
            maxiter=max_evaluations // population_size,
            popsize=population_size,
            seed=42,
            disp=True
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=self._parameters_to_dict(result.x),
            best_score=-result.fun,  # Convert back to maximization
            optimization_history=self.objective.evaluation_history,
            total_evaluations=result.nfev,
            optimization_time=optimization_time,
            converged=result.success,
            method="Differential_Evolution"
        )
    
    def _parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names = self.bounds.get_parameter_names()
        return {name: float(param) for name, param in zip(param_names, parameters)}


class GridSearchOptimizer:
    """Grid search optimizer for systematic exploration."""
    
    def __init__(self, objective_function: WormholeObjectiveFunction,
                 parameter_bounds: ParameterBounds):
        """Initialize grid search optimizer."""
        self.objective = objective_function
        self.bounds = parameter_bounds
    
    def optimize(self, grid_points_per_dim: int = 5) -> OptimizationResult:
        """Run grid search optimization."""
        start_time = time.time()
        bounds_array = self.bounds.to_bounds_array()
        param_names = self.bounds.get_parameter_names()
        
        logger.info(f"Starting Grid Search: {grid_points_per_dim}^{len(param_names)} = {grid_points_per_dim**len(param_names)} evaluations")
        
        # Generate grid points
        grid_ranges = []
        for i, (low, high) in enumerate(bounds_array):
            grid_ranges.append(np.linspace(low, high, grid_points_per_dim))
        
        # Evaluate all grid points
        best_score = -float('inf')
        best_params = None
        total_evaluations = 0
        
        # Use meshgrid for systematic evaluation
        grids = np.meshgrid(*grid_ranges, indexing='ij')
        grid_points = np.array([grid.ravel() for grid in grids]).T
        
        logger.info(f"Evaluating {len(grid_points)} grid points...")
        
        for i, point in enumerate(grid_points):
            score = self.objective(point)
            total_evaluations += 1
            
            if score > best_score:
                best_score = score
                best_params = point.copy()
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{len(grid_points)}, best score: {best_score:.6f}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=self._parameters_to_dict(best_params),
            best_score=best_score,
            optimization_history=self.objective.evaluation_history,
            total_evaluations=total_evaluations,
            optimization_time=optimization_time,
            converged=True,
            method="Grid_Search"
        )
    
    def _parameters_to_dict(self, parameters: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names = self.bounds.get_parameter_names()
        return {name: float(param) for name, param in zip(param_names, parameters)}


class ParallelOptimizer:
    """Parallel optimization wrapper for multi-threaded exploration."""
    
    def __init__(self, base_optimizer_class, n_workers: int = 4):
        """Initialize parallel optimizer.
        
        Args:
            base_optimizer_class: Base optimizer class to parallelize
            n_workers: Number of parallel workers
        """
        self.base_optimizer_class = base_optimizer_class
        self.n_workers = n_workers
    
    def optimize(self, objective_function: WormholeObjectiveFunction,
                 parameter_bounds: ParameterBounds,
                 **optimizer_kwargs) -> List[OptimizationResult]:
        """Run parallel optimization."""
        
        logger.info(f"Starting parallel optimization with {self.n_workers} workers")
        
        # Create multiple optimizer instances
        def run_single_optimization(worker_id):
            logger.info(f"Worker {worker_id} starting optimization")
            
            # Create separate objective function for each worker to avoid conflicts
            worker_objective = WormholeObjectiveFunction(
                objective_function.circuit_factory,
                objective_function.weights
            )
            
            optimizer = self.base_optimizer_class(worker_objective, parameter_bounds)
            result = optimizer.optimize(**optimizer_kwargs)
            
            logger.info(f"Worker {worker_id} completed: best_score={result.best_score:.6f}")
            return result
        
        # Run optimizations in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(run_single_optimization, i) 
                for i in range(self.n_workers)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
        
        return results


def create_quantum_circuit_factory(num_qubits: int = 4):
    """Create a quantum circuit factory function."""
    
    def factory(config: Dict[str, float]):
        """Create quantum circuit with given configuration."""
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
            logger.error(f"Failed to create quantum circuit: {e}")
            raise
    
    return factory


def run_optimization_comparison(optimization_budget: int = 50) -> Dict[str, OptimizationResult]:
    """Run comparison of different optimization methods."""
    
    logger.info("Starting optimization method comparison")
    
    # Setup
    circuit_factory = create_quantum_circuit_factory(num_qubits=4)
    objective = WormholeObjectiveFunction(circuit_factory)
    bounds = ParameterBounds()
    
    results = {}
    
    # 1. Bayesian Optimization
    try:
        logger.info("Running Bayesian optimization...")
        bayesian_opt = BayesianOptimizer(objective, bounds)
        results['Bayesian'] = bayesian_opt.optimize(n_iterations=optimization_budget)
        logger.info(f"Bayesian completed: best_score={results['Bayesian'].best_score:.6f}")
    except Exception as e:
        logger.error(f"Bayesian optimization failed: {e}")
    
    # Reset objective function
    objective.evaluation_count = 0
    objective.evaluation_history = []
    
    # 2. Differential Evolution
    try:
        logger.info("Running Differential Evolution...")
        de_opt = DifferentialEvolutionOptimizer(objective, bounds)
        results['Differential_Evolution'] = de_opt.optimize(max_evaluations=optimization_budget)
        logger.info(f"DE completed: best_score={results['Differential_Evolution'].best_score:.6f}")
    except Exception as e:
        logger.error(f"Differential Evolution failed: {e}")
    
    # Reset objective function
    objective.evaluation_count = 0
    objective.evaluation_history = []
    
    # 3. Grid Search (smaller grid due to budget)
    try:
        logger.info("Running Grid Search...")
        grid_opt = GridSearchOptimizer(objective, bounds)
        grid_points = max(2, int(optimization_budget**(1/5)))  # 5D grid
        results['Grid_Search'] = grid_opt.optimize(grid_points_per_dim=grid_points)
        logger.info(f"Grid Search completed: best_score={results['Grid_Search'].best_score:.6f}")
    except Exception as e:
        logger.error(f"Grid Search failed: {e}")
    
    return results


def save_optimization_results(results: Dict[str, OptimizationResult], filename: str):
    """Save optimization results to file."""
    
    # Convert results to JSON-serializable format
    json_results = {}
    
    for method, result in results.items():
        json_results[method] = {
            'best_parameters': result.best_parameters,
            'best_score': result.best_score,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'converged': result.converged,
            'method': result.method,
            'optimization_history': result.optimization_history
        }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Optimization results saved to {filename}")