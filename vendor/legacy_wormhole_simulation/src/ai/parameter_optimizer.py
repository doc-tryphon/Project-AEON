"""
Optimization algorithms for wormhole parameter tuning.

This module implements various optimization techniques including:
- Gradient-based optimization
- Evolutionary algorithms
- Particle swarm optimization
to find optimal wormhole configurations.
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class OptimizationObjective:
    """Objective function configuration for optimization."""
    stability_weight: float = 1.0
    energy_weight: float = 0.5
    size_weight: float = 0.3
    entanglement_weight: float = 0.2


class ParameterOptimizer:
    """Optimizes wormhole parameters using multiple strategies."""
    
    def __init__(self,
                objective: OptimizationObjective,
                parameter_bounds: Dict[str, Tuple[float, float]]):
        """Initialize parameter optimizer.
        
        Args:
            objective: Optimization objective configuration
            parameter_bounds: Min/max bounds for each parameter
        """
        self.objective = objective
        self.bounds = parameter_bounds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize parameters for gradient-based optimization
        self.parameters = {
            name: torch.nn.Parameter(
                torch.FloatTensor([(bounds[1] + bounds[0])/2]).to(self.device)
            )
            for name, bounds in parameter_bounds.items()
        }
        
        # Initialize evolutionary algorithm population
        self.population_size = 100
        self.population = self._initialize_population()
        
    def _initialize_population(self) -> List[Dict[str, float]]:
        """Initialize random population for evolutionary optimization."""
        population = []
        for _ in range(self.population_size):
            individual = {
                name: np.random.uniform(low, high)
                for name, (low, high) in self.bounds.items()
            }
            population.append(individual)
        return population
        
    def objective_function(self, params: Dict[str, Union[torch.Tensor, float]]) -> Union[torch.Tensor, float]:
        """Calculate optimization objective value."""
        # Convert numpy/float parameters to tensors if needed
        if not isinstance(next(iter(params.values())), torch.Tensor):
            params = {
                name: torch.tensor([float(value)], device=self.device)
                for name, value in params.items()
            }
            
        # Calculate components
        stability_loss = self._calculate_stability_loss(params)
        energy_loss = self._calculate_energy_loss(params)
        size_loss = self._calculate_size_loss(params)
        entanglement_loss = self._calculate_entanglement_loss(params)
        
        # Weighted sum
        total_loss = (
            self.objective.stability_weight * stability_loss +
            self.objective.energy_weight * energy_loss +
            self.objective.size_weight * size_loss +
            self.objective.entanglement_weight * entanglement_loss
        )
        
        return total_loss
        
    def _calculate_stability_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate stability component of loss."""
        mass = params['mass']
        radius = params['throat_radius']
        
        # Stability metrics
        schwarzschild_radius = 2 * mass  # Simplified units G=c=1
        stability_metric = torch.relu(schwarzschild_radius - radius)
        
        return stability_metric
        
    def _calculate_energy_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate energy efficiency component of loss."""
        mass = params['mass']
        radius = params['throat_radius']
        energy = mass * radius
        return energy
        
    def _calculate_size_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate size optimization component of loss."""
        radius = params['throat_radius']
        target_radius = torch.tensor([1.0], device=self.device)
        return torch.abs(radius - target_radius)
        
    def _calculate_entanglement_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate quantum properties component of loss."""
        radius = params['throat_radius']
        mass = params['mass']
        entanglement_metric = 1 / (radius * mass)
        return entanglement_metric
        
    def optimize_gradient(self, 
                        num_steps: int = 1000,
                        learning_rate: float = 0.01) -> Dict[str, float]:
        """Run gradient-based optimization."""
        optimizer = optim.Adam(self.parameters.values(), lr=learning_rate)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.objective_function(self.parameters)
            loss.backward()
            optimizer.step()
            
            # Project to bounds
            with torch.no_grad():
                for name, param in self.parameters.items():
                    min_val, max_val = self.bounds[name]
                    param.data.clamp_(min_val, max_val)
                    
        return {
            name: float(param.data)
            for name, param in self.parameters.items()
        }
        
    def optimize_evolutionary(self, 
                           num_generations: int = 100,
                           mutation_rate: float = 0.1) -> Dict[str, float]:
        """Run evolutionary optimization."""
        for generation in range(num_generations):
            # Evaluate fitness
            fitness = [
                -float(self.objective_function(individual))  # Negative because we maximize fitness
                for individual in self.population
            ]
            
            # Select parents
            parent_indices = np.argsort(fitness)[-self.population_size//2:]
            parents = [self.population[i] for i in parent_indices]
            
            # Create next generation
            next_population = parents.copy()
            while len(next_population) < self.population_size:
                # Crossover
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                child = {}
                for param_name in self.bounds:
                    if np.random.random() < 0.5:
                        child[param_name] = parent1[param_name]
                    else:
                        child[param_name] = parent2[param_name]
                        
                # Mutation
                for param_name, (low, high) in self.bounds.items():
                    if np.random.random() < mutation_rate:
                        child[param_name] = np.random.uniform(low, high)
                        
                next_population.append(child)
                
            self.population = next_population
            
        # Return best individual
        best_idx = np.argmax([
            -float(self.objective_function(individual))
            for individual in self.population
        ])
        return self.population[best_idx]
        
    def optimize(self, method: str = 'both') -> Dict[str, float]:
        """Run optimization using specified method(s).
        
        Args:
            method: One of 'gradient', 'evolutionary', or 'both'
            
        Returns:
            Best parameters found
        """
        if method == 'gradient':
            return self.optimize_gradient()
        elif method == 'evolutionary':
            return self.optimize_evolutionary()
        else:
            # Run both and take best result
            grad_result = self.optimize_gradient()
            evo_result = self.optimize_evolutionary()
            
            grad_loss = float(self.objective_function(grad_result))
            evo_loss = float(self.objective_function(evo_result))
            
            return grad_result if grad_loss < evo_loss else evo_result
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import warnings
warnings.filterwarnings('ignore')

from src.physics.exotic_matter import ExoticMatter, optimize_exotic_matter_distribution
from src.physics.spacetime_metrics import SpacetimeMetric
from src.ai.stability_predictor import BaseStabilityPredictor


@dataclass
class OptimizationBounds:
    """Parameter bounds for optimization."""
    
    throat_radius: Tuple[float, float] = (1e3, 1e6)  # meters
    exotic_energy_scale: Tuple[float, float] = (-1e20, -1e10)  # J/m³
    shape_parameter: Tuple[float, float] = (0.1, 10.0)
    coupling_strength: Tuple[float, float] = (0.01, 10.0)
    temperature: Tuple[float, float] = (1e-10, 1e-5)  # Kelvin
    mass_parameter: Tuple[float, float] = (1e20, 1e35)  # kg
    quantum_correction: Tuple[float, float] = (0.0, 1.0)


@dataclass
class OptimizationResult:
    """Results of parameter optimization."""
    
    best_parameters: Dict[str, float]
    best_fitness: float
    convergence_history: List[float]
    population_history: List[List[Dict[str, float]]]
    optimization_time: float
    function_evaluations: int
    success: bool
    termination_reason: str


class ObjectiveFunction:
    """Multi-objective function for wormhole optimization."""
    
    def __init__(self, stability_predictor: Optional[BaseStabilityPredictor] = None,
                 physics_simulator: Optional[Callable] = None,
                 weights: Dict[str, float] = None):
        """Initialize objective function.
        
        Args:
            stability_predictor: Trained stability prediction model
            physics_simulator: Physics simulation function
            weights: Weights for different objectives
        """
        self.stability_predictor = stability_predictor
        self.physics_simulator = physics_simulator
        
        # Default weights for multi-objective optimization
        default_weights = {
            'stability': 0.4,
            'energy_efficiency': 0.3,
            'traversability': 0.2,
            'quantum_consistency': 0.1
        }
        
        self.weights = weights or default_weights
        self.evaluation_count = 0
        self.evaluation_history = []
    
    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate objective function for given parameters.
        
        Args:
            parameters: Dictionary of wormhole parameters
        
        Returns:
            Fitness value (higher is better)
        """
        self.evaluation_count += 1
        
        try:
            # Individual objective components
            stability_score = self._evaluate_stability(parameters)
            energy_score = self._evaluate_energy_efficiency(parameters)
            traversability_score = self._evaluate_traversability(parameters)
            quantum_score = self._evaluate_quantum_consistency(parameters)
            
            # Weighted combination
            total_fitness = (
                self.weights['stability'] * stability_score +
                self.weights['energy_efficiency'] * energy_score +
                self.weights['traversability'] * traversability_score +
                self.weights['quantum_consistency'] * quantum_score
            )
            
            # Store evaluation
            evaluation_record = {
                'parameters': parameters.copy(),
                'fitness': total_fitness,
                'stability': stability_score,
                'energy_efficiency': energy_score,
                'traversability': traversability_score,
                'quantum_consistency': quantum_score
            }
            
            self.evaluation_history.append(evaluation_record)
            
            return total_fitness
            
        except Exception as e:
            # Return penalty for invalid parameters
            return -1000.0
    
    def _evaluate_stability(self, parameters: Dict[str, float]) -> float:
        """Evaluate stability score."""
        
        if self.stability_predictor and self.stability_predictor.is_trained:
            # Use trained ML model for stability prediction
            features = self._parameters_to_features(parameters)
            stability_prob = self.stability_predictor.predict_proba(features.reshape(1, -1))[0][0]
            return stability_prob
        else:
            # Fallback: physics-based heuristics
            throat_r = parameters.get('throat_radius', 1e4)
            exotic_rho = parameters.get('exotic_energy_scale', -1e15)
            shape_param = parameters.get('shape_parameter', 1.0)
            
            # Simple stability heuristics
            flare_out_condition = max(0, 1 - shape_param / throat_r * 1e-3)
            energy_condition = min(1, abs(exotic_rho) / 1e15)
            geometric_stability = min(1, throat_r / 1e4)
            
            return (flare_out_condition + energy_condition + geometric_stability) / 3
    
    def _evaluate_energy_efficiency(self, parameters: Dict[str, float]) -> float:
        """Evaluate energy efficiency (lower energy requirement is better)."""
        
        throat_r = parameters.get('throat_radius', 1e4)
        exotic_rho = parameters.get('exotic_energy_scale', -1e15)
        quantum_corr = parameters.get('quantum_correction', 0.1)
        
        # Total energy requirement estimate
        volume = 4 * np.pi * throat_r**3 / 3
        total_energy = abs(exotic_rho) * volume * (1 + quantum_corr)
        
        # Normalize and invert (lower energy is better)
        normalized_energy = total_energy / 1e25  # Reference energy scale
        energy_efficiency = 1 / (1 + normalized_energy)
        
        return energy_efficiency
    
    def _evaluate_traversability(self, parameters: Dict[str, float]) -> float:
        """Evaluate traversability score."""
        
        throat_r = parameters.get('throat_radius', 1e4)
        shape_param = parameters.get('shape_parameter', 1.0)
        temperature = parameters.get('temperature', 1e-8)
        
        # Traversability factors
        throat_size_factor = min(1, throat_r / 1e3)  # Larger throat is better
        geometry_factor = max(0, 1 - shape_param / 10)  # Smoother geometry is better
        thermal_factor = max(0, 1 - temperature * 1e8)  # Lower temperature is better
        
        return (throat_size_factor + geometry_factor + thermal_factor) / 3
    
    def _evaluate_quantum_consistency(self, parameters: Dict[str, float]) -> float:
        """Evaluate quantum consistency."""
        
        throat_r = parameters.get('throat_radius', 1e4)
        exotic_rho = parameters.get('exotic_energy_scale', -1e15)
        quantum_corr = parameters.get('quantum_correction', 0.1)
        
        # Planck scale consistency
        planck_length = 1.616e-35  # meters
        planck_energy_density = 4.6e113  # J/m³
        
        # Check if parameters are within reasonable quantum bounds
        size_consistency = min(1, throat_r / (1000 * planck_length))
        energy_consistency = min(1, abs(exotic_rho) / planck_energy_density)
        correction_consistency = 1 - abs(quantum_corr - 0.1) / 0.9  # Prefer moderate corrections
        
        return (size_consistency + energy_consistency + correction_consistency) / 3
    
    def _parameters_to_features(self, parameters: Dict[str, float]) -> np.ndarray:
        """Convert parameters to feature vector for ML model."""
        
        # Extract parameters with defaults
        throat_r = parameters.get('throat_radius', 1e4)
        exotic_rho = parameters.get('exotic_energy_scale', -1e15)
        shape_param = parameters.get('shape_parameter', 1.0)
        coupling = parameters.get('coupling_strength', 1.0)
        temperature = parameters.get('temperature', 1e-8)
        mass = parameters.get('mass_parameter', 1e30)
        quantum_corr = parameters.get('quantum_correction', 0.1)
        
        # Compute derived features (simplified)
        energy_density = abs(exotic_rho)
        radial_pressure = -exotic_rho * (1 + quantum_corr)
        tangential_pressure = -exotic_rho * 0.5
        stress_anisotropy = abs(radial_pressure - tangential_pressure) / energy_density
        
        ricci_scalar = -6 * shape_param / throat_r**2
        kretschmann_scalar = 48 * shape_param**2 / throat_r**4
        
        flare_out = max(0, 1 - shape_param / throat_r)
        tidal_forces = kretschmann_scalar * throat_r**2
        
        # Feature vector matching stability predictor input
        features = np.array([
            energy_density, radial_pressure, tangential_pressure,
            stress_anisotropy, ricci_scalar, kretschmann_scalar,
            ricci_scalar/3, exotic_rho, radial_pressure/energy_density,
            1.0, throat_r, -shape_param/throat_r,  # energy condition violations count
            flare_out, quantum_corr * energy_density, quantum_corr * 1e-10,
            temperature, tidal_forces, abs(ricci_scalar) * throat_r,
            np.sqrt(abs(ricci_scalar))
        ])
        
        return features


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, objective_function: ObjectiveFunction,
                 bounds: OptimizationBounds = None):
        """Initialize optimizer.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
        """
        self.objective_function = objective_function
        self.bounds = bounds or OptimizationBounds()
        
    @abstractmethod
    def optimize(self, max_iterations: int = 1000, **kwargs) -> OptimizationResult:
        """Run optimization algorithm."""
        pass
    
    def _bounds_to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert bounds to arrays for scipy optimizers."""
        
        param_names = [
            'throat_radius', 'exotic_energy_scale', 'shape_parameter',
            'coupling_strength', 'temperature', 'mass_parameter', 'quantum_correction'
        ]
        
        lower_bounds = []
        upper_bounds = []
        
        for name in param_names:
            bound = getattr(self.bounds, name)
            lower_bounds.append(bound[0])
            upper_bounds.append(bound[1])
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def _array_to_parameters(self, x: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dictionary."""
        
        param_names = [
            'throat_radius', 'exotic_energy_scale', 'shape_parameter',
            'coupling_strength', 'temperature', 'mass_parameter', 'quantum_correction'
        ]
        
        return {name: float(val) for name, val in zip(param_names, x)}
    
    def _parameters_to_array(self, parameters: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        
        param_names = [
            'throat_radius', 'exotic_energy_scale', 'shape_parameter',
            'coupling_strength', 'temperature', 'mass_parameter', 'quantum_correction'
        ]
        
        return np.array([parameters.get(name, 0.0) for name in param_names])


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic algorithm for wormhole parameter optimization."""
    
    def __init__(self, objective_function: ObjectiveFunction,
                 bounds: OptimizationBounds = None,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """Initialize genetic algorithm.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        super().__init__(objective_function, bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def optimize(self, max_iterations: int = 1000, **kwargs) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        
        import time
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            parameters = self._array_to_parameters(individual)
            fitness = self.objective_function.evaluate(parameters)
            fitness_scores.append(fitness)
        
        # Evolution tracking
        convergence_history = []
        population_history = []
        best_fitness = max(fitness_scores)
        best_individual = population[np.argmax(fitness_scores)]
        
        # Evolution loop
        for generation in range(max_iterations):
            
            # Selection
            selected_parents = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected_parents) - 1, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self._simulated_binary_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    child1 = self._polynomial_mutation(child1)
                if np.random.rand() < self.mutation_rate:
                    child2 = self._polynomial_mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Ensure offspring size matches population size
            offspring = offspring[:self.population_size]
            
            # Evaluate offspring
            offspring_fitness = []
            for individual in offspring:
                parameters = self._array_to_parameters(individual)
                fitness = self.objective_function.evaluate(parameters)
                offspring_fitness.append(fitness)
            
            # Environmental selection (elitism + tournament)
            combined_population = population + offspring
            combined_fitness = fitness_scores + offspring_fitness
            
            # Keep best individuals
            elite_indices = np.argsort(combined_fitness)[-self.population_size:]
            population = [combined_population[i] for i in elite_indices]
            fitness_scores = [combined_fitness[i] for i in elite_indices]
            
            # Update best solution
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmax(fitness_scores)]
            
            # Record progress
            convergence_history.append(best_fitness)
            population_params = [self._array_to_parameters(ind) for ind in population]
            population_history.append(population_params)
            
            # Early stopping check
            if len(convergence_history) > 50:
                recent_improvement = convergence_history[-1] - convergence_history[-50]
                if abs(recent_improvement) < 1e-6:
                    termination_reason = "Early convergence"
                    break
        else:
            termination_reason = "Maximum iterations reached"
        
        # Results
        optimization_time = time.time() - start_time
        best_parameters = self._array_to_parameters(best_individual)
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_fitness=best_fitness,
            convergence_history=convergence_history,
            population_history=population_history,
            optimization_time=optimization_time,
            function_evaluations=self.objective_function.evaluation_count,
            success=True,
            termination_reason=termination_reason
        )
    
    def _initialize_population(self) -> List[np.ndarray]:
        """Initialize random population within bounds."""
        
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        population = []
        
        for _ in range(self.population_size):
            individual = np.random.uniform(lower_bounds, upper_bounds)
            population.append(individual)
        
        return population
    
    def _tournament_selection(self, population: List[np.ndarray],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List[np.ndarray]:
        """Tournament selection for parent selection."""
        
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select best from tournament
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        
        return selected
    
    def _simulated_binary_crossover(self, parent1: np.ndarray, 
                                  parent2: np.ndarray,
                                  eta_c: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover (SBX)."""
        
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:  # Crossover probability per gene
                
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    
                    # Calculate beta
                    rand = np.random.rand()
                    
                    if rand <= 0.5:
                        beta = (2.0 * rand) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta_c + 1.0))
                    
                    # Generate offspring
                    child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i])
                    
                    # Ensure bounds
                    child1[i] = np.clip(child1[i], lower_bounds[i], upper_bounds[i])
                    child2[i] = np.clip(child2[i], lower_bounds[i], upper_bounds[i])
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray, 
                           eta_m: float = 20.0) -> np.ndarray:
        """Polynomial mutation."""
        
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.rand() <= (1.0 / len(individual)):  # Mutation probability per gene
                
                val = individual[i]
                lb = lower_bounds[i]
                ub = upper_bounds[i]
                
                delta1 = (val - lb) / (ub - lb)
                delta2 = (ub - val) / (ub - lb)
                
                rand = np.random.rand()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val_new = val - delta1 * ((2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1.0)) ** mut_pow - 1.0)
                else:
                    xy = 1.0 - delta2
                    val_new = val + delta2 * (1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1.0)) ** mut_pow)
                
                mutated[i] = np.clip(val_new, lb, ub)
        
        return mutated


class ParticleSwarmOptimizer(BaseOptimizer):
    """Particle Swarm Optimization for wormhole parameters."""
    
    def __init__(self, objective_function: ObjectiveFunction,
                 bounds: OptimizationBounds = None,
                 swarm_size: int = 30,
                 w: float = 0.729,  # Inertia weight
                 c1: float = 1.494,  # Cognitive parameter
                 c2: float = 1.494):  # Social parameter
        """Initialize particle swarm optimizer.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
            swarm_size: Number of particles
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        super().__init__(objective_function, bounds)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> OptimizationResult:
        """Run particle swarm optimization."""
        
        import time
        start_time = time.time()
        
        # Initialize swarm
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        dim = len(lower_bounds)
        
        # Particle positions and velocities
        positions = np.random.uniform(lower_bounds, upper_bounds, (self.swarm_size, dim))
        velocities = np.random.uniform(-0.1 * (upper_bounds - lower_bounds),
                                     0.1 * (upper_bounds - lower_bounds),
                                     (self.swarm_size, dim))
        
        # Personal best positions and fitness
        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.swarm_size, -np.inf)
        
        # Global best
        global_best_position = None
        global_best_fitness = -np.inf
        
        # Evolution tracking
        convergence_history = []
        population_history = []
        
        # PSO main loop
        for iteration in range(max_iterations):
            
            # Evaluate particles
            for i in range(self.swarm_size):
                parameters = self._array_to_parameters(positions[i])
                fitness = self.objective_function.evaluate(parameters)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i].copy()
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                
                # Random factors
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                # Velocity update
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               self.c2 * r2 * (global_best_position - positions[i]))
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Enforce bounds
                positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)
                
                # Velocity clamping
                v_max = 0.2 * (upper_bounds - lower_bounds)
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Record progress
            convergence_history.append(global_best_fitness)
            current_population = [self._array_to_parameters(pos) for pos in positions]
            population_history.append(current_population)
            
            # Early stopping check
            if len(convergence_history) > 30:
                recent_improvement = convergence_history[-1] - convergence_history[-30]
                if abs(recent_improvement) < 1e-6:
                    termination_reason = "Early convergence"
                    break
        else:
            termination_reason = "Maximum iterations reached"
        
        # Results
        optimization_time = time.time() - start_time
        best_parameters = self._array_to_parameters(global_best_position)
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_fitness=global_best_fitness,
            convergence_history=convergence_history,
            population_history=population_history,
            optimization_time=optimization_time,
            function_evaluations=self.objective_function.evaluation_count,
            success=True,
            termination_reason=termination_reason
        )


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian processes."""
    
    def __init__(self, objective_function: ObjectiveFunction,
                 bounds: OptimizationBounds = None,
                 acquisition_function: str = 'expected_improvement'):
        """Initialize Bayesian optimizer.
        
        Args:
            objective_function: Function to optimize
            bounds: Parameter bounds
            acquisition_function: 'expected_improvement', 'upper_confidence_bound', 'probability_improvement'
        """
        super().__init__(objective_function, bounds)
        self.acquisition_function = acquisition_function
        self.gp = None
        
    def optimize(self, max_iterations: int = 100, n_initial_points: int = 10, **kwargs) -> OptimizationResult:
        """Run Bayesian optimization."""
        
        import time
        start_time = time.time()
        
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        dim = len(lower_bounds)
        
        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # Initial sampling
        X_samples = []
        y_samples = []
        
        for _ in range(n_initial_points):
            x = np.random.uniform(lower_bounds, upper_bounds)
            parameters = self._array_to_parameters(x)
            y = self.objective_function.evaluate(parameters)
            
            X_samples.append(x)
            y_samples.append(y)
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        # Evolution tracking
        convergence_history = list(y_samples)
        population_history = []
        
        best_fitness = max(y_samples)
        best_x = X_samples[np.argmax(y_samples)]
        
        # Bayesian optimization loop
        for iteration in range(max_iterations - n_initial_points):
            
            # Fit GP to current data
            self.gp.fit(X_samples, y_samples)
            
            # Find next point to evaluate using acquisition function
            next_x = self._optimize_acquisition(lower_bounds, upper_bounds)
            
            # Evaluate objective at next point
            parameters = self._array_to_parameters(next_x)
            next_y = self.objective_function.evaluate(parameters)
            
            # Add to dataset
            X_samples = np.vstack([X_samples, next_x])
            y_samples = np.append(y_samples, next_y)
            
            # Update best
            if next_y > best_fitness:
                best_fitness = next_y
                best_x = next_x
            
            # Record progress
            convergence_history.append(best_fitness)
            current_population = [self._array_to_parameters(x) for x in X_samples[-10:]]  # Last 10
            population_history.append(current_population)
        
        # Results
        optimization_time = time.time() - start_time
        best_parameters = self._array_to_parameters(best_x)
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_fitness=best_fitness,
            convergence_history=convergence_history,
            population_history=population_history,
            optimization_time=optimization_time,
            function_evaluations=self.objective_function.evaluation_count,
            success=True,
            termination_reason="Maximum iterations reached"
        )
    
    def _optimize_acquisition(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
        """Optimize acquisition function to find next evaluation point."""
        
        # Define acquisition function
        def acquisition(x):
            x = x.reshape(1, -1)
            
            if self.acquisition_function == 'expected_improvement':
                return -self._expected_improvement(x)
            elif self.acquisition_function == 'upper_confidence_bound':
                return -self._upper_confidence_bound(x)
            else:  # probability_improvement
                return -self._probability_improvement(x)
        
        # Multi-start optimization of acquisition function
        best_x = None
        best_acq_value = np.inf
        
        for _ in range(10):  # Multiple random starts
            x0 = np.random.uniform(lower_bounds, upper_bounds)
            
            result = minimize(acquisition, x0, method='L-BFGS-B',
                            bounds=list(zip(lower_bounds, upper_bounds)))
            
            if result.fun < best_acq_value:
                best_acq_value = result.fun
                best_x = result.x
        
        return best_x
    
    def _expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected improvement acquisition function."""
        
        mu, sigma = self.gp.predict(x, return_std=True)
        mu_sample_opt = np.max(self.gp.y_train_)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, x: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """Upper confidence bound acquisition function."""
        
        mu, sigma = self.gp.predict(x, return_std=True)
        return mu + kappa * sigma
    
    def _probability_improvement(self, x: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Probability of improvement acquisition function."""
        
        mu, sigma = self.gp.predict(x, return_std=True)
        mu_sample_opt = np.max(self.gp.y_train_)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi


class MultiObjectiveOptimizer:
    """Multi-objective optimization using NSGA-II algorithm."""
    
    def __init__(self, objectives: List[Callable], bounds: OptimizationBounds = None):
        """Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective functions to optimize
            bounds: Parameter bounds
        """
        self.objectives = objectives
        self.bounds = bounds or OptimizationBounds()
        self.population_size = 100
    
    def optimize(self, max_generations: int = 500) -> Dict:
        """Run NSGA-II multi-objective optimization."""
        
        # This is a simplified implementation
        # Full NSGA-II would require non-dominated sorting and crowding distance
        
        lower_bounds, upper_bounds = self._bounds_to_arrays()
        dim = len(lower_bounds)
        
        # Initialize population
        population = np.random.uniform(lower_bounds, upper_bounds, 
                                     (self.population_size, dim))
        
        # Evolution loop
        pareto_front = []
        
        for generation in range(max_generations):
            
            # Evaluate population on all objectives
            objective_values = []
            
            for individual in population:
                obj_vals = []
                for obj_func in self.objectives:
                    if hasattr(obj_func, 'evaluate'):
                        parameters = self._array_to_parameters(individual)
                        val = obj_func.evaluate(parameters)
                    else:
                        val = obj_func(individual)
                    obj_vals.append(val)
                
                objective_values.append(obj_vals)
            
            # Simple Pareto front identification (non-dominated solutions)
            pareto_indices = self._find_pareto_front(objective_values)
            
            if generation == max_generations - 1:
                pareto_front = [
                    {
                        'parameters': self._array_to_parameters(population[i]),
                        'objectives': objective_values[i]
                    }
                    for i in pareto_indices
                ]
        
        return {
            'pareto_front': pareto_front,
            'num_solutions': len(pareto_front)
        }
    
    def _find_pareto_front(self, objective_values: List[List[float]]) -> List[int]:
        """Find Pareto front (non-dominated solutions)."""
        
        pareto_indices = []
        objective_values = np.array(objective_values)
        
        for i, obj_i in enumerate(objective_values):
            is_dominated = False
            
            for j, obj_j in enumerate(objective_values):
                if i != j:
                    # Check if j dominates i (all objectives better or equal, at least one strictly better)
                    if all(obj_j >= obj_i) and any(obj_j > obj_i):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _bounds_to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert bounds to arrays."""
        param_names = [
            'throat_radius', 'exotic_energy_scale', 'shape_parameter',
            'coupling_strength', 'temperature', 'mass_parameter', 'quantum_correction'
        ]
        
        lower_bounds = []
        upper_bounds = []
        
        for name in param_names:
            bound = getattr(self.bounds, name)
            lower_bounds.append(bound[0])
            upper_bounds.append(bound[1])
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def _array_to_parameters(self, x: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dictionary."""
        param_names = [
            'throat_radius', 'exotic_energy_scale', 'shape_parameter',
            'coupling_strength', 'temperature', 'mass_parameter', 'quantum_correction'
        ]
        
        return {name: float(val) for name, val in zip(param_names, x)}


# Import required for Bayesian optimization
try:
    from scipy.stats import norm
except ImportError:
    # Fallback implementation
    class norm:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        @staticmethod
        def pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def run_optimization_comparison(objective_function: ObjectiveFunction,
                              bounds: OptimizationBounds = None,
                              max_evaluations: int = 1000) -> Dict:
    """Compare different optimization algorithms."""
    
    algorithms = {
        'Genetic Algorithm': GeneticAlgorithmOptimizer(objective_function, bounds),
        'Particle Swarm': ParticleSwarmOptimizer(objective_function, bounds),
        'Bayesian Optimization': BayesianOptimizer(objective_function, bounds)
    }
    
    results = {}
    
    for name, optimizer in algorithms.items():
        print(f"Running {name}...")
        
        # Reset evaluation counter
        objective_function.evaluation_count = 0
        
        try:
            if name == 'Bayesian Optimization':
                result = optimizer.optimize(max_iterations=max_evaluations//10)
            else:
                result = optimizer.optimize(max_iterations=max_evaluations//50)
            
            results[name] = {
                'best_fitness': result.best_fitness,
                'best_parameters': result.best_parameters,
                'convergence_time': result.optimization_time,
                'function_evaluations': result.function_evaluations,
                'success': result.success
            }
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


def visualize_optimization_results(results: OptimizationResult, save_path: Optional[str] = None):
    """Visualize optimization results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convergence history
    axes[0, 0].plot(results.convergence_history)
    axes[0, 0].set_title('Convergence History')
    axes[0, 0].set_xlabel('Generation/Iteration')
    axes[0, 0].set_ylabel('Best Fitness')
    axes[0, 0].grid(True)
    
    # Parameter evolution (first few parameters)
    if results.population_history:
        param_names = list(results.best_parameters.keys())[:3]  # First 3 parameters
        
        for i, param_name in enumerate(param_names):
            param_evolution = []
            for pop in results.population_history[::10]:  # Sample every 10th generation
                values = [ind[param_name] for ind in pop]
                param_evolution.append(np.mean(values))
            
            axes[0, 1].plot(param_evolution, label=param_name)
        
        axes[0, 1].set_title('Parameter Evolution')
        axes[0, 1].set_xlabel('Generation (sampled)')
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Final parameter values
    param_names = list(results.best_parameters.keys())
    param_values = list(results.best_parameters.values())
    
    axes[1, 0].barh(param_names, param_values)
    axes[1, 0].set_title('Best Parameters')
    axes[1, 0].set_xlabel('Parameter Value')
    
    # Optimization statistics
    stats_text = f"""
    Best Fitness: {results.best_fitness:.4f}
    Optimization Time: {results.optimization_time:.2f} s
    Function Evaluations: {results.function_evaluations}
    Success: {results.success}
    Termination: {results.termination_reason}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()