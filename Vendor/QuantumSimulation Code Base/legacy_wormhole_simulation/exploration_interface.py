"""
Unified Scientific Exploration Interface for Quantum Wormhole Simulation.

This module provides a comprehensive interface that integrates all simulation components
into a unified scientific exploration platform with advanced analysis capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import concurrent.futures
from pathlib import Path

# Import all simulation components
from src.visualization.spacetime_plotter import SpacetimePlotter, SpacetimeVisualizationConfig
from src.visualization.quantum_state_animator import QuantumStateAnimator, AnimationConfig
from src.visualization.field_visualizer import FieldVisualizer, FieldVisualizationConfig
from src.visualization.interactive_dashboard import InteractiveDashboard, DashboardConfig, SimulationState

from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.quantum.entanglement_dynamics import EntanglementDynamics
from src.physics.spacetime_metrics import MorrisThorneeWormhole, SpacetimeMetric
from src.physics.exotic_matter import ExoticMatter
from src.physics.stress_energy_tensor import StressEnergyTensor
from src.ai.stability_predictor import StabilityPredictor
from src.ai.parameter_optimizer import ParameterOptimizer


@dataclass
class ExplorationConfig:
    """Configuration for scientific exploration interface."""
    
    # General parameters
    workspace_dir: str = "exploration_workspace"
    session_name: str = "wormhole_exploration"
    auto_save: bool = True
    save_interval: int = 300  # seconds
    
    # Simulation parameters
    default_throat_radius: float = 1e3
    default_mass: float = 1e30
    default_qubits: int = 4
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'throat_radius': (1e2, 1e4),
        'mass': (1e29, 1e31),
        'num_qubits': (2, 8),
        'exotic_matter_density': (-1e16, -1e14)
    })
    
    # Analysis parameters
    parameter_sweep_points: int = 20
    stability_threshold: float = 0.7
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    # Visualization parameters
    enable_3d_visualization: bool = True
    enable_animations: bool = True
    enable_interactive_plots: bool = True
    export_high_res: bool = False
    
    # Performance parameters
    parallel_processing: bool = True
    max_workers: int = 4


class ScientificExplorationInterface:
    """Unified interface for comprehensive wormhole simulation exploration."""
    
    def __init__(self, config: ExplorationConfig = None):
        """Initialize scientific exploration interface.
        
        Args:
            config: Exploration configuration
        """
        self.config = config or ExplorationConfig()
        
        # Create workspace directory
        self.workspace_path = Path(self.config.workspace_dir)
        self.workspace_path.mkdir(exist_ok=True)
        
        # Initialize core components
        self.dashboard = None
        self.spacetime_plotter = None
        self.quantum_animator = None
        self.field_visualizer = None
        
        # Initialize AI components
        self.stability_predictor = None
        self.parameter_optimizer = None
        
        # Initialize physics components
        self.wormhole_metric = None
        self.exotic_matter = None
        self.stress_energy = None
        self.quantum_circuit = None
        
        # Exploration state
        self.current_parameters = {}
        self.exploration_history = []
        self.analysis_results = {}
        self.visualization_cache = {}
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize all system components with default parameters."""
        
        # Set initial parameters
        self.current_parameters = {
            'throat_radius': self.config.default_throat_radius,
            'mass': self.config.default_mass,
            'num_qubits': self.config.default_qubits,
            'exotic_matter_density': -1e15,
            'traversal_probability': 0.8,
            'entanglement_strength': 1.0,
            'decoherence_rate': 0.01
        }
        
        # Initialize physics components
        self._update_physics_components()
        
        # Initialize visualization components
        self._initialize_visualization_components()
        
        # Initialize AI components
        self._initialize_ai_components()
        
        print(f"Scientific Exploration Interface initialized")
        print(f"Workspace: {self.workspace_path.absolute()}")
        
    def _update_physics_components(self):
        """Update physics components with current parameters."""
        
        # Wormhole metric
        self.wormhole_metric = MorrisThorneeWormhole(
            throat_radius=self.current_parameters['throat_radius']
        )
        
        # Exotic matter
        self.exotic_matter = ExoticMatter(
            energy_density=self.current_parameters['exotic_matter_density']
        )
        
        # Stress-energy tensor
        self.stress_energy = StressEnergyTensor(
            exotic_matter=self.exotic_matter,
            metric=self.wormhole_metric
        )
        
        # Quantum circuit
        self.quantum_circuit = WormholeQuantumCircuit(
            num_qubits=self.current_parameters['num_qubits'],
            traversal_probability=self.current_parameters['traversal_probability']
        )
        
    def _initialize_visualization_components(self):
        """Initialize visualization components."""
        
        # Spacetime visualization
        spacetime_config = SpacetimeVisualizationConfig(
            r_min=self.current_parameters['throat_radius'] * 1.1,
            r_max=self.current_parameters['throat_radius'] * 10
        )
        self.spacetime_plotter = SpacetimePlotter(self.wormhole_metric, spacetime_config)
        
        # Quantum animation
        animation_config = AnimationConfig()
        self.quantum_animator = QuantumStateAnimator(animation_config)
        self.quantum_animator.setup_wormhole_system(
            num_qubits=self.current_parameters['num_qubits'],
            traversal_probability=self.current_parameters['traversal_probability']
        )
        
        # Field visualization
        field_config = FieldVisualizationConfig()
        self.field_visualizer = FieldVisualizer(field_config)
        
        # Interactive dashboard
        dashboard_config = DashboardConfig()
        self.dashboard = InteractiveDashboard(dashboard_config)
        
    def _initialize_ai_components(self):
        """Initialize AI analysis components."""
        
        try:
            self.stability_predictor = StabilityPredictor()
            self.parameter_optimizer = ParameterOptimizer()
        except Exception as e:
            print(f"Warning: AI components initialization failed: {e}")
            self.stability_predictor = None
            self.parameter_optimizer = None
    
    def set_parameters(self, **kwargs) -> None:
        """Set simulation parameters and update components.
        
        Args:
            **kwargs: Parameter updates
        """
        
        # Validate parameters
        for key, value in kwargs.items():
            if key in self.config.parameter_ranges:
                min_val, max_val = self.config.parameter_ranges[key]
                if not (min_val <= value <= max_val):
                    print(f"Warning: {key}={value} outside range [{min_val}, {max_val}]")
        
        # Update parameters
        self.current_parameters.update(kwargs)
        
        # Update components
        self._update_physics_components()
        self._initialize_visualization_components()
        
        # Log parameter change
        self.exploration_history.append({
            'timestamp': datetime.now(),
            'action': 'parameter_update',
            'parameters': kwargs.copy(),
            'full_state': self.current_parameters.copy()
        })
        
        print(f"Parameters updated: {kwargs}")
    
    def create_comprehensive_visualization(self) -> go.Figure:
        """Create comprehensive multi-panel visualization of current state.
        
        Returns:
            Comprehensive visualization figure
        """
        
        print("Creating comprehensive visualization...")
        
        # Create main dashboard
        main_fig = self.dashboard.create_main_dashboard()
        
        # Update dashboard with current parameters
        self.dashboard.update_simulation_state(**self.current_parameters)
        
        return main_fig
    
    def run_parameter_sweep(self, parameter_name: str, 
                          value_range: Tuple[float, float] = None,
                          num_points: int = None) -> Dict[str, Any]:
        """Run parameter sweep analysis.
        
        Args:
            parameter_name: Name of parameter to sweep
            value_range: Range of values to explore
            num_points: Number of points in sweep
            
        Returns:
            Sweep results dictionary
        """
        
        if value_range is None:
            if parameter_name in self.config.parameter_ranges:
                value_range = self.config.parameter_ranges[parameter_name]
            else:
                raise ValueError(f"No default range for parameter {parameter_name}")
        
        if num_points is None:
            num_points = self.config.parameter_sweep_points
        
        print(f"Running parameter sweep: {parameter_name} from {value_range[0]} to {value_range[1]}")
        
        # Create parameter values
        if parameter_name == 'num_qubits':
            # Integer parameter
            param_values = np.linspace(value_range[0], value_range[1], num_points, dtype=int)
            param_values = np.unique(param_values)  # Remove duplicates
        else:
            # Float parameter
            param_values = np.linspace(value_range[0], value_range[1], num_points)
        
        # Store original parameter value
        original_value = self.current_parameters[parameter_name]
        
        # Initialize results storage
        results = {
            'parameter_name': parameter_name,
            'parameter_values': param_values,
            'stability_scores': [],
            'entanglement_measures': [],
            'field_strengths': [],
            'computation_times': [],
            'convergence_status': []
        }
        
        # Run sweep
        for i, param_value in enumerate(param_values):
            start_time = time.time()
            
            try:
                # Set parameter
                self.set_parameters(**{parameter_name: param_value})
                
                # Compute stability score
                stability = self._compute_stability_score()
                results['stability_scores'].append(stability)
                
                # Compute entanglement measure
                entanglement = self._compute_entanglement_measure()
                results['entanglement_measures'].append(entanglement)
                
                # Compute field strength
                field_strength = self._compute_field_strength()
                results['field_strengths'].append(field_strength)
                
                # Check convergence
                converged = self._check_convergence()
                results['convergence_status'].append(converged)
                
                computation_time = time.time() - start_time
                results['computation_times'].append(computation_time)
                
                print(f"  Point {i+1}/{len(param_values)}: {parameter_name}={param_value:.2e}, "
                      f"stability={stability:.3f}, time={computation_time:.2f}s")
                
            except Exception as e:
                print(f"  Error at {parameter_name}={param_value}: {e}")
                results['stability_scores'].append(0.0)
                results['entanglement_measures'].append(0.0)
                results['field_strengths'].append(0.0)
                results['convergence_status'].append(False)
                results['computation_times'].append(time.time() - start_time)
        
        # Restore original parameter
        self.set_parameters(**{parameter_name: original_value})
        
        # Store results
        self.analysis_results[f'sweep_{parameter_name}'] = results
        
        # Log analysis
        self.exploration_history.append({
            'timestamp': datetime.now(),
            'action': 'parameter_sweep',
            'parameter': parameter_name,
            'range': value_range,
            'points': len(param_values),
            'results_summary': {
                'max_stability': max(results['stability_scores']),
                'avg_computation_time': np.mean(results['computation_times'])
            }
        })
        
        print(f"Parameter sweep completed. Max stability: {max(results['stability_scores']):.3f}")
        
        return results
    
    def _compute_stability_score(self) -> float:
        """Compute stability score for current configuration."""
        
        try:
            if self.stability_predictor:
                # Use AI predictor if available
                features = np.array([
                    self.current_parameters['throat_radius'],
                    self.current_parameters['mass'],
                    self.current_parameters['exotic_matter_density'],
                    self.current_parameters['entanglement_strength']
                ]).reshape(1, -1)
                
                stability = self.stability_predictor.predict_stability(features)[0]
            else:
                # Use physics-based heuristic
                r_throat = self.current_parameters['throat_radius']
                mass = self.current_parameters['mass']
                rho_exotic = abs(self.current_parameters['exotic_matter_density'])
                
                # Simple stability metric based on geometry and exotic matter
                schwarzschild_radius = 2 * 6.67e-11 * mass / (3e8**2)
                
                if r_throat > schwarzschild_radius:
                    geometry_factor = np.exp(-(r_throat / schwarzschild_radius - 1))
                else:
                    geometry_factor = 0.1  # Unstable if throat too small
                
                exotic_factor = min(1.0, rho_exotic / 1e15)  # Normalize exotic matter density
                quantum_factor = 0.5 + 0.5 * self.current_parameters['entanglement_strength']
                
                stability = geometry_factor * exotic_factor * quantum_factor
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            print(f"Warning: Stability computation failed: {e}")
            return 0.0
    
    def _compute_entanglement_measure(self) -> float:
        """Compute quantum entanglement measure."""
        
        try:
            # Simulate entanglement evolution
            entanglement = self.current_parameters['entanglement_strength']
            decoherence = self.current_parameters['decoherence_rate']
            
            # Simple model: entanglement decays with decoherence
            effective_entanglement = entanglement * np.exp(-decoherence * 10)  # time factor
            
            return max(0.0, min(1.0, effective_entanglement))
            
        except Exception:
            return 0.0
    
    def _compute_field_strength(self) -> float:
        """Compute characteristic field strength."""
        
        try:
            # Gravitational field strength estimate
            mass = self.current_parameters['mass']
            r_throat = self.current_parameters['throat_radius']
            
            # Field strength at throat
            G = 6.67e-11
            field_strength = G * mass / r_throat**2
            
            # Normalize to reasonable scale
            return min(1.0, field_strength / 1e10)
            
        except Exception:
            return 0.0
    
    def _check_convergence(self) -> bool:
        """Check if current configuration has converged."""
        
        # Simple convergence check based on parameter stability
        return True  # Placeholder
    
    def visualize_parameter_sweep(self, sweep_results: Dict[str, Any]) -> go.Figure:
        """Create visualization of parameter sweep results.
        
        Args:
            sweep_results: Results from parameter sweep
            
        Returns:
            Parameter sweep visualization figure
        """
        
        param_name = sweep_results['parameter_name']
        param_values = sweep_results['parameter_values']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Stability vs Parameter',
                'Entanglement vs Parameter', 
                'Field Strength vs Parameter',
                'Computation Time vs Parameter'
            ]
        )
        
        # Stability plot
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=sweep_results['stability_scores'],
                mode='lines+markers',
                name='Stability Score',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # Entanglement plot
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=sweep_results['entanglement_measures'],
                mode='lines+markers',
                name='Entanglement',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        # Field strength plot
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=sweep_results['field_strengths'],
                mode='lines+markers',
                name='Field Strength',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
        
        # Computation time plot
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=sweep_results['computation_times'],
                mode='lines+markers',
                name='Computation Time',
                line=dict(color='orange', width=3)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text=param_name, row=1, col=1)
        fig.update_xaxes(title_text=param_name, row=1, col=2)
        fig.update_xaxes(title_text=param_name, row=2, col=1)
        fig.update_xaxes(title_text=param_name, row=2, col=2)
        
        fig.update_yaxes(title_text='Stability Score', row=1, col=1)
        fig.update_yaxes(title_text='Entanglement Measure', row=1, col=2)
        fig.update_yaxes(title_text='Field Strength', row=2, col=1)
        fig.update_yaxes(title_text='Time (s)', row=2, col=2)
        
        fig.update_layout(
            title=f'Parameter Sweep Analysis: {param_name}',
            height=600,
            width=1000,
            showlegend=False
        )
        
        return fig
    
    def create_stability_landscape(self, param1: str, param2: str,
                                 resolution: int = 15) -> go.Figure:
        """Create 2D stability landscape for two parameters.
        
        Args:
            param1: First parameter name
            param2: Second parameter name  
            resolution: Grid resolution
            
        Returns:
            Stability landscape visualization
        """
        
        print(f"Creating stability landscape for {param1} vs {param2}")
        
        # Get parameter ranges
        range1 = self.config.parameter_ranges.get(param1, (0.1, 10))
        range2 = self.config.parameter_ranges.get(param2, (0.1, 10))
        
        # Create parameter grids
        if param1 == 'num_qubits':
            p1_values = np.linspace(range1[0], range1[1], resolution, dtype=int)
            p1_values = np.unique(p1_values)
        else:
            p1_values = np.linspace(range1[0], range1[1], resolution)
        
        if param2 == 'num_qubits':
            p2_values = np.linspace(range2[0], range2[1], resolution, dtype=int) 
            p2_values = np.unique(p2_values)
        else:
            p2_values = np.linspace(range2[0], range2[1], resolution)
        
        # Store original values
        orig_p1 = self.current_parameters[param1]
        orig_p2 = self.current_parameters[param2]
        
        # Compute stability landscape
        stability_landscape = np.zeros((len(p1_values), len(p2_values)))
        
        total_points = len(p1_values) * len(p2_values)
        point_count = 0
        
        for i, p1_val in enumerate(p1_values):
            for j, p2_val in enumerate(p2_values):
                point_count += 1
                
                try:
                    # Set parameters
                    self.set_parameters(**{param1: p1_val, param2: p2_val})
                    
                    # Compute stability
                    stability = self._compute_stability_score()
                    stability_landscape[i, j] = stability
                    
                    if point_count % 10 == 0:
                        print(f"  Progress: {point_count}/{total_points} points computed")
                        
                except Exception as e:
                    stability_landscape[i, j] = 0.0
        
        # Restore original parameters
        self.set_parameters(**{param1: orig_p1, param2: orig_p2})
        
        # Create visualization
        fig = go.Figure(data=go.Heatmap(
            z=stability_landscape,
            x=p2_values,
            y=p1_values,
            colorscale='Viridis',
            colorbar=dict(title='Stability Score')
        ))
        
        # Add contour lines
        fig.add_trace(go.Contour(
            z=stability_landscape,
            x=p2_values,
            y=p1_values,
            showscale=False,
            contours=dict(
                start=0.0,
                end=1.0,
                size=0.1,
                coloring='lines'
            ),
            line=dict(width=1, color='white', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Stability Landscape: {param1} vs {param2}',
            xaxis_title=param2,
            yaxis_title=param1,
            width=700,
            height=600
        )
        
        # Store results
        landscape_key = f'stability_landscape_{param1}_{param2}'
        self.analysis_results[landscape_key] = {
            'param1': param1,
            'param2': param2,
            'param1_values': p1_values,
            'param2_values': p2_values,
            'stability_landscape': stability_landscape,
            'max_stability': np.max(stability_landscape),
            'optimal_params': {
                param1: p1_values[np.unravel_index(np.argmax(stability_landscape), stability_landscape.shape)[0]],
                param2: p2_values[np.unravel_index(np.argmax(stability_landscape), stability_landscape.shape)[1]]
            }
        }
        
        print(f"Stability landscape completed. Max stability: {np.max(stability_landscape):.3f}")
        
        return fig
    
    def run_optimization(self, target_metric: str = 'stability',
                        max_iterations: int = None) -> Dict[str, Any]:
        """Run parameter optimization.
        
        Args:
            target_metric: Metric to optimize ('stability', 'entanglement', 'field_strength')
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results
        """
        
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        print(f"Running parameter optimization for {target_metric}")
        
        # Define objective function
        def objective_function(params_array):
            """Objective function for optimization."""
            
            try:
                # Map array to parameters
                param_dict = {}
                param_names = ['throat_radius', 'mass', 'exotic_matter_density', 'entanglement_strength']
                
                for i, name in enumerate(param_names):
                    if i < len(params_array):
                        param_dict[name] = params_array[i]
                
                # Set parameters
                self.set_parameters(**param_dict)
                
                # Compute target metric
                if target_metric == 'stability':
                    return -self._compute_stability_score()  # Minimize negative
                elif target_metric == 'entanglement':
                    return -self._compute_entanglement_measure()
                elif target_metric == 'field_strength':
                    return -self._compute_field_strength()
                else:
                    return 0.0
                    
            except Exception:
                return 1e6  # Large penalty for invalid parameters
        
        # Initial parameter vector
        initial_params = np.array([
            self.current_parameters['throat_radius'],
            self.current_parameters['mass'], 
            self.current_parameters['exotic_matter_density'],
            self.current_parameters['entanglement_strength']
        ])
        
        # Parameter bounds
        bounds = [
            self.config.parameter_ranges.get('throat_radius', (1e2, 1e4)),
            self.config.parameter_ranges.get('mass', (1e29, 1e31)),
            self.config.parameter_ranges.get('exotic_matter_density', (-1e16, -1e14)),
            (0.1, 2.0)  # entanglement_strength bounds
        ]
        
        # Simple optimization using grid search (placeholder for more advanced methods)
        best_score = float('inf')
        best_params = initial_params.copy()
        optimization_history = []
        
        # Random search optimization
        np.random.seed(42)
        
        for iteration in range(max_iterations):
            # Generate random parameter variation
            random_params = initial_params.copy()
            
            for i, (min_val, max_val) in enumerate(bounds):
                # Add random perturbation
                perturbation = np.random.uniform(-0.1, 0.1) * (max_val - min_val)
                random_params[i] = np.clip(
                    random_params[i] + perturbation, 
                    min_val, max_val
                )
            
            # Evaluate objective
            score = objective_function(random_params)
            
            # Update best if improved
            if score < best_score:
                best_score = score
                best_params = random_params.copy()
                
                print(f"  Iteration {iteration+1}: New best {target_metric} = {-score:.4f}")
            
            # Store history
            optimization_history.append({
                'iteration': iteration + 1,
                'score': -score,  # Convert back to positive
                'parameters': random_params.copy()
            })
            
            # Early stopping check
            if -score > 0.99:  # Very good score
                print(f"  Early stopping: Excellent {target_metric} achieved")
                break
        
        # Set optimal parameters
        optimal_params = {
            'throat_radius': best_params[0],
            'mass': best_params[1],
            'exotic_matter_density': best_params[2], 
            'entanglement_strength': best_params[3]
        }
        
        self.set_parameters(**optimal_params)
        
        # Prepare results
        results = {
            'target_metric': target_metric,
            'optimal_score': -best_score,
            'optimal_parameters': optimal_params,
            'initial_parameters': {
                'throat_radius': initial_params[0],
                'mass': initial_params[1],
                'exotic_matter_density': initial_params[2],
                'entanglement_strength': initial_params[3]
            },
            'optimization_history': optimization_history,
            'iterations': len(optimization_history)
        }
        
        # Store results
        self.analysis_results[f'optimization_{target_metric}'] = results
        
        print(f"Optimization completed. Best {target_metric}: {-best_score:.4f}")
        
        return results
    
    def export_results(self, filename: str = None, format_type: str = 'json') -> None:
        """Export analysis results and current state.
        
        Args:
            filename: Output filename
            format_type: Export format ('json', 'csv', 'html')
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.session_name}_{timestamp}"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'current_parameters': self.current_parameters,
            'analysis_results': self.analysis_results,
            'exploration_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()} 
                for entry in self.exploration_history
            ]
        }
        
        output_path = self.workspace_path / f"{filename}.{format_type}"
        
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format_type == 'csv':
            # Export analysis results as CSV
            results_df = pd.DataFrame()
            
            for key, results in self.analysis_results.items():
                if isinstance(results, dict) and 'parameter_values' in results:
                    df = pd.DataFrame({
                        'analysis_type': key,
                        'parameter_values': results['parameter_values'],
                        'stability_scores': results.get('stability_scores', []),
                        'entanglement_measures': results.get('entanglement_measures', []),
                        'field_strengths': results.get('field_strengths', [])
                    })
                    results_df = pd.concat([results_df, df], ignore_index=True)
            
            results_df.to_csv(output_path, index=False)
        
        elif format_type == 'html':
            # Create comprehensive HTML report
            html_content = self._generate_html_report(export_data)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        print(f"Results exported to: {output_path}")
    
    def _generate_html_report(self, export_data: Dict) -> str:
        """Generate comprehensive HTML report."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Wormhole Exploration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .parameter {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .results {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quantum Wormhole Exploration Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Session: {session_name}</p>
            </div>
            
            <div class="section">
                <h2>Current Parameters</h2>
                {parameters_html}
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
                {results_html}
            </div>
            
            <div class="section">
                <h2>Exploration History</h2>
                <p>Total actions: {history_count}</p>
                {history_html}
            </div>
        </body>
        </html>
        """
        
        # Generate parameters HTML
        parameters_html = ""
        for key, value in export_data['current_parameters'].items():
            parameters_html += f'<div class="parameter"><strong>{key}:</strong> {value}</div>\n'
        
        # Generate results HTML
        results_html = ""
        for key, results in export_data['analysis_results'].items():
            if isinstance(results, dict):
                results_html += f'<div class="results"><h3>{key}</h3>\n'
                for res_key, res_value in results.items():
                    if not isinstance(res_value, (list, np.ndarray)):
                        results_html += f'<p><strong>{res_key}:</strong> {res_value}</p>\n'
                results_html += '</div>\n'
        
        # Generate history HTML (last 10 entries)
        history_html = ""
        recent_history = export_data['exploration_history'][-10:]
        for entry in recent_history:
            history_html += f'<div class="parameter">{entry["timestamp"]}: {entry["action"]}</div>\n'
        
        return html_template.format(
            timestamp=export_data['timestamp'],
            session_name=self.config.session_name,
            parameters_html=parameters_html,
            results_html=results_html,
            history_count=len(export_data['exploration_history']),
            history_html=history_html
        )
    
    def create_exploration_summary(self) -> go.Figure:
        """Create summary visualization of all exploration results.
        
        Returns:
            Summary figure with key insights
        """
        
        # Create comprehensive summary dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Parameter Evolution',
                'Stability Analysis', 
                'Performance Metrics',
                'Exploration Timeline'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Parameter evolution over time
        if self.exploration_history:
            timestamps = []
            throat_radii = []
            masses = []
            
            for entry in self.exploration_history:
                if entry['action'] == 'parameter_update':
                    timestamps.append(entry['timestamp'])
                    throat_radii.append(entry['full_state'].get('throat_radius', 0))
                    masses.append(entry['full_state'].get('mass', 0))
            
            if timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(timestamps))),
                        y=throat_radii,
                        mode='lines+markers',
                        name='Throat Radius',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
        
        # Stability analysis summary
        stability_data = []
        for key, results in self.analysis_results.items():
            if 'stability' in key and isinstance(results, dict):
                if 'stability_scores' in results:
                    stability_data.extend(results['stability_scores'])
                elif 'optimal_score' in results:
                    stability_data.append(results['optimal_score'])
        
        if stability_data:
            fig.add_trace(
                go.Histogram(
                    x=stability_data,
                    nbinsx=20,
                    name='Stability Distribution',
                    marker_color='green'
                ),
                row=1, col=2
            )
        
        # Performance metrics
        computation_times = []
        for key, results in self.analysis_results.items():
            if isinstance(results, dict) and 'computation_times' in results:
                computation_times.extend(results['computation_times'])
        
        if computation_times:
            fig.add_trace(
                go.Box(
                    y=computation_times,
                    name='Computation Times',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # Exploration timeline
        action_counts = {}
        for entry in self.exploration_history:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        if action_counts:
            fig.add_trace(
                go.Bar(
                    x=list(action_counts.keys()),
                    y=list(action_counts.values()),
                    name='Action Counts',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Exploration Summary Dashboard',
            height=700,
            width=1000,
            showlegend=True
        )
        
        return fig


def launch_exploration_interface(config: ExplorationConfig = None) -> ScientificExplorationInterface:
    """Launch the scientific exploration interface.
    
    Args:
        config: Exploration configuration
        
    Returns:
        Initialized exploration interface
    """
    
    interface = ScientificExplorationInterface(config)
    
    print("\n" + "="*60)
    print("QUANTUM WORMHOLE SCIENTIFIC EXPLORATION INTERFACE")
    print("="*60)
    print("\nAvailable methods:")
    print("• set_parameters(**kwargs) - Update simulation parameters")
    print("• run_parameter_sweep(param, range, points) - Analyze parameter space")
    print("• create_stability_landscape(param1, param2) - 2D stability analysis")
    print("• run_optimization(target) - Optimize for stability/entanglement")
    print("• create_comprehensive_visualization() - Generate full dashboard")
    print("• export_results(filename, format) - Export analysis results")
    print("• create_exploration_summary() - Summary of all explorations")
    print("\nExample usage:")
    print("  interface.set_parameters(throat_radius=5e3, num_qubits=6)")
    print("  results = interface.run_parameter_sweep('throat_radius')")
    print("  fig = interface.visualize_parameter_sweep(results)")
    print("  fig.show()")
    print("\n" + "="*60)
    
    return interface