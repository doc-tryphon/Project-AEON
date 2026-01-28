#!/usr/bin/env python3
"""
Advanced Analysis Tools Demo

This script demonstrates advanced analysis capabilities including:
- Parameter sensitivity analysis
- Stability boundary mapping
- Quantum coherence tracking
- Field dynamics analysis
- Batch simulation workflows
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import pearsonr
import concurrent.futures
from pathlib import Path
import time

from src.exploration_interface import ScientificExplorationInterface, ExplorationConfig


class AdvancedAnalysisTools:
    """Advanced analysis tools for quantum wormhole exploration."""
    
    def __init__(self, interface: ScientificExplorationInterface):
        """Initialize advanced analysis tools.
        
        Args:
            interface: Scientific exploration interface
        """
        self.interface = interface
        self.sensitivity_cache = {}
        self.stability_boundary_cache = {}
        
    def run_sensitivity_analysis(self, base_parameters: dict = None,
                                perturbation_size: float = 0.1,
                                num_samples: int = 100) -> dict:
        """Run comprehensive parameter sensitivity analysis.
        
        Args:
            base_parameters: Base parameter set
            perturbation_size: Size of parameter perturbations (fraction)
            num_samples: Number of sensitivity samples
            
        Returns:
            Sensitivity analysis results
        """
        
        print(f"Running sensitivity analysis with {num_samples} samples...")
        
        if base_parameters is None:
            base_parameters = self.interface.current_parameters.copy()
        
        # Parameters to analyze
        sensitive_params = ['throat_radius', 'mass', 'exotic_matter_density', 'entanglement_strength']
        
        # Initialize results storage
        sensitivity_results = {
            'base_parameters': base_parameters,
            'perturbation_size': perturbation_size,
            'parameter_sensitivities': {},
            'correlation_matrix': {},
            'stability_response': [],
            'entanglement_response': [],
            'field_response': []
        }
        
        # Generate parameter samples using Latin Hypercube-like sampling
        np.random.seed(42)
        parameter_samples = []
        
        for i in range(num_samples):
            sample = base_parameters.copy()
            
            for param in sensitive_params:
                base_value = base_parameters[param]
                
                # Generate perturbation
                if param == 'num_qubits':
                    # Integer parameter - discrete perturbation
                    perturbation = np.random.choice([-1, 0, 1])
                    sample[param] = max(2, min(8, int(base_value + perturbation)))
                else:
                    # Continuous parameter - fractional perturbation
                    perturbation = np.random.uniform(-perturbation_size, perturbation_size)
                    sample[param] = base_value * (1 + perturbation)
                    
                    # Apply parameter bounds
                    if param in self.interface.config.parameter_ranges:
                        min_val, max_val = self.interface.config.parameter_ranges[param]
                        sample[param] = np.clip(sample[param], min_val, max_val)
            
            parameter_samples.append(sample)
        
        # Evaluate samples
        responses = {'stability': [], 'entanglement': [], 'field': []}
        
        for i, sample in enumerate(parameter_samples):
            try:
                # Set parameters
                self.interface.set_parameters(**sample)
                
                # Compute responses
                stability = self.interface._compute_stability_score()
                entanglement = self.interface._compute_entanglement_measure()
                field_strength = self.interface._compute_field_strength()
                
                responses['stability'].append(stability)
                responses['entanglement'].append(entanglement)
                responses['field'].append(field_strength)
                
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{num_samples} samples evaluated")
                    
            except Exception as e:
                print(f"  Error evaluating sample {i}: {e}")
                responses['stability'].append(0.0)
                responses['entanglement'].append(0.0) 
                responses['field'].append(0.0)
        
        # Restore base parameters
        self.interface.set_parameters(**base_parameters)
        
        # Analyze parameter sensitivities
        parameter_matrix = np.zeros((num_samples, len(sensitive_params)))
        for i, sample in enumerate(parameter_samples):
            for j, param in enumerate(sensitive_params):
                parameter_matrix[i, j] = sample[param]
        
        # Compute correlations
        for i, param in enumerate(sensitive_params):
            param_values = parameter_matrix[:, i]
            
            # Correlation with stability
            corr_stability, p_val_stability = pearsonr(param_values, responses['stability'])
            
            # Correlation with entanglement
            corr_entanglement, p_val_entanglement = pearsonr(param_values, responses['entanglement'])
            
            # Correlation with field strength
            corr_field, p_val_field = pearsonr(param_values, responses['field'])
            
            sensitivity_results['parameter_sensitivities'][param] = {
                'stability_correlation': corr_stability,
                'stability_p_value': p_val_stability,
                'entanglement_correlation': corr_entanglement,
                'entanglement_p_value': p_val_entanglement,
                'field_correlation': corr_field,
                'field_p_value': p_val_field,
                'sensitivity_magnitude': np.sqrt(corr_stability**2 + corr_entanglement**2 + corr_field**2)
            }
        
        # Parameter correlation matrix
        param_corr_matrix = np.corrcoef(parameter_matrix.T)
        sensitivity_results['correlation_matrix'] = {
            'parameters': sensitive_params,
            'matrix': param_corr_matrix
        }
        
        # Store raw responses
        sensitivity_results['stability_response'] = responses['stability']
        sensitivity_results['entanglement_response'] = responses['entanglement']
        sensitivity_results['field_response'] = responses['field']
        sensitivity_results['parameter_samples'] = parameter_samples
        
        print("Sensitivity analysis completed!")
        
        # Print key findings
        print("\nSensitivity Rankings (by total sensitivity magnitude):")
        sensitivities = [(param, data['sensitivity_magnitude']) 
                        for param, data in sensitivity_results['parameter_sensitivities'].items()]
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        for i, (param, magnitude) in enumerate(sensitivities):
            stability_corr = sensitivity_results['parameter_sensitivities'][param]['stability_correlation']
            print(f"  {i+1}. {param}: magnitude={magnitude:.3f}, stability_corr={stability_corr:.3f}")
        
        return sensitivity_results
    
    def visualize_sensitivity_analysis(self, sensitivity_results: dict) -> go.Figure:
        """Create comprehensive sensitivity analysis visualization.
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            
        Returns:
            Sensitivity visualization figure
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Parameter Sensitivity to Stability',
                'Parameter Correlation Matrix',
                'Response Distributions', 
                'Sensitivity Scatter Plot'
            ],
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )
        
        # Parameter sensitivity bar chart
        params = list(sensitivity_results['parameter_sensitivities'].keys())
        stability_correlations = [sensitivity_results['parameter_sensitivities'][p]['stability_correlation'] 
                                for p in params]
        entanglement_correlations = [sensitivity_results['parameter_sensitivities'][p]['entanglement_correlation'] 
                                   for p in params]
        
        fig.add_trace(
            go.Bar(
                x=params,
                y=stability_correlations,
                name='Stability Correlation',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=params,
                y=entanglement_correlations,
                name='Entanglement Correlation',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Parameter correlation heatmap
        corr_matrix = sensitivity_results['correlation_matrix']['matrix']
        corr_params = sensitivity_results['correlation_matrix']['parameters']
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=corr_params,
                y=corr_params,
                colorscale='RdBu',
                zmid=0,
                showscale=True
            ),
            row=1, col=2
        )
        
        # Response distributions
        responses = ['stability', 'entanglement', 'field']
        response_data = [
            sensitivity_results['stability_response'],
            sensitivity_results['entanglement_response'], 
            sensitivity_results['field_response']
        ]
        
        for i, (response, data) in enumerate(zip(responses, response_data)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=response.title(),
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        # Sensitivity scatter plot
        stability_vals = sensitivity_results['stability_response']
        entanglement_vals = sensitivity_results['entanglement_response']
        
        fig.add_trace(
            go.Scatter(
                x=stability_vals,
                y=entanglement_vals,
                mode='markers',
                marker=dict(
                    color=sensitivity_results['field_response'],
                    colorscale='Viridis',
                    size=6,
                    colorbar=dict(title='Field Strength', x=1.1)
                ),
                name='Sensitivity Samples'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Comprehensive Parameter Sensitivity Analysis',
            height=800,
            width=1200,
            showlegend=True
        )
        
        return fig
    
    def map_stability_boundaries(self, parameter_pairs: list = None,
                               resolution: int = 25,
                               stability_threshold: float = 0.5) -> dict:
        """Map stability boundaries in parameter space.
        
        Args:
            parameter_pairs: List of (param1, param2) pairs to analyze
            resolution: Grid resolution for boundary mapping
            stability_threshold: Stability threshold for boundaries
            
        Returns:
            Stability boundary mapping results
        """
        
        if parameter_pairs is None:
            parameter_pairs = [
                ('throat_radius', 'mass'),
                ('throat_radius', 'exotic_matter_density'),
                ('mass', 'exotic_matter_density')
            ]
        
        print(f"Mapping stability boundaries for {len(parameter_pairs)} parameter pairs...")
        
        boundary_results = {}
        
        for param1, param2 in parameter_pairs:
            print(f"  Analyzing {param1} vs {param2}...")
            
            # Get parameter ranges
            range1 = self.interface.config.parameter_ranges.get(param1, (0.1, 10))
            range2 = self.interface.config.parameter_ranges.get(param2, (0.1, 10))
            
            # Create parameter grids
            if param1 == 'num_qubits':
                p1_values = np.linspace(range1[0], range1[1], resolution//2, dtype=int)
                p1_values = np.unique(p1_values)
            else:
                p1_values = np.linspace(range1[0], range1[1], resolution)
            
            if param2 == 'num_qubits':
                p2_values = np.linspace(range2[0], range2[1], resolution//2, dtype=int)
                p2_values = np.unique(p2_values)
            else:
                p2_values = np.linspace(range2[0], range2[1], resolution)
            
            # Store original parameters
            orig_params = {param1: self.interface.current_parameters[param1],
                          param2: self.interface.current_parameters[param2]}
            
            # Compute stability map
            stability_map = np.zeros((len(p1_values), len(p2_values)))
            
            for i, p1_val in enumerate(p1_values):
                for j, p2_val in enumerate(p2_values):
                    try:
                        self.interface.set_parameters(**{param1: p1_val, param2: p2_val})
                        stability = self.interface._compute_stability_score()
                        stability_map[i, j] = stability
                    except:
                        stability_map[i, j] = 0.0
            
            # Find stability boundaries
            stable_region = stability_map >= stability_threshold
            unstable_region = stability_map < stability_threshold
            
            # Calculate boundary points
            boundary_points = []
            for i in range(len(p1_values)-1):
                for j in range(len(p2_values)-1):
                    # Check if boundary crosses this cell
                    cell_values = [
                        stability_map[i, j], stability_map[i+1, j],
                        stability_map[i, j+1], stability_map[i+1, j+1]
                    ]
                    
                    stable_count = sum(1 for v in cell_values if v >= stability_threshold)
                    
                    if 0 < stable_count < 4:  # Boundary crosses this cell
                        boundary_points.append((p1_values[i], p2_values[j]))
            
            # Store results
            boundary_results[f"{param1}_vs_{param2}"] = {
                'param1': param1,
                'param2': param2,
                'param1_values': p1_values,
                'param2_values': p2_values,
                'stability_map': stability_map,
                'stability_threshold': stability_threshold,
                'stable_fraction': np.sum(stable_region) / stability_map.size,
                'boundary_points': boundary_points,
                'max_stability': np.max(stability_map),
                'min_stability': np.min(stability_map)
            }
            
            # Restore parameters
            self.interface.set_parameters(**orig_params)
            
            print(f"    Stable region fraction: {np.sum(stable_region) / stability_map.size:.3f}")
            print(f"    Boundary points found: {len(boundary_points)}")
        
        print("Stability boundary mapping completed!")
        
        return boundary_results
    
    def visualize_stability_boundaries(self, boundary_results: dict) -> go.Figure:
        """Visualize stability boundaries.
        
        Args:
            boundary_results: Results from boundary mapping
            
        Returns:
            Boundary visualization figure
        """
        
        num_pairs = len(boundary_results)
        
        if num_pairs == 1:
            rows, cols = 1, 1
        elif num_pairs == 2:
            rows, cols = 1, 2
        elif num_pairs <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(boundary_results.keys()),
            specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)]
        )
        
        for idx, (key, results) in enumerate(boundary_results.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            # Stability heatmap
            fig.add_trace(
                go.Heatmap(
                    z=results['stability_map'],
                    x=results['param2_values'],
                    y=results['param1_values'],
                    colorscale='RdYlBu',
                    zmin=0, zmax=1,
                    showscale=(idx == 0),
                    colorbar=dict(title='Stability Score') if idx == 0 else None
                ),
                row=row, col=col
            )
            
            # Boundary contour
            fig.add_trace(
                go.Contour(
                    z=results['stability_map'],
                    x=results['param2_values'],
                    y=results['param1_values'],
                    contours=dict(
                        start=results['stability_threshold'],
                        end=results['stability_threshold'],
                        size=0.01,
                        coloring='lines'
                    ),
                    line=dict(color='black', width=3),
                    showscale=False,
                    name=f'Stability Boundary (threshold={results["stability_threshold"]})'
                ),
                row=row, col=col
            )
            
            # Update axes labels
            fig.update_xaxes(title_text=results['param2'], row=row, col=col)
            fig.update_yaxes(title_text=results['param1'], row=row, col=col)
        
        fig.update_layout(
            title='Stability Boundaries in Parameter Space',
            height=400 * rows,
            width=600 * cols
        )
        
        return fig
    
    def run_batch_simulations(self, parameter_sets: list,
                            analysis_functions: list = None,
                            parallel: bool = True) -> dict:
        """Run batch simulations with multiple parameter sets.
        
        Args:
            parameter_sets: List of parameter dictionaries
            analysis_functions: List of analysis functions to run
            parallel: Whether to run simulations in parallel
            
        Returns:
            Batch simulation results
        """
        
        if analysis_functions is None:
            analysis_functions = [
                ('stability', self.interface._compute_stability_score),
                ('entanglement', self.interface._compute_entanglement_measure),
                ('field_strength', self.interface._compute_field_strength)
            ]
        
        print(f"Running batch simulations for {len(parameter_sets)} parameter sets...")
        
        def run_single_simulation(param_set):
            """Run simulation for single parameter set."""
            
            results = {'parameters': param_set}
            
            try:
                # Set parameters
                self.interface.set_parameters(**param_set)
                
                # Run analysis functions
                for name, func in analysis_functions:
                    try:
                        result = func()
                        results[name] = result
                    except Exception as e:
                        results[name] = None
                        results[f'{name}_error'] = str(e)
                
                return results
                
            except Exception as e:
                results['error'] = str(e)
                return results
        
        # Run simulations
        if parallel and len(parameter_sets) > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run_single_simulation, param_set) 
                          for param_set in parameter_sets]
                
                batch_results = []
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    batch_results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/{len(parameter_sets)} simulations")
        else:
            # Sequential execution
            batch_results = []
            for i, param_set in enumerate(parameter_sets):
                result = run_single_simulation(param_set)
                batch_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(parameter_sets)} simulations")
        
        # Organize results
        organized_results = {
            'parameter_sets': parameter_sets,
            'analysis_functions': [name for name, _ in analysis_functions],
            'results': batch_results,
            'summary_statistics': {}
        }
        
        # Compute summary statistics
        for name, _ in analysis_functions:
            values = [r.get(name) for r in batch_results if r.get(name) is not None]
            
            if values:
                organized_results['summary_statistics'][name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        print("Batch simulations completed!")
        
        return organized_results
    
    def create_advanced_analysis_report(self, sensitivity_results: dict = None,
                                      boundary_results: dict = None,
                                      batch_results: dict = None) -> go.Figure:
        """Create comprehensive advanced analysis report.
        
        Args:
            sensitivity_results: Sensitivity analysis results
            boundary_results: Boundary mapping results  
            batch_results: Batch simulation results
            
        Returns:
            Comprehensive analysis report figure
        """
        
        # Create adaptive subplot layout
        subplot_specs = []
        subplot_titles = []
        
        if sensitivity_results:
            subplot_specs.append({'type': 'bar'})
            subplot_titles.append('Parameter Sensitivities')
        
        if boundary_results:
            subplot_specs.append({'type': 'scatter'}) 
            subplot_titles.append('Stability Boundaries')
        
        if batch_results:
            subplot_specs.append({'type': 'box'})
            subplot_titles.append('Batch Results Distribution')
        
        # Add general analysis panel
        subplot_specs.append({'type': 'scatter'})
        subplot_titles.append('Analysis Summary')
        
        # Determine layout
        num_plots = len(subplot_specs)
        if num_plots <= 2:
            rows, cols = 1, num_plots
        else:
            rows = 2
            cols = (num_plots + 1) // 2
        
        # Pad specs to match grid
        while len(subplot_specs) < rows * cols:
            subplot_specs.append({'type': 'scatter'})
            subplot_titles.append('')
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[subplot_specs[i:i+cols] for i in range(0, len(subplot_specs), cols)]
        )
        
        plot_idx = 0
        
        # Add sensitivity analysis
        if sensitivity_results:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            params = list(sensitivity_results['parameter_sensitivities'].keys())
            magnitudes = [sensitivity_results['parameter_sensitivities'][p]['sensitivity_magnitude'] 
                         for p in params]
            
            fig.add_trace(
                go.Bar(
                    x=params,
                    y=magnitudes,
                    name='Sensitivity Magnitude',
                    marker_color='blue'
                ),
                row=row, col=col
            )
        
        # Add boundary analysis
        if boundary_results:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            # Combine boundary data from all parameter pairs
            all_stable_fractions = []
            pair_names = []
            
            for key, results in boundary_results.items():
                all_stable_fractions.append(results['stable_fraction'])
                pair_names.append(key.replace('_vs_', ' vs '))
            
            fig.add_trace(
                go.Bar(
                    x=pair_names,
                    y=all_stable_fractions,
                    name='Stable Region Fraction',
                    marker_color='green'
                ),
                row=row, col=col
            )
        
        # Add batch results
        if batch_results:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            for metric in batch_results['analysis_functions']:
                values = [r.get(metric) for r in batch_results['results'] 
                         if r.get(metric) is not None]
                
                if values:
                    fig.add_trace(
                        go.Box(
                            y=values,
                            name=metric.title(),
                            boxpoints='outliers'
                        ),
                        row=row, col=col
                    )
        
        # Add summary analysis
        plot_idx += 1
        row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
        
        # Create summary metrics
        summary_metrics = ['Sensitivity Analysis', 'Boundary Mapping', 'Batch Simulations']
        completion_status = [
            1 if sensitivity_results else 0,
            1 if boundary_results else 0,
            1 if batch_results else 0
        ]
        
        fig.add_trace(
            go.Bar(
                x=summary_metrics,
                y=completion_status,
                name='Analysis Completion',
                marker_color=['green' if x else 'red' for x in completion_status]
            ),
            row=row, col=col
        )
        
        fig.update_layout(
            title='Advanced Analysis Tools Report',
            height=400 * rows,
            width=600 * cols,
            showlegend=True
        )
        
        return fig


def main():
    """Run advanced analysis tools demonstration."""
    
    print("="*70)
    print("ADVANCED ANALYSIS TOOLS DEMONSTRATION")
    print("="*70)
    
    # Initialize exploration interface
    config = ExplorationConfig(
        workspace_dir="advanced_analysis_results",
        session_name="advanced_analysis_demo",
        parameter_sweep_points=20
    )
    
    print("\n1. Initializing interface...")
    interface = ScientificExplorationInterface(config)
    
    # Initialize advanced analysis tools
    print("2. Initializing advanced analysis tools...")
    analysis_tools = AdvancedAnalysisTools(interface)
    
    # Set base parameters
    base_params = {
        'throat_radius': 3e3,
        'mass': 8e29,
        'num_qubits': 4,
        'exotic_matter_density': -2e15,
        'entanglement_strength': 1.0
    }
    
    interface.set_parameters(**base_params)
    
    # Run sensitivity analysis
    print("\n3. Running parameter sensitivity analysis...")
    try:
        sensitivity_results = analysis_tools.run_sensitivity_analysis(
            base_parameters=base_params,
            perturbation_size=0.2,
            num_samples=50
        )
        
        # Visualize sensitivity
        sens_fig = analysis_tools.visualize_sensitivity_analysis(sensitivity_results)
        sens_fig.write_html("sensitivity_analysis.html")
        print("   ✓ Sensitivity analysis saved as 'sensitivity_analysis.html'")
        
    except Exception as e:
        print(f"   ⚠ Sensitivity analysis failed: {e}")
        sensitivity_results = None
    
    # Map stability boundaries
    print("\n4. Mapping stability boundaries...")
    try:
        boundary_results = analysis_tools.map_stability_boundaries(
            parameter_pairs=[
                ('throat_radius', 'mass'),
                ('throat_radius', 'exotic_matter_density')
            ],
            resolution=15,
            stability_threshold=0.6
        )
        
        # Visualize boundaries
        boundary_fig = analysis_tools.visualize_stability_boundaries(boundary_results)
        boundary_fig.write_html("stability_boundaries.html")
        print("   ✓ Stability boundaries saved as 'stability_boundaries.html'")
        
    except Exception as e:
        print(f"   ⚠ Boundary mapping failed: {e}")
        boundary_results = None
    
    # Run batch simulations
    print("\n5. Running batch simulations...")
    try:
        # Generate parameter sets
        np.random.seed(123)
        parameter_sets = []
        
        for i in range(25):
            param_set = base_params.copy()
            
            # Random variations
            param_set['throat_radius'] *= np.random.uniform(0.5, 2.0)
            param_set['mass'] *= np.random.uniform(0.3, 3.0)
            param_set['exotic_matter_density'] *= np.random.uniform(0.1, 10.0)
            param_set['entanglement_strength'] *= np.random.uniform(0.5, 1.5)
            
            # Apply bounds
            param_set['throat_radius'] = np.clip(param_set['throat_radius'], 5e2, 1e4)
            param_set['mass'] = np.clip(param_set['mass'], 1e29, 1e31)
            param_set['exotic_matter_density'] = np.clip(param_set['exotic_matter_density'], -1e16, -1e14)
            param_set['entanglement_strength'] = np.clip(param_set['entanglement_strength'], 0.1, 2.0)
            
            parameter_sets.append(param_set)
        
        batch_results = analysis_tools.run_batch_simulations(
            parameter_sets=parameter_sets,
            parallel=True
        )
        
        print("   ✓ Batch simulations completed")
        print(f"   → Successful simulations: {len([r for r in batch_results['results'] if 'error' not in r])}")
        
        # Print summary statistics
        for metric, stats in batch_results['summary_statistics'].items():
            print(f"   → {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
    except Exception as e:
        print(f"   ⚠ Batch simulations failed: {e}")
        batch_results = None
    
    # Create comprehensive report
    print("\n6. Creating advanced analysis report...")
    try:
        report_fig = analysis_tools.create_advanced_analysis_report(
            sensitivity_results=sensitivity_results,
            boundary_results=boundary_results,
            batch_results=batch_results
        )
        
        report_fig.write_html("advanced_analysis_report.html")
        print("   ✓ Advanced analysis report saved as 'advanced_analysis_report.html'")
        
    except Exception as e:
        print(f"   ⚠ Report generation failed: {e}")
    
    # Export comprehensive results
    print("\n7. Exporting results...")
    try:
        # Combine all results
        combined_results = {
            'sensitivity_analysis': sensitivity_results,
            'boundary_mapping': boundary_results,
            'batch_simulations': batch_results,
            'base_parameters': base_params
        }
        
        # Save as JSON
        import json
        with open('advanced_analysis_complete_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        print("   ✓ Complete results saved as 'advanced_analysis_complete_results.json'")
        
    except Exception as e:
        print(f"   ⚠ Export failed: {e}")
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS DEMONSTRATION COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("• sensitivity_analysis.html - Parameter sensitivity analysis")
    print("• stability_boundaries.html - Stability boundary mapping")
    print("• advanced_analysis_report.html - Comprehensive analysis report")
    print("• advanced_analysis_complete_results.json - All analysis data")
    print("\nThese tools provide deep insights into:")
    print("• Parameter sensitivities and correlations")
    print("• Stability boundaries in parameter space")
    print("• Batch simulation capabilities")
    print("• Comprehensive analysis workflows")


if __name__ == "__main__":
    main()