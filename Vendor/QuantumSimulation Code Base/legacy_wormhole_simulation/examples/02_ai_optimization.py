#!/usr/bin/env python3
"""
AI-Driven Parameter Optimization Example

This example demonstrates how to use the AI system to optimize wormhole
parameters for maximum stability and traversability.

Topics covered:
- Multi-objective parameter optimization
- Stability prediction using machine learning
- Anomaly detection in simulation results
- Pareto front analysis
- Advanced AI configuration
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.ai.parameter_optimizer import ParameterOptimizer
from src.ai.stability_predictor import StabilityPredictor
from src.ai.anomaly_detector import AnomalyDetector


class WormholeOptimizationStudy:
    """Advanced wormhole optimization study using AI."""
    
    def __init__(self):
        """Initialize the optimization study."""
        
        # Define parameter bounds for optimization
        self.parameter_bounds = {
            'throat_radius': (500.0, 5000.0),      # 0.5 to 5 km
            'mass': (1e29, 1e31),                  # 0.1 to 10 solar masses
            'traversal_probability': (0.3, 1.0),   # 30% to 100%
            'entanglement_strength': (0.1, 2.0),   # 10% to 200%
            'casimir_energy': (-1e16, -1e14)       # Exotic matter range
        }
        
        # Initialize AI components
        self.optimizer = ParameterOptimizer(self.parameter_bounds)
        self.stability_predictor = StabilityPredictor(model_type="ensemble")
        self.anomaly_detector = AnomalyDetector(algorithm="isolation_forest")
        
        # Results storage
        self.optimization_history = []
        self.evaluation_data = []
        self.pareto_front = []
    
    def evaluate_wormhole_configuration(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate a wormhole configuration and return objective scores.
        
        Args:
            params: Dictionary of wormhole parameters
            
        Returns:
            Dictionary of objective function values
        """
        
        print(f"   Evaluating: R={params['throat_radius']:.0f}m, "
              f"M={params['mass']:.1e}kg, P={params['traversal_probability']:.2f}")
        
        try:
            # Create configuration for this evaluation
            config = IntegrationConfig(
                simulation_name=f"optimization_eval",
                time_steps=50,  # Shorter for optimization speed
                dt=0.2,
                num_qubits=6,
                enable_stability_prediction=True,
                enable_anomaly_detection=True,
                enable_real_time_visualization=False
            )
            
            # Initialize framework
            framework = WormholeSimulationFramework(config)
            
            # Set up parameters
            wormhole_params = {
                'b0': params['throat_radius'],
                'mass': params['mass'],
                'casimir_energy': params['casimir_energy']
            }
            
            quantum_params = {
                'num_qubits': 6,
                'traversal_probability': params['traversal_probability'],
                'entanglement_strength': params['entanglement_strength'],
                'decoherence_rate': 0.01
            }
            
            # Initialize system
            framework.initialize_system(
                wormhole_params=wormhole_params,
                quantum_params=quantum_params
            )
            
            # Run simulation
            results = framework.run_simulation()
            
            # Calculate objectives
            objectives = self._calculate_objectives(params, results, framework)
            
            # Store evaluation data for analysis
            evaluation_data = {
                'parameters': params.copy(),
                'objectives': objectives.copy(),
                'results': results
            }
            self.evaluation_data.append(evaluation_data)
            
            return objectives
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            # Return poor objectives for failed evaluations
            return {
                'stability': 0.0,
                'traversability': 0.0,
                'energy_efficiency': 0.0,
                'quantum_coherence': 0.0
            }
    
    def _calculate_objectives(self, params: Dict[str, float], 
                            results, framework) -> Dict[str, float]:
        """Calculate objective function values from simulation results."""
        
        objectives = {}
        
        # 1. Stability objective
        if results.stability_predictions:
            stability_mean = np.mean(results.stability_predictions)
            stability_std = np.std(results.stability_predictions)
            # Reward high mean stability with low variance
            objectives['stability'] = stability_mean * (1 - 0.5 * stability_std)
        else:
            objectives['stability'] = 0.0
        
        # 2. Traversability objective
        # Based on traversal probability and throat size
        traversability = params['traversal_probability']
        # Bonus for larger throats (easier traversal)
        size_bonus = min(1.0, params['throat_radius'] / 2000.0)
        objectives['traversability'] = traversability * (0.7 + 0.3 * size_bonus)
        
        # 3. Energy efficiency objective
        # Minimize required exotic matter (maximize efficiency)
        energy_magnitude = abs(params['casimir_energy'])
        throat_volume = 4/3 * np.pi * params['throat_radius']**3
        energy_per_volume = energy_magnitude / throat_volume
        # Normalize and invert (lower energy requirement = higher efficiency)
        objectives['energy_efficiency'] = 1.0 / (1.0 + energy_per_volume / 1e12)
        
        # 4. Quantum coherence objective
        if results.quantum_state_evolution:
            # Extract concurrence values
            concurrences = []
            for data in results.quantum_state_evolution:
                if 'concurrence' in data:
                    concurrences.append(data['concurrence'])
            
            if concurrences:
                # Reward sustained entanglement
                mean_concurrence = np.mean(concurrences)
                final_concurrence = concurrences[-1] if concurrences else 0
                objectives['quantum_coherence'] = 0.7 * mean_concurrence + 0.3 * final_concurrence
            else:
                objectives['quantum_coherence'] = params['entanglement_strength'] * 0.5
        else:
            objectives['quantum_coherence'] = params['entanglement_strength'] * 0.5
        
        return objectives
    
    def run_optimization_study(self, num_evaluations: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive optimization study.
        
        Args:
            num_evaluations: Number of parameter configurations to evaluate
            
        Returns:
            Complete optimization results
        """
        
        print("🤖 AI-Driven Wormhole Optimization Study")
        print("=" * 60)
        
        print(f"\n1. Configuration:")
        print(f"   Parameter bounds: {len(self.parameter_bounds)} dimensions")
        print(f"   Evaluations planned: {num_evaluations}")
        print(f"   Objectives: stability, traversability, energy efficiency, quantum coherence")
        
        # Run multi-objective optimization
        print(f"\n2. Running multi-objective optimization...")
        
        optimization_results = self.optimizer.optimize(
            objective_function=self.evaluate_wormhole_configuration,
            objectives=['stability', 'traversability', 'energy_efficiency', 'quantum_coherence'],
            max_iterations=num_evaluations
        )
        
        print(f"   ✓ Optimization completed")
        print(f"   ✓ {len(self.evaluation_data)} configurations evaluated")
        
        # Analyze results
        print(f"\n3. Analyzing optimization results...")
        
        analysis = self._analyze_optimization_results(optimization_results)
        
        # Train stability predictor
        print(f"\n4. Training stability predictor...")
        
        self._train_stability_predictor()
        
        # Detect anomalies
        print(f"\n5. Running anomaly detection...")
        
        anomalies = self._detect_anomalies()
        
        # Create visualizations
        print(f"\n6. Creating optimization visualizations...")
        
        self._create_optimization_plots(analysis, anomalies)
        
        # Compile final results
        final_results = {
            'optimization_results': optimization_results,
            'analysis': analysis,
            'anomalies': anomalies,
            'best_configurations': self._get_best_configurations(),
            'evaluation_data': self.evaluation_data
        }
        
        return final_results
    
    def _analyze_optimization_results(self, opt_results) -> Dict[str, Any]:
        """Analyze optimization results and compute statistics."""
        
        analysis = {}
        
        # Extract all objective values
        all_objectives = {
            'stability': [],
            'traversability': [],
            'energy_efficiency': [],
            'quantum_coherence': []
        }
        
        for eval_data in self.evaluation_data:
            objectives = eval_data['objectives']
            for obj_name, obj_value in objectives.items():
                if obj_name in all_objectives:
                    all_objectives[obj_name].append(obj_value)
        
        # Compute statistics
        analysis['objective_statistics'] = {}
        for obj_name, values in all_objectives.items():
            if values:
                analysis['objective_statistics'][obj_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        # Find Pareto-optimal solutions
        analysis['pareto_front'] = self._compute_pareto_front()
        
        # Parameter sensitivity analysis
        analysis['parameter_sensitivity'] = self._analyze_parameter_sensitivity()
        
        # Correlation analysis
        analysis['correlations'] = self._compute_objective_correlations()
        
        return analysis
    
    def _compute_pareto_front(self) -> List[Dict]:
        """Compute Pareto-optimal solutions."""
        
        pareto_solutions = []
        
        # Extract objective values for all evaluations
        objectives_matrix = []
        for eval_data in self.evaluation_data:
            objectives = eval_data['objectives']
            obj_vector = [
                objectives['stability'],
                objectives['traversability'], 
                objectives['energy_efficiency'],
                objectives['quantum_coherence']
            ]
            objectives_matrix.append(obj_vector)
        
        objectives_matrix = np.array(objectives_matrix)
        
        # Find Pareto-optimal points
        pareto_indices = []
        for i in range(len(objectives_matrix)):
            is_pareto = True
            for j in range(len(objectives_matrix)):
                if i != j:
                    # Check if j dominates i (all objectives >= and at least one >)
                    dominates = all(objectives_matrix[j] >= objectives_matrix[i])
                    strictly_better = any(objectives_matrix[j] > objectives_matrix[i])
                    
                    if dominates and strictly_better:
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        # Store Pareto solutions
        for idx in pareto_indices:
            pareto_solutions.append({
                'parameters': self.evaluation_data[idx]['parameters'],
                'objectives': self.evaluation_data[idx]['objectives'],
                'index': idx
            })
        
        self.pareto_front = pareto_solutions
        return pareto_solutions
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, float]:
        """Analyze parameter sensitivity to objectives."""
        
        sensitivity = {}
        
        # Compute correlation between each parameter and combined objective
        for param_name in self.parameter_bounds.keys():
            param_values = []
            combined_objectives = []
            
            for eval_data in self.evaluation_data:
                param_values.append(eval_data['parameters'][param_name])
                
                # Combined objective (weighted sum)
                objectives = eval_data['objectives']
                combined = (0.3 * objectives['stability'] + 
                           0.25 * objectives['traversability'] +
                           0.25 * objectives['energy_efficiency'] +
                           0.2 * objectives['quantum_coherence'])
                combined_objectives.append(combined)
            
            # Compute correlation coefficient
            if len(param_values) > 1:
                correlation = np.corrcoef(param_values, combined_objectives)[0, 1]
                sensitivity[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                sensitivity[param_name] = 0.0
        
        return sensitivity
    
    def _compute_objective_correlations(self) -> Dict[str, Dict[str, float]]:
        """Compute correlations between objectives."""
        
        obj_names = ['stability', 'traversability', 'energy_efficiency', 'quantum_coherence']
        correlations = {}
        
        # Extract objective values
        obj_data = {name: [] for name in obj_names}
        for eval_data in self.evaluation_data:
            for obj_name in obj_names:
                obj_data[obj_name].append(eval_data['objectives'][obj_name])
        
        # Compute pairwise correlations
        for obj1 in obj_names:
            correlations[obj1] = {}
            for obj2 in obj_names:
                if len(obj_data[obj1]) > 1 and len(obj_data[obj2]) > 1:
                    corr = np.corrcoef(obj_data[obj1], obj_data[obj2])[0, 1]
                    correlations[obj1][obj2] = corr if not np.isnan(corr) else 0.0
                else:
                    correlations[obj1][obj2] = 0.0
        
        return correlations
    
    def _train_stability_predictor(self):
        """Train stability predictor on optimization data."""
        
        if len(self.evaluation_data) < 10:
            print("   ⚠️  Not enough data for training (need at least 10 samples)")
            return
        
        # Prepare training data
        features = []
        labels = []
        
        for eval_data in self.evaluation_data:
            params = eval_data['parameters']
            feature_vector = [
                params['throat_radius'] / 1000.0,  # Normalize to km
                params['mass'] / 1e30,             # Normalize to solar masses  
                params['traversal_probability'],
                params['entanglement_strength'],
                params['casimir_energy'] / 1e15    # Normalize
            ]
            features.append(feature_vector)
            labels.append(eval_data['objectives']['stability'])
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Train predictor
        training_metrics = self.stability_predictor.train(features, labels)
        
        print(f"   ✓ Predictor trained (accuracy: {training_metrics.get('accuracy', 0):.3f})")
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous configurations."""
        
        if len(self.evaluation_data) < 5:
            return {'anomalies': [], 'anomaly_scores': []}
        
        # Prepare data for anomaly detection
        feature_data = []
        for eval_data in self.evaluation_data:
            objectives = eval_data['objectives']
            feature_vector = [
                objectives['stability'],
                objectives['traversability'],
                objectives['energy_efficiency'],
                objectives['quantum_coherence']
            ]
            feature_data.append(feature_vector)
        
        feature_data = np.array(feature_data)
        
        # Fit anomaly detector on all data
        self.anomaly_detector.fit(feature_data)
        
        # Detect anomalies
        anomaly_labels, anomaly_scores = self.anomaly_detector.predict_anomaly(feature_data)
        
        # Identify anomalous configurations
        anomalies = []
        for i, (is_anomaly, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
            if is_anomaly:
                anomalies.append({
                    'index': i,
                    'parameters': self.evaluation_data[i]['parameters'],
                    'objectives': self.evaluation_data[i]['objectives'],
                    'anomaly_score': score
                })
        
        print(f"   ✓ Found {len(anomalies)} anomalous configurations")
        
        return {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores.tolist()
        }
    
    def _get_best_configurations(self) -> Dict[str, Dict]:
        """Get best configurations for each objective."""
        
        best_configs = {}
        
        obj_names = ['stability', 'traversability', 'energy_efficiency', 'quantum_coherence']
        
        for obj_name in obj_names:
            best_value = -1
            best_config = None
            
            for eval_data in self.evaluation_data:
                obj_value = eval_data['objectives'][obj_name]
                if obj_value > best_value:
                    best_value = obj_value
                    best_config = {
                        'parameters': eval_data['parameters'],
                        'objectives': eval_data['objectives'],
                        'objective_value': obj_value
                    }
            
            if best_config:
                best_configs[obj_name] = best_config
        
        return best_configs
    
    def _create_optimization_plots(self, analysis: Dict, anomalies: Dict):
        """Create comprehensive optimization visualizations."""
        
        # Create output directory
        os.makedirs('examples/output', exist_ok=True)
        
        # Plot 1: Pareto front (2D projections)
        self._plot_pareto_front()
        
        # Plot 2: Parameter sensitivity
        self._plot_parameter_sensitivity(analysis['parameter_sensitivity'])
        
        # Plot 3: Objective correlations
        self._plot_objective_correlations(analysis['correlations'])
        
        # Plot 4: Optimization history
        self._plot_optimization_history()
        
        # Plot 5: Anomaly analysis
        self._plot_anomaly_analysis(anomalies)
    
    def _plot_pareto_front(self):
        """Plot Pareto front projections."""
        
        if not self.pareto_front:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        obj_names = ['stability', 'traversability', 'energy_efficiency', 'quantum_coherence']
        obj_pairs = [
            ('stability', 'traversability'),
            ('stability', 'energy_efficiency'),
            ('stability', 'quantum_coherence'),
            ('traversability', 'energy_efficiency'),
            ('traversability', 'quantum_coherence'),
            ('energy_efficiency', 'quantum_coherence')
        ]
        
        for i, (obj1, obj2) in enumerate(obj_pairs):
            ax = axes[i]
            
            # Plot all points
            all_obj1 = [ed['objectives'][obj1] for ed in self.evaluation_data]
            all_obj2 = [ed['objectives'][obj2] for ed in self.evaluation_data]
            ax.scatter(all_obj1, all_obj2, alpha=0.3, c='lightblue', s=30, label='All points')
            
            # Plot Pareto points
            pareto_obj1 = [pf['objectives'][obj1] for pf in self.pareto_front]
            pareto_obj2 = [pf['objectives'][obj2] for pf in self.pareto_front]
            ax.scatter(pareto_obj1, pareto_obj2, c='red', s=60, label='Pareto front', zorder=5)
            
            ax.set_xlabel(obj1.replace('_', ' ').title())
            ax.set_ylabel(obj2.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Pareto Front Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('examples/output/pareto_front.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, sensitivity: Dict[str, float]):
        """Plot parameter sensitivity analysis."""
        
        plt.figure(figsize=(10, 6))
        
        params = list(sensitivity.keys())
        values = list(sensitivity.values())
        
        # Clean parameter names for display
        param_labels = []
        for param in params:
            clean_name = param.replace('_', ' ').title()
            if 'Radius' in clean_name:
                clean_name = 'Throat Radius'
            elif 'Probability' in clean_name:
                clean_name = 'Traversal Prob.'
            elif 'Strength' in clean_name:
                clean_name = 'Entanglement'
            elif 'Energy' in clean_name:
                clean_name = 'Casimir Energy'
            param_labels.append(clean_name)
        
        bars = plt.bar(param_labels, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Parameters')
        plt.ylabel('Sensitivity (|Correlation|)')
        plt.title('Parameter Sensitivity to Combined Objective')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('examples/output/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_objective_correlations(self, correlations: Dict[str, Dict[str, float]]):
        """Plot objective correlation matrix."""
        
        obj_names = ['stability', 'traversability', 'energy_efficiency', 'quantum_coherence']
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(obj_names), len(obj_names)))
        for i, obj1 in enumerate(obj_names):
            for j, obj2 in enumerate(obj_names):
                corr_matrix[i, j] = correlations[obj1][obj2]
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, label='Correlation Coefficient')
        
        # Set ticks and labels
        clean_labels = ['Stability', 'Traversability', 'Energy Eff.', 'Quantum Coh.']
        plt.xticks(range(len(obj_names)), clean_labels, rotation=45)
        plt.yticks(range(len(obj_names)), clean_labels)
        
        # Add correlation values as text
        for i in range(len(obj_names)):
            for j in range(len(obj_names)):
                text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.title('Objective Correlation Matrix')
        plt.tight_layout()
        plt.savefig('examples/output/objective_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_history(self):
        """Plot optimization convergence history."""
        
        # Extract best objective values over time
        best_stability = []
        best_combined = []
        
        current_best_stability = 0
        current_best_combined = 0
        
        for eval_data in self.evaluation_data:
            objectives = eval_data['objectives']
            stability = objectives['stability']
            combined = (0.3 * objectives['stability'] + 
                       0.25 * objectives['traversability'] +
                       0.25 * objectives['energy_efficiency'] +
                       0.2 * objectives['quantum_coherence'])
            
            current_best_stability = max(current_best_stability, stability)
            current_best_combined = max(current_best_combined, combined)
            
            best_stability.append(current_best_stability)
            best_combined.append(current_best_combined)
        
        plt.figure(figsize=(10, 6))
        
        evaluations = range(1, len(best_stability) + 1)
        plt.plot(evaluations, best_stability, 'b-', linewidth=2, label='Best Stability')
        plt.plot(evaluations, best_combined, 'r-', linewidth=2, label='Best Combined Objective')
        
        plt.xlabel('Evaluation Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization Convergence History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('examples/output/optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_analysis(self, anomalies: Dict):
        """Plot anomaly detection results."""
        
        if not anomalies['anomalies']:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot anomaly scores
        plt.subplot(2, 2, 1)
        plt.hist(anomalies['anomaly_scores'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot anomalies in objective space
        plt.subplot(2, 2, 2)
        normal_stability = []
        normal_traversability = []
        anomaly_stability = []
        anomaly_traversability = []
        
        anomaly_indices = {a['index'] for a in anomalies['anomalies']}
        
        for i, eval_data in enumerate(self.evaluation_data):
            objectives = eval_data['objectives']
            if i in anomaly_indices:
                anomaly_stability.append(objectives['stability'])
                anomaly_traversability.append(objectives['traversability'])
            else:
                normal_stability.append(objectives['stability'])
                normal_traversability.append(objectives['traversability'])
        
        plt.scatter(normal_stability, normal_traversability, alpha=0.6, label='Normal', s=30)
        plt.scatter(anomaly_stability, anomaly_traversability, c='red', label='Anomaly', s=60)
        plt.xlabel('Stability')
        plt.ylabel('Traversability')
        plt.title('Anomalies in Objective Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot parameter distributions for anomalies
        plt.subplot(2, 2, 3)
        if anomalies['anomalies']:
            anomaly_throat_radii = [a['parameters']['throat_radius'] for a in anomalies['anomalies']]
            normal_throat_radii = [ed['parameters']['throat_radius'] for i, ed in enumerate(self.evaluation_data) 
                                 if i not in anomaly_indices]
            
            plt.hist([normal_throat_radii, anomaly_throat_radii], bins=15, alpha=0.7, 
                    label=['Normal', 'Anomaly'], color=['blue', 'red'])
            plt.xlabel('Throat Radius (m)')
            plt.ylabel('Frequency')
            plt.title('Throat Radius Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Anomaly characteristics
        plt.subplot(2, 2, 4)
        if anomalies['anomalies']:
            anomaly_scores = [a['anomaly_score'] for a in anomalies['anomalies']]
            anomaly_stabilities = [a['objectives']['stability'] for a in anomalies['anomalies']]
            
            plt.scatter(anomaly_scores, anomaly_stabilities, c='red', s=60)
            plt.xlabel('Anomaly Score')
            plt.ylabel('Stability')
            plt.title('Anomaly Score vs Stability')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('examples/output/anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run the AI optimization example."""
    
    # Initialize optimization study
    study = WormholeOptimizationStudy()
    
    # Run optimization (use smaller number for demonstration)
    results = study.run_optimization_study(num_evaluations=30)
    
    # Display results
    print(f"\n📊 Optimization Results Summary:")
    print(f"=" * 60)
    
    # Show Pareto front
    pareto_front = results['analysis']['pareto_front']
    print(f"\nPareto-optimal solutions found: {len(pareto_front)}")
    
    for i, solution in enumerate(pareto_front[:3]):  # Show top 3
        params = solution['parameters']
        objectives = solution['objectives']
        
        print(f"\n  Solution {i+1}:")
        print(f"    Throat radius: {params['throat_radius']:.0f} m")
        print(f"    Mass: {params['mass']:.1e} kg")
        print(f"    Traversal prob: {params['traversal_probability']:.3f}")
        print(f"    Stability: {objectives['stability']:.3f}")
        print(f"    Traversability: {objectives['traversability']:.3f}")
        print(f"    Energy efficiency: {objectives['energy_efficiency']:.3f}")
    
    # Show best individual objectives
    best_configs = results['best_configurations']
    print(f"\n🏆 Best Individual Objectives:")
    
    for obj_name, config in best_configs.items():
        print(f"\n  Best {obj_name.replace('_', ' ').title()}:")
        print(f"    Value: {config['objective_value']:.3f}")
        print(f"    Throat radius: {config['parameters']['throat_radius']:.0f} m")
        print(f"    Mass: {config['parameters']['mass']:.1e} kg")
    
    # Show parameter sensitivity
    sensitivity = results['analysis']['parameter_sensitivity']
    print(f"\n📈 Parameter Sensitivity (most influential first):")
    sorted_params = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    for param_name, sens_value in sorted_params:
        print(f"    {param_name.replace('_', ' ').title()}: {sens_value:.3f}")
    
    # Show anomalies
    anomalies = results['anomalies']['anomalies']
    print(f"\n⚠️  Anomalous Configurations Detected: {len(anomalies)}")
    
    for i, anomaly in enumerate(anomalies[:2]):  # Show first 2
        print(f"    Anomaly {i+1}: Score = {anomaly['anomaly_score']:.2f}")
        print(f"      Stability = {anomaly['objectives']['stability']:.3f}")
    
    print(f"\n✅ AI optimization example completed!")
    print(f"   📁 Visualization plots saved to examples/output/")
    print(f"   📊 {len(results['evaluation_data'])} configurations evaluated")
    print(f"   🎯 {len(pareto_front)} Pareto-optimal solutions found")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error running AI optimization example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)