#!/usr/bin/env python3
"""
Stability Benchmarking Example

This example demonstrates comprehensive stability benchmarking of wormhole
configurations, including stress tests, parameter sensitivity analysis,
and performance optimization studies.

Topics covered:
- Systematic stability testing
- Parameter space exploration
- Stress testing under extreme conditions
- Performance benchmarking
- Statistical analysis of stability
- Robustness evaluation
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import differential_evolution
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import CasimirExoticMatter


class StabilityBenchmark:
    """Comprehensive stability benchmarking suite."""
    
    def __init__(self):
        """Initialize benchmarking system."""
        self.benchmark_results = {}
        self.parameter_ranges = {
            'throat_radius': (100.0, 10000.0),
            'mass': (1e29, 1e32),
            'casimir_energy': (-1e17, -1e13),
            'num_qubits': (4, 12),
            'traversal_probability': (0.1, 1.0),
            'decoherence_rate': (0.001, 0.1)
        }
        
    def run_stability_test(self, params, test_duration=100, test_id=None):
        """Run single stability test with given parameters."""
        
        try:
            start_time = time.time()
            
            # Create configuration
            config = IntegrationConfig(
                simulation_name=f"stability_test_{test_id}",
                time_steps=test_duration,
                dt=0.1,
                num_qubits=int(params['num_qubits']),
                enable_stability_prediction=True,
                enable_real_time_visualization=False
            )
            
            # Initialize framework
            framework = WormholeSimulationFramework(config)
            
            wormhole_params = {
                'b0': params['throat_radius'],
                'mass': params['mass'],
                'casimir_energy': params['casimir_energy']
            }
            
            quantum_params = {
                'num_qubits': int(params['num_qubits']),
                'traversal_probability': params['traversal_probability'],
                'entanglement_strength': 1.0,
                'decoherence_rate': params['decoherence_rate']
            }
            
            ai_params = {
                'stability_threshold': 0.5,
                'optimization_target': 'stability'
            }
            
            framework.initialize_system(
                wormhole_params=wormhole_params,
                quantum_params=quantum_params,
                ai_params=ai_params
            )
            
            # Run simulation
            results = framework.run_simulation()
            
            # Analyze stability
            if results.stability_predictions:
                stability_scores = results.stability_predictions
                mean_stability = np.mean(stability_scores)
                std_stability = np.std(stability_scores)
                min_stability = np.min(stability_scores)
                max_stability = np.max(stability_scores)
                
                # Stability metrics
                stable_fraction = np.sum(np.array(stability_scores) > 0.5) / len(stability_scores)
                stability_trend = np.polyfit(range(len(stability_scores)), stability_scores, 1)[0]
                
                # Convergence analysis
                convergence_time = self._calculate_convergence_time(stability_scores)
                
            else:
                mean_stability = 0.0
                std_stability = 0.0
                min_stability = 0.0
                max_stability = 0.0
                stable_fraction = 0.0
                stability_trend = 0.0
                convergence_time = -1
            
            execution_time = time.time() - start_time
            
            return {
                'test_id': test_id,
                'parameters': params,
                'execution_time': execution_time,
                'stability_metrics': {
                    'mean': mean_stability,
                    'std': std_stability,
                    'min': min_stability,
                    'max': max_stability,
                    'stable_fraction': stable_fraction,
                    'trend': stability_trend,
                    'convergence_time': convergence_time
                },
                'simulation_successful': True,
                'raw_stability_data': stability_scores if results.stability_predictions else []
            }
            
        except Exception as e:
            return {
                'test_id': test_id,
                'parameters': params,
                'execution_time': time.time() - start_time,
                'error': str(e),
                'simulation_successful': False,
                'stability_metrics': None
            }
    
    def _calculate_convergence_time(self, stability_scores, threshold=0.01):
        """Calculate time for stability to converge."""
        if len(stability_scores) < 10:
            return -1
            
        # Look for when stability values stop changing significantly
        window_size = 10
        for i in range(window_size, len(stability_scores)):
            recent_values = stability_scores[i-window_size:i]
            if np.std(recent_values) < threshold:
                return i
        
        return -1  # Did not converge
    
    def parameter_sweep_study(self, param_name, num_points=20, num_trials=5):
        """Systematic parameter sweep study."""
        
        print(f"🔍 Running parameter sweep for {param_name}...")
        
        # Define parameter range
        param_min, param_max = self.parameter_ranges[param_name]
        
        if param_name == 'num_qubits':
            # Integer parameter
            param_values = np.linspace(param_min, param_max, num_points, dtype=int)
        else:
            # Continuous parameter
            param_values = np.linspace(param_min, param_max, num_points)
        
        sweep_results = []
        
        for i, param_value in enumerate(param_values):
            print(f"   Testing {param_name} = {param_value:.2e} ({i+1}/{num_points})")
            
            # Default parameters
            default_params = {
                'throat_radius': 1000.0,
                'mass': 1e30,
                'casimir_energy': -1e15,
                'num_qubits': 6,
                'traversal_probability': 0.8,
                'decoherence_rate': 0.01
            }
            
            # Set the parameter being swept
            default_params[param_name] = param_value
            
            # Run multiple trials for statistical significance
            trial_results = []
            for trial in range(num_trials):
                result = self.run_stability_test(
                    default_params, 
                    test_duration=50,
                    test_id=f"sweep_{param_name}_{i}_{trial}"
                )
                trial_results.append(result)
            
            # Aggregate trial results
            successful_trials = [r for r in trial_results if r['simulation_successful']]
            
            if successful_trials:
                mean_stabilities = [r['stability_metrics']['mean'] for r in successful_trials]
                aggregate_result = {
                    'parameter_value': param_value,
                    'mean_stability': np.mean(mean_stabilities),
                    'std_stability': np.std(mean_stabilities),
                    'success_rate': len(successful_trials) / num_trials,
                    'all_trials': trial_results
                }
            else:
                aggregate_result = {
                    'parameter_value': param_value,
                    'mean_stability': 0.0,
                    'std_stability': 0.0,
                    'success_rate': 0.0,
                    'all_trials': trial_results
                }
            
            sweep_results.append(aggregate_result)
        
        self.benchmark_results[f'{param_name}_sweep'] = {
            'parameter_name': param_name,
            'parameter_values': param_values.tolist(),
            'results': sweep_results
        }
        
        print(f"   ✓ Completed {param_name} parameter sweep")
        return sweep_results
    
    def stress_test_suite(self, num_stress_tests=50):
        """Run stress tests with extreme parameter combinations."""
        
        print("💪 Running stability stress tests...")
        
        stress_results = []
        
        for i in range(num_stress_tests):
            # Generate random extreme parameters
            stress_params = self._generate_stress_parameters()
            
            print(f"   Stress test {i+1}/{num_stress_tests}: Extreme configuration")
            
            result = self.run_stability_test(
                stress_params,
                test_duration=75,
                test_id=f"stress_{i}"
            )
            
            stress_results.append(result)
            
            if not result['simulation_successful']:
                print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
            else:
                stability = result['stability_metrics']['mean']
                print(f"     ✓ Stability: {stability:.3f}")
        
        # Analyze stress test results
        successful_tests = [r for r in stress_results if r['simulation_successful']]
        failure_rate = 1.0 - len(successful_tests) / num_stress_tests
        
        if successful_tests:
            stress_stabilities = [r['stability_metrics']['mean'] for r in successful_tests]
            mean_stress_stability = np.mean(stress_stabilities)
            robustness_score = len(successful_tests) / num_stress_tests
        else:
            mean_stress_stability = 0.0
            robustness_score = 0.0
        
        self.benchmark_results['stress_tests'] = {
            'num_tests': num_stress_tests,
            'failure_rate': failure_rate,
            'mean_stability_under_stress': mean_stress_stability,
            'robustness_score': robustness_score,
            'detailed_results': stress_results
        }
        
        print(f"   ✓ Stress tests completed")
        print(f"   Failure rate: {failure_rate:.1%}")
        print(f"   Robustness score: {robustness_score:.3f}")
        
        return stress_results
    
    def _generate_stress_parameters(self):
        """Generate extreme parameter combinations for stress testing."""
        
        # Use extreme values from parameter ranges
        stress_params = {}
        
        for param_name, (param_min, param_max) in self.parameter_ranges.items():
            # Randomly choose from extreme ends of parameter space
            if np.random.random() > 0.5:
                # High extreme
                if param_name == 'num_qubits':
                    stress_params[param_name] = int(param_max)
                else:
                    stress_params[param_name] = param_max * (0.8 + 0.2 * np.random.random())
            else:
                # Low extreme
                if param_name == 'num_qubits':
                    stress_params[param_name] = int(param_min)
                else:
                    stress_params[param_name] = param_min * (1.0 + 0.2 * np.random.random())
        
        return stress_params
    
    def performance_benchmark(self, configurations):
        """Benchmark computational performance across different configurations."""
        
        print("⚡ Running performance benchmarks...")
        
        performance_results = []
        
        for i, config in enumerate(configurations):
            print(f"   Benchmark {i+1}/{len(configurations)}: {config['name']}")
            
            # Run multiple times for statistical accuracy
            execution_times = []
            memory_usage = []
            
            for trial in range(3):  # 3 trials per configuration
                result = self.run_stability_test(
                    config['parameters'],
                    test_duration=config.get('duration', 100),
                    test_id=f"perf_{i}_{trial}"
                )
                
                if result['simulation_successful']:
                    execution_times.append(result['execution_time'])
                    # Memory usage would need to be tracked separately
                    memory_usage.append(0)  # Placeholder
            
            if execution_times:
                perf_result = {
                    'configuration_name': config['name'],
                    'mean_execution_time': np.mean(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'max_execution_time': np.max(execution_times),
                    'success_rate': len(execution_times) / 3,
                    'parameters': config['parameters']
                }
            else:
                perf_result = {
                    'configuration_name': config['name'],
                    'mean_execution_time': float('inf'),
                    'success_rate': 0.0,
                    'parameters': config['parameters']
                }
            
            performance_results.append(perf_result)
            print(f"     Avg time: {perf_result['mean_execution_time']:.2f}s")
        
        self.benchmark_results['performance'] = performance_results
        
        print("   ✓ Performance benchmarks completed")
        return performance_results
    
    def statistical_analysis(self):
        """Perform statistical analysis of all benchmark results."""
        
        print("📊 Performing statistical analysis...")
        
        analysis = {
            'timestamp': str(np.datetime64('now')),
            'summary_statistics': {},
            'correlations': {},
            'distributions': {},
            'recommendations': []
        }
        
        # Analyze parameter sweeps
        for sweep_name, sweep_data in self.benchmark_results.items():
            if '_sweep' in sweep_name:
                param_name = sweep_data['parameter_name']
                results = sweep_data['results']
                
                param_values = [r['parameter_value'] for r in results]
                stabilities = [r['mean_stability'] for r in results]
                
                # Correlation analysis
                if len(param_values) > 3:
                    correlation, p_value = stats.pearsonr(param_values, stabilities)
                    analysis['correlations'][param_name] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significance': p_value < 0.05
                    }
        
        # Stress test analysis
        if 'stress_tests' in self.benchmark_results:
            stress_data = self.benchmark_results['stress_tests']
            analysis['summary_statistics']['stress_tests'] = {
                'failure_rate': stress_data['failure_rate'],
                'robustness_score': stress_data['robustness_score'],
                'mean_stability_under_stress': stress_data['mean_stability_under_stress']
            }
        
        # Performance analysis
        if 'performance' in self.benchmark_results:
            perf_data = self.benchmark_results['performance']
            execution_times = [p['mean_execution_time'] for p in perf_data 
                             if p['mean_execution_time'] != float('inf')]
            
            if execution_times:
                analysis['summary_statistics']['performance'] = {
                    'mean_execution_time': np.mean(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'fastest_configuration': min(perf_data, 
                                               key=lambda x: x['mean_execution_time'])['configuration_name']
                }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.benchmark_results['statistical_analysis'] = analysis
        
        print("   ✓ Statistical analysis completed")
        return analysis
    
    def _generate_recommendations(self, analysis):
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        
        # Parameter optimization recommendations
        for param_name, corr_data in analysis['correlations'].items():
            if corr_data['significance']:
                if corr_data['correlation'] > 0.5:
                    recommendations.append(
                        f"Increase {param_name} for better stability (strong positive correlation)")
                elif corr_data['correlation'] < -0.5:
                    recommendations.append(
                        f"Decrease {param_name} for better stability (strong negative correlation)")
        
        # Robustness recommendations
        if 'stress_tests' in analysis['summary_statistics']:
            stress_stats = analysis['summary_statistics']['stress_tests']
            if stress_stats['failure_rate'] > 0.2:
                recommendations.append(
                    "System shows limited robustness - consider parameter bounds validation")
            if stress_stats['robustness_score'] > 0.8:
                recommendations.append(
                    "System demonstrates excellent robustness under stress conditions")
        
        # Performance recommendations
        if 'performance' in analysis['summary_statistics']:
            perf_stats = analysis['summary_statistics']['performance']
            fastest_config = perf_stats.get('fastest_configuration', 'Unknown')
            recommendations.append(
                f"For best performance, use configuration: {fastest_config}")
        
        return recommendations
    
    def save_benchmark_results(self, filename="benchmark_results.json"):
        """Save all benchmark results to file."""
        
        output_path = f"examples/output/{filename}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.benchmark_results.items():
            serializable_results[key] = self._make_serializable(value)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📁 Benchmark results saved to {output_path}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def create_benchmark_visualizations(benchmark):
    """Create comprehensive visualizations of benchmark results."""
    
    print("📈 Creating benchmark visualizations...")
    
    os.makedirs('examples/output', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Parameter sweep plots
    sweep_results = {k: v for k, v in benchmark.benchmark_results.items() if '_sweep' in k}
    
    if sweep_results:
        n_sweeps = len(sweep_results)
        fig, axes = plt.subplots(2, (n_sweeps + 1) // 2, figsize=(15, 10))
        if n_sweeps == 1:
            axes = [axes]
        axes = axes.flatten() if n_sweeps > 1 else axes
        
        for i, (sweep_name, sweep_data) in enumerate(sweep_results.items()):
            param_name = sweep_data['parameter_name']
            results = sweep_data['results']
            
            param_values = [r['parameter_value'] for r in results]
            stabilities = [r['mean_stability'] for r in results]
            errors = [r['std_stability'] for r in results]
            
            axes[i].errorbar(param_values, stabilities, yerr=errors, 
                           marker='o', linewidth=2, capsize=5)
            axes[i].set_xlabel(param_name.replace('_', ' ').title())
            axes[i].set_ylabel('Mean Stability')
            axes[i].set_title(f'Stability vs {param_name}')
            axes[i].grid(True, alpha=0.3)
            
            # Add trendline
            if len(param_values) > 2:
                z = np.polyfit(param_values, stabilities, 1)
                p = np.poly1d(z)
                axes[i].plot(param_values, p(param_values), '--', alpha=0.7, color='red')
        
        # Hide unused subplots
        for i in range(len(sweep_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('examples/output/parameter_sweeps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Stress test results
    if 'stress_tests' in benchmark.benchmark_results:
        stress_data = benchmark.benchmark_results['stress_tests']
        detailed_results = stress_data['detailed_results']
        
        # Extract data for successful tests
        successful_tests = [r for r in detailed_results if r['simulation_successful']]
        
        if successful_tests:
            stabilities = [r['stability_metrics']['mean'] for r in successful_tests]
            convergence_times = [r['stability_metrics']['convergence_time'] for r in successful_tests]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Stability distribution
            ax1.hist(stabilities, bins=20, alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(stabilities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(stabilities):.3f}')
            ax1.set_xlabel('Stability Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Stability Distribution Under Stress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Convergence time distribution
            valid_convergence = [c for c in convergence_times if c > 0]
            if valid_convergence:
                ax2.hist(valid_convergence, bins=15, alpha=0.7, edgecolor='black', color='green')
                ax2.set_xlabel('Convergence Time (steps)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Convergence Time Distribution')
                ax2.grid(True, alpha=0.3)
            
            # Success/Failure pie chart
            success_count = len(successful_tests)
            failure_count = len(detailed_results) - success_count
            
            ax3.pie([success_count, failure_count], 
                   labels=['Successful', 'Failed'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
            ax3.set_title('Stress Test Success Rate')
            
            # Robustness metrics
            metrics = ['Failure Rate', 'Robustness Score', 'Mean Stability']
            values = [stress_data['failure_rate'], 
                     stress_data['robustness_score'],
                     stress_data['mean_stability_under_stress']]
            
            bars = ax4.bar(metrics, values, color=['coral', 'lightblue', 'lightgreen'])
            ax4.set_ylabel('Score')
            ax4.set_title('Robustness Metrics')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('examples/output/stress_test_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Performance benchmarks
    if 'performance' in benchmark.benchmark_results:
        perf_data = benchmark.benchmark_results['performance']
        
        config_names = [p['configuration_name'] for p in perf_data]
        exec_times = [p['mean_execution_time'] for p in perf_data]
        exec_errors = [p.get('std_execution_time', 0) for p in perf_data]
        
        # Filter out infinite times
        valid_indices = [i for i, t in enumerate(exec_times) if t != float('inf')]
        if valid_indices:
            config_names = [config_names[i] for i in valid_indices]
            exec_times = [exec_times[i] for i in valid_indices]
            exec_errors = [exec_errors[i] for i in valid_indices]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(config_names)), exec_times, yerr=exec_errors,
                          capsize=5, alpha=0.7)
            plt.xlabel('Configuration')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Performance Benchmark Results')
            plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Color bars by performance (faster = greener)
            max_time = max(exec_times)
            for bar, time in zip(bars, exec_times):
                normalized_time = 1.0 - (time / max_time)  # Faster = higher value
                bar.set_color(plt.cm.RdYlGn(normalized_time))
            
            plt.tight_layout()
            plt.savefig('examples/output/performance_benchmark.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Statistical analysis summary
    if 'statistical_analysis' in benchmark.benchmark_results:
        stats_data = benchmark.benchmark_results['statistical_analysis']
        
        # Correlation heatmap
        if stats_data['correlations']:
            param_names = list(stats_data['correlations'].keys())
            correlations = [stats_data['correlations'][p]['correlation'] for p in param_names]
            significances = [stats_data['correlations'][p]['significance'] for p in param_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Correlation values
            bars1 = ax1.bar(param_names, correlations, color=['green' if c > 0 else 'red' 
                                                            for c in correlations])
            ax1.set_ylabel('Correlation with Stability')
            ax1.set_title('Parameter-Stability Correlations')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Significance indicators
            significance_values = [1 if s else 0 for s in significances]
            bars2 = ax2.bar(param_names, significance_values, 
                           color=['darkgreen' if s else 'lightgray' for s in significances])
            ax2.set_ylabel('Statistical Significance')
            ax2.set_title('Correlation Significance (p < 0.05)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1.2)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('examples/output/statistical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("   ✓ All benchmark visualizations created")


def main():
    """Run comprehensive stability benchmarking study."""
    
    print("🧪 Comprehensive Wormhole Stability Benchmarking")
    print("=" * 60)
    
    # Initialize benchmark system
    benchmark = StabilityBenchmark()
    
    # Step 1: Parameter sweep studies
    print("\n1. Running parameter sweep studies...")
    
    key_parameters = ['throat_radius', 'mass', 'casimir_energy', 'num_qubits', 'decoherence_rate']
    
    for param in key_parameters:
        benchmark.parameter_sweep_study(param, num_points=15, num_trials=3)
    
    # Step 2: Stress testing
    print("\n2. Running stability stress tests...")
    
    benchmark.stress_test_suite(num_stress_tests=30)
    
    # Step 3: Performance benchmarking
    print("\n3. Running performance benchmarks...")
    
    performance_configs = [
        {
            'name': 'Minimal',
            'parameters': {
                'throat_radius': 500.0, 'mass': 1e29, 'casimir_energy': -1e14,
                'num_qubits': 4, 'traversal_probability': 0.5, 'decoherence_rate': 0.01
            },
            'duration': 50
        },
        {
            'name': 'Standard',
            'parameters': {
                'throat_radius': 1000.0, 'mass': 1e30, 'casimir_energy': -1e15,
                'num_qubits': 6, 'traversal_probability': 0.8, 'decoherence_rate': 0.01
            },
            'duration': 100
        },
        {
            'name': 'High-Fidelity',
            'parameters': {
                'throat_radius': 2000.0, 'mass': 2e30, 'casimir_energy': -5e15,
                'num_qubits': 8, 'traversal_probability': 0.9, 'decoherence_rate': 0.005
            },
            'duration': 150
        },
        {
            'name': 'Maximum',
            'parameters': {
                'throat_radius': 5000.0, 'mass': 5e30, 'casimir_energy': -1e16,
                'num_qubits': 10, 'traversal_probability': 0.95, 'decoherence_rate': 0.001
            },
            'duration': 200
        }
    ]
    
    benchmark.performance_benchmark(performance_configs)
    
    # Step 4: Statistical analysis
    print("\n4. Performing statistical analysis...")
    
    stats_analysis = benchmark.statistical_analysis()
    
    # Step 5: Generate report
    print("\n5. Generating comprehensive report...")
    
    # Display key findings
    print("\n📊 Key Findings:")
    
    # Parameter correlations
    if stats_analysis['correlations']:
        print("\n   Parameter-Stability Correlations:")
        for param, corr_data in stats_analysis['correlations'].items():
            significance = "✓" if corr_data['significance'] else "✗"
            print(f"     {param}: {corr_data['correlation']:+.3f} {significance}")
    
    # Stress test results
    if 'stress_tests' in benchmark.benchmark_results:
        stress_stats = benchmark.benchmark_results['stress_tests']
        print(f"\n   Stress Test Results:")
        print(f"     Failure rate: {stress_stats['failure_rate']:.1%}")
        print(f"     Robustness score: {stress_stats['robustness_score']:.3f}")
        print(f"     Mean stability under stress: {stress_stats['mean_stability_under_stress']:.3f}")
    
    # Performance results
    if 'performance' in benchmark.benchmark_results:
        perf_data = benchmark.benchmark_results['performance']
        fastest = min(perf_data, key=lambda x: x.get('mean_execution_time', float('inf')))
        print(f"\n   Performance Results:")
        print(f"     Fastest configuration: {fastest['configuration_name']}")
        print(f"     Execution time: {fastest.get('mean_execution_time', 0):.2f}s")
    
    # Step 6: Display recommendations
    print("\n🎯 Optimization Recommendations:")
    for i, rec in enumerate(stats_analysis['recommendations'], 1):
        print(f"    {i}. {rec}")
    
    # Step 7: Save results and create visualizations
    print("\n6. Saving results and creating visualizations...")
    
    benchmark.save_benchmark_results()
    create_benchmark_visualizations(benchmark)
    
    print(f"\n🎉 Comprehensive benchmarking completed!")
    print(f"   Parameter sweeps: {len([k for k in benchmark.benchmark_results.keys() if '_sweep' in k])}")
    print(f"   Stress tests: {benchmark.benchmark_results['stress_tests']['num_tests']}")
    print(f"   Performance configs: {len(performance_configs)}")
    print(f"   Statistical analysis: Complete")
    print(f"   Results saved to: examples/output/")
    
    return benchmark, stats_analysis


if __name__ == "__main__":
    try:
        benchmark, analysis = main()
        
        print(f"\n📈 Benchmarking Summary:")
        print(f"   - Parameter optimization guidance available")
        print(f"   - System robustness quantified")
        print(f"   - Performance characteristics mapped")
        print(f"   - Statistical significance validated")
        print(f"   - Comprehensive visualizations generated")
        
    except Exception as e:
        print(f"\n❌ Error in stability benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)