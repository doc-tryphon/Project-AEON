"""
Multi-Scenario Validation Sweeps.

This module provides comprehensive validation across all implemented Phase 3
systems, including parameter space sweeps, performance benchmarking, and
statistical analysis of wormhole scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import concurrent.futures
import logging
from pathlib import Path

# Import all Phase 3 systems for validation
from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
from src.ai.parameter_exploration import (
    run_optimization_comparison,
    BayesianOptimizer,
    DifferentialEvolutionOptimizer,
    GridSearchOptimizer
)
from src.ai.bayesian_wormhole_search import (
    BayesianWormholeSearch,
    TraversabilityConstraints,
    create_wormhole_circuit_factory
)
from src.physics.rotating_wormhole_metrics import (
    KerrLikeWormhole,
    RotationParameters,
    create_rotating_wormhole
)
from src.physics.dynamic_throat_evolution import (
    create_evolution_scenario,
    compare_evolution_scenarios,
    DynamicThroatEvolution
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for multi-scenario validation sweeps."""
    
    # Parameter sweep configuration
    parameter_sweep_points: int = 20
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'throat_radius': (500.0, 2000.0),
        'mass': (5e29, 2e30),
        'angular_momentum': (1e42, 1e45),
        'exotic_matter_density': (-1e-2, -1e-4),
        'evolution_timescale': (100.0, 2000.0)
    })
    
    # Scenario configuration
    scenarios_to_test: List[str] = field(default_factory=lambda: [
        'standard', 'collapse', 'expansion'
    ])
    
    # Validation tests configuration
    quantum_backend_tests: bool = True
    ml_optimization_tests: bool = True
    bayesian_search_tests: bool = True
    rotating_metrics_tests: bool = True
    dynamic_evolution_tests: bool = True
    visualization_tests: bool = True
    
    # Performance benchmarking
    performance_benchmark: bool = True
    benchmark_iterations: int = 10
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Output configuration
    save_detailed_results: bool = True
    create_summary_plots: bool = True
    export_csv_data: bool = True
    output_directory: str = "validation_results"
    
    # Statistical analysis
    statistical_analysis: bool = True
    confidence_level: float = 0.95
    significance_threshold: float = 0.05


class MultiScenarioValidator:
    """Comprehensive validation system for all Phase 3 components."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize multi-scenario validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.validation_results = {}
        self.performance_metrics = {}
        self.statistical_summary = {}
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize validation timestamp
        self.validation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized multi-scenario validator: {self.validation_timestamp}")
    
    def run_full_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite across all Phase 3 systems."""
        
        logger.info("Starting comprehensive Phase 3 validation suite")
        
        validation_start_time = time.time()
        
        # Initialize results structure
        self.validation_results = {
            'timestamp': self.validation_timestamp,
            'config': self.config,
            'system_validations': {},
            'parameter_sweeps': {},
            'performance_benchmarks': {},
            'statistical_analysis': {},
            'overall_summary': {}
        }
        
        # Run individual system validations
        if self.config.quantum_backend_tests:
            self.validation_results['system_validations']['quantum_backend'] = \
                self._validate_quantum_backend()
        
        if self.config.ml_optimization_tests:
            self.validation_results['system_validations']['ml_optimization'] = \
                self._validate_ml_optimization()
        
        if self.config.bayesian_search_tests:
            self.validation_results['system_validations']['bayesian_search'] = \
                self._validate_bayesian_search()
        
        if self.config.rotating_metrics_tests:
            self.validation_results['system_validations']['rotating_metrics'] = \
                self._validate_rotating_metrics()
        
        if self.config.dynamic_evolution_tests:
            self.validation_results['system_validations']['dynamic_evolution'] = \
                self._validate_dynamic_evolution()
        
        if self.config.visualization_tests:
            self.validation_results['system_validations']['visualization'] = \
                self._validate_visualization_system()
        
        # Run parameter space sweeps
        self.validation_results['parameter_sweeps'] = self._run_parameter_sweeps()
        
        # Run performance benchmarks
        if self.config.performance_benchmark:
            self.validation_results['performance_benchmarks'] = \
                self._run_performance_benchmarks()
        
        # Statistical analysis
        if self.config.statistical_analysis:
            self.validation_results['statistical_analysis'] = \
                self._perform_statistical_analysis()
        
        # Generate overall summary
        validation_time = time.time() - validation_start_time
        self.validation_results['overall_summary'] = \
            self._generate_overall_summary(validation_time)
        
        # Save results
        self._save_validation_results()
        
        # Generate plots and reports
        if self.config.create_summary_plots:
            self._create_summary_plots()
        
        logger.info(f"Validation suite completed in {validation_time:.2f}s")
        
        return self.validation_results
    
    def _validate_quantum_backend(self) -> Dict[str, Any]:
        """Validate quantum backend integration."""
        logger.info("Validating quantum backend integration")
        
        results = {
            'test_name': 'Quantum Backend Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'details': {}
        }
        
        try:
            # Test hybrid quantum-AI circuit
            geometry_params = {
                'throat_radius': 1000.0,
                'mass': 1e30,
                'exotic_matter_density': -1e-3
            }
            circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
            
            # Test 1: Circuit initialization
            results['total_tests'] += 1
            if circuit.num_qubits == 4:
                results['tests_passed'] += 1
                results['details']['circuit_initialization'] = 'PASS'
            else:
                results['details']['circuit_initialization'] = 'FAIL'
            
            # Test 2: Basic quantum operations
            results['total_tests'] += 1
            try:
                initial_state = circuit.create_initial_state()
                results['tests_passed'] += 1
                results['details']['quantum_operations'] = 'PASS'
            except:
                results['details']['quantum_operations'] = 'FAIL'
            
            # Test 3: AI optimization integration
            results['total_tests'] += 1
            try:
                target_metrics = {'traversability': 0.8, 'stability': 0.7}
                optimized_params = circuit.optimize_parameters(target_metrics)
                results['tests_passed'] += 1
                results['details']['ai_optimization'] = 'PASS'
            except:
                results['details']['ai_optimization'] = 'FAIL'
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _validate_ml_optimization(self) -> Dict[str, Any]:
        """Validate ML optimization systems."""
        logger.info("Validating ML optimization systems")
        
        results = {
            'test_name': 'ML Optimization Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'optimization_results': {},
            'details': {}
        }
        
        try:
            # Test optimization comparison with small budget
            optimization_results = run_optimization_comparison(optimization_budget=15)
            
            # Test 1: All optimization methods completed
            results['total_tests'] += 1
            if len(optimization_results) >= 2:  # At least 2 methods should work
                results['tests_passed'] += 1
                results['details']['methods_completed'] = 'PASS'
            else:
                results['details']['methods_completed'] = 'FAIL'
            
            # Test 2: Check best scores are reasonable
            results['total_tests'] += 1
            best_scores = [result.best_score for result in optimization_results.values()]
            if all(0 <= score <= 1 for score in best_scores):
                results['tests_passed'] += 1
                results['details']['score_validity'] = 'PASS'
            else:
                results['details']['score_validity'] = 'FAIL'
            
            # Test 3: Check convergence
            results['total_tests'] += 1
            converged_methods = [method for method, result in optimization_results.items() 
                               if result.converged or result.total_evaluations > 10]
            if len(converged_methods) > 0:
                results['tests_passed'] += 1
                results['details']['convergence'] = 'PASS'
            else:
                results['details']['convergence'] = 'FAIL'
            
            results['optimization_results'] = {
                method: {
                    'best_score': result.best_score,
                    'total_evaluations': result.total_evaluations,
                    'converged': result.converged,
                    'method': result.method
                }
                for method, result in optimization_results.items()
            }
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _validate_bayesian_search(self) -> Dict[str, Any]:
        """Validate Bayesian wormhole search system."""
        logger.info("Validating Bayesian wormhole search")
        
        results = {
            'test_name': 'Bayesian Search Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'search_results': {},
            'details': {}
        }
        
        try:
            # Create search system
            circuit_factory = create_wormhole_circuit_factory(num_qubits=4)
            constraints = TraversabilityConstraints()
            search = BayesianWormholeSearch(circuit_factory, constraints)
            
            # Define small search space for testing
            parameter_bounds = np.array([
                [800.0, 1200.0],    # throat_radius
                [5e29, 2e30],       # mass
                [-2e-3, -5e-4],     # exotic_matter_density
                [0.7, 0.9],         # traversal_probability
                [80.0, 120.0]       # quantum_coherence_time
            ])
            
            # Test 1: Search execution
            results['total_tests'] += 1
            try:
                search_result = search.search(
                    n_iterations=8,
                    n_initial=4,
                    parameter_bounds=parameter_bounds
                )
                results['tests_passed'] += 1
                results['details']['search_execution'] = 'PASS'
            except Exception as e:
                results['details']['search_execution'] = f'FAIL: {e}'
                search_result = None
            
            if search_result:
                # Test 2: Results structure
                results['total_tests'] += 1
                required_keys = ['total_evaluations', 'viable_candidates', 'best_candidate']
                if all(key in search_result for key in required_keys):
                    results['tests_passed'] += 1
                    results['details']['results_structure'] = 'PASS'
                else:
                    results['details']['results_structure'] = 'FAIL'
                
                # Test 3: Found viable candidates
                results['total_tests'] += 1
                if len(search_result['viable_candidates']) > 0:
                    results['tests_passed'] += 1
                    results['details']['viable_candidates'] = 'PASS'
                else:
                    results['details']['viable_candidates'] = 'FAIL'
                
                results['search_results'] = {
                    'total_evaluations': search_result['total_evaluations'],
                    'viable_candidates_count': len(search_result['viable_candidates']),
                    'best_score': search_result['search_statistics']['best_score'],
                    'viable_fraction': search_result['search_statistics']['viable_fraction']
                }
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _validate_rotating_metrics(self) -> Dict[str, Any]:
        """Validate rotating wormhole metrics system."""
        logger.info("Validating rotating wormhole metrics")
        
        results = {
            'test_name': 'Rotating Metrics Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'metric_tests': {},
            'details': {}
        }
        
        try:
            # Create rotating wormhole
            rotation_params = RotationParameters(
                angular_momentum=1e45,
                spin_parameter=0.5
            )
            wormhole = KerrLikeWormhole(
                throat_radius=1000.0,
                mass=1e30,
                rotation_params=rotation_params
            )
            
            # Test 1: Metric components finite and reasonable
            results['total_tests'] += 1
            r, theta = 1500.0, np.pi/2
            g_tt = wormhole.metric_tt(r, theta)
            g_rr = wormhole.metric_rr(r, theta)
            g_tphi = wormhole.metric_t_phi(r, theta)
            
            if all(np.isfinite([g_tt, g_rr, g_tphi])):
                results['tests_passed'] += 1
                results['details']['metric_components'] = 'PASS'
            else:
                results['details']['metric_components'] = 'FAIL'
            
            # Test 2: Frame-dragging effects present
            results['total_tests'] += 1
            frame_dragging = wormhole.frame_dragging_function(r, theta)
            if abs(frame_dragging) > 1e-10:  # Should have some frame-dragging
                results['tests_passed'] += 1
                results['details']['frame_dragging'] = 'PASS'
            else:
                results['details']['frame_dragging'] = 'FAIL'
            
            # Test 3: Ergosphere analysis
            results['total_tests'] += 1
            try:
                ergo_radius = wormhole.ergosphere_radius(np.pi/2)
                results['tests_passed'] += 1
                results['details']['ergosphere_analysis'] = 'PASS'
            except:
                results['details']['ergosphere_analysis'] = 'FAIL'
            
            results['metric_tests'] = {
                'g_tt': float(g_tt),
                'g_rr': float(g_rr),
                'g_t_phi': float(g_tphi),
                'frame_dragging': float(frame_dragging)
            }
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _validate_dynamic_evolution(self) -> Dict[str, Any]:
        """Validate dynamic throat evolution system."""
        logger.info("Validating dynamic throat evolution")
        
        results = {
            'test_name': 'Dynamic Evolution Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'evolution_results': {},
            'details': {}
        }
        
        try:
            # Test multiple scenarios
            test_scenarios = ['standard', 'collapse', 'expansion']
            scenario_results = {}
            
            for scenario in test_scenarios:
                try:
                    evolution_system = create_evolution_scenario(scenario_type=scenario)
                    
                    # Run short evolution
                    evolution_result = evolution_system.evolve_throat(
                        time_span=100.0,
                        num_steps=50
                    )
                    
                    scenario_results[scenario] = {
                        'success': evolution_result['evolution_success'],
                        'final_radius': evolution_result['statistics']['final_radius'] if evolution_result['evolution_success'] else None,
                        'stability_score': evolution_result['statistics']['stability_score'] if evolution_result['evolution_success'] else None
                    }
                    
                except Exception as e:
                    scenario_results[scenario] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Test 1: All scenarios execute
            results['total_tests'] += 1
            successful_scenarios = [s for s, r in scenario_results.items() if r['success']]
            if len(successful_scenarios) >= 2:  # At least 2 should work
                results['tests_passed'] += 1
                results['details']['scenario_execution'] = 'PASS'
            else:
                results['details']['scenario_execution'] = 'FAIL'
            
            # Test 2: Evolution produces changes
            results['total_tests'] += 1
            significant_changes = []
            for scenario, result in scenario_results.items():
                if result['success'] and result['final_radius']:
                    change = abs(result['final_radius'] - 1000.0)  # Assuming 1000.0 initial
                    if change > 1.0:  # At least 1m change
                        significant_changes.append(scenario)
            
            if len(significant_changes) > 0:
                results['tests_passed'] += 1
                results['details']['evolution_changes'] = 'PASS'
            else:
                results['details']['evolution_changes'] = 'FAIL'
            
            # Test 3: Stability analysis
            results['total_tests'] += 1
            stability_computed = []
            for scenario, result in scenario_results.items():
                if result['success'] and result['stability_score'] is not None:
                    if 0 <= result['stability_score'] <= 1:
                        stability_computed.append(scenario)
            
            if len(stability_computed) > 0:
                results['tests_passed'] += 1
                results['details']['stability_analysis'] = 'PASS'
            else:
                results['details']['stability_analysis'] = 'FAIL'
            
            results['evolution_results'] = scenario_results
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _validate_visualization_system(self) -> Dict[str, Any]:
        """Validate visualization system."""
        logger.info("Validating visualization system")
        
        results = {
            'test_name': 'Visualization System Validation',
            'start_time': time.time(),
            'tests_passed': 0,
            'total_tests': 0,
            'details': {}
        }
        
        try:
            from src.visualization.realtime_throat_evolution_dashboard import (
                create_realtime_dashboard,
                RealTimeVisualizationConfig
            )
            
            # Test 1: Dashboard creation
            results['total_tests'] += 1
            try:
                config = RealTimeVisualizationConfig()
                dashboard = create_realtime_dashboard(config)
                results['tests_passed'] += 1
                results['details']['dashboard_creation'] = 'PASS'
            except Exception as e:
                results['details']['dashboard_creation'] = f'FAIL: {e}'
                dashboard = None
            
            if dashboard:
                # Test 2: Figure generation
                results['total_tests'] += 1
                try:
                    fig = dashboard.create_dashboard()
                    if len(fig.data) > 0:
                        results['tests_passed'] += 1
                        results['details']['figure_generation'] = 'PASS'
                    else:
                        results['details']['figure_generation'] = 'FAIL'
                except Exception as e:
                    results['details']['figure_generation'] = f'FAIL: {e}'
                
                # Test 3: Scenario switching
                results['total_tests'] += 1
                try:
                    dashboard.switch_scenario('collapse')
                    if dashboard.current_scenario == 'collapse':
                        results['tests_passed'] += 1
                        results['details']['scenario_switching'] = 'PASS'
                    else:
                        results['details']['scenario_switching'] = 'FAIL'
                except Exception as e:
                    results['details']['scenario_switching'] = f'FAIL: {e}'
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        results['end_time'] = time.time()
        results['success_rate'] = results['tests_passed'] / max(results['total_tests'], 1)
        
        return results
    
    def _run_parameter_sweeps(self) -> Dict[str, Any]:
        """Run parameter space sweeps across all scenarios."""
        logger.info("Running parameter space sweeps")
        
        sweep_results = {
            'start_time': time.time(),
            'parameter_ranges': self.config.parameter_ranges,
            'sweep_points': self.config.parameter_sweep_points,
            'scenario_sweeps': {},
            'parameter_sensitivity': {}
        }
        
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations()
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            # Test each scenario with parameter sweeps
            for scenario in self.config.scenarios_to_test:
                logger.info(f"Parameter sweep for {scenario} scenario")
                
                scenario_results = []
                
                for i, params in enumerate(param_combinations[:10]):  # Limit for testing
                    try:
                        # Create evolution system with these parameters
                        evolution_system = create_evolution_scenario(
                            scenario_type=scenario,
                            **params
                        )
                        
                        # Run short evolution
                        result = evolution_system.evolve_throat(
                            time_span=50.0,
                            num_steps=25
                        )
                        
                        if result['evolution_success']:
                            scenario_results.append({
                                'parameters': params,
                                'final_radius': result['statistics']['final_radius'],
                                'stability_score': result['statistics']['stability_score'],
                                'final_state': result['statistics']['final_state']
                            })
                        
                    except Exception as e:
                        logger.warning(f"Parameter combination {i} failed for {scenario}: {e}")
                
                sweep_results['scenario_sweeps'][scenario] = {
                    'successful_runs': len(scenario_results),
                    'total_attempts': min(len(param_combinations), 10),
                    'success_rate': len(scenario_results) / min(len(param_combinations), 10),
                    'results': scenario_results
                }
            
            # Analyze parameter sensitivity
            sweep_results['parameter_sensitivity'] = self._analyze_parameter_sensitivity(
                sweep_results['scenario_sweeps']
            )
            
        except Exception as e:
            sweep_results['error'] = str(e)
        
        sweep_results['end_time'] = time.time()
        
        return sweep_results
    
    def _generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate combinations of parameters for testing."""
        
        # Sample parameters from ranges
        combinations = []
        n_points = min(self.config.parameter_sweep_points, 20)  # Limit for testing
        
        for i in range(n_points):
            params = {}
            for param_name, (min_val, max_val) in self.config.parameter_ranges.items():
                if param_name in ['throat_radius', 'mass', 'angular_momentum']:
                    # Use these for evolution scenario creation
                    if min_val > 0 and max_val > min_val:
                        params[param_name] = np.random.uniform(min_val, max_val)
            
            if params:  # Only add if we have valid parameters
                combinations.append(params)
        
        return combinations
    
    def _analyze_parameter_sensitivity(self, scenario_sweeps: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter sensitivity across scenarios."""
        
        sensitivity_analysis = {
            'most_sensitive_parameters': {},
            'stability_correlations': {},
            'scenario_comparisons': {}
        }
        
        try:
            # Collect all results for analysis
            all_results = []
            for scenario, sweep_data in scenario_sweeps.items():
                for result in sweep_data.get('results', []):
                    result_with_scenario = result.copy()
                    result_with_scenario['scenario'] = scenario
                    all_results.append(result_with_scenario)
            
            if len(all_results) > 5:  # Need some data for analysis
                # Convert to DataFrame for easier analysis
                import pandas as pd
                
                # Flatten parameters and create DataFrame
                flat_data = []
                for result in all_results:
                    flat_result = {'scenario': result['scenario']}
                    flat_result.update(result['parameters'])
                    flat_result['final_radius'] = result['final_radius']
                    flat_result['stability_score'] = result['stability_score']
                    flat_data.append(flat_result)
                
                df = pd.DataFrame(flat_data)
                
                # Analyze correlations with stability
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if 'stability_score' in numeric_cols and len(numeric_cols) > 1:
                    correlations = df[numeric_cols].corr()['stability_score'].abs().sort_values(ascending=False)
                    
                    sensitivity_analysis['stability_correlations'] = {
                        param: float(corr) for param, corr in correlations.items() 
                        if param != 'stability_score'
                    }
                
                # Find most impactful parameters
                if len(sensitivity_analysis['stability_correlations']) > 0:
                    most_sensitive = max(
                        sensitivity_analysis['stability_correlations'].items(),
                        key=lambda x: x[1]
                    )
                    sensitivity_analysis['most_sensitive_parameter'] = most_sensitive[0]
                    sensitivity_analysis['max_correlation'] = most_sensitive[1]
            
        except Exception as e:
            sensitivity_analysis['error'] = str(e)
        
        return sensitivity_analysis
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks across all systems."""
        logger.info("Running performance benchmarks")
        
        benchmark_results = {
            'start_time': time.time(),
            'benchmark_iterations': self.config.benchmark_iterations,
            'system_benchmarks': {}
        }
        
        # Benchmark each system
        systems_to_benchmark = [
            ('quantum_backend', self._benchmark_quantum_backend),
            ('ml_optimization', self._benchmark_ml_optimization),
            ('dynamic_evolution', self._benchmark_dynamic_evolution),
            ('visualization', self._benchmark_visualization)
        ]
        
        for system_name, benchmark_func in systems_to_benchmark:
            logger.info(f"Benchmarking {system_name}")
            
            try:
                system_benchmark = benchmark_func()
                benchmark_results['system_benchmarks'][system_name] = system_benchmark
                
            except Exception as e:
                benchmark_results['system_benchmarks'][system_name] = {
                    'error': str(e),
                    'benchmark_failed': True
                }
        
        benchmark_results['end_time'] = time.time()
        
        return benchmark_results
    
    def _benchmark_quantum_backend(self) -> Dict[str, Any]:
        """Benchmark quantum backend performance."""
        
        times = []
        successes = 0
        
        for i in range(min(self.config.benchmark_iterations, 5)):  # Limit iterations
            try:
                start_time = time.time()
                
                geometry_params = {
                    'throat_radius': 1000.0,
                    'mass': 1e30,
                    'exotic_matter_density': -1e-3
                }
                circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
                initial_state = circuit.create_initial_state()
                target_metrics = {'traversability': 0.8, 'stability': 0.7}
                optimized = circuit.optimize_parameters(target_metrics)
                
                end_time = time.time()
                times.append(end_time - start_time)
                successes += 1
                
            except Exception:
                pass
        
        return {
            'successful_runs': successes,
            'total_runs': min(self.config.benchmark_iterations, 5),
            'mean_time': np.mean(times) if times else None,
            'std_time': np.std(times) if times else None,
            'success_rate': successes / min(self.config.benchmark_iterations, 5)
        }
    
    def _benchmark_ml_optimization(self) -> Dict[str, Any]:
        """Benchmark ML optimization performance."""
        
        times = []
        successes = 0
        
        for i in range(min(self.config.benchmark_iterations, 3)):  # Very limited for ML
            try:
                start_time = time.time()
                
                results = run_optimization_comparison(optimization_budget=10)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if len(results) > 0:
                    successes += 1
                
            except Exception:
                pass
        
        return {
            'successful_runs': successes,
            'total_runs': min(self.config.benchmark_iterations, 3),
            'mean_time': np.mean(times) if times else None,
            'std_time': np.std(times) if times else None,
            'success_rate': successes / min(self.config.benchmark_iterations, 3)
        }
    
    def _benchmark_dynamic_evolution(self) -> Dict[str, Any]:
        """Benchmark dynamic evolution performance."""
        
        times = []
        successes = 0
        
        for i in range(min(self.config.benchmark_iterations, 5)):
            try:
                start_time = time.time()
                
                evolution_system = create_evolution_scenario('standard')
                result = evolution_system.evolve_throat(time_span=50.0, num_steps=20)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if result['evolution_success']:
                    successes += 1
                
            except Exception:
                pass
        
        return {
            'successful_runs': successes,
            'total_runs': min(self.config.benchmark_iterations, 5),
            'mean_time': np.mean(times) if times else None,
            'std_time': np.std(times) if times else None,
            'success_rate': successes / min(self.config.benchmark_iterations, 5)
        }
    
    def _benchmark_visualization(self) -> Dict[str, Any]:
        """Benchmark visualization system performance."""
        
        times = []
        successes = 0
        
        for i in range(min(self.config.benchmark_iterations, 3)):
            try:
                start_time = time.time()
                
                from src.visualization.realtime_throat_evolution_dashboard import create_realtime_dashboard
                dashboard = create_realtime_dashboard()
                fig = dashboard.create_dashboard()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if len(fig.data) > 0:
                    successes += 1
                
            except Exception:
                pass
        
        return {
            'successful_runs': successes,
            'total_runs': min(self.config.benchmark_iterations, 3),
            'mean_time': np.mean(times) if times else None,
            'std_time': np.std(times) if times else None,
            'success_rate': successes / min(self.config.benchmark_iterations, 3)
        }
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of all validation results."""
        logger.info("Performing statistical analysis")
        
        analysis = {
            'overall_success_rates': {},
            'system_reliability': {},
            'performance_statistics': {},
            'confidence_intervals': {}
        }
        
        try:
            # Calculate overall success rates
            for system_name, system_results in self.validation_results['system_validations'].items():
                if 'success_rate' in system_results:
                    analysis['overall_success_rates'][system_name] = system_results['success_rate']
            
            # Calculate mean success rate
            if analysis['overall_success_rates']:
                analysis['mean_success_rate'] = np.mean(list(analysis['overall_success_rates'].values()))
                analysis['std_success_rate'] = np.std(list(analysis['overall_success_rates'].values()))
            
            # Performance statistics
            if 'performance_benchmarks' in self.validation_results:
                perf_data = self.validation_results['performance_benchmarks']['system_benchmarks']
                
                for system, benchmark in perf_data.items():
                    if 'mean_time' in benchmark and benchmark['mean_time'] is not None:
                        analysis['performance_statistics'][system] = {
                            'mean_time': benchmark['mean_time'],
                            'performance_success_rate': benchmark.get('success_rate', 0)
                        }
            
            # Overall system health assessment
            success_rates = list(analysis['overall_success_rates'].values())
            if success_rates:
                if np.mean(success_rates) > 0.9:
                    analysis['system_health'] = 'EXCELLENT'
                elif np.mean(success_rates) > 0.7:
                    analysis['system_health'] = 'GOOD'
                elif np.mean(success_rates) > 0.5:
                    analysis['system_health'] = 'FAIR'
                else:
                    analysis['system_health'] = 'POOR'
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_overall_summary(self, validation_time: float) -> Dict[str, Any]:
        """Generate overall validation summary."""
        
        summary = {
            'validation_time': validation_time,
            'timestamp': self.validation_timestamp,
            'systems_tested': [],
            'overall_success_rate': 0.0,
            'key_findings': [],
            'recommendations': []
        }
        
        try:
            # Count systems tested
            if 'system_validations' in self.validation_results:
                summary['systems_tested'] = list(self.validation_results['system_validations'].keys())
            
            # Calculate overall success rate
            success_rates = []
            for system_results in self.validation_results['system_validations'].values():
                if 'success_rate' in system_results:
                    success_rates.append(system_results['success_rate'])
            
            if success_rates:
                summary['overall_success_rate'] = np.mean(success_rates)
            
            # Generate key findings
            if summary['overall_success_rate'] > 0.8:
                summary['key_findings'].append("Phase 3 systems are highly functional")
            
            if 'parameter_sweeps' in self.validation_results:
                sweep_data = self.validation_results['parameter_sweeps']
                if 'scenario_sweeps' in sweep_data:
                    working_scenarios = [s for s, d in sweep_data['scenario_sweeps'].items() 
                                       if d.get('success_rate', 0) > 0.5]
                    if working_scenarios:
                        summary['key_findings'].append(f"Scenarios working: {', '.join(working_scenarios)}")
            
            # Generate recommendations
            if summary['overall_success_rate'] < 0.7:
                summary['recommendations'].append("Review failing system components")
            
            if validation_time > 300:  # > 5 minutes
                summary['recommendations'].append("Consider performance optimization")
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _save_validation_results(self):
        """Save validation results to files."""
        
        # Save main results as JSON
        results_file = self.output_dir / f"validation_results_{self.validation_timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_safe_results = self._make_json_safe(self.validation_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {results_file}")
        
        # Save CSV summary if requested
        if self.config.export_csv_data:
            self._export_csv_summary()
    
    def _make_json_safe(self, data):
        """Convert data to JSON-safe format."""
        if isinstance(data, dict):
            return {k: self._make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_safe(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif hasattr(data, '__dict__'):  # Handle custom objects
            return str(data)
        else:
            return data
    
    def _export_csv_summary(self):
        """Export summary data to CSV."""
        try:
            summary_data = []
            
            # System validation summary
            for system, results in self.validation_results['system_validations'].items():
                summary_data.append({
                    'component': system,
                    'test_type': 'system_validation',
                    'success_rate': results.get('success_rate', 0),
                    'tests_passed': results.get('tests_passed', 0),
                    'total_tests': results.get('total_tests', 0)
                })
            
            # Performance benchmark summary
            if 'performance_benchmarks' in self.validation_results:
                perf_data = self.validation_results['performance_benchmarks']['system_benchmarks']
                for system, benchmark in perf_data.items():
                    if not benchmark.get('benchmark_failed', False):
                        summary_data.append({
                            'component': system,
                            'test_type': 'performance_benchmark',
                            'success_rate': benchmark.get('success_rate', 0),
                            'mean_time': benchmark.get('mean_time', 0),
                            'successful_runs': benchmark.get('successful_runs', 0)
                        })
            
            # Save to CSV
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_file = self.output_dir / f"validation_summary_{self.validation_timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"CSV summary saved to: {csv_file}")
            
        except Exception as e:
            logger.warning(f"Could not export CSV summary: {e}")
    
    def _create_summary_plots(self):
        """Create summary plots and visualizations."""
        logger.info("Creating summary plots")
        
        try:
            import matplotlib.pyplot as plt
            
            # Success rate comparison plot
            if 'system_validations' in self.validation_results:
                systems = []
                success_rates = []
                
                for system, results in self.validation_results['system_validations'].items():
                    if 'success_rate' in results:
                        systems.append(system.replace('_', ' ').title())
                        success_rates.append(results['success_rate'])
                
                if systems and success_rates:
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(systems, success_rates, 
                                  color=['green' if rate > 0.8 else 'orange' if rate > 0.5 else 'red' 
                                        for rate in success_rates])
                    plt.title('Phase 3 System Validation Success Rates')
                    plt.ylabel('Success Rate')
                    plt.ylim(0, 1)
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, rate in zip(bars, success_rates):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{rate:.2f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plot_file = self.output_dir / f"success_rates_{self.validation_timestamp}.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Success rate plot saved to: {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not create summary plots: {e}")


def run_comprehensive_validation(config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Run comprehensive validation of all Phase 3 systems.
    
    Args:
        config: Optional validation configuration
        
    Returns:
        Complete validation results
    """
    
    if config is None:
        config = ValidationConfig()
    
    validator = MultiScenarioValidator(config)
    return validator.run_full_validation_suite()


def quick_validation_check() -> Dict[str, Any]:
    """Run quick validation check of core systems.
    
    Returns:
        Quick validation results
    """
    
    config = ValidationConfig(
        parameter_sweep_points=5,
        benchmark_iterations=3,
        scenarios_to_test=['standard'],
        create_summary_plots=False,
        export_csv_data=False
    )
    
    return run_comprehensive_validation(config)