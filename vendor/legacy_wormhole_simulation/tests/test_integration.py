"""
Comprehensive integration tests for the unified wormhole simulation framework.

This module provides extensive testing of the integrated system including:
- Component initialization and interaction
- Simulation execution and convergence
- Cross-system consistency checks
- Performance and memory testing  
- Error handling and recovery
- Results validation and analysis
"""

import pytest
import numpy as np
import time
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the integration framework
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import (
    WormholeSimulationFramework,
    IntegrationConfig,
    SimulationResults,
    create_default_simulation,
    run_quick_demo
)


class TestIntegrationConfig:
    """Test the integration configuration system."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = IntegrationConfig()
        
        assert config.simulation_name == "quantum_wormhole_simulation"
        assert config.time_steps == 1000
        assert config.dt == 0.1
        assert config.num_qubits == 8
        assert config.enable_real_time_visualization == True
    
    def test_custom_config_creation(self):
        """Test creation of custom configuration."""
        config = IntegrationConfig(
            simulation_name="test_simulation",
            time_steps=500,
            dt=0.2,
            num_qubits=4
        )
        
        assert config.simulation_name == "test_simulation"
        assert config.time_steps == 500
        assert config.dt == 0.2
        assert config.num_qubits == 4
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration should not raise errors
        config = IntegrationConfig(time_steps=100, dt=0.1)
        assert config.time_steps == 100
        
        # Test boundary values
        config = IntegrationConfig(num_qubits=2)
        assert config.num_qubits == 2


class TestSimulationResults:
    """Test the simulation results container."""
    
    def test_results_initialization(self):
        """Test results container initialization."""
        results = SimulationResults()
        
        assert isinstance(results.spacetime_evolution, list)
        assert isinstance(results.quantum_state_evolution, list)
        assert isinstance(results.stability_predictions, list)
        assert isinstance(results.field_strengths, dict)
        assert results.timestamp is not None
    
    def test_results_data_storage(self):
        """Test storing data in results container."""
        results = SimulationResults()
        
        # Add some test data
        results.spacetime_evolution.append({'step': 0, 'energy': -1e15})
        results.stability_predictions.append(0.75)
        results.field_strengths['electromagnetic'] = [1e8, 1e7, 1e6]
        
        assert len(results.spacetime_evolution) == 1
        assert results.spacetime_evolution[0]['step'] == 0
        assert len(results.stability_predictions) == 1
        assert 'electromagnetic' in results.field_strengths


class TestWormholeSimulationFramework:
    """Test the main simulation framework."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.config = IntegrationConfig(
            simulation_name="test_simulation",
            time_steps=10,  # Small for testing
            dt=0.1,
            num_qubits=4,
            enable_real_time_visualization=False
        )
        self.framework = WormholeSimulationFramework(self.config)
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.config.simulation_name == "test_simulation"
        assert self.framework.is_initialized == False
        assert self.framework.is_running == False
        assert self.framework.current_step == 0
    
    def test_system_initialization(self):
        """Test initialization of all subsystems."""
        # Test successful initialization
        self.framework.initialize_system()
        
        assert self.framework.is_initialized == True
        assert self.framework.physics_engine is not None
        assert self.framework.quantum_system is not None
        assert self.framework.ai_system is not None
        assert self.framework.visualization_system is not None
        
        # Check physics engine components
        assert 'metric' in self.framework.physics_engine
        assert 'exotic_matter' in self.framework.physics_engine
        assert 'stress_energy' in self.framework.physics_engine
        
        # Check quantum system components
        assert 'circuit' in self.framework.quantum_system
        assert 'entanglement' in self.framework.quantum_system
        assert 'vacuum' in self.framework.quantum_system
        
        # Check AI system components
        assert 'stability' in self.framework.ai_system
        assert 'quantum_ml' in self.framework.ai_system
        
        # Check visualization components
        assert 'spacetime' in self.framework.visualization_system
        assert 'quantum' in self.framework.visualization_system
    
    def test_system_initialization_with_custom_params(self):
        """Test system initialization with custom parameters."""
        wormhole_params = {'b0': 5e3, 'mass': 2e30}
        quantum_params = {'num_qubits': 6, 'traversal_probability': 0.9}
        
        self.framework.initialize_system(
            wormhole_params=wormhole_params,
            quantum_params=quantum_params
        )
        
        assert self.framework.is_initialized == True
        
        # Verify custom parameters were applied
        metric = self.framework.physics_engine['metric']
        assert metric.b0 == 5e3
        assert metric.mass == 2e30
    
    def test_simulation_execution(self):
        """Test basic simulation execution."""
        self.framework.initialize_system()
        
        # Run short simulation
        results = self.framework.run_simulation(duration=1.0)  # 1 second
        
        assert isinstance(results, SimulationResults)
        assert len(results.spacetime_evolution) > 0
        assert len(results.quantum_state_evolution) > 0
        assert not self.framework.is_running
    
    def test_simulation_step_execution(self):
        """Test individual simulation step execution."""
        self.framework.initialize_system()
        
        # Run single step
        step_results = self.framework._run_simulation_step(0)
        
        assert 'time' in step_results
        assert 'step' in step_results
        assert 'physics' in step_results
        assert 'quantum' in step_results
        assert 'ai' in step_results
        
        # Check physics results structure
        physics_results = step_results['physics']
        assert isinstance(physics_results, dict)
        
        # Check quantum results structure  
        quantum_results = step_results['quantum']
        assert isinstance(quantum_results, dict)
        
        # Check AI results structure
        ai_results = step_results['ai']
        assert isinstance(ai_results, dict)
    
    def test_physics_evolution(self):
        """Test physics system evolution."""
        self.framework.initialize_system()
        
        physics_results = self.framework._evolve_physics(0, 0.0)
        
        assert isinstance(physics_results, dict)
        
        # Should have basic physics quantities
        expected_keys = ['energy_density', 'pressure']
        for key in expected_keys:
            if key in physics_results:
                assert isinstance(physics_results[key], (int, float))
    
    def test_quantum_evolution(self):
        """Test quantum system evolution."""
        self.framework.initialize_system()
        
        # Mock physics results for quantum evolution
        physics_results = {'metric_determinant': 1.0}
        
        quantum_results = self.framework._evolve_quantum(0, 0.0, physics_results)
        
        assert isinstance(quantum_results, dict)
        
        # Should have quantum measures
        quantum_keys = ['concurrence', 'negativity', 'entropy']
        for key in quantum_keys:
            if key in quantum_results:
                assert isinstance(quantum_results[key], (int, float))
                assert 0 <= quantum_results[key] <= 1  # Valid quantum measures
    
    def test_ai_analysis(self):
        """Test AI analysis system."""
        self.framework.initialize_system()
        
        # Mock input results
        physics_results = {
            'energy_density': -1e15,
            'pressure': -1e14,
            'metric_determinant': 0.8
        }
        
        quantum_results = {
            'concurrence': 0.5,
            'negativity': 0.3,
            'entropy': 0.7
        }
        
        ai_results = self.framework._run_ai_analysis(0, physics_results, quantum_results)
        
        assert isinstance(ai_results, dict)
        
        # Should have AI analysis results
        if 'stability_score' in ai_results:
            assert 0 <= ai_results['stability_score'] <= 1
        
        if 'anomaly_score' in ai_results:
            assert isinstance(ai_results['anomaly_score'], (int, float))
    
    def test_feature_extraction(self):
        """Test feature extraction from simulation results."""
        physics_results = {
            'energy_density': -1e15,
            'pressure': -1e14,
            'metric_determinant': 0.8,
            'invalid_feature': 'not_a_number'
        }
        
        quantum_results = {
            'concurrence': 0.5,
            'entropy': 0.7,
            'another_invalid': None
        }
        
        features = self.framework._extract_features(physics_results, quantum_results)
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # All features should be numeric
        for feature in features:
            assert isinstance(feature, (int, float))
            assert not np.isnan(feature)
    
    def test_results_storage(self):
        """Test storage of simulation results."""
        self.framework.initialize_system()
        
        # Create mock step results
        step_results = {
            'time': 0.1,
            'physics': {'energy_density': -1e15, 'pressure': -1e14},
            'quantum': {'concurrence': 0.5, 'entropy': 0.3},
            'ai': {'stability_score': 0.7, 'anomaly_score': 1.2}
        }
        
        self.framework._store_step_results(0, step_results)
        
        # Check that results were stored
        assert len(self.framework.results.spacetime_evolution) == 1
        assert len(self.framework.results.quantum_state_evolution) == 1
        assert len(self.framework.results.stability_predictions) == 1
        assert len(self.framework.results.anomaly_scores) == 1
        
        # Verify data integrity
        physics_data = self.framework.results.spacetime_evolution[0]
        assert physics_data['step'] == 0
        assert physics_data['time'] == 0.1
        assert physics_data['energy_density'] == -1e15
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        self.framework.initialize_system()
        
        # Fill results with test data
        for i in range(1500):  # More than max_cache_size (1000)
            self.framework.results.spacetime_evolution.append({'step': i})
            self.framework.results.quantum_state_evolution.append({'step': i})
        
        # Trigger memory cleanup
        self.framework._cleanup_memory()
        
        # Check that memory was cleaned up
        assert len(self.framework.results.spacetime_evolution) <= 1000
        assert len(self.framework.results.quantum_state_evolution) <= 1000
    
    def test_simulation_finalization(self):
        """Test simulation finalization."""
        self.framework.initialize_system()
        
        # Add some test data
        self.framework.results.stability_predictions = [0.8, 0.7, 0.9, 0.6]
        self.framework.results.quantum_state_evolution = [
            {'concurrence': 0.5}, {'concurrence': 0.3}, {'concurrence': 0.7}
        ]
        self.framework._step_times = [0.1, 0.15, 0.12, 0.14]
        
        self.framework._finalize_simulation()
        
        # Check that convergence metrics were computed
        metrics = self.framework.results.convergence_metrics
        
        assert 'avg_stability' in metrics
        assert 'stability_std' in metrics
        assert 'min_stability' in metrics
        assert 'max_stability' in metrics
        assert 'avg_entanglement' in metrics
        assert 'max_entanglement' in metrics
        assert 'avg_step_time' in metrics
        
        # Verify metric values
        assert metrics['avg_stability'] == 0.75
        assert metrics['min_stability'] == 0.6
        assert metrics['max_stability'] == 0.9
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        self.framework.initialize_system()
        
        # Add test data
        self.framework.results.spacetime_evolution = [
            {'energy_density': -1e15, 'pressure': -1e14, 'metric_determinant': 0.8},
            {'energy_density': -1.1e15, 'pressure': -1.1e14, 'metric_determinant': 0.9}
        ]
        
        self.framework.results.quantum_state_evolution = [
            {'concurrence': 0.5, 'entropy': 0.3},
            {'concurrence': 0.6, 'entropy': 0.4}
        ]
        
        self.framework.results.stability_predictions = [0.7, 0.8]
        self.framework.results.anomaly_scores = [1.0, 1.2]
        
        report = self.framework.generate_comprehensive_report()
        
        # Check report structure
        assert 'metadata' in report
        assert 'physics_analysis' in report
        assert 'quantum_analysis' in report
        assert 'ai_analysis' in report
        assert 'performance_analysis' in report
        assert 'stability_analysis' in report
        assert 'summary' in report
        assert 'recommendations' in report
        
        # Check metadata
        metadata = report['metadata']
        assert 'simulation_id' in metadata
        assert 'timestamp' in metadata
        assert 'config' in metadata
        assert 'total_steps' in metadata
        
        # Check physics analysis
        physics = report['physics_analysis']
        if 'energy_statistics' in physics:
            assert 'mean' in physics['energy_statistics']
            assert 'std' in physics['energy_statistics']
        
        # Check quantum analysis
        quantum = report['quantum_analysis']
        if 'entanglement_statistics' in quantum:
            assert 'max_concurrence' in quantum['entanglement_statistics']
        
        # Check recommendations
        recommendations = report['recommendations']
        assert isinstance(recommendations, list)
    
    def test_error_handling_during_simulation(self):
        """Test error handling during simulation execution."""
        self.framework.initialize_system()
        
        # Mock a component to raise an error
        original_evolve = self.framework._evolve_physics
        
        def mock_evolve_with_error(step, time):
            if step == 2:  # Fail on step 2
                raise ValueError("Simulated physics error")
            return original_evolve(step, time)
        
        self.framework._evolve_physics = mock_evolve_with_error
        
        # Should complete despite error (with error handling)
        results = self.framework.run_simulation(duration=0.5)
        
        assert isinstance(results, SimulationResults)
        # Should have some results before the error
        assert len(results.spacetime_evolution) >= 2
    
    def test_save_and_load_results(self):
        """Test saving and loading simulation results."""
        self.framework.initialize_system()
        
        # Add some test data
        self.framework.results.stability_predictions = [0.7, 0.8, 0.6]
        self.framework.results.spacetime_evolution = [
            {'step': 0, 'energy_density': -1e15},
            {'step': 1, 'energy_density': -1.1e15}
        ]
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            self.framework.save_results(temp_filename, 'json')
            assert os.path.exists(temp_filename)
            
            # Test loading
            new_framework = WormholeSimulationFramework()
            new_framework.load_results(temp_filename, 'json')
            
            # Verify loaded data
            assert len(new_framework.results.stability_predictions) == 3
            assert len(new_framework.results.spacetime_evolution) == 2
            assert new_framework.results.stability_predictions[0] == 0.7
            
        finally:
            # Cleanup
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestIntegrationPerformance:
    """Test performance characteristics of the integrated system."""
    
    def test_simulation_performance(self):
        """Test simulation performance metrics."""
        config = IntegrationConfig(
            time_steps=50,
            num_qubits=4,
            enable_real_time_visualization=False
        )
        
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        start_time = time.time()
        results = framework.run_simulation()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time
        assert execution_time < 30.0  # 30 seconds max for 50 steps
        
        # Should have performance metrics
        metrics = results.convergence_metrics
        assert 'avg_step_time' in metrics
        assert 'total_time' in metrics
        
        # Check performance consistency
        if 'step_time' in results.computation_times:
            step_times = results.computation_times['step_time']
            if len(step_times) > 1:
                # Step times shouldn't vary too much (within factor of 10)
                max_time = max(step_times)
                min_time = min(step_times)
                assert max_time / min_time < 10.0
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        config = IntegrationConfig(
            time_steps=100,
            num_qubits=4,
            enable_real_time_visualization=False
        )
        
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        # Run simulation and check memory doesn't grow unbounded
        initial_length = len(framework.results.spacetime_evolution)
        
        # Simulate many steps
        for step in range(200):
            step_results = framework._run_simulation_step(step)
            framework._store_step_results(step, step_results)
            
            # Trigger cleanup periodically
            if step % 100 == 0:
                framework._cleanup_memory()
        
        # Memory should be bounded
        final_length = len(framework.results.spacetime_evolution)
        assert final_length <= 1000  # Should not exceed cache limit
    
    def test_parallel_processing_capability(self):
        """Test that parallel processing can be enabled without errors."""
        config = IntegrationConfig(
            parallel_processing=True,
            max_workers=2,
            time_steps=10
        )
        
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        # Should initialize without errors
        assert framework.config.parallel_processing == True
        assert framework.config.max_workers == 2


class TestIntegrationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_extreme_parameters(self):
        """Test system behavior with extreme parameters."""
        # Very small wormhole
        extreme_config = IntegrationConfig(
            time_steps=10,
            num_qubits=2,
            enable_real_time_visualization=False
        )
        
        framework = WormholeSimulationFramework(extreme_config)
        
        # Should handle extreme parameters gracefully
        framework.initialize_system(
            wormhole_params={'b0': 1e-10, 'mass': 1e20}  # Very small, light wormhole
        )
        
        assert framework.is_initialized == True
        
        # Should be able to run simulation
        results = framework.run_simulation(duration=0.1)
        assert isinstance(results, SimulationResults)
    
    def test_zero_time_steps(self):
        """Test behavior with zero time steps."""
        config = IntegrationConfig(time_steps=0)
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        results = framework.run_simulation()
        
        # Should complete without error but with no evolution data
        assert len(results.spacetime_evolution) == 0
        assert len(results.quantum_state_evolution) == 0
    
    def test_single_qubit_system(self):
        """Test behavior with minimal quantum system."""
        config = IntegrationConfig(
            num_qubits=1,  # Minimal quantum system
            time_steps=5
        )
        
        framework = WormholeSimulationFramework(config)
        
        # Should handle single qubit system
        framework.initialize_system(quantum_params={'num_qubits': 1})
        
        assert framework.is_initialized == True
        
        results = framework.run_simulation()
        assert isinstance(results, SimulationResults)
    
    def test_disabled_subsystems(self):
        """Test behavior with some subsystems disabled."""
        config = IntegrationConfig(
            enable_stability_prediction=False,
            enable_parameter_optimization=False,
            enable_anomaly_detection=False,
            enable_reinforcement_learning=False,
            time_steps=5
        )
        
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        # Should initialize with reduced AI system
        ai_system = framework.ai_system
        assert 'stability' not in ai_system
        assert 'optimizer' not in ai_system
        assert 'anomaly' not in ai_system
        assert 'rl_agent' not in ai_system
        
        # But should still have quantum ML
        assert 'quantum_ml' in ai_system
        
        # Should still run simulation
        results = framework.run_simulation()
        assert isinstance(results, SimulationResults)


class TestIntegrationUtilities:
    """Test utility functions and helpers."""
    
    def test_create_default_simulation(self):
        """Test default simulation creation utility."""
        framework = create_default_simulation()
        
        assert isinstance(framework, WormholeSimulationFramework)
        assert framework.is_initialized == True
        assert framework.config.simulation_name == "default_wormhole_simulation"
        
        # Should have all systems initialized
        assert framework.physics_engine is not None
        assert framework.quantum_system is not None
        assert framework.ai_system is not None
        assert framework.visualization_system is not None
    
    def test_run_quick_demo(self):
        """Test quick demo functionality."""
        # This will actually run a short simulation
        results = run_quick_demo()
        
        assert isinstance(results, SimulationResults)
        assert len(results.spacetime_evolution) > 0
        assert len(results.quantum_state_evolution) > 0
        
        # Demo should complete successfully
        # (We can't guarantee exact metrics due to randomness, but should have some data)
    
    @patch('src.integration.logger')
    def test_logging_functionality(self, mock_logger):
        """Test that logging works correctly."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Should have logged initialization
        mock_logger.info.assert_called()
        
        # Check that log messages were appropriate
        log_calls = mock_logger.info.call_args_list
        log_messages = [call[0][0] for call in log_calls if call[0]]
        
        # Should have logged system initialization steps
        initialization_messages = [msg for msg in log_messages if 'initializ' in msg.lower()]
        assert len(initialization_messages) > 0


class TestIntegrationConsistency:
    """Test consistency between different system components."""
    
    def test_physics_quantum_consistency(self):
        """Test consistency between physics and quantum systems."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Run a few steps to get data
        for step in range(3):
            step_results = framework._run_simulation_step(step)
            framework._store_step_results(step, step_results)
        
        # Check that physics and quantum data are consistent
        spacetime_data = framework.results.spacetime_evolution
        quantum_data = framework.results.quantum_state_evolution
        
        assert len(spacetime_data) == len(quantum_data)
        
        for i, (phys, quantum) in enumerate(zip(spacetime_data, quantum_data)):
            # Time stamps should match
            assert phys['time'] == quantum['time']
            assert phys['step'] == quantum['step'] == i
    
    def test_ai_predictions_consistency(self):
        """Test consistency of AI predictions over time."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Run simulation to get predictions
        results = framework.run_simulation(duration=1.0)
        
        predictions = results.stability_predictions
        
        if len(predictions) > 1:
            # Predictions should be bounded
            for pred in predictions:
                assert 0 <= pred <= 1
            
            # Predictions shouldn't jump too wildly (continuity check)
            for i in range(1, len(predictions)):
                change = abs(predictions[i] - predictions[i-1])
                assert change < 0.5  # No jumps > 0.5 in stability
    
    def test_energy_conservation_checks(self):
        """Test energy conservation in the simulation."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Run simulation
        results = framework.run_simulation(duration=1.0)
        
        # Extract energy densities
        energy_densities = []
        for data in results.spacetime_evolution:
            if 'energy_density' in data:
                energy_densities.append(data['energy_density'])
        
        if len(energy_densities) > 1:
            # Energy should not change too rapidly (conservation-like check)
            energy_changes = [abs(energy_densities[i] - energy_densities[i-1]) 
                            for i in range(1, len(energy_densities))]
            
            # Most energy changes should be small
            small_changes = [change for change in energy_changes 
                           if change / abs(energy_densities[0]) < 0.1]
            
            # At least 70% of changes should be small
            assert len(small_changes) >= 0.7 * len(energy_changes)


class TestIntegrationValidation:
    """Validate the integrated simulation against known physics."""
    
    def test_exotic_matter_requirements(self):
        """Test that exotic matter requirements are satisfied."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Run simulation and check exotic matter properties
        results = framework.run_simulation(duration=0.5)
        
        # Check for negative energy density (exotic matter signature)
        negative_energy_found = False
        null_energy_violations = 0
        
        for data in results.spacetime_evolution:
            if 'energy_density' in data:
                energy = data['energy_density']
                pressure = data.get('pressure', 0)
                
                if energy < 0:
                    negative_energy_found = True
                
                # Check null energy condition
                if 'null_energy_condition' in data:
                    if not data['null_energy_condition']:
                        null_energy_violations += 1
        
        # Should have exotic matter (negative energy)
        assert negative_energy_found, "No exotic matter detected"
        
        # Should have some null energy condition violations (required for traversable wormholes)
        if len(results.spacetime_evolution) > 0:
            violation_rate = null_energy_violations / len(results.spacetime_evolution)
            assert violation_rate > 0.1, "Insufficient exotic matter for wormhole"
    
    def test_quantum_entanglement_evolution(self):
        """Test quantum entanglement evolution physics."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        results = framework.run_simulation(duration=1.0)
        
        # Extract entanglement measures
        entanglement_values = []
        for data in results.quantum_state_evolution:
            if 'concurrence' in data:
                entanglement_values.append(data['concurrence'])
        
        if len(entanglement_values) > 1:
            # Entanglement should start positive and may decay
            initial_entanglement = entanglement_values[0]
            final_entanglement = entanglement_values[-1]
            
            # Should have some initial entanglement
            assert initial_entanglement >= 0
            
            # Entanglement should be bounded
            for ent in entanglement_values:
                assert 0 <= ent <= 1
    
    def test_spacetime_metric_properties(self):
        """Test spacetime metric mathematical properties."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        metric = framework.physics_engine['metric']
        
        # Test metric at throat
        throat_coords = (0.0, metric.b0, np.pi/2, 0.0)
        g_throat = metric.metric_tensor(throat_coords)
        
        # Metric should be real and symmetric
        assert np.all(np.isreal(g_throat))
        assert np.allclose(g_throat, g_throat.T)
        
        # Determinant should be negative (signature -,+,+,+)
        det_g = np.linalg.det(g_throat)
        assert det_g < 0
        
        # Test metric away from throat
        far_coords = (0.0, metric.b0 * 10, np.pi/2, 0.0)
        g_far = metric.metric_tensor(far_coords)
        
        # Should also be well-behaved
        assert np.all(np.isfinite(g_far))
        assert np.linalg.det(g_far) < 0


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestIntegrationIO:
    """Test input/output operations of the integrated system."""
    
    def test_results_json_serialization(self, temp_output_dir):
        """Test JSON serialization of results."""
        framework = WormholeSimulationFramework()
        framework.initialize_system()
        
        # Run short simulation
        framework.run_simulation(duration=0.1)
        
        # Save results
        output_file = os.path.join(temp_output_dir, "test_results.json")
        framework.save_results(output_file, 'json')
        
        # Verify file was created and is valid JSON
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Check structure
        assert 'simulation_id' in loaded_data
        assert 'timestamp' in loaded_data
        assert 'config' in loaded_data
        assert 'spacetime_evolution' in loaded_data
        assert 'quantum_state_evolution' in loaded_data
    
    def test_configuration_persistence(self, temp_output_dir):
        """Test that configuration is properly saved and can be reproduced."""
        config = IntegrationConfig(
            simulation_name="test_config_persistence",
            time_steps=42,
            num_qubits=6,
            dt=0.05
        )
        
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        framework.run_simulation(duration=0.1)
        
        # Save results
        output_file = os.path.join(temp_output_dir, "config_test.json")
        framework.save_results(output_file, 'json')
        
        # Load and verify config
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        saved_config = data['config']
        assert saved_config['simulation_name'] == "test_config_persistence"
        assert saved_config['time_steps'] == 42
        assert saved_config['num_qubits'] == 6
        assert saved_config['dt'] == 0.05


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])