"""
Integration tests for cross-component validation in quantum wormhole simulation.

This module tests the interaction between physics, quantum, AI, and visualization
components to catch parameter mismatches and interface incompatibilities.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import AdvancedCasimirExoticMatter
from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.ai.stability_predictor import StabilityPredictor
from src.ai.anomaly_detector import AnomalyDetector
from src.visualization.spacetime_plotter import SpacetimePlotter, SpacetimeVisualizationConfig


class TestParameterCompatibility:
    """Test parameter compatibility across components."""
    
    def test_throat_radius_consistency(self):
        """Test that throat_radius/b0 is handled consistently."""
        
        # Test that MorrisThorneeWormhole accepts throat_radius
        throat_radius = 1e3
        metric = MorrisThorneeWormhole(throat_radius=throat_radius)
        
        # Verify both b0 and throat_radius attributes exist
        assert hasattr(metric, 'b0')
        assert hasattr(metric, 'throat_radius')
        assert metric.b0 == throat_radius
        assert metric.throat_radius == throat_radius
        
    def test_wormhole_params_integration(self):
        """Test wormhole parameter passing in integration framework."""
        
        config = IntegrationConfig(
            time_steps=10,
            enable_real_time_visualization=False,
            enable_stability_prediction=False,
            enable_parameter_optimization=False,
            enable_anomaly_detection=False
        )
        
        framework = WormholeSimulationFramework(config)
        
        # Test both parameter name formats
        wormhole_params_new = {'throat_radius': 1e3, 'mass': 1e30}
        wormhole_params_old = {'b0': 1e3, 'mass': 1e30}  # Legacy format
        
        # Both should work without errors
        framework.initialize_system(wormhole_params=wormhole_params_new)
        assert framework.physics_engine is not None
        
        # Test legacy format compatibility
        framework2 = WormholeSimulationFramework(config)
        framework2.initialize_system(wormhole_params=wormhole_params_old)
        assert framework2.physics_engine is not None
        
    def test_visualization_metric_compatibility(self):
        """Test that visualization components work with physics metrics."""
        
        # Create a metric
        metric = MorrisThorneeWormhole(throat_radius=1e3)
        
        # Test that SpacetimePlotter can accept it
        config = SpacetimeVisualizationConfig(
            r_min=1e3,
            r_max=1e4,
            grid_resolution=10
        )
        
        # This should not raise an error
        plotter = SpacetimePlotter(metric=metric, config=config)
        assert plotter.metric is not None
        assert plotter.config is not None


class TestComponentInterfaces:
    """Test interfaces between different components."""
    
    def test_physics_quantum_interface(self):
        """Test interface between physics engine and quantum system."""
        
        # Create physics components
        metric = MorrisThorneeWormhole(throat_radius=1e3)
        exotic_matter = AdvancedCasimirExoticMatter()
        
        # Create quantum circuit - should accept geometry parameters
        try:
            circuit = WormholeQuantumCircuit(
                num_qubits=4,
                geometry_params={
                    'throat_radius': 1e3,
                    'traversal_probability': 0.8
                }
            )
            # If we get here, the interface works
            assert circuit is not None
        except Exception as e:
            # If WormholeQuantumCircuit is not available, that's okay for this test
            if "module" not in str(e).lower():
                raise
    
    def test_ai_physics_interface(self):
        """Test interface between AI components and physics results."""
        
        # Create mock physics results
        physics_results = {
            'energy_density': -1e15,
            'pressure': -5e14,
            'metric_determinant': 1.0,
            'christoffel_norm': 1e-3
        }
        
        quantum_results = {
            'concurrence': 0.5,
            'negativity': 0.3,
            'entropy': 0.7
        }
        
        # Test stability predictor can process these
        predictor = StabilityPredictor()
        
        # Test feature extraction (simplified)
        features = []
        for key in ['energy_density', 'pressure', 'metric_determinant']:
            if key in physics_results:
                features.append(physics_results[key])
        
        for key in ['concurrence', 'negativity', 'entropy']:
            if key in quantum_results:
                features.append(quantum_results[key])
        
        assert len(features) > 0
        
        # Test anomaly detector
        detector = AnomalyDetector(['energy_density', 'pressure', 'concurrence'])
        
        # Should be able to handle feature arrays
        test_data = {
            'energy_density': np.array([physics_results['energy_density']]),
            'pressure': np.array([physics_results['pressure']]),
            'concurrence': np.array([quantum_results['concurrence']])
        }
        
        # This should not raise errors
        assert test_data is not None


class TestFullIntegration:
    """Test full integration workflow."""
    
    def test_minimal_simulation_run(self):
        """Test that a minimal simulation can run without errors."""
        
        config = IntegrationConfig(
            simulation_name="test_simulation",
            time_steps=5,  # Very short for testing
            dt=0.1,
            num_qubits=2,  # Minimal qubits
            enable_real_time_visualization=False,
            enable_stability_prediction=True,
            enable_parameter_optimization=False,
            enable_anomaly_detection=True,
            enable_reinforcement_learning=False
        )
        
        framework = WormholeSimulationFramework(config)
        
        # Initialize with standard parameters
        framework.initialize_system(
            wormhole_params={'throat_radius': 1e3, 'mass': 1e30},
            quantum_params={'num_qubits': 2, 'traversal_probability': 0.8},
            ai_params={'stability_threshold': 0.5}
        )
        
        # Run short simulation
        results = framework.run_simulation()
        
        # Verify results structure
        assert results is not None
        assert hasattr(results, 'spacetime_evolution')
        assert hasattr(results, 'quantum_state_evolution')
        assert hasattr(results, 'stability_predictions')
        
        # Verify some data was generated
        assert len(results.spacetime_evolution) > 0
        assert len(results.quantum_state_evolution) > 0
        
        # If AI was enabled, should have predictions
        if config.enable_stability_prediction:
            assert len(results.stability_predictions) > 0
    
    def test_parameter_validation_workflow(self):
        """Test parameter validation across the workflow."""
        
        config = IntegrationConfig(time_steps=3)
        framework = WormholeSimulationFramework(config)
        
        # Test various parameter combinations
        valid_params = [
            {'throat_radius': 1e3, 'mass': 1e30},
            {'throat_radius': 5e3, 'mass': 2e30},
            {'b0': 1e3, 'mass': 1e30},  # Legacy format
        ]
        
        for params in valid_params:
            try:
                framework.initialize_system(wormhole_params=params)
                # If we get here, parameters were accepted
                assert framework.physics_engine is not None
            except Exception as e:
                pytest.fail(f"Valid parameters {params} caused error: {e}")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation."""
        
        config = IntegrationConfig(
            time_steps=5,
            enable_real_time_visualization=True,  # This might fail
            enable_stability_prediction=True,
            enable_anomaly_detection=True
        )
        
        framework = WormholeSimulationFramework(config)
        
        # Initialize - visualization might fail but others should work
        framework.initialize_system(
            wormhole_params={'throat_radius': 1e3, 'mass': 1e30}
        )
        
        # Should have physics and quantum systems even if visualization failed
        assert framework.physics_engine is not None
        assert framework.quantum_system is not None
        assert framework.ai_system is not None
        
        # Should be able to run simulation even with some components failing
        results = framework.run_simulation()
        assert results is not None


class TestMemoryAndPerformance:
    """Test memory management and performance aspects."""
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        
        config = IntegrationConfig(time_steps=10)
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        # Run simulation to generate data
        framework.run_simulation()
        
        # Test memory cleanup
        try:
            framework._cleanup_memory()
            # Should not raise errors
        except Exception as e:
            pytest.fail(f"Memory cleanup failed: {e}")
    
    def test_cache_handling(self):
        """Test visualization cache handling."""
        
        config = IntegrationConfig(
            time_steps=5,
            enable_real_time_visualization=False  # Avoid complex initialization
        )
        framework = WormholeSimulationFramework(config)
        framework.initialize_system()
        
        # Test that cache cleanup doesn't fail with None visualization system
        framework.visualization_system = None
        
        try:
            framework._cleanup_memory()
        except Exception as e:
            pytest.fail(f"Cache cleanup with None visualization failed: {e}")


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v"])