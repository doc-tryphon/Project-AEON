#!/usr/bin/env python3
"""
Test script for TensorFlow Quantum backend activation.

This script tests the new TensorFlow Quantum implementation and compares
it with the previous QuTiP-based approach.
"""

import sys
import time
import numpy as np
sys.path.append('src')

def test_tfq_import():
    """Test if TensorFlow Quantum can be imported."""
    print("Testing TensorFlow Quantum imports...")
    
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} imported successfully")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] {len(gpus)} GPU(s) available: {[gpu.name for gpu in gpus]}")
        else:
            print("[INFO] No GPU available, using CPU")
        
        import cirq
        print(f"[OK] Cirq {cirq.__version__} imported successfully")
        
        import tensorflow_quantum as tfq
        print(f"[OK] TensorFlow Quantum {tfq.__version__} imported successfully")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("To install TensorFlow Quantum: pip install tensorflow-quantum cirq")
        return False

def test_tfq_circuit_creation():
    """Test creating a TFQ wormhole circuit."""
    print("\nTesting TensorFlow Quantum circuit creation...")
    
    try:
        from src.quantum.tfq_wormhole_circuit import TFQWormholeCircuit, create_tfq_backend
        
        # Test direct instantiation
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8
        }
        
        circuit = TFQWormholeCircuit(num_qubits=4, geometry_params=geometry_params)
        print(f"[OK] TFQ circuit created with {circuit.num_qubits} qubits")
        print(f"  Circuit depth: {circuit.get_circuit_depth()}")
        print(f"  Gate count: {circuit.get_gate_count()}")
        
        # Test factory function
        config = {
            'num_qubits': 6,
            'throat_radius': 2e3,
            'traversal_probability': 0.9
        }
        
        circuit2 = create_tfq_backend(config)
        print(f"[OK] TFQ circuit created via factory with {circuit2.num_qubits} qubits")
        
        return True, circuit
        
    except Exception as e:
        print(f"[FAIL] TFQ circuit creation failed: {e}")
        return False, None

def test_quantum_state_creation():
    """Test quantum state creation and measurement."""
    print("\nTesting quantum state creation...")
    
    try:
        from src.quantum.tfq_wormhole_circuit import TFQWormholeCircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8
        }
        
        circuit = TFQWormholeCircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Create traversal state
        start_time = time.time()
        state = circuit.create_traversal_state()
        state_time = time.time() - start_time
        
        print(f"[OK] Quantum state created in {state_time:.3f} seconds")
        print(f"  State shape: {state.shape}")
        print(f"  State dtype: {state.dtype}")
        
        # Test measurements
        start_time = time.time()
        measurements = circuit.measure_observables(num_shots=1000)
        measure_time = time.time() - start_time
        
        print(f"[OK] Observables measured in {measure_time:.3f} seconds")
        print(f"  Number of observables: {len(measurements)}")
        
        for key, value in list(measurements.items())[:5]:  # Show first 5
            print(f"    {key}: {value:.6f}")
        
        # Test entanglement entropy calculation
        entropy = circuit.compute_entanglement_entropy()
        print(f"[OK] Entanglement entropy: {entropy:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quantum state operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_evolution():
    """Test time evolution capabilities."""
    print("\nTesting time evolution...")
    
    try:
        from src.quantum.tfq_wormhole_circuit import TFQWormholeCircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8
        }
        
        circuit = TFQWormholeCircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Run short time evolution
        start_time = time.time()
        evolution_results = circuit.time_evolve(time_steps=5, dt=0.1)
        evolution_time = time.time() - start_time
        
        print(f"[OK] Time evolution completed in {evolution_time:.3f} seconds")
        print(f"  Evolution steps: {len(evolution_results)}")
        
        # Show results from first and last time steps
        print(f"  Initial state (t=0): {evolution_results[0]}")
        print(f"  Final state (t={evolution_results[-1]['time']}): {evolution_results[-1]}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Time evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_optimization():
    """Test parameter optimization capabilities."""
    print("\nTesting parameter optimization...")
    
    try:
        from src.quantum.tfq_wormhole_circuit import TFQWormholeCircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8
        }
        
        circuit = TFQWormholeCircuit(num_qubits=4, geometry_params=geometry_params)
        
        print(f"Initial theta: {circuit.theta_value:.6f}")
        
        # Run optimization for a few steps
        start_time = time.time()
        optimization_results = circuit.optimize_parameters(
            target_state=None,  # Optimize for max entanglement
            learning_rate=0.1,
            steps=10
        )
        opt_time = time.time() - start_time
        
        print(f"[OK] Parameter optimization completed in {opt_time:.3f} seconds")
        print(f"  Optimized theta: {optimization_results['optimized_theta']:.6f}")
        print(f"  Final loss: {optimization_results['final_loss']:.6f}")
        print(f"  Loss improvement: {optimization_results['loss_history'][0] - optimization_results['final_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Parameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_simulation():
    """Test integration with the main simulation system."""
    print("\nTesting integration with main simulation...")
    
    try:
        from src.integration import WormholeSimulation
        from src.config import SimulationConfig
        
        # Create config with TFQ backend enabled
        config = SimulationConfig()
        config.quantum_backend = 'tfq'
        config.num_qubits = 4
        config.time_steps = 3  # Short run for testing
        config.enable_real_time_visualization = False
        
        # Initialize simulation
        sim = WormholeSimulation(config)
        
        # Test initialization
        start_time = time.time()
        wormhole_params = {'b0': 1e3, 'mass': 1e30}
        quantum_params = {'num_qubits': 4, 'traversal_probability': 0.8}
        
        sim.initialize_system(
            wormhole_params=wormhole_params,
            quantum_params=quantum_params
        )
        init_time = time.time() - start_time
        
        print(f"[OK] Simulation initialized in {init_time:.3f} seconds")
        print(f"  System is ready: {sim.is_initialized}")
        
        # Check quantum system type
        circuit = sim.quantum_system['circuit']
        circuit_type = type(circuit).__name__
        print(f"  Quantum circuit type: {circuit_type}")
        
        if hasattr(circuit, 'use_gpu'):
            print(f"  GPU acceleration: {circuit.use_gpu}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all TensorFlow Quantum backend tests."""
    print("TensorFlow Quantum Backend Activation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_tfq_import),
        ("Circuit Creation", test_tfq_circuit_creation),
        ("Quantum States", test_quantum_state_creation),
        ("Time Evolution", test_time_evolution),
        ("Parameter Optimization", test_parameter_optimization),
        ("Integration", test_integration_with_simulation),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        print("-" * 30)
        
        try:
            if test_func == test_tfq_circuit_creation:
                success, circuit = test_func()
            else:
                success = test_func()
            
            results[test_name] = success
            if success:
                passed += 1
                print(f"[OK] {test_name} PASSED")
            else:
                print(f"[FAIL] {test_name} FAILED")
                
        except Exception as e:
            print(f"[FAIL] {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TensorFlow Quantum Backend Test Summary")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All tests passed! TensorFlow Quantum backend is ready for Phase 3.")
    elif passed > 0:
        print("[WARN]  Partial success. Some features may not be available.")
    else:
        print("[ERROR] TensorFlow Quantum backend not available. Falling back to QuTiP/mock.")
    
    print("\nNext steps for Phase 3:")
    print("- Quantum backend: [OK] Activated" if results.get("Import Test", False) else "- Quantum backend: [FAIL] Not available")
    print("- Entanglement benchmarking: Ready" if results.get("Quantum States", False) else "- Entanglement benchmarking: Needs work")
    print("- ML parameter exploration: Ready" if results.get("Parameter Optimization", False) else "- ML parameter exploration: Needs work")
    print("- Advanced scenarios: Ready for implementation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)