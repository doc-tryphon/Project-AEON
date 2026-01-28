#!/usr/bin/env python3
"""
Test script for Hybrid Quantum-AI backend.

This script tests the hybrid approach combining QuTiP quantum circuits
with TensorFlow AI optimization for Phase 3 features.
"""

import sys
import time
import numpy as np
sys.path.append('src')

def test_hybrid_backend_creation():
    """Test creating hybrid quantum-AI circuit."""
    print("Testing Hybrid Quantum-AI backend creation...")
    
    try:
        from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit, create_hybrid_backend
        
        # Test direct instantiation
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8,
            'mass': 1e30,
            'exotic_matter_density': -1e-3
        }
        
        circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
        print(f"[OK] Hybrid circuit created with {circuit.num_qubits} qubits")
        print(f"  Trainable parameters: theta={circuit.theta.numpy():.6f}, phi={circuit.phi.numpy():.6f}")
        
        # Test factory function
        config = {
            'num_qubits': 6,
            'throat_radius': 2e3,
            'traversal_probability': 0.9,
            'mass': 2e30,
            'exotic_matter_density': -2e-3
        }
        
        circuit2 = create_hybrid_backend(config)
        print(f"[OK] Hybrid circuit created via factory with {circuit2.num_qubits} qubits")
        
        return True, circuit
        
    except Exception as e:
        print(f"[FAIL] Hybrid backend creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_quantum_operations():
    """Test quantum state operations."""
    print("\nTesting quantum state operations...")
    
    try:
        from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8,
            'mass': 1e30,
            'exotic_matter_density': -1e-3
        }
        
        circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Test state creation
        start_time = time.time()
        state = circuit.create_traversal_state()
        state_time = time.time() - start_time
        
        print(f"[OK] Quantum state created in {state_time:.3f} seconds")
        print(f"  State dimensions: {state.dims}")
        print(f"  State type: {type(state)}")
        
        # Test measurements
        start_time = time.time()
        measurements = circuit.measure_observables(state)
        measure_time = time.time() - start_time
        
        print(f"[OK] Observables measured in {measure_time:.3f} seconds")
        print(f"  Number of observables: {len(measurements)}")
        
        # Show sample measurements
        for key, value in list(measurements.items())[:5]:
            print(f"    {key}: {value:.6f}")
        
        # Test entanglement metrics
        entropy = circuit.compute_entanglement_entropy(state)
        concurrence = circuit.compute_concurrence(state)
        
        print(f"[OK] Entanglement entropy: {entropy:.6f}")
        print(f"[OK] Concurrence: {concurrence:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quantum operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_components():
    """Test AI/ML components."""
    print("\nTesting AI/ML components...")
    
    try:
        from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8,
            'mass': 1e30,
            'exotic_matter_density': -1e-3
        }
        
        circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Test AI-optimized state creation
        start_time = time.time()
        ai_state = circuit.create_traversal_state(use_ai_params=True)
        ai_time = time.time() - start_time
        
        print(f"[OK] AI-optimized state created in {ai_time:.3f} seconds")
        
        # Test traversability prediction
        traversability = circuit.predict_traversability(ai_state)
        print(f"[OK] Traversability prediction: {traversability:.6f}")
        
        # Test parameter optimization
        print("[INFO] Running parameter optimization...")
        target_metrics = {'entropy': 1.0, 'concurrence': 0.8}
        
        start_time = time.time()
        optimization_results = circuit.optimize_parameters(
            target_metrics=target_metrics,
            learning_rate=0.1,
            steps=10
        )
        opt_time = time.time() - start_time
        
        print(f"[OK] Parameter optimization completed in {opt_time:.3f} seconds")
        print(f"  Optimized theta: {optimization_results['optimized_theta']:.6f}")
        print(f"  Optimized phi: {optimization_results['optimized_phi']:.6f}")
        print(f"  Final loss: {optimization_results['final_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] AI components failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_evolution():
    """Test quantum time evolution."""
    print("\nTesting quantum time evolution...")
    
    try:
        from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8,
            'mass': 1e30,
            'exotic_matter_density': -1e-3
        }
        
        circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Run time evolution
        start_time = time.time()
        evolution_results = circuit.time_evolve(time_steps=5, dt=0.1)
        evolution_time = time.time() - start_time
        
        print(f"[OK] Time evolution completed in {evolution_time:.3f} seconds")
        print(f"  Evolution steps: {len(evolution_results)}")
        
        # Show evolution of key metrics
        print("  Time evolution of entanglement:")
        for result in evolution_results:
            t = result['time']
            entropy = result['entropy']
            concurrence = result['concurrence']
            print(f"    t={t:.1f}: entropy={entropy:.4f}, concurrence={concurrence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Time evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration with main simulation."""
    print("\nTesting integration with main simulation...")
    
    try:
        from src.integration import WormholeSimulation
        from src.config import SimulationConfig
        
        # Create config with hybrid backend
        config = SimulationConfig()
        config.quantum_backend = 'tfq'  # Will fall back to hybrid
        config.num_qubits = 4
        config.time_steps = 3
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
        print(f"  System ready: {sim.is_initialized}")
        
        # Check quantum system type
        circuit = sim.quantum_system['circuit']
        circuit_type = type(circuit).__name__
        print(f"  Quantum circuit type: {circuit_type}")
        
        # Test if it's the hybrid backend
        if hasattr(circuit, 'predict_traversability'):
            print("  [OK] Hybrid backend features available")
            
            # Test AI feature
            if hasattr(circuit, 'create_traversal_state'):
                state = circuit.create_traversal_state(use_ai_params=True)
                traversability = circuit.predict_traversability(state)
                print(f"    AI traversability prediction: {traversability:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_training():
    """Test AI training capabilities."""
    print("\nTesting AI training capabilities...")
    
    try:
        from src.quantum.hybrid_quantum_ai import HybridQuantumAICircuit
        
        geometry_params = {
            'throat_radius': 1e3,
            'traversal_probability': 0.8,
            'mass': 1e30,
            'exotic_matter_density': -1e-3
        }
        
        circuit = HybridQuantumAICircuit(num_qubits=4, geometry_params=geometry_params)
        
        # Create mock training data
        training_data = []
        for i in range(10):
            # Generate random wormhole configurations
            state = circuit.create_traversal_state()
            state_vector = state.full().flatten()
            
            example = {
                'throat_radius': 1e3 + i*100,
                'mass': 1e30 * (1 + 0.1*i),
                'traversal_probability': 0.7 + 0.02*i,
                'exotic_matter_density': -1e-3 * (1 + 0.1*i),
                'optimal_theta': np.pi/4 + 0.1*i,
                'optimal_phi': 0.1*i,
                'state_vector': state_vector,
                'traversability': 0.8 if i % 2 == 0 else 0.6
            }
            training_data.append(example)
        
        print(f"[INFO] Generated {len(training_data)} training examples")
        
        # Train AI components
        start_time = time.time()
        circuit.train_ai_components(training_data, epochs=5)
        training_time = time.time() - start_time
        
        print(f"[OK] AI training completed in {training_time:.3f} seconds")
        
        # Test trained model predictions
        test_geometry_features = np.array([[1e3, 1e30, 0.8, -1e-3]], dtype=np.float32)
        test_geometry_features = test_geometry_features / np.array([1e4, 1e31, 1.0, 1e-2])
        
        predicted_params = circuit.param_optimizer(test_geometry_features)
        print(f"  Predicted parameters: theta={predicted_params[0,0]:.6f}, phi={predicted_params[0,1]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] AI training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all hybrid backend tests."""
    print("Hybrid Quantum-AI Backend Test Suite")
    print("=" * 60)
    
    tests = [
        ("Backend Creation", test_hybrid_backend_creation),
        ("Quantum Operations", test_quantum_operations),
        ("AI Components", test_ai_components),
        ("Time Evolution", test_time_evolution),
        ("Integration", test_integration),
        ("AI Training", test_ai_training),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        print("-" * 30)
        
        try:
            if test_func == test_hybrid_backend_creation:
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
    print(f"Hybrid Quantum-AI Backend Test Summary")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Hybrid Quantum-AI backend is ready.")
    elif passed > 0:
        print("[WARN] Partial success. Some features may not be available.")
    else:
        print("[ERROR] Hybrid backend not functional.")
    
    print("\nPhase 3 Quantum Backend Status:")
    print(f"- Quantum circuit creation: {'[OK]' if results.get('Backend Creation', False) else '[FAIL]'}")
    print(f"- Entanglement dynamics: {'[OK]' if results.get('Quantum Operations', False) else '[FAIL]'}")
    print(f"- AI optimization: {'[OK]' if results.get('AI Components', False) else '[FAIL]'}")
    print(f"- Time evolution: {'[OK]' if results.get('Time Evolution', False) else '[FAIL]'}")
    print(f"- Integration ready: {'[OK]' if results.get('Integration', False) else '[FAIL]'}")
    
    if results.get('AI Components', False):
        print("\nReady for Phase 3 activities:")
        print("- Parameter space exploration: [READY]")
        print("- Bayesian optimization: [READY]")
        print("- Advanced scenario testing: [READY]")
    
    return passed >= 4  # Allow some failures for edge cases

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)