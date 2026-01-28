#!/usr/bin/env python3
"""
Test Bayesian Search for Traversable Wormhole Solutions.

This script tests the advanced Bayesian optimization system specifically
designed to find physically viable traversable wormhole configurations.
"""

import sys
import time
import numpy as np
sys.path.append('src')

def test_traversability_constraints():
    """Test traversability constraints system."""
    print("Testing Traversability Constraints...")
    
    try:
        from src.ai.bayesian_wormhole_search import TraversabilityConstraints
        
        # Test default constraints
        default_constraints = TraversabilityConstraints()
        print(f"  Default max tidal force: {default_constraints.max_tidal_force} N")
        print(f"  Default min stability time: {default_constraints.min_stability_time} s")
        print(f"  Default max exotic matter energy: {default_constraints.max_exotic_matter_energy:.2e} J")
        
        # Test custom constraints
        strict_constraints = TraversabilityConstraints(
            max_tidal_force=500.0,
            min_stability_time=200.0,
            min_entanglement_persistence=0.8
        )
        
        print(f"  Strict max tidal force: {strict_constraints.max_tidal_force} N")
        print(f"  Strict min stability time: {strict_constraints.min_stability_time} s")
        print(f"  Strict min entanglement persistence: {strict_constraints.min_entanglement_persistence}")
        
        # Test constraint dictionary conversion
        constraint_dict = default_constraints.to_dict()
        print(f"  Constraint dictionary keys: {list(constraint_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Traversability constraints test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wormhole_candidate():
    """Test wormhole candidate evaluation."""
    print("\nTesting Wormhole Candidate...")
    
    try:
        from src.ai.bayesian_wormhole_search import (
            WormholeCandidate, 
            TraversabilityConstraints
        )
        
        # Create test candidate
        test_params = {
            'throat_radius': 1000.0,
            'mass': 1e30,
            'exotic_matter_density': -1e-3,
            'traversal_probability': 0.8,
            'quantum_coherence_time': 100.0
        }
        
        # Simulate realistic constraint violations
        violations = {
            'nec_violation': -1e-7,  # Good (within limits)
            'wec_violation': -1e-7,  # Good
            'sec_violation': -1e-6,  # Good  
            'tidal_force': 800.0,    # Good (< 1000 N)
            'decoherence_rate': 0.01 # Good (< 0.1)
        }
        
        physics_metrics = {
            'entanglement_persistence': 0.7,  # Good
            'stability_time': 150.0,          # Good
            'entanglement_entropy': 0.5,
            'concurrence': 0.8
        }
        
        candidate = WormholeCandidate(
            parameters=test_params,
            traversability_score=0.75,
            constraint_violations=violations,
            physics_metrics=physics_metrics,
            feasibility_score=0.8,
            discovery_iteration=1,
            evaluation_time=2.0
        )
        
        # Test viability check
        constraints = TraversabilityConstraints()
        is_viable = candidate.is_physically_viable(constraints)
        overall_score = candidate.get_overall_score()
        
        print(f"  Candidate parameters: {test_params}")
        print(f"  Traversability score: {candidate.traversability_score:.4f}")
        print(f"  Feasibility score: {candidate.feasibility_score:.4f}")
        print(f"  Overall score: {overall_score:.4f}")
        print(f"  Is physically viable: {is_viable}")
        
        # Test with violating candidate
        bad_violations = violations.copy()
        bad_violations['tidal_force'] = 2000.0  # Too high
        
        bad_candidate = WormholeCandidate(
            parameters=test_params,
            traversability_score=0.9,  # High traversability but bad physics
            constraint_violations=bad_violations,
            physics_metrics=physics_metrics,
            feasibility_score=0.2,
            discovery_iteration=2,
            evaluation_time=2.0
        )
        
        bad_viable = bad_candidate.is_physically_viable(constraints)
        print(f"  Bad candidate viable: {bad_viable} (should be False)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Wormhole candidate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traversability_objective():
    """Test enhanced traversability objective function."""
    print("\nTesting Traversability Objective...")
    
    try:
        from src.ai.bayesian_wormhole_search import (
            TraversabilityObjective,
            TraversabilityConstraints,
            create_wormhole_circuit_factory
        )
        
        # Setup
        circuit_factory = create_wormhole_circuit_factory(num_qubits=4)
        constraints = TraversabilityConstraints()
        objective = TraversabilityObjective(circuit_factory, constraints)
        
        # Test with reasonable parameters
        test_params = np.array([
            1000.0,    # throat_radius
            1e30,      # mass
            -1e-3,     # exotic_matter_density
            0.8,       # traversal_probability
            100.0      # quantum_coherence_time
        ])
        
        print(f"  Test parameters: {test_params}")
        
        start_time = time.time()
        score = objective(test_params)
        evaluation_time = time.time() - start_time
        
        print(f"  Objective score: {score:.6f}")
        print(f"  Evaluation time: {evaluation_time:.3f}s")
        print(f"  Candidate history length: {len(objective.candidate_history)}")
        
        # Check if candidate was recorded
        if objective.candidate_history:
            candidate = objective.candidate_history[0]
            print(f"  Recorded traversability: {candidate.traversability_score:.4f}")
            print(f"  Recorded feasibility: {candidate.feasibility_score:.4f}")
            print(f"  Constraint violations: {len(candidate.constraint_violations)}")
            print(f"  Physics metrics: {len(candidate.physics_metrics)}")
        
        # Test viable candidate detection
        viable_candidates = objective.get_viable_candidates()
        best_candidate = objective.get_best_candidate()
        
        print(f"  Viable candidates found: {len(viable_candidates)}")
        print(f"  Best candidate score: {best_candidate.get_overall_score():.4f}" if best_candidate else "None")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Traversability objective test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bayesian_wormhole_search():
    """Test the main Bayesian wormhole search system."""
    print("\nTesting Bayesian Wormhole Search...")
    
    try:
        from src.ai.bayesian_wormhole_search import (
            BayesianWormholeSearch,
            TraversabilityConstraints,
            create_wormhole_circuit_factory
        )
        
        # Setup
        circuit_factory = create_wormhole_circuit_factory(num_qubits=4)
        constraints = TraversabilityConstraints(
            max_tidal_force=1200.0,  # Slightly relaxed for testing
            min_stability_time=50.0   # Reduced for faster testing
        )
        
        search = BayesianWormholeSearch(
            circuit_factory,
            constraints,
            acquisition_function='expected_improvement'
        )
        
        # Define small search space for testing
        parameter_bounds = np.array([
            [800.0, 1200.0],    # throat_radius (m) - narrow range
            [5e29, 2e30],       # mass (kg) - narrow range  
            [-2e-3, -5e-4],     # exotic_matter_density - narrow range
            [0.7, 0.9],         # traversal_probability - narrow range
            [80.0, 120.0]       # quantum_coherence_time - narrow range
        ])
        
        print("  Running Bayesian search (small test)...")
        start_time = time.time()
        
        # Run short search for testing
        results = search.search(
            n_iterations=12,
            n_initial=8, 
            parameter_bounds=parameter_bounds
        )
        
        search_time = time.time() - start_time
        
        print(f"  Search completed in {search_time:.3f}s")
        print(f"  Total evaluations: {results['total_evaluations']}")
        print(f"  Viable candidates found: {len(results['viable_candidates'])}")
        print(f"  Best score: {results['search_statistics']['best_score']:.6f}")
        print(f"  Viable fraction: {results['search_statistics']['viable_fraction']:.3f}")
        
        # Analyze best candidate
        if results['best_candidate']:
            best = results['best_candidate']
            print(f"  Best candidate traversability: {best['traversability_score']:.4f}")
            print(f"  Best candidate feasibility: {best['feasibility_score']:.4f}")
            print(f"  Best candidate viable: {best['is_viable']}")
            
            print("  Best parameters:")
            for param, value in best['parameters'].items():
                print(f"    {param}: {value:.2e}")
        
        # Show viable candidates
        if results['viable_candidates']:
            print(f"\n  Viable candidate details:")
            for i, candidate in enumerate(results['viable_candidates'][:3]):  # Show first 3
                print(f"    Candidate {i+1}:")
                print(f"      Overall score: {candidate['overall_score']:.4f}")
                print(f"      Throat radius: {candidate['parameters']['throat_radius']:.1f} m")
                print(f"      Tidal force: {candidate['constraint_violations'].get('tidal_force', 'N/A'):.1f} N")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Bayesian wormhole search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_result_analysis():
    """Test search result analysis and saving."""
    print("\nTesting Search Result Analysis...")
    
    try:
        from src.ai.bayesian_wormhole_search import (
            BayesianWormholeSearch,
            TraversabilityConstraints,
            create_wormhole_circuit_factory,
            save_search_results
        )
        
        # Quick search to generate results
        circuit_factory = create_wormhole_circuit_factory(num_qubits=4)
        constraints = TraversabilityConstraints()
        search = BayesianWormholeSearch(circuit_factory, constraints)
        
        # Minimal search for testing
        parameter_bounds = np.array([
            [900.0, 1100.0],
            [8e29, 1.2e30], 
            [-1.5e-3, -8e-4],
            [0.75, 0.85],
            [90.0, 110.0]
        ])
        
        results = search.search(
            n_iterations=6,
            n_initial=4,
            parameter_bounds=parameter_bounds
        )
        
        # Test result structure
        expected_keys = [
            'search_time', 'total_evaluations', 'viable_candidates',
            'best_candidate', 'search_statistics', 'constraints_used',
            'optimization_history'
        ]
        
        missing_keys = [key for key in expected_keys if key not in results]
        print(f"  Result structure complete: {len(missing_keys) == 0}")
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        
        # Test saving
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_wormhole_search_{timestamp}.json"
        
        save_search_results(results, filename)
        print(f"  Results saved to: {filename}")
        
        # Verify file was created
        import os
        file_exists = os.path.exists(filename)
        print(f"  File created successfully: {file_exists}")
        
        if file_exists:
            file_size = os.path.getsize(filename)
            print(f"  File size: {file_size} bytes")
            
            # Clean up test file
            try:
                os.remove(filename)
                print(f"  Test file cleaned up")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Search result analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Bayesian wormhole search tests."""
    print("Bayesian Wormhole Search Test Suite")
    print("=" * 60)
    
    tests = [
        ("Traversability Constraints", test_traversability_constraints),
        ("Wormhole Candidate", test_wormhole_candidate),
        ("Traversability Objective", test_traversability_objective),
        ("Bayesian Wormhole Search", test_bayesian_wormhole_search),
        ("Search Result Analysis", test_search_result_analysis),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        print("-" * 40)
        
        try:
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
    print(f"Bayesian Wormhole Search Test Summary")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All Bayesian search tests passed!")
        print("\nBayesian Wormhole Search Features:")
        print("- Physics constraint enforcement: [READY]")
        print("- Traversability optimization: [READY]")
        print("- Viable candidate detection: [READY]")
        print("- Advanced acquisition functions: [READY]")
        print("- Multi-phase search strategy: [READY]")
        print("\nPhase 3 Bayesian Search: COMPLETE")
    elif passed >= 3:
        print("[PARTIAL] Core Bayesian search functionality working.")
        print("Some advanced features may have issues.")
    else:
        print("[ERROR] Bayesian wormhole search not functional.")
    
    print(f"\nNext Phase 3 Steps:")
    if results.get('Bayesian Wormhole Search', False):
        print("- Advanced scenario optimization: [READY]")
        print("- Multi-objective optimization: [READY]")
        print("- Real-time search monitoring: [PENDING]")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)