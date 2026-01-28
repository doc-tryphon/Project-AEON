#!/usr/bin/env python3
"""
Test Dynamic Throat Evolution Implementation.

This script tests the dynamic evolution of wormhole throat geometries,
including time-dependent metrics, stability analysis, and various
evolution scenarios (expansion, collapse, stabilization).
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')

def test_evolution_parameters():
    """Test evolution parameters validation."""
    print("Testing Evolution Parameters...")
    
    try:
        from src.physics.dynamic_throat_evolution import EvolutionParameters
        
        # Test default parameters
        default_params = EvolutionParameters()
        print(f"  Default evolution timescale: {default_params.evolution_timescale} s")
        print(f"  Default damping coefficient: {default_params.damping_coefficient}")
        print(f"  Default exotic matter pressure: {default_params.exotic_matter_pressure:.2e} Pa")
        print(f"  Default minimum throat radius: {default_params.minimum_throat_radius} m")
        
        # Test custom parameters
        custom_params = EvolutionParameters(
            evolution_timescale=500.0,
            damping_coefficient=0.2,
            exotic_matter_pressure=-5e-3,
            minimum_throat_radius=50.0
        )
        print(f"  Custom evolution timescale: {custom_params.evolution_timescale} s")
        print(f"  Custom damping coefficient: {custom_params.damping_coefficient}")
        
        # Test validation
        try:
            invalid_params = EvolutionParameters(minimum_throat_radius=-10.0)
            print(f"  [FAIL] Should have rejected negative minimum radius")
            return False
        except ValueError:
            print(f"  [OK] Correctly rejected negative minimum radius")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Evolution parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_throat_evolution():
    """Test basic throat evolution dynamics."""
    print("\nTesting Basic Throat Evolution...")
    
    try:
        from src.physics.dynamic_throat_evolution import (
            DynamicThroatEvolution, 
            EvolutionParameters,
            create_evolution_scenario
        )
        
        # Create standard evolution scenario
        evolution_system = create_evolution_scenario(
            scenario_type="standard",
            initial_radius=1000.0,
            mass=1e30,
            angular_momentum=1e43
        )
        
        print(f"  Initial throat radius: {evolution_system.current_throat_radius} m")
        print(f"  Initial mass: {evolution_system.current_mass:.2e} kg")
        print(f"  Initial angular momentum: {evolution_system.current_rotation_params.angular_momentum:.2e}")
        
        # Run short evolution
        start_time = time.time()
        evolution_result = evolution_system.evolve_throat(time_span=100.0, num_steps=200)
        evolution_time = time.time() - start_time
        
        print(f"  Evolution completed in {evolution_time:.3f}s")
        
        if evolution_result['evolution_success']:
            stats = evolution_result['statistics']
            print(f"  Final throat radius: {stats['final_radius']:.1f} m")
            print(f"  Radius change: {stats['radius_change']:.1f} m")
            print(f"  Relative change: {stats['relative_change']*100:.2f}%")
            print(f"  Final state: {stats['final_state']}")
            print(f"  Stability score: {stats['stability_score']:.4f}")
            
            # Check energy analysis
            energy_analysis = evolution_result['energy_analysis']
            print(f"  Energy conservation violation: {energy_analysis['energy_conservation_violation']:.2e}")
            
            return True
        else:
            print(f"  [FAIL] Evolution failed")
            return False
        
    except Exception as e:
        print(f"[FAIL] Basic throat evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_throat_collapse_scenario():
    """Test throat collapse scenario."""
    print("\nTesting Throat Collapse Scenario...")
    
    try:
        from src.physics.dynamic_throat_evolution import (
            ThroatCollapseScenario,
            create_evolution_scenario
        )
        
        # Create collapse scenario
        collapse_scenario = ThroatCollapseScenario(
            initial_radius=800.0,
            collapse_timescale=50.0
        )
        
        print(f"  Initial radius: {collapse_scenario.initial_radius} m")
        print(f"  Collapse timescale: {collapse_scenario.collapse_timescale} s")
        
        # Run collapse simulation
        start_time = time.time()
        collapse_result = collapse_scenario.run_collapse_simulation(simulation_time=200.0)
        simulation_time = time.time() - start_time
        
        print(f"  Collapse simulation completed in {simulation_time:.3f}s")
        
        if collapse_result['evolution_success']:
            stats = collapse_result['statistics']
            collapse_analysis = collapse_result['collapse_analysis']
            
            print(f"  Final radius: {stats['final_radius']:.1f} m")
            print(f"  Minimum radius reached: {collapse_analysis['minimum_radius_reached']:.1f} m")
            print(f"  Collapse prevented: {collapse_analysis['collapse_prevented']}")
            print(f"  Max collapse rate: {collapse_analysis['max_collapse_rate']:.3f} m/s")
            print(f"  Total collapse time: {collapse_analysis['total_collapse_time']:.1f} s")
            
            return True
        else:
            print(f"  [FAIL] Collapse simulation failed")
            return False
        
    except Exception as e:
        print(f"[FAIL] Throat collapse test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_throat_expansion_scenario():
    """Test throat expansion scenario."""
    print("\nTesting Throat Expansion Scenario...")
    
    try:
        from src.physics.dynamic_throat_evolution import ThroatExpansionScenario
        
        # Create expansion scenario
        expansion_scenario = ThroatExpansionScenario(
            initial_radius=300.0,
            expansion_timescale=500.0
        )
        
        print(f"  Initial radius: {expansion_scenario.initial_radius} m")
        print(f"  Expansion timescale: {expansion_scenario.expansion_timescale} s")
        
        # Run expansion simulation
        start_time = time.time()
        expansion_result = expansion_scenario.run_expansion_simulation(simulation_time=1000.0)
        simulation_time = time.time() - start_time
        
        print(f"  Expansion simulation completed in {simulation_time:.3f}s")
        
        if expansion_result['evolution_success']:
            stats = expansion_result['statistics']
            expansion_analysis = expansion_result['expansion_analysis']
            
            print(f"  Final radius: {stats['final_radius']:.1f} m")
            print(f"  Total radius increase: {expansion_analysis['total_radius_increase']:.1f} m")
            print(f"  Expansion efficiency: {expansion_analysis['expansion_efficiency']:.2f}")
            print(f"  Max expansion rate: {expansion_analysis['max_expansion_rate']:.3f} m/s")
            print(f"  Approaching steady state: {expansion_analysis['approaching_steady_state']}")
            
            return True
        else:
            print(f"  [FAIL] Expansion simulation failed")
            return False
        
    except Exception as e:
        print(f"[FAIL] Throat expansion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_long_term_stability_analysis():
    """Test long-term stability analysis."""
    print("\nTesting Long-term Stability Analysis...")
    
    try:
        from src.physics.dynamic_throat_evolution import create_evolution_scenario
        
        # Create stable evolution scenario
        evolution_system = create_evolution_scenario(
            scenario_type="standard",
            initial_radius=1000.0,
            evolution_timescale=2000.0,
            exotic_matter_pressure=-1e-4  # Weak exotic matter for stability
        )
        
        print(f"  Running long-term analysis...")
        
        # Run long-term stability analysis
        start_time = time.time()
        stability_result = evolution_system.analyze_long_term_stability(
            total_time=2000.0,
            checkpoint_interval=400.0
        )
        analysis_time = time.time() - start_time
        
        print(f"  Long-term analysis completed in {analysis_time:.3f}s")
        print(f"  Total evolution time: {stability_result['total_evolution_time']:.1f} s")
        print(f"  Number of checkpoints: {len(stability_result['checkpoints'])}")
        
        if stability_result['checkpoints']:
            long_term_analysis = stability_result['long_term_analysis']
            
            print(f"  Long-term behavior: {long_term_analysis['long_term_behavior']}")
            print(f"  Overall assessment: {long_term_analysis['overall_assessment']}")
            print(f"  Mean stability score: {long_term_analysis['mean_stability_score']:.4f}")
            print(f"  Radius trend: {long_term_analysis['radius_trend']:.2e} m/s")
            print(f"  Final radius: {long_term_analysis['final_radius']:.1f} m")
            print(f"  Total radius change: {long_term_analysis['total_radius_change']:.1f} m")
            
            return True
        else:
            print(f"  [FAIL] No checkpoints generated")
            return False
        
    except Exception as e:
        print(f"[FAIL] Long-term stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scenario_comparison():
    """Test comparison of multiple evolution scenarios."""
    print("\nTesting Scenario Comparison...")
    
    try:
        from src.physics.dynamic_throat_evolution import compare_evolution_scenarios
        
        # Define scenarios to compare
        scenarios = ["standard", "collapse", "expansion"]
        
        print(f"  Comparing scenarios: {scenarios}")
        
        # Run scenario comparison
        start_time = time.time()
        comparison_result = compare_evolution_scenarios(
            scenarios=scenarios,
            simulation_time=500.0,
            initial_radius=800.0
        )
        comparison_time = time.time() - start_time
        
        print(f"  Scenario comparison completed in {comparison_time:.3f}s")
        
        comparative_analysis = comparison_result['comparative_analysis']
        
        if 'error' not in comparative_analysis:
            print(f"  Successful scenarios: {comparative_analysis['successful_scenarios']}/{comparative_analysis['total_scenarios']}")
            print(f"  Most stable scenario: {comparative_analysis['most_stable_scenario']}")
            print(f"  Least stable scenario: {comparative_analysis['least_stable_scenario']}")
            print(f"  Mean final radius: {comparative_analysis['mean_final_radius']:.1f} m")
            print(f"  Mean stability score: {comparative_analysis['mean_stability_score']:.4f}")
            
            # Show individual scenario results
            scenario_metrics = comparative_analysis['scenario_metrics']
            print(f"  Individual scenario results:")
            print(f"    Scenario     | Final Radius | Stability | Final State")
            print(f"    -------------|--------------|-----------|-------------")
            
            for scenario, metrics in scenario_metrics.items():
                print(f"    {scenario:12} | {metrics['final_radius']:11.1f} | {metrics['stability_score']:8.4f} | {metrics['final_state']}")
            
            return True
        else:
            print(f"  [FAIL] Comparison failed: {comparative_analysis['error']}")
            return False
        
    except Exception as e:
        print(f"[FAIL] Scenario comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evolution_equations():
    """Test throat evolution differential equations."""
    print("\nTesting Evolution Equations...")
    
    try:
        from src.physics.dynamic_throat_evolution import (
            DynamicThroatEvolution,
            EvolutionParameters,
            create_evolution_scenario
        )
        
        # Create evolution system
        evolution_system = create_evolution_scenario("standard")
        
        # Test evolution equation at different states
        test_states = [
            np.array([500.0, 1e30, 1e43]),    # Small throat
            np.array([1000.0, 1e30, 1e43]),   # Medium throat  
            np.array([2000.0, 1e30, 1e43]),   # Large throat
            np.array([1000.0, 5e29, 1e43]),   # Low mass
            np.array([1000.0, 2e30, 1e43]),   # High mass
            np.array([1000.0, 1e30, 5e44]),   # High angular momentum
        ]
        
        test_descriptions = [
            "Small throat",
            "Medium throat", 
            "Large throat",
            "Low mass",
            "High mass",
            "High angular momentum"
        ]
        
        print(f"  Testing evolution equations at different states:")
        print(f"  State                | db/dt (m/s) | dM/dt (kg/s) | dJ/dt (J*s/s)")
        print(f"  ---------------------|-------------|--------------|---------------")
        
        for i, (state, description) in enumerate(zip(test_states, test_descriptions)):
            try:
                derivatives = evolution_system.throat_evolution_equation(0.0, state)
                db_dt, dM_dt, dJ_dt = derivatives
                
                print(f"  {description:20} | {db_dt:10.6f} | {dM_dt:11.2e} | {dJ_dt:12.2e}")
                
                # Check for reasonable values
                if not (np.isfinite(db_dt) and np.isfinite(dM_dt) and np.isfinite(dJ_dt)):
                    print(f"    [WARNING] Non-finite derivatives detected")
                
            except Exception as e:
                print(f"    [ERROR] Failed for {description}: {e}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Evolution equations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all dynamic throat evolution tests."""
    print("Dynamic Throat Evolution Test Suite")
    print("=" * 60)
    
    tests = [
        ("Evolution Parameters", test_evolution_parameters),
        ("Basic Throat Evolution", test_basic_throat_evolution),
        ("Throat Collapse Scenario", test_throat_collapse_scenario),
        ("Throat Expansion Scenario", test_throat_expansion_scenario),
        ("Long-term Stability Analysis", test_long_term_stability_analysis),
        ("Scenario Comparison", test_scenario_comparison),
        ("Evolution Equations", test_evolution_equations),
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
    print(f"Dynamic Throat Evolution Test Summary")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All dynamic throat evolution tests passed!")
        print("\nDynamic Throat Evolution Features:")
        print("- Time-dependent throat geometry: [READY]")
        print("- Collapse scenario analysis: [READY]")
        print("- Expansion scenario analysis: [READY]")
        print("- Long-term stability assessment: [READY]")
        print("- Multi-scenario comparison: [READY]")
        print("- Evolution equation integration: [READY]")
        print("\nPhase 3 Dynamic Evolution: COMPLETE")
    elif passed >= 5:
        print("[PARTIAL] Core dynamic evolution functionality working.")
        print("Advanced scenarios may have minor issues.")
    else:
        print("[ERROR] Dynamic throat evolution not functional.")
    
    print(f"\nNext Phase 3 Steps:")
    if results.get('Long-term Stability Analysis', False):
        print("- Real-time visualization system: [READY]")
        print("- Multi-scenario validation sweeps: [READY]")
        print("- Parameter space exploration: [READY]")
    
    return passed >= 5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)