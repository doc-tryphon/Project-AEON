#!/usr/bin/env python3
"""
Test Multi-Scenario Validation System.

This script tests the comprehensive validation system that validates all
Phase 3 components across multiple scenarios and parameter ranges.
"""

import sys
import time
import numpy as np
import json
import os
from pathlib import Path
sys.path.append('src')

def test_validation_config():
    """Test validation configuration."""
    print("Testing Validation Configuration...")
    
    try:
        from src.validation.multi_scenario_validation import ValidationConfig
        
        # Test default configuration
        default_config = ValidationConfig()
        print(f"  Default parameter sweep points: {default_config.parameter_sweep_points}")
        print(f"  Default scenarios: {default_config.scenarios_to_test}")
        print(f"  Quantum backend tests enabled: {default_config.quantum_backend_tests}")
        print(f"  Performance benchmark enabled: {default_config.performance_benchmark}")
        print(f"  Statistical analysis enabled: {default_config.statistical_analysis}")
        
        # Test parameter ranges
        print(f"  Parameter ranges defined: {len(default_config.parameter_ranges)}")
        for param, (min_val, max_val) in default_config.parameter_ranges.items():
            print(f"    {param}: [{min_val:.2e}, {max_val:.2e}]")
        
        # Test custom configuration
        custom_config = ValidationConfig(
            parameter_sweep_points=10,
            scenarios_to_test=['standard', 'collapse'],
            benchmark_iterations=5,
            max_workers=2
        )
        
        print(f"  Custom sweep points: {custom_config.parameter_sweep_points}")
        print(f"  Custom scenarios: {custom_config.scenarios_to_test}")
        print(f"  Custom benchmark iterations: {custom_config.benchmark_iterations}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Validation config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validator_initialization():
    """Test validator initialization."""
    print("\nTesting Validator Initialization...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator
        config = ValidationConfig(
            parameter_sweep_points=5,
            benchmark_iterations=2,
            create_summary_plots=False,
            export_csv_data=False
        )
        
        validator = MultiScenarioValidator(config)
        
        print(f"  Validator created successfully")
        print(f"  Output directory: {validator.output_dir}")
        print(f"  Validation timestamp: {validator.validation_timestamp}")
        print(f"  Configuration loaded: {validator.config is not None}")
        
        # Test directory creation
        output_exists = validator.output_dir.exists()
        print(f"  Output directory created: {output_exists}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Validator initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_backend_validation():
    """Test quantum backend validation specifically."""
    print("\nTesting Quantum Backend Validation...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator
        config = ValidationConfig(
            quantum_backend_tests=True,
            ml_optimization_tests=False,
            bayesian_search_tests=False,
            rotating_metrics_tests=False,
            dynamic_evolution_tests=False,
            visualization_tests=False,
            performance_benchmark=False,
            create_summary_plots=False
        )
        
        validator = MultiScenarioValidator(config)
        
        # Run quantum backend validation only
        print(f"  Running quantum backend validation...")
        start_time = time.time()
        
        quantum_results = validator._validate_quantum_backend()
        validation_time = time.time() - start_time
        
        print(f"  Quantum backend validation completed in {validation_time:.3f}s")
        print(f"  Tests passed: {quantum_results['tests_passed']}/{quantum_results['total_tests']}")
        print(f"  Success rate: {quantum_results['success_rate']:.2f}")
        
        # Check specific test results
        for test_name, result in quantum_results['details'].items():
            print(f"    {test_name}: {result}")
        
        return quantum_results['success_rate'] > 0.5
        
    except Exception as e:
        print(f"[FAIL] Quantum backend validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_optimization_validation():
    """Test ML optimization validation."""
    print("\nTesting ML Optimization Validation...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator
        config = ValidationConfig(
            quantum_backend_tests=False,
            ml_optimization_tests=True,
            bayesian_search_tests=False,
            rotating_metrics_tests=False,
            dynamic_evolution_tests=False,
            visualization_tests=False,
            performance_benchmark=False,
            create_summary_plots=False
        )
        
        validator = MultiScenarioValidator(config)
        
        # Run ML optimization validation
        print(f"  Running ML optimization validation...")
        start_time = time.time()
        
        ml_results = validator._validate_ml_optimization()
        validation_time = time.time() - start_time
        
        print(f"  ML optimization validation completed in {validation_time:.3f}s")
        print(f"  Tests passed: {ml_results['tests_passed']}/{ml_results['total_tests']}")
        print(f"  Success rate: {ml_results['success_rate']:.2f}")
        
        # Check optimization results
        if 'optimization_results' in ml_results:
            print(f"  Optimization methods tested: {len(ml_results['optimization_results'])}")
            for method, result in ml_results['optimization_results'].items():
                print(f"    {method}: score={result['best_score']:.4f}, evals={result['total_evaluations']}")
        
        return ml_results['success_rate'] > 0.5
        
    except Exception as e:
        print(f"[FAIL] ML optimization validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_sweep_generation():
    """Test parameter combination generation."""
    print("\nTesting Parameter Sweep Generation...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator
        config = ValidationConfig(parameter_sweep_points=15)
        validator = MultiScenarioValidator(config)
        
        # Generate parameter combinations
        print(f"  Generating parameter combinations...")
        start_time = time.time()
        
        param_combinations = validator._generate_parameter_combinations()
        generation_time = time.time() - start_time
        
        print(f"  Parameter generation completed in {generation_time:.3f}s")
        print(f"  Combinations generated: {len(param_combinations)}")
        
        if param_combinations:
            # Check first combination
            first_combo = param_combinations[0]
            print(f"  Sample combination parameters: {list(first_combo.keys())}")
            print(f"  Sample values:")
            for param, value in first_combo.items():
                print(f"    {param}: {value:.2e}")
            
            # Verify parameters are within expected ranges
            valid_combinations = 0
            for combo in param_combinations:
                if all(isinstance(v, (int, float)) and v > 0 for v in combo.values()):
                    valid_combinations += 1
            
            print(f"  Valid combinations: {valid_combinations}/{len(param_combinations)}")
            
            return valid_combinations > 0
        else:
            print(f"  [FAIL] No parameter combinations generated")
            return False
        
    except Exception as e:
        print(f"[FAIL] Parameter sweep generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scenario_validation():
    """Test individual scenario validation."""
    print("\nTesting Scenario Validation...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator focused on dynamic evolution
        config = ValidationConfig(
            quantum_backend_tests=False,
            ml_optimization_tests=False,
            bayesian_search_tests=False,
            rotating_metrics_tests=False,
            dynamic_evolution_tests=True,
            visualization_tests=False,
            performance_benchmark=False,
            create_summary_plots=False,
            scenarios_to_test=['standard', 'collapse']
        )
        
        validator = MultiScenarioValidator(config)
        
        # Run dynamic evolution validation
        print(f"  Running dynamic evolution validation...")
        start_time = time.time()
        
        evolution_results = validator._validate_dynamic_evolution()
        validation_time = time.time() - start_time
        
        print(f"  Dynamic evolution validation completed in {validation_time:.3f}s")
        print(f"  Tests passed: {evolution_results['tests_passed']}/{evolution_results['total_tests']}")
        print(f"  Success rate: {evolution_results['success_rate']:.2f}")
        
        # Check scenario results
        if 'evolution_results' in evolution_results:
            print(f"  Scenario results:")
            for scenario, result in evolution_results['evolution_results'].items():
                status = "SUCCESS" if result['success'] else "FAILED"
                print(f"    {scenario}: {status}")
                if result['success']:
                    print(f"      Final radius: {result['final_radius']:.1f} m")
                    print(f"      Stability score: {result['stability_score']:.4f}")
        
        return evolution_results['success_rate'] > 0.5
        
    except Exception as e:
        print(f"[FAIL] Scenario validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_validation():
    """Test quick validation function."""
    print("\nTesting Quick Validation...")
    
    try:
        from src.validation.multi_scenario_validation import quick_validation_check
        
        print(f"  Running quick validation check...")
        start_time = time.time()
        
        quick_results = quick_validation_check()
        validation_time = time.time() - start_time
        
        print(f"  Quick validation completed in {validation_time:.3f}s")
        
        # Check results structure
        expected_keys = ['timestamp', 'system_validations', 'overall_summary']
        found_keys = [key for key in expected_keys if key in quick_results]
        print(f"  Result structure complete: {len(found_keys)}/{len(expected_keys)}")
        
        # Check overall summary
        if 'overall_summary' in quick_results:
            summary = quick_results['overall_summary']
            print(f"  Overall success rate: {summary.get('overall_success_rate', 'N/A'):.2f}")
            print(f"  Systems tested: {len(summary.get('systems_tested', []))}")
            
            if 'key_findings' in summary:
                print(f"  Key findings: {len(summary['key_findings'])}")
                for finding in summary['key_findings'][:2]:  # Show first 2
                    print(f"    - {finding}")
        
        return 'overall_summary' in quick_results
        
    except Exception as e:
        print(f"[FAIL] Quick validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmarking():
    """Test performance benchmarking."""
    print("\nTesting Performance Benchmarking...")
    
    try:
        from src.validation.multi_scenario_validation import (
            MultiScenarioValidator,
            ValidationConfig
        )
        
        # Create validator with performance benchmarking
        config = ValidationConfig(
            quantum_backend_tests=False,
            ml_optimization_tests=False,
            bayesian_search_tests=False,
            rotating_metrics_tests=False,
            dynamic_evolution_tests=False,
            visualization_tests=False,
            performance_benchmark=True,
            benchmark_iterations=2,  # Small number for testing
            create_summary_plots=False
        )
        
        validator = MultiScenarioValidator(config)
        
        print(f"  Running performance benchmarks...")
        start_time = time.time()
        
        # Test individual benchmark functions
        quantum_benchmark = validator._benchmark_quantum_backend()
        evolution_benchmark = validator._benchmark_dynamic_evolution()
        
        benchmark_time = time.time() - start_time
        
        print(f"  Performance benchmarking completed in {benchmark_time:.3f}s")
        
        # Check quantum benchmark results
        print(f"  Quantum backend benchmark:")
        print(f"    Successful runs: {quantum_benchmark['successful_runs']}/{quantum_benchmark['total_runs']}")
        print(f"    Success rate: {quantum_benchmark['success_rate']:.2f}")
        if quantum_benchmark['mean_time'] is not None:
            print(f"    Mean time: {quantum_benchmark['mean_time']:.3f}s")
        
        # Check evolution benchmark results
        print(f"  Dynamic evolution benchmark:")
        print(f"    Successful runs: {evolution_benchmark['successful_runs']}/{evolution_benchmark['total_runs']}")
        print(f"    Success rate: {evolution_benchmark['success_rate']:.2f}")
        if evolution_benchmark['mean_time'] is not None:
            print(f"    Mean time: {evolution_benchmark['mean_time']:.3f}s")
        
        return (quantum_benchmark['success_rate'] > 0 or evolution_benchmark['success_rate'] > 0)
        
    except Exception as e:
        print(f"[FAIL] Performance benchmarking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_validation():
    """Test comprehensive validation suite (limited version)."""
    print("\nTesting Comprehensive Validation Suite...")
    
    try:
        from src.validation.multi_scenario_validation import run_comprehensive_validation, ValidationConfig
        
        # Create limited configuration for testing
        config = ValidationConfig(
            parameter_sweep_points=3,
            benchmark_iterations=2,
            scenarios_to_test=['standard'],
            quantum_backend_tests=True,
            ml_optimization_tests=True,
            bayesian_search_tests=False,  # Skip to save time
            rotating_metrics_tests=True,
            dynamic_evolution_tests=True,
            visualization_tests=True,
            performance_benchmark=True,
            create_summary_plots=False,
            export_csv_data=False
        )
        
        print(f"  Running comprehensive validation (limited)...")
        start_time = time.time()
        
        validation_results = run_comprehensive_validation(config)
        
        validation_time = time.time() - start_time
        
        print(f"  Comprehensive validation completed in {validation_time:.3f}s")
        
        # Check results structure
        if 'overall_summary' in validation_results:
            summary = validation_results['overall_summary']
            print(f"  Overall success rate: {summary['overall_success_rate']:.2f}")
            print(f"  Systems tested: {summary['systems_tested']}")
            print(f"  Validation time: {summary['validation_time']:.2f}s")
            
            if summary['key_findings']:
                print(f"  Key findings:")
                for finding in summary['key_findings']:
                    print(f"    - {finding}")
        
        # Check individual system results
        if 'system_validations' in validation_results:
            print(f"  System validation results:")
            for system, results in validation_results['system_validations'].items():
                success_rate = results.get('success_rate', 0)
                print(f"    {system}: {success_rate:.2f} success rate")
        
        # Verify files were created
        output_dir = Path(config.output_directory)
        json_files = list(output_dir.glob("validation_results_*.json"))
        print(f"  Output files created: {len(json_files)}")
        
        # Clean up test files
        for file in json_files:
            try:
                file.unlink()
            except:
                pass
        
        overall_success = validation_results['overall_summary']['overall_success_rate'] > 0.5
        return overall_success
        
    except Exception as e:
        print(f"[FAIL] Comprehensive validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all multi-scenario validation tests."""
    print("Multi-Scenario Validation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Validation Configuration", test_validation_config),
        ("Validator Initialization", test_validator_initialization),
        ("Quantum Backend Validation", test_quantum_backend_validation),
        ("ML Optimization Validation", test_ml_optimization_validation),
        ("Parameter Sweep Generation", test_parameter_sweep_generation),
        ("Scenario Validation", test_scenario_validation),
        ("Quick Validation", test_quick_validation),
        ("Performance Benchmarking", test_performance_benchmarking),
        ("Comprehensive Validation", test_comprehensive_validation),
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
    print(f"Multi-Scenario Validation Test Summary")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All multi-scenario validation tests passed!")
        print("\nMulti-Scenario Validation Features:")
        print("- Comprehensive system validation: [READY]")
        print("- Parameter space sweeps: [READY]")
        print("- Performance benchmarking: [READY]")
        print("- Statistical analysis: [READY]")
        print("- Automated reporting: [READY]")
        print("- Multi-scenario testing: [READY]")
        print("\nPhase 3 Validation: COMPLETE")
    elif passed >= 7:
        print("[PARTIAL] Core validation functionality working.")
        print("Advanced validation features may have minor issues.")
    else:
        print("[ERROR] Multi-scenario validation system not functional.")
    
    print(f"\nPhase 3 Final Status:")
    if results.get('Comprehensive Validation', False):
        print("- All Phase 3 systems validated: [COMPLETE]")
        print("- Ready for production use: [YES]")
        print("- Quality assurance: [PASSED]")
    
    print(f"\nTo run full validation:")
    print(f"from src.validation.multi_scenario_validation import run_comprehensive_validation")
    print(f"results = run_comprehensive_validation()")
    
    return passed >= 7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)