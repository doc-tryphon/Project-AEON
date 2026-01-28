#!/usr/bin/env python3
"""
Comprehensive Exploration Demo

This script demonstrates the complete scientific exploration capabilities
of the quantum wormhole simulation framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

from src.exploration_interface import launch_exploration_interface, ExplorationConfig


def main():
    """Run comprehensive exploration demonstration."""
    
    print("="*70)
    print("QUANTUM WORMHOLE COMPREHENSIVE EXPLORATION DEMO")
    print("="*70)
    
    # Configuration for exploration
    config = ExplorationConfig(
        workspace_dir="exploration_demo_results",
        session_name="comprehensive_demo",
        parameter_sweep_points=15,
        parallel_processing=True,
        enable_3d_visualization=True,
        enable_animations=True
    )
    
    # Launch exploration interface
    print("\n1. Initializing Scientific Exploration Interface...")
    interface = launch_exploration_interface(config)
    
    # Set initial parameters for exploration
    print("\n2. Setting initial parameters...")
    interface.set_parameters(
        throat_radius=2e3,
        mass=5e29,
        num_qubits=4,
        exotic_matter_density=-5e15,
        traversal_probability=0.8,
        entanglement_strength=1.2,
        decoherence_rate=0.005
    )
    
    print(f"Current parameters: {interface.current_parameters}")
    
    # Create initial comprehensive visualization
    print("\n3. Creating comprehensive visualization...")
    try:
        main_dashboard = interface.create_comprehensive_visualization()
        main_dashboard.write_html("comprehensive_dashboard.html")
        print("   ✓ Main dashboard saved as 'comprehensive_dashboard.html'")
    except Exception as e:
        print(f"   ⚠ Dashboard creation failed: {e}")
    
    # Parameter sweep analysis
    print("\n4. Running parameter sweep analyses...")
    
    # Sweep throat radius
    print("   4a. Sweeping throat radius...")
    try:
        throat_sweep = interface.run_parameter_sweep(
            'throat_radius', 
            value_range=(5e2, 5e3),
            num_points=12
        )
        
        # Visualize sweep results
        throat_fig = interface.visualize_parameter_sweep(throat_sweep)
        throat_fig.write_html("throat_radius_sweep.html")
        print("   ✓ Throat radius sweep completed and saved")
        
        # Find optimal throat radius
        optimal_idx = np.argmax(throat_sweep['stability_scores'])
        optimal_throat = throat_sweep['parameter_values'][optimal_idx]
        print(f"   → Optimal throat radius: {optimal_throat:.2e} (stability: {throat_sweep['stability_scores'][optimal_idx]:.3f})")
        
    except Exception as e:
        print(f"   ⚠ Throat radius sweep failed: {e}")
    
    # Sweep number of qubits
    print("   4b. Sweeping number of qubits...")
    try:
        qubit_sweep = interface.run_parameter_sweep(
            'num_qubits',
            value_range=(2, 8),
            num_points=7
        )
        
        qubit_fig = interface.visualize_parameter_sweep(qubit_sweep)
        qubit_fig.write_html("num_qubits_sweep.html")
        print("   ✓ Qubit sweep completed and saved")
        
        # Find optimal qubit count
        optimal_idx = np.argmax(qubit_sweep['stability_scores'])
        optimal_qubits = int(qubit_sweep['parameter_values'][optimal_idx])
        print(f"   → Optimal qubit count: {optimal_qubits} (stability: {qubit_sweep['stability_scores'][optimal_idx]:.3f})")
        
    except Exception as e:
        print(f"   ⚠ Qubit sweep failed: {e}")
    
    # Sweep exotic matter density
    print("   4c. Sweeping exotic matter density...")
    try:
        exotic_sweep = interface.run_parameter_sweep(
            'exotic_matter_density',
            value_range=(-1e16, -1e14),
            num_points=10
        )
        
        exotic_fig = interface.visualize_parameter_sweep(exotic_sweep)
        exotic_fig.write_html("exotic_matter_sweep.html")
        print("   ✓ Exotic matter sweep completed and saved")
        
    except Exception as e:
        print(f"   ⚠ Exotic matter sweep failed: {e}")
    
    # 2D Stability landscape analysis
    print("\n5. Creating stability landscapes...")
    
    # Throat radius vs mass landscape
    print("   5a. Throat radius vs mass landscape...")
    try:
        landscape1 = interface.create_stability_landscape(
            'throat_radius', 'mass', resolution=10
        )
        landscape1.write_html("stability_landscape_throat_mass.html")
        
        # Extract optimal parameters
        landscape_key = 'stability_landscape_throat_radius_mass'
        if landscape_key in interface.analysis_results:
            optimal_params = interface.analysis_results[landscape_key]['optimal_params']
            max_stability = interface.analysis_results[landscape_key]['max_stability']
            print(f"   ✓ Landscape completed. Max stability: {max_stability:.3f}")
            print(f"   → Optimal throat radius: {optimal_params['throat_radius']:.2e}")
            print(f"   → Optimal mass: {optimal_params['mass']:.2e}")
        
    except Exception as e:
        print(f"   ⚠ Landscape creation failed: {e}")
    
    # Qubits vs entanglement landscape
    print("   5b. Qubits vs entanglement strength landscape...")
    try:
        interface.config.parameter_ranges['entanglement_strength'] = (0.1, 2.0)
        landscape2 = interface.create_stability_landscape(
            'num_qubits', 'entanglement_strength', resolution=8
        )
        landscape2.write_html("stability_landscape_qubits_entanglement.html")
        print("   ✓ Qubits-entanglement landscape completed")
        
    except Exception as e:
        print(f"   ⚠ Second landscape creation failed: {e}")
    
    # Parameter optimization
    print("\n6. Running parameter optimization...")
    
    # Optimize for stability
    print("   6a. Optimizing for stability...")
    try:
        stability_opt = interface.run_optimization(
            target_metric='stability',
            max_iterations=50
        )
        
        print(f"   ✓ Stability optimization completed")
        print(f"   → Optimal stability score: {stability_opt['optimal_score']:.4f}")
        print(f"   → Optimal throat radius: {stability_opt['optimal_parameters']['throat_radius']:.2e}")
        print(f"   → Optimal mass: {stability_opt['optimal_parameters']['mass']:.2e}")
        print(f"   → Iterations: {stability_opt['iterations']}")
        
    except Exception as e:
        print(f"   ⚠ Stability optimization failed: {e}")
    
    # Optimize for entanglement
    print("   6b. Optimizing for entanglement...")
    try:
        entanglement_opt = interface.run_optimization(
            target_metric='entanglement',
            max_iterations=30
        )
        
        print(f"   ✓ Entanglement optimization completed")
        print(f"   → Optimal entanglement measure: {entanglement_opt['optimal_score']:.4f}")
        
    except Exception as e:
        print(f"   ⚠ Entanglement optimization failed: {e}")
    
    # Create exploration summary
    print("\n7. Creating exploration summary...")
    try:
        summary_fig = interface.create_exploration_summary()
        summary_fig.write_html("exploration_summary.html")
        print("   ✓ Exploration summary created and saved")
    except Exception as e:
        print(f"   ⚠ Summary creation failed: {e}")
    
    # Generate comprehensive report
    print("\n8. Generating comprehensive reports...")
    
    try:
        # JSON export
        interface.export_results("comprehensive_exploration_results", "json")
        print("   ✓ JSON results exported")
        
        # CSV export
        interface.export_results("comprehensive_exploration_data", "csv")
        print("   ✓ CSV data exported")
        
        # HTML report
        interface.export_results("comprehensive_exploration_report", "html")
        print("   ✓ HTML report generated")
        
    except Exception as e:
        print(f"   ⚠ Report generation failed: {e}")
    
    # Final analysis summary
    print("\n" + "="*70)
    print("EXPLORATION SUMMARY")
    print("="*70)
    
    # Print key findings
    total_analyses = len(interface.analysis_results)
    print(f"Total analyses completed: {total_analyses}")
    
    # Best stability scores
    all_stability_scores = []
    for key, results in interface.analysis_results.items():
        if isinstance(results, dict):
            if 'stability_scores' in results:
                all_stability_scores.extend(results['stability_scores'])
            elif 'optimal_score' in results:
                all_stability_scores.append(results['optimal_score'])
            elif 'max_stability' in results:
                all_stability_scores.append(results['max_stability'])
    
    if all_stability_scores:
        max_stability = max(all_stability_scores)
        avg_stability = np.mean(all_stability_scores)
        print(f"Maximum stability achieved: {max_stability:.4f}")
        print(f"Average stability across all analyses: {avg_stability:.4f}")
    
    # Current optimal parameters
    print(f"\nCurrent optimal parameters:")
    for key, value in interface.current_parameters.items():
        print(f"  {key}: {value}")
    
    # Files generated
    print(f"\nFiles generated in current directory:")
    html_files = list(Path('.').glob('*.html'))
    for html_file in html_files:
        print(f"  • {html_file.name}")
    
    workspace_files = list(Path(config.workspace_dir).glob('*'))
    if workspace_files:
        print(f"\nFiles in workspace ({config.workspace_dir}):")
        for file in workspace_files:
            print(f"  • {file.name}")
    
    print(f"\n" + "="*70)
    print("COMPREHENSIVE EXPLORATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nTo view results:")
    print("1. Open 'comprehensive_dashboard.html' for interactive dashboard")
    print("2. Open 'exploration_summary.html' for analysis summary") 
    print("3. Open 'comprehensive_exploration_report.html' for detailed report")
    print("4. Review parameter sweep visualizations (*.html files)")
    print("\nAll results have been saved for further analysis.")


if __name__ == "__main__":
    main()