#!/usr/bin/env python3
"""
Basic Wormhole Simulation Example

This example demonstrates the basic usage of the quantum wormhole simulation
framework with a simple Morris-Thorne wormhole configuration.

Topics covered:
- Basic framework initialization
- Morris-Thorne wormhole setup
- Simple simulation execution
- Results analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import CasimirExoticMatter


def main():
    """Run basic wormhole simulation example."""
    
    print("🌌 Basic Wormhole Simulation Example")
    print("=" * 50)
    
    # Step 1: Create configuration
    print("\n1. Setting up simulation configuration...")
    
    config = IntegrationConfig(
        simulation_name="basic_wormhole_example",
        time_steps=100,  # Short simulation for demonstration
        dt=0.1,
        num_qubits=4,
        enable_stability_prediction=True,
        enable_real_time_visualization=False  # Disable for simplicity
    )
    
    print(f"   Simulation: {config.simulation_name}")
    print(f"   Time steps: {config.time_steps}")
    print(f"   Qubits: {config.num_qubits}")
    
    # Step 2: Initialize framework
    print("\n2. Initializing simulation framework...")
    
    framework = WormholeSimulationFramework(config)
    
    # Step 3: Configure wormhole parameters
    print("\n3. Configuring wormhole parameters...")
    
    wormhole_params = {
        'b0': 1000.0,           # 1 km throat radius
        'mass': 1e30,           # Solar mass
        'casimir_energy': -1e15  # Casimir energy scale
    }
    
    quantum_params = {
        'num_qubits': 4,
        'traversal_probability': 0.8,
        'entanglement_strength': 1.0,
        'decoherence_rate': 0.01
    }
    
    ai_params = {
        'stability_threshold': 0.5,
        'optimization_target': 'stability'
    }
    
    print(f"   Throat radius: {wormhole_params['b0']} m")
    print(f"   Mass: {wormhole_params['mass']:.1e} kg")
    print(f"   Traversal probability: {quantum_params['traversal_probability']}")
    
    # Step 4: Initialize system
    print("\n4. Initializing physics and quantum systems...")
    
    framework.initialize_system(
        wormhole_params=wormhole_params,
        quantum_params=quantum_params,
        ai_params=ai_params
    )
    
    print("   ✓ Physics engine initialized")
    print("   ✓ Quantum system initialized") 
    print("   ✓ AI components initialized")
    
    # Step 5: Run simulation
    print("\n5. Running simulation...")
    
    def progress_callback(step, step_results):
        """Callback to show progress."""
        if step % 20 == 0:  # Print every 20 steps
            stability = step_results.get('ai', {}).get('stability_score', 0)
            print(f"   Step {step:3d}: Stability = {stability:.3f}")
    
    results = framework.run_simulation(callback=progress_callback)
    
    print("   ✓ Simulation completed successfully")
    
    # Step 6: Analyze results
    print("\n6. Analyzing results...")
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report()
    
    # Display key metrics
    print(f"   Total steps completed: {len(results.spacetime_evolution)}")
    
    if results.stability_predictions:
        avg_stability = np.mean(results.stability_predictions)
        min_stability = np.min(results.stability_predictions)
        max_stability = np.max(results.stability_predictions)
        
        print(f"   Average stability: {avg_stability:.3f}")
        print(f"   Stability range: {min_stability:.3f} - {max_stability:.3f}")
    
    if 'summary' in report:
        summary = report['summary']
        print(f"   Simulation successful: {summary.get('simulation_successful', 'Unknown')}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1%}")
    
    # Step 7: Create basic plots
    print("\n7. Creating visualization plots...")
    
    create_basic_plots(results, report)
    
    print("   ✓ Plots saved as PNG files")
    
    # Step 8: Display recommendations
    print("\n8. AI Recommendations:")
    
    if 'recommendations' in report:
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    else:
        print("   No specific recommendations at this time")
    
    print(f"\n🎉 Basic simulation example completed successfully!")
    print(f"   Check the generated plots for visualization results.")
    
    return results, report


def create_basic_plots(results, report):
    """Create basic visualization plots."""
    
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # Plot 1: Stability evolution
    if results.stability_predictions:
        plt.figure(figsize=(10, 6))
        steps = range(len(results.stability_predictions))
        plt.plot(steps, results.stability_predictions, 'b-', linewidth=2, label='Stability Score')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
        plt.xlabel('Simulation Step')
        plt.ylabel('Stability Score')
        plt.title('Wormhole Stability Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('examples/output/stability_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Physics metrics
    if results.spacetime_evolution:
        physics_data = results.spacetime_evolution
        
        # Extract energy densities
        energy_densities = []
        pressures = []
        steps = []
        
        for data in physics_data:
            if 'energy_density' in data and 'pressure' in data:
                steps.append(data.get('step', 0))
                energy_densities.append(data['energy_density'])
                pressures.append(data['pressure'])
        
        if energy_densities and pressures:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Energy density plot
            ax1.plot(steps, energy_densities, 'g-', linewidth=2)
            ax1.set_ylabel('Energy Density (J/m³)')
            ax1.set_title('Exotic Matter Properties')
            ax1.grid(True, alpha=0.3)
            ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Pressure plot
            ax2.plot(steps, pressures, 'r-', linewidth=2)
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Pressure (Pa)')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig('examples/output/physics_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 3: Quantum metrics
    if results.quantum_state_evolution:
        quantum_data = results.quantum_state_evolution
        
        concurrences = []
        entropies = []
        steps = []
        
        for data in quantum_data:
            if 'concurrence' in data and 'entropy' in data:
                steps.append(data.get('step', 0))
                concurrences.append(data['concurrence'])
                entropies.append(data['entropy'])
        
        if concurrences and entropies:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Concurrence plot
            ax1.plot(steps, concurrences, 'b-', linewidth=2, label='Concurrence')
            ax1.set_ylabel('Concurrence')
            ax1.set_title('Quantum Entanglement Measures')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Entropy plot
            ax2.plot(steps, entropies, 'purple', linewidth=2, label='Entanglement Entropy')
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Entropy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('examples/output/quantum_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 4: Summary dashboard
    if 'summary' in report:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Stability histogram
        if results.stability_predictions:
            ax1.hist(results.stability_predictions, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Stability Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Stability Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        if 'performance_analysis' in report:
            perf = report['performance_analysis']
            if 'timing' in perf:
                timing = perf['timing']
                metrics = list(timing.keys())
                values = list(timing.values())
                
                ax2.bar(metrics, values, color='green', alpha=0.7)
                ax2.set_ylabel('Time (seconds)')
                ax2.set_title('Performance Metrics')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
        
        # Physics analysis
        if 'physics_analysis' in report and 'energy_statistics' in report['physics_analysis']:
            energy_stats = report['physics_analysis']['energy_statistics']
            stats_names = ['Mean', 'Std', 'Min', 'Max']
            stats_values = [
                energy_stats.get('mean', 0),
                energy_stats.get('std', 0), 
                energy_stats.get('min', 0),
                energy_stats.get('max', 0)
            ]
            
            ax3.bar(stats_names, stats_values, color='orange', alpha=0.7)
            ax3.set_ylabel('Energy Density (J/m³)')
            ax3.set_title('Energy Statistics')
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax3.grid(True, alpha=0.3)
        
        # Quantum analysis
        if 'quantum_analysis' in report and 'entanglement_statistics' in report['quantum_analysis']:
            ent_stats = report['quantum_analysis']['entanglement_statistics']
            
            categories = ['Max Concurrence', 'Avg Concurrence', 'Persistence']
            values = [
                ent_stats.get('max_concurrence', 0),
                ent_stats.get('avg_concurrence', 0),
                ent_stats.get('entanglement_persistence', 0)
            ]
            
            ax4.bar(categories, values, color='purple', alpha=0.7)
            ax4.set_ylabel('Measure Value')
            ax4.set_title('Quantum Statistics')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Simulation Summary Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('examples/output/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    try:
        results, report = main()
        print(f"\n📊 Results summary:")
        print(f"   - Physics evolution steps: {len(results.spacetime_evolution)}")
        print(f"   - Quantum evolution steps: {len(results.quantum_state_evolution)}")
        print(f"   - Stability predictions: {len(results.stability_predictions)}")
        
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)