#!/usr/bin/env python3
"""
Quantum Field Analysis Example

This example demonstrates advanced quantum field analysis in curved spacetime,
focusing on vacuum fluctuations, field propagation through wormholes, and
quantum field theory calculations.

Topics covered:
- Quantum field theory in curved spacetime  
- Vacuum fluctuation analysis
- Field propagation through wormholes
- Hawking radiation calculations
- Unruh effect demonstration
- Field correlation functions
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as special
from scipy import integrate
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import CasimirExoticMatter


class QuantumFieldAnalysis:
    """Advanced quantum field analysis in wormhole spacetime."""
    
    def __init__(self, framework):
        """Initialize field analysis with simulation framework."""
        self.framework = framework
        self.field_modes = []
        self.vacuum_fluctuations = []
        self.correlation_functions = []
        
    def analyze_vacuum_fluctuations(self, grid_size=50, max_frequency=1000.0):
        """Analyze quantum vacuum fluctuations in wormhole geometry."""
        
        print("🌊 Analyzing quantum vacuum fluctuations...")
        
        # Create spatial grid around wormhole
        r_min, r_max = 0.1, 10.0
        r_grid = np.linspace(r_min, r_max, grid_size)
        
        # Frequency modes for field expansion
        frequencies = np.linspace(0.1, max_frequency, 100)
        
        vacuum_energy_density = []
        field_variance = []
        
        for r in r_grid:
            # Calculate vacuum energy density at position r
            energy_density = 0.0
            variance = 0.0
            
            for omega in frequencies:
                # Mode functions in wormhole geometry (simplified)
                # Using modified Bessel functions for wormhole geometry
                k = omega  # Natural units where c = 1
                
                # Effective potential in wormhole
                b0 = 1000.0  # Throat radius
                V_eff = self._effective_potential(r, b0)
                
                # Modified dispersion relation
                k_eff = np.sqrt(max(0, k**2 - V_eff))
                
                # Vacuum contribution (zero-point energy)
                energy_contribution = 0.5 * omega * np.exp(-k_eff * abs(r - b0))
                energy_density += energy_contribution
                
                # Field variance (quantum fluctuations)
                variance += (1.0 / (2.0 * k_eff)) * np.exp(-2 * k_eff * abs(r - b0))
            
            vacuum_energy_density.append(energy_density / len(frequencies))
            field_variance.append(variance)
        
        self.vacuum_fluctuations = {
            'positions': r_grid,
            'energy_density': vacuum_energy_density,
            'field_variance': field_variance,
            'frequencies': frequencies
        }
        
        print(f"   ✓ Calculated vacuum fluctuations at {len(r_grid)} positions")
        return self.vacuum_fluctuations
    
    def _effective_potential(self, r, b0):
        """Calculate effective potential for field modes in wormhole geometry."""
        # Simplified potential for Morris-Thorne wormhole
        if r < b0:
            return 0.0  # No potential at throat
        else:
            # Potential due to curvature
            return 1.0 / (r**2 + b0**2)
    
    def calculate_field_propagation(self, initial_field, time_steps=100):
        """Calculate quantum field propagation through wormhole."""
        
        print("📡 Calculating field propagation through wormhole...")
        
        # Spatial grid
        r_grid = np.linspace(-10, 10, 100)  # Extend through both sides
        dt = 0.01
        
        # Initialize field on one side of wormhole
        field = np.zeros((time_steps, len(r_grid)), dtype=complex)
        
        # Gaussian wave packet initial condition
        r0 = -5.0  # Start on one side
        sigma = 1.0
        k0 = 10.0  # Initial momentum
        
        field[0] = initial_field * np.exp(-(r_grid - r0)**2 / (2*sigma**2)) * \
                   np.exp(1j * k0 * r_grid)
        
        # Propagate field using finite difference method
        for t in range(1, time_steps):
            field[t] = self._propagate_field_step(field[t-1], r_grid, dt)
        
        self.field_propagation = {
            'field_evolution': field,
            'positions': r_grid,
            'times': np.arange(time_steps) * dt
        }
        
        print(f"   ✓ Propagated field for {time_steps} time steps")
        return self.field_propagation
    
    def _propagate_field_step(self, current_field, r_grid, dt):
        """Single step of field propagation using Schrödinger equation."""
        dr = r_grid[1] - r_grid[0]
        
        # Kinetic energy operator (second derivative)
        kinetic = np.zeros_like(current_field)
        kinetic[1:-1] = (current_field[2:] - 2*current_field[1:-1] + current_field[:-2]) / dr**2
        
        # Potential energy (wormhole geometry effects)
        potential = np.zeros_like(current_field)
        b0 = 1000.0  # Throat radius (scaled)
        for i, r in enumerate(r_grid):
            potential[i] = self._effective_potential(abs(r), b0) * current_field[i]
        
        # Time evolution using Schrödinger equation
        # ψ(t+dt) = ψ(t) - i*dt*H*ψ(t)
        hamiltonian = -0.5 * kinetic + potential
        next_field = current_field - 1j * dt * hamiltonian
        
        return next_field
    
    def analyze_hawking_radiation(self, temperature_scale=1e-7):
        """Analyze Hawking radiation from wormhole horizon effects."""
        
        print("🔥 Analyzing Hawking radiation effects...")
        
        # Frequency range for thermal spectrum
        frequencies = np.logspace(-3, 3, 1000)
        
        # Hawking temperature (scaled for wormhole)
        T_hawking = temperature_scale  # Kelvin
        k_B = 1.38e-23  # Boltzmann constant
        h_bar = 1.05e-34  # Reduced Planck constant
        
        # Thermal distribution
        beta = 1.0 / (k_B * T_hawking)
        thermal_spectrum = []
        
        for omega in frequencies:
            # Planck distribution for Hawking radiation
            if omega > 0:
                n_thermal = 1.0 / (np.exp(h_bar * omega * beta) - 1.0)
            else:
                n_thermal = 0.0
            thermal_spectrum.append(n_thermal)
        
        # Calculate total radiated power
        stefan_boltzmann = 5.67e-8  # Stefan-Boltzmann constant
        total_power = stefan_boltzmann * T_hawking**4  # Simplified
        
        self.hawking_analysis = {
            'frequencies': frequencies,
            'thermal_spectrum': np.array(thermal_spectrum),
            'temperature': T_hawking,
            'total_power': total_power
        }
        
        print(f"   ✓ Hawking temperature: {T_hawking:.2e} K")
        print(f"   ✓ Total radiated power: {total_power:.2e} W")
        
        return self.hawking_analysis
    
    def calculate_unruh_effect(self, accelerations):
        """Calculate Unruh effect for accelerated observers near wormhole."""
        
        print("🚀 Calculating Unruh effect for accelerated observers...")
        
        unruh_temperatures = []
        particle_creation_rates = []
        
        c = 3e8  # Speed of light
        k_B = 1.38e-23  # Boltzmann constant
        h_bar = 1.05e-34  # Reduced Planck constant
        
        for a in accelerations:
            # Unruh temperature
            T_unruh = h_bar * a / (2 * np.pi * k_B * c)
            unruh_temperatures.append(T_unruh)
            
            # Particle creation rate (simplified)
            # Rate proportional to acceleration squared
            creation_rate = (a**2) / (12 * np.pi**2 * c**3) * (h_bar * c)
            particle_creation_rates.append(creation_rate)
        
        self.unruh_analysis = {
            'accelerations': np.array(accelerations),
            'temperatures': np.array(unruh_temperatures),
            'creation_rates': np.array(particle_creation_rates)
        }
        
        print(f"   ✓ Calculated Unruh effect for {len(accelerations)} acceleration values")
        return self.unruh_analysis
    
    def compute_correlation_functions(self, separation_distances):
        """Compute quantum field correlation functions."""
        
        print("📊 Computing quantum field correlation functions...")
        
        correlations = []
        vacuum_correlations = []
        
        for distance in separation_distances:
            # Two-point correlation function in vacuum
            # <0|φ(x)φ(x+r)|0> for wormhole geometry
            
            # Simplified correlation function
            # Falls off as 1/r in flat space, modified by wormhole geometry
            b0 = 1000.0  # Throat radius
            
            if distance == 0:
                correlation = float('inf')  # Divergent at zero separation
                vacuum_corr = 1.0
            else:
                # Modified by wormhole geometry
                geometry_factor = 1.0 + b0**2 / (distance**2 + b0**2)
                correlation = geometry_factor / distance
                
                # Vacuum correlation (regularized)
                vacuum_corr = np.exp(-distance / (2 * b0)) / np.sqrt(distance + b0)
            
            correlations.append(correlation)
            vacuum_correlations.append(vacuum_corr)
        
        self.correlation_analysis = {
            'distances': np.array(separation_distances),
            'field_correlations': np.array(correlations),
            'vacuum_correlations': np.array(vacuum_correlations)
        }
        
        print(f"   ✓ Computed correlations for {len(separation_distances)} separations")
        return self.correlation_analysis
    
    def generate_comprehensive_report(self):
        """Generate comprehensive quantum field analysis report."""
        
        report = {
            'timestamp': np.datetime64('now'),
            'analysis_type': 'quantum_field_analysis',
            'summary': {},
            'detailed_results': {},
            'physical_interpretation': {},
            'recommendations': []
        }
        
        # Vacuum fluctuation analysis
        if hasattr(self, 'vacuum_fluctuations'):
            vf = self.vacuum_fluctuations
            max_energy = np.max(vf['energy_density'])
            avg_variance = np.mean(vf['field_variance'])
            
            report['summary']['vacuum_energy_max'] = max_energy
            report['summary']['average_field_variance'] = avg_variance
            report['detailed_results']['vacuum_fluctuations'] = vf
            
        # Hawking radiation analysis
        if hasattr(self, 'hawking_analysis'):
            ha = self.hawking_analysis
            report['summary']['hawking_temperature'] = ha['temperature']
            report['summary']['hawking_power'] = ha['total_power']
            report['detailed_results']['hawking_radiation'] = ha
            
        # Unruh effect analysis
        if hasattr(self, 'unruh_analysis'):
            ua = self.unruh_analysis
            max_unruh_temp = np.max(ua['temperatures'])
            report['summary']['max_unruh_temperature'] = max_unruh_temp
            report['detailed_results']['unruh_effect'] = ua
        
        # Physical interpretation
        report['physical_interpretation'] = {
            'vacuum_state': 'Quantum vacuum exhibits enhanced fluctuations near wormhole throat',
            'field_propagation': 'Field modes can traverse wormhole with geometric modifications',
            'thermal_effects': 'Hawking and Unruh effects create thermal radiation signatures',
            'correlations': 'Quantum correlations modified by non-trivial spacetime topology'
        }
        
        # Recommendations
        report['recommendations'] = [
            'Monitor vacuum energy density for stability indicators',
            'Consider thermal effects in wormhole traversal calculations',
            'Account for field correlation modifications in quantum information protocols',
            'Investigate accelerated motion effects near wormhole regions'
        ]
        
        return report


def create_field_visualizations(analysis):
    """Create comprehensive visualizations of quantum field analysis."""
    
    print("📈 Creating quantum field visualizations...")
    
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # 1. Vacuum fluctuation plot
    if hasattr(analysis, 'vacuum_fluctuations'):
        vf = analysis.vacuum_fluctuations
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Energy density
        ax1.semilogy(vf['positions'], vf['energy_density'], 'b-', linewidth=2, label='Vacuum Energy Density')
        ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Throat Position')
        ax1.set_xlabel('Position (normalized units)')
        ax1.set_ylabel('Energy Density (arbitrary units)')
        ax1.set_title('Quantum Vacuum Energy Density in Wormhole')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Field variance
        ax2.plot(vf['positions'], vf['field_variance'], 'g-', linewidth=2, label='Field Variance')
        ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Throat Position')
        ax2.set_xlabel('Position (normalized units)')
        ax2.set_ylabel('Field Variance')
        ax2.set_title('Quantum Field Fluctuations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('examples/output/vacuum_fluctuations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Field propagation animation data
    if hasattr(analysis, 'field_propagation'):
        fp = analysis.field_propagation
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Field intensity evolution
        intensity = np.abs(fp['field_evolution'])**2
        im1 = ax1.imshow(intensity, extent=[fp['positions'][0], fp['positions'][-1], 
                                           0, fp['times'][-1]], aspect='auto', cmap='hot')
        ax1.axvline(x=0, color='white', linestyle='--', alpha=0.7, label='Wormhole Center')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Time')
        ax1.set_title('Quantum Field Intensity Evolution')
        plt.colorbar(im1, ax=ax1, label='Field Intensity')
        
        # Phase evolution
        phase = np.angle(fp['field_evolution'])
        im2 = ax2.imshow(phase, extent=[fp['positions'][0], fp['positions'][-1], 
                                       0, fp['times'][-1]], aspect='auto', cmap='hsv')
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7, label='Wormhole Center')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Time')
        ax2.set_title('Quantum Field Phase Evolution')
        plt.colorbar(im2, ax=ax2, label='Phase (radians)')
        
        plt.tight_layout()
        plt.savefig('examples/output/field_propagation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Hawking radiation spectrum
    if hasattr(analysis, 'hawking_analysis'):
        ha = analysis.hawking_analysis
        
        plt.figure(figsize=(10, 6))
        plt.loglog(ha['frequencies'], ha['thermal_spectrum'], 'r-', linewidth=2, 
                  label=f'Hawking Spectrum (T = {ha["temperature"]:.2e} K)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Thermal Occupation Number')
        plt.title('Hawking Radiation Spectrum from Wormhole')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('examples/output/hawking_spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Unruh effect
    if hasattr(analysis, 'unruh_analysis'):
        ua = analysis.unruh_analysis
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Unruh temperatures
        ax1.loglog(ua['accelerations'], ua['temperatures'], 'b-', linewidth=2, 
                  marker='o', markersize=6, label='Unruh Temperature')
        ax1.set_xlabel('Acceleration (m/s²)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Unruh Effect: Temperature vs Acceleration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Particle creation rates
        ax2.loglog(ua['accelerations'], ua['creation_rates'], 'g-', linewidth=2,
                  marker='s', markersize=6, label='Creation Rate')
        ax2.set_xlabel('Acceleration (m/s²)')
        ax2.set_ylabel('Particle Creation Rate')
        ax2.set_title('Unruh Effect: Particle Creation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('examples/output/unruh_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Correlation functions
    if hasattr(analysis, 'correlation_analysis'):
        ca = analysis.correlation_analysis
        
        plt.figure(figsize=(10, 6))
        plt.loglog(ca['distances'], ca['field_correlations'], 'purple', linewidth=2,
                  label='Field Correlations', marker='o')
        plt.loglog(ca['distances'], ca['vacuum_correlations'], 'orange', linewidth=2,
                  label='Vacuum Correlations', marker='s')
        plt.xlabel('Separation Distance (normalized units)')
        plt.ylabel('Correlation Function Value')
        plt.title('Quantum Field Correlation Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('examples/output/correlation_functions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("   ✓ All quantum field visualizations saved")


def main():
    """Run quantum field analysis example."""
    
    print("⚛️  Quantum Field Analysis in Wormhole Spacetime")
    print("=" * 60)
    
    # Step 1: Initialize framework
    print("\n1. Initializing simulation framework...")
    
    config = IntegrationConfig(
        simulation_name="quantum_field_analysis",
        time_steps=200,
        dt=0.05,
        num_qubits=6,
        enable_stability_prediction=True,
        enable_real_time_visualization=False
    )
    
    framework = WormholeSimulationFramework(config)
    
    # Initialize with field-analysis optimized parameters
    wormhole_params = {
        'b0': 1000.0,  # 1 km throat radius
        'mass': 1e30,  # Solar mass
        'casimir_energy': -1e15
    }
    
    quantum_params = {
        'num_qubits': 6,
        'traversal_probability': 0.9,
        'entanglement_strength': 1.0,
        'decoherence_rate': 0.005
    }
    
    ai_params = {
        'stability_threshold': 0.6,
        'optimization_target': 'stability'
    }
    
    framework.initialize_system(
        wormhole_params=wormhole_params,
        quantum_params=quantum_params,
        ai_params=ai_params
    )
    
    print("   ✓ Framework initialized for quantum field analysis")
    
    # Step 2: Create field analysis object
    print("\n2. Setting up quantum field analysis...")
    
    field_analysis = QuantumFieldAnalysis(framework)
    
    # Step 3: Analyze vacuum fluctuations
    print("\n3. Analyzing vacuum fluctuations...")
    
    vacuum_results = field_analysis.analyze_vacuum_fluctuations(grid_size=100, max_frequency=1000.0)
    
    # Step 4: Calculate field propagation
    print("\n4. Calculating field propagation...")
    
    initial_amplitude = 1.0
    propagation_results = field_analysis.calculate_field_propagation(
        initial_field=initial_amplitude, 
        time_steps=150
    )
    
    # Step 5: Analyze Hawking radiation
    print("\n5. Analyzing Hawking radiation...")
    
    hawking_results = field_analysis.analyze_hawking_radiation(temperature_scale=1e-7)
    
    # Step 6: Calculate Unruh effect
    print("\n6. Calculating Unruh effect...")
    
    accelerations = np.logspace(1, 20, 50)  # Range of accelerations
    unruh_results = field_analysis.calculate_unruh_effect(accelerations)
    
    # Step 7: Compute correlation functions  
    print("\n7. Computing correlation functions...")
    
    distances = np.logspace(-1, 2, 100)  # Separation distances
    correlation_results = field_analysis.compute_correlation_functions(distances)
    
    # Step 8: Generate comprehensive report
    print("\n8. Generating comprehensive analysis report...")
    
    report = field_analysis.generate_comprehensive_report()
    
    # Display key results
    print("\n📊 Key Results:")
    if 'vacuum_energy_max' in report['summary']:
        print(f"   Max vacuum energy density: {report['summary']['vacuum_energy_max']:.2e}")
    if 'hawking_temperature' in report['summary']:
        print(f"   Hawking temperature: {report['summary']['hawking_temperature']:.2e} K")
    if 'max_unruh_temperature' in report['summary']:
        print(f"   Max Unruh temperature: {report['summary']['max_unruh_temperature']:.2e} K")
    
    # Step 9: Create visualizations
    print("\n9. Creating quantum field visualizations...")
    
    create_field_visualizations(field_analysis)
    
    # Step 10: Display physical interpretation
    print("\n10. Physical Interpretation:")
    for key, interpretation in report['physical_interpretation'].items():
        print(f"    • {key}: {interpretation}")
    
    print("\n🎯 Recommendations:")
    for i, recommendation in enumerate(report['recommendations'], 1):
        print(f"    {i}. {recommendation}")
    
    print(f"\n🎉 Quantum field analysis completed successfully!")
    print(f"   Check 'examples/output/' for detailed visualizations")
    print(f"   Analysis covered: vacuum fluctuations, field propagation,")
    print(f"   Hawking radiation, Unruh effect, and correlation functions")
    
    return field_analysis, report


if __name__ == "__main__":
    try:
        analysis, report = main()
        print(f"\n📈 Analysis Summary:")
        print(f"   - Vacuum fluctuation analysis: Complete")
        print(f"   - Field propagation: Complete") 
        print(f"   - Hawking radiation: Complete")
        print(f"   - Unruh effect analysis: Complete")
        print(f"   - Correlation functions: Complete")
        
    except Exception as e:
        print(f"\n❌ Error in quantum field analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)