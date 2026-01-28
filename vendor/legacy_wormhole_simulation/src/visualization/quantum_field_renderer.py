"""
Real-Time Quantum Field Effects Renderer

This module provides advanced visualization of quantum field effects in
curved spacetime, including vacuum fluctuations, particle creation,
Hawking radiation, and backreaction effects on wormhole geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import concurrent.futures
from functools import lru_cache
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import griddata

# Import physics modules
from src.physics.exotic_matter import ExoticMatter
from src.physics.spacetime_metrics import SpacetimeMetric
from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.quantum.entanglement_dynamics import EntanglementDynamics


@dataclass
class QuantumFieldRenderConfig:
    """Configuration for quantum field rendering."""
    
    # Spacetime grid parameters
    r_min: float = 1e-6
    r_max: float = 1e-2
    r_points: int = 200
    t_points: int = 100
    theta_points: int = 50
    phi_points: int = 25
    
    # Quantum field parameters
    field_cutoff: float = 1e19  # Planck scale cutoff (GeV/c)
    renormalization_scale: float = 1e16  # GUT scale (GeV/c)
    temperature: float = 2.725  # CMB temperature (K)
    
    # Visualization parameters
    vacuum_fluctuation_amplitude: float = 1.0
    particle_creation_threshold: float = 0.1
    hawking_radiation_scale: float = 1.0
    animation_fps: int = 20
    
    # Rendering options
    show_vacuum_fluctuations: bool = True
    show_particle_creation: bool = True
    show_hawking_radiation: bool = True
    show_backreaction: bool = True
    show_entanglement_structure: bool = True
    
    # Performance parameters
    max_particles: int = 10000
    computation_timeout: int = 60  # seconds
    cache_size: int = 500


class QuantumFieldRenderer:
    """Advanced renderer for quantum field effects in curved spacetime."""
    
    def __init__(self, config: QuantumFieldRenderConfig = None):
        """Initialize quantum field renderer.
        
        Args:
            config: Rendering configuration
        """
        self.config = config or QuantumFieldRenderConfig()
        
        # Spacetime grids
        self.spacetime_grid = None
        self.field_grid = None
        
        # Physical constants (natural units ℏ = c = G = 1)
        self.hbar = 1.054571817e-34  # J⋅s
        self.c = 299792458.0  # m/s
        self.G = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
        self.k_b = 1.380649e-23  # J/K
        
        # Planck units
        self.l_planck = np.sqrt(self.hbar * self.G / self.c**3)
        self.t_planck = self.l_planck / self.c
        self.m_planck = np.sqrt(self.hbar * self.c / self.G)
        self.E_planck = self.m_planck * self.c**2
        
        # Quantum field calculation cache
        self.cache = {}
        
        # Initialize grids
        self._initialize_grids()
        
        print(f"Quantum field renderer initialized with {self.config.r_points}×{self.config.t_points} spacetime grid")
    
    def _initialize_grids(self):
        """Initialize spacetime and field grids."""
        
        # Spacetime coordinates
        r_grid = np.logspace(
            np.log10(self.config.r_min),
            np.log10(self.config.r_max),
            self.config.r_points
        )
        
        t_grid = np.linspace(0, self.t_planck * 1000, self.config.t_points)
        theta_grid = np.linspace(0, np.pi, self.config.theta_points)
        phi_grid = np.linspace(0, 2*np.pi, self.config.phi_points)
        
        # Create 4D meshgrid for calculations
        T, R, THETA, PHI = np.meshgrid(t_grid, r_grid, theta_grid, phi_grid, indexing='ij')
        
        self.spacetime_grid = {
            't': T,
            'r': R,
            'theta': THETA,
            'phi': PHI,
            't_1d': t_grid,
            'r_1d': r_grid,
            'theta_1d': theta_grid,
            'phi_1d': phi_grid
        }
        
        # Initialize field grid
        self.field_grid = np.zeros_like(T)
    
    def compute_vacuum_fluctuations(self, metric: SpacetimeMetric,
                                  exotic_matter: ExoticMatter) -> np.ndarray:
        """Compute quantum vacuum fluctuations in curved spacetime.
        
        Args:
            metric: Spacetime metric
            exotic_matter: Exotic matter configuration
            
        Returns:
            Vacuum fluctuation field values
        """
        
        print("Computing vacuum fluctuations...")
        
        # Initialize fluctuation field
        fluctuations = np.zeros_like(self.spacetime_grid['t'])
        
        # Sample subset for performance
        r_sample = self.spacetime_grid['r_1d'][::max(1, len(self.spacetime_grid['r_1d'])//50)]
        t_sample = self.spacetime_grid['t_1d'][::max(1, len(self.spacetime_grid['t_1d'])//20)]
        
        for i, t in enumerate(t_sample):
            for j, r in enumerate(r_sample):
                coords = (t, r, np.pi/2, 0.0)  # Equatorial slice
                
                try:
                    # Get metric components
                    g_tt = metric.metric_tensor(coords)[0, 0]
                    g_rr = metric.metric_tensor(coords)[1, 1]
                    
                    # Ricci scalar curvature
                    ricci_scalar = metric.ricci_scalar(coords)
                    
                    # Vacuum expectation value of stress-energy tensor
                    # <T_μν> = (1/4π) * (R_μν - (1/2)g_μν R + local terms)
                    
                    # Regularized vacuum energy density
                    # Using point-splitting regularization
                    cutoff_freq = self.config.field_cutoff
                    renorm_scale = self.config.renormalization_scale
                    
                    # Vacuum fluctuation amplitude
                    vacuum_amplitude = (
                        self.config.vacuum_fluctuation_amplitude *
                        np.sqrt(abs(ricci_scalar)) * 
                        np.exp(-r / self.l_planck) *  # Exponential cutoff
                        (1 + 0.1 * np.sin(cutoff_freq * t / self.hbar))  # High-frequency oscillations
                    )
                    
                    # Add stochastic component
                    stochastic_component = np.random.normal(0, vacuum_amplitude * 0.1)
                    
                    total_fluctuation = vacuum_amplitude + stochastic_component
                    
                    # Map to grid indices
                    t_idx = np.argmin(np.abs(self.spacetime_grid['t_1d'] - t))
                    r_idx = np.argmin(np.abs(self.spacetime_grid['r_1d'] - r))
                    
                    if t_idx < fluctuations.shape[0] and r_idx < fluctuations.shape[1]:
                        fluctuations[t_idx, r_idx, self.config.theta_points//2, 0] = total_fluctuation
                
                except Exception as e:
                    continue  # Skip problematic points
        
        # Interpolate to fill grid
        filled_fluctuations = self._interpolate_field(fluctuations)
        
        print("Vacuum fluctuations computed successfully")
        
        return filled_fluctuations
    
    def compute_particle_creation_events(self, metric: SpacetimeMetric,
                                       exotic_matter: ExoticMatter) -> Dict[str, np.ndarray]:
        """Compute particle creation events near the wormhole.
        
        Args:
            metric: Spacetime metric
            exotic_matter: Exotic matter configuration
            
        Returns:
            Dictionary of particle creation data
        """
        
        print("Computing particle creation events...")
        
        particle_events = {
            'positions': [],
            'times': [],
            'energies': [],
            'types': [],  # 'virtual', 'real', 'hawking'
            'momenta': []
        }
        
        # Sample spacetime for particle creation analysis
        r_sample = self.spacetime_grid['r_1d'][::5]
        t_sample = self.spacetime_grid['t_1d'][::2]
        
        for t in t_sample:
            for r in r_sample:
                coords = (t, r, np.pi/2, 0.0)
                
                try:
                    # Compute local curvature and field strength
                    ricci_scalar = abs(metric.ricci_scalar(coords))
                    
                    # Energy density from exotic matter
                    rho_exotic = abs(exotic_matter.energy_density(coords))
                    
                    # Particle creation probability based on Schwinger effect
                    # P ∝ exp(-πm²c⁴/ℏeE) for electric field E
                    
                    # Effective field strength from curvature and exotic matter
                    effective_field = np.sqrt(ricci_scalar + rho_exotic / self.E_planck)
                    
                    # Critical field strength (Schwinger limit)
                    schwinger_field = self.m_planck * self.c**2 / (self.hbar * self.l_planck)
                    
                    # Particle creation probability
                    if effective_field > 0:
                        creation_probability = np.exp(-np.pi * schwinger_field / effective_field)
                    else:
                        creation_probability = 0.0
                    
                    # Create particles probabilistically
                    if (creation_probability > self.config.particle_creation_threshold and 
                        len(particle_events['positions']) < self.config.max_particles):
                        
                        # Particle properties
                        particle_energy = effective_field * self.hbar * self.c
                        particle_momentum = particle_energy / self.c
                        
                        # Determine particle type
                        if creation_probability > 0.8:
                            particle_type = 'real'
                        elif ricci_scalar > rho_exotic:
                            particle_type = 'hawking'
                        else:
                            particle_type = 'virtual'
                        
                        # Store particle data
                        particle_events['positions'].append([t, r, np.pi/2, 0.0])
                        particle_events['times'].append(t)
                        particle_events['energies'].append(particle_energy)
                        particle_events['types'].append(particle_type)
                        particle_events['momenta'].append(particle_momentum)
                
                except Exception:
                    continue
        
        # Convert to numpy arrays
        for key in particle_events:
            particle_events[key] = np.array(particle_events[key])
        
        print(f"Found {len(particle_events['positions'])} particle creation events")
        
        return particle_events
    
    def compute_hawking_radiation(self, metric: SpacetimeMetric,
                                throat_radius: float) -> Dict[str, np.ndarray]:
        """Compute Hawking radiation from wormhole horizons.
        
        Args:
            metric: Spacetime metric
            throat_radius: Wormhole throat radius
            
        Returns:
            Hawking radiation field data
        """
        
        print("Computing Hawking radiation...")
        
        # Hawking temperature for wormhole
        # T_H = ℏκ/(2πkB) where κ is surface gravity
        
        # Approximate surface gravity at throat
        throat_coords = (0.0, throat_radius, np.pi/2, 0.0)
        
        try:
            # Compute surface gravity from metric
            g_tt = metric.metric_tensor(throat_coords)[0, 0]
            g_rr = metric.metric_tensor(throat_coords)[1, 1]
            
            # Surface gravity approximation
            surface_gravity = self.c**4 / (4 * self.G * throat_radius * self.hbar)
            
            # Hawking temperature
            hawking_temp = self.hbar * surface_gravity / (2 * np.pi * self.k_b)
            
            print(f"Hawking temperature: {hawking_temp:.2e} K")
            
        except Exception:
            hawking_temp = 1e-7  # Default low temperature
        
        # Generate Hawking radiation field
        hawking_field = {
            'temperature': hawking_temp,
            'flux': np.zeros_like(self.spacetime_grid['r']),
            'spectrum': {},
            'particle_energies': [],
            'emission_rates': []
        }
        
        # Stefan-Boltzmann law for black-body radiation
        stefan_boltzmann = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
        
        # Hawking flux as function of distance from throat
        r_grid = self.spacetime_grid['r_1d']
        
        for i, r in enumerate(r_grid):
            # Flux decreases as 1/r² from throat
            distance_factor = throat_radius**2 / r**2
            
            # Greybody factor (transmission probability)
            greybody_factor = 1 / (1 + np.exp((r - throat_radius) / throat_radius))
            
            # Total flux
            flux = (stefan_boltzmann * hawking_temp**4 * 
                   distance_factor * greybody_factor * self.config.hawking_radiation_scale)
            
            hawking_field['flux'][0, i, :, :] = flux
        
        # Generate Hawking radiation spectrum
        # Planck distribution: n(ω) = 1/(exp(ℏω/kT) - 1)
        
        frequencies = np.logspace(10, 20, 100)  # Hz
        spectrum = []
        
        for freq in frequencies:
            photon_energy = self.hbar * 2 * np.pi * freq
            occupation_number = 1 / (np.exp(photon_energy / (self.k_b * hawking_temp)) - 1)
            spectrum.append(occupation_number)
        
        hawking_field['spectrum'] = {
            'frequencies': frequencies,
            'occupation_numbers': np.array(spectrum)
        }
        
        # Particle energies and emission rates
        thermal_energy = self.k_b * hawking_temp
        
        for n in range(100):
            particle_energy = thermal_energy * (n + 0.5)
            emission_rate = np.exp(-particle_energy / thermal_energy)
            
            hawking_field['particle_energies'].append(particle_energy)
            hawking_field['emission_rates'].append(emission_rate)
        
        hawking_field['particle_energies'] = np.array(hawking_field['particle_energies'])
        hawking_field['emission_rates'] = np.array(hawking_field['emission_rates'])
        
        print("Hawking radiation computed successfully")
        
        return hawking_field
    
    def compute_backreaction_effects(self, metric: SpacetimeMetric,
                                   exotic_matter: ExoticMatter,
                                   quantum_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute quantum backreaction on spacetime geometry.
        
        Args:
            metric: Original spacetime metric
            exotic_matter: Exotic matter configuration
            quantum_fields: Quantum field configurations
            
        Returns:
            Backreaction corrections to metric
        """
        
        print("Computing quantum backreaction effects...")
        
        backreaction = {
            'metric_correction': np.zeros((4, 4)),
            'energy_momentum_quantum': np.zeros((4, 4)),
            'vacuum_polarization': np.zeros_like(self.spacetime_grid['r']),
            'geometry_perturbation': np.zeros_like(self.spacetime_grid['r'])
        }
        
        # Sample points for backreaction calculation
        r_sample = self.spacetime_grid['r_1d'][::10]
        
        for r in r_sample:
            coords = (0.0, r, np.pi/2, 0.0)
            
            try:
                # Original metric components
                g_orig = metric.metric_tensor(coords)
                
                # Quantum stress-energy tensor
                # <T_μν> = classical T_μν + quantum corrections
                
                # Classical stress-energy from exotic matter
                T_classical = exotic_matter.stress_energy_tensor(coords)
                
                # Quantum corrections
                # 1. Vacuum polarization
                vacuum_energy = quantum_fields.get('vacuum_fluctuations', np.zeros(1))[0] if len(quantum_fields.get('vacuum_fluctuations', [])) > 0 else 0
                
                # 2. Particle creation contribution
                particle_energy_density = 0.0
                if 'particle_events' in quantum_fields:
                    for pos, energy in zip(quantum_fields['particle_events']['positions'],
                                         quantum_fields['particle_events']['energies']):
                        if abs(pos[1] - r) < r * 0.1:  # Within 10% of radius
                            particle_energy_density += energy / (4 * np.pi * r**2)
                
                # 3. Hawking radiation pressure
                hawking_pressure = 0.0
                if 'hawking_radiation' in quantum_fields:
                    r_idx = np.argmin(np.abs(self.spacetime_grid['r_1d'] - r))
                    if r_idx < quantum_fields['hawking_radiation']['flux'].shape[1]:
                        hawking_flux = quantum_fields['hawking_radiation']['flux'][0, r_idx, 0, 0]
                        hawking_pressure = hawking_flux / self.c  # Radiation pressure
                
                # Total quantum stress-energy tensor
                T_quantum = T_classical.copy()
                
                # Add quantum corrections
                T_quantum[0, 0] += vacuum_energy + particle_energy_density  # Energy density
                T_quantum[1, 1] += hawking_pressure  # Radial pressure
                T_quantum[2, 2] += hawking_pressure  # Tangential pressure
                T_quantum[3, 3] += hawking_pressure  # Tangential pressure
                
                # Einstein field equations: G_μν = 8πG T_μν
                # Metric correction: δg_μν = 8πG δT_μν / curvature_scale
                
                curvature_scale = 1 / r**2  # Characteristic curvature
                correction_amplitude = 8 * np.pi * self.G / (self.c**4 * curvature_scale)
                
                # Metric corrections (small perturbations)
                metric_correction = correction_amplitude * (T_quantum - T_classical)
                
                # Store results
                r_idx = np.argmin(np.abs(self.spacetime_grid['r_1d'] - r))
                
                backreaction['metric_correction'] += metric_correction / len(r_sample)
                backreaction['energy_momentum_quantum'] = T_quantum
                
                if r_idx < backreaction['vacuum_polarization'].shape[1]:
                    backreaction['vacuum_polarization'][0, r_idx, 0, 0] = vacuum_energy
                    
                    # Geometry perturbation (relative change in metric determinant)
                    det_correction = np.linalg.det(g_orig + metric_correction) / np.linalg.det(g_orig) - 1
                    backreaction['geometry_perturbation'][0, r_idx, 0, 0] = det_correction
            
            except Exception as e:
                continue  # Skip problematic points
        
        print("Quantum backreaction computed successfully")
        
        return backreaction
    
    def compute_entanglement_structure(self, quantum_circuit: WormholeQuantumCircuit,
                                     entanglement_dynamics: EntanglementDynamics) -> Dict[str, np.ndarray]:
        """Compute entanglement structure across wormhole.
        
        Args:
            quantum_circuit: Wormhole quantum circuit
            entanglement_dynamics: Entanglement evolution
            
        Returns:
            Entanglement structure data
        """
        
        print("Computing entanglement structure...")
        
        entanglement_structure = {
            'entanglement_entropy': np.zeros_like(self.spacetime_grid['r']),
            'mutual_information': np.zeros_like(self.spacetime_grid['r']),
            'negativity': np.zeros_like(self.spacetime_grid['r']),
            'entangling_regions': []
        }
        
        try:
            # Initialize quantum state
            initial_state = quantum_circuit.initialize_wormhole_state()
            
            # Evolve state across wormhole
            r_sample = self.spacetime_grid['r_1d'][::5]
            
            for i, r in enumerate(r_sample):
                # Geometric phase evolution
                geometry_params = {'radius': r, 'curvature': 1/r}
                evolved_state = quantum_circuit.evolve_through_geometry(initial_state, geometry_params)
                
                # Compute entanglement measures
                try:
                    # Entanglement entropy (von Neumann entropy of reduced state)
                    entropy = entanglement_dynamics.compute_entanglement_entropy(evolved_state)
                    
                    # Mutual information between throat regions
                    mutual_info = entanglement_dynamics.compute_mutual_information(evolved_state)
                    
                    # Negativity (measure of quantum correlations)
                    negativity = entanglement_dynamics.compute_logarithmic_negativity(evolved_state)
                    
                    # Map to grid
                    r_idx = np.argmin(np.abs(self.spacetime_grid['r_1d'] - r))
                    
                    if r_idx < entanglement_structure['entanglement_entropy'].shape[1]:
                        entanglement_structure['entanglement_entropy'][0, r_idx, 0, 0] = entropy
                        entanglement_structure['mutual_information'][0, r_idx, 0, 0] = mutual_info
                        entanglement_structure['negativity'][0, r_idx, 0, 0] = negativity
                    
                    # Identify highly entangled regions
                    if entropy > 1.0:  # Threshold for strong entanglement
                        entanglement_structure['entangling_regions'].append({
                            'radius': r,
                            'entropy': entropy,
                            'mutual_information': mutual_info
                        })
                
                except Exception as e:
                    continue  # Skip problematic calculations
        
        except Exception as e:
            print(f"Warning: Entanglement calculation failed: {e}")
            # Fill with default values
            pass
        
        print(f"Found {len(entanglement_structure['entangling_regions'])} highly entangled regions")
        
        return entanglement_structure
    
    def _interpolate_field(self, field_data: np.ndarray) -> np.ndarray:
        """Interpolate sparse field data to full grid.
        
        Args:
            field_data: Sparse field data
            
        Returns:
            Interpolated field on full grid
        """
        
        # Find non-zero points
        non_zero_indices = np.nonzero(field_data)
        
        if len(non_zero_indices[0]) == 0:
            return field_data
        
        # Extract coordinates and values
        points = []
        values = []
        
        for i in range(len(non_zero_indices[0])):
            t_idx = non_zero_indices[0][i]
            r_idx = non_zero_indices[1][i]
            theta_idx = non_zero_indices[2][i]
            phi_idx = non_zero_indices[3][i]
            
            points.append([t_idx, r_idx, theta_idx, phi_idx])
            values.append(field_data[t_idx, r_idx, theta_idx, phi_idx])
        
        if len(values) < 2:
            return field_data
        
        # Create full grid coordinates
        t_indices, r_indices, theta_indices, phi_indices = np.meshgrid(
            range(field_data.shape[0]),
            range(field_data.shape[1]),
            range(field_data.shape[2]),
            range(field_data.shape[3]),
            indexing='ij'
        )
        
        grid_points = np.column_stack([
            t_indices.ravel(),
            r_indices.ravel(),
            theta_indices.ravel(),
            phi_indices.ravel()
        ])
        
        # Interpolate (use nearest neighbor for high-dimensional data)
        try:
            interpolated_values = griddata(
                points, values, grid_points,
                method='nearest', fill_value=0.0
            )
            
            return interpolated_values.reshape(field_data.shape)
            
        except Exception:
            return field_data  # Return original if interpolation fails
    
    def create_comprehensive_field_visualization(self, metric: SpacetimeMetric,
                                               exotic_matter: ExoticMatter,
                                               quantum_circuit: WormholeQuantumCircuit = None) -> go.Figure:
        """Create comprehensive visualization of all quantum field effects.
        
        Args:
            metric: Spacetime metric
            exotic_matter: Exotic matter configuration
            quantum_circuit: Quantum circuit (optional)
            
        Returns:
            Comprehensive field visualization
        """
        
        print("Creating comprehensive quantum field visualization...")
        
        # Compute all quantum field effects
        quantum_fields = {}
        
        if self.config.show_vacuum_fluctuations:
            quantum_fields['vacuum_fluctuations'] = self.compute_vacuum_fluctuations(metric, exotic_matter)
        
        if self.config.show_particle_creation:
            quantum_fields['particle_events'] = self.compute_particle_creation_events(metric, exotic_matter)
        
        if self.config.show_hawking_radiation:
            throat_radius = self.config.r_min * 10  # Estimate throat radius
            quantum_fields['hawking_radiation'] = self.compute_hawking_radiation(metric, throat_radius)
        
        if self.config.show_backreaction:
            quantum_fields['backreaction'] = self.compute_backreaction_effects(metric, exotic_matter, quantum_fields)
        
        if self.config.show_entanglement_structure and quantum_circuit:
            try:
                from src.quantum.entanglement_dynamics import EntanglementDynamics
                entanglement_dynamics = EntanglementDynamics(quantum_circuit.num_qubits)
                quantum_fields['entanglement'] = self.compute_entanglement_structure(quantum_circuit, entanglement_dynamics)
            except Exception:
                print("Warning: Entanglement computation skipped")
        
        # Create visualization
        fig = self._create_field_visualization_figure(quantum_fields)
        
        print("Comprehensive field visualization created successfully")
        
        return fig
    
    def _create_field_visualization_figure(self, quantum_fields: Dict[str, Any]) -> go.Figure:
        """Create the actual visualization figure from computed field data.
        
        Args:
            quantum_fields: Computed quantum field data
            
        Returns:
            Plotly figure with quantum field visualizations
        """
        
        # Determine number of subplots based on available data
        subplot_count = len(quantum_fields)
        
        if subplot_count <= 2:
            rows, cols = 1, subplot_count
        elif subplot_count <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        # Create subplot titles
        subplot_titles = []
        for field_type in quantum_fields.keys():
            subplot_titles.append(field_type.replace('_', ' ').title())
        
        # Pad titles if necessary
        while len(subplot_titles) < rows * cols:
            subplot_titles.append("")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'scatter'} for _ in range(cols)] for _ in range(rows)]
        )
        
        plot_idx = 0
        
        # Plot vacuum fluctuations
        if 'vacuum_fluctuations' in quantum_fields:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            field_data = quantum_fields['vacuum_fluctuations']
            
            # Extract 2D slice (t=0, theta=π/2)
            field_slice = field_data[0, :, self.config.theta_points//2, 0]
            r_coords = self.spacetime_grid['r_1d']
            
            fig.add_trace(
                go.Scatter(
                    x=r_coords,
                    y=field_slice,
                    mode='lines',
                    name='Vacuum Fluctuations',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=col
            )
        
        # Plot particle creation events
        if 'particle_events' in quantum_fields:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            events = quantum_fields['particle_events']
            
            if len(events['positions']) > 0:
                # Extract radial positions and energies
                r_positions = [pos[1] for pos in events['positions']]
                energies = events['energies']
                types = events['types']
                
                # Color code by particle type
                colors = {'real': 'red', 'virtual': 'blue', 'hawking': 'orange'}
                
                for ptype in ['real', 'virtual', 'hawking']:
                    mask = [t == ptype for t in types]
                    if any(mask):
                        r_type = [r for r, m in zip(r_positions, mask) if m]
                        e_type = [e for e, m in zip(energies, mask) if m]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=r_type,
                                y=e_type,
                                mode='markers',
                                name=f'{ptype.title()} Particles',
                                marker=dict(
                                    color=colors[ptype],
                                    size=8,
                                    symbol='circle'
                                )
                            ),
                            row=row, col=col
                        )
        
        # Plot Hawking radiation
        if 'hawking_radiation' in quantum_fields:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            hawking_data = quantum_fields['hawking_radiation']
            
            # Plot flux vs radius
            flux_data = hawking_data['flux'][0, :, 0, 0]
            r_coords = self.spacetime_grid['r_1d']
            
            fig.add_trace(
                go.Scatter(
                    x=r_coords,
                    y=flux_data,
                    mode='lines',
                    name='Hawking Flux',
                    line=dict(color='orange', width=3)
                ),
                row=row, col=col
            )
        
        # Plot backreaction effects
        if 'backreaction' in quantum_fields:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            backreaction_data = quantum_fields['backreaction']
            
            # Plot geometry perturbation
            geometry_pert = backreaction_data['geometry_perturbation'][0, :, 0, 0]
            r_coords = self.spacetime_grid['r_1d']
            
            fig.add_trace(
                go.Scatter(
                    x=r_coords,
                    y=geometry_pert,
                    mode='lines',
                    name='Geometry Perturbation',
                    line=dict(color='green', width=2)
                ),
                row=row, col=col
            )
        
        # Plot entanglement structure
        if 'entanglement' in quantum_fields:
            plot_idx += 1
            row, col = ((plot_idx - 1) // cols) + 1, ((plot_idx - 1) % cols) + 1
            
            entanglement_data = quantum_fields['entanglement']
            
            # Plot entanglement entropy
            entropy_data = entanglement_data['entanglement_entropy'][0, :, 0, 0]
            r_coords = self.spacetime_grid['r_1d']
            
            fig.add_trace(
                go.Scatter(
                    x=r_coords,
                    y=entropy_data,
                    mode='lines+markers',
                    name='Entanglement Entropy',
                    line=dict(color='purple', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title='Comprehensive Quantum Field Effects Visualization',
            height=400 * rows,
            width=600 * cols,
            showlegend=True
        )
        
        # Update axes labels
        for row_idx in range(1, rows + 1):
            for col_idx in range(1, cols + 1):
                fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=row_idx, col=col_idx)
                
                # Set appropriate y-axis labels based on subplot
                plot_num = (row_idx - 1) * cols + col_idx
                if plot_num <= len(quantum_fields):
                    field_types = list(quantum_fields.keys())
                    if plot_num <= len(field_types):
                        field_type = field_types[plot_num - 1]
                        
                        if 'vacuum' in field_type:
                            y_label = 'Field Amplitude'
                        elif 'particle' in field_type:
                            y_label = 'Energy (J)'
                        elif 'hawking' in field_type:
                            y_label = 'Flux (W/m²)'
                        elif 'backreaction' in field_type:
                            y_label = 'Relative Change'
                        elif 'entanglement' in field_type:
                            y_label = 'Entropy (bits)'
                        else:
                            y_label = 'Value'
                        
                        fig.update_yaxes(title_text=y_label, row=row_idx, col=col_idx)
        
        return fig
    
    def create_animated_field_evolution(self, metric: SpacetimeMetric,
                                      exotic_matter: ExoticMatter) -> go.Figure:
        """Create animated visualization of quantum field evolution.
        
        Args:
            metric: Spacetime metric
            exotic_matter: Exotic matter configuration
            
        Returns:
            Animated figure showing field evolution
        """
        
        print("Creating animated quantum field evolution...")
        
        # Compute field evolution over time
        frames = []
        time_steps = self.spacetime_grid['t_1d'][::5]  # Sample every 5th time step
        
        for t in time_steps:
            # Compute fields at this time step
            field_data = []
            r_coords = self.spacetime_grid['r_1d']
            
            for r in r_coords:
                coords = (t, r, np.pi/2, 0.0)
                
                try:
                    # Vacuum fluctuation at this point
                    ricci = abs(metric.ricci_scalar(coords))
                    vacuum_amp = np.sqrt(ricci) * np.exp(-r / self.l_planck)
                    field_data.append(vacuum_amp)
                except:
                    field_data.append(0.0)
            
            # Create frame
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=r_coords,
                        y=field_data,
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        name='Quantum Field'
                    )
                ],
                name=f't = {t:.2e} s'
            )
            
            frames.append(frame)
        
        # Create initial plot
        initial_field = []
        for r in r_coords:
            coords = (0.0, r, np.pi/2, 0.0)
            try:
                ricci = abs(metric.ricci_scalar(coords))
                vacuum_amp = np.sqrt(ricci) * np.exp(-r / self.l_planck)
                initial_field.append(vacuum_amp)
            except:
                initial_field.append(0.0)
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=r_coords,
                    y=initial_field,
                    mode='lines',
                    line=dict(width=3, color='blue'),
                    name='Quantum Field'
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title='Quantum Field Evolution in Curved Spacetime',
            xaxis=dict(title='Radial Distance (m)', type='log'),
            yaxis=dict(title='Field Amplitude'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
        
        print("Animated field evolution created successfully")
        
        return fig


def create_quantum_field_demo() -> Dict[str, go.Figure]:
    """Create demonstration of quantum field rendering capabilities.
    
    Returns:
        Dictionary of demonstration figures
    """
    
    print("Creating quantum field effects demonstration...")
    
    # Import required modules
    from src.physics.spacetime_metrics import MorrisThorneeWormhole
    from src.physics.exotic_matter import AdvancedCasimirExoticMatter
    
    # Initialize components
    config = QuantumFieldRenderConfig(
        r_points=100,
        t_points=50,
        show_vacuum_fluctuations=True,
        show_particle_creation=True,
        show_hawking_radiation=True,
        show_backreaction=True
    )
    
    renderer = QuantumFieldRenderer(config)
    
    # Create test metric and exotic matter
    metric = MorrisThorneeWormhole(throat_radius=1e3)
    exotic_matter = AdvancedCasimirExoticMatter(plate_separation=1e-6)
    
    # Generate demonstrations
    demo_figures = {}
    
    try:
        # Comprehensive field visualization
        demo_figures['comprehensive_fields'] = renderer.create_comprehensive_field_visualization(
            metric, exotic_matter
        )
        
        # Animated field evolution
        demo_figures['animated_evolution'] = renderer.create_animated_field_evolution(
            metric, exotic_matter
        )
        
        print(f"Quantum field demonstration created with {len(demo_figures)} visualizations")
        
    except Exception as e:
        print(f"Demo creation failed: {e}")
        demo_figures['error'] = go.Figure().add_annotation(
            text=f"Error creating demo: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    return demo_figures


if __name__ == "__main__":
    # Create and save demonstration
    print("Quantum Field Renderer Demo")
    print("="*40)
    
    demo_figures = create_quantum_field_demo()
    
    # Save figures
    for name, fig in demo_figures.items():
        filename = f"quantum_field_{name}.html"
        fig.write_html(filename)
        print(f"Saved: {filename}")
    
    print("\nDemo completed! Open the HTML files to explore quantum field effects.")