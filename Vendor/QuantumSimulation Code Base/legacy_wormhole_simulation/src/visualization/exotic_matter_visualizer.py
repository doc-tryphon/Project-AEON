"""
Advanced Exotic Matter Visualization System

This module provides comprehensive visualization capabilities for exotic matter
distributions, energy condition violations, stability analysis, and quantum
field effects in wormhole geometries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import concurrent.futures
from functools import lru_cache

# Import enhanced exotic matter module
from src.physics.exotic_matter import (
    ExoticMatter, AdvancedCasimirExoticMatter, PhantomDarkEnergyField,
    QuantumInequalityConstrainedMatter, StringTheoryDerivedMatter,
    HybridExoticMatter, EnergyConditionResult, StabilityAnalysis,
    optimize_exotic_matter_configuration, load_exotic_matter_from_catalog
)

# Import other visualization modules
from src.spacetime_plotter import SpacetimePlotter
from src.quantum_state_animator import QuantumStateAnimator
from src.field_visualizer import FieldVisualizer


@dataclass
class ExoticMatterVisualizationConfig:
    """Configuration for exotic matter visualization."""
    
    # Grid parameters
    r_min: float = 1e-6
    r_max: float = 1e-2
    r_points: int = 200
    theta_points: int = 100
    phi_points: int = 50
    
    # Time evolution
    t_min: float = 0.0
    t_max: float = 1e-20
    t_points: int = 50
    
    # Visualization parameters
    colormap: str = 'RdBu_r'
    energy_condition_colormap: str = 'viridis'
    stability_colormap: str = 'plasma'
    animation_fps: int = 10
    save_animation: bool = False
    
    # Analysis parameters
    energy_condition_threshold: float = 1e-10
    stability_threshold: float = 0.5
    violation_magnitude_scale: str = 'log'  # 'linear' or 'log'
    
    # Interactive parameters
    enable_widgets: bool = True
    real_time_update: bool = True
    auto_optimization: bool = False


class ExoticMatterVisualizer:
    """Advanced visualizer for exotic matter distributions and properties."""
    
    def __init__(self, config: ExoticMatterVisualizationConfig = None):
        """Initialize exotic matter visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or ExoticMatterVisualizationConfig()
        
        # Coordinate grids
        self.r_grid = None
        self.theta_grid = None
        self.phi_grid = None
        self.coords_grid = None
        
        # Cached calculations
        self.cache = {}
        self.cache_timestamps = {}
        
        # Current exotic matter instance
        self.current_matter = None
        
        # Animation objects
        self.animation_objects = {}
        
        # Initialize coordinate grids
        self._initialize_grids()
        
        print(f"Exotic matter visualizer initialized with {self.config.r_points}×{self.config.theta_points}×{self.config.phi_points} grid")
    
    def _initialize_grids(self):
        """Initialize coordinate grids for visualization."""
        
        # Radial grid (logarithmic for better resolution near throat)
        self.r_grid = np.logspace(
            np.log10(self.config.r_min), 
            np.log10(self.config.r_max), 
            self.config.r_points
        )
        
        # Angular grids
        self.theta_grid = np.linspace(0, np.pi, self.config.theta_points)
        self.phi_grid = np.linspace(0, 2*np.pi, self.config.phi_points)
        
        # Time grid
        self.t_grid = np.linspace(self.config.t_min, self.config.t_max, self.config.t_points)
        
        # 4D coordinate mesh for calculations
        self.coords_grid = self._create_coordinate_mesh()
    
    def _create_coordinate_mesh(self) -> np.ndarray:
        """Create 4D coordinate mesh for calculations."""
        
        # Create coordinate arrays for vectorized calculations
        coords_list = []
        
        # Sample subset for performance
        r_sample = self.r_grid[::max(1, len(self.r_grid)//50)]
        theta_sample = self.theta_grid[::max(1, len(self.theta_grid)//20)]
        
        for t in [0.0]:  # Static case for now
            for r in r_sample:
                for theta in theta_sample:
                    for phi in [0.0]:  # Azimuthal symmetry
                        coords_list.append((t, r, theta, phi))
        
        return np.array(coords_list)
    
    @lru_cache(maxsize=100)
    def _compute_matter_properties(self, matter_id: str, property_type: str) -> np.ndarray:
        """Compute matter properties with caching.
        
        Args:
            matter_id: Unique identifier for matter configuration
            property_type: Type of property to compute
            
        Returns:
            Property values at grid points
        """
        
        if self.current_matter is None:
            return np.zeros(len(self.coords_grid))
        
        properties = []
        
        for coords in self.coords_grid:
            try:
                if property_type == 'energy_density':
                    value = self.current_matter.energy_density(tuple(coords))
                elif property_type == 'pressure_radial':
                    value = self.current_matter.pressure_radial(tuple(coords))
                elif property_type == 'pressure_tangential':
                    value = self.current_matter.pressure_tangential(tuple(coords))
                elif property_type == 'equation_of_state':
                    eos_params = self.current_matter.equation_of_state_parameters(coords[1])
                    value = eos_params['w_radial']
                else:
                    value = 0.0
                    
                properties.append(value)
                
            except Exception as e:
                properties.append(0.0)
        
        return np.array(properties)
    
    def set_exotic_matter(self, matter: ExoticMatter):
        """Set the current exotic matter for visualization.
        
        Args:
            matter: Exotic matter instance
        """
        self.current_matter = matter
        
        # Clear cache when matter changes
        self.cache.clear()
        self.cache_timestamps.clear()
        
        print(f"Set exotic matter: {matter.name}")
    
    def create_energy_condition_map(self, matter: ExoticMatter = None) -> go.Figure:
        """Create comprehensive energy condition violation map.
        
        Args:
            matter: Exotic matter instance (uses current if None)
            
        Returns:
            Energy condition visualization figure
        """
        
        if matter is not None:
            self.set_exotic_matter(matter)
        
        if self.current_matter is None:
            raise ValueError("No exotic matter set for visualization")
        
        print("Creating energy condition violation map...")
        
        # Compute energy conditions across grid
        energy_conditions = []
        violation_magnitudes = []
        causality_status = []
        
        for coords in self.coords_grid:
            try:
                ec_result = self.current_matter.check_energy_conditions(tuple(coords))
                
                # Encode energy conditions as integer
                ec_flags = (
                    int(ec_result.null_energy_condition) * 1 +
                    int(ec_result.weak_energy_condition) * 2 +
                    int(ec_result.strong_energy_condition) * 4 +
                    int(ec_result.dominant_energy_condition) * 8
                )
                
                energy_conditions.append(ec_flags)
                violation_magnitudes.append(ec_result.violation_magnitude)
                causality_status.append(int(ec_result.causality_preserved))
                
            except Exception:
                energy_conditions.append(0)
                violation_magnitudes.append(0.0)
                causality_status.append(1)
        
        energy_conditions = np.array(energy_conditions)
        violation_magnitudes = np.array(violation_magnitudes)
        causality_status = np.array(causality_status)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Energy Condition Violations',
                'Violation Magnitudes',
                'Causality Map',
                'Combined Analysis'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # Extract r coordinates for plotting
        r_coords = self.coords_grid[:, 1]
        theta_coords = self.coords_grid[:, 2]
        
        # 1. Energy condition violations
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=energy_conditions,
                    colorscale='viridis',
                    size=8,
                    colorbar=dict(
                        title='Energy Conditions<br>(Binary Encoded)',
                        x=0.45
                    ),
                    cmin=0,
                    cmax=15
                ),
                name='Energy Conditions',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Violations: %{marker.color}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # 2. Violation magnitudes
        log_violations = np.log10(violation_magnitudes + 1e-50)  # Avoid log(0)
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=log_violations,
                    colorscale='plasma',
                    size=8,
                    colorbar=dict(
                        title='log₁₀(Violation<br>Magnitude)',
                        x=1.02
                    )
                ),
                name='Violation Magnitude',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Magnitude: %{marker.color:.2f}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=2
        )
        
        # 3. Causality map
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=causality_status,
                    colorscale=[[0, 'red'], [1, 'green']],
                    size=10,
                    colorbar=dict(
                        title='Causality<br>Preserved',
                        x=0.45
                    ),
                    cmin=0,
                    cmax=1
                ),
                name='Causality',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Causal: %{marker.color}<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # 4. Combined analysis (violations weighted by magnitude)
        combined_score = energy_conditions * np.log10(violation_magnitudes + 1e-10)
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=combined_score,
                    colorscale='RdYlBu_r',
                    size=8,
                    colorbar=dict(
                        title='Combined<br>Violation Score',
                        x=1.02
                    )
                ),
                name='Combined Score',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Score: %{marker.color:.2f}<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Energy Condition Analysis: {self.current_matter.name}',
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=row, col=col)
                fig.update_yaxes(title_text='Polar Angle (rad)', row=row, col=col)
        
        print("Energy condition map created successfully")
        
        return fig
    
    def create_stability_landscape(self, matter: ExoticMatter = None) -> go.Figure:
        """Create comprehensive stability analysis visualization.
        
        Args:
            matter: Exotic matter instance
            
        Returns:
            Stability landscape figure
        """
        
        if matter is not None:
            self.set_exotic_matter(matter)
        
        if self.current_matter is None:
            raise ValueError("No exotic matter set for visualization")
        
        print("Creating stability landscape...")
        
        # Compute stability properties
        radial_sound_speeds = []
        tangential_sound_speeds = []
        radial_eigenvalues_real = []
        radial_eigenvalues_imag = []
        jeans_wavelengths = []
        
        for coords in self.coords_grid:
            try:
                stability = self.current_matter.stability_analysis(tuple(coords))
                
                radial_sound_speeds.append(stability.radial_sound_speed)
                tangential_sound_speeds.append(stability.tangential_sound_speed)
                radial_eigenvalues_real.append(stability.radial_perturbation_eigenvalue.real)
                radial_eigenvalues_imag.append(stability.radial_perturbation_eigenvalue.imag)
                jeans_wavelengths.append(min(stability.jeans_instability_wavelength, 1e10))
                
            except Exception:
                radial_sound_speeds.append(0.0)
                tangential_sound_speeds.append(0.0)
                radial_eigenvalues_real.append(0.0)
                radial_eigenvalues_imag.append(0.0)
                jeans_wavelengths.append(1e10)
        
        # Convert to arrays
        radial_sound_speeds = np.array(radial_sound_speeds)
        tangential_sound_speeds = np.array(tangential_sound_speeds)
        radial_eigenvalues_real = np.array(radial_eigenvalues_real)
        radial_eigenvalues_imag = np.array(radial_eigenvalues_imag)
        jeans_wavelengths = np.array(jeans_wavelengths)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Sound Speed Analysis',
                'Perturbation Eigenvalues',
                'Jeans Instability Scale',
                'Stability Regions'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        r_coords = self.coords_grid[:, 1]
        theta_coords = self.coords_grid[:, 2]
        
        # 1. Sound speed analysis
        sound_speed_ratio = tangential_sound_speeds / (radial_sound_speeds + 1e-10)
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=sound_speed_ratio,
                    colorscale='viridis',
                    size=8,
                    colorbar=dict(
                        title='Sound Speed<br>Ratio (t/r)',
                        x=0.45
                    )
                ),
                name='Sound Speed Ratio',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Ratio: %{marker.color:.3f}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # 2. Perturbation eigenvalues
        eigenvalue_magnitude = np.sqrt(radial_eigenvalues_real**2 + radial_eigenvalues_imag**2)
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=eigenvalue_magnitude,
                    colorscale='plasma',
                    size=8,
                    colorbar=dict(
                        title='|λ| Eigenvalue<br>Magnitude',
                        x=1.02
                    )
                ),
                name='Eigenvalue Magnitude',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    '|λ|: %{marker.color:.3f}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=2
        )
        
        # 3. Jeans instability scale
        log_jeans = np.log10(jeans_wavelengths)
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=log_jeans,
                    colorscale='blues',
                    size=8,
                    colorbar=dict(
                        title='log₁₀(Jeans<br>Wavelength)',
                        x=0.45
                    )
                ),
                name='Jeans Scale',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'log λ_J: %{marker.color:.2f}<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # 4. Stability regions
        stability_score = (
            (radial_sound_speeds < 3e8) * 1 +  # Subluminal radial
            (tangential_sound_speeds < 3e8) * 1 +  # Subluminal tangential
            (radial_eigenvalues_real < 0) * 1 +  # Stable real part
            (abs(radial_eigenvalues_imag) < 1) * 1  # Bounded oscillations
        ) / 4.0
        
        fig.add_trace(
            go.Scatter(
                x=r_coords,
                y=theta_coords,
                mode='markers',
                marker=dict(
                    color=stability_score,
                    colorscale='RdYlGn',
                    size=10,
                    colorbar=dict(
                        title='Stability<br>Score',
                        x=1.02
                    ),
                    cmin=0,
                    cmax=1
                ),
                name='Stability Score',
                hovertemplate=(
                    'r: %{x:.2e} m<br>'
                    'θ: %{y:.2f} rad<br>'
                    'Stability: %{marker.color:.3f}<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Stability Analysis: {self.current_matter.name}',
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=row, col=col)
                fig.update_yaxes(title_text='Polar Angle (rad)', row=row, col=col)
        
        print("Stability landscape created successfully")
        
        return fig
    
    def create_matter_distribution_3d(self, matter: ExoticMatter = None,
                                    property_type: str = 'energy_density') -> go.Figure:
        """Create 3D visualization of exotic matter distribution.
        
        Args:
            matter: Exotic matter instance
            property_type: Property to visualize
            
        Returns:
            3D visualization figure
        """
        
        if matter is not None:
            self.set_exotic_matter(matter)
        
        if self.current_matter is None:
            raise ValueError("No exotic matter set for visualization")
        
        print(f"Creating 3D {property_type} distribution...")
        
        # Create 3D grid
        r_3d = np.logspace(np.log10(self.config.r_min), np.log10(self.config.r_max), 50)
        theta_3d = np.linspace(0, np.pi, 25)
        phi_3d = np.linspace(0, 2*np.pi, 25)
        
        R, THETA, PHI = np.meshgrid(r_3d, theta_3d, phi_3d, indexing='ij')
        
        # Convert to Cartesian coordinates
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Compute property values
        property_values = np.zeros_like(R)
        
        for i in range(len(r_3d)):
            for j in range(len(theta_3d)):
                for k in range(len(phi_3d)):
                    coords = (0.0, R[i,j,k], THETA[i,j,k], PHI[i,j,k])
                    
                    try:
                        if property_type == 'energy_density':
                            value = self.current_matter.energy_density(coords)
                        elif property_type == 'pressure_radial':
                            value = self.current_matter.pressure_radial(coords)
                        elif property_type == 'pressure_tangential':
                            value = self.current_matter.pressure_tangential(coords)
                        else:
                            value = 0.0
                            
                        property_values[i,j,k] = value
                        
                    except Exception:
                        property_values[i,j,k] = 0.0
        
        # Create 3D visualization
        fig = go.Figure()
        
        # Add isosurfaces for different property levels
        if np.any(property_values != 0):
            # Determine good isosurface levels
            non_zero_values = property_values[property_values != 0]
            
            if len(non_zero_values) > 0:
                if property_type == 'energy_density' and np.any(non_zero_values < 0):
                    # For negative energy density
                    levels = np.percentile(non_zero_values[non_zero_values < 0], [10, 30, 50, 70, 90])
                else:
                    # For positive values
                    levels = np.percentile(np.abs(non_zero_values), [10, 30, 50, 70, 90])
                
                # Add isosurfaces
                for i, level in enumerate(levels):
                    if not np.isnan(level):
                        fig.add_trace(
                            go.Isosurface(
                                x=X.flatten(),
                                y=Y.flatten(),
                                z=Z.flatten(),
                                value=property_values.flatten(),
                                isomin=level,
                                isomax=level,
                                opacity=0.3,
                                surface_count=1,
                                colorscale='viridis',
                                name=f'Level {i+1}',
                                showscale=(i == 0)
                            )
                        )
        
        # Add wormhole throat representation
        throat_radius = self.config.r_min * 10  # Approximate throat
        
        # Throat sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_throat = throat_radius * np.outer(np.cos(u), np.sin(v))
        y_throat = throat_radius * np.outer(np.sin(u), np.sin(v))
        z_throat = throat_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(
            go.Surface(
                x=x_throat,
                y=y_throat,
                z=z_throat,
                colorscale='greys',
                opacity=0.5,
                name='Wormhole Throat',
                showscale=False
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'3D {property_type.replace("_", " ").title()}: {self.current_matter.name}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            width=800,
            height=800
        )
        
        print(f"3D {property_type} visualization created successfully")
        
        return fig
    
    def create_time_evolution_animation(self, matter: ExoticMatter = None,
                                      property_type: str = 'energy_density') -> go.Figure:
        """Create animated visualization of time evolution.
        
        Args:
            matter: Exotic matter instance
            property_type: Property to animate
            
        Returns:
            Animated figure
        """
        
        if matter is not None:
            self.set_exotic_matter(matter)
        
        if self.current_matter is None:
            raise ValueError("No exotic matter set for visualization")
        
        print(f"Creating time evolution animation for {property_type}...")
        
        # Time-dependent visualization
        frames = []
        
        # Sample radial coordinates for animation
        r_sample = np.logspace(np.log10(self.config.r_min), np.log10(self.config.r_max), 100)
        
        for t in self.t_grid:
            property_values = []
            
            for r in r_sample:
                coords = (t, r, np.pi/2, 0.0)  # Equatorial slice
                
                try:
                    if property_type == 'energy_density':
                        value = self.current_matter.energy_density(coords)
                    elif property_type == 'pressure_radial':
                        value = self.current_matter.pressure_radial(coords)
                    elif property_type == 'pressure_tangential':
                        value = self.current_matter.pressure_tangential(coords)
                    else:
                        value = 0.0
                        
                    property_values.append(value)
                    
                except Exception:
                    property_values.append(0.0)
            
            # Create frame
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=r_sample,
                        y=property_values,
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        name=property_type.replace('_', ' ').title()
                    )
                ],
                name=f't = {t:.2e} s'
            )
            
            frames.append(frame)
        
        # Create initial plot
        initial_values = []
        for r in r_sample:
            coords = (0.0, r, np.pi/2, 0.0)
            try:
                if property_type == 'energy_density':
                    value = self.current_matter.energy_density(coords)
                elif property_type == 'pressure_radial':
                    value = self.current_matter.pressure_radial(coords)
                elif property_type == 'pressure_tangential':
                    value = self.current_matter.pressure_tangential(coords)
                else:
                    value = 0.0
                initial_values.append(value)
            except Exception:
                initial_values.append(0.0)
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=r_sample,
                    y=initial_values,
                    mode='lines',
                    line=dict(width=3, color='blue'),
                    name=property_type.replace('_', ' ').title()
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=f'Time Evolution: {property_type.replace("_", " ").title()} - {self.current_matter.name}',
            xaxis=dict(title='Radial Distance (m)', type='log'),
            yaxis=dict(title=f'{property_type.replace("_", " ").title()} (SI units)'),
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
        
        print("Time evolution animation created successfully")
        
        return fig
    
    def create_comparative_analysis(self, matter_list: List[ExoticMatter],
                                  analysis_type: str = 'energy_conditions') -> go.Figure:
        """Create comparative analysis of multiple exotic matter types.
        
        Args:
            matter_list: List of exotic matter instances
            analysis_type: Type of analysis to compare
            
        Returns:
            Comparative analysis figure
        """
        
        print(f"Creating comparative {analysis_type} analysis...")
        
        if analysis_type == 'energy_conditions':
            fig = self._create_energy_condition_comparison(matter_list)
        elif analysis_type == 'stability':
            fig = self._create_stability_comparison(matter_list)
        elif analysis_type == 'properties':
            fig = self._create_properties_comparison(matter_list)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        print(f"Comparative {analysis_type} analysis created successfully")
        
        return fig
    
    def _create_energy_condition_comparison(self, matter_list: List[ExoticMatter]) -> go.Figure:
        """Create energy condition comparison plot."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'NEC Violations vs Distance',
                'WEC Violations vs Distance',
                'Violation Magnitudes',
                'Summary Statistics'
            ]
        )
        
        # Sample points for comparison
        r_sample = np.logspace(np.log10(self.config.r_min), np.log10(self.config.r_max), 50)
        
        colors = px.colors.qualitative.Set1[:len(matter_list)]
        
        for i, matter in enumerate(matter_list):
            nec_violations = []
            wec_violations = []
            violation_mags = []
            
            for r in r_sample:
                coords = (0.0, r, np.pi/2, 0.0)
                
                try:
                    ec_result = matter.check_energy_conditions(coords)
                    nec_violations.append(int(not ec_result.null_energy_condition))
                    wec_violations.append(int(not ec_result.weak_energy_condition))
                    violation_mags.append(ec_result.violation_magnitude)
                except:
                    nec_violations.append(0)
                    wec_violations.append(0)
                    violation_mags.append(0.0)
            
            # NEC violations
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=nec_violations,
                    mode='lines+markers',
                    name=f'{matter.name} NEC',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # WEC violations
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=wec_violations,
                    mode='lines+markers',
                    name=f'{matter.name} WEC',
                    line=dict(color=colors[i], width=2, dash='dash'),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Violation magnitudes
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=np.log10(np.array(violation_mags) + 1e-50),
                    mode='lines',
                    name=f'{matter.name} Magnitude',
                    line=dict(color=colors[i], width=3),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Summary statistics
        summary_data = []
        for matter in matter_list:
            # Compute summary statistics
            total_nec_violations = sum(nec_violations)
            total_wec_violations = sum(wec_violations)
            avg_violation_mag = np.mean(violation_mags)
            
            summary_data.append({
                'Matter Type': matter.name,
                'NEC Violations': total_nec_violations,
                'WEC Violations': total_wec_violations,
                'Avg Violation Mag': avg_violation_mag
            })
        
        # Create bar chart for summary
        matter_names = [matter.name for matter in matter_list]
        nec_totals = [sum(nec_violations) for matter in matter_list]
        
        fig.add_trace(
            go.Bar(
                x=matter_names,
                y=nec_totals,
                name='Total NEC Violations',
                marker_color='red',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Comparative Energy Condition Analysis',
            height=800,
            width=1200
        )
        
        # Update axes
        fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=1, col=1)
        fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=1, col=2)
        fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=2, col=1)
        fig.update_xaxes(title_text='Matter Type', row=2, col=2)
        
        fig.update_yaxes(title_text='NEC Violation', row=1, col=1)
        fig.update_yaxes(title_text='WEC Violation', row=1, col=2)
        fig.update_yaxes(title_text='log₁₀(Violation Mag)', row=2, col=1)
        fig.update_yaxes(title_text='Total Violations', row=2, col=2)
        
        return fig
    
    def _create_stability_comparison(self, matter_list: List[ExoticMatter]) -> go.Figure:
        """Create stability comparison plot."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Radial Sound Speeds',
                'Tangential Sound Speeds',
                'Eigenvalue Analysis',
                'Stability Scores'
            ]
        )
        
        r_sample = np.logspace(np.log10(self.config.r_min), np.log10(self.config.r_max), 30)
        colors = px.colors.qualitative.Set1[:len(matter_list)]
        
        for i, matter in enumerate(matter_list):
            radial_speeds = []
            tangential_speeds = []
            eigenvalue_mags = []
            stability_scores = []
            
            for r in r_sample:
                coords = (0.0, r, np.pi/2, 0.0)
                
                try:
                    stability = matter.stability_analysis(coords)
                    radial_speeds.append(stability.radial_sound_speed)
                    tangential_speeds.append(stability.tangential_sound_speed)
                    
                    eigenval_mag = abs(stability.radial_perturbation_eigenvalue)
                    eigenvalue_mags.append(eigenval_mag)
                    
                    # Simple stability score
                    score = (
                        (stability.radial_sound_speed < 3e8) * 0.25 +
                        (stability.tangential_sound_speed < 3e8) * 0.25 +
                        (stability.radial_perturbation_eigenvalue.real < 0) * 0.25 +
                        stability.causality_preserved * 0.25
                    )
                    stability_scores.append(score)
                    
                except:
                    radial_speeds.append(0.0)
                    tangential_speeds.append(0.0)
                    eigenvalue_mags.append(0.0)
                    stability_scores.append(0.0)
            
            # Plot data
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=radial_speeds,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=tangential_speeds,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=eigenvalue_mags,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=stability_scores,
                    mode='lines+markers',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Add speed of light reference line
        c_line_y = [3e8] * len(r_sample)
        fig.add_trace(
            go.Scatter(
                x=r_sample,
                y=c_line_y,
                mode='lines',
                name='Speed of Light',
                line=dict(color='black', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=r_sample,
                y=c_line_y,
                mode='lines',
                name='Speed of Light',
                line=dict(color='black', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Comparative Stability Analysis',
            height=800,
            width=1200
        )
        
        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=row, col=col)
        
        fig.update_yaxes(title_text='Radial Sound Speed (m/s)', row=1, col=1)
        fig.update_yaxes(title_text='Tangential Sound Speed (m/s)', row=1, col=2)
        fig.update_yaxes(title_text='|Eigenvalue|', row=2, col=1, type='log')
        fig.update_yaxes(title_text='Stability Score', row=2, col=2)
        
        return fig
    
    def _create_properties_comparison(self, matter_list: List[ExoticMatter]) -> go.Figure:
        """Create properties comparison plot."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Energy Density',
                'Radial Pressure',
                'Tangential Pressure',
                'Equation of State'
            ]
        )
        
        r_sample = np.logspace(np.log10(self.config.r_min), np.log10(self.config.r_max), 50)
        colors = px.colors.qualitative.Set1[:len(matter_list)]
        
        for i, matter in enumerate(matter_list):
            energy_densities = []
            radial_pressures = []
            tangential_pressures = []
            eos_parameters = []
            
            for r in r_sample:
                coords = (0.0, r, np.pi/2, 0.0)
                
                try:
                    rho = matter.energy_density(coords)
                    p_r = matter.pressure_radial(coords)
                    p_t = matter.pressure_tangential(coords)
                    
                    energy_densities.append(rho)
                    radial_pressures.append(p_r)
                    tangential_pressures.append(p_t)
                    
                    # Equation of state parameter
                    if abs(rho) > 1e-50:
                        w_r = p_r / rho
                    else:
                        w_r = 0.0
                    eos_parameters.append(w_r)
                    
                except:
                    energy_densities.append(0.0)
                    radial_pressures.append(0.0)
                    tangential_pressures.append(0.0)
                    eos_parameters.append(0.0)
            
            # Plot properties
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=energy_densities,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=radial_pressures,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=tangential_pressures,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=r_sample,
                    y=eos_parameters,
                    mode='lines',
                    name=f'{matter.name}',
                    line=dict(color=colors[i], width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Comparative Matter Properties',
            height=800,
            width=1200
        )
        
        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(title_text='Radial Distance (m)', type='log', row=row, col=col)
        
        fig.update_yaxes(title_text='Energy Density (J/m³)', row=1, col=1)
        fig.update_yaxes(title_text='Radial Pressure (Pa)', row=1, col=2)
        fig.update_yaxes(title_text='Tangential Pressure (Pa)', row=2, col=1)
        fig.update_yaxes(title_text='w = p/ρ', row=2, col=2)
        
        return fig
    
    def create_interactive_explorer(self, matter_catalog: List[str] = None) -> go.Figure:
        """Create interactive exotic matter explorer.
        
        Args:
            matter_catalog: List of matter types to include
            
        Returns:
            Interactive exploration dashboard
        """
        
        if matter_catalog is None:
            matter_catalog = ['advanced_casimir', 'phantom_dark_energy', 
                            'quantum_inequality', 'string_theory']
        
        print("Creating interactive exotic matter explorer...")
        
        # Create dashboard with dropdown for matter selection
        fig = go.Figure()
        
        # Default matter type
        default_matter = load_exotic_matter_from_catalog(matter_catalog[0])
        
        # Create initial visualization
        energy_conditions_fig = self.create_energy_condition_map(default_matter)
        
        # Extract traces from energy conditions figure
        for trace in energy_conditions_fig.data:
            fig.add_trace(trace)
        
        # Add dropdown menu for matter type selection
        dropdown_buttons = []
        
        for matter_type in matter_catalog:
            button = dict(
                label=matter_type.replace('_', ' ').title(),
                method='restyle',
                args=[{'visible': [False] * len(fig.data)}]  # Hide all traces initially
            )
            dropdown_buttons.append(button)
        
        fig.update_layout(
            title='Interactive Exotic Matter Explorer',
            updatemenus=[{
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.02
            }],
            height=800,
            width=1200
        )
        
        print("Interactive explorer created successfully")
        
        return fig
    
    def generate_comprehensive_report(self, matter: ExoticMatter,
                                    output_format: str = 'html') -> str:
        """Generate comprehensive visualization report.
        
        Args:
            matter: Exotic matter instance
            output_format: Output format ('html', 'pdf')
            
        Returns:
            Path to generated report
        """
        
        print(f"Generating comprehensive report for {matter.name}...")
        
        self.set_exotic_matter(matter)
        
        # Generate all visualizations
        energy_conditions_fig = self.create_energy_condition_map()
        stability_fig = self.create_stability_landscape()
        distribution_3d_fig = self.create_matter_distribution_3d()
        
        if output_format == 'html':
            # Create HTML report with embedded figures
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exotic_matter_report_{matter.name.replace(' ', '_')}_{timestamp}.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Exotic Matter Analysis Report: {matter.name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                    .section {{ margin: 20px 0; }}
                    .figure {{ margin: 20px 0; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Exotic Matter Analysis Report</h1>
                    <h2>{matter.name}</h2>
                    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Energy Condition Analysis</h2>
                    <div class="figure">
                        {energy_conditions_fig.to_html(include_plotlyjs='inline', div_id="energy_conditions")}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Stability Analysis</h2>
                    <div class="figure">
                        {stability_fig.to_html(include_plotlyjs='inline', div_id="stability")}
                    </div>
                </div>
                
                <div class="section">
                    <h2>3D Matter Distribution</h2>
                    <div class="figure">
                        {distribution_3d_fig.to_html(include_plotlyjs='inline', div_id="distribution_3d")}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Summary</h2>
                    <p>This report provides comprehensive analysis of the {matter.name} exotic matter configuration.</p>
                    <p>Key findings and physical constraints are visualized above.</p>
                </div>
            </body>
            </html>
            """
            
            with open(filename, 'w') as f:
                f.write(html_content)
            
            print(f"HTML report generated: {filename}")
            return filename
            
        else:
            raise ValueError(f"Output format {output_format} not supported")


def create_exotic_matter_showcase() -> Dict[str, go.Figure]:
    """Create showcase of all exotic matter visualization capabilities.
    
    Returns:
        Dictionary of visualization figures
    """
    
    print("Creating exotic matter visualization showcase...")
    
    # Initialize visualizer
    config = ExoticMatterVisualizationConfig(
        r_points=100,
        theta_points=50,
        colormap='RdBu_r'
    )
    
    visualizer = ExoticMatterVisualizer(config)
    
    # Create different exotic matter instances
    casimir_matter = AdvancedCasimirExoticMatter(
        plate_separation=5e-7,
        experimental_calibration='decca_2003'
    )
    
    phantom_matter = PhantomDarkEnergyField()
    
    qi_matter = QuantumInequalityConstrainedMatter(throat_radius=1e3)
    
    string_matter = StringTheoryDerivedMatter(string_model='heterotic')
    
    matter_list = [casimir_matter, phantom_matter, qi_matter, string_matter]
    
    # Create showcase figures
    figures = {}
    
    # Individual visualizations
    for matter in matter_list:
        matter_name = matter.name.replace(' ', '_').lower()
        
        # Energy condition map
        figures[f"{matter_name}_energy_conditions"] = visualizer.create_energy_condition_map(matter)
        
        # Stability landscape
        figures[f"{matter_name}_stability"] = visualizer.create_stability_landscape(matter)
        
        # 3D distribution
        figures[f"{matter_name}_3d_distribution"] = visualizer.create_matter_distribution_3d(matter)
    
    # Comparative analyses
    figures['comparative_energy_conditions'] = visualizer.create_comparative_analysis(
        matter_list, 'energy_conditions'
    )
    
    figures['comparative_stability'] = visualizer.create_comparative_analysis(
        matter_list, 'stability'
    )
    
    figures['comparative_properties'] = visualizer.create_comparative_analysis(
        matter_list, 'properties'
    )
    
    # Interactive explorer
    figures['interactive_explorer'] = visualizer.create_interactive_explorer()
    
    print(f"Showcase created with {len(figures)} visualizations")
    
    return figures


if __name__ == "__main__":
    # Demonstration
    print("Exotic Matter Visualization System Demo")
    print("="*50)
    
    # Create showcase
    showcase_figures = create_exotic_matter_showcase()
    
    # Save selected figures
    showcase_figures['comparative_energy_conditions'].write_html("comparative_energy_conditions.html")
    showcase_figures['comparative_stability'].write_html("comparative_stability.html")
    showcase_figures['interactive_explorer'].write_html("interactive_exotic_matter_explorer.html")
    
    print("\nDemo completed!")
    print("Generated files:")
    print("- comparative_energy_conditions.html")
    print("- comparative_stability.html") 
    print("- interactive_exotic_matter_explorer.html")
    print("\nOpen these files in your browser for interactive exploration!")