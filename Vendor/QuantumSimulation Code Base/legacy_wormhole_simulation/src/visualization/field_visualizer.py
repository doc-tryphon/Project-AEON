"""
Electromagnetic and gravitational field visualization.

This module provides comprehensive visualization capabilities for:
- Electromagnetic field lines and flux
- Gravitational field visualization
- Exotic matter field distributions
- Stress-energy tensor components
- Field interaction dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.interpolate import griddata
from scipy.special import sph_harm

from src.physics.spacetime_metrics import SpacetimeMetric
from src.physics.stress_energy_tensor import StressEnergyTensor
from src.physics.exotic_matter import ExoticMatter
from src.physics.constants import C, G, E_0 as EPSILON_0, MU_0


@dataclass
class FieldVisualizationConfig:
    """Configuration for field visualization."""
    
    # Grid parameters
    field_resolution: int = 30
    streamline_density: int = 20
    vector_scale: float = 1.0
    
    # Visualization parameters
    colormap: str = 'plasma'
    alpha: float = 0.7
    line_width: float = 2.0
    arrow_scale: float = 1.0
    
    # Field calculation parameters
    max_field_strength: float = 1e10
    min_field_strength: float = 1e-15
    logarithmic_scale: bool = True
    
    # Integration parameters
    streamline_max_length: float = 100.0
    streamline_step_size: float = 0.1


class FieldVisualizer:
    """Advanced field visualization for electromagnetic and gravitational fields."""
    
    def __init__(self, config: FieldVisualizationConfig = None):
        """Initialize field visualizer.
        
        Args:
            config: Field visualization configuration
        """
        self.config = config or FieldVisualizationConfig()
        
        # Cached field calculations
        self._field_cache = {}
        self._streamline_cache = {}
        
    def visualize_electromagnetic_field(self, charge_distribution: Callable[[np.ndarray], float],
                                      current_distribution: Callable[[np.ndarray], np.ndarray] = None,
                                      region_bounds: Tuple[float, float, float] = (-5, 5, 10)) -> go.Figure:
        """Visualize electromagnetic field from charge and current distributions.
        
        Args:
            charge_distribution: Function ρ(r) giving charge density
            current_distribution: Function J(r) giving current density vector
            region_bounds: (x_range, y_range, z_range) for visualization region
        
        Returns:
            Plotly figure with electromagnetic field visualization
        """
        
        x_range, y_range, z_range = region_bounds
        
        # Create coordinate grids
        x = np.linspace(-x_range, x_range, self.config.field_resolution)
        y = np.linspace(-y_range, y_range, self.config.field_resolution)
        z = np.linspace(-z_range/2, z_range/2, self.config.field_resolution//2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate electric and magnetic fields
        E_field = np.zeros((*X.shape, 3))
        B_field = np.zeros((*X.shape, 3)) if current_distribution else None
        charge_density = np.zeros(X.shape)
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    r = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    
                    # Calculate charge density
                    charge_density[i,j,k] = charge_distribution(r)
                    
                    # Calculate electric field from all charges (simplified)
                    E_total = np.zeros(3)
                    
                    # Integrate over charge distribution (simplified approach)
                    for ii in range(0, len(x), 2):  # Sample for performance
                        for jj in range(0, len(y), 2):
                            for kk in range(0, len(z), 2):
                                r_source = np.array([x[ii], y[jj], z[kk]])
                                rho = charge_distribution(r_source)
                                
                                if abs(rho) > self.config.min_field_strength:
                                    r_vec = r - r_source
                                    r_mag = np.linalg.norm(r_vec)
                                    
                                    if r_mag > 1e-6:  # Avoid singularity
                                        E_contribution = rho * r_vec / (4 * np.pi * EPSILON_0 * r_mag**3)
                                        E_total += E_contribution * (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
                    
                    # Clamp field strength
                    E_mag = np.linalg.norm(E_total)
                    if E_mag > self.config.max_field_strength:
                        E_total = E_total / E_mag * self.config.max_field_strength
                    
                    E_field[i,j,k] = E_total
                    
                    # Calculate magnetic field from current (if provided)
                    if current_distribution is not None:
                        B_total = np.zeros(3)
                        
                        # Biot-Savart law (simplified)
                        for ii in range(0, len(x), 3):
                            for jj in range(0, len(y), 3):
                                for kk in range(0, len(z), 3):
                                    r_source = np.array([x[ii], y[jj], z[kk]])
                                    J = current_distribution(r_source)
                                    
                                    if np.linalg.norm(J) > self.config.min_field_strength:
                                        r_vec = r - r_source
                                        r_mag = np.linalg.norm(r_vec)
                                        
                                        if r_mag > 1e-6:
                                            B_contribution = MU_0 * np.cross(J, r_vec) / (4 * np.pi * r_mag**3)
                                            B_total += B_contribution * (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
                        
                        B_field[i,j,k] = B_total
        
        # Create visualization
        fig = go.Figure()
        
        # Charge density visualization
        charge_nonzero = np.abs(charge_density) > self.config.min_field_strength
        
        if np.any(charge_nonzero):
            # Positive charges
            pos_charges = charge_density > self.config.min_field_strength
            if np.any(pos_charges):
                fig.add_trace(go.Scatter3d(
                    x=X[pos_charges],
                    y=Y[pos_charges], 
                    z=Z[pos_charges],
                    mode='markers',
                    marker=dict(
                        size=np.abs(charge_density[pos_charges]) * 1e12,
                        color='red',
                        opacity=0.8
                    ),
                    name='Positive Charges'
                ))
            
            # Negative charges
            neg_charges = charge_density < -self.config.min_field_strength
            if np.any(neg_charges):
                fig.add_trace(go.Scatter3d(
                    x=X[neg_charges],
                    y=Y[neg_charges],
                    z=Z[neg_charges],
                    mode='markers',
                    marker=dict(
                        size=np.abs(charge_density[neg_charges]) * 1e12,
                        color='blue',
                        opacity=0.8
                    ),
                    name='Negative Charges'
                ))
        
        # Electric field vectors (sample for visualization)
        step = 3  # Sample every 3rd point
        for i in range(0, len(x), step):
            for j in range(0, len(y), step):
                for k in range(0, len(z), step):
                    E_vec = E_field[i,j,k]
                    E_mag = np.linalg.norm(E_vec)
                    
                    if E_mag > self.config.min_field_strength:
                        # Normalize and scale arrow
                        E_normalized = E_vec / E_mag * self.config.arrow_scale
                        
                        fig.add_trace(go.Scatter3d(
                            x=[X[i,j,k], X[i,j,k] + E_normalized[0]],
                            y=[Y[i,j,k], Y[i,j,k] + E_normalized[1]],
                            z=[Z[i,j,k], Z[i,j,k] + E_normalized[2]],
                            mode='lines',
                            line=dict(
                                color=np.log10(E_mag) if self.config.logarithmic_scale else E_mag,
                                colorscale='Viridis',
                                width=3
                            ),
                            showlegend=False
                        ))
        
        # Magnetic field vectors (if available)
        if B_field is not None:
            for i in range(0, len(x), step):
                for j in range(0, len(y), step):
                    for k in range(0, len(z), step):
                        B_vec = B_field[i,j,k]
                        B_mag = np.linalg.norm(B_vec)
                        
                        if B_mag > self.config.min_field_strength:
                            B_normalized = B_vec / B_mag * self.config.arrow_scale * 0.5
                            
                            fig.add_trace(go.Scatter3d(
                                x=[X[i,j,k], X[i,j,k] + B_normalized[0]],
                                y=[Y[i,j,k], Y[i,j,k] + B_normalized[1]],
                                z=[Z[i,j,k], Z[i,j,k] + B_normalized[2]],
                                mode='lines',
                                line=dict(
                                    color='purple',
                                    width=2,
                                    dash='dash'
                                ),
                                showlegend=False
                            ))
        
        fig.update_layout(
            title='Electromagnetic Field Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_gravitational_field(self, metric: SpacetimeMetric,
                                    region_bounds: Tuple[float, float, float] = (10, 10, 10)) -> go.Figure:
        """Visualize gravitational field from spacetime metric.
        
        Args:
            metric: Spacetime metric
            region_bounds: Spatial bounds for visualization
        
        Returns:
            Gravitational field visualization
        """
        
        x_range, y_range, z_range = region_bounds
        
        # Create coordinate grid
        x = np.linspace(-x_range, x_range, self.config.field_resolution)
        y = np.linspace(-y_range, y_range, self.config.field_resolution) 
        z = np.linspace(-z_range, z_range, self.config.field_resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate gravitational field (tidal forces)
        gravitational_field = np.zeros((*X.shape, 3))
        curvature_scalar = np.zeros(X.shape)
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    # Convert to spherical coordinates for metric evaluation
                    r = np.sqrt(X[i,j,k]**2 + Y[i,j,k]**2 + Z[i,j,k]**2)
                    
                    if r > 1e-6:  # Avoid singularity
                        theta = np.arccos(Z[i,j,k] / r)
                        phi = np.arctan2(Y[i,j,k], X[i,j,k])
                        
                        coordinates = (0.0, r, theta, phi)  # t=0 slice
                        
                        try:
                            # Calculate gravitational acceleration from metric
                            g_tensor = metric.metric_tensor(coordinates)
                            
                            # Simplified tidal force calculation
                            # In practice, this would use Riemann tensor
                            det_g = np.linalg.det(g_tensor)
                            curvature_scalar[i,j,k] = np.log(abs(det_g)) if det_g != 0 else 0
                            
                            # Approximate gravitational field direction
                            if hasattr(metric, 'mass') and metric.mass > 0:
                                # Radial gravitational field
                                g_magnitude = G * metric.mass / r**2
                                g_direction = -np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]) / r
                                gravitational_field[i,j,k] = g_magnitude * g_direction
                            else:
                                # Use metric curvature to estimate field
                                if r > 1:  # Outside throat
                                    field_mag = abs(curvature_scalar[i,j,k]) * 1e-10
                                    field_dir = -np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]) / r
                                    gravitational_field[i,j,k] = field_mag * field_dir
                            
                        except Exception as e:
                            gravitational_field[i,j,k] = np.zeros(3)
                            curvature_scalar[i,j,k] = 0
        
        # Create visualization
        fig = go.Figure()
        
        # Curvature scalar as volume rendering
        if np.any(np.abs(curvature_scalar) > self.config.min_field_strength):
            # Sample points for volume visualization
            sample_mask = (np.abs(curvature_scalar) > np.percentile(np.abs(curvature_scalar), 90))
            
            if np.any(sample_mask):
                fig.add_trace(go.Scatter3d(
                    x=X[sample_mask],
                    y=Y[sample_mask],
                    z=Z[sample_mask],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=curvature_scalar[sample_mask],
                        colorscale=self.config.colormap,
                        opacity=self.config.alpha,
                        colorbar=dict(title='Curvature Scalar')
                    ),
                    name='Spacetime Curvature'
                ))
        
        # Gravitational field vectors
        step = 4  # Sample for visualization
        for i in range(0, len(x), step):
            for j in range(0, len(y), step):
                for k in range(0, len(z), step):
                    g_vec = gravitational_field[i,j,k]
                    g_mag = np.linalg.norm(g_vec)
                    
                    if g_mag > self.config.min_field_strength:
                        # Scale vector for visibility
                        g_scaled = g_vec / g_mag * self.config.arrow_scale * np.log10(g_mag + 1)
                        
                        fig.add_trace(go.Scatter3d(
                            x=[X[i,j,k], X[i,j,k] + g_scaled[0]],
                            y=[Y[i,j,k], Y[i,j,k] + g_scaled[1]],
                            z=[Z[i,j,k], Z[i,j,k] + g_scaled[2]],
                            mode='lines',
                            line=dict(
                                color='red',
                                width=2
                            ),
                            showlegend=False
                        ))
        
        # Add wormhole throat if applicable
        if hasattr(metric, 'b0'):
            throat_radius = metric.b0
            
            # Create throat visualization
            phi_throat = np.linspace(0, 2*np.pi, 50)
            theta_throat = np.linspace(0, np.pi, 25)
            
            phi_mesh, theta_mesh = np.meshgrid(phi_throat, theta_throat)
            
            x_throat = throat_radius * np.sin(theta_mesh) * np.cos(phi_mesh)
            y_throat = throat_radius * np.sin(theta_mesh) * np.sin(phi_mesh)
            z_throat = throat_radius * np.cos(theta_mesh)
            
            fig.add_trace(go.Surface(
                x=x_throat, y=y_throat, z=z_throat,
                opacity=0.5,
                colorscale='Greys',
                showscale=False,
                name='Wormhole Throat'
            ))
        
        fig.update_layout(
            title='Gravitational Field Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_exotic_matter_field(self, exotic_matter: ExoticMatter,
                                    region_bounds: Tuple[float, float, float] = (5, 5, 5)) -> go.Figure:
        """Visualize exotic matter field distribution.
        
        Args:
            exotic_matter: Exotic matter configuration
            region_bounds: Spatial bounds for visualization
        
        Returns:
            Exotic matter field visualization
        """
        
        x_range, y_range, z_range = region_bounds
        
        # Create coordinate grid
        x = np.linspace(-x_range, x_range, self.config.field_resolution)
        y = np.linspace(-y_range, y_range, self.config.field_resolution)
        z = np.linspace(-z_range, z_range, self.config.field_resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate exotic matter properties
        energy_density = np.zeros(X.shape)
        pressure_field = np.zeros((*X.shape, 3))  # Pressure gradient
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    r = np.sqrt(X[i,j,k]**2 + Y[i,j,k]**2 + Z[i,j,k]**2)
                    
                    if r > 1e-6:
                        theta = np.arccos(Z[i,j,k] / r) if r > abs(Z[i,j,k]) else 0
                        phi = np.arctan2(Y[i,j,k], X[i,j,k])
                        
                        coordinates = (0.0, r, theta, phi)
                        
                        try:
                            # Calculate energy density
                            rho = exotic_matter.energy_density(coordinates)
                            energy_density[i,j,k] = rho
                            
                            # Calculate pressure gradient (simplified)
                            p = exotic_matter.pressure(coordinates)
                            if r > 0.1:  # Avoid numerical issues
                                dr = 0.01
                                r_plus = r + dr
                                r_minus = max(r - dr, 0.01)
                                
                                coords_plus = (0.0, r_plus, theta, phi)
                                coords_minus = (0.0, r_minus, theta, phi)
                                
                                try:
                                    p_plus = exotic_matter.pressure(coords_plus)
                                    p_minus = exotic_matter.pressure(coords_minus)
                                    dp_dr = (p_plus - p_minus) / (2 * dr)
                                    
                                    # Pressure gradient in radial direction
                                    r_hat = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]) / r
                                    pressure_field[i,j,k] = -dp_dr * r_hat  # Negative for force direction
                                except:
                                    pressure_field[i,j,k] = np.zeros(3)
                            
                        except Exception:
                            energy_density[i,j,k] = 0
                            pressure_field[i,j,k] = np.zeros(3)
        
        # Create visualization
        fig = go.Figure()
        
        # Energy density visualization
        energy_nonzero = np.abs(energy_density) > self.config.min_field_strength
        
        if np.any(energy_nonzero):
            # Negative energy density (exotic matter)
            negative_energy = energy_density < -self.config.min_field_strength
            if np.any(negative_energy):
                fig.add_trace(go.Scatter3d(
                    x=X[negative_energy],
                    y=Y[negative_energy],
                    z=Z[negative_energy],
                    mode='markers',
                    marker=dict(
                        size=np.abs(energy_density[negative_energy]) * 1e20,
                        color=energy_density[negative_energy],
                        colorscale='Blues_r',
                        opacity=0.8,
                        colorbar=dict(title='Energy Density')
                    ),
                    name='Negative Energy Density'
                ))
            
            # Positive energy density
            positive_energy = energy_density > self.config.min_field_strength
            if np.any(positive_energy):
                fig.add_trace(go.Scatter3d(
                    x=X[positive_energy],
                    y=Y[positive_energy],
                    z=Z[positive_energy],
                    mode='markers',
                    marker=dict(
                        size=energy_density[positive_energy] * 1e20,
                        color='orange',
                        opacity=0.6
                    ),
                    name='Positive Energy Density'
                ))
        
        # Pressure gradient field vectors
        step = 3
        for i in range(0, len(x), step):
            for j in range(0, len(y), step):
                for k in range(0, len(z), step):
                    p_grad = pressure_field[i,j,k]
                    p_mag = np.linalg.norm(p_grad)
                    
                    if p_mag > self.config.min_field_strength:
                        p_scaled = p_grad / p_mag * self.config.arrow_scale
                        
                        fig.add_trace(go.Scatter3d(
                            x=[X[i,j,k], X[i,j,k] + p_scaled[0]],
                            y=[Y[i,j,k], Y[i,j,k] + p_scaled[1]],
                            z=[Z[i,j,k], Z[i,j,k] + p_scaled[2]],
                            mode='lines',
                            line=dict(
                                color='green',
                                width=2
                            ),
                            showlegend=False
                        ))
        
        fig.update_layout(
            title='Exotic Matter Field Distribution',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_stress_energy_tensor(self, stress_energy: StressEnergyTensor,
                                     tensor_component: Tuple[int, int] = (0, 0),
                                     region_bounds: Tuple[float, float, float] = (5, 5, 5)) -> go.Figure:
        """Visualize stress-energy tensor components.
        
        Args:
            stress_energy: Stress-energy tensor
            tensor_component: Which component to visualize (μ, ν)
            region_bounds: Spatial bounds for visualization
        
        Returns:
            Stress-energy tensor visualization
        """
        
        x_range, y_range, z_range = region_bounds
        mu, nu = tensor_component
        
        # Create coordinate grid
        x = np.linspace(-x_range, x_range, self.config.field_resolution)
        y = np.linspace(-y_range, y_range, self.config.field_resolution)
        z = np.linspace(-z_range, z_range, self.config.field_resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate tensor component
        tensor_values = np.zeros(X.shape)
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    r = np.sqrt(X[i,j,k]**2 + Y[i,j,k]**2 + Z[i,j,k]**2)
                    
                    if r > 1e-6:
                        theta = np.arccos(Z[i,j,k] / r) if r > abs(Z[i,j,k]) else 0
                        phi = np.arctan2(Y[i,j,k], X[i,j,k])
                        
                        coordinates = (0.0, r, theta, phi)
                        
                        try:
                            T = stress_energy.compute_tensor(coordinates)
                            tensor_values[i,j,k] = T[mu, nu]
                        except Exception:
                            tensor_values[i,j,k] = 0
        
        # Create isosurfaces for different tensor values
        fig = go.Figure()
        
        # Find significant values
        max_val = np.max(np.abs(tensor_values))
        if max_val > self.config.min_field_strength:
            levels = np.linspace(-max_val, max_val, 10)
            
            for level in levels[::2]:  # Sample levels
                if abs(level) > max_val * 0.1:  # Only significant levels
                    # Create isosurface (simplified as scatter points)
                    mask = np.abs(tensor_values - level) < max_val * 0.05
                    
                    if np.any(mask):
                        fig.add_trace(go.Scatter3d(
                            x=X[mask],
                            y=Y[mask],
                            z=Z[mask],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color=level,
                                colorscale=self.config.colormap,
                                opacity=0.6
                            ),
                            name=f'T_{mu}{nu} = {level:.2e}'
                        ))
        
        # Add tensor field visualization as vector field for off-diagonal components
        if mu != nu:
            # Interpret as vector field component
            step = 4
            for i in range(0, len(x), step):
                for j in range(0, len(y), step):
                    for k in range(0, len(z), step):
                        T_val = tensor_values[i,j,k]
                        
                        if abs(T_val) > self.config.min_field_strength:
                            # Create vector representation
                            vec_direction = np.zeros(3)
                            if mu < 3 and nu < 3:  # Spatial components
                                vec_direction[mu-1] = T_val if mu > 0 else 0
                                vec_direction[nu-1] += T_val if nu > 0 else 0
                                
                                vec_mag = np.linalg.norm(vec_direction)
                                if vec_mag > 0:
                                    vec_scaled = vec_direction / vec_mag * self.config.arrow_scale
                                    
                                    fig.add_trace(go.Scatter3d(
                                        x=[X[i,j,k], X[i,j,k] + vec_scaled[0]],
                                        y=[Y[i,j,k], Y[i,j,k] + vec_scaled[1]],
                                        z=[Z[i,j,k], Z[i,j,k] + vec_scaled[2]],
                                        mode='lines',
                                        line=dict(
                                            color=T_val,
                                            colorscale='RdBu',
                                            width=2
                                        ),
                                        showlegend=False
                                    ))
        
        component_names = ['t', 'r', 'θ', 'φ']
        fig.update_layout(
            title=f'Stress-Energy Tensor Component T_{component_names[mu]}{component_names[nu]}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_field_streamlines(self, vector_field: Callable[[np.ndarray], np.ndarray],
                               start_points: List[np.ndarray],
                               region_bounds: Tuple[float, float, float] = (5, 5, 5)) -> go.Figure:
        """Create streamlines for vector fields.
        
        Args:
            vector_field: Function returning 3D vector at each point
            start_points: Starting points for streamlines
            region_bounds: Spatial bounds
        
        Returns:
            Streamline visualization
        """
        
        def integrate_streamline(start_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Integrate single streamline."""
            
            def field_derivative(t, y):
                if np.any(np.abs(y) > max(region_bounds)):
                    return np.zeros(3)  # Stop integration outside bounds
                
                try:
                    field = vector_field(y)
                    field_mag = np.linalg.norm(field)
                    
                    if field_mag > self.config.min_field_strength:
                        return field / field_mag  # Unit tangent vector
                    else:
                        return np.zeros(3)
                except:
                    return np.zeros(3)
            
            # Integrate streamline
            t_span = (0, self.config.streamline_max_length)
            t_eval = np.linspace(0, self.config.streamline_max_length, 
                               int(self.config.streamline_max_length / self.config.streamline_step_size))
            
            try:
                sol = solve_ivp(field_derivative, t_span, start_point, 
                              t_eval=t_eval, rtol=1e-6, atol=1e-9)
                
                return sol.t, sol.y.T
            except:
                return np.array([0]), start_point.reshape(1, -1)
        
        # Calculate streamlines
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, start_point in enumerate(start_points):
            t_vals, positions = integrate_streamline(start_point)
            
            if len(positions) > 1:
                # Calculate field magnitude along streamline for coloring
                field_magnitudes = []
                for pos in positions:
                    try:
                        field = vector_field(pos)
                        field_magnitudes.append(np.linalg.norm(field))
                    except:
                        field_magnitudes.append(0)
                
                field_magnitudes = np.array(field_magnitudes)
                
                # Plot streamline
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    line=dict(
                        color=field_magnitudes if len(field_magnitudes) > 1 else colors[i % len(colors)],
                        colorscale='Viridis' if len(field_magnitudes) > 1 else None,
                        width=self.config.line_width
                    ),
                    marker=dict(size=2),
                    name=f'Streamline {i+1}'
                ))
                
                # Mark starting point
                fig.add_trace(go.Scatter3d(
                    x=[start_point[0]],
                    y=[start_point[1]], 
                    z=[start_point[2]],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='diamond'),
                    name=f'Start {i+1}',
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Field Streamlines',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_comprehensive_field_view(self, metric: SpacetimeMetric,
                                      exotic_matter: ExoticMatter = None,
                                      stress_energy: StressEnergyTensor = None) -> go.Figure:
        """Create comprehensive multi-panel field visualization.
        
        Args:
            metric: Spacetime metric
            exotic_matter: Exotic matter configuration
            stress_energy: Stress-energy tensor
        
        Returns:
            Multi-panel field visualization
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Gravitational Field', 'Exotic Matter Field',
                          'Stress-Energy Tensor', 'Field Interactions'],
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'scene'}, {'type': 'scene'}]]
        )
        
        # Gravitational field
        grav_fig = self.visualize_gravitational_field(metric)
        for trace in grav_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Exotic matter field
        if exotic_matter:
            exotic_fig = self.visualize_exotic_matter_field(exotic_matter)
            for trace in exotic_fig.data:
                fig.add_trace(trace, row=1, col=2)
        
        # Stress-energy tensor
        if stress_energy:
            tensor_fig = self.visualize_stress_energy_tensor(stress_energy)
            for trace in tensor_fig.data:
                fig.add_trace(trace, row=2, col=1)
        
        # Combined field interactions would go in (2,2)
        
        fig.update_layout(
            title='Comprehensive Field Analysis',
            height=800,
            width=1200
        )
        
        return fig


def create_electromagnetic_dipole_field(dipole_moment: np.ndarray,
                                      dipole_position: np.ndarray = np.array([0, 0, 0])) -> Callable[[np.ndarray], np.ndarray]:
    """Create electromagnetic field function for electric dipole.
    
    Args:
        dipole_moment: Electric dipole moment vector
        dipole_position: Position of dipole
    
    Returns:
        Function computing electric field at any point
    """
    
    def electric_field(r: np.ndarray) -> np.ndarray:
        """Electric field of dipole at position r."""
        
        r_vec = r - dipole_position
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e-6:
            return np.zeros(3)
        
        r_hat = r_vec / r_mag
        
        # Electric dipole field
        k = 1 / (4 * np.pi * EPSILON_0)
        
        E_field = k / r_mag**3 * (3 * np.dot(dipole_moment, r_hat) * r_hat - dipole_moment)
        
        return E_field
    
    return electric_field