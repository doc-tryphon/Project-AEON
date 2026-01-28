"""
4D spacetime rendering with interactive 3D projections.

This module provides advanced visualization of wormhole geometry, geodesics,
and spacetime curvature using interactive 3D plots and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from matplotlib import cm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# BasicSpacetimePlotter class removed to avoid conflicts with advanced SpacetimePlotter
        
    def plot_embedding_diagram(self,
                            throat_radius: float,
                            length: float) -> None:
        """Create 3D embedding diagram of wormhole geometry.
        
        Args:
            throat_radius: Radius of wormhole throat
            length: Length of wormhole tunnel
        """
        if self.config.interactive:
            self._plot_interactive_embedding(throat_radius, length)
        else:
            self._plot_static_embedding(throat_radius, length)
            
    def _plot_static_embedding(self,
                            throat_radius: float,
                            length: float) -> None:
        """Create static matplotlib embedding diagram."""
        # Set up coordinate grid
        r = np.linspace(-length, length, self.config.resolution)
        theta = np.linspace(0, 2*np.pi, self.config.resolution)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Calculate embedding surface
        z = r_grid
        rho = np.sqrt(throat_radius**2 + r_grid**2)
        x = rho * np.cos(theta_grid)
        y = rho * np.sin(theta_grid)
        
        # Create 3D plot
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = self.ax.plot_surface(x, y, z,
                                 cmap=self.config.colormap,
                                 alpha=0.8)
        
        # Customize plot
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_title('Wormhole Embedding Diagram')
        
        if self.config.show_grid:
            self.ax.grid(True)
            
        # Add colorbar
        self.fig.colorbar(surf)
            
    def _plot_interactive_embedding(self,
                                 throat_radius: float,
                                 length: float) -> None:
        """Create interactive plotly embedding diagram."""
        # Set up coordinate grid
        r = np.linspace(-length, length, self.config.resolution)
        theta = np.linspace(0, 2*np.pi, self.config.resolution)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Calculate embedding surface
        z = r_grid
        rho = np.sqrt(throat_radius**2 + r_grid**2)
        x = rho * np.cos(theta_grid)
        y = rho * np.sin(theta_grid)
        
        # Create surface plot
        surface = go.Surface(x=x, y=y, z=z,
                          colorscale=self.config.colormap)
        
        # Set up figure
        fig = go.Figure(data=[surface])
        
        # Update layout
        fig.update_layout(
            title='Interactive Wormhole Embedding Diagram',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=800,
            height=800
        )
        
        if not self.config.show_grid:
            fig.update_layout(scene=dict(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False)
            ))
            
        # Store figure
        self.fig = fig
        
    def plot_metric_components(self,
                            coordinates: np.ndarray,
                            metric: np.ndarray) -> None:
        """Plot metric tensor components.
        
        Args:
            coordinates: Spacetime coordinates
            metric: Metric tensor components
        """
        # Number of components to plot
        n = metric.shape[0]
        
        # Create subplots
        fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))
        
        # Plot each component
        for i in range(n):
            for j in range(n):
                if n > 1:
                    ax = axes[i,j]
                else:
                    ax = axes
                    
                im = ax.contourf(coordinates[0], coordinates[1],
                              metric[i,j], cmap=self.config.colormap)
                plt.colorbar(im, ax=ax)
                
                ax.set_title(f'g_{i}{j}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
                if self.config.show_grid:
                    ax.grid(True)
                    
        plt.tight_layout()
        self.fig = fig
        
    def plot_geodesics(self,
                     initial_positions: np.ndarray,
                     trajectories: List[np.ndarray]) -> None:
        """Plot geodesic trajectories.
        
        Args:
            initial_positions: Starting points
            trajectories: List of trajectory coordinate arrays
        """
        if self.config.interactive:
            self._plot_interactive_geodesics(initial_positions, trajectories)
        else:
            self._plot_static_geodesics(initial_positions, trajectories)
            
    def _plot_static_geodesics(self,
                             initial_positions: np.ndarray,
                             trajectories: List[np.ndarray]) -> None:
        """Create static geodesic plot."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each trajectory
        for pos, traj in zip(initial_positions, trajectories):
            ax.plot3D(traj[:,0], traj[:,1], traj[:,2],
                    label=f'Start: {tuple(pos)}')
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Geodesic Trajectories')
        
        if self.config.show_grid:
            ax.grid(True)
            
        ax.legend()
        self.fig = fig
        self.ax = ax
        
    def _plot_interactive_geodesics(self,
                                 initial_positions: np.ndarray,
                                 trajectories: List[np.ndarray]) -> None:
        """Create interactive geodesic plot."""
        fig = go.Figure()
        
        # Plot each trajectory
        for pos, traj in zip(initial_positions, trajectories):
            fig.add_trace(go.Scatter3d(
                x=traj[:,0], y=traj[:,1], z=traj[:,2],
                mode='lines',
                name=f'Start: {tuple(pos)}'
            ))
            
        # Update layout
        fig.update_layout(
            title='Interactive Geodesic Trajectories',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=800,
            height=800
        )
        
        self.fig = fig
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.spatial import SphericalVoronoi
from scipy.interpolate import griddata, interp1d

from src.physics.spacetime_metrics import SpacetimeMetric, MorrisThorneeWormhole
from src.physics.constants import C, G, PLANCK_LENGTH


@dataclass
class SpacetimeVisualizationConfig:
    """Configuration for spacetime visualization."""
    
    # Grid parameters
    r_min: float = 1e3
    r_max: float = 1e6
    theta_range: Tuple[float, float] = (0, np.pi)
    phi_range: Tuple[float, float] = (0, 2*np.pi)
    grid_resolution: int = 50
    
    # Time parameters
    t_range: Tuple[float, float] = (-100, 100)
    time_steps: int = 100
    
    # Visualization parameters
    colormap: str = 'viridis'
    opacity: float = 0.8
    line_width: float = 2.0
    marker_size: float = 5.0
    
    # Curvature visualization
    curvature_scale: float = 1e-10
    max_curvature_display: float = 1e15
    
    # Geodesic parameters
    geodesic_steps: int = 1000
    geodesic_time: float = 100.0


class SpacetimePlotter:
    """Advanced 4D spacetime visualization with 3D projections."""
    
    def __init__(self, metric: SpacetimeMetric, config: SpacetimeVisualizationConfig = None):
        """Initialize spacetime plotter.
        
        Args:
            metric: Spacetime metric to visualize
            config: Visualization configuration
        """
        self.metric = metric
        self.config = config or SpacetimeVisualizationConfig()
        
        # Precompute grids
        self.r_grid = None
        self.theta_grid = None
        self.phi_grid = None
        self.t_grid = None
        self._setup_coordinate_grids()
        
        # Cached computations
        self._curvature_cache = {}
        self._geodesic_cache = {}
        
    def _setup_coordinate_grids(self):
        """Set up coordinate grids for visualization."""
        
        # Radial grid (logarithmic for wormhole visualization)
        self.r_grid = np.logspace(
            np.log10(self.config.r_min), 
            np.log10(self.config.r_max),
            self.config.grid_resolution
        )
        
        # Angular grids
        self.theta_grid = np.linspace(
            self.config.theta_range[0], 
            self.config.theta_range[1],
            self.config.grid_resolution
        )
        
        self.phi_grid = np.linspace(
            self.config.phi_range[0], 
            self.config.phi_range[1],
            self.config.grid_resolution
        )
        
        # Time grid
        self.t_grid = np.linspace(
            self.config.t_range[0], 
            self.config.t_range[1],
            self.config.time_steps
        )
    
    def plot_wormhole_geometry_3d(self, time_slice: float = 0.0,
                                 visualization_type: str = 'surface') -> go.Figure:
        """Create 3D visualization of wormhole geometry.
        
        Args:
            time_slice: Time coordinate for spatial slice
            visualization_type: Type of visualization ('surface', 'wireframe', 'contour')
        
        Returns:
            Plotly figure object
        """
        
        # Create coordinate meshes for equatorial plane
        phi_2d, r_2d = np.meshgrid(self.phi_grid, self.r_grid)
        theta_fixed = np.pi / 2  # Equatorial plane
        
        # Convert to Cartesian coordinates
        x = r_2d * np.cos(phi_2d)
        y = r_2d * np.sin(phi_2d)
        
        # Compute metric signature as height (simplified)
        z = np.zeros_like(x)
        metric_det = np.zeros_like(x)
        
        for i, r in enumerate(self.r_grid):
            for j, phi in enumerate(self.phi_grid):
                coordinates = (time_slice, r, theta_fixed, phi)
                
                try:
                    g = self.metric.metric_tensor(coordinates)
                    det_g = np.linalg.det(g)
                    
                    # Use metric determinant to show geometry
                    metric_det[i, j] = np.log(abs(det_g)) if det_g != 0 else 0
                    
                    # For wormhole, show throat structure
                    if hasattr(self.metric, 'shape_function'):
                        b = self.metric.shape_function(r)
                        z[i, j] = np.sqrt(max(0, r**2 - b**2)) * 0.1  # Embedding height
                    else:
                        z[i, j] = 0
                        
                except Exception as e:
                    metric_det[i, j] = 0
                    z[i, j] = 0
        
        fig = go.Figure()
        
        if visualization_type == 'surface':
            # 3D surface plot
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=metric_det,
                colorscale=self.config.colormap,
                opacity=self.config.opacity,
                name='Wormhole Geometry'
            ))
            
        elif visualization_type == 'wireframe':
            # Wireframe representation
            for i in range(0, len(self.r_grid), 5):
                fig.add_trace(go.Scatter3d(
                    x=x[i, :], y=y[i, :], z=z[i, :],
                    mode='lines',
                    line=dict(width=self.config.line_width, color='blue'),
                    showlegend=False
                ))
            
            for j in range(0, len(self.phi_grid), 5):
                fig.add_trace(go.Scatter3d(
                    x=x[:, j], y=y[:, j], z=z[:, j],
                    mode='lines',
                    line=dict(width=self.config.line_width, color='red'),
                    showlegend=False
                ))
        
        elif visualization_type == 'contour':
            # Contour lines at different heights
            levels = np.linspace(z.min(), z.max(), 10)
            
            for level in levels:
                contour_mask = np.abs(z - level) < (z.max() - z.min()) / 20
                if np.any(contour_mask):
                    contour_x = x[contour_mask]
                    contour_y = y[contour_mask]
                    contour_z = z[contour_mask]
                    
                    fig.add_trace(go.Scatter3d(
                        x=contour_x, y=contour_y, z=contour_z,
                        mode='markers',
                        marker=dict(size=2, color=level, colorscale=self.config.colormap),
                        showlegend=False
                    ))
        
        # Formatting
        fig.update_layout(
            title=f'Wormhole Geometry at t={time_slice}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Embedding)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_curvature_visualization(self, curvature_type: str = 'ricci_scalar') -> go.Figure:
        """Visualize spacetime curvature in 3D.
        
        Args:
            curvature_type: Type of curvature ('ricci_scalar', 'kretschmann', 'weyl')
        
        Returns:
            Plotly figure with curvature visualization
        """
        
        # Create grid for curvature calculation
        r_sample = self.r_grid[::5]  # Sample for performance
        theta_sample = self.theta_grid[::5]
        phi_sample = self.phi_grid[::5]
        
        points = []
        curvature_values = []
        
        for r in r_sample:
            for theta in theta_sample:
                for phi in phi_sample:
                    coordinates = (0.0, r, theta, phi)  # t=0 slice
                    
                    try:
                        if curvature_type == 'ricci_scalar':
                            # Simplified Ricci scalar estimate
                            if hasattr(self.metric, 'shape_function'):
                                b = self.metric.shape_function(r)
                                curvature = -6 * b / r**3 if r > b else 0
                            else:
                                curvature = 1 / r**2
                        
                        elif curvature_type == 'kretschmann':
                            # Simplified Kretschmann scalar
                            if hasattr(self.metric, 'shape_function'):
                                b = self.metric.shape_function(r)
                                curvature = 48 * b**2 / r**6 if r > b else 0
                            else:
                                curvature = 1 / r**4
                        
                        else:  # weyl
                            curvature = 1 / r**3
                        
                        # Clamp extreme values
                        curvature = np.clip(curvature, 
                                          -self.config.max_curvature_display,
                                          self.config.max_curvature_display)
                        
                        # Convert to Cartesian
                        x = r * np.sin(theta) * np.cos(phi)
                        y = r * np.sin(theta) * np.sin(phi)
                        z = r * np.cos(theta)
                        
                        points.append([x, y, z])
                        curvature_values.append(curvature)
                        
                    except Exception:
                        continue
        
        if not points:
            raise ValueError("No valid curvature points computed")
        
        points = np.array(points)
        curvature_values = np.array(curvature_values)
        
        # Create 3D scatter plot with curvature coloring
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1], 
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=self.config.marker_size,
                color=curvature_values,
                colorscale=self.config.colormap,
                colorbar=dict(title=f'{curvature_type} Curvature'),
                opacity=self.config.opacity
            ),
            text=[f'R={np.linalg.norm(p):.2e}<br>Curvature={c:.2e}' 
                  for p, c in zip(points, curvature_values)],
            hovertemplate='<b>Position:</b><br>X=%{x}<br>Y=%{y}<br>Z=%{z}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Spacetime Curvature: {curvature_type}',
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
    
    def compute_geodesics(self, initial_position: np.ndarray,
                         initial_velocity: np.ndarray,
                         geodesic_type: str = 'timelike') -> Tuple[np.ndarray, np.ndarray]:
        """Compute geodesic trajectories.
        
        Args:
            initial_position: Initial 4-position
            initial_velocity: Initial 4-velocity  
            geodesic_type: Type of geodesic ('timelike', 'null', 'spacelike')
        
        Returns:
            Positions and velocities along geodesic
        """
        
        cache_key = (tuple(initial_position), tuple(initial_velocity), geodesic_type)
        if cache_key in self._geodesic_cache:
            return self._geodesic_cache[cache_key]
        
        # Integration parameters
        tau_max = self.config.geodesic_time
        num_steps = self.config.geodesic_steps
        dtau = tau_max / num_steps
        
        # Initialize arrays
        positions = np.zeros((num_steps + 1, 4))
        velocities = np.zeros((num_steps + 1, 4))
        
        positions[0] = initial_position
        velocities[0] = initial_velocity
        
        # Geodesic integration using RK4
        for i in range(num_steps):
            pos = positions[i]
            vel = velocities[i]
            
            # Compute acceleration from geodesic equation
            try:
                acceleration = self._compute_geodesic_acceleration(pos, vel)
                
                # RK4 integration
                k1_pos = vel
                k1_vel = acceleration
                
                pos_mid1 = pos + 0.5 * dtau * k1_pos
                vel_mid1 = vel + 0.5 * dtau * k1_vel
                acceleration_mid1 = self._compute_geodesic_acceleration(pos_mid1, vel_mid1)
                
                k2_pos = vel_mid1
                k2_vel = acceleration_mid1
                
                pos_mid2 = pos + 0.5 * dtau * k2_pos
                vel_mid2 = vel + 0.5 * dtau * k2_vel
                acceleration_mid2 = self._compute_geodesic_acceleration(pos_mid2, vel_mid2)
                
                k3_pos = vel_mid2
                k3_vel = acceleration_mid2
                
                pos_end = pos + dtau * k3_pos
                vel_end = vel + dtau * k3_vel
                acceleration_end = self._compute_geodesic_acceleration(pos_end, vel_end)
                
                k4_pos = vel_end
                k4_vel = acceleration_end
                
                # Update position and velocity
                positions[i + 1] = pos + (dtau / 6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
                velocities[i + 1] = vel + (dtau / 6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
                
                # Ensure we don't go to unphysical regions
                if positions[i + 1][1] < self.config.r_min:  # r coordinate
                    break
                    
            except Exception as e:
                print(f"Geodesic integration failed at step {i}: {e}")
                positions = positions[:i+1]
                velocities = velocities[:i+1]
                break
        
        # Cache result
        self._geodesic_cache[cache_key] = (positions, velocities)
        
        return positions, velocities
    
    def _compute_geodesic_acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Compute geodesic acceleration d²x^μ/dτ² = -Γ^μ_νρ dx^ν/dτ dx^ρ/dτ."""
        
        coordinates = tuple(position)
        acceleration = np.zeros(4)
        
        try:
            # Get Christoffel symbols
            gamma = self.metric.christoffel_symbols(coordinates)
            
            # Compute acceleration
            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        acceleration[mu] -= gamma[mu, nu, rho] * velocity[nu] * velocity[rho]
                        
        except Exception as e:
            print(f"Error computing acceleration: {e}")
            # Return small acceleration to avoid numerical issues
            acceleration = np.random.normal(0, 1e-10, 4)
        
        return acceleration
    
    def plot_geodesics_3d(self, geodesic_list: List[Tuple[np.ndarray, np.ndarray]] = None,
                         num_geodesics: int = 10) -> go.Figure:
        """Plot geodesic trajectories in 3D.
        
        Args:
            geodesic_list: List of (initial_pos, initial_vel) tuples
            num_geodesics: Number of random geodesics to generate if list not provided
        
        Returns:
            Plotly figure with geodesic trajectories
        """
        
        fig = go.Figure()
        
        if geodesic_list is None:
            # Generate random initial conditions
            geodesic_list = []
            
            for i in range(num_geodesics):
                # Random initial position
                r0 = np.random.uniform(self.config.r_min * 1.1, self.config.r_max * 0.9)
                theta0 = np.random.uniform(0.1, np.pi - 0.1)
                phi0 = np.random.uniform(0, 2*np.pi)
                
                initial_pos = np.array([0.0, r0, theta0, phi0])
                
                # Random initial velocity (normalized)
                initial_vel = np.random.uniform(-1, 1, 4)
                initial_vel = initial_vel / np.linalg.norm(initial_vel) * 0.1
                
                geodesic_list.append((initial_pos, initial_vel))
        
        colors = px.colors.qualitative.Set3
        
        for i, (initial_pos, initial_vel) in enumerate(geodesic_list):
            try:
                positions, velocities = self.compute_geodesics(initial_pos, initial_vel)
                
                # Convert to Cartesian coordinates
                x_coords = []
                y_coords = []
                z_coords = []
                
                for pos in positions:
                    t, r, theta, phi = pos
                    
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                
                # Plot geodesic
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines+markers',
                    line=dict(width=self.config.line_width, color=color),
                    marker=dict(size=2, color=color),
                    name=f'Geodesic {i+1}',
                    hovertemplate='<b>Geodesic %{fullData.name}</b><br>X=%{x}<br>Y=%{y}<br>Z=%{z}<extra></extra>'
                ))
                
                # Mark initial point
                fig.add_trace(go.Scatter3d(
                    x=[x_coords[0]],
                    y=[y_coords[0]],
                    z=[z_coords[0]],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='diamond'),
                    name=f'Start {i+1}',
                    showlegend=False
                ))
                
            except Exception as e:
                print(f"Failed to compute geodesic {i}: {e}")
                continue
        
        # Add wormhole throat (if applicable)
        if hasattr(self.metric, 'b0'):
            throat_radius = self.metric.b0
            
            # Create throat visualization
            phi_throat = np.linspace(0, 2*np.pi, 50)
            theta_throat = np.linspace(0, np.pi, 25)
            
            phi_mesh, theta_mesh = np.meshgrid(phi_throat, theta_throat)
            
            x_throat = throat_radius * np.sin(theta_mesh) * np.cos(phi_mesh)
            y_throat = throat_radius * np.sin(theta_mesh) * np.sin(phi_mesh)
            z_throat = throat_radius * np.cos(theta_mesh)
            
            fig.add_trace(go.Surface(
                x=x_throat, y=y_throat, z=z_throat,
                opacity=0.3,
                colorscale='Reds',
                showscale=False,
                name='Wormhole Throat'
            ))
        
        fig.update_layout(
            title='Geodesic Trajectories in Wormhole Spacetime',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_spacetime_embedding(self, embedding_dimension: int = 3) -> go.Figure:
        """Create spacetime embedding visualization.
        
        Args:
            embedding_dimension: Dimension of embedding space (2 or 3)
        
        Returns:
            Plotly figure showing spacetime embedding
        """
        
        # For Morris-Thorne wormhole, use standard embedding
        if hasattr(self.metric, 'shape_function'):
            return self._morris_thorne_embedding(embedding_dimension)
        else:
            return self._generic_embedding(embedding_dimension)
    
    def _morris_thorne_embedding(self, embedding_dim: int) -> go.Figure:
        """Create Morris-Thorne wormhole embedding diagram."""
        
        # Create coordinate grids
        r_vals = np.linspace(self.metric.b0 * 1.01, self.config.r_max * 0.1, 100)
        phi_vals = np.linspace(0, 2*np.pi, 50)
        
        r_mesh, phi_mesh = np.meshgrid(r_vals, phi_vals)
        
        # Compute embedding surface
        b0 = self.metric.b0
        
        # Upper sheet (z > 0)
        z_upper = np.sqrt(np.maximum(0, (r_mesh - b0)**2))
        x_upper = r_mesh * np.cos(phi_mesh)
        y_upper = r_mesh * np.sin(phi_mesh)
        
        # Lower sheet (z < 0) - mirror image
        z_lower = -z_upper
        x_lower = x_upper
        y_lower = y_upper
        
        fig = go.Figure()
        
        # Plot upper sheet
        fig.add_trace(go.Surface(
            x=x_upper, y=y_upper, z=z_upper,
            colorscale='Blues',
            opacity=0.7,
            name='Upper Universe'
        ))
        
        # Plot lower sheet
        fig.add_trace(go.Surface(
            x=x_lower, y=y_lower, z=z_lower,
            colorscale='Reds', 
            opacity=0.7,
            name='Lower Universe'
        ))
        
        # Plot throat
        throat_phi = np.linspace(0, 2*np.pi, 100)
        throat_x = b0 * np.cos(throat_phi)
        throat_y = b0 * np.sin(throat_phi)
        throat_z = np.zeros_like(throat_x)
        
        fig.add_trace(go.Scatter3d(
            x=throat_x, y=throat_y, z=throat_z,
            mode='lines',
            line=dict(width=8, color='yellow'),
            name='Wormhole Throat'
        ))
        
        fig.update_layout(
            title='Morris-Thorne Wormhole Embedding',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Embedding)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _generic_embedding(self, embedding_dim: int) -> go.Figure:
        """Create generic spacetime embedding."""
        
        # Simplified embedding for general metrics
        r_vals = np.linspace(self.config.r_min, self.config.r_max * 0.1, 100)
        phi_vals = np.linspace(0, 2*np.pi, 50)
        
        r_mesh, phi_mesh = np.meshgrid(r_vals, phi_vals)
        
        # Use metric determinant to define embedding
        z_mesh = np.zeros_like(r_mesh)
        
        for i, r in enumerate(r_vals):
            for j, phi in enumerate(phi_vals):
                coordinates = (0.0, r, np.pi/2, phi)
                
                try:
                    g = self.metric.metric_tensor(coordinates)
                    det_g = np.linalg.det(g)
                    z_mesh[j, i] = np.log(abs(det_g)) if det_g != 0 else 0
                except Exception:
                    z_mesh[j, i] = 0
        
        x_mesh = r_mesh * np.cos(phi_mesh)
        y_mesh = r_mesh * np.sin(phi_mesh)
        
        fig = go.Figure(data=go.Surface(
            x=x_mesh, y=y_mesh, z=z_mesh,
            colorscale=self.config.colormap,
            opacity=self.config.opacity
        ))
        
        fig.update_layout(
            title='Spacetime Embedding',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z (Metric Determinant)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_4d_hypersurface_animation(self, time_steps: int = 50) -> go.Figure:
        """Create animated 4D hypersurface visualization.
        
        Args:
            time_steps: Number of time steps for animation
        
        Returns:
            Animated Plotly figure
        """
        
        # Create frames for different time slices
        frames = []
        
        t_vals = np.linspace(self.config.t_range[0], self.config.t_range[1], time_steps)
        
        for t in t_vals:
            # Create spatial slice at time t
            r_sample = self.r_grid[::3]  # Sample for performance
            phi_sample = np.linspace(0, 2*np.pi, 30)
            
            r_mesh, phi_mesh = np.meshgrid(r_sample, phi_sample)
            
            # Compute some 4D quantity (e.g., spacetime interval)
            interval_mesh = np.zeros_like(r_mesh)
            
            for i, r in enumerate(r_sample):
                for j, phi in enumerate(phi_sample):
                    coordinates = (t, r, np.pi/2, phi)
                    
                    try:
                        g = self.metric.metric_tensor(coordinates)
                        # Compute spacetime interval ds²
                        dx = np.array([1, 0.1, 0, 0])  # Small coordinate displacement
                        interval = np.dot(dx, np.dot(g, dx))
                        interval_mesh[j, i] = interval
                    except Exception:
                        interval_mesh[j, i] = 0
            
            x_mesh = r_mesh * np.cos(phi_mesh)
            y_mesh = r_mesh * np.sin(phi_mesh)
            
            frame = go.Frame(
                data=go.Surface(
                    x=x_mesh, y=y_mesh, z=interval_mesh,
                    colorscale=self.config.colormap,
                    opacity=self.config.opacity
                ),
                name=str(t)
            )
            
            frames.append(frame)
        
        # Create initial plot
        fig = go.Figure(data=frames[0].data)
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title='4D Spacetime Hypersurface Evolution',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Spacetime Interval',
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}])
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )],
            sliders=[dict(
                steps=[dict(args=[[f.name], {"frame": {"duration": 100, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 100}}],
                           label=f.name,
                           method="animate") for f in frames],
                active=0,
                currentvalue={"prefix": "Time: "},
                len=0.9,
                x=0.1,
                xanchor="left",
                y=0,
                yanchor="top"
            )]
        )
        
        return fig
    
    def plot_light_cones(self, event_position: Tuple[float, float, float, float],
                        cone_type: str = 'future') -> go.Figure:
        """Visualize light cones in curved spacetime.
        
        Args:
            event_position: Position of event (t, r, θ, φ)
            cone_type: Type of light cone ('future', 'past', 'both')
        
        Returns:
            Plotly figure with light cone visualization
        """
        
        t0, r0, theta0, phi0 = event_position
        
        fig = go.Figure()
        
        # Create light cone surface
        cone_times = np.linspace(-10, 10, 20) if cone_type == 'both' else \
                    np.linspace(0, 10, 20) if cone_type == 'future' else \
                    np.linspace(-10, 0, 20)
        
        cone_angles = np.linspace(0, 2*np.pi, 30)
        
        for dt in cone_times:
            if dt == 0:
                continue
                
            # Light ray radius in curved spacetime (simplified)
            light_radius = abs(dt) * C  # In flat spacetime approximation
            
            # Modify for curvature effects
            coordinates = (t0 + dt, r0, theta0, phi0)
            try:
                g = self.metric.metric_tensor(coordinates)
                # Use metric to modify light propagation
                light_radius *= np.sqrt(abs(g[1, 1]))  # Radial metric component
            except Exception:
                pass
            
            # Create cone surface at this time
            x_cone = light_radius * np.cos(cone_angles)
            y_cone = light_radius * np.sin(cone_angles)
            z_cone = np.full_like(x_cone, dt)
            
            color = 'red' if dt > 0 else 'blue'
            opacity = 0.3 if abs(dt) < 5 else 0.1
            
            fig.add_trace(go.Scatter3d(
                x=x_cone, y=y_cone, z=z_cone,
                mode='lines',
                line=dict(width=2, color=color),
                opacity=opacity,
                showlegend=False
            ))
        
        # Add event point
        x0 = r0 * np.sin(theta0) * np.cos(phi0)
        y0 = r0 * np.sin(theta0) * np.sin(phi0)
        
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[t0],
            mode='markers',
            marker=dict(size=10, color='yellow', symbol='diamond'),
            name='Event'
        ))
        
        fig.update_layout(
            title=f'{cone_type.title()} Light Cone(s)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Time',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def export_visualization(self, figure: go.Figure, filename: str,
                           format_type: str = 'html') -> None:
        """Export visualization to file.
        
        Args:
            figure: Plotly figure to export
            filename: Output filename
            format_type: Export format ('html', 'png', 'svg', 'pdf')
        """
        
        if format_type == 'html':
            figure.write_html(filename)
        elif format_type == 'png':
            figure.write_image(filename, format='png')
        elif format_type == 'svg':
            figure.write_image(filename, format='svg')
        elif format_type == 'pdf':
            figure.write_image(filename, format='pdf')
        else:
            raise ValueError(f"Unsupported format: {format_type}")


def create_multi_panel_spacetime_view(metric: SpacetimeMetric,
                                    config: SpacetimeVisualizationConfig = None) -> go.Figure:
    """Create comprehensive multi-panel spacetime visualization.
    
    Args:
        metric: Spacetime metric to visualize
        config: Visualization configuration
    
    Returns:
        Multi-panel Plotly figure
    """
    
    plotter = SpacetimePlotter(metric, config)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Wormhole Geometry', 'Curvature', 'Geodesics', 'Embedding'],
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]]
    )
    
    # Geometry plot
    geom_fig = plotter.plot_wormhole_geometry_3d()
    for trace in geom_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Curvature plot
    curv_fig = plotter.plot_curvature_visualization()
    for trace in curv_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Geodesics plot  
    geod_fig = plotter.plot_geodesics_3d(num_geodesics=5)
    for trace in geod_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Embedding plot
    embed_fig = plotter.create_spacetime_embedding()
    for trace in embed_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    fig.update_layout(
        title='Comprehensive Spacetime Analysis',
        height=800,
        width=1200
    )
    
    return fig