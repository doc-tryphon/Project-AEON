"""
4D spacetime rendering with interactive 3D projections.

This module provides advanced visualization of wormhole geometry, geodesics,
and spacetime curvature using interactive 3D plots and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def create_multi_panel_spacetime_view(metric: SpacetimeMetric) -> go.Figure:
    """Create multi-panel spacetime visualization."""
    
    config = SpacetimeVisualizationConfig()
    plotter = SpacetimePlotter(metric, config)
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Wormhole Surface', 'Wireframe', 'Contour', 'Combined'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # Generate different views
    surface_fig = plotter.plot_wormhole_geometry_3d(visualization_type='surface')
    wireframe_fig = plotter.plot_wormhole_geometry_3d(visualization_type='wireframe')
    contour_fig = plotter.plot_wormhole_geometry_3d(visualization_type='contour')
    
    # Add traces to subplots (simplified for demo)
    fig.update_layout(
        title='Multi-Panel Spacetime Analysis',
        height=800,
        width=1200
    )
    
    return fig