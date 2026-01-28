"""
Interactive dashboard for comprehensive wormhole simulation control panel.

This module provides a complete interactive interface for:
- Real-time parameter control and visualization
- Multi-panel simulation monitoring
- Interactive 3D visualizations
- Data analysis and export tools
- Simulation state management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
from plotly import tools
import plotly.io as pio

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import json
import time
from datetime import datetime

# Import simulation components
from src.visualization.spacetime_plotter import SpacetimePlotter, SpacetimeVisualizationConfig
from src.visualization.quantum_state_animator import QuantumStateAnimator, AnimationConfig
from src.visualization.field_visualizer import FieldVisualizer, FieldVisualizationConfig

from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import ExoticMatter
from src.physics.stress_energy_tensor import StressEnergyTensor
from src.ai.stability_predictor import StabilityPredictor
from src.ai.parameter_optimizer import ParameterOptimizer


@dataclass
class DashboardConfig:
    """Configuration for interactive dashboard."""
    
    # Layout parameters
    width: int = 1600
    height: int = 1000
    update_interval: float = 100  # milliseconds
    
    # Panel configurations
    show_spacetime_panel: bool = True
    show_quantum_panel: bool = True
    show_field_panel: bool = True
    show_ai_panel: bool = True
    show_controls_panel: bool = True
    
    # Real-time features
    enable_real_time: bool = True
    auto_refresh: bool = True
    save_session: bool = True
    
    # Export settings
    default_export_format: str = 'html'
    data_export_format: str = 'json'


@dataclass 
class SimulationState:
    """Current state of the wormhole simulation."""
    
    # Wormhole parameters
    throat_radius: float = 1e3
    mass: float = 1e30
    traversal_probability: float = 0.8
    exotic_matter_density: float = -1e15
    
    # Quantum parameters
    num_qubits: int = 4
    entanglement_strength: float = 1.0
    decoherence_rate: float = 0.01
    
    # Visualization parameters
    time_evolution: float = 0.0
    spatial_bounds: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    
    # AI parameters
    stability_threshold: float = 0.5
    optimization_target: str = 'stability'
    
    # Simulation status
    is_running: bool = False
    current_step: int = 0
    total_steps: int = 1000
    last_update: Optional[datetime] = None
    
    # Computed results
    stability_score: float = 0.0
    entanglement_measure: float = 0.0
    field_strength: float = 0.0
    
    # Performance metrics
    computation_time: float = 0.0
    memory_usage: float = 0.0


class InteractiveDashboard:
    """Comprehensive interactive dashboard for wormhole simulation control."""
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize interactive dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.state = SimulationState()
        
        # Initialize simulation components
        self.spacetime_plotter = None
        self.quantum_animator = None  
        self.field_visualizer = None
        
        # Initialize AI components
        self.stability_predictor = None
        self.parameter_optimizer = None
        
        # Dashboard components
        self.main_figure = None
        self.control_widgets = {}
        self.data_cache = {}
        
        # Initialize components
        self._initialize_simulation_components()
        
    def _initialize_simulation_components(self):
        """Initialize all simulation components."""
        
        # Spacetime visualization
        spacetime_config = SpacetimeVisualizationConfig(
            r_min=self.state.throat_radius * 1.1,
            r_max=self.state.throat_radius * 10
        )
        
        # Create wormhole metric
        wormhole_metric = MorrisThorneeWormhole(
            throat_radius=self.state.throat_radius
        )
        
        self.spacetime_plotter = SpacetimePlotter(wormhole_metric, spacetime_config)
        
        # Quantum animation  
        animation_config = AnimationConfig(
            total_time=10.0,
            time_steps=100
        )
        
        self.quantum_animator = QuantumStateAnimator(animation_config)
        self.quantum_animator.setup_wormhole_system(
            num_qubits=self.state.num_qubits,
            traversal_probability=self.state.traversal_probability
        )
        
        # Field visualization
        field_config = FieldVisualizationConfig()
        self.field_visualizer = FieldVisualizer(field_config)
        
        # AI components
        self.stability_predictor = StabilityPredictor()
        self.parameter_optimizer = ParameterOptimizer()
        
    def create_main_dashboard(self) -> go.Figure:
        """Create main interactive dashboard with all panels.
        
        Returns:
            Interactive dashboard figure
        """
        
        # Create subplot layout
        panel_count = sum([
            self.config.show_spacetime_panel,
            self.config.show_quantum_panel, 
            self.config.show_field_panel,
            self.config.show_ai_panel
        ])
        
        if panel_count == 4:
            rows, cols = 2, 2
        elif panel_count == 3:
            rows, cols = 2, 2  # Leave one empty
        elif panel_count == 2:
            rows, cols = 1, 2
        else:
            rows, cols = 1, 1
        
        # Define subplot specs
        specs = []
        titles = []
        
        if self.config.show_spacetime_panel:
            titles.append('Spacetime Geometry')
        if self.config.show_quantum_panel:
            titles.append('Quantum State Evolution') 
        if self.config.show_field_panel:
            titles.append('Field Visualization')
        if self.config.show_ai_panel:
            titles.append('AI Analysis')
        
        # Pad titles to match grid
        while len(titles) < rows * cols:
            titles.append('')
        
        # Create specs array
        for i in range(rows):
            row_specs = []
            for j in range(cols):
                idx = i * cols + j
                if idx < panel_count:
                    if idx in [0, 2]:  # Spacetime and Field panels
                        row_specs.append({'type': 'scene'})
                    else:  # Quantum and AI panels
                        row_specs.append({'type': 'scatter'})
                else:
                    row_specs.append({'type': 'scatter'})
            specs.append(row_specs)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles,
            specs=specs,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Add panels
        panel_idx = 0
        
        if self.config.show_spacetime_panel:
            row, col = (panel_idx // cols) + 1, (panel_idx % cols) + 1
            self._add_spacetime_panel(fig, row, col)
            panel_idx += 1
            
        if self.config.show_quantum_panel:
            row, col = (panel_idx // cols) + 1, (panel_idx % cols) + 1
            self._add_quantum_panel(fig, row, col)
            panel_idx += 1
            
        if self.config.show_field_panel:
            row, col = (panel_idx // cols) + 1, (panel_idx % cols) + 1
            self._add_field_panel(fig, row, col)
            panel_idx += 1
            
        if self.config.show_ai_panel:
            row, col = (panel_idx // cols) + 1, (panel_idx % cols) + 1
            self._add_ai_panel(fig, row, col)
            panel_idx += 1
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text='Quantum Wormhole Simulation Dashboard',
                font=dict(size=24, color='darkblue'),
                x=0.5
            ),
            showlegend=True,
            width=self.config.width,
            height=self.config.height,
            updatemenus=self._create_control_menus(),
            annotations=self._create_status_annotations()
        )
        
        self.main_figure = fig
        return fig
    
    def _add_spacetime_panel(self, fig: go.Figure, row: int, col: int):
        """Add spacetime visualization panel."""
        
        # Get spacetime visualization
        spacetime_fig = self.spacetime_plotter.plot_wormhole_geometry_3d(
            time_slice=self.state.time_evolution
        )
        
        # Add traces to subplot
        for trace in spacetime_fig.data:
            fig.add_trace(trace, row=row, col=col)
        
        # Update scene
        scene_key = f'scene{row}{col}' if row > 1 or col > 1 else 'scene'
        fig.update_layout(**{
            scene_key: dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            )
        })
    
    def _add_quantum_panel(self, fig: go.Figure, row: int, col: int):
        """Add quantum state panel."""
        
        # Create quantum state evolution data
        time_points = np.linspace(0, 10, 100)
        
        # Simulate quantum state evolution
        entanglement_evolution = []
        coherence_evolution = []
        
        for t in time_points:
            # Simulate entanglement measure
            entanglement = self.state.entanglement_strength * np.sin(t) * np.exp(-self.state.decoherence_rate * t)
            entanglement_evolution.append(max(0, entanglement))
            
            # Simulate coherence
            coherence = np.exp(-self.state.decoherence_rate * t)
            coherence_evolution.append(coherence)
        
        # Add entanglement trace
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=entanglement_evolution,
                mode='lines',
                name='Entanglement',
                line=dict(color='red', width=3)
            ),
            row=row, col=col
        )
        
        # Add coherence trace
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=coherence_evolution,
                mode='lines', 
                name='Coherence',
                line=dict(color='blue', width=3)
            ),
            row=row, col=col
        )
        
        # Update axes
        x_axis_key = f'xaxis{row}{col}' if row > 1 or col > 1 else 'xaxis'
        y_axis_key = f'yaxis{row}{col}' if row > 1 or col > 1 else 'yaxis'
        
        fig.update_layout(**{
            x_axis_key: dict(title='Time'),
            y_axis_key: dict(title='Quantum Measure')
        })
    
    def _add_field_panel(self, fig: go.Figure, row: int, col: int):
        """Add field visualization panel."""
        
        # Create sample field visualization
        # For demo, create a simple gravitational field visualization
        theta = np.linspace(0, 2*np.pi, 50)
        r_vals = np.linspace(self.state.throat_radius, self.state.throat_radius * 3, 20)
        
        field_strength = []
        for r in r_vals:
            # Simplified field strength
            strength = 1 / (r - self.state.throat_radius + 1)**2
            field_strength.append(strength)
        
        # Create 3D field visualization
        theta_mesh, r_mesh = np.meshgrid(theta, r_vals)
        x_mesh = r_mesh * np.cos(theta_mesh)
        y_mesh = r_mesh * np.sin(theta_mesh)
        z_mesh = np.array([field_strength]).T @ np.ones((1, len(theta)))
        
        fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh, 
                z=z_mesh,
                colorscale='Viridis',
                opacity=0.8,
                name='Field Strength'
            ),
            row=row, col=col
        )
        
        # Update scene
        scene_key = f'scene{row}{col}' if row > 1 or col > 1 else 'scene'
        if row > 1 or col > 1:
            fig.update_layout(**{
                scene_key: dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Field Strength',
                    aspectmode='cube'
                )
            })
    
    def _add_ai_panel(self, fig: go.Figure, row: int, col: int):
        """Add AI analysis panel."""
        
        # Create AI analysis visualization
        time_points = np.linspace(0, 100, 100)
        
        # Simulate stability evolution
        stability_evolution = []
        for t in time_points:
            # Simulate stability with some noise
            base_stability = self.state.stability_threshold
            noise = 0.1 * np.sin(t/10) + 0.05 * np.random.randn()
            stability = max(0, min(1, base_stability + noise))
            stability_evolution.append(stability)
        
        # Add stability trace
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=stability_evolution,
                mode='lines+markers',
                name='Stability Score',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[self.state.stability_threshold, self.state.stability_threshold],
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash', width=2)
            ),
            row=row, col=col
        )
        
        # Update axes
        x_axis_key = f'xaxis{row}{col}' if row > 1 or col > 1 else 'xaxis'
        y_axis_key = f'yaxis{row}{col}' if row > 1 or col > 1 else 'yaxis'
        
        fig.update_layout(**{
            x_axis_key: dict(title='Simulation Step'),
            y_axis_key: dict(title='Stability Score', range=[0, 1])
        })
    
    def _create_control_menus(self) -> List[Dict]:
        """Create interactive control menus."""
        
        control_menus = [
            # Simulation control menu
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True] * 20}],  # Show all traces
                        label="Start Simulation",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False] * 20}],  # Hide all traces
                        label="Pause Simulation", 
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True] * 20}],
                        label="Reset Simulation",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.05,
                yanchor="top",
            ),
            
            # View control menu
            dict(
                type="dropdown",
                direction="down",
                buttons=[
                    dict(
                        args=[{"title": "3D Spacetime View"}],
                        label="3D Spacetime",
                        method="relayout"
                    ),
                    dict(
                        args=[{"title": "Quantum Evolution View"}], 
                        label="Quantum Evolution",
                        method="relayout"
                    ),
                    dict(
                        args=[{"title": "Field Analysis View"}],
                        label="Field Analysis",
                        method="relayout"
                    ),
                    dict(
                        args=[{"title": "AI Dashboard View"}],
                        label="AI Dashboard", 
                        method="relayout"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.3,
                xanchor="left",
                y=1.05,
                yanchor="top",
            )
        ]
        
        return control_menus
    
    def _create_status_annotations(self) -> List[Dict]:
        """Create status display annotations."""
        
        status_annotations = [
            # Simulation status
            dict(
                text=f"Status: {'Running' if self.state.is_running else 'Stopped'}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                bgcolor="lightgreen" if self.state.is_running else "lightcoral",
                bordercolor="black",
                borderwidth=1
            ),
            
            # Current parameters
            dict(
                text=f"Throat Radius: {self.state.throat_radius:.1e}<br>"
                     f"Mass: {self.state.mass:.1e}<br>" 
                     f"Qubits: {self.state.num_qubits}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.98, y=0.85,
                xanchor="right", yanchor="top",
                bgcolor="lightblue",
                bordercolor="black",
                borderwidth=1
            ),
            
            # Performance metrics
            dict(
                text=f"Stability: {self.state.stability_score:.3f}<br>"
                     f"Entanglement: {self.state.entanglement_measure:.3f}<br>"
                     f"Compute Time: {self.state.computation_time:.2f}ms",
                showarrow=False,
                xref="paper", yref="paper", 
                x=0.98, y=0.65,
                xanchor="right", yanchor="top",
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=1
            )
        ]
        
        return status_annotations
    
    def update_simulation_state(self, **kwargs):
        """Update simulation state with new parameters.
        
        Args:
            **kwargs: Parameter updates
        """
        
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        self.state.last_update = datetime.now()
        
        # Trigger component updates
        if any(key in ['throat_radius', 'mass'] for key in kwargs):
            self._update_spacetime_components()
        
        if any(key in ['num_qubits', 'entanglement_strength'] for key in kwargs):
            self._update_quantum_components()
    
    def _update_spacetime_components(self):
        """Update spacetime visualization components."""
        
        # Update wormhole metric
        wormhole_metric = MorrisThorneeWormhole(
            throat_radius=self.state.throat_radius
        )
        
        # Update spacetime plotter
        spacetime_config = SpacetimeVisualizationConfig(
            r_min=self.state.throat_radius * 1.1,
            r_max=self.state.throat_radius * 10
        )
        
        self.spacetime_plotter = SpacetimePlotter(wormhole_metric, spacetime_config)
    
    def _update_quantum_components(self):
        """Update quantum simulation components."""
        
        # Update quantum animator
        self.quantum_animator.setup_wormhole_system(
            num_qubits=self.state.num_qubits,
            traversal_probability=self.state.traversal_probability
        )
    
    def run_real_time_simulation(self, duration: float = 60.0):
        """Run real-time simulation with dashboard updates.
        
        Args:
            duration: Simulation duration in seconds
        """
        
        if not self.config.enable_real_time:
            print("Real-time simulation not enabled")
            return
        
        self.state.is_running = True
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration and self.state.is_running:
            step_start = time.time()
            
            # Simulate one time step
            self._simulate_step(step)
            
            # Update dashboard if needed
            if step % 10 == 0:  # Update every 10 steps
                self._update_dashboard_data()
            
            # Control update rate
            step_time = (time.time() - step_start) * 1000
            if step_time < self.config.update_interval:
                time.sleep((self.config.update_interval - step_time) / 1000)
            
            step += 1
        
        self.state.is_running = False
        self.state.current_step = step
    
    def _simulate_step(self, step: int):
        """Simulate one time step.
        
        Args:
            step: Current simulation step
        """
        
        step_start = time.time()
        
        # Update time evolution
        self.state.time_evolution = step * 0.1
        
        # Simulate stability evolution
        base_stability = 0.8
        noise = 0.1 * np.sin(step * 0.1) + 0.02 * np.random.randn()
        self.state.stability_score = max(0, min(1, base_stability + noise))
        
        # Simulate entanglement evolution  
        self.state.entanglement_measure = abs(np.sin(step * 0.05)) * np.exp(-step * 0.001)
        
        # Simulate field strength
        self.state.field_strength = 1 / (1 + step * 0.01)
        
        # Update performance metrics
        self.state.computation_time = (time.time() - step_start) * 1000
        
        # Update AI predictions (simplified)
        if self.stability_predictor and step % 50 == 0:
            # Create feature vector
            features = np.array([
                self.state.throat_radius,
                self.state.mass,
                self.state.entanglement_measure,
                self.state.field_strength
            ]).reshape(1, -1)
            
            try:
                # Predict stability (would need trained model)
                predicted_stability = np.random.random()  # Placeholder
                self.state.stability_score = predicted_stability
            except:
                pass  # Model not trained
    
    def _update_dashboard_data(self):
        """Update cached dashboard data."""
        
        # Cache current state for dashboard
        self.data_cache['timestamp'] = datetime.now()
        self.data_cache['state'] = self.state.__dict__.copy()
        
        # Store time series data
        if 'time_series' not in self.data_cache:
            self.data_cache['time_series'] = {
                'time': [],
                'stability': [],
                'entanglement': [],
                'field_strength': []
            }
        
        self.data_cache['time_series']['time'].append(self.state.time_evolution)
        self.data_cache['time_series']['stability'].append(self.state.stability_score)
        self.data_cache['time_series']['entanglement'].append(self.state.entanglement_measure)
        self.data_cache['time_series']['field_strength'].append(self.state.field_strength)
    
    def create_parameter_control_panel(self) -> go.Figure:
        """Create interactive parameter control panel.
        
        Returns:
            Parameter control panel figure
        """
        
        # Create parameter sliders using plotly widgets
        fig = go.Figure()
        
        # This would create interactive sliders for parameters
        # For now, create a visualization of parameter space
        
        # Create parameter sensitivity analysis
        param_names = ['Throat Radius', 'Mass', 'Qubits', 'Entanglement']
        param_values = [
            self.state.throat_radius / 1e3,  # Scaled for display
            self.state.mass / 1e30,
            self.state.num_qubits,
            self.state.entanglement_strength
        ]
        
        fig.add_trace(go.Bar(
            x=param_names,
            y=param_values,
            marker_color=['red', 'blue', 'green', 'purple'],
            name='Current Values'
        ))
        
        fig.update_layout(
            title='Simulation Parameters',
            xaxis_title='Parameters',
            yaxis_title='Normalized Values',
            width=800,
            height=400
        )
        
        return fig
    
    def export_dashboard_data(self, filename: str, format_type: str = None) -> None:
        """Export dashboard data and state.
        
        Args:
            filename: Output filename
            format_type: Export format ('json', 'csv', 'html')
        """
        
        if format_type is None:
            format_type = self.config.data_export_format
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'state': self.state.__dict__,
            'cache': self.data_cache
        }
        
        if format_type == 'json':
            # Convert datetime objects to strings
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError("Type not serializable")
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, default=serialize_datetime, indent=2)
                
        elif format_type == 'csv':
            # Export time series data as CSV
            if 'time_series' in self.data_cache:
                df = pd.DataFrame(self.data_cache['time_series'])
                df.to_csv(filename, index=False)
            
        elif format_type == 'html':
            # Export dashboard figure as HTML
            if self.main_figure:
                self.main_figure.write_html(filename)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def load_dashboard_session(self, filename: str) -> None:
        """Load saved dashboard session.
        
        Args:
            filename: Session file to load
        """
        
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            # Restore state
            state_data = session_data.get('state', {})
            for key, value in state_data.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            
            # Restore cache
            self.data_cache = session_data.get('cache', {})
            
            # Reinitialize components
            self._initialize_simulation_components()
            
            print(f"Dashboard session loaded from {filename}")
            
        except Exception as e:
            print(f"Error loading session: {e}")
    
    def create_comprehensive_dashboard_app(self) -> go.Figure:
        """Create complete dashboard application.
        
        Returns:
            Complete interactive dashboard
        """
        
        # Create main dashboard
        main_fig = self.create_main_dashboard()
        
        # Add additional interactive features
        main_fig.update_layout(
            # Add range sliders for time series data
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            ),
            
            # Add hover data
            hovermode='closest',
            
            # Add animation frames if needed
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"visible": [True] * 50}],
                            label="Show All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False] * 50}],
                            label="Hide All", 
                            method="update"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.1,
                    yanchor="top",
                )
            ]
        )
        
        return main_fig


def launch_dashboard(config: DashboardConfig = None, port: int = 8050) -> None:
    """Launch the interactive dashboard web application.
    
    Args:
        config: Dashboard configuration
        port: Port number for web server
    """
    
    dashboard = InteractiveDashboard(config)
    
    # Create main dashboard
    fig = dashboard.create_comprehensive_dashboard_app()
    
    # Create parameter control panel
    control_fig = dashboard.create_parameter_control_panel()
    
    print(f"Launching Quantum Wormhole Simulation Dashboard...")
    print(f"Access the dashboard at http://localhost:{port}")
    
    # Save dashboard as HTML for viewing
    fig.write_html("quantum_wormhole_dashboard.html", auto_open=True)
    control_fig.write_html("parameter_control_panel.html")
    
    print("Dashboard saved as quantum_wormhole_dashboard.html")
    print("Parameter panel saved as parameter_control_panel.html")


def create_dashboard_suite(config: DashboardConfig = None) -> Dict[str, go.Figure]:
    """Create complete suite of dashboard visualizations.
    
    Args:
        config: Dashboard configuration
    
    Returns:
        Dictionary of dashboard figures
    """
    
    dashboard = InteractiveDashboard(config)
    
    dashboard_suite = {
        'main_dashboard': dashboard.create_comprehensive_dashboard_app(),
        'parameter_control': dashboard.create_parameter_control_panel(),
        'spacetime_view': dashboard.spacetime_plotter.plot_wormhole_geometry_3d(),
        'quantum_animation': dashboard.quantum_animator.animate_bloch_sphere_evolution(),
        'field_visualization': dashboard.field_visualizer.visualize_gravitational_field(
            dashboard.spacetime_plotter.metric
        )
    }
    
    return dashboard_suite