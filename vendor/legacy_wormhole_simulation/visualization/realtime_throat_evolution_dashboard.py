"""
Real-time Throat Evolution Dashboard.

This module provides live visualization of dynamic wormhole throat evolution,
integrating with the existing visualization infrastructure to display
time-dependent geometries, metrics, and evolution trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly import tools
import plotly.offline as pyo

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import threading
import queue

# Import existing visualization infrastructure
from src.visualization.spacetime_plotter import SpacetimePlotter, SpacetimeVisualizationConfig
from src.visualization.interactive_dashboard import DashboardConfig

# Import physics and evolution systems
from src.physics.dynamic_throat_evolution import (
    DynamicThroatEvolution, 
    EvolutionParameters,
    create_evolution_scenario
)
from src.physics.rotating_wormhole_metrics import KerrLikeWormhole, RotationParameters


@dataclass
class RealTimeVisualizationConfig:
    """Configuration for real-time throat evolution visualization."""
    
    # Update parameters
    update_interval: float = 100.0  # milliseconds
    evolution_step_size: float = 10.0  # evolution time per update (seconds)
    max_history_points: int = 1000  # maximum data points to store
    
    # Display parameters
    show_3d_embedding: bool = True
    show_metric_evolution: bool = True
    show_energy_analysis: bool = True
    show_stability_indicators: bool = True
    
    # Layout parameters
    plot_width: int = 1400
    plot_height: int = 800
    subplot_rows: int = 2
    subplot_cols: int = 3
    
    # Animation parameters
    frame_rate: int = 30  # FPS for animations
    trail_length: int = 50  # number of points in evolution trail
    
    # Interaction parameters
    enable_parameter_controls: bool = True
    enable_scenario_switching: bool = True
    enable_real_time_export: bool = False


class RealTimeThroatEvolutionDashboard:
    """Real-time dashboard for throat evolution visualization."""
    
    def __init__(self, config: RealTimeVisualizationConfig):
        """Initialize real-time dashboard.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.evolution_system = None
        self.current_scenario = "standard"
        
        # Data storage for real-time updates
        self.time_history = []
        self.throat_radius_history = []
        self.mass_history = []
        self.angular_momentum_history = []
        self.stability_score_history = []
        self.energy_history = []
        
        # Animation control
        self.is_running = False
        self.animation = None
        self.update_thread = None
        self.data_queue = queue.Queue()
        
        # Plotly figure components
        self.fig = None
        self.plots_initialized = False
        
        # Initialize with default scenario
        self._initialize_evolution_system()
        
    def _initialize_evolution_system(self, scenario_type: str = "standard"):
        """Initialize evolution system with specified scenario."""
        self.evolution_system = create_evolution_scenario(
            scenario_type=scenario_type,
            initial_radius=1000.0,
            mass=1e30,
            angular_momentum=1e43
        )
        self.current_scenario = scenario_type
        
        # Reset data history
        self.time_history = [0.0]
        self.throat_radius_history = [self.evolution_system.current_throat_radius]
        self.mass_history = [self.evolution_system.current_mass]
        self.angular_momentum_history = [self.evolution_system.current_rotation_params.angular_momentum]
        self.stability_score_history = [1.0]  # Start with perfect stability
        self.energy_history = [0.0]
    
    def create_dashboard(self) -> go.Figure:
        """Create the complete real-time dashboard layout."""
        
        # Create subplot structure
        self.fig = make_subplots(
            rows=self.config.subplot_rows,
            cols=self.config.subplot_cols,
            subplot_titles=[
                "3D Wormhole Embedding",
                "Throat Radius Evolution", 
                "Mass & Angular Momentum",
                "Metric Components",
                "Stability Analysis",
                "Energy Conservation"
            ],
            specs=[
                [{"type": "scene"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Initialize all subplot components
        self._initialize_3d_embedding()
        self._initialize_throat_evolution_plot()
        self._initialize_mass_momentum_plot()
        self._initialize_metric_components_plot()
        self._initialize_stability_plot()
        self._initialize_energy_plot()
        
        # Configure layout
        self.fig.update_layout(
            title="Real-Time Wormhole Throat Evolution Dashboard",
            width=self.config.plot_width,
            height=self.config.plot_height,
            showlegend=True,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 1000/self.config.frame_rate}}]
                        },
                        {
                            "label": "Pause", 
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ]
        )
        
        self.plots_initialized = True
        return self.fig
    
    def _initialize_3d_embedding(self):
        """Initialize 3D wormhole embedding visualization."""
        current_radius = self.evolution_system.current_throat_radius
        
        # Create embedding surface
        u = np.linspace(-2*current_radius, 2*current_radius, 50)
        v = np.linspace(0, 2*np.pi, 50)
        U, V = np.meshgrid(u, v)
        
        # Morris-Thorne embedding (simplified)
        r = np.sqrt(U**2 + current_radius**2)
        X = r * np.cos(V)
        Y = r * np.sin(V) 
        Z = np.sqrt(np.maximum(0, r**2 - current_radius**2))
        
        # Create upper and lower sheets
        self.fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, name="Upper Sheet",
                      colorscale="Viridis", opacity=0.8),
            row=1, col=1
        )
        
        self.fig.add_trace(
            go.Surface(x=X, y=Y, z=-Z, name="Lower Sheet", 
                      colorscale="Viridis", opacity=0.8),
            row=1, col=1
        )
        
        # Add throat circle
        theta = np.linspace(0, 2*np.pi, 100)
        throat_x = current_radius * np.cos(theta)
        throat_y = current_radius * np.sin(theta)
        throat_z = np.zeros_like(theta)
        
        self.fig.add_trace(
            go.Scatter3d(x=throat_x, y=throat_y, z=throat_z,
                        mode="lines", name="Throat",
                        line=dict(color="red", width=5)),
            row=1, col=1
        )
    
    def _initialize_throat_evolution_plot(self):
        """Initialize throat radius evolution plot."""
        self.fig.add_trace(
            go.Scatter(x=self.time_history, y=self.throat_radius_history,
                      mode="lines+markers", name="Throat Radius",
                      line=dict(color="blue", width=2)),
            row=1, col=2
        )
        
        self.fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        self.fig.update_yaxes(title_text="Throat Radius (m)", row=1, col=2)
    
    def _initialize_mass_momentum_plot(self):
        """Initialize mass and angular momentum evolution plot."""
        self.fig.add_trace(
            go.Scatter(x=self.time_history, y=self.mass_history,
                      mode="lines", name="Mass", 
                      line=dict(color="green", width=2),
                      yaxis="y"),
            row=1, col=3
        )
        
        # Scale angular momentum for display
        scaled_momentum = [j/1e43 for j in self.angular_momentum_history]
        self.fig.add_trace(
            go.Scatter(x=self.time_history, y=scaled_momentum,
                      mode="lines", name="Angular Momentum (×10⁴³)",
                      line=dict(color="orange", width=2),
                      yaxis="y2"),
            row=1, col=3
        )
        
        self.fig.update_xaxes(title_text="Time (s)", row=1, col=3)
        self.fig.update_yaxes(title_text="Mass (kg)", row=1, col=3)
    
    def _initialize_metric_components_plot(self):
        """Initialize metric components visualization."""
        # Sample metric at throat
        current_wormhole = self.evolution_system.get_current_wormhole()
        r_throat = current_wormhole.throat_radius
        
        g_tt = current_wormhole.metric_tt(r_throat, np.pi/2)
        g_rr = current_wormhole.metric_rr(r_throat, np.pi/2)
        g_tphi = current_wormhole.metric_t_phi(r_throat, np.pi/2)
        
        metrics = ["g_tt", "g_rr", "g_tφ"]
        values = [g_tt, g_rr, g_tphi]
        
        self.fig.add_trace(
            go.Bar(x=metrics, y=values, name="Metric Components",
                  marker_color=["red", "blue", "green"]),
            row=2, col=1
        )
        
        self.fig.update_xaxes(title_text="Metric Component", row=2, col=1)
        self.fig.update_yaxes(title_text="Value", row=2, col=1)
    
    def _initialize_stability_plot(self):
        """Initialize stability analysis plot."""
        self.fig.add_trace(
            go.Scatter(x=self.time_history, y=self.stability_score_history,
                      mode="lines+markers", name="Stability Score",
                      line=dict(color="purple", width=2)),
            row=2, col=2
        )
        
        # Add stability threshold line manually
        self.fig.add_trace(
            go.Scatter(x=[0, 1], y=[0.5, 0.5],
                      mode="lines", name="Stability Threshold",
                      line=dict(color="red", dash="dash"),
                      showlegend=False),
            row=2, col=2
        )
        
        self.fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        self.fig.update_yaxes(title_text="Stability Score", range=[0, 1], row=2, col=2)
    
    def _initialize_energy_plot(self):
        """Initialize energy conservation plot."""
        self.fig.add_trace(
            go.Scatter(x=self.time_history, y=self.energy_history,
                      mode="lines", name="Total Energy",
                      line=dict(color="darkred", width=2)),
            row=2, col=3
        )
        
        self.fig.update_xaxes(title_text="Time (s)", row=2, col=3)
        self.fig.update_yaxes(title_text="Energy Change", row=2, col=3)
    
    def start_real_time_evolution(self):
        """Start real-time evolution updates."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background evolution thread
        self.update_thread = threading.Thread(target=self._evolution_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print(f"Started real-time evolution for scenario: {self.current_scenario}")
    
    def stop_real_time_evolution(self):
        """Stop real-time evolution updates."""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        print("Stopped real-time evolution")
    
    def _evolution_update_loop(self):
        """Background thread for evolution updates."""
        while self.is_running:
            try:
                # Evolve system for one step
                evolution_result = self.evolution_system.evolve_throat(
                    time_span=self.config.evolution_step_size,
                    num_steps=50
                )
                
                if evolution_result['evolution_success']:
                    # Extract latest data
                    times = evolution_result['times']
                    radii = evolution_result['throat_radii']
                    masses = evolution_result['masses']
                    angular_momenta = evolution_result['angular_momenta']
                    
                    # Calculate stability score
                    velocity = evolution_result['throat_velocity'][-1]
                    stability_score = 1.0 / (1.0 + abs(velocity))
                    
                    # Energy analysis
                    energy_analysis = evolution_result['energy_analysis']
                    energy_change = energy_analysis['energy_change']
                    
                    # Add to history (keep only recent points)
                    current_time = times[-1]
                    self.time_history.append(current_time)
                    self.throat_radius_history.append(radii[-1])
                    self.mass_history.append(masses[-1])
                    self.angular_momentum_history.append(angular_momenta[-1])
                    self.stability_score_history.append(stability_score)
                    self.energy_history.append(energy_change)
                    
                    # Limit history size
                    if len(self.time_history) > self.config.max_history_points:
                        self.time_history = self.time_history[-self.config.max_history_points:]
                        self.throat_radius_history = self.throat_radius_history[-self.config.max_history_points:]
                        self.mass_history = self.mass_history[-self.config.max_history_points:]
                        self.angular_momentum_history = self.angular_momentum_history[-self.config.max_history_points:]
                        self.stability_score_history = self.stability_score_history[-self.config.max_history_points:]
                        self.energy_history = self.energy_history[-self.config.max_history_points:]
                    
                    # Queue update for main thread
                    self.data_queue.put({
                        'time': current_time,
                        'throat_radius': radii[-1],
                        'mass': masses[-1],
                        'angular_momentum': angular_momenta[-1],
                        'stability_score': stability_score,
                        'energy_change': energy_change
                    })
                
                # Sleep for update interval
                time.sleep(self.config.update_interval / 1000.0)
                
            except Exception as e:
                print(f"Evolution update error: {e}")
                time.sleep(0.1)
    
    def update_plots(self):
        """Update all plots with latest data."""
        if not self.plots_initialized:
            return
        
        # Process queued updates
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                
                # Update throat evolution plot
                with self.fig.batch_update():
                    # Update throat radius plot
                    self.fig.data[3].x = self.time_history
                    self.fig.data[3].y = self.throat_radius_history
                    
                    # Update mass plot
                    self.fig.data[4].x = self.time_history
                    self.fig.data[4].y = self.mass_history
                    
                    # Update angular momentum plot
                    scaled_momentum = [j/1e43 for j in self.angular_momentum_history]
                    self.fig.data[5].x = self.time_history
                    self.fig.data[5].y = scaled_momentum
                    
                    # Update stability plot
                    self.fig.data[7].x = self.time_history
                    self.fig.data[7].y = self.stability_score_history
                    
                    # Update energy plot
                    self.fig.data[9].x = self.time_history
                    self.fig.data[9].y = self.energy_history
                    
                    # Update 3D embedding (simplified - just update throat size)
                    current_radius = self.throat_radius_history[-1]
                    self._update_3d_embedding(current_radius)
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Plot update error: {e}")
    
    def _update_3d_embedding(self, new_radius: float):
        """Update 3D embedding with new throat radius."""
        # Update throat circle
        theta = np.linspace(0, 2*np.pi, 100)
        throat_x = new_radius * np.cos(theta)
        throat_y = new_radius * np.sin(theta)
        throat_z = np.zeros_like(theta)
        
        # Update the throat trace (assuming it's trace index 2)
        if len(self.fig.data) > 2:
            self.fig.data[2].x = throat_x
            self.fig.data[2].y = throat_y
            self.fig.data[2].z = throat_z
    
    def switch_scenario(self, scenario_type: str):
        """Switch to a different evolution scenario."""
        if scenario_type == self.current_scenario:
            return
        
        # Stop current evolution
        was_running = self.is_running
        if was_running:
            self.stop_real_time_evolution()
        
        # Initialize new scenario
        self._initialize_evolution_system(scenario_type)
        
        # Restart if it was running
        if was_running:
            self.start_real_time_evolution()
        
        print(f"Switched to scenario: {scenario_type}")
    
    def export_current_data(self, filename: Optional[str] = None) -> str:
        """Export current evolution data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"throat_evolution_data_{timestamp}.json"
        
        export_data = {
            'scenario_type': self.current_scenario,
            'export_time': datetime.now().isoformat(),
            'time_history': self.time_history,
            'throat_radius_history': self.throat_radius_history,
            'mass_history': self.mass_history,
            'angular_momentum_history': self.angular_momentum_history,
            'stability_score_history': self.stability_score_history,
            'energy_history': self.energy_history,
            'config': {
                'update_interval': self.config.update_interval,
                'evolution_step_size': self.config.evolution_step_size,
                'max_history_points': self.config.max_history_points
            }
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def create_static_comparison_plot(self, scenarios: List[str],
                                    simulation_time: float = 1000.0) -> go.Figure:
        """Create static comparison plot of multiple scenarios."""
        from src.physics.dynamic_throat_evolution import compare_evolution_scenarios
        
        # Run scenario comparison
        comparison_results = compare_evolution_scenarios(
            scenarios=scenarios,
            simulation_time=simulation_time,
            initial_radius=1000.0
        )
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Throat Radius Evolution",
                "Stability Scores", 
                "Final State Distribution",
                "Energy Conservation"
            ]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, scenario in enumerate(scenarios):
            if scenario in comparison_results['individual_results']:
                result = comparison_results['individual_results'][scenario]
                
                if result.get('evolution_success', False):
                    times = result['times']
                    radii = result['throat_radii']
                    stability = result['statistics']['stability_score']
                    
                    # Throat radius evolution
                    fig.add_trace(
                        go.Scatter(x=times, y=radii, mode="lines",
                                  name=f"{scenario} - Radius",
                                  line=dict(color=colors[i % len(colors)])),
                        row=1, col=1
                    )
                    
                    # Stability score
                    fig.add_trace(
                        go.Bar(x=[scenario], y=[stability],
                              name=f"{scenario} - Stability",
                              marker_color=colors[i % len(colors)]),
                        row=1, col=2
                    )
        
        fig.update_layout(
            title="Evolution Scenario Comparison",
            height=800,
            showlegend=True
        )
        
        return fig


def create_realtime_dashboard(config: Optional[RealTimeVisualizationConfig] = None) -> RealTimeThroatEvolutionDashboard:
    """Factory function to create real-time dashboard.
    
    Args:
        config: Optional visualization configuration
        
    Returns:
        Configured real-time dashboard
    """
    if config is None:
        config = RealTimeVisualizationConfig()
    
    dashboard = RealTimeThroatEvolutionDashboard(config)
    return dashboard


def launch_interactive_session(scenarios: List[str] = None) -> None:
    """Launch interactive real-time visualization session.
    
    Args:
        scenarios: List of scenarios to make available for switching
    """
    if scenarios is None:
        scenarios = ["standard", "collapse", "expansion"]
    
    # Create dashboard
    config = RealTimeVisualizationConfig(
        update_interval=200.0,  # 5 FPS for smooth updates
        evolution_step_size=20.0,  # 20 seconds per update
        show_3d_embedding=True,
        enable_parameter_controls=True
    )
    
    dashboard = create_realtime_dashboard(config)
    
    # Create and display dashboard
    fig = dashboard.create_dashboard()
    
    print("Launching real-time throat evolution dashboard...")
    print(f"Available scenarios: {scenarios}")
    print("Use dashboard.switch_scenario('scenario_name') to change scenarios")
    print("Use dashboard.start_real_time_evolution() to begin evolution")
    print("Use dashboard.stop_real_time_evolution() to pause")
    
    # Show the dashboard
    pyo.plot(fig, filename="realtime_throat_evolution.html", auto_open=True)
    
    return dashboard