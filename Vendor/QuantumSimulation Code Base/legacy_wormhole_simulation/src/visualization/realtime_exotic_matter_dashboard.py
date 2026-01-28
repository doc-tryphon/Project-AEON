"""
Real-Time Exotic Matter Dashboard

A comprehensive real-time visualization and analysis platform for exploring exotic
matter configurations in wormhole physics. The dashboard provides interactive controls
for parameter adjustment, real-time stability analysis, and visualization of quantum
field effects.

Key Features:
------------
1. Interactive Visualization Components:
   - 3D energy density distribution
   - Energy condition violation maps
   - Stability landscape analysis
   - Quantum field evolution
   - Parameter space explorer

2. Real-time Analysis Tools:
   - Stability prediction using ML
   - Energy condition checking
   - Quantum effects computation
   - Geometric analysis
   - Performance metrics

3. Parameter Control:
   - Matter type selection
   - Energy density adjustment
   - Pressure components
   - Quantum corrections
   - Geometric parameters

Example Usage:
-------------
Basic Setup:
>>> from src.visualization import RealTimeExoticMatterDashboard
>>> dashboard = RealTimeExoticMatterDashboard()
>>> dashboard.run(port=8050)

Custom Configuration:
>>> config = DashboardConfig(
...     title="Wormhole Explorer",
...     theme="dark",
...     auto_refresh_interval=1000,
...     enable_live_updates=True
... )
>>> dashboard = RealTimeExoticMatterDashboard(config=config)

Parameter Optimization:
>>> dashboard.optimize_configuration(
...     target_stability=0.95,
...     constraint_tolerance=0.01
... )

Data Export:
>>> dashboard.export_results("analysis_results.json")
>>> dashboard.save_visualizations("output_directory")

Available Visualizations:
-----------------------
1. Energy Analysis:
   - 3D energy density plots
   - Pressure distribution maps
   - Energy condition violations
   - Stress-energy tensor components

2. Stability Analysis:
   - Parameter sweep landscapes
   - Stability threshold maps
   - Perturbation analysis
   - Time evolution plots

3. Quantum Effects:
   - Vacuum fluctuations
   - Hawking radiation
   - Entanglement entropy
   - Field correlations

4. Comparative Analysis:
   - Multiple configuration overlay
   - Matter type comparison
   - Parameter sensitivity
   - Optimization results

Dashboard Architecture:
--------------------
```
+------------------------+
|     Control Panel      |
+------------------------+
|  Parameter | Analysis  |
|  Controls  | Output    |
|------------------------+
|    Visualization       |
|      Display          |
+------------------------+
|    Status & Metrics    |
+------------------------+
```

Configuration Options:
-------------------
- theme: Visual theme ('light', 'dark', 'custom')
- auto_refresh: Update interval in milliseconds
- layout: Dashboard layout configuration
- data_sources: Input data configuration
- visualization_options: Plot customization
- optimization_settings: Parameter optimization
- export_options: Data export settings

Notes:
-----
- The dashboard requires a modern web browser
- GPU acceleration recommended for real-time 3D
- Minimum screen resolution: 1920x1080
- WebGL support required for 3D visualization
   - Dynamic stability analysis
   - Interactive 3D energy density plots
   - Energy condition violation mapping

2. Exotic Matter Types Supported
   - Advanced Casimir Effect
   - Phantom Dark Energy Fields
   - Quantum Inequality Constrained Matter
   - String Theory Derived Matter
   - Hybrid Configurations

3. Analysis Capabilities
   - Energy condition violations (NEC, WEC, SEC, DEC)
   - Stability analysis with eigenvalue computation
   - Sound speed calculations
   - Quantum field backreaction effects
   - Parameter optimization

4. Visualization Components
   - 3D matter distribution plots
   - Energy condition maps
   - Stability landscapes
   - Comparative configuration analysis
   - Parameter sweep visualizations
   - Quantum field evolution animations

Usage:
------
To launch the dashboard:
    dashboard = RealTimeExoticMatterDashboard()
    dashboard.run(port=8050)

To customize the configuration:
    config = DashboardConfig(
        title="Custom Dashboard",
        theme="dark",
        enable_live_updates=True,
        auto_refresh_interval=1000
    )
    dashboard = RealTimeExoticMatterDashboard(config=config)

For parameter optimization:
    dashboard.optimize_configuration(
        target_stability=0.95,
        constraint_tolerance=0.01
    )

Interactive Features:
------------------
- Real-time parameter adjustment sliders
- Matter type selection dropdown
- Energy condition toggles
- Stability threshold controls
- Animation speed adjustment
- Data export capabilities
- Configuration save/load
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import time
import threading
import json
from datetime import datetime
import queue
import logging

# Import exotic matter and visualization modules
from src.physics.exotic_matter import (
    ExoticMatter, AdvancedCasimirExoticMatter, PhantomDarkEnergyField,
    QuantumInequalityConstrainedMatter, StringTheoryDerivedMatter,
    HybridExoticMatter, optimize_exotic_matter_configuration,
    load_exotic_matter_from_catalog, ENHANCED_EXOTIC_MATTER_CATALOG
)

from src.visualization.exotic_matter_visualizer import ExoticMatterVisualizer, ExoticMatterVisualizationConfig


@dataclass
class DashboardConfig:
    """Configuration for real-time dashboard."""
    
    # Dashboard parameters
    title: str = "Quantum Wormhole Exotic Matter Explorer"
    theme: str = "bootstrap"  # 'bootstrap', 'dark', 'light'
    auto_refresh_interval: int = 2000  # milliseconds
    enable_live_updates: bool = True
    
    # Visualization parameters
    default_matter_type: str = "advanced_casimir"
    parameter_update_delay: int = 500  # milliseconds
    animation_speed: int = 100  # milliseconds per frame
    
    # Performance parameters
    max_grid_points: int = 5000
    computation_timeout: int = 30  # seconds
    cache_size: int = 100
    
    # Analysis parameters
    stability_threshold: float = 0.7
    energy_condition_threshold: float = 1e-10
    optimization_iterations: int = 50


class RealTimeExoticMatterDashboard:
    """Real-time interactive dashboard for exotic matter exploration."""
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize real-time dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        
        # Initialize components
        self.visualizer = ExoticMatterVisualizer()
        self.current_matter = None
        self.current_parameters = {}
        
        # Dash app
        self.app = None
        
        # Data cache and update queue
        self.data_cache = {}
        self.update_queue = queue.Queue()
        
        # Performance monitoring
        self.computation_times = []
        self.update_times = []
        
        # Initialize dashboard
        self._initialize_dashboard()
        
        print(f"Real-time exotic matter dashboard initialized")
    
    def _initialize_dashboard(self):
        """Initialize Dash application with layout and callbacks."""
        
        # Create Dash app
        if self.config.theme == 'bootstrap':
            external_stylesheets = [dbc.themes.BOOTSTRAP]
        elif self.config.theme == 'dark':
            external_stylesheets = [dbc.themes.CYBORG]
        else:
            external_stylesheets = [dbc.themes.FLATLY]
        
        self.app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            suppress_callback_exceptions=True
        )
        
        self.app.title = self.config.title
        
        # Set layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
    
    def _create_layout(self) -> html.Div:
        """Create dashboard layout."""
        
        # Header
        header = dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand(self.config.title, className="ms-2")),
                    ], align="center", className="g-0"),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            ]),
            color="primary",
            dark=True,
        )
        
        # Control panel
        control_panel = dbc.Card([
            dbc.CardHeader("Exotic Matter Configuration"),
            dbc.CardBody([
                # Matter type selection
                html.Div([
                    html.Label("Exotic Matter Type:", className="form-label"),
                    dcc.Dropdown(
                        id="matter-type-dropdown",
                        options=[
                            {"label": catalog_entry["description"], "value": matter_type}
                            for matter_type, catalog_entry in ENHANCED_EXOTIC_MATTER_CATALOG.items()
                        ],
                        value=self.config.default_matter_type,
                        clearable=False
                    )
                ], className="mb-3"),
                
                # Dynamic parameter controls (will be populated based on matter type)
                html.Div(id="parameter-controls"),
                
                # Analysis controls
                html.Hr(),
                html.H6("Analysis Options", className="text-muted"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id="analysis-options",
                            options=[
                                {"label": "Energy Conditions", "value": "energy_conditions"},
                                {"label": "Stability Analysis", "value": "stability"},
                                {"label": "3D Distribution", "value": "3d_distribution"},
                                {"label": "Time Evolution", "value": "time_evolution"}
                            ],
                            value=["energy_conditions", "stability"],
                            inline=True
                        )
                    ])
                ]),
                
                # Optimization controls
                html.Hr(),
                html.H6("Optimization", className="text-muted"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Optimize Configuration",
                            id="optimize-button",
                            color="success",
                            size="sm",
                            className="me-2"
                        ),
                        dbc.Button(
                            "Reset Parameters", 
                            id="reset-button",
                            color="secondary",
                            size="sm"
                        )
                    ])
                ]),
                
                # Real-time updates toggle
                html.Hr(),
                dbc.Switch(
                    id="realtime-toggle",
                    label="Real-time Updates",
                    value=self.config.enable_live_updates
                )
            ])
        ], className="mb-4")
        
        # Main visualization area
        main_content = dbc.Card([
            dbc.CardHeader([
                html.H5("Visualization Dashboard", className="mb-0"),
                dbc.Badge("Live", color="success", className="ms-2", id="status-badge")
            ]),
            dbc.CardBody([
                # Tabs for different visualization types
                dbc.Tabs([
                    dbc.Tab(
                        label="Energy Conditions",
                        tab_id="energy-conditions-tab",
                        children=[
                            dcc.Graph(
                                id="energy-conditions-plot",
                                style={"height": "600px"}
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="Stability Analysis", 
                        tab_id="stability-tab",
                        children=[
                            dcc.Graph(
                                id="stability-plot",
                                style={"height": "600px"}
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="3D Distribution",
                        tab_id="3d-distribution-tab",
                        children=[
                            dcc.Graph(
                                id="3d-distribution-plot",
                                style={"height": "600px"}
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="Comparative Analysis",
                        tab_id="comparative-tab",
                        children=[
                            dcc.Graph(
                                id="comparative-plot",
                                style={"height": "600px"}
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="Time Evolution",
                        tab_id="time-evolution-tab", 
                        children=[
                            dcc.Graph(
                                id="time-evolution-plot",
                                style={"height": "600px"}
                            )
                        ]
                    )
                ], id="visualization-tabs", active_tab="energy-conditions-tab")
            ])
        ])
        
        # Performance monitoring panel
        performance_panel = dbc.Card([
            dbc.CardHeader("Performance Monitor"),
            dbc.CardBody([
                html.Div(id="performance-stats"),
                dcc.Graph(
                    id="performance-plot",
                    style={"height": "200px"}
                )
            ])
        ], className="mt-4")
        
        # Layout assembly
        layout = dbc.Container([
            header,
            
            html.Div([
                dbc.Row([
                    dbc.Col([control_panel, performance_panel], width=3),
                    dbc.Col([main_content], width=9)
                ])
            ], className="mt-4"),
            
            # Hidden components for data storage
            dcc.Store(id="matter-data-store"),
            dcc.Store(id="parameter-store"),
            dcc.Store(id="computation-store"),
            
            # Interval component for real-time updates
            dcc.Interval(
                id="update-interval",
                interval=self.config.auto_refresh_interval,
                n_intervals=0,
                disabled=not self.config.enable_live_updates
            )
        ], fluid=True)
        
        return layout
    
    def _register_callbacks(self):
        """Register all dashboard callbacks."""
        
        # Matter type selection callback
        @self.app.callback(
            [Output("parameter-controls", "children"),
             Output("matter-data-store", "data")],
            [Input("matter-type-dropdown", "value")]
        )
        def update_matter_type(matter_type):
            """Update parameter controls when matter type changes."""
            
            try:
                # Load matter instance
                matter = load_exotic_matter_from_catalog(matter_type)
                self.current_matter = matter
                
                # Create parameter controls based on matter type
                parameter_controls = self._create_parameter_controls(matter_type)
                
                # Store matter data
                matter_data = {
                    "type": matter_type,
                    "name": matter.name,
                    "timestamp": datetime.now().isoformat()
                }
                
                return parameter_controls, matter_data
                
            except Exception as e:
                print(f"Error updating matter type: {e}")
                return html.Div("Error loading matter type"), {}
        
        # Parameter update callback
        @self.app.callback(
            Output("parameter-store", "data"),
            [Input(f"param-{param}", "value") for param in ["throat_radius", "plate_separation", "temperature", "field_amplitude"]],
            [State("matter-type-dropdown", "value")]
        )
        def update_parameters(*args):
            """Update parameters when sliders change."""
            
            matter_type = args[-1]  # Last argument is the state (matter type)
            param_values = args[:-1]  # All other arguments are parameter values
            
            # Map parameter values to their names
            param_names = ["throat_radius", "plate_separation", "temperature", "field_amplitude"]
            parameters = {}
            
            for name, value in zip(param_names, param_values):
                if value is not None:
                    parameters[name] = value
            
            # Update current matter instance with new parameters
            try:
                if matter_type and parameters:
                    self.current_matter = load_exotic_matter_from_catalog(matter_type, **parameters)
                    self.current_parameters = parameters
            except Exception as e:
                print(f"Error updating parameters: {e}")
            
            return {
                "matter_type": matter_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
        
        # Main visualization update callback
        @self.app.callback(
            [Output("energy-conditions-plot", "figure"),
             Output("stability-plot", "figure"),
             Output("3d-distribution-plot", "figure"),
             Output("comparative-plot", "figure"),
             Output("time-evolution-plot", "figure"),
             Output("computation-store", "data")],
            [Input("parameter-store", "data"),
             Input("analysis-options", "value"),
             Input("update-interval", "n_intervals")],
            [State("realtime-toggle", "value")]
        )
        def update_visualizations(parameter_data, analysis_options, n_intervals, realtime_enabled):
            """Update all visualizations based on current parameters."""
            
            start_time = time.time()
            
            try:
                if self.current_matter is None:
                    # Return empty figures if no matter is loaded
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(
                        text="Select exotic matter type to begin",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return [empty_fig] * 5, {}
                
                # Update visualizer with current matter
                self.visualizer.set_exotic_matter(self.current_matter)
                
                # Create visualizations based on selected options
                figures = {}
                
                if "energy_conditions" in analysis_options:
                    figures["energy_conditions"] = self.visualizer.create_energy_condition_map()
                else:
                    figures["energy_conditions"] = go.Figure()
                
                if "stability" in analysis_options:
                    figures["stability"] = self.visualizer.create_stability_landscape()
                else:
                    figures["stability"] = go.Figure()
                
                if "3d_distribution" in analysis_options:
                    figures["3d_distribution"] = self.visualizer.create_matter_distribution_3d()
                else:
                    figures["3d_distribution"] = go.Figure()
                
                # Comparative analysis (always compute for multiple matter types)
                matter_list = [
                    load_exotic_matter_from_catalog("advanced_casimir"),
                    load_exotic_matter_from_catalog("phantom_dark_energy")
                ]
                figures["comparative"] = self.visualizer.create_comparative_analysis(
                    matter_list, "energy_conditions"
                )
                
                if "time_evolution" in analysis_options:
                    figures["time_evolution"] = self.visualizer.create_time_evolution_animation()
                else:
                    figures["time_evolution"] = go.Figure()
                
                computation_time = time.time() - start_time
                self.computation_times.append(computation_time)
                
                # Keep only last 100 computation times
                if len(self.computation_times) > 100:
                    self.computation_times = self.computation_times[-100:]
                
                computation_data = {
                    "computation_time": computation_time,
                    "total_computations": len(self.computation_times),
                    "avg_computation_time": np.mean(self.computation_times),
                    "timestamp": datetime.now().isoformat()
                }
                
                return (
                    figures["energy_conditions"],
                    figures["stability"],
                    figures["3d_distribution"],
                    figures["comparative"],
                    figures["time_evolution"],
                    computation_data
                )
                
            except Exception as e:
                print(f"Error updating visualizations: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return [error_fig] * 5, {"error": str(e)}
        
        # Performance monitoring callback
        @self.app.callback(
            [Output("performance-stats", "children"),
             Output("performance-plot", "figure")],
            [Input("computation-store", "data")]
        )
        def update_performance_monitor(computation_data):
            """Update performance monitoring display."""
            
            if not computation_data:
                return "No data available", go.Figure()
            
            # Performance statistics
            stats = [
                html.P(f"Last Computation: {computation_data.get('computation_time', 0):.3f}s"),
                html.P(f"Average Time: {computation_data.get('avg_computation_time', 0):.3f}s"),
                html.P(f"Total Computations: {computation_data.get('total_computations', 0)}")
            ]
            
            # Performance plot
            if self.computation_times:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(self.computation_times))),
                        y=self.computation_times,
                        mode='lines+markers',
                        name='Computation Time',
                        line=dict(color='blue', width=2)
                    )
                )
                
                fig.update_layout(
                    title="Computation Performance",
                    xaxis_title="Computation #",
                    yaxis_title="Time (s)",
                    height=200,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            else:
                fig = go.Figure()
            
            return stats, fig
        
        # Optimization callback
        @self.app.callback(
            Output("matter-type-dropdown", "value", allow_duplicate=True),
            [Input("optimize-button", "n_clicks")],
            [State("matter-type-dropdown", "value")],
            prevent_initial_call=True
        )
        def run_optimization(n_clicks, current_matter_type):
            """Run parameter optimization."""
            
            if n_clicks and self.current_matter:
                try:
                    # Run optimization
                    throat_radius = self.current_parameters.get("throat_radius", 1e3)
                    
                    optimization_result = optimize_exotic_matter_configuration(
                        throat_radius=throat_radius,
                        matter_types=[current_matter_type],
                        optimization_method="differential_evolution"
                    )
                    
                    # Update parameters with optimized values
                    if optimization_result["energy_budget_satisfied"]:
                        optimal_params = optimization_result["best_configuration"]["optimal_parameters"]
                        
                        # Update current matter with optimal parameters
                        self.current_matter = load_exotic_matter_from_catalog(
                            current_matter_type, **optimal_params
                        )
                        self.current_parameters.update(optimal_params)
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
            
            return current_matter_type
        
        # Real-time updates toggle
        @self.app.callback(
            Output("update-interval", "disabled"),
            [Input("realtime-toggle", "value")]
        )
        def toggle_realtime_updates(enabled):
            """Toggle real-time updates on/off."""
            return not enabled
    
    def _create_parameter_controls(self, matter_type: str) -> html.Div:
        """Create parameter control widgets based on matter type.
        
        Args:
            matter_type: Type of exotic matter
            
        Returns:
            HTML div containing parameter controls
        """
        
        controls = []
        
        if matter_type == "advanced_casimir":
            controls.extend([
                html.Label("Plate Separation (m):", className="form-label"),
                dcc.Slider(
                    id="param-plate_separation",
                    min=1e-9, max=1e-5,
                    value=1e-6,
                    marks={1e-9: "1nm", 1e-6: "1μm", 1e-5: "10μm"},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Label("Temperature (K):", className="form-label mt-3"),
                dcc.Slider(
                    id="param-temperature",
                    min=0.1, max=1000,
                    value=300,
                    marks={0.1: "0.1K", 300: "300K", 1000: "1000K"},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
            
        elif matter_type == "phantom_dark_energy":
            controls.extend([
                html.Label("Field Amplitude:", className="form-label"),
                dcc.Slider(
                    id="param-field_amplitude",
                    min=0.1, max=10.0,
                    value=1.0,
                    marks={0.1: "0.1", 1.0: "1.0", 10.0: "10.0"},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
            
        elif matter_type == "quantum_inequality":
            controls.extend([
                html.Label("Throat Radius (m):", className="form-label"),
                dcc.Slider(
                    id="param-throat_radius",
                    min=1e2, max=1e5,
                    value=1e3,
                    marks={1e2: "100m", 1e3: "1km", 1e5: "100km"},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
            
        elif matter_type == "string_theory":
            controls.extend([
                html.Label("Compactification Scale (m):", className="form-label"),
                dcc.Slider(
                    id="param-compactification_scale",
                    min=1e-36, max=1e-30,
                    value=1e-35,
                    marks={1e-36: "10⁻³⁶", 1e-35: "10⁻³⁵", 1e-30: "10⁻³⁰"},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        
        # Add placeholder sliders for unused parameters (to avoid callback errors)
        all_params = ["throat_radius", "plate_separation", "temperature", "field_amplitude"]
        used_params = {control.id.split("-")[1] for control in controls if hasattr(control, 'id') and control.id}
        
        for param in all_params:
            if f"param-{param}" not in [f"param-{p}" for p in used_params]:
                controls.append(
                    dcc.Slider(
                        id=f"param-{param}",
                        min=0, max=1, value=0.5,
                        style={"display": "none"}  # Hidden
                    )
                )
        
        return html.Div(controls)
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """Run the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        
        print(f"Starting dashboard server at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        self.app.run_server(
            host=host,
            port=port, 
            debug=debug,
            use_reloader=False  # Avoid issues with threading
        )
    
    def save_dashboard_state(self, filename: str):
        """Save current dashboard state to file.
        
        Args:
            filename: Output filename
        """
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "current_matter_type": self.current_matter.name if self.current_matter else None,
            "current_parameters": self.current_parameters,
            "computation_times": self.computation_times,
            "config": self.config.__dict__
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"Dashboard state saved to {filename}")
    
    def load_dashboard_state(self, filename: str):
        """Load dashboard state from file.
        
        Args:
            filename: Input filename
        """
        
        with open(filename, 'r') as f:
            state = json.load(f)
        
        # Restore state
        self.current_parameters = state.get("current_parameters", {})
        self.computation_times = state.get("computation_times", [])
        
        print(f"Dashboard state loaded from {filename}")


def create_demo_dashboard() -> RealTimeExoticMatterDashboard:
    """Create demonstration dashboard with sample configuration.
    
    Returns:
        Configured dashboard instance
    """
    
    config = DashboardConfig(
        title="Quantum Wormhole Exotic Matter Explorer - Demo",
        theme="bootstrap",
        auto_refresh_interval=3000,
        enable_live_updates=True,
        default_matter_type="advanced_casimir"
    )
    
    dashboard = RealTimeExoticMatterDashboard(config)
    
    return dashboard


if __name__ == "__main__":
    # Create and run demo dashboard
    print("Launching Real-Time Exotic Matter Dashboard Demo")
    print("="*60)
    
    dashboard = create_demo_dashboard()
    
    # Run server
    dashboard.run_server(host="127.0.0.1", port=8050, debug=False)