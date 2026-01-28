#!/usr/bin/env python3
"""
Interactive Visualization Example

This example demonstrates the interactive visualization capabilities of the quantum
wormhole simulation framework, including real-time parameter adjustment and 
3D spacetime visualization.

Topics covered:
- Interactive dashboard setup
- Real-time parameter adjustment
- 3D spacetime visualization
- Live data streaming
- User interaction handling
"""

import sys
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import threading
import time
import queue
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration import WormholeSimulationFramework, IntegrationConfig
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import CasimirExoticMatter


class InteractiveVisualizationDemo:
    """Interactive visualization demo with real-time controls."""
    
    def __init__(self):
        """Initialize the interactive demo."""
        self.framework = None
        self.simulation_running = False
        self.data_queue = queue.Queue()
        self.current_step = 0
        self.simulation_thread = None
        
        # Default parameters
        self.default_params = {
            'throat_radius': 1000.0,
            'mass': 1e30,
            'casimir_energy': -1e15,
            'num_qubits': 6,
            'traversal_probability': 0.8,
            'decoherence_rate': 0.01
        }
        
        # Data storage
        self.stability_data = []
        self.physics_data = []
        self.quantum_data = []
        self.time_stamps = []

    def create_dash_app(self):
        """Create the Dash web application."""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("🌌 Quantum Wormhole Interactive Visualization", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control Panel
            html.Div([
                html.H3("Simulation Controls", style={'marginBottom': 20}),
                
                # Physics Parameters
                html.Div([
                    html.H4("Physics Parameters"),
                    html.Label("Throat Radius (m):"),
                    dcc.Slider(
                        id='throat-radius-slider',
                        min=100, max=10000, step=100,
                        value=self.default_params['throat_radius'],
                        marks={i: f'{i}' for i in range(1000, 11000, 2000)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Mass (kg):"),
                    dcc.Slider(
                        id='mass-slider',
                        min=1e29, max=1e32, step=1e29,
                        value=self.default_params['mass'],
                        marks={int(i): f'{i:.0e}' for i in np.logspace(29, 32, 4)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Casimir Energy (J):"),
                    dcc.Slider(
                        id='casimir-energy-slider',
                        min=-1e16, max=-1e14, step=1e14,
                        value=self.default_params['casimir_energy'],
                        marks={int(i): f'{i:.0e}' for i in np.logspace(14, 16, 3) * -1},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Quantum Parameters
                html.Div([
                    html.H4("Quantum Parameters"),
                    html.Label("Number of Qubits:"),
                    dcc.Slider(
                        id='qubits-slider',
                        min=2, max=12, step=1,
                        value=self.default_params['num_qubits'],
                        marks={i: str(i) for i in range(2, 13, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Traversal Probability:"),
                    dcc.Slider(
                        id='traversal-slider',
                        min=0.1, max=1.0, step=0.1,
                        value=self.default_params['traversal_probability'],
                        marks={i/10: f'{i/10:.1f}' for i in range(1, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Decoherence Rate:"),
                    dcc.Slider(
                        id='decoherence-slider',
                        min=0.001, max=0.1, step=0.001,
                        value=self.default_params['decoherence_rate'],
                        marks={i/1000: f'{i/1000:.3f}' for i in range(1, 101, 25)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'marginBottom': 30, 'padding': 20, 'border': '1px solid #ddd', 'borderRadius': 5}),
            
            # Control Buttons
            html.Div([
                html.Button('Start Simulation', id='start-button', n_clicks=0, 
                           style={'backgroundColor': '#4CAF50', 'color': 'white', 'marginRight': 10}),
                html.Button('Stop Simulation', id='stop-button', n_clicks=0,
                           style={'backgroundColor': '#f44336', 'color': 'white', 'marginRight': 10}),
                html.Button('Reset Parameters', id='reset-button', n_clicks=0,
                           style={'backgroundColor': '#008CBA', 'color': 'white', 'marginRight': 10}),
                html.Button('Save Results', id='save-button', n_clicks=0,
                           style={'backgroundColor': '#FF9800', 'color': 'white'})
            ], style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Status Display
            html.Div(id='status-display', 
                    style={'textAlign': 'center', 'fontSize': 18, 'marginBottom': 20}),
            
            # Real-time Plots
            html.Div([
                dcc.Graph(id='stability-plot', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='quantum-plot', style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='physics-plot', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='spacetime-3d', style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            ),
            
            # Hidden div to store simulation data
            html.Div(id='simulation-data', style={'display': 'none'})
        ])
        
        # Callbacks
        self.setup_callbacks(app)
        
        return app

    def setup_callbacks(self, app):
        """Setup Dash callbacks for interactivity."""
        
        @app.callback(
            [Output('status-display', 'children'),
             Output('stability-plot', 'figure'),
             Output('quantum-plot', 'figure'),
             Output('physics-plot', 'figure'),
             Output('spacetime-3d', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('start-button', 'n_clicks'),
             Input('stop-button', 'n_clicks'),
             Input('reset-button', 'n_clicks')],
            [State('throat-radius-slider', 'value'),
             State('mass-slider', 'value'),
             State('casimir-energy-slider', 'value'),
             State('qubits-slider', 'value'),
             State('traversal-slider', 'value'),
             State('decoherence-slider', 'value')]
        )
        def update_dashboard(n_intervals, start_clicks, stop_clicks, reset_clicks,
                           throat_radius, mass, casimir_energy, num_qubits, 
                           traversal_prob, decoherence_rate):
            
            ctx = dash.callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'start-button' and not self.simulation_running:
                    self.start_simulation(throat_radius, mass, casimir_energy,
                                        num_qubits, traversal_prob, decoherence_rate)
                elif button_id == 'stop-button':
                    self.stop_simulation()
                elif button_id == 'reset-button':
                    self.reset_simulation()
            
            # Update plots with latest data
            status = self.get_status_message()
            stability_fig = self.create_stability_plot()
            quantum_fig = self.create_quantum_plot()
            physics_fig = self.create_physics_plot()
            spacetime_fig = self.create_spacetime_3d()
            
            return status, stability_fig, quantum_fig, physics_fig, spacetime_fig

    def start_simulation(self, throat_radius, mass, casimir_energy, 
                        num_qubits, traversal_prob, decoherence_rate):
        """Start the simulation with given parameters."""
        if self.simulation_running:
            return
            
        print(f"🚀 Starting interactive simulation...")
        print(f"   Throat radius: {throat_radius} m")
        print(f"   Mass: {mass:.1e} kg")
        print(f"   Qubits: {num_qubits}")
        
        # Clear previous data
        self.stability_data.clear()
        self.physics_data.clear()
        self.quantum_data.clear()
        self.time_stamps.clear()
        self.current_step = 0
        
        # Create configuration
        config = IntegrationConfig(
            simulation_name="interactive_visualization",
            time_steps=1000,  # Long-running simulation
            dt=0.1,
            num_qubits=int(num_qubits),
            enable_stability_prediction=True,
            enable_real_time_visualization=True
        )
        
        # Initialize framework
        self.framework = WormholeSimulationFramework(config)
        
        # Set parameters
        wormhole_params = {
            'b0': throat_radius,
            'mass': mass,
            'casimir_energy': casimir_energy
        }
        
        quantum_params = {
            'num_qubits': int(num_qubits),
            'traversal_probability': traversal_prob,
            'entanglement_strength': 1.0,
            'decoherence_rate': decoherence_rate
        }
        
        ai_params = {
            'stability_threshold': 0.5,
            'optimization_target': 'stability'
        }
        
        self.framework.initialize_system(
            wormhole_params=wormhole_params,
            quantum_params=quantum_params,
            ai_params=ai_params
        )
        
        # Start simulation in separate thread
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def _run_simulation_loop(self):
        """Run the simulation loop in a separate thread."""
        try:
            for step in range(1000):  # Run for many steps
                if not self.simulation_running:
                    break
                    
                # Simulate one step
                step_results = self._simulate_step(step)
                
                # Store data
                self.stability_data.append(step_results.get('stability', 0))
                self.physics_data.append(step_results.get('physics', {}))
                self.quantum_data.append(step_results.get('quantum', {}))
                self.time_stamps.append(datetime.now())
                self.current_step = step
                
                # Limit data storage to last 100 points
                if len(self.stability_data) > 100:
                    self.stability_data.pop(0)
                    self.physics_data.pop(0)
                    self.quantum_data.pop(0)
                    self.time_stamps.pop(0)
                
                time.sleep(0.5)  # Slow down for visualization
                
        except Exception as e:
            print(f"❌ Simulation error: {e}")
        finally:
            self.simulation_running = False

    def _simulate_step(self, step):
        """Simulate a single step and return results."""
        # Simplified simulation step for demonstration
        try:
            # Physics calculations
            energy_density = -1e15 * (1 + 0.1 * np.sin(step * 0.1))
            pressure = -0.5 * energy_density
            
            # Stability calculation
            stability = 0.7 + 0.2 * np.sin(step * 0.05) + np.random.normal(0, 0.05)
            stability = np.clip(stability, 0, 1)
            
            # Quantum metrics
            concurrence = 0.8 * np.exp(-step * 0.001) + np.random.normal(0, 0.02)
            entropy = 1.5 + 0.3 * np.sin(step * 0.08)
            
            return {
                'stability': stability,
                'physics': {
                    'energy_density': energy_density,
                    'pressure': pressure,
                    'step': step
                },
                'quantum': {
                    'concurrence': max(0, concurrence),
                    'entropy': entropy,
                    'step': step
                }
            }
        except Exception as e:
            print(f"Step simulation error: {e}")
            return {'stability': 0, 'physics': {}, 'quantum': {}}

    def stop_simulation(self):
        """Stop the running simulation."""
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
        print("🛑 Simulation stopped")

    def reset_simulation(self):
        """Reset simulation data."""
        self.stop_simulation()
        self.stability_data.clear()
        self.physics_data.clear()
        self.quantum_data.clear()
        self.time_stamps.clear()
        self.current_step = 0
        print("🔄 Simulation reset")

    def get_status_message(self):
        """Get current simulation status."""
        if self.simulation_running:
            return f"🟢 Simulation Running - Step: {self.current_step}"
        else:
            return f"🔴 Simulation Stopped - Steps completed: {len(self.stability_data)}"

    def create_stability_plot(self):
        """Create real-time stability plot."""
        fig = go.Figure()
        
        if self.stability_data:
            steps = list(range(len(self.stability_data)))
            fig.add_trace(go.Scatter(
                x=steps,
                y=self.stability_data,
                mode='lines+markers',
                name='Stability Score',
                line=dict(color='blue', width=2)
            ))
            
            # Add threshold line
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Stability Threshold")
        
        fig.update_layout(
            title="Real-Time Stability Evolution",
            xaxis_title="Simulation Step",
            yaxis_title="Stability Score",
            yaxis=dict(range=[0, 1]),
            height=300
        )
        
        return fig

    def create_quantum_plot(self):
        """Create quantum metrics plot."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if self.quantum_data:
            steps = list(range(len(self.quantum_data)))
            concurrences = [q.get('concurrence', 0) for q in self.quantum_data]
            entropies = [q.get('entropy', 0) for q in self.quantum_data]
            
            fig.add_trace(go.Scatter(
                x=steps, y=concurrences,
                mode='lines', name='Concurrence',
                line=dict(color='purple', width=2)
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=steps, y=entropies,
                mode='lines', name='Entropy',
                line=dict(color='orange', width=2)
            ), secondary_y=True)
        
        fig.update_xaxes(title_text="Simulation Step")
        fig.update_yaxes(title_text="Concurrence", secondary_y=False)
        fig.update_yaxes(title_text="Entropy", secondary_y=True)
        fig.update_layout(title="Quantum Entanglement Metrics", height=300)
        
        return fig

    def create_physics_plot(self):
        """Create physics metrics plot."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if self.physics_data:
            steps = list(range(len(self.physics_data)))
            energies = [p.get('energy_density', 0) for p in self.physics_data]
            pressures = [p.get('pressure', 0) for p in self.physics_data]
            
            fig.add_trace(go.Scatter(
                x=steps, y=energies,
                mode='lines', name='Energy Density',
                line=dict(color='green', width=2)
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=steps, y=pressures,
                mode='lines', name='Pressure',
                line=dict(color='red', width=2)
            ), secondary_y=True)
        
        fig.update_xaxes(title_text="Simulation Step")
        fig.update_yaxes(title_text="Energy Density (J/m³)", secondary_y=False)
        fig.update_yaxes(title_text="Pressure (Pa)", secondary_y=True)
        fig.update_layout(title="Exotic Matter Properties", height=300)
        
        return fig

    def create_spacetime_3d(self):
        """Create 3D spacetime visualization."""
        # Create wormhole geometry visualization
        r = np.linspace(0.1, 5, 30)
        theta = np.linspace(0, 2*np.pi, 30)
        R, THETA = np.meshgrid(r, theta)
        
        # Morris-Thorne throat geometry (simplified)
        b0 = 1.0  # Normalized throat radius
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        Z = np.sqrt(R**2 - b0**2)  # Embedding function approximation
        Z = np.where(R < b0, 0, Z)  # Handle throat region
        
        # Add time evolution effect if simulation is running
        if self.simulation_running and self.stability_data:
            time_factor = 1 + 0.1 * self.stability_data[-1] if self.stability_data else 1
            Z = Z * time_factor
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Curvature")
        )])
        
        # Add throat visualization
        throat_r = np.linspace(0, b0, 10)
        throat_theta = np.linspace(0, 2*np.pi, 20)
        TR, TTHETA = np.meshgrid(throat_r, throat_theta)
        TX = TR * np.cos(TTHETA)
        TY = TR * np.sin(TTHETA)
        TZ = np.zeros_like(TX)
        
        fig.add_trace(go.Surface(
            x=TX, y=TY, z=TZ,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            opacity=0.8
        ))
        
        fig.update_layout(
            title="3D Wormhole Spacetime Geometry",
            scene=dict(
                xaxis_title="X (normalized)",
                yaxis_title="Y (normalized)",
                zaxis_title="Z (embedding)",
                aspectmode='cube'
            ),
            height=400
        )
        
        return fig

    def run_interactive_demo(self, port=8050, debug=False):
        """Run the interactive visualization demo."""
        print(f"🌐 Starting interactive visualization on port {port}")
        print(f"   Open your browser to: http://localhost:{port}")
        print(f"   Use the controls to adjust parameters in real-time!")
        
        app = self.create_dash_app()
        app.run_server(host='127.0.0.1', port=port, debug=debug)


def main():
    """Run the interactive visualization example."""
    print("🎮 Interactive Quantum Wormhole Visualization")
    print("=" * 50)
    
    demo = InteractiveVisualizationDemo()
    
    try:
        print("\n📋 Instructions:")
        print("1. The web interface will open automatically")
        print("2. Use sliders to adjust physics and quantum parameters")
        print("3. Click 'Start Simulation' to begin real-time simulation")
        print("4. Watch the plots update in real-time")
        print("5. Use 'Stop' and 'Reset' buttons to control the simulation")
        print("6. The 3D plot shows the wormhole spacetime geometry")
        
        print("\n🚀 Launching interactive dashboard...")
        demo.run_interactive_demo(port=8050, debug=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Interactive demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error running interactive demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()