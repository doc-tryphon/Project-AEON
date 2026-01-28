# API Reference

Complete API documentation for the Quantum Wormhole Simulation Framework.

## Table of Contents

1. [Overview](#overview)
2. [Core Integration](#core-integration)
3. [Physics Engine](#physics-engine)
4. [Quantum System](#quantum-system)
5. [AI Components](#ai-components)
6. [Visualization](#visualization)
7. [Utilities](#utilities)
8. [Examples](#examples)

---

## Overview

The Quantum Wormhole Simulation Framework provides a comprehensive API for simulating traversable wormholes with quantum mechanics and AI analysis.

### Package Structure

```
src/
├── integration.py          # Main integration framework
├── physics/                # Physics engine components
├── quantum/                # Quantum mechanics modules
├── ai/                     # Machine learning components
├── visualization/          # Visualization and plotting
└── utils/                  # Utility functions
```

### Import Pattern

```python
# Main framework
from src.integration import WormholeSimulationFramework, IntegrationConfig

# Physics components  
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.exotic_matter import CasimirExoticMatter

# Quantum components
from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.quantum.entanglement_dynamics import EntanglementDynamics

# AI components
from src.ai.stability_predictor import StabilityPredictor
from src.ai.parameter_optimizer import ParameterOptimizer

# Visualization
from src.visualization.interactive_dashboard import InteractiveDashboard
```

---

## Core Integration

### WormholeSimulationFramework

**Class**: `src.integration.WormholeSimulationFramework`

The main class that orchestrates all simulation components.

#### Constructor

```python
def __init__(self, config: IntegrationConfig = None)
```

**Parameters:**
- `config` (IntegrationConfig, optional): Configuration object for simulation parameters

**Example:**
```python
from src.integration import WormholeSimulationFramework, IntegrationConfig

config = IntegrationConfig(
    simulation_name="my_wormhole_sim",
    time_steps=1000,
    num_qubits=8
)

framework = WormholeSimulationFramework(config)
```

#### Methods

##### initialize_system()

```python
def initialize_system(self, 
                     wormhole_params: Dict[str, Any] = None,
                     quantum_params: Dict[str, Any] = None,
                     ai_params: Dict[str, Any] = None,
                     visualization_params: Dict[str, Any] = None) -> None
```

Initialize all simulation subsystems.

**Parameters:**
- `wormhole_params`: Physics parameters (throat radius, mass, etc.)
- `quantum_params`: Quantum system parameters (qubits, entanglement, etc.)
- `ai_params`: AI system parameters (thresholds, objectives, etc.)
- `visualization_params`: Visualization settings

**Example:**
```python
framework.initialize_system(
    wormhole_params={'b0': 1000.0, 'mass': 1e30},
    quantum_params={'num_qubits': 8, 'traversal_probability': 0.8},
    ai_params={'stability_threshold': 0.5}
)
```

##### run_simulation()

```python
def run_simulation(self, 
                  duration: Optional[float] = None,
                  callback: Optional[Callable] = None) -> SimulationResults
```

Execute the complete simulation.

**Parameters:**
- `duration`: Simulation duration (overrides time_steps if provided)
- `callback`: Optional callback function called each step

**Returns:**
- `SimulationResults`: Complete simulation results and analysis

**Example:**
```python
def progress_callback(step, step_results):
    print(f"Step {step}: Stability = {step_results.get('stability', 0):.3f}")

results = framework.run_simulation(duration=10.0, callback=progress_callback)
```

##### generate_comprehensive_report()

```python
def generate_comprehensive_report(self) -> Dict[str, Any]
```

Generate detailed analysis report.

**Returns:**
- `Dict`: Comprehensive analysis including physics, quantum, AI, and performance metrics

**Example:**
```python
report = framework.generate_comprehensive_report()
print(f"Average stability: {report['summary']['average_stability']:.3f}")
```

### IntegrationConfig

**Class**: `src.integration.IntegrationConfig`

Configuration dataclass for simulation parameters.

#### Attributes

```python
@dataclass
class IntegrationConfig:
    # Simulation metadata
    simulation_name: str = "quantum_wormhole_simulation"
    time_steps: int = 1000
    dt: float = 0.1
    
    # Physics configuration
    use_relativistic_corrections: bool = True
    include_quantum_corrections: bool = True
    enable_exotic_matter: bool = True
    
    # Quantum system configuration
    num_qubits: int = 8
    quantum_coherence_time: float = 100.0
    enable_decoherence: bool = True
    
    # AI configuration
    enable_stability_prediction: bool = True
    enable_parameter_optimization: bool = True
    enable_anomaly_detection: bool = True
    
    # Performance configuration
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
```

**Example:**
```python
config = IntegrationConfig(
    simulation_name="high_precision_simulation",
    time_steps=5000,
    dt=0.01,
    num_qubits=12,
    enable_stability_prediction=True,
    parallel_processing=True,
    max_workers=8
)
```

### SimulationResults

**Class**: `src.integration.SimulationResults`

Container for simulation results and analysis.

#### Attributes

```python
@dataclass
class SimulationResults:
    # Metadata
    simulation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[IntegrationConfig] = None
    
    # Physics results
    spacetime_evolution: List[Dict] = field(default_factory=list)
    field_strengths: Dict[str, List[float]] = field(default_factory=dict)
    stress_energy_evolution: List[np.ndarray] = field(default_factory=list)
    
    # Quantum results
    quantum_state_evolution: List[Dict] = field(default_factory=list)
    entanglement_measures: Dict[str, List[float]] = field(default_factory=dict)
    quantum_correlations: List[np.ndarray] = field(default_factory=list)
    
    # AI analysis results
    stability_predictions: List[float] = field(default_factory=list)
    anomaly_scores: List[float] = field(default_factory=list)
    optimization_history: List[Dict] = field(default_factory=list)
```

---

## Physics Engine

### SpacetimeMetric

**Abstract Base Class**: `src.physics.spacetime_metrics.SpacetimeMetric`

Base class for all spacetime metrics.

#### Methods

##### metric_tensor()

```python
def metric_tensor(self, coordinates: Tuple[float, float, float, float]) -> np.ndarray
```

Compute the metric tensor at given coordinates.

**Parameters:**
- `coordinates`: (t, r, θ, φ) spacetime coordinates

**Returns:**
- `np.ndarray`: 4×4 metric tensor

##### christoffel_symbols()

```python
def christoffel_symbols(self, coordinates: Tuple[float, float, float, float]) -> np.ndarray
```

Compute Christoffel symbols.

**Parameters:**
- `coordinates`: (t, r, θ, φ) spacetime coordinates

**Returns:**
- `np.ndarray`: 4×4×4 Christoffel symbol array

### MorrisThorneeWormhole

**Class**: `src.physics.spacetime_metrics.MorrisThorneeWormhole`

Morris-Thorne traversable wormhole implementation.

#### Constructor

```python
def __init__(self, b0: float, mass: float = 0.0, shape_function_type: str = "exponential")
```

**Parameters:**
- `b0`: Throat radius in meters
- `mass`: Wormhole mass in kg (default: 0.0)
- `shape_function_type`: Type of shape function ("exponential", "power_law", "hyperbolic")

**Example:**
```python
wormhole = MorrisThorneeWormhole(
    b0=1000.0,           # 1 km throat radius
    mass=1e30,           # Solar mass
    shape_function_type="exponential"
)

# Get metric at throat
coordinates = (0.0, 1000.0, np.pi/2, 0.0)
g = wormhole.metric_tensor(coordinates)
```

#### Methods

##### shape_function()

```python
def shape_function(self, r: float) -> float
```

Compute the wormhole shape function b(r).

**Parameters:**
- `r`: Radial coordinate

**Returns:**
- `float`: Shape function value

##### is_traversable()

```python
def is_traversable(self, r_range: Tuple[float, float] = None) -> bool
```

Check if wormhole is traversable.

**Parameters:**
- `r_range`: Radial range to check (default: throat to infinity)

**Returns:**
- `bool`: True if traversable

### ExoticMatter

**Abstract Base Class**: `src.physics.exotic_matter.ExoticMatter`

Base class for exotic matter configurations.

#### Methods

##### energy_density()

```python
def energy_density(self, coordinates: Tuple[float, float, float, float]) -> float
```

Compute energy density at given coordinates.

**Parameters:**
- `coordinates`: (t, r, θ, φ) spacetime coordinates

**Returns:**
- `float`: Energy density in J/m³

##### pressure()

```python
def pressure(self, coordinates: Tuple[float, float, float, float]) -> float
```

Compute pressure at given coordinates.

**Parameters:**
- `coordinates`: (t, r, θ, φ) spacetime coordinates

**Returns:**
- `float`: Pressure in Pa

### CasimirExoticMatter

**Class**: `src.physics.exotic_matter.CasimirExoticMatter`

Casimir effect-based exotic matter.

#### Constructor

```python
def __init__(self, throat_radius: float, casimir_energy_scale: float = -1e15)
```

**Parameters:**
- `throat_radius`: Wormhole throat radius
- `casimir_energy_scale`: Energy density scale factor

**Example:**
```python
casimir_matter = CasimirExoticMatter(
    throat_radius=1000.0,
    casimir_energy_scale=-1e15
)

# Check energy density at throat
coordinates = (0.0, 1000.0, np.pi/2, 0.0)
rho = casimir_matter.energy_density(coordinates)
print(f"Energy density: {rho:.2e} J/m³")
```

### StressEnergyTensor

**Class**: `src.physics.stress_energy_tensor.StressEnergyTensor`

Base class for stress-energy tensor calculations.

#### Methods

##### compute_tensor()

```python
def compute_tensor(self, coordinates: Tuple[float, float, float, float]) -> np.ndarray
```

Compute full stress-energy tensor.

**Parameters:**
- `coordinates`: (t, r, θ, φ) spacetime coordinates

**Returns:**
- `np.ndarray`: 4×4 stress-energy tensor

---

## Quantum System

### WormholeQuantumCircuit

**Class**: `src.quantum.wormhole_circuit.WormholeQuantumCircuit`

Quantum circuit for wormhole traversal simulation.

#### Constructor

```python
def __init__(self, num_qubits: int, traversal_probability: float = 0.8)
```

**Parameters:**
- `num_qubits`: Number of qubits in the circuit
- `traversal_probability`: Success probability for traversal

**Example:**
```python
circuit = WormholeQuantumCircuit(
    num_qubits=8,
    traversal_probability=0.85
)
```

#### Methods

##### create_traversal_state()

```python
def create_traversal_state(self) -> qutip.Qobj
```

Create quantum state for wormhole traversal.

**Returns:**
- `qutip.Qobj`: Quantum state object

##### simulate_teleportation()

```python
def simulate_teleportation(self, input_state: qutip.Qobj) -> Tuple[qutip.Qobj, float]
```

Simulate quantum teleportation through wormhole.

**Parameters:**
- `input_state`: State to teleport

**Returns:**
- `Tuple[qutip.Qobj, float]`: (output_state, fidelity)

**Example:**
```python
import qutip as qt

# Create input state
input_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

# Simulate teleportation
output_state, fidelity = circuit.simulate_teleportation(input_state)
print(f"Teleportation fidelity: {fidelity:.3f}")
```

### EntanglementDynamics

**Class**: `src.quantum.entanglement_dynamics.EntanglementDynamics`

Simulation of quantum entanglement evolution.

#### Constructor

```python
def __init__(self, num_particles: int, dimension: int = 2)
```

**Parameters:**
- `num_particles`: Number of entangled particles
- `dimension`: Hilbert space dimension per particle

**Example:**
```python
entanglement = EntanglementDynamics(
    num_particles=4,
    dimension=2
)
```

#### Methods

##### compute_concurrence()

```python
def compute_concurrence(self, state: qutip.Qobj) -> float
```

Compute concurrence entanglement measure.

**Parameters:**
- `state`: Quantum state

**Returns:**
- `float`: Concurrence value (0 to 1)

##### compute_negativity()

```python
def compute_negativity(self, state: qutip.Qobj) -> float
```

Compute negativity entanglement measure.

**Parameters:**
- `state`: Quantum state

**Returns:**
- `float`: Negativity value

##### evolve_entanglement()

```python
def evolve_entanglement(self, initial_state: qutip.Qobj, 
                       hamiltonian: qutip.Qobj, 
                       time_list: List[float]) -> List[Dict[str, float]]
```

Evolve entanglement over time.

**Parameters:**
- `initial_state`: Initial quantum state
- `hamiltonian`: System Hamiltonian
- `time_list`: Time points for evolution

**Returns:**
- `List[Dict]`: Entanglement measures at each time

**Example:**
```python
import qutip as qt
import numpy as np

# Create Bell state
bell_state = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + 
              qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()

# Random Hamiltonian
H = qt.tensor(qt.rand_herm(2), qt.rand_herm(2))

# Evolution times
times = np.linspace(0, 10, 100)

# Evolve entanglement
evolution = entanglement.evolve_entanglement(bell_state, H, times)

# Plot results
concurrences = [e['concurrence'] for e in evolution]
```

### VacuumFluctuations

**Class**: `src.quantum.vacuum_fluctuations.VacuumFluctuations`

Quantum vacuum fluctuation calculations.

#### Constructor

```python
def __init__(self, field_dimensions: int = 3, cutoff_energy: float = 1e19)
```

**Parameters:**
- `field_dimensions`: Number of spatial dimensions
- `cutoff_energy`: UV cutoff energy

#### Methods

##### compute_vacuum_energy()

```python
def compute_vacuum_energy(self, volume: float = 1.0) -> float
```

Compute vacuum energy in given volume.

**Parameters:**
- `volume`: Volume in m³

**Returns:**
- `float`: Vacuum energy in Joules

---

## AI Components

### StabilityPredictor

**Class**: `src.ai.stability_predictor.StabilityPredictor`

Machine learning model for wormhole stability prediction.

#### Constructor

```python
def __init__(self, model_type: str = "ensemble")
```

**Parameters:**
- `model_type`: Type of ML model ("neural_network", "ensemble", "transformer")

**Example:**
```python
predictor = StabilityPredictor(model_type="ensemble")
```

#### Methods

##### train()

```python
def train(self, training_data: np.ndarray, 
          training_labels: np.ndarray,
          validation_split: float = 0.2) -> Dict[str, float]
```

Train the stability prediction model.

**Parameters:**
- `training_data`: Feature matrix (n_samples × n_features)
- `training_labels`: Stability labels (0 to 1)
- `validation_split`: Fraction of data for validation

**Returns:**
- `Dict`: Training metrics (accuracy, loss, etc.)

##### predict()

```python
def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Predict wormhole stability.

**Parameters:**
- `features`: Feature matrix

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (predictions, confidence_scores)

**Example:**
```python
# Create feature vector
features = np.array([[
    1000.0,  # throat radius
    1e30,    # mass
    0.8,     # traversal probability
    -1e15,   # energy density
    0.5      # entanglement strength
]])

# Predict stability
stability, confidence = predictor.predict(features)
print(f"Stability: {stability[0]:.3f} ± {confidence[0]:.3f}")
```

### ParameterOptimizer

**Class**: `src.ai.parameter_optimizer.ParameterOptimizer`

Multi-objective parameter optimization.

#### Constructor

```python
def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]])
```

**Parameters:**
- `parameter_bounds`: Dictionary of parameter names and (min, max) bounds

**Example:**
```python
optimizer = ParameterOptimizer(
    parameter_bounds={
        'throat_radius': (100.0, 10000.0),
        'mass': (1e29, 1e31),
        'traversal_probability': (0.1, 1.0)
    }
)
```

#### Methods

##### optimize()

```python
def optimize(self, objective_function: Callable,
             objectives: List[str] = None,
             max_iterations: int = 100) -> OptimizationResults
```

Perform multi-objective optimization.

**Parameters:**
- `objective_function`: Function to optimize (takes parameters, returns objectives)
- `objectives`: List of objective names
- `max_iterations`: Maximum optimization iterations

**Returns:**
- `OptimizationResults`: Optimization results and Pareto front

**Example:**
```python
def objective_function(params):
    # Run simulation with parameters
    framework = WormholeSimulationFramework()
    framework.initialize_system(wormhole_params=params)
    results = framework.run_simulation()
    
    # Return objectives to maximize
    return {
        'stability': np.mean(results.stability_predictions),
        'traversability': params.get('traversal_probability', 0.5),
        'efficiency': -abs(params.get('energy_density', -1e15))
    }

# Run optimization
opt_results = optimizer.optimize(
    objective_function=objective_function,
    objectives=['stability', 'traversability', 'efficiency'],
    max_iterations=50
)

print(f"Best parameters: {opt_results.best_parameters}")
```

### AnomalyDetector

**Class**: `src.ai.anomaly_detector.AnomalyDetector`

Unsupervised anomaly detection for simulation monitoring.

#### Constructor

```python
def __init__(self, algorithm: str = "isolation_forest")
```

**Parameters:**
- `algorithm`: Detection algorithm ("isolation_forest", "one_class_svm", "autoencoder")

#### Methods

##### fit()

```python
def fit(self, normal_data: np.ndarray) -> None
```

Train detector on normal simulation data.

**Parameters:**
- `normal_data`: Training data matrix

##### predict_anomaly()

```python
def predict_anomaly(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Detect anomalies in data.

**Parameters:**
- `data`: Data to analyze

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (anomaly_labels, anomaly_scores)

---

## Visualization

### InteractiveDashboard

**Class**: `src.visualization.interactive_dashboard.InteractiveDashboard`

Web-based interactive dashboard for real-time control.

#### Constructor

```python
def __init__(self, config: DashboardConfig = None)
```

**Parameters:**
- `config`: Dashboard configuration object

**Example:**
```python
from src.visualization.interactive_dashboard import InteractiveDashboard, DashboardConfig

dashboard_config = DashboardConfig(
    width=1600,
    height=1000,
    enable_real_time=True
)

dashboard = InteractiveDashboard(dashboard_config)
```

#### Methods

##### create_main_dashboard()

```python
def create_main_dashboard(self) -> go.Figure
```

Create the main dashboard interface.

**Returns:**
- `go.Figure`: Plotly figure with interactive panels

##### launch()

```python
def launch(self, port: int = 8050) -> None
```

Launch dashboard web server.

**Parameters:**
- `port`: Port number for web server

### SpacetimePlotter

**Class**: `src.visualization.spacetime_plotter.SpacetimePlotter`

Advanced 4D spacetime visualization.

#### Constructor

```python
def __init__(self, metric: SpacetimeMetric, 
             config: SpacetimeVisualizationConfig = None)
```

**Parameters:**
- `metric`: Spacetime metric to visualize
- `config`: Visualization configuration

#### Methods

##### plot_wormhole_geometry_3d()

```python
def plot_wormhole_geometry_3d(self, time_slice: float = 0.0,
                             visualization_type: str = 'surface') -> go.Figure
```

Create 3D wormhole geometry visualization.

**Parameters:**
- `time_slice`: Time coordinate for spatial slice
- `visualization_type`: Type ("surface", "wireframe", "contour")

**Returns:**
- `go.Figure`: Interactive 3D plot

##### compute_geodesics()

```python
def compute_geodesics(self, initial_position: np.ndarray,
                     initial_velocity: np.ndarray,
                     geodesic_type: str = 'timelike') -> Tuple[np.ndarray, np.ndarray]
```

Compute geodesic trajectories.

**Parameters:**
- `initial_position`: Starting 4-position
- `initial_velocity`: Starting 4-velocity
- `geodesic_type`: Type ("timelike", "null", "spacelike")

**Returns:**
- `Tuple`: (positions, velocities) along geodesic

**Example:**
```python
# Create Morris-Thorne wormhole
wormhole = MorrisThorneeWormhole(b0=1000.0)
plotter = SpacetimePlotter(wormhole)

# Plot 3D geometry
fig = plotter.plot_wormhole_geometry_3d(visualization_type='surface')
fig.show()

# Compute geodesic
initial_pos = np.array([0.0, 1500.0, np.pi/2, 0.0])
initial_vel = np.array([1.0, 0.1, 0.0, 0.0])
positions, velocities = plotter.compute_geodesics(initial_pos, initial_vel)
```

### QuantumStateAnimator

**Class**: `src.visualization.quantum_state_animator.QuantumStateAnimator`

Quantum state evolution animations.

#### Methods

##### animate_bloch_sphere_evolution()

```python
def animate_bloch_sphere_evolution(self, initial_state: qutip.Qobj = None,
                                 hamiltonian: qutip.Qobj = None) -> go.Figure
```

Animate quantum state on Bloch sphere.

**Parameters:**
- `initial_state`: Initial quantum state
- `hamiltonian`: System Hamiltonian

**Returns:**
- `go.Figure`: Animated Bloch sphere visualization

##### animate_entanglement_dynamics()

```python
def animate_entanglement_dynamics(self, initial_entanglement: float = 0.0,
                                interaction_strength: float = 1.0) -> go.Figure
```

Animate entanglement evolution.

**Parameters:**
- `initial_entanglement`: Initial entanglement measure
- `interaction_strength`: Coupling strength

**Returns:**
- `go.Figure`: Entanglement dynamics animation

---

## Utilities

### Configuration Management

#### load_config()

```python
def load_config(config_file: str) -> Dict[str, Any]
```

Load configuration from YAML/JSON file.

**Parameters:**
- `config_file`: Path to configuration file

**Returns:**
- `Dict`: Configuration dictionary

**Example:**
```python
from src.utils.config import load_config

config = load_config('config/simulation_config.yaml')
framework = WormholeSimulationFramework(IntegrationConfig(**config))
```

#### save_config()

```python
def save_config(config: Dict[str, Any], config_file: str) -> None
```

Save configuration to file.

**Parameters:**
- `config`: Configuration dictionary
- `config_file`: Output file path

### Data Processing

#### export_results()

```python
def export_results(results: SimulationResults, 
                  output_file: str, 
                  format: str = 'json') -> None
```

Export simulation results.

**Parameters:**
- `results`: Simulation results object
- `output_file`: Output file path
- `format`: Export format ("json", "csv", "hdf5")

#### load_results()

```python
def load_results(input_file: str, format: str = 'json') -> SimulationResults
```

Load previously saved results.

**Parameters:**
- `input_file`: Input file path
- `format`: Input format

**Returns:**
- `SimulationResults`: Loaded results object

---

## Examples

### Basic Simulation

```python
from src.integration import WormholeSimulationFramework, IntegrationConfig
import numpy as np

# Create configuration
config = IntegrationConfig(
    simulation_name="basic_example",
    time_steps=500,
    num_qubits=6
)

# Initialize framework
framework = WormholeSimulationFramework(config)

# Set up wormhole parameters
wormhole_params = {
    'b0': 1500.0,        # 1.5 km throat
    'mass': 2e30         # 2 solar masses
}

quantum_params = {
    'num_qubits': 6,
    'traversal_probability': 0.85
}

# Initialize and run
framework.initialize_system(
    wormhole_params=wormhole_params,
    quantum_params=quantum_params
)

results = framework.run_simulation()

# Analyze results
report = framework.generate_comprehensive_report()
print(f"Simulation successful: {report['summary']['simulation_successful']}")
print(f"Average stability: {report['summary']['average_stability']:.3f}")
```

### AI-Enhanced Optimization

```python
from src.ai.parameter_optimizer import ParameterOptimizer
from src.integration import WormholeSimulationFramework

# Define parameter space
bounds = {
    'throat_radius': (500.0, 5000.0),
    'mass': (1e29, 1e31),
    'traversal_probability': (0.5, 1.0)
}

optimizer = ParameterOptimizer(bounds)

# Objective function
def evaluate_wormhole(params):
    framework = WormholeSimulationFramework()
    framework.initialize_system(
        wormhole_params={
            'b0': params['throat_radius'],
            'mass': params['mass']
        },
        quantum_params={
            'traversal_probability': params['traversal_probability']
        }
    )
    
    results = framework.run_simulation(duration=5.0)
    
    return {
        'stability': np.mean(results.stability_predictions),
        'efficiency': -abs(params['mass'] / params['throat_radius'])
    }

# Run optimization
opt_results = optimizer.optimize(
    objective_function=evaluate_wormhole,
    objectives=['stability', 'efficiency'],
    max_iterations=20
)

print(f"Optimal parameters: {opt_results.best_parameters}")
```

### Custom Visualization

```python
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.visualization.spacetime_plotter import SpacetimePlotter
import plotly.graph_objects as go

# Create wormhole
wormhole = MorrisThorneeWormhole(b0=2000.0, mass=1e30)
plotter = SpacetimePlotter(wormhole)

# Create visualization
fig = plotter.plot_wormhole_geometry_3d(visualization_type='surface')

# Customize plot
fig.update_layout(
    title="Custom Wormhole Visualization",
    scene=dict(
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        zaxis_title="Embedding Dimension",
        camera=dict(eye=dict(x=2, y=2, z=2))
    )
)

fig.show()

# Add geodesics
initial_pos = np.array([0.0, 3000.0, np.pi/2, 0.0])
initial_vel = np.array([1.0, -0.1, 0.0, 0.0])
positions, _ = plotter.compute_geodesics(initial_pos, initial_vel)

# Convert to Cartesian for plotting
x_coords = positions[:, 1] * np.cos(positions[:, 3])
y_coords = positions[:, 1] * np.sin(positions[:, 3])
z_coords = np.zeros_like(x_coords)

fig.add_trace(go.Scatter3d(
    x=x_coords, y=y_coords, z=z_coords,
    mode='lines',
    line=dict(color='red', width=5),
    name='Geodesic'
))

fig.show()
```

### Quantum Circuit Analysis

```python
from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.quantum.entanglement_dynamics import EntanglementDynamics
import qutip as qt
import matplotlib.pyplot as plt

# Create quantum system
circuit = WormholeQuantumCircuit(num_qubits=4, traversal_probability=0.9)
entanglement = EntanglementDynamics(num_particles=2, dimension=2)

# Create Bell state
bell = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + 
        qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()

# Test teleportation
original_state = (qt.basis(2,0) + 1j*qt.basis(2,1)).unit()
teleported_state, fidelity = circuit.simulate_teleportation(original_state)

print(f"Teleportation fidelity: {fidelity:.4f}")

# Analyze entanglement evolution
H = qt.tensor(qt.sigmaz(), qt.sigmaz()) + 0.1*qt.tensor(qt.sigmax(), qt.sigmax())
times = np.linspace(0, 10, 100)

evolution = entanglement.evolve_entanglement(bell, H, times)

# Plot results
concurrences = [e['concurrence'] for e in evolution]
plt.figure(figsize=(10, 6))
plt.plot(times, concurrences, 'b-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Concurrence')
plt.title('Entanglement Evolution Through Wormhole')
plt.grid(True)
plt.show()
```

---

## Error Handling

### Common Exceptions

#### ValidationError

Raised when simulation parameters are invalid.

```python
try:
    framework.initialize_system(wormhole_params={'b0': -1000.0})  # Invalid
except ValidationError as e:
    print(f"Parameter validation failed: {e}")
```

#### ConvergenceError

Raised when numerical methods fail to converge.

```python
try:
    results = framework.run_simulation()
except ConvergenceError as e:
    print(f"Simulation convergence failed: {e}")
    # Reduce time step or increase tolerance
```

#### ResourceError

Raised when computational resources are exceeded.

```python
try:
    framework = WormholeSimulationFramework(config)
except ResourceError as e:
    print(f"Insufficient resources: {e}")
    # Reduce problem size or increase memory limit
```

### Best Practices

1. **Always validate parameters** before running simulations
2. **Check convergence** of numerical solutions  
3. **Monitor memory usage** for large simulations
4. **Use appropriate precision** for calculations
5. **Save intermediate results** for long simulations
6. **Handle exceptions gracefully** in production code

---

*This API reference covers the main classes and methods. For implementation details, see the source code comments. For usage examples, see the examples/ directory.*

**Last Updated**: January 2024  
**Framework Version**: 1.0.0