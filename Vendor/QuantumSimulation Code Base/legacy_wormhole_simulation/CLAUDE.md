# CLAUDE.md - Quantum Wormhole Simulation Project

## Project Overview

This is a professional quantum wormhole simulation framework that combines quantum mechanics, classical physics, and AI prediction models to study exotic spacetime phenomena.

## Project Structure

```
QuantumSimulation/
├── src/                    # Main source code
│   ├── quantum/           # Quantum simulation modules
│   ├── physics/           # Physics calculations  
│   ├── ai/                # AI prediction models
│   └── visualization/     # Plotting and graphics
├── data/                  # Simulation results
├── config/                # Configuration files
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── README.md             # Project documentation
└── main.py               # Entry point

```

## Core Components

### 1. Quantum Module (`src/quantum/`)
- **QuantumState**: Manages quantum state vectors and operations
- **WormholeGeometry**: Quantum geometric descriptions of wormhole structures
- **QuantumEvolution**: Time evolution operators and dynamics

### 2. Physics Module (`src/physics/`)
- **SpacetimeGeometry**: Einstein field equations and metric tensors
- **RelativisticPhysics**: General relativity calculations
- **FieldEquations**: Stress-energy tensor computations

### 3. AI Module (`src/ai/`)
- **QuantumPredictor**: Machine learning models for quantum state prediction
- **NeuralNetworks**: Deep learning architectures for physics simulation
- **DataProcessor**: Data preprocessing and feature extraction

### 4. Visualization Module (`src/visualization/`)
- **QuantumVisualizer**: 3D plotting of quantum states and fields
- **SpacetimePlotter**: Visualization of curved spacetime metrics
- **AnimationTools**: Time-evolution animations and interactive plots

## Working with the Codebase

### Getting Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   # or for development:
   pip install -e .
   ```

2. **Run Basic Simulation**:
   ```bash
   python main.py
   ```

3. **Run with Custom Parameters**:
   ```bash
   python main.py --dimensions 12 --time-steps 2000 --verbose
   ```

### Development Workflow

1. **Adding New Features**:
   - Create modules in appropriate `src/` subdirectories
   - Follow existing naming conventions
   - Import dependencies from requirements.txt only
   - Add corresponding tests in `tests/` directory

2. **Configuration Management**:
   - Store config files in `config/` directory
   - Use JSON format for configuration files
   - Override configs with command-line arguments

3. **Data Management**:
   - All simulation results go in `data/` directory
   - Use NumPy arrays for numerical data
   - Include metadata files with simulation parameters

### Code Conventions

- **Naming**: Use descriptive variable names and follow PEP 8
- **Documentation**: Include docstrings for all classes and functions
- **Type Hints**: Use type annotations for function parameters and returns
- **Error Handling**: Use try/except blocks with appropriate logging
- **Imports**: Group imports by standard library, third-party, and local modules

### Testing

- Run tests with: `pytest tests/`
- Write unit tests for all core functions
- Use mock objects for expensive computations
- Include integration tests for complete workflows

### Dependencies

The project uses these key libraries:
- **NumPy/SciPy**: Numerical computations
- **Matplotlib/Plotly**: Visualization
- **TensorFlow**: AI/ML models
- **QuTiP**: Quantum mechanics toolkit
- **SymPy**: Symbolic mathematics

### Common Tasks

1. **Add New Quantum State**:
   - Extend `QuantumState` class in `src/quantum/quantum_state.py`
   - Add corresponding visualization methods
   - Include unit tests

2. **Implement New Physics Model**:
   - Create module in `src/physics/`
   - Integrate with `SpacetimeGeometry` class
   - Update main simulation loop

3. **Add ML Model**:
   - Extend `QuantumPredictor` in `src/ai/predictor.py`
   - Train on existing simulation data
   - Validate predictions against known results

4. **Create Visualizations**:
   - Add methods to `QuantumVisualizer`
   - Support both static and animated plots
   - Include interactive features where appropriate

## Architecture Notes

- **Modular Design**: Each component is self-contained with clear interfaces
- **Data Flow**: main.py → quantum simulation → physics calculations → AI prediction → visualization
- **Extensibility**: Easy to add new physics models, quantum states, or visualization types
- **Performance**: Optimized for large-scale numerical computations

## Tips for Claude Code Users

- Use the project structure as a guide for organizing code
- Leverage existing modules rather than creating new files
- Check `requirements.txt` before adding new dependencies  
- Run simulations with small parameters first to test changes
- Use the visualization tools to understand quantum states and physics

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed and paths are correct
- **Memory Issues**: Reduce dimensions or time steps for large simulations
- **Convergence Problems**: Check initial conditions and simulation parameters
- **Visualization Errors**: Verify backend compatibility and data formats