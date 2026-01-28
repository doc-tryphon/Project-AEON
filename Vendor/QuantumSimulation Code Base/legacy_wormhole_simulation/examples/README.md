# Quantum Wormhole Simulation Examples

This directory contains comprehensive examples demonstrating different aspects and capabilities of the Quantum Wormhole Simulation Framework. Each example is self-contained and focuses on specific features or use cases.

## 📁 Available Examples

### 01_basic_wormhole.py
**Basic Wormhole Simulation Walkthrough**

A comprehensive introduction to the framework demonstrating:
- Framework initialization and configuration
- Morris-Thorne wormhole setup
- Basic simulation execution
- Results analysis and visualization
- Stability monitoring
- Performance metrics

**Usage:**
```bash
python examples/01_basic_wormhole.py
```

**Output:**
- Stability evolution plots
- Physics metrics visualization
- Quantum entanglement measures
- Summary dashboard
- Console progress tracking

**Suitable for:** Beginners, first-time users, basic analysis

---

### 02_ai_optimization.py
**AI-Driven Parameter Optimization**

Advanced parameter optimization using machine learning:
- Multi-objective optimization (stability, traversability, energy efficiency)
- Genetic algorithm implementation
- Pareto front analysis
- Parameter sensitivity studies
- Anomaly detection
- Statistical validation

**Usage:**
```bash
python examples/02_ai_optimization.py
```

**Output:**
- Optimization convergence plots
- Pareto front visualizations
- Parameter sensitivity analysis
- Anomaly detection results
- Best configuration recommendations

**Suitable for:** Researchers, parameter tuning, optimization studies

---

### 03_interactive_visualization.py
**Interactive Real-Time Visualization**

Web-based interactive dashboard with:
- Real-time parameter adjustment
- Live simulation monitoring
- Interactive 3D spacetime visualization
- Multi-plot dashboard
- Parameter sliders and controls
- Start/stop/reset functionality

**Usage:**
```bash
python examples/03_interactive_visualization.py
```

**Access:** Open browser to `http://localhost:8050`

**Features:**
- Physics parameter sliders (throat radius, mass, Casimir energy)
- Quantum parameter controls (qubits, traversal probability, decoherence)
- Real-time plots (stability, quantum metrics, physics properties)
- 3D wormhole geometry visualization
- Live data streaming

**Suitable for:** Educational demonstrations, interactive exploration, presentations

---

### 04_quantum_field_analysis.py
**Quantum Field Theory Analysis**

Advanced quantum field analysis in curved spacetime:
- Vacuum fluctuation calculations
- Field propagation through wormholes
- Hawking radiation analysis
- Unruh effect demonstration
- Quantum field correlations
- Thermal spectrum analysis

**Usage:**
```bash
python examples/04_quantum_field_analysis.py
```

**Output:**
- Vacuum energy density plots
- Field propagation animations
- Hawking radiation spectrum
- Unruh temperature calculations
- Correlation function analysis
- Comprehensive field theory report

**Suitable for:** Advanced physics research, quantum field theory studies

---

### 05_stability_benchmarking.py
**Comprehensive Stability Benchmarking**

Systematic stability analysis and performance benchmarking:
- Parameter space exploration
- Stress testing under extreme conditions
- Performance optimization
- Statistical significance testing
- Robustness evaluation
- Correlation analysis

**Usage:**
```bash
python examples/05_stability_benchmarking.py
```

**Output:**
- Parameter sweep studies
- Stress test analysis
- Performance benchmarks
- Statistical correlations
- Optimization recommendations
- Comprehensive benchmark report

**Suitable for:** System validation, performance tuning, robustness analysis

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify Installation:**
```bash
python main.py --validate
```

### Running Examples

#### Simple Execution
```bash
# Run any example directly
python examples/01_basic_wormhole.py
```

#### With Modifications
```bash
# Modify parameters via environment variables or config files
python examples/01_basic_wormhole.py --verbose
```

#### Interactive Mode
```bash
# For interactive examples
python examples/03_interactive_visualization.py
# Then open http://localhost:8050 in browser
```

## 📊 Output Structure

All examples generate organized output in the `examples/output/` directory:

```
examples/output/
├── stability_evolution.png          # Stability over time
├── physics_metrics.png             # Energy density, pressure
├── quantum_metrics.png             # Entanglement, coherence
├── summary_dashboard.png           # Overview dashboard
├── optimization_results.png        # AI optimization plots
├── pareto_front.png                # Multi-objective results
├── parameter_sensitivity.png       # Sensitivity analysis
├── interactive_dashboard.html      # Interactive plots
├── vacuum_fluctuations.png         # Quantum field analysis
├── field_propagation.png           # Field evolution
├── hawking_spectrum.png            # Thermal radiation
├── correlation_functions.png       # Quantum correlations
├── parameter_sweeps.png            # Benchmark sweeps
├── stress_test_analysis.png        # Robustness testing
├── performance_benchmark.png       # Timing analysis
└── benchmark_results.json          # Raw numerical data
```

## 🛠️ Customization Guide

### Modifying Parameters

1. **Direct Parameter Changes:**
```python
# Edit parameters in example files
wormhole_params = {
    'b0': 2000.0,           # Throat radius (m)
    'mass': 2e30,           # Mass (kg)
    'casimir_energy': -2e15 # Casimir energy (J)
}
```

2. **Configuration Files:**
```python
# Use custom configuration files
config = IntegrationConfig.from_file('my_custom_config.yaml')
```

3. **Command Line Overrides:**
```python
# Add argparse support to examples
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--throat-radius', type=float, default=1000.0)
```

### Adding New Visualizations

```python
def create_custom_plot(results):
    """Add custom visualization."""
    plt.figure(figsize=(10, 6))
    # Custom plotting code
    plt.savefig('examples/output/custom_plot.png')
```

### Extending Examples

```python
class CustomAnalysis(ExistingClass):
    """Extend existing analysis classes."""
    
    def custom_analysis_method(self):
        """Add new analysis capabilities."""
        # Custom analysis code
        pass
```

## 🔧 Troubleshooting

### Common Issues

**1. Memory Errors**
```python
# Reduce simulation complexity
config = IntegrationConfig(
    time_steps=50,      # Reduce from 100+
    num_qubits=4,       # Reduce from 8+
    spatial_resolution=25  # Reduce grid size
)
```

**2. Slow Performance**
```python
# Enable parallel processing
config.enable_parallel = True
config.num_workers = 4
```

**3. Visualization Errors**
```python
# Check matplotlib backend
import matplotlib
matplotlib.use('Agg')  # For headless systems
```

**4. Import Errors**
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation

Test examples before running:
```bash
python -m py_compile examples/01_basic_wormhole.py
```

## 📚 Educational Progression

**Recommended Learning Path:**

1. **Start Here:** `01_basic_wormhole.py`
   - Learn framework basics
   - Understand core concepts
   - Review visualization outputs

2. **Next:** `03_interactive_visualization.py`
   - Explore parameter effects interactively
   - Understand real-time dynamics
   - Experiment with different settings

3. **Advanced:** `02_ai_optimization.py`
   - Learn optimization techniques
   - Understand multi-objective analysis
   - Study parameter relationships

4. **Research Level:** `04_quantum_field_analysis.py`
   - Dive into quantum field theory
   - Understand advanced physics
   - Explore field correlations

5. **System Analysis:** `05_stability_benchmarking.py`
   - Comprehensive system validation
   - Performance characterization
   - Statistical analysis methods

## 🎯 Use Case Scenarios

### Educational Demonstrations
- Use `01_basic_wormhole.py` for introductory lessons
- Use `03_interactive_visualization.py` for classroom demonstrations
- Focus on stability plots and physics interpretation

### Research Applications
- Use `02_ai_optimization.py` for parameter studies
- Use `04_quantum_field_analysis.py` for theoretical investigations
- Use `05_stability_benchmarking.py` for validation studies

### Development and Testing
- Use `05_stability_benchmarking.py` for system validation
- Modify examples to test new features
- Create custom examples for specific research needs

## 📝 Citation and Attribution

If you use these examples in research or educational materials, please cite:

```bibtex
@software{quantum_wormhole_simulation,
  title={Quantum Wormhole Simulation Framework},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]}
}
```

## 🤝 Contributing

To contribute new examples:

1. Follow the existing example structure
2. Include comprehensive documentation
3. Add appropriate visualizations
4. Test thoroughly across different parameter ranges
5. Submit pull request with detailed description

---

## 📞 Support

For questions or issues with examples:
- Check the main documentation in `docs/`
- Review the troubleshooting section above
- Open an issue on the project repository
- Consult the API reference documentation

---

*Last updated: January 2024*  
*Framework Version: 1.0.0*