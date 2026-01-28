#!/usr/bin/env python3
"""
Quantum Wormhole Simulation - Main Entry Point

Comprehensive command-line interface for quantum wormhole simulation with
multiple modes: basic analysis, AI-optimized parameters, and real-time visualization.
"""

import argparse
import sys
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the integrated framework
from src.integration import (
    WormholeSimulationFramework,
    IntegrationConfig,
    SimulationResults,
    create_default_simulation,
    run_quick_demo
)

# Import specific components for advanced modes
from src.physics.constants import PhysicsConstants, NaturalUnits
from src.visualization.interactive_dashboard import launch_dashboard, DashboardConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulationModes:
    """Available simulation modes."""
    BASIC = "basic"
    AI_OPTIMIZED = "ai-optimized" 
    INTERACTIVE = "interactive"
    VISUALIZATION = "visualization"
    BENCHMARK = "benchmark"
    DEMO = "demo"
    CUSTOM = "custom"


class CommandLineInterface:
    """Comprehensive command-line interface for wormhole simulation."""
    
    def __init__(self):
        """Initialize the command-line interface."""
        self.parser = self._create_parser()
        self.framework = None
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        
        parser = argparse.ArgumentParser(
            description='Quantum Wormhole Simulation Framework',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                              # Run quick demo
  %(prog)s --mode basic                 # Basic wormhole analysis
  %(prog)s --mode ai-optimized --steps 2000 # AI-optimized simulation
  %(prog)s --mode interactive           # Interactive dashboard
  %(prog)s --mode visualization --real-time # Real-time visualization
  %(prog)s --mode benchmark             # Performance benchmark
  %(prog)s --config my_config.json     # Custom configuration
  %(prog)s --throat-radius 5000 --mass 2e30 # Custom parameters
            """
        )
        
        # Simulation mode
        parser.add_argument(
            '--mode', '-m',
            choices=[SimulationModes.BASIC, SimulationModes.AI_OPTIMIZED, 
                    SimulationModes.INTERACTIVE, SimulationModes.VISUALIZATION,
                    SimulationModes.BENCHMARK, SimulationModes.DEMO, SimulationModes.CUSTOM],
            default=SimulationModes.DEMO,
            help='Simulation mode to run (default: demo)'
        )
        
        # Configuration files
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to JSON configuration file'
        )
        
        parser.add_argument(
            '--save-config',
            type=str,
            help='Save current configuration to file'
        )
        
        # Basic simulation parameters
        sim_group = parser.add_argument_group('Simulation Parameters')
        
        sim_group.add_argument(
            '--name',
            type=str,
            default=f"wormhole_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help='Simulation name (default: timestamped)'
        )
        
        sim_group.add_argument(
            '--steps',
            type=int,
            default=1000,
            help='Number of simulation time steps (default: 1000)'
        )
        
        sim_group.add_argument(
            '--dt',
            type=float,
            default=0.1,
            help='Time step size (default: 0.1)'
        )
        
        sim_group.add_argument(
            '--duration',
            type=float,
            help='Total simulation duration (overrides --steps if provided)'
        )
        
        # Physics parameters
        physics_group = parser.add_argument_group('Physics Parameters')
        
        physics_group.add_argument(
            '--throat-radius',
            type=float,
            default=1e3,
            help='Wormhole throat radius in meters (default: 1e3)'
        )
        
        physics_group.add_argument(
            '--mass',
            type=float,
            default=1e30,
            help='Wormhole mass in kg (default: 1e30)'
        )
        
        physics_group.add_argument(
            '--exotic-matter-density',
            type=float,
            default=-1e15,
            help='Exotic matter energy density (default: -1e15)'
        )
        
        physics_group.add_argument(
            '--casimir-energy',
            type=float,
            default=-1e15,
            help='Casimir energy scale (default: -1e15)'
        )
        
        # Quantum parameters
        quantum_group = parser.add_argument_group('Quantum Parameters')
        
        quantum_group.add_argument(
            '--qubits',
            type=int,
            default=8,
            help='Number of qubits in quantum system (default: 8)'
        )
        
        quantum_group.add_argument(
            '--traversal-probability',
            type=float,
            default=0.8,
            help='Wormhole traversal probability (default: 0.8)'
        )
        
        quantum_group.add_argument(
            '--entanglement-strength',
            type=float,
            default=1.0,
            help='Initial entanglement strength (default: 1.0)'
        )
        
        quantum_group.add_argument(
            '--decoherence-rate',
            type=float,
            default=0.01,
            help='Quantum decoherence rate (default: 0.01)'
        )
        
        quantum_group.add_argument(
            '--coherence-time',
            type=float,
            default=100.0,
            help='Quantum coherence time (default: 100.0)'
        )
        
        # AI parameters
        ai_group = parser.add_argument_group('AI Parameters')
        
        ai_group.add_argument(
            '--stability-threshold',
            type=float,
            default=0.5,
            help='Stability prediction threshold (default: 0.5)'
        )
        
        ai_group.add_argument(
            '--optimization-target',
            choices=['stability', 'traversability', 'energy', 'entanglement'],
            default='stability',
            help='AI optimization target (default: stability)'
        )
        
        ai_group.add_argument(
            '--enable-ml',
            action='store_true',
            help='Enable machine learning components'
        )
        
        ai_group.add_argument(
            '--enable-optimization',
            action='store_true',
            help='Enable parameter optimization'
        )
        
        ai_group.add_argument(
            '--enable-anomaly-detection',
            action='store_true',
            help='Enable anomaly detection'
        )
        
        # Visualization parameters
        vis_group = parser.add_argument_group('Visualization Parameters')
        
        vis_group.add_argument(
            '--real-time',
            action='store_true',
            help='Enable real-time visualization'
        )
        
        vis_group.add_argument(
            '--dashboard',
            action='store_true',
            help='Launch interactive dashboard'
        )
        
        vis_group.add_argument(
            '--save-plots',
            action='store_true',
            help='Save visualization plots to files'
        )
        
        vis_group.add_argument(
            '--plot-format',
            choices=['png', 'svg', 'pdf', 'html'],
            default='html',
            help='Plot output format (default: html)'
        )
        
        vis_group.add_argument(
            '--animation-frames',
            type=int,
            default=100,
            help='Number of animation frames (default: 100)'
        )
        
        # Performance parameters
        perf_group = parser.add_argument_group('Performance Parameters')
        
        perf_group.add_argument(
            '--parallel',
            action='store_true',
            help='Enable parallel processing'
        )
        
        perf_group.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of worker processes (default: 4)'
        )
        
        perf_group.add_argument(
            '--memory-limit',
            type=float,
            default=8.0,
            help='Memory limit in GB (default: 8.0)'
        )
        
        perf_group.add_argument(
            '--profile',
            action='store_true',
            help='Enable performance profiling'
        )
        
        # Output parameters
        output_group = parser.add_argument_group('Output Parameters')
        
        output_group.add_argument(
            '--output-dir',
            type=str,
            default='simulation_results',
            help='Output directory (default: simulation_results)'
        )
        
        output_group.add_argument(
            '--save-results',
            action='store_true',
            default=True,
            help='Save simulation results (default: True)'
        )
        
        output_group.add_argument(
            '--results-format',
            choices=['json', 'pickle', 'hdf5'],
            default='json',
            help='Results output format (default: json)'
        )
        
        output_group.add_argument(
            '--generate-report',
            action='store_true',
            help='Generate comprehensive analysis report'
        )
        
        # Debug and utility options
        debug_group = parser.add_argument_group('Debug and Utility Options')
        
        debug_group.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='Increase verbosity (use -v, -vv, or -vvv)'
        )
        
        debug_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress all output except errors'
        )
        
        debug_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show configuration without running simulation'
        )
        
        debug_group.add_argument(
            '--validate',
            action='store_true',
            help='Validate configuration and parameters'
        )
        
        debug_group.add_argument(
            '--list-modes',
            action='store_true',
            help='List available simulation modes'
        )
        
        debug_group.add_argument(
            '--version',
            action='store_true',
            help='Show version information'
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)
    
    def setup_logging(self, args: argparse.Namespace):
        """Setup logging based on command-line arguments."""
        
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose == 0:
            logging.getLogger().setLevel(logging.INFO)
        elif args.verbose == 1:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
            # Enable debug for specific modules
            logging.getLogger('src.integration').setLevel(logging.DEBUG)
            logging.getLogger('src.physics').setLevel(logging.DEBUG)
            logging.getLogger('src.quantum').setLevel(logging.DEBUG)
    
    def load_config(self, args: argparse.Namespace) -> IntegrationConfig:
        """Load configuration from file and command-line arguments."""
        
        # Start with default config
        config_dict = {}
        
        # Load from file if specified
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    file_config = json.load(f)
                config_dict.update(file_config)
                logger.info(f"Loaded configuration from {args.config}")
            except FileNotFoundError:
                logger.warning(f"Configuration file {args.config} not found")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file: {e}")
                sys.exit(1)
        
        # Override with command-line arguments
        config_dict.update({
            'simulation_name': args.name,
            'time_steps': args.steps,
            'dt': args.dt,
            'num_qubits': args.qubits,
            'enable_real_time_visualization': args.real_time,
            'parallel_processing': args.parallel,
            'max_workers': args.workers,
            'memory_limit_gb': args.memory_limit,
            'output_directory': args.output_dir,
            'save_intermediate_results': args.save_results,
            
            # AI configuration
            'enable_stability_prediction': args.enable_ml or args.mode == SimulationModes.AI_OPTIMIZED,
            'enable_parameter_optimization': args.enable_optimization or args.mode == SimulationModes.AI_OPTIMIZED,
            'enable_anomaly_detection': args.enable_anomaly_detection or args.mode == SimulationModes.AI_OPTIMIZED,
            
            # Visualization configuration
            'enable_real_time_visualization': args.real_time or args.mode == SimulationModes.VISUALIZATION,
            'save_visualization_frames': args.save_plots,
            'visualization_update_interval': 10 if args.real_time else 50
        })
        
        # Create configuration object
        config = IntegrationConfig(**config_dict)
        
        return config
    
    def create_simulation_parameters(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        """Create simulation parameters from command-line arguments."""
        
        wormhole_params = {
            'throat_radius': args.throat_radius,
            'mass': args.mass,
            'casimir_energy': args.casimir_energy
        }
        
        quantum_params = {
            'num_qubits': args.qubits,
            'traversal_probability': args.traversal_probability,
            'entanglement_strength': args.entanglement_strength,
            'decoherence_rate': args.decoherence_rate,
            'coherence_time': args.coherence_time
        }
        
        ai_params = {
            'stability_threshold': args.stability_threshold,
            'optimization_target': args.optimization_target
        }
        
        visualization_params = {
            'enable_real_time': args.real_time,
            'save_frames': args.save_plots,
            'animation_frames': args.animation_frames,
            'output_format': args.plot_format
        }
        
        return {
            'wormhole_params': wormhole_params,
            'quantum_params': quantum_params,
            'ai_params': ai_params,
            'visualization_params': visualization_params
        }
    
    def save_config(self, config: IntegrationConfig, filename: str):
        """Save configuration to file."""
        
        config_dict = config.__dict__.copy()
        
        # Convert datetime objects to strings
        for key, value in config_dict.items():
            if isinstance(value, datetime):
                config_dict[key] = value.isoformat()
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {filename}")
    
    def validate_parameters(self, args: argparse.Namespace) -> bool:
        """Validate simulation parameters."""
        
        errors = []
        
        # Physics validation
        if args.throat_radius <= 0:
            errors.append("Throat radius must be positive")
        
        if args.mass < 0:
            errors.append("Mass cannot be negative")
        
        if args.exotic_matter_density >= 0:
            logger.warning("Positive exotic matter density - wormhole may not be traversable")
        
        # Quantum validation
        if args.qubits < 1:
            errors.append("Number of qubits must be at least 1")
        
        if not 0 <= args.traversal_probability <= 1:
            errors.append("Traversal probability must be between 0 and 1")
        
        if args.decoherence_rate < 0:
            errors.append("Decoherence rate cannot be negative")
        
        # Simulation validation
        if args.steps <= 0:
            errors.append("Number of steps must be positive")
        
        if args.dt <= 0:
            errors.append("Time step must be positive")
        
        # Performance validation
        if args.workers < 1:
            errors.append("Number of workers must be at least 1")
        
        if args.memory_limit <= 0:
            errors.append("Memory limit must be positive")
        
        # Print errors
        if errors:
            logger.error("Parameter validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Parameter validation passed")
        return True
    
    def print_mode_information(self):
        """Print information about available simulation modes."""
        
        modes = {
            SimulationModes.DEMO: "Quick demonstration simulation (default)",
            SimulationModes.BASIC: "Basic wormhole analysis with standard physics",
            SimulationModes.AI_OPTIMIZED: "AI-driven parameter optimization and analysis",
            SimulationModes.INTERACTIVE: "Interactive dashboard with real-time controls",
            SimulationModes.VISUALIZATION: "Focus on advanced visualization and animation",
            SimulationModes.BENCHMARK: "Performance benchmarking and testing",
            SimulationModes.CUSTOM: "Custom simulation with user-defined parameters"
        }
        
        print("Available simulation modes:")
        for mode, description in modes.items():
            print(f"  {mode:15} - {description}")
    
    def print_version(self):
        """Print version information."""
        
        print("Quantum Wormhole Simulation Framework")
        print("Version: 1.0.0")
        print("Author: Claude Code AI Assistant")
        print("Built with: NumPy, SciPy, TensorFlow, QuTiP, Plotly")
        print()
        print("Components:")
        print("  - Physics Engine: Einstein field equations, exotic matter")
        print("  - Quantum System: Wormhole circuits, entanglement dynamics")
        print("  - AI Analysis: Stability prediction, parameter optimization")
        print("  - Visualization: Interactive 4D spacetime rendering")
    
    def run_simulation_mode(self, mode: str, args: argparse.Namespace) -> Optional[SimulationResults]:
        """Run simulation based on selected mode."""
        
        if mode == SimulationModes.DEMO:
            return self.run_demo_mode(args)
        elif mode == SimulationModes.BASIC:
            return self.run_basic_mode(args)
        elif mode == SimulationModes.AI_OPTIMIZED:
            return self.run_ai_optimized_mode(args)
        elif mode == SimulationModes.INTERACTIVE:
            return self.run_interactive_mode(args)
        elif mode == SimulationModes.VISUALIZATION:
            return self.run_visualization_mode(args)
        elif mode == SimulationModes.BENCHMARK:
            return self.run_benchmark_mode(args)
        elif mode == SimulationModes.CUSTOM:
            return self.run_custom_mode(args)
        else:
            logger.error(f"Unknown simulation mode: {mode}")
            return None
    
    def run_demo_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run quick demonstration simulation."""
        
        print(" Running Quantum Wormhole Simulation Demo")
        print("=" * 50)
        
        logger.info("Starting demo simulation...")
        
        # Run the built-in demo
        results = run_quick_demo()
        
        print("\n Demo completed successfully!")
        print(f"Simulation steps: {len(results.spacetime_evolution)}")
        
        if results.stability_predictions:
            avg_stability = np.mean(results.stability_predictions)
            print(f"Average stability: {avg_stability:.3f}")
        
        return results
    
    def run_basic_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run basic wormhole analysis."""
        
        print(" Running Basic Wormhole Analysis")
        print("=" * 50)
        
        # Load configuration
        config = self.load_config(args)
        config.enable_real_time_visualization = False  # Keep it simple for basic mode
        
        # Create framework
        self.framework = WormholeSimulationFramework(config)
        
        # Create parameters
        params = self.create_simulation_parameters(args)
        
        # Initialize system
        logger.info("Initializing wormhole simulation system...")
        self.framework.initialize_system(**params)
        
        # Run simulation
        logger.info("Running basic wormhole analysis...")
        results = self.framework.run_simulation()
        
        # Generate basic report
        report = self.framework.generate_comprehensive_report()
        
        # Display results
        self.display_basic_results(results, report)
        
        return results
    
    def run_ai_optimized_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run AI-optimized parameter simulation."""
        
        print(" Running AI-Optimized Wormhole Simulation")
        print("=" * 50)
        
        # Load configuration with AI enabled
        config = self.load_config(args)
        config.enable_stability_prediction = True
        config.enable_parameter_optimization = True
        config.enable_anomaly_detection = True
        
        # Create framework
        self.framework = WormholeSimulationFramework(config)
        
        # Create parameters
        params = self.create_simulation_parameters(args)
        
        # Initialize system
        logger.info("Initializing AI-enhanced wormhole system...")
        self.framework.initialize_system(**params)
        
        # Run optimization loop
        logger.info("Running AI parameter optimization...")
        
        best_results = None
        best_stability = 0.0
        
        optimization_rounds = 3  # Quick optimization for demo
        
        for round_num in range(optimization_rounds):
            print(f"\n🔄 Optimization Round {round_num + 1}/{optimization_rounds}")
            
            # Run simulation
            results = self.framework.run_simulation()
            
            # Evaluate performance
            if results.stability_predictions:
                avg_stability = np.mean(results.stability_predictions)
                print(f"Average stability: {avg_stability:.3f}")
                
                if avg_stability > best_stability:
                    best_stability = avg_stability
                    best_results = results
                    print(" New best configuration found!")
            
            # Adjust parameters for next round (simplified optimization)
            if round_num < optimization_rounds - 1:
                self._adjust_parameters_for_optimization(params)
                self.framework.initialize_system(**params)
        
        # Display optimization results
        print(f"\n Optimization Complete!")
        print(f"Best stability achieved: {best_stability:.3f}")
        
        return best_results or results
    
    def run_interactive_mode(self, args: argparse.Namespace) -> Optional[SimulationResults]:
        """Run interactive dashboard mode."""
        
        print("  Launching Interactive Wormhole Dashboard")
        print("=" * 50)
        
        # Create dashboard configuration
        dashboard_config = DashboardConfig(
            enable_real_time=True,
            auto_refresh=True,
            width=1600,
            height=1000
        )
        
        # Launch dashboard
        logger.info("Starting interactive dashboard...")
        try:
            launch_dashboard(dashboard_config)
            print("🚀 Dashboard launched successfully!")
            print("Check your web browser for the interactive interface.")
            
            # Keep the process running
            input("\nPress Enter to exit...")
            
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
            return None
        
        return None  # Interactive mode doesn't return traditional results
    
    def run_visualization_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run visualization-focused simulation."""
        
        print(" Running Advanced Visualization Simulation")
        print("=" * 50)
        
        # Load configuration with visualization enabled
        config = self.load_config(args)
        config.enable_real_time_visualization = True
        config.save_visualization_frames = args.save_plots
        
        # Create framework
        self.framework = WormholeSimulationFramework(config)
        
        # Create parameters
        params = self.create_simulation_parameters(args)
        
        # Initialize system
        logger.info("Initializing visualization-enhanced system...")
        self.framework.initialize_system(**params)
        
        # Run simulation with enhanced visualization
        logger.info("Running simulation with real-time visualization...")
        
        def visualization_callback(step, step_results):
            """Callback for visualization updates."""
            if step % 10 == 0:
                print(f"Step {step}: Generating visualizations...")
        
        results = self.framework.run_simulation(callback=visualization_callback)
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations(results)
        
        print("🎬 Visualization suite created!")
        if args.save_plots:
            print(f"Plots saved to {args.output_dir}")
        
        return results
    
    def run_benchmark_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run performance benchmark simulation."""
        
        print(" Running Performance Benchmark")
        print("=" * 50)
        
        benchmark_configs = [
            {'steps': 100, 'qubits': 4, 'name': 'Small'},
            {'steps': 500, 'qubits': 6, 'name': 'Medium'},
            {'steps': 1000, 'qubits': 8, 'name': 'Large'}
        ]
        
        benchmark_results = []
        
        for bench_config in benchmark_configs:
            print(f"\n🏃 Running {bench_config['name']} benchmark...")
            
            # Create config for this benchmark
            config = self.load_config(args)
            config.time_steps = bench_config['steps']
            config.num_qubits = bench_config['qubits']
            config.enable_real_time_visualization = False  # Disable for speed
            
            # Create framework
            framework = WormholeSimulationFramework(config)
            
            # Create parameters
            params = self.create_simulation_parameters(args)
            params['quantum_params']['num_qubits'] = bench_config['qubits']
            
            # Time the execution
            start_time = time.time()
            
            framework.initialize_system(**params)
            results = framework.run_simulation()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store benchmark results
            benchmark_result = {
                'name': bench_config['name'],
                'steps': bench_config['steps'],
                'qubits': bench_config['qubits'],
                'execution_time': execution_time,
                'steps_per_second': bench_config['steps'] / execution_time,
                'memory_usage': len(results.spacetime_evolution) * 0.001  # Rough estimate
            }
            
            benchmark_results.append(benchmark_result)
            
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Steps per second: {benchmark_result['steps_per_second']:.1f}")
        
        # Display benchmark summary
        self.display_benchmark_results(benchmark_results)
        
        return results  # Return the last results
    
    def run_custom_mode(self, args: argparse.Namespace) -> SimulationResults:
        """Run custom simulation with user-defined parameters."""
        
        print("  Running Custom Wormhole Simulation")
        print("=" * 50)
        
        # Load configuration
        config = self.load_config(args)
        
        # Create framework
        self.framework = WormholeSimulationFramework(config)
        
        # Create parameters
        params = self.create_simulation_parameters(args)
        
        # Display configuration
        self.display_configuration(config, params)
        
        # Initialize system
        logger.info("Initializing custom simulation system...")
        self.framework.initialize_system(**params)
        
        # Run simulation
        logger.info("Running custom simulation...")
        results = self.framework.run_simulation()
        
        # Generate and display report
        report = self.framework.generate_comprehensive_report()
        self.display_comprehensive_report(report)
        
        return results
    
    def _adjust_parameters_for_optimization(self, params: Dict[str, Dict[str, Any]]):
        """Adjust parameters for optimization (simplified)."""
        
        # Simple parameter adjustment strategy
        wormhole_params = params['wormhole_params']
        quantum_params = params['quantum_params']
        
        # Slightly adjust throat radius
        wormhole_params['b0'] *= (1 + 0.1 * (np.random.random() - 0.5))
        
        # Adjust traversal probability
        quantum_params['traversal_probability'] *= (1 + 0.05 * (np.random.random() - 0.5))
        quantum_params['traversal_probability'] = np.clip(quantum_params['traversal_probability'], 0.1, 1.0)
    
    def display_basic_results(self, results: SimulationResults, report: Dict[str, Any]):
        """Display basic simulation results."""
        
        print(f"\n Basic Analysis Results:")
        print(f"  Total simulation steps: {len(results.spacetime_evolution)}")
        
        if results.stability_predictions:
            avg_stability = np.mean(results.stability_predictions)
            print(f"  Average stability: {avg_stability:.3f}")
            print(f"  Stability range: {np.min(results.stability_predictions):.3f} - {np.max(results.stability_predictions):.3f}")
        
        if 'summary' in report:
            summary = report['summary']
            print(f"  Simulation successful: {summary.get('simulation_successful', 'Unknown')}")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
        
        if 'physics_analysis' in report and 'energy_statistics' in report['physics_analysis']:
            energy_stats = report['physics_analysis']['energy_statistics']
            print(f"  Mean energy density: {energy_stats.get('mean', 0):.2e}")
        
        if 'recommendations' in report:
            print(f"\n Recommendations:")
            for i, rec in enumerate(report['recommendations'][:3], 1):  # Show first 3
                print(f"  {i}. {rec}")
    
    def display_benchmark_results(self, benchmark_results: List[Dict[str, Any]]):
        """Display benchmark results."""
        
        print(f"\n Benchmark Results Summary:")
        print(f"{'Config':<10} {'Steps':<8} {'Qubits':<8} {'Time(s)':<10} {'Steps/s':<10} {'Memory(MB)':<12}")
        print("-" * 68)
        
        for result in benchmark_results:
            print(f"{result['name']:<10} {result['steps']:<8} {result['qubits']:<8} "
                  f"{result['execution_time']:<10.2f} {result['steps_per_second']:<10.1f} "
                  f"{result['memory_usage']:<12.1f}")
        
        print(f"\n Performance Analysis:")
        
        if len(benchmark_results) >= 2:
            small_sps = benchmark_results[0]['steps_per_second']
            large_sps = benchmark_results[-1]['steps_per_second']
            efficiency = large_sps / small_sps if small_sps > 0 else 0
            print(f"  Scaling efficiency: {efficiency:.2f}")
        
        fastest = max(benchmark_results, key=lambda x: x['steps_per_second'])
        print(f"  Fastest configuration: {fastest['name']} ({fastest['steps_per_second']:.1f} steps/s)")
    
    def display_configuration(self, config: IntegrationConfig, params: Dict[str, Dict[str, Any]]):
        """Display current configuration."""
        
        print(f"\n  Current Configuration:")
        print(f"  Simulation: {config.simulation_name}")
        print(f"  Time steps: {config.time_steps}")
        print(f"  Time step size: {config.dt}")
        print(f"  Qubits: {config.num_qubits}")
        
        wormhole = params['wormhole_params']
        print(f"  Throat radius: {wormhole['b0']:.2e} m")
        print(f"  Mass: {wormhole['mass']:.2e} kg")
        
        quantum = params['quantum_params']
        print(f"  Traversal probability: {quantum['traversal_probability']:.2f}")
        print(f"  Entanglement strength: {quantum['entanglement_strength']:.2f}")
        
        print(f"  AI components: {config.enable_stability_prediction}")
        print(f"  Real-time visualization: {config.enable_real_time_visualization}")
    
    def display_comprehensive_report(self, report: Dict[str, Any]):
        """Display comprehensive analysis report."""
        
        print(f"\n Comprehensive Analysis Report:")
        
        if 'summary' in report:
            summary = report['summary']
            print(f"\n Summary:")
            print(f"  Simulation successful: {summary.get('simulation_successful', 'Unknown')}")
            print(f"  Total steps: {summary.get('total_steps', 0)}")
            print(f"  Average stability: {summary.get('average_stability', 0):.3f}")
            print(f"  Max entanglement: {summary.get('max_entanglement', 0):.3f}")
        
        if 'stability_analysis' in report:
            stability = report['stability_analysis']
            print(f"\nStability Analysis:")
            for key, value in stability.items():
                if isinstance(value, (int, float)):
                    print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        
        if 'recommendations' in report:
            print(f"\n Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    def create_comprehensive_visualizations(self, results: SimulationResults):
        """Create comprehensive visualization suite."""
        
        logger.info("Creating comprehensive visualizations...")
        
        try:
            # Import visualization components
            from src.visualization.spacetime_plotter import create_multi_panel_spacetime_view
            from src.visualization.quantum_state_animator import create_quantum_animation_suite
            from src.visualization.field_visualizer import FieldVisualizer
            from src.visualization.interactive_dashboard import create_dashboard_suite
            
            # Create spacetime visualizations
            if self.framework and self.framework.physics_engine:
                metric = self.framework.physics_engine['metric']
                spacetime_fig = create_multi_panel_spacetime_view(metric)
                spacetime_fig.write_html(f"{self.framework.config.output_directory}/spacetime_analysis.html")
                print("   Spacetime visualization created")
            
            # Create quantum animations
            quantum_animations = create_quantum_animation_suite()
            for name, fig in quantum_animations.items():
                fig.write_html(f"{self.framework.config.output_directory}/quantum_{name}.html")
            print("   Quantum animations created")
            
            # Create dashboard suite
            dashboard_suite = create_dashboard_suite()
            for name, fig in dashboard_suite.items():
                fig.write_html(f"{self.framework.config.output_directory}/dashboard_{name}.html")
            print("   Interactive dashboard created")
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
    
    def save_results(self, results: SimulationResults, args: argparse.Namespace):
        """Save simulation results."""
        
        if not args.save_results:
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results
        if self.framework:
            results_file = os.path.join(args.output_dir, f"results_{args.name}.{args.results_format}")
            self.framework.save_results(results_file, args.results_format)
            
            # Generate and save report
            if args.generate_report:
                report = self.framework.generate_comprehensive_report()
                report_file = os.path.join(args.output_dir, f"report_{args.name}.json")
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to {report_file}")
        
        logger.info(f"Results saved to {args.output_dir}")


def main():
    """Main entry point for the quantum wormhole simulation."""
    
    # Create command-line interface
    cli = CommandLineInterface()
    args = cli.parse_args()
    
    # Handle utility options first
    if args.version:
        cli.print_version()
        return 0
    
    if args.list_modes:
        cli.print_mode_information()
        return 0
    
    # Setup logging
    cli.setup_logging(args)
    
    # Validate parameters
    if args.validate or args.dry_run:
        is_valid = cli.validate_parameters(args)
        if not is_valid:
            return 1
        
        if args.dry_run:
            print("Configuration validation passed")
            config = cli.load_config(args)
            params = cli.create_simulation_parameters(args)
            cli.display_configuration(config, params)
            return 0
    
    # Save configuration if requested
    if args.save_config:
        config = cli.load_config(args)
        cli.save_config(config, args.save_config)
    
    try:
        # Run simulation based on mode
        logger.info(f"Starting simulation in {args.mode} mode...")
        
        start_time = time.time()
        results = cli.run_simulation_mode(args.mode, args)
        end_time = time.time()
        
        if results is not None:
            execution_time = end_time - start_time
            print(f"\n Simulation completed in {execution_time:.2f} seconds!")
            
            # Save results
            cli.save_results(results, args)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())