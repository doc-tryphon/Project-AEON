"""
Simulation configuration module for quantum wormhole simulations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SimulationConfig:
    """Configuration parameters for wormhole simulation."""
    
    # Basic simulation parameters
    simulation_name: str = "default_wormhole_sim"
    time_steps: int = 100
    dt: float = 0.1
    
    # Physics parameters
    use_relativistic_corrections: bool = True
    include_quantum_corrections: bool = True
    enable_exotic_matter: bool = True
    
    # Quantum system parameters
    num_qubits: int = 4
    quantum_coherence_time: float = 100.0
    enable_decoherence: bool = True
    quantum_backend: str = 'tfq'  # 'tfq', 'qutip', 'hybrid'
    
    # AI/ML parameters
    enable_stability_prediction: bool = False
    enable_parameter_optimization: bool = False
    enable_anomaly_detection: bool = False
    enable_reinforcement_learning: bool = False
    
    # Visualization parameters
    enable_real_time_visualization: bool = False
    visualization_update_interval: int = 50
    save_visualization_frames: bool = False
    
    # Performance parameters
    parallel_processing: bool = False
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Output parameters
    save_intermediate_results: bool = True
    output_directory: str = "simulation_results"
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'SimulationConfig':
        """Update configuration with new values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


def get_default_config() -> SimulationConfig:
    """Get default simulation configuration."""
    return SimulationConfig()


def get_demo_config() -> SimulationConfig:
    """Get configuration for quick demo runs."""
    return SimulationConfig(
        simulation_name="demo_run",
        time_steps=10,
        dt=0.1,
        num_qubits=4,
        enable_real_time_visualization=False,
        parallel_processing=False
    )


def get_research_config() -> SimulationConfig:
    """Get configuration for research-grade simulations."""
    return SimulationConfig(
        simulation_name="research_run",
        time_steps=1000,
        dt=0.01,
        num_qubits=8,
        enable_stability_prediction=True,
        enable_parameter_optimization=True,
        enable_anomaly_detection=True,
        parallel_processing=True,
        max_workers=8,
        memory_limit_gb=16.0,
        save_intermediate_results=True
    )


def get_phase3_config() -> SimulationConfig:
    """Get configuration optimized for Phase 3 quantum-AI features."""
    return SimulationConfig(
        simulation_name="phase3_quantum_ai",
        time_steps=100,
        dt=0.05,
        num_qubits=6,
        quantum_backend='tfq',  # Prefer TensorFlow Quantum
        enable_stability_prediction=True,
        enable_parameter_optimization=True,
        enable_anomaly_detection=True,
        enable_reinforcement_learning=True,
        enable_real_time_visualization=True,
        visualization_update_interval=10,
        parallel_processing=True,
        max_workers=4,
        save_intermediate_results=True
    )