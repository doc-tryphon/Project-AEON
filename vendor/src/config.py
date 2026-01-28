"""
Configuration module for verified quantum computing framework.

This module provides configuration for experimentally verified quantum protocols
from the 15-step roadmap (Steps 1-5 currently implemented).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class QuantumConfig:
    """Configuration parameters for verified quantum computing framework."""

    # Basic parameters
    framework_name: str = "quantum_computing_framework"
    numerical_tolerance: float = 1e-10  # For SymPy verification

    # Quantum system parameters
    num_qubits: int = 4
    default_backend: str = 'qutip'  # QuTiP for verified simulations

    # Decoherence parameters (Step 4)
    enable_decoherence: bool = False
    coherence_time_t1: float = 100.0  # T1 relaxation time
    coherence_time_t2: float = 50.0   # T2 dephasing time

    # Error correction parameters (Step 5)
    enable_error_correction: bool = False
    physical_error_rate: float = 0.01  # For QEC simulations

    # Output parameters
    save_results: bool = True
    output_directory: str = "quantum_results"
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantumConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def update(self, **kwargs) -> 'QuantumConfig':
        """Update configuration with new values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


def get_default_config() -> QuantumConfig:
    """Get default quantum computing configuration."""
    return QuantumConfig()


def get_step1_config() -> QuantumConfig:
    """Configuration for Step 1: Bell States & Entanglement."""
    return QuantumConfig(
        framework_name="step1_bell_states",
        num_qubits=2,
        enable_decoherence=False,
        enable_error_correction=False
    )


def get_step2_config() -> QuantumConfig:
    """Configuration for Step 2: Quantum Teleportation."""
    return QuantumConfig(
        framework_name="step2_teleportation",
        num_qubits=3,
        enable_decoherence=False,
        enable_error_correction=False
    )


def get_step3_config() -> QuantumConfig:
    """Configuration for Step 3: Superdense Coding."""
    return QuantumConfig(
        framework_name="step3_superdense_coding",
        num_qubits=2,
        enable_decoherence=False,
        enable_error_correction=False
    )


def get_step4_config() -> QuantumConfig:
    """Configuration for Step 4: Quantum Decoherence."""
    return QuantumConfig(
        framework_name="step4_decoherence",
        num_qubits=2,
        enable_decoherence=True,
        coherence_time_t1=100.0,
        coherence_time_t2=50.0,
        enable_error_correction=False
    )


def get_step5_config() -> QuantumConfig:
    """Configuration for Step 5: Quantum Error Correction."""
    return QuantumConfig(
        framework_name="step5_error_correction",
        num_qubits=3,  # For 3-qubit bit flip code
        enable_decoherence=True,
        enable_error_correction=True,
        physical_error_rate=0.01
    )


def get_research_config() -> QuantumConfig:
    """Configuration for research-grade quantum computing."""
    return QuantumConfig(
        framework_name="research_quantum_computing",
        num_qubits=8,
        numerical_tolerance=1e-12,
        enable_decoherence=True,
        enable_error_correction=True,
        physical_error_rate=0.001,
        save_results=True
    )
