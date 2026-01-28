"""
Quantum Algorithms Module.

Implements verified quantum algorithms with symbolic verification:
- Deutsch-Jozsa Algorithm (exponential speedup for function classification)
- Grover's Search Algorithm (quadratic speedup for unstructured search)
- Quantum Phase Estimation (eigenvalue estimation, foundation of Shor's algorithm)
"""

from .deutsch_jozsa import (
    DeutschJozsaAlgorithm,
    DeutschJozsaResult,
    OracleType,
    create_constant_oracle,
    create_balanced_oracle,
    hadamard_transform,
    verify_deutsch_jozsa_symbolic,
)

from .grover import (
    GroversAlgorithm,
    GroverResult,
    construct_oracle,
    construct_diffuser,
    calculate_optimal_iterations,
    calculate_success_probability,
)

from .phase_estimation import (
    QuantumFourierTransform,
    PhaseEstimationAlgorithm,
    PhaseEstimationResult,
    apply_controlled_unitary_power,
    estimate_phase_from_measurement,
    verify_qft_symbolic,
    verify_phase_estimation_symbolic,
)

__all__ = [
    # Deutsch-Jozsa
    'DeutschJozsaAlgorithm',
    'DeutschJozsaResult',
    'OracleType',
    'create_constant_oracle',
    'create_balanced_oracle',
    'hadamard_transform',
    'verify_deutsch_jozsa_symbolic',
    # Grover
    'GroversAlgorithm',
    'GroverResult',
    'construct_oracle',
    'construct_diffuser',
    'calculate_optimal_iterations',
    'calculate_success_probability',
    # Phase Estimation
    'QuantumFourierTransform',
    'PhaseEstimationAlgorithm',
    'PhaseEstimationResult',
    'apply_controlled_unitary_power',
    'estimate_phase_from_measurement',
    'verify_qft_symbolic',
    'verify_phase_estimation_symbolic',
]
