"""
Quantum error correction codes for protecting quantum information.

This module implements the 3-qubit bit flip code, the simplest quantum error
correction code that protects against single bit flip errors.

References:
- Shor, "Scheme for reducing decoherence in quantum computer memory", Phys. Rev. A 52, R2493 (1995)
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000), Ch. 10
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from src.quantum.gates import CNOTGate, GateComposition


@dataclass
class ErrorCorrectionResult:
    """Results from error correction cycle."""
    input_state: np.ndarray
    encoded_state: np.ndarray
    error_applied: Optional[int]
    corrupted_state: np.ndarray
    syndrome: Tuple[int, int]
    corrected_state: np.ndarray
    decoded_state: np.ndarray
    fidelity: float
    success: bool


class BitFlipCode:
    """
    3-qubit bit flip quantum error correction code.

    Encoding: |ψ⟩ = α|0⟩ + β|1⟩ → |ψ_L⟩ = α|000⟩ + β|111⟩

    This code can detect and correct a single bit flip error on any of the
    three qubits. The logical code space is spanned by:
        |0_L⟩ = |000⟩
        |1_L⟩ = |111⟩

    Stabilizers:
        S₁ = Z₀Z₁ (eigenvalue +1 for code space)
        S₂ = Z₁Z₂ (eigenvalue +1 for code space)

    Syndrome measurement gives (s₁, s₂):
        (0,0) → No error
        (1,0) → Qubit 0 flipped
        (1,1) → Qubit 1 flipped
        (0,1) → Qubit 2 flipped
    """

    def __init__(self):
        """Initialize 3-qubit bit flip code."""
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Define encoding/decoding gates
        self._setup_gates()

        # Syndrome lookup table
        self.syndrome_to_error = {
            (0, 0): None,  # No error
            (1, 0): 0,     # Qubit 0 flipped
            (1, 1): 1,     # Qubit 1 flipped
            (0, 1): 2,     # Qubit 2 flipped
        }

    def _setup_gates(self):
        """Set up encoding and decoding circuits."""
        # Encoding circuit: CNOT₀₂ · CNOT₀₁
        self.cnot01 = CNOTGate(control=0, target=1, n_qubits=3)
        self.cnot02 = CNOTGate(control=0, target=2, n_qubits=3)

        # Decoding is reverse: CNOT₀₁ · CNOT₀₂
        # (same gates, reverse order)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Encode single-qubit state into 3-qubit code space.

        Encoding circuit:
            |ψ,0,0⟩ --CNOT₀₁--> |ψ,ψ,0⟩ --CNOT₀₂--> α|000⟩ + β|111⟩

        Args:
            state: Single-qubit state |ψ⟩ = α|0⟩ + β|1⟩

        Returns:
            Encoded 3-qubit state α|000⟩ + β|111⟩

        Raises:
            ValueError: If input state dimension incorrect
        """
        if state.shape[0] != 2:
            raise ValueError(f"Input must be single-qubit state (dim=2), got dim={state.shape[0]}")

        # Verify normalization
        if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
            raise ValueError("Input state must be normalized")

        # Prepare 3-qubit state: |ψ⟩ ⊗ |0⟩ ⊗ |0⟩
        zero = np.array([1, 0], dtype=np.complex128)
        initial_state = np.kron(np.kron(state, zero), zero)

        # Apply encoding circuit
        encoded = GateComposition.sequential_application(
            [self.cnot01, self.cnot02],
            initial_state
        )

        return encoded

    def decode(self, encoded_state: np.ndarray) -> np.ndarray:
        """
        Decode 3-qubit code space back to single-qubit state.

        Decoding circuit (reverse of encoding):
            α|000⟩ + β|111⟩ --CNOT₀₂--> |ψ,ψ,0⟩ --CNOT₀₁--> |ψ,0,0⟩

        Args:
            encoded_state: 3-qubit encoded state

        Returns:
            Decoded single-qubit state (first qubit)

        Raises:
            ValueError: If input dimension incorrect
        """
        if encoded_state.shape[0] != 8:
            raise ValueError(f"Input must be 3-qubit state (dim=8), got dim={encoded_state.shape[0]}")

        # Apply decoding circuit (reverse order)
        decoded_full = GateComposition.sequential_application(
            [self.cnot02, self.cnot01],
            encoded_state
        )

        # Trace out qubits 1 and 2 to get single-qubit state
        # After decoding, state should be |ψ,0,0⟩ so we can just extract qubit 0
        logical_state = self._extract_logical_qubit(decoded_full)

        return logical_state

    def _extract_logical_qubit(self, state_3qubit: np.ndarray) -> np.ndarray:
        """
        Extract logical qubit (qubit 0) from 3-qubit state.

        Assumes state is in form |ψ⟩⊗|00⟩ after decoding.
        """
        # Partial trace over qubits 1 and 2
        # State basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
        # Extract |0**⟩ and |1**⟩ components

        # |0⟩ component (indices 0-3)
        alpha = 0.0
        for i in range(4):
            alpha += state_3qubit[i]

        # |1⟩ component (indices 4-7)
        beta = 0.0
        for i in range(4, 8):
            beta += state_3qubit[i]

        logical_state = np.array([alpha, beta], dtype=np.complex128)

        # Normalize (should already be normalized if input was)
        norm = np.linalg.norm(logical_state)
        if norm > 1e-10:
            logical_state = logical_state / norm

        return logical_state

    def inject_error(self, state: np.ndarray, error_qubit: int) -> np.ndarray:
        """
        Inject bit flip error on specified qubit.

        Args:
            state: 3-qubit state
            error_qubit: Which qubit to flip (0, 1, or 2)

        Returns:
            State with error applied

        Raises:
            ValueError: If error_qubit not in {0, 1, 2}
        """
        if error_qubit not in {0, 1, 2}:
            raise ValueError(f"Error qubit must be 0, 1, or 2, got {error_qubit}")

        # Build X gate on specified qubit
        operators = [self.I, self.I, self.I]
        operators[error_qubit] = self.X

        error_operator = np.kron(np.kron(operators[0], operators[1]), operators[2])

        return error_operator @ state

    def measure_syndrome(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Measure error syndrome without collapsing logical state.

        Measures stabilizers:
            S₁ = Z₀Z₁
            S₂ = Z₁Z₂

        Args:
            state: 3-qubit state (possibly with errors)

        Returns:
            Syndrome tuple (s₁, s₂) where each is 0 or 1
        """
        # Build stabilizer operators
        ZZ_01 = np.kron(np.kron(self.Z, self.Z), self.I)  # Z₀Z₁
        ZZ_12 = np.kron(np.kron(self.I, self.Z), self.Z)  # Z₁Z₂

        # Measure expectation values
        # For stabilizer measurement, eigenvalue +1 → syndrome bit 0
        #                                eigenvalue -1 → syndrome bit 1
        expectation_s1 = np.real(np.conj(state) @ ZZ_01 @ state)
        expectation_s2 = np.real(np.conj(state) @ ZZ_12 @ state)

        # Convert to syndrome bits
        s1 = 0 if expectation_s1 > 0 else 1
        s2 = 0 if expectation_s2 > 0 else 1

        return (s1, s2)

    def apply_correction(self, state: np.ndarray, syndrome: Tuple[int, int]) -> np.ndarray:
        """
        Apply correction based on syndrome measurement.

        Syndrome lookup:
            (0,0) → No error    → Apply I
            (1,0) → Qubit 0 flip → Apply X₀
            (1,1) → Qubit 1 flip → Apply X₁
            (0,1) → Qubit 2 flip → Apply X₂

        Args:
            state: Corrupted 3-qubit state
            syndrome: Measured syndrome (s₁, s₂)

        Returns:
            Corrected state
        """
        error_qubit = self.syndrome_to_error.get(syndrome)

        if error_qubit is None:
            # No error detected, return state unchanged
            return state

        # Apply X gate to error qubit (correcting the flip)
        operators = [self.I, self.I, self.I]
        operators[error_qubit] = self.X

        correction_operator = np.kron(np.kron(operators[0], operators[1]), operators[2])

        return correction_operator @ state

    def full_cycle(self,
                   input_state: np.ndarray,
                   error_qubit: Optional[int] = None,
                   error_probability: float = 0.0) -> ErrorCorrectionResult:
        """
        Perform complete error correction cycle.

        Steps:
            1. Encode: |ψ⟩ → α|000⟩ + β|111⟩
            2. Error: Apply bit flip with probability p
            3. Syndrome: Measure stabilizers
            4. Correct: Apply correction based on syndrome
            5. Decode: Extract logical qubit

        Args:
            input_state: Single-qubit state to protect
            error_qubit: Specific qubit to flip (if None, random based on probability)
            error_probability: Probability of error per qubit

        Returns:
            ErrorCorrectionResult with full cycle information
        """
        # Step 1: Encode
        encoded = self.encode(input_state)

        # Step 2: Apply error
        if error_qubit is not None:
            # Deterministic error
            corrupted = self.inject_error(encoded, error_qubit)
            error_applied = error_qubit
        elif error_probability > 0:
            # Probabilistic error
            rng = np.random.RandomState()
            if rng.random() < error_probability:
                error_applied = rng.choice([0, 1, 2])
                corrupted = self.inject_error(encoded, error_applied)
            else:
                corrupted = encoded
                error_applied = None
        else:
            # No error
            corrupted = encoded
            error_applied = None

        # Step 3: Measure syndrome
        syndrome = self.measure_syndrome(corrupted)

        # Step 4: Apply correction
        corrected = self.apply_correction(corrupted, syndrome)

        # Step 5: Decode
        decoded = self.decode(corrected)

        # Calculate fidelity
        fidelity = np.abs(np.conj(input_state) @ decoded)**2

        # Check success (fidelity > 0.999 = success)
        success = fidelity > 0.999

        return ErrorCorrectionResult(
            input_state=input_state,
            encoded_state=encoded,
            error_applied=error_applied,
            corrupted_state=corrupted,
            syndrome=syndrome,
            corrected_state=corrected,
            decoded_state=decoded,
            fidelity=fidelity,
            success=success
        )

    def verify_code_space(self, tolerance: float = 1e-10) -> Dict:
        """
        Verify properties of the code space.

        Returns:
            Dictionary with verification results
        """
        # Logical basis states
        zero_logical = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        one_logical = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)   # |111⟩

        results = {
            'logical_basis_normalized': True,
            'logical_basis_orthogonal': True,
            'stabilizer_eigenvalues': {},
            'distance': 3,  # Minimum Hamming distance
        }

        # Check normalization
        if not np.isclose(np.linalg.norm(zero_logical), 1.0, atol=tolerance):
            results['logical_basis_normalized'] = False
        if not np.isclose(np.linalg.norm(one_logical), 1.0, atol=tolerance):
            results['logical_basis_normalized'] = False

        # Check orthogonality
        overlap = np.abs(np.conj(zero_logical) @ one_logical)
        if not np.isclose(overlap, 0.0, atol=tolerance):
            results['logical_basis_orthogonal'] = False

        # Verify stabilizer eigenvalues
        ZZ_01 = np.kron(np.kron(self.Z, self.Z), self.I)
        ZZ_12 = np.kron(np.kron(self.I, self.Z), self.Z)

        for name, state in [('|000⟩', zero_logical), ('|111⟩', one_logical)]:
            s1_eigenval = np.real(np.conj(state) @ ZZ_01 @ state)
            s2_eigenval = np.real(np.conj(state) @ ZZ_12 @ state)

            results['stabilizer_eigenvalues'][name] = {
                'S1': s1_eigenval,
                'S2': s2_eigenval,
                'in_code_space': np.isclose(s1_eigenval, 1.0, atol=tolerance) and
                                 np.isclose(s2_eigenval, 1.0, atol=tolerance)
            }

        return results


class LogicalErrorRate:
    """
    Analysis of logical error rates for the 3-qubit bit flip code.

    For independent bit flip errors with probability p per qubit:
        p_L = P(2 or 3 errors) = 3p² - 2p³

    For small p: p_L ≈ 3p²
    Break-even point: p < 1/3
    """

    @staticmethod
    def calculate_logical_error_rate(physical_error_rate: float) -> float:
        """
        Calculate logical error rate from physical error rate.

        Args:
            physical_error_rate: Probability p of error per physical qubit

        Returns:
            Logical error rate p_L
        """
        p = physical_error_rate

        # Exact formula: p_L = 3p² - 2p³
        # This accounts for:
        #   - 3p²(1-p): exactly 2 errors (uncorrectable)
        #   - p³: all 3 errors (uncorrectable)
        p_L = 3 * p**2 - 2 * p**3

        return p_L

    @staticmethod
    def improvement_factor(physical_error_rate: float) -> float:
        """
        Calculate improvement factor: p_L / p.

        For small p, this is approximately 3p, showing suppression.

        Args:
            physical_error_rate: Physical error rate p

        Returns:
            Ratio p_L / p
        """
        p = physical_error_rate
        p_L = LogicalErrorRate.calculate_logical_error_rate(p)

        if p > 0:
            return p_L / p
        else:
            return 0.0

    @staticmethod
    def break_even_threshold() -> float:
        """
        Return break-even threshold where p_L = p.

        Solving 3p² - 2p³ = p:
            2p² - 3p + 1 = 0
            p = (3 ± √(9-8))/4 = (3 ± 1)/4
            p = 1/2 or p = 1

        Physical break-even is p = 1/2.

        Returns:
            Break-even threshold (1/2)
        """
        return 1.0 / 2.0

    @staticmethod
    def analyze_performance(physical_error_rates: List[float]) -> Dict:
        """
        Analyze code performance across range of physical error rates.

        Args:
            physical_error_rates: List of p values to analyze

        Returns:
            Dictionary with performance analysis
        """
        results = {
            'physical_rates': [],
            'logical_rates': [],
            'improvement_factors': [],
            'suppression_achieved': []
        }

        for p in physical_error_rates:
            p_L = LogicalErrorRate.calculate_logical_error_rate(p)
            improvement = LogicalErrorRate.improvement_factor(p)
            suppression = p_L < p  # Code helps if p_L < p

            results['physical_rates'].append(p)
            results['logical_rates'].append(p_L)
            results['improvement_factors'].append(improvement)
            results['suppression_achieved'].append(suppression)

        return results

    @staticmethod
    def pseudo_threshold() -> Dict:
        """
        Calculate pseudo-threshold for the 3-qubit code.

        Pseudo-threshold: p* where d(p_L)/dp = 1
        (maximum benefit from error correction)

        Returns:
            Dictionary with threshold analysis
        """
        # d(p_L)/dp = 6p - 6p² = 1
        # Solving: 6p² - 6p + 1 = 0
        # p = (6 ± √(36-24))/12 = (6 ± √12)/12 = (3 ± √3)/6

        p_threshold = (3 - np.sqrt(3)) / 6  # Take smaller root
        p_L_threshold = LogicalErrorRate.calculate_logical_error_rate(p_threshold)

        return {
            'pseudo_threshold': p_threshold,
            'logical_error_at_threshold': p_L_threshold,
            'derivative_at_threshold': 1.0,
            'break_even': 1.0 / 3.0,
            'optimal_regime': f'p < {p_threshold:.4f}'
        }