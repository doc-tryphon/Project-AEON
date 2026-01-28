"""
Multi-qubit quantum gates for quantum error correction.

This module implements controlled gates (CNOT, Toffoli) with full unitarity
verification and tensor product utilities for n-qubit operations.

References:
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000), Ch. 4
- Barenco et al., "Elementary gates for quantum computation", Phys. Rev. A 52, 3457 (1995)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import itertools


class MultiQubitGate:
    """Base class for multi-qubit quantum gates."""

    # Single-qubit Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Projection operators for control qubits
    P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0⟩⟨0|
    P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1⟩⟨1|

    @staticmethod
    def tensor_product(operators: List[np.ndarray]) -> np.ndarray:
        """
        Compute tensor product of operators.

        Args:
            operators: List of 2x2 matrices

        Returns:
            Tensor product of all operators
        """
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return result

    @staticmethod
    def partial_gate(gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
        """
        Apply single-qubit gate to target qubit in n-qubit system.

        Args:
            gate: 2x2 gate matrix
            target: Target qubit index (0-indexed)
            n_qubits: Total number of qubits

        Returns:
            2^n × 2^n matrix with gate applied to target qubit
        """
        operators = [MultiQubitGate.I] * n_qubits
        operators[target] = gate
        return MultiQubitGate.tensor_product(operators)

    @staticmethod
    def verify_unitarity(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Verify that U†U = I (unitarity condition).

        Args:
            U: Gate matrix
            tolerance: Numerical tolerance

        Returns:
            True if unitary within tolerance
        """
        n = U.shape[0]
        identity = np.eye(n, dtype=np.complex128)
        product = U.conj().T @ U
        return np.allclose(product, identity, atol=tolerance)

    @staticmethod
    def verify_hermitian(H: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Verify that H† = H (Hermiticity condition).

        Args:
            H: Matrix to check
            tolerance: Numerical tolerance

        Returns:
            True if Hermitian within tolerance
        """
        return np.allclose(H, H.conj().T, atol=tolerance)


class CNOTGate(MultiQubitGate):
    """
    Controlled-NOT gate: flips target qubit if control qubit is |1⟩.

    Matrix form:
        CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X

    Properties:
        - CNOT† = CNOT (self-adjoint)
        - CNOT² = I (involutory)
        - Generates entanglement: CNOT|+0⟩ = (|00⟩ + |11⟩)/√2
    """

    def __init__(self, control: int, target: int, n_qubits: int):
        """
        Initialize CNOT gate.

        Args:
            control: Control qubit index (0-indexed)
            target: Target qubit index (0-indexed)
            n_qubits: Total number of qubits in system

        Raises:
            ValueError: If indices invalid or control == target
        """
        if control < 0 or control >= n_qubits:
            raise ValueError(f"Control qubit {control} out of range [0, {n_qubits-1}]")
        if target < 0 or target >= n_qubits:
            raise ValueError(f"Target qubit {target} out of range [0, {n_qubits-1}]")
        if control == target:
            raise ValueError("Control and target qubits must be different")

        self.control = control
        self.target = target
        self.n_qubits = n_qubits
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """
        Build CNOT matrix using projection operators.

        CNOT = P₀ ⊗ I + P₁ ⊗ X (with identity on other qubits)
        """
        dim = 2 ** self.n_qubits
        cnot = np.zeros((dim, dim), dtype=np.complex128)

        # Iterate over all computational basis states
        for i in range(dim):
            # Convert index to binary string
            state_in = format(i, f'0{self.n_qubits}b')
            state_list = list(state_in)

            # Apply CNOT logic
            if state_list[self.control] == '1':
                # Flip target qubit
                state_list[self.target] = '0' if state_list[self.target] == '1' else '1'

            # Convert back to index
            state_out = ''.join(state_list)
            j = int(state_out, 2)

            # Set matrix element
            cnot[j, i] = 1.0

        return cnot

    @property
    def matrix(self) -> np.ndarray:
        """Get CNOT matrix."""
        return self._matrix

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply CNOT gate to quantum state.

        Args:
            state: State vector of dimension 2^n_qubits

        Returns:
            Transformed state vector
        """
        if state.shape[0] != 2 ** self.n_qubits:
            raise ValueError(f"State dimension {state.shape[0]} incompatible with {self.n_qubits} qubits")
        return self._matrix @ state

    def verify_properties(self, tolerance: float = 1e-10) -> dict:
        """
        Verify CNOT gate properties.

        Returns:
            Dictionary with verification results
        """
        U = self._matrix
        results = {
            'is_unitary': self.verify_unitarity(U, tolerance),
            'is_self_adjoint': np.allclose(U, U.conj().T, atol=tolerance),
            'is_involutory': np.allclose(U @ U, np.eye(2**self.n_qubits), atol=tolerance),
        }

        # Test action on basis states
        results['basis_state_tests'] = self._test_basis_states(tolerance)

        return results

    def _test_basis_states(self, tolerance: float) -> dict:
        """Test CNOT action on computational basis states."""
        tests = {}

        # For 2-qubit case (most common)
        if self.n_qubits == 2:
            # |00⟩ → |00⟩
            state_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
            expected_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
            result_00 = self.apply(state_00)
            tests['00_to_00'] = np.allclose(result_00, expected_00, atol=tolerance)

            # |01⟩ → |01⟩
            state_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
            expected_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
            result_01 = self.apply(state_01)
            tests['01_to_01'] = np.allclose(result_01, expected_01, atol=tolerance)

            # |10⟩ → |11⟩ (control = 1, flip target)
            state_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
            expected_11 = np.array([0, 0, 0, 1], dtype=np.complex128)
            result_10 = self.apply(state_10)
            tests['10_to_11'] = np.allclose(result_10, expected_11, atol=tolerance)

            # |11⟩ → |10⟩ (control = 1, flip target)
            state_11 = np.array([0, 0, 0, 1], dtype=np.complex128)
            expected_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
            result_11 = self.apply(state_11)
            tests['11_to_10'] = np.allclose(result_11, expected_10, atol=tolerance)

        return tests


class ToffoliGate(MultiQubitGate):
    """
    Toffoli gate (CCNOT): flips target if both control qubits are |1⟩.

    Matrix form:
        Toffoli = |00⟩⟨00| ⊗ I + |01⟩⟨01| ⊗ I + |10⟩⟨10| ⊗ I + |11⟩⟨11| ⊗ X

    Properties:
        - Toffoli† = Toffoli (self-adjoint)
        - Toffoli² = I (involutory)
        - Universal for classical computation (NAND gate)
        - Used for syndrome measurement in QEC
    """

    def __init__(self, control1: int, control2: int, target: int, n_qubits: int):
        """
        Initialize Toffoli gate.

        Args:
            control1: First control qubit index
            control2: Second control qubit index
            target: Target qubit index
            n_qubits: Total number of qubits

        Raises:
            ValueError: If indices invalid or not all different
        """
        if control1 < 0 or control1 >= n_qubits:
            raise ValueError(f"Control1 qubit {control1} out of range [0, {n_qubits-1}]")
        if control2 < 0 or control2 >= n_qubits:
            raise ValueError(f"Control2 qubit {control2} out of range [0, {n_qubits-1}]")
        if target < 0 or target >= n_qubits:
            raise ValueError(f"Target qubit {target} out of range [0, {n_qubits-1}]")

        if len({control1, control2, target}) != 3:
            raise ValueError("Control1, control2, and target must be different qubits")

        self.control1 = control1
        self.control2 = control2
        self.target = target
        self.n_qubits = n_qubits
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """
        Build Toffoli matrix.

        Toffoli flips target only when both controls are |1⟩.
        """
        dim = 2 ** self.n_qubits
        toffoli = np.zeros((dim, dim), dtype=np.complex128)

        # Iterate over all computational basis states
        for i in range(dim):
            # Convert index to binary string
            state_in = format(i, f'0{self.n_qubits}b')
            state_list = list(state_in)

            # Apply Toffoli logic
            if state_list[self.control1] == '1' and state_list[self.control2] == '1':
                # Both controls are 1, flip target
                state_list[self.target] = '0' if state_list[self.target] == '1' else '1'

            # Convert back to index
            state_out = ''.join(state_list)
            j = int(state_out, 2)

            # Set matrix element
            toffoli[j, i] = 1.0

        return toffoli

    @property
    def matrix(self) -> np.ndarray:
        """Get Toffoli matrix."""
        return self._matrix

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply Toffoli gate to quantum state.

        Args:
            state: State vector of dimension 2^n_qubits

        Returns:
            Transformed state vector
        """
        if state.shape[0] != 2 ** self.n_qubits:
            raise ValueError(f"State dimension {state.shape[0]} incompatible with {self.n_qubits} qubits")
        return self._matrix @ state

    def verify_properties(self, tolerance: float = 1e-10) -> dict:
        """
        Verify Toffoli gate properties.

        Returns:
            Dictionary with verification results
        """
        U = self._matrix
        results = {
            'is_unitary': self.verify_unitarity(U, tolerance),
            'is_self_adjoint': np.allclose(U, U.conj().T, atol=tolerance),
            'is_involutory': np.allclose(U @ U, np.eye(2**self.n_qubits), atol=tolerance),
        }

        # Test action on basis states
        results['basis_state_tests'] = self._test_basis_states(tolerance)

        return results

    def _test_basis_states(self, tolerance: float) -> dict:
        """Test Toffoli action on computational basis states."""
        tests = {}

        # For 3-qubit case (most common)
        if self.n_qubits == 3 and self.control1 == 0 and self.control2 == 1 and self.target == 2:
            # |110⟩ → |111⟩ (both controls = 1, flip target)
            state_110 = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.complex128)
            expected_111 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)
            result_110 = self.apply(state_110)
            tests['110_to_111'] = np.allclose(result_110, expected_111, atol=tolerance)

            # |111⟩ → |110⟩ (both controls = 1, flip target)
            state_111 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)
            expected_110 = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.complex128)
            result_111 = self.apply(state_111)
            tests['111_to_110'] = np.allclose(result_111, expected_110, atol=tolerance)

            # |010⟩ → |010⟩ (control1 = 0, no flip)
            state_010 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.complex128)
            result_010 = self.apply(state_010)
            tests['010_unchanged'] = np.allclose(result_010, state_010, atol=tolerance)

            # |100⟩ → |100⟩ (control2 = 0, no flip)
            state_100 = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.complex128)
            result_100 = self.apply(state_100)
            tests['100_unchanged'] = np.allclose(result_100, state_100, atol=tolerance)

        return tests


class GateComposition:
    """Utilities for composing multi-qubit gates."""

    @staticmethod
    def sequential_application(gates: List, state: np.ndarray) -> np.ndarray:
        """
        Apply gates sequentially: (G_n ... G_2 G_1) |ψ⟩.

        Args:
            gates: List of gate objects (CNOT or Toffoli)
            state: Input state vector

        Returns:
            Final state after all gates applied
        """
        result = state.copy()
        for gate in gates:
            result = gate.apply(result)
        return result

    @staticmethod
    def compose_matrices(gates: List) -> np.ndarray:
        """
        Compose gate matrices: U_total = U_n ... U_2 U_1.

        Args:
            gates: List of gate objects

        Returns:
            Product of all gate matrices
        """
        result = gates[0].matrix
        for gate in gates[1:]:
            result = gate.matrix @ result
        return result

    @staticmethod
    def verify_composition_unitarity(gates: List, tolerance: float = 1e-10) -> bool:
        """
        Verify that composition of unitary gates is unitary.

        Args:
            gates: List of gate objects
            tolerance: Numerical tolerance

        Returns:
            True if composition is unitary
        """
        U_total = GateComposition.compose_matrices(gates)
        return MultiQubitGate.verify_unitarity(U_total, tolerance)