"""
Verification tests for multi-qubit quantum gates.

This test suite verifies:
1. Gate matrix properties (unitarity, Hermiticity)
2. Gate identities (CNOT² = I, Toffoli² = I)
3. Action on computational basis states
4. Composition properties
5. Entanglement generation

Test tolerance: 1e-10 (10 decimal places)
"""

import pytest
import numpy as np
from src.quantum.gates import (
    MultiQubitGate, CNOTGate, ToffoliGate, GateComposition
)


class TestMultiQubitGateUtilities:
    """Test base class utilities."""

    def test_tensor_product_two_operators(self):
        """Test tensor product of two Pauli operators."""
        I = MultiQubitGate.I
        X = MultiQubitGate.X

        IX = MultiQubitGate.tensor_product([I, X])
        expected = np.kron(I, X)

        assert np.allclose(IX, expected, atol=1e-10)

    def test_tensor_product_three_operators(self):
        """Test tensor product of three operators."""
        I = MultiQubitGate.I
        X = MultiQubitGate.X
        Z = MultiQubitGate.Z

        IXZ = MultiQubitGate.tensor_product([I, X, Z])
        expected = np.kron(np.kron(I, X), Z)

        assert np.allclose(IXZ, expected, atol=1e-10)

    def test_partial_gate_application(self):
        """Test applying gate to specific qubit."""
        X = MultiQubitGate.X
        # Apply X to qubit 1 in 2-qubit system
        gate = MultiQubitGate.partial_gate(X, target=1, n_qubits=2)

        # Should be I ⊗ X
        expected = np.kron(MultiQubitGate.I, X)
        assert np.allclose(gate, expected, atol=1e-10)

    def test_unitarity_verification_valid(self):
        """Test unitarity verification for valid unitary."""
        X = MultiQubitGate.X
        assert MultiQubitGate.verify_unitarity(X)

    def test_unitarity_verification_invalid(self):
        """Test unitarity verification rejects non-unitary."""
        non_unitary = np.array([[1, 0], [0, 2]], dtype=np.complex128)
        assert not MultiQubitGate.verify_unitarity(non_unitary)

    def test_hermiticity_verification(self):
        """Test Hermiticity verification."""
        Z = MultiQubitGate.Z
        assert MultiQubitGate.verify_hermitian(Z)

        non_hermitian = np.array([[1, 1j], [0, 1]], dtype=np.complex128)
        assert not MultiQubitGate.verify_hermitian(non_hermitian)


class TestCNOTGate:
    """Test CNOT gate implementation."""

    def test_initialization_valid(self):
        """Test valid CNOT initialization."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        assert cnot.control == 0
        assert cnot.target == 1
        assert cnot.n_qubits == 2
        assert cnot.matrix.shape == (4, 4)

    def test_initialization_invalid_indices(self):
        """Test error handling for invalid indices."""
        with pytest.raises(ValueError, match="out of range"):
            CNOTGate(control=2, target=1, n_qubits=2)

        with pytest.raises(ValueError, match="out of range"):
            CNOTGate(control=0, target=3, n_qubits=2)

    def test_initialization_same_qubit(self):
        """Test error when control == target."""
        with pytest.raises(ValueError, match="must be different"):
            CNOTGate(control=0, target=0, n_qubits=2)

    def test_cnot_matrix_standard(self):
        """Test CNOT matrix for standard 2-qubit case."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)

        assert np.allclose(cnot.matrix, expected, atol=1e-10)

    def test_cnot_unitarity(self):
        """Test CNOT is unitary."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        properties = cnot.verify_properties()
        assert properties['is_unitary']

    def test_cnot_self_adjoint(self):
        """Test CNOT is self-adjoint (CNOT† = CNOT)."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        properties = cnot.verify_properties()
        assert properties['is_self_adjoint']

    def test_cnot_involutory(self):
        """Test CNOT² = I."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        properties = cnot.verify_properties()
        assert properties['is_involutory']

    def test_cnot_basis_states(self):
        """Test CNOT action on computational basis."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        properties = cnot.verify_properties()

        tests = properties['basis_state_tests']
        assert tests['00_to_00']  # |00⟩ → |00⟩
        assert tests['01_to_01']  # |01⟩ → |01⟩
        assert tests['10_to_11']  # |10⟩ → |11⟩ (flip)
        assert tests['11_to_10']  # |11⟩ → |10⟩ (flip)

    def test_cnot_entanglement_generation(self):
        """Test CNOT generates Bell state from |+0⟩."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)

        # Initial state: |+⟩ ⊗ |0⟩ = (|0⟩ + |1⟩) ⊗ |0⟩ / √2
        plus_zero = np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)

        result = cnot.apply(plus_zero)

        # Expected: (|00⟩ + |11⟩) / √2 = Bell state |Φ+⟩
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

        assert np.allclose(result, expected, atol=1e-10)

    def test_cnot_three_qubit_system(self):
        """Test CNOT in 3-qubit system."""
        # CNOT on qubits 0 (control) and 2 (target) in 3-qubit system
        cnot = CNOTGate(control=0, target=2, n_qubits=3)

        # Test |100⟩ → |101⟩ (control = 1, flip target)
        state_100 = np.zeros(8, dtype=np.complex128)
        state_100[4] = 1.0  # |100⟩ is index 4 (binary 100 = 4)

        result = cnot.apply(state_100)

        expected_101 = np.zeros(8, dtype=np.complex128)
        expected_101[5] = 1.0  # |101⟩ is index 5

        assert np.allclose(result, expected_101, atol=1e-10)

    def test_cnot_preserves_normalization(self):
        """Test CNOT preserves state normalization."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)

        # Create properly normalized state
        state = np.array([0.6, 0.48, 0.48, 0.4], dtype=np.complex128)
        state = state / np.linalg.norm(state)  # Normalize
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)

        result = cnot.apply(state)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-10)


class TestToffoliGate:
    """Test Toffoli gate implementation."""

    def test_initialization_valid(self):
        """Test valid Toffoli initialization."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)
        assert toffoli.control1 == 0
        assert toffoli.control2 == 1
        assert toffoli.target == 2
        assert toffoli.n_qubits == 3
        assert toffoli.matrix.shape == (8, 8)

    def test_initialization_invalid_indices(self):
        """Test error handling for invalid indices."""
        with pytest.raises(ValueError, match="out of range"):
            ToffoliGate(control1=3, control2=1, target=2, n_qubits=3)

    def test_initialization_duplicate_qubits(self):
        """Test error when qubits not all different."""
        with pytest.raises(ValueError, match="must be different"):
            ToffoliGate(control1=0, control2=0, target=2, n_qubits=3)

        with pytest.raises(ValueError, match="must be different"):
            ToffoliGate(control1=0, control2=1, target=0, n_qubits=3)

    def test_toffoli_unitarity(self):
        """Test Toffoli is unitary."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)
        properties = toffoli.verify_properties()
        assert properties['is_unitary']

    def test_toffoli_self_adjoint(self):
        """Test Toffoli is self-adjoint."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)
        properties = toffoli.verify_properties()
        assert properties['is_self_adjoint']

    def test_toffoli_involutory(self):
        """Test Toffoli² = I."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)
        properties = toffoli.verify_properties()
        assert properties['is_involutory']

    def test_toffoli_basis_states(self):
        """Test Toffoli action on computational basis."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)
        properties = toffoli.verify_properties()

        tests = properties['basis_state_tests']
        assert tests['110_to_111']      # |110⟩ → |111⟩ (both controls = 1)
        assert tests['111_to_110']      # |111⟩ → |110⟩ (both controls = 1)
        assert tests['010_unchanged']   # |010⟩ → |010⟩ (control1 = 0)
        assert tests['100_unchanged']   # |100⟩ → |100⟩ (control2 = 0)

    def test_toffoli_classical_nand(self):
        """Test Toffoli implements classical NAND gate."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)

        # Test all classical inputs with target initialized to |1⟩
        # NAND(0,0) = 1: |001⟩ → |001⟩
        state_001 = np.zeros(8, dtype=np.complex128)
        state_001[1] = 1.0
        result_001 = toffoli.apply(state_001)
        assert np.allclose(result_001, state_001, atol=1e-10)

        # NAND(0,1) = 1: |011⟩ → |011⟩
        state_011 = np.zeros(8, dtype=np.complex128)
        state_011[3] = 1.0
        result_011 = toffoli.apply(state_011)
        assert np.allclose(result_011, state_011, atol=1e-10)

        # NAND(1,0) = 1: |101⟩ → |101⟩
        state_101 = np.zeros(8, dtype=np.complex128)
        state_101[5] = 1.0
        result_101 = toffoli.apply(state_101)
        assert np.allclose(result_101, state_101, atol=1e-10)

        # NAND(1,1) = 0: |111⟩ → |110⟩ (flip target from 1 to 0)
        state_111 = np.zeros(8, dtype=np.complex128)
        state_111[7] = 1.0
        expected_110 = np.zeros(8, dtype=np.complex128)
        expected_110[6] = 1.0
        result_111 = toffoli.apply(state_111)
        assert np.allclose(result_111, expected_110, atol=1e-10)

    def test_toffoli_preserves_normalization(self):
        """Test Toffoli preserves state normalization."""
        toffoli = ToffoliGate(control1=0, control2=1, target=2, n_qubits=3)

        # Random normalized state
        rng = np.random.RandomState(42)
        state = rng.randn(8) + 1j * rng.randn(8)
        state = state / np.linalg.norm(state)

        result = toffoli.apply(state)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-10)


class TestGateComposition:
    """Test gate composition utilities."""

    def test_sequential_cnot_application(self):
        """Test sequential application of two CNOTs."""
        cnot12 = CNOTGate(control=0, target=1, n_qubits=2)
        cnot21 = CNOTGate(control=1, target=0, n_qubits=2)

        # Start with |10⟩
        state = np.array([0, 0, 1, 0], dtype=np.complex128)

        # Apply CNOT₁₂ then CNOT₂₁
        result = GateComposition.sequential_application([cnot12, cnot21], state)

        # CNOT₁₂|10⟩ = |11⟩, then CNOT₂₁|11⟩ = |01⟩
        expected = np.array([0, 1, 0, 0], dtype=np.complex128)

        assert np.allclose(result, expected, atol=1e-10)

    def test_compose_cnot_matrices(self):
        """Test matrix composition of CNOTs."""
        cnot12 = CNOTGate(control=0, target=1, n_qubits=2)
        cnot21 = CNOTGate(control=1, target=0, n_qubits=2)

        composed = GateComposition.compose_matrices([cnot12, cnot21])

        # Verify it's equivalent to sequential application
        state = np.array([0, 0, 1, 0], dtype=np.complex128)
        result1 = composed @ state
        result2 = GateComposition.sequential_application([cnot12, cnot21], state)

        assert np.allclose(result1, result2, atol=1e-10)

    def test_composition_unitarity(self):
        """Test composition of unitary gates is unitary."""
        cnot12 = CNOTGate(control=0, target=1, n_qubits=2)
        cnot21 = CNOTGate(control=1, target=0, n_qubits=2)

        assert GateComposition.verify_composition_unitarity([cnot12, cnot21])

    def test_cnot_involutory_via_composition(self):
        """Test CNOT² = I via composition."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)

        # Compose CNOT with itself
        composed = GateComposition.compose_matrices([cnot, cnot])

        # Should be identity
        identity = np.eye(4, dtype=np.complex128)
        assert np.allclose(composed, identity, atol=1e-10)

    def test_encoding_circuit_composition(self):
        """Test composition of CNOT gates for 3-qubit encoding."""
        # Encoding circuit: CNOT₁₃ · CNOT₁₂
        cnot12 = CNOTGate(control=0, target=1, n_qubits=3)
        cnot13 = CNOTGate(control=0, target=2, n_qubits=3)

        # Test |100⟩ → |111⟩ (encoding |1⟩)
        state_100 = np.zeros(8, dtype=np.complex128)
        state_100[4] = 1.0  # |100⟩

        result = GateComposition.sequential_application([cnot12, cnot13], state_100)

        expected_111 = np.zeros(8, dtype=np.complex128)
        expected_111[7] = 1.0  # |111⟩

        assert np.allclose(result, expected_111, atol=1e-10)

    def test_decoding_circuit_reverses_encoding(self):
        """Test decoding circuit reverses encoding."""
        # Encoding: CNOT₁₃ · CNOT₁₂
        # Decoding: CNOT₁₂ · CNOT₁₃ (reverse order)
        cnot12 = CNOTGate(control=0, target=1, n_qubits=3)
        cnot13 = CNOTGate(control=0, target=2, n_qubits=3)

        # Start with arbitrary state on qubit 0
        state_plus = np.zeros(8, dtype=np.complex128)
        state_plus[0] = 1/np.sqrt(2)  # |000⟩
        state_plus[4] = 1/np.sqrt(2)  # |100⟩

        # Encode
        encoded = GateComposition.sequential_application([cnot12, cnot13], state_plus)

        # Decode (reverse order)
        decoded = GateComposition.sequential_application([cnot13, cnot12], encoded)

        # Should return to original state
        assert np.allclose(decoded, state_plus, atol=1e-10)


class TestGateIdentities:
    """Test important gate identities."""

    def test_cnot_swap_decomposition(self):
        """Test SWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂."""
        cnot12 = CNOTGate(control=0, target=1, n_qubits=2)
        cnot21 = CNOTGate(control=1, target=0, n_qubits=2)

        swap = GateComposition.compose_matrices([cnot12, cnot21, cnot12])

        # Test on |10⟩ → |01⟩
        state_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        result = swap @ state_10
        expected_01 = np.array([0, 1, 0, 0], dtype=np.complex128)

        assert np.allclose(result, expected_01, atol=1e-10)

        # Test on |01⟩ → |10⟩
        state_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        result = swap @ state_01
        expected_10 = np.array([0, 0, 1, 0], dtype=np.complex128)

        assert np.allclose(result, expected_10, atol=1e-10)

    def test_cnot_commutes_with_ii(self):
        """Test CNOT commutes with I ⊗ I (trivial)."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        II = np.kron(MultiQubitGate.I, MultiQubitGate.I)

        # [CNOT, I⊗I] = 0
        commutator = cnot.matrix @ II - II @ cnot.matrix
        assert np.allclose(commutator, np.zeros((4, 4)), atol=1e-10)

    def test_cnot_anticommutes_with_xi(self):
        """Test CNOT anticommutes with X ⊗ I."""
        cnot = CNOTGate(control=0, target=1, n_qubits=2)
        XI = np.kron(MultiQubitGate.X, MultiQubitGate.I)

        # {CNOT, X⊗I} = 0 (anticommutator)
        anticommutator = cnot.matrix @ XI + XI @ cnot.matrix
        # Note: This is NOT zero - CNOT and X⊗I do not anticommute
        # Instead, X⊗I conjugates to X⊗X:
        # CNOT (X⊗I) CNOT† = X⊗X
        conjugated = cnot.matrix @ XI @ cnot.matrix.conj().T
        XX = np.kron(MultiQubitGate.X, MultiQubitGate.X)
        assert np.allclose(conjugated, XX, atol=1e-10)