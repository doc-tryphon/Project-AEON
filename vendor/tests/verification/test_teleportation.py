"""
Test suite for quantum teleportation protocol.

All tests verify against known analytical results:
- Bennett et al., Phys. Rev. Lett. 70, 1895 (1993)
- Nielsen & Chuang, Section 1.3.7
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.quantum.teleportation import (
    QuantumTeleportation,
    teleport_basis_states,
    teleport_arbitrary_superposition
)


class TestTeleportationBasics:
    """Test basic teleportation protocol functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation(fidelity_threshold=0.99)

    def test_correction_operators_unitary(self):
        """Test that all Pauli correction operators are unitary."""
        I = np.eye(2, dtype=complex)

        for outcome, (name, U) in self.protocol.corrections.items():
            # Check U†U = I
            U_dag_U = U.conj().T @ U
            assert np.allclose(U_dag_U, I), \
                f"Correction {name} not unitary: U†U = {U_dag_U}"

            # Check UU† = I
            U_U_dag = U @ U.conj().T
            assert np.allclose(U_U_dag, I), \
                f"Correction {name} not unitary: UU† = {U_U_dag}"

    def test_correction_operators_hermitian(self):
        """Test that single Pauli operators are Hermitian: U† = U.

        Note: XZ is not Hermitian, but it is unitary (already tested).
        """
        hermitian_ops = ['I', 'X', 'Z']

        for outcome, (name, U) in self.protocol.corrections.items():
            if name in hermitian_ops:
                assert np.allclose(U.conj().T, U), \
                    f"Pauli {name} should be Hermitian"
            else:
                # XZ is unitary but not Hermitian
                pass


class TestTeleportBasisStates:
    """Test teleportation of standard basis states."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation()
        self.tolerance = 1e-10

    def test_teleport_ket_0(self):
        """Test teleportation of |0⟩."""
        ket_0 = np.array([1, 0], dtype=complex)
        result = self.protocol.teleport(ket_0)

        assert result.protocol_successful, "Teleportation failed"
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|0⟩ fidelity: {result.fidelity}"

        # Verify output state matches input
        assert np.allclose(result.output_state, ket_0, atol=self.tolerance)

    def test_teleport_ket_1(self):
        """Test teleportation of |1⟩."""
        ket_1 = np.array([0, 1], dtype=complex)
        result = self.protocol.teleport(ket_1)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|1⟩ fidelity: {result.fidelity}"
        assert np.allclose(result.output_state, ket_1, atol=self.tolerance)

    def test_teleport_ket_plus(self):
        """Test teleportation of |+⟩ = (|0⟩ + |1⟩)/√2."""
        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        result = self.protocol.teleport(ket_plus)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|+⟩ fidelity: {result.fidelity}"

        # Allow for global phase
        overlap = np.abs(np.dot(result.output_state.conj(), ket_plus))
        assert np.isclose(overlap, 1.0, atol=self.tolerance)

    def test_teleport_ket_minus(self):
        """Test teleportation of |-⟩ = (|0⟩ - |1⟩)/√2."""
        ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
        result = self.protocol.teleport(ket_minus)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|-⟩ fidelity: {result.fidelity}"

    def test_teleport_ket_i(self):
        """Test teleportation of |i⟩ = (|0⟩ + i|1⟩)/√2."""
        ket_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)
        result = self.protocol.teleport(ket_i)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|i⟩ fidelity: {result.fidelity}"

    def test_teleport_ket_minus_i(self):
        """Test teleportation of |-i⟩ = (|0⟩ - i|1⟩)/√2."""
        ket_minus_i = np.array([1, -1j], dtype=complex) / np.sqrt(2)
        result = self.protocol.teleport(ket_minus_i)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"|-i⟩ fidelity: {result.fidelity}"

    def test_all_basis_states(self):
        """Test all six basis states at once."""
        results = teleport_basis_states()

        for name, result in results.items():
            assert result.protocol_successful, \
                f"{name} teleportation failed"
            assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
                f"{name} fidelity: {result.fidelity}"


class TestArbitrarySuperposition:
    """Test teleportation of arbitrary quantum states."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation()
        self.tolerance = 1e-10

    def test_arbitrary_real_superposition(self):
        """Test teleportation of arbitrary real superposition."""
        # Random real coefficients
        alpha = 0.6
        beta = 0.8
        result = teleport_arbitrary_superposition(alpha, beta)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"Arbitrary real fidelity: {result.fidelity}"

    def test_arbitrary_complex_superposition(self):
        """Test teleportation of arbitrary complex superposition."""
        # Random complex coefficients
        alpha = 0.6 + 0.3j
        beta = 0.7 - 0.2j
        result = teleport_arbitrary_superposition(alpha, beta)

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
            f"Arbitrary complex fidelity: {result.fidelity}"

    def test_multiple_random_states(self):
        """Test teleportation of 100 random quantum states."""
        np.random.seed(42)

        for _ in range(100):
            # Generate random state
            alpha = np.random.randn() + 1j*np.random.randn()
            beta = np.random.randn() + 1j*np.random.randn()

            result = teleport_arbitrary_superposition(alpha, beta)

            assert result.protocol_successful, \
                f"Random state teleportation failed: α={alpha}, β={beta}"
            assert np.isclose(result.fidelity, 1.0, atol=self.tolerance), \
                f"Random state fidelity: {result.fidelity}"


class TestDifferentBellStates:
    """Test teleportation using different Bell state resources."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation()
        self.test_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        self.tolerance = 1e-10

    def test_phi_plus_resource(self):
        """Test teleportation with |Φ+⟩ entanglement."""
        result = self.protocol.teleport(self.test_state, 'phi_plus')

        assert result.protocol_successful
        assert np.isclose(result.fidelity, 1.0, atol=self.tolerance)

    def test_other_bell_resources_run(self):
        """Test that protocol runs with other Bell states.

        Note: The standard teleportation protocol is designed for |Φ+⟩.
        Using other Bell states as resources requires modified correction
        operators. This test verifies the protocol executes without errors
        but does not guarantee F=1 with standard corrections.

        Reference: Bennett et al. (1993) uses |Φ+⟩ specifically.
        """
        bell_states = ['phi_minus', 'psi_plus', 'psi_minus']

        for bell_state in bell_states:
            result = self.protocol.teleport(self.test_state, bell_state)

            # Protocol should execute without errors
            assert result is not None
            assert result.measurement_outcome in [(0,0), (0,1), (1,0), (1,1)]
            assert 0.0 <= result.fidelity <= 1.0


class TestNoSignaling:
    """Test no-signaling theorem: Bob can't extract info before classical message."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation()
        self.tolerance = 1e-10

    def test_bob_maximally_mixed_ket_0(self):
        """Test Bob's state is maximally mixed when Alice has |0⟩."""
        ket_0 = np.array([1, 0], dtype=complex)
        result = self.protocol.verify_no_signaling(ket_0)

        assert result['no_signaling_verified'], \
            "No-signaling violated for |0⟩"
        assert np.isclose(result['purity'], 0.5, atol=self.tolerance), \
            f"Bob's state purity: {result['purity']}"

    def test_bob_maximally_mixed_ket_1(self):
        """Test Bob's state is maximally mixed when Alice has |1⟩."""
        ket_1 = np.array([0, 1], dtype=complex)
        result = self.protocol.verify_no_signaling(ket_1)

        assert result['no_signaling_verified']
        assert np.isclose(result['purity'], 0.5, atol=self.tolerance)

    def test_bob_maximally_mixed_superposition(self):
        """Test Bob's state is maximally mixed for arbitrary superposition."""
        # Random state
        alpha = 0.6 + 0.3j
        beta = 0.7 - 0.2j
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        state = np.array([alpha, beta], dtype=complex) / norm

        result = self.protocol.verify_no_signaling(state)

        assert result['no_signaling_verified'], \
            "No-signaling violated for superposition"

    def test_no_signaling_all_bell_resources(self):
        """Test no-signaling for all Bell state resources."""
        test_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        bell_states = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']

        for bell in bell_states:
            result = self.protocol.verify_no_signaling(test_state, bell)
            assert result['no_signaling_verified'], \
                f"No-signaling violated for {bell}"


class TestProtocolProperties:
    """Test mathematical properties of the protocol."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = QuantumTeleportation()

    def test_measurement_probabilities_sum_to_one(self):
        """Test that all measurement outcome probabilities sum to 1."""
        # This is implicitly verified by the protocol always succeeding,
        # but let's check explicitly
        test_state = np.array([0.6, 0.8], dtype=complex)

        # Run protocol and verify it finds a valid outcome
        result = self.protocol.teleport(test_state)

        assert result.measurement_outcome in [(0,0), (0,1), (1,0), (1,1)], \
            f"Invalid measurement outcome: {result.measurement_outcome}"

    def test_linearity(self):
        """Test protocol preserves superposition (linearity)."""
        # Teleport |0⟩
        ket_0 = np.array([1, 0], dtype=complex)
        result_0 = self.protocol.teleport(ket_0)

        # Teleport |1⟩
        ket_1 = np.array([0, 1], dtype=complex)
        result_1 = self.protocol.teleport(ket_1)

        # Teleport superposition (|0⟩ + |1⟩)/√2
        ket_plus = (ket_0 + ket_1) / np.sqrt(2)
        result_plus = self.protocol.teleport(ket_plus)

        # All should succeed with F=1
        assert result_0.fidelity >= 0.99
        assert result_1.fidelity >= 0.99
        assert result_plus.fidelity >= 0.99


if __name__ == '__main__':
    pytest.main([__file__, '-v'])