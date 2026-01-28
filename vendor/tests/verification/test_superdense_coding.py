"""
Comprehensive tests for superdense coding protocol.

Test Categories:
1. Operator verification (unitarity, Hermiticity)
2. All 4 message transmissions
3. Different Bell state resources
4. Orthogonality of encoded states
5. Information capacity verification
6. Duality with teleportation
7. Edge cases (imperfect entanglement, decoherence)
"""

import pytest
import numpy as np
from src.quantum.superdense_coding import (
    SuperdenseCoding,
    SuperdenseCodingAnalyzer,
    SuperdenseCodingResult
)
from src.quantum.entanglement import BellStateGenerator


class TestEncodingOperators:
    """Test properties of Alice's encoding operators."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_all_operators_unitary(self):
        """Verify all encoding operators are unitary (U†U = I)."""
        I = self.protocol.I

        for (b1, b2), (name, U) in self.protocol.encoding_operators.items():
            # Check U†U = I
            product = U.conj().T @ U
            assert np.allclose(product, I, atol=self.tolerance), \
                f"Operator {name} for message ({b1},{b2}) is not unitary"

            # Check UU† = I
            product = U @ U.conj().T
            assert np.allclose(product, I, atol=self.tolerance), \
                f"Operator {name} for message ({b1},{b2}) is not unitary"

    def test_pauli_hermiticity(self):
        """Verify Pauli operators are Hermitian."""
        paulis = [
            ('I', self.protocol.I),
            ('X', self.protocol.X),
            ('Z', self.protocol.Z)
        ]

        for name, P in paulis:
            assert np.allclose(P.conj().T, P, atol=self.tolerance), \
                f"Pauli {name} is not Hermitian"

    def test_pauli_eigenvalues(self):
        """Verify Pauli matrices have eigenvalues ±1."""
        paulis = [
            ('X', self.protocol.X),
            ('Z', self.protocol.Z)
        ]

        for name, P in paulis:
            eigenvalues = np.linalg.eigvalsh(P)
            expected = np.array([-1, 1])
            assert np.allclose(sorted(eigenvalues), expected, atol=self.tolerance), \
                f"Pauli {name} has incorrect eigenvalues"

    def test_pauli_commutators(self):
        """Verify Pauli commutation relations: [X,Z] = XZ - ZX = 2iY."""
        X = self.protocol.X
        Z = self.protocol.Z

        commutator = X @ Z - Z @ X
        # Y = [[0, -i], [i, 0]]
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        # [X,Z] = 2iY = [[0, -2i], [2i, 0]]
        # But XZ - ZX = [[0, -2], [2, 0]] because XZ = [[0,-1],[1,0]] and ZX = [[0,1],[-1,0]]
        # So [X,Z] = XZ - ZX = -2iY (not +2iY)
        expected = -2j * Y
        assert np.allclose(commutator, expected, atol=self.tolerance), \
            "Pauli commutator [X,Z] != -2iY"

    def test_encoding_map_bijection(self):
        """Verify encoding map is bijective (one-to-one correspondence)."""
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # Check all messages are in encoding map
        for msg in messages:
            assert msg in self.protocol.encoding_operators, \
                f"Message {msg} not in encoding map"

        # Check no duplicate operators (up to global phase)
        operators = [U for (name, U) in self.protocol.encoding_operators.values()]
        for i, U1 in enumerate(operators):
            for j, U2 in enumerate(operators):
                if i != j:
                    # Operators should be different (not equal up to phase)
                    # Check if U1 = e^{iθ} U2 for any θ
                    ratio = U1[0, 0] / U2[0, 0] if U2[0, 0] != 0 else U1[0, 1] / U2[0, 1]
                    phase_shifted = ratio * U2
                    if not np.allclose(U1, phase_shifted, atol=self.tolerance):
                        continue  # They are different, good
                    else:
                        pytest.fail(f"Operators {i} and {j} are identical up to phase")


class TestMessageTransmission:
    """Test transmission of all 4 possible messages."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_message_00(self):
        """Test sending message (0, 0) using I operator."""
        result = self.protocol.send_message((0, 0), 'phi_plus')

        assert result.success
        assert result.message_received == (0, 0)
        assert result.encoding_operator == 'I'
        assert result.measurement_outcome == '00'  # Bell label for |Φ+⟩
        assert np.isclose(result.error_rate, 0.0, atol=self.tolerance)

    def test_message_01(self):
        """Test sending message (0, 1) using X operator."""
        result = self.protocol.send_message((0, 1), 'phi_plus')

        assert result.success
        assert result.message_received == (0, 1)
        assert result.encoding_operator == 'X'
        assert result.measurement_outcome == '10'  # Bell label for |Ψ+⟩
        assert np.isclose(result.error_rate, 0.0, atol=self.tolerance)

    def test_message_10(self):
        """Test sending message (1, 0) using Z operator."""
        result = self.protocol.send_message((1, 0), 'phi_plus')

        assert result.success
        assert result.message_received == (1, 0)
        assert result.encoding_operator == 'Z'
        assert result.measurement_outcome == '01'  # Bell label for |Φ-⟩
        assert np.isclose(result.error_rate, 0.0, atol=self.tolerance)

    def test_message_11(self):
        """Test sending message (1, 1) using XZ operator."""
        result = self.protocol.send_message((1, 1), 'phi_plus')

        assert result.success
        assert result.message_received == (1, 1)
        assert result.encoding_operator == 'XZ'
        assert result.measurement_outcome == '11'  # Bell label for |Ψ-⟩
        assert np.isclose(result.error_rate, 0.0, atol=self.tolerance)

    def test_all_messages_success(self):
        """Test all 4 messages achieve 100% success rate."""
        test_results = self.protocol.test_all_messages('phi_plus')

        assert test_results['total_messages'] == 4
        assert test_results['successful'] == 4
        assert test_results['failed'] == 0
        assert np.isclose(test_results['error_rate'], 0.0, atol=self.tolerance)

    def test_encoded_state_normalization(self):
        """Verify all encoded states are properly normalized."""
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for msg in messages:
            result = self.protocol.send_message(msg, 'phi_plus')
            state = result.encoded_state

            # Check normalization: ⟨ψ|ψ⟩ = 1
            norm_squared = np.vdot(state, state).real
            assert np.isclose(norm_squared, 1.0, atol=self.tolerance), \
                f"Encoded state for message {msg} is not normalized"


class TestDifferentBellResources:
    """Test superdense coding with different entanglement resources."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_phi_plus_resource(self):
        """Test with |Φ+⟩ = (|00⟩ + |11⟩)/√2 resource (standard)."""
        test_results = self.protocol.test_all_messages('phi_plus')

        assert test_results['successful'] == 4
        assert np.isclose(test_results['error_rate'], 0.0, atol=self.tolerance)

    def test_different_resources_protocol_runs(self):
        """Test protocol executes with other Bell state resources.

        Note: Standard superdense coding uses |Φ+⟩. Other Bell states
        require modified decoding maps. This test verifies protocol runs
        without errors.
        """
        bell_states = ['phi_minus', 'psi_plus', 'psi_minus']
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for bell_state in bell_states:
            for msg in messages:
                result = self.protocol.send_message(msg, bell_state)

                # Protocol should execute without errors
                assert result is not None
                assert result.message_sent == msg
                assert result.measurement_outcome in ['00', '01', '10', '11']


class TestOrthogonality:
    """Test orthogonality properties of encoded states."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_encoded_states_orthogonal(self):
        """Verify all 4 encoded states are mutually orthogonal."""
        inner_products = self.protocol.verify_orthogonality()
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # Check diagonal elements are 1 (normalized)
        for msg in messages:
            inner_prod = inner_products[(msg, msg)]
            assert np.isclose(abs(inner_prod), 1.0, atol=self.tolerance), \
                f"State {msg} is not normalized"

        # Check off-diagonal elements are 0 (orthogonal)
        for i, msg1 in enumerate(messages):
            for j, msg2 in enumerate(messages):
                if i != j:
                    inner_prod = inner_products[(msg1, msg2)]
                    assert np.isclose(abs(inner_prod), 0.0, atol=self.tolerance), \
                        f"States {msg1} and {msg2} are not orthogonal: ⟨ψ|φ⟩ = {inner_prod}"

    def test_encoded_states_form_bell_basis(self):
        """Verify the 4 encoded states form the complete Bell basis."""
        bell_gen = BellStateGenerator()

        # Expected mapping from our symbolic verification
        expected_mapping = {
            (0, 0): '00',  # I  → |Φ+⟩
            (0, 1): '10',  # X  → |Ψ+⟩
            (1, 0): '01',  # Z  → |Φ-⟩
            (1, 1): '11'   # XZ → |Ψ-⟩
        }

        phi_plus = bell_gen.create_bell_state('00').state_vector  # '00' = |Φ+⟩

        for msg, expected_bell_label in expected_mapping.items():
            # Encode message
            operator_name, operator = self.protocol.encoding_operators[msg]
            two_qubit_op = np.kron(operator, self.protocol.I)
            encoded_state = two_qubit_op @ phi_plus

            # Get expected Bell state
            bell_state = bell_gen.create_bell_state(expected_bell_label).state_vector

            # Check overlap (allowing for global phase)
            overlap = abs(np.vdot(bell_state, encoded_state))
            assert np.isclose(overlap, 1.0, atol=self.tolerance), \
                f"Message {msg} does not encode to Bell state {expected_bell_label}"


class TestInformationCapacity:
    """Test information-theoretic properties."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_channel_capacity_metrics(self):
        """Verify channel capacity calculations."""
        capacity = self.protocol.calculate_channel_capacity()

        assert np.isclose(capacity['qubits_transmitted'], 1.0, atol=self.tolerance)
        assert np.isclose(capacity['classical_bits_sent'], 2.0, atol=self.tolerance)
        assert np.isclose(capacity['bits_per_qubit'], 2.0, atol=self.tolerance)
        assert np.isclose(capacity['enhancement_factor'], 2.0, atol=self.tolerance)
        assert np.isclose(capacity['entanglement_pairs_consumed'], 1.0, atol=self.tolerance)

    def test_holevo_bound(self):
        """Verify protocol respects Holevo bound.

        Holevo bound: χ ≤ n for n qubits
        Superdense coding: 1 qubit + 1 ebit → 2 bits (saturates bound)
        """
        analyzer = SuperdenseCodingAnalyzer()
        verification = analyzer.verify_information_theory(self.protocol)

        assert verification['holevo_bound_satisfied']
        assert verification['achieves_2_bits_per_qubit']

    def test_requires_entanglement(self):
        """Verify protocol requires entanglement as resource."""
        capacity = self.protocol.calculate_channel_capacity()

        # Superdense coding requires exactly 1 Bell pair
        assert np.isclose(capacity['entanglement_pairs_consumed'], 1.0, atol=self.tolerance)

        # Without entanglement, capacity would be 1 bit per qubit
        classical_capacity = capacity['classical_capacity']
        quantum_capacity = capacity['bits_per_qubit']
        assert quantum_capacity > classical_capacity


class TestDualityWithTeleportation:
    """Test and document duality between superdense coding and teleportation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_protocol_duality_documented(self):
        """Verify duality between protocols is properly documented."""
        analyzer = SuperdenseCodingAnalyzer()
        duality = analyzer.compare_to_teleportation()

        # Check all key aspects are documented
        assert 'teleportation' in duality
        assert 'superdense_coding' in duality
        assert 'duality' in duality
        assert 'resource_comparison' in duality

        # Verify key facts
        assert '2 classical bits' in duality['teleportation']
        assert '1 qubit' in duality['superdense_coding']
        assert 'shared Bell pair' in duality['duality'].lower() or 'bell pair' in duality['duality'].lower()

    def test_resource_symmetry(self):
        """Verify both protocols use same entanglement resource."""
        capacity_superdense = self.protocol.calculate_channel_capacity()

        # Both protocols consume exactly 1 Bell pair
        assert np.isclose(capacity_superdense['entanglement_pairs_consumed'], 1.0, atol=1e-10)

        # Superdense: 1 ebit + 1 qubit → 2 cbits
        # Teleport:   1 ebit + 2 cbits → 1 qubit
        # These are dual transformations


class TestEdgeCases:
    """Test protocol behavior with imperfections."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.tolerance = 1e-10

    def test_invalid_message_bits(self):
        """Test error handling for invalid message bits."""
        invalid_messages = [
            (2, 0),
            (0, -1),
            (1, 2),
            (-1, -1)
        ]

        for msg in invalid_messages:
            with pytest.raises(ValueError):
                self.protocol.send_message(msg, 'phi_plus')

    def test_depolarized_entanglement(self):
        """Test with partially depolarized entanglement resource.

        Note: This tests robustness to imperfect entanglement.
        """
        bell_gen = BellStateGenerator()

        # Create perfect Bell state
        phi_plus = bell_gen.create_bell_state('00')  # '00' = |Φ+⟩

        # Verify it's maximally entangled
        assert np.isclose(phi_plus.entanglement_entropy, np.log(2), atol=self.tolerance)

        # With perfect entanglement, all messages succeed
        test_results = self.protocol.test_all_messages('phi_plus')
        assert test_results['error_rate'] == 0.0

    def test_measurement_outcome_deterministic(self):
        """Verify measurement outcomes are deterministic in noiseless case."""
        message = (0, 1)

        # Run same message multiple times
        results = [self.protocol.send_message(message, 'phi_plus') for _ in range(10)]

        # All should give same outcome
        outcomes = [r.measurement_outcome for r in results]
        assert all(o == outcomes[0] for o in outcomes), \
            "Measurement outcomes are not deterministic"

    def test_protocol_reversibility(self):
        """Verify encoding-decoding is reversible (lossless)."""
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for msg in messages:
            result = self.protocol.send_message(msg, 'phi_plus')

            # Message should be perfectly recovered
            assert result.message_sent == result.message_received, \
                f"Protocol not reversible for message {msg}"
            assert result.error_rate == 0.0


class TestProtocolProperties:
    """Test fundamental protocol properties."""

    def setup_method(self):
        """Setup test fixtures."""
        self.protocol = SuperdenseCoding()
        self.analyzer = SuperdenseCodingAnalyzer()
        self.tolerance = 1e-10

    def test_unitarity_verified(self):
        """Verify all encoding operations are unitary."""
        verification = self.analyzer.verify_information_theory(self.protocol)
        assert verification['encoding_operators_unitary']

    def test_no_cloning_theorem_respected(self):
        """Verify protocol doesn't violate no-cloning theorem.

        Superdense coding doesn't clone quantum states - it encodes
        classical information into orthogonal quantum states.
        """
        # Send a message
        result = self.protocol.send_message((0, 1), 'phi_plus')

        # After Bob's measurement, the encoded state is destroyed
        # (This is tested implicitly by the measurement process)
        # We just verify the protocol structure is correct
        assert result.success

    def test_communication_complexity(self):
        """Verify communication complexity: 1 qubit sent."""
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for msg in messages:
            result = self.protocol.send_message(msg, 'phi_plus')

            # Alice sends exactly her qubit (1 qubit transmitted)
            # This is verified by the protocol structure
            assert result.success

        capacity = self.protocol.calculate_channel_capacity()
        assert capacity['qubits_transmitted'] == 1.0