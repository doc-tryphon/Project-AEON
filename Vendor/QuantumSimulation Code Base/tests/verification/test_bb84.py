"""
TDD Test Suite for BB84 Quantum Key Distribution Protocol (Step 8).

Tests verify:
1. Protocol correctness (no eavesdropper)
2. Eavesdropper detection (intercept-resend attack)
3. Symbolic verification of security properties

Run with: pytest tests/verification/test_bb84.py -v
"""

import pytest
import numpy as np
from sympy import sqrt, simplify, Rational

import sys
sys.path.insert(0, 'src')

from quantum.bb84 import (
    BB84Protocol, BB84Result, BB84State, Basis,
    get_bb84_states_symbolic, verify_mutually_unbiased_bases,
    verify_no_cloning_impossibility, verify_measurement_disturbance
)
from verification.symbolic_solver import QuantumVerifier


class TestBB84States:
    """Test BB84 state preparation and properties."""

    @pytest.fixture
    def protocol(self):
        return BB84Protocol(num_qubits=100)

    def test_four_bb84_states_exist(self, protocol):
        """Test that all four BB84 states are defined."""
        assert len(protocol.states) == 4

        # Check all combinations
        assert (0, Basis.COMPUTATIONAL) in protocol.states
        assert (1, Basis.COMPUTATIONAL) in protocol.states
        assert (0, Basis.HADAMARD) in protocol.states
        assert (1, Basis.HADAMARD) in protocol.states

    def test_states_are_normalized(self, protocol):
        """Test all BB84 states have unit norm."""
        for key, state in protocol.states.items():
            norm = np.linalg.norm(state.state_vector)
            assert np.isclose(norm, 1.0), f"State {state.label} not normalized"

    def test_z_basis_orthogonal(self, protocol):
        """Test |0⟩ and |1⟩ are orthogonal."""
        state_0 = protocol.states[(0, Basis.COMPUTATIONAL)].state_vector
        state_1 = protocol.states[(1, Basis.COMPUTATIONAL)].state_vector
        inner = np.abs(np.vdot(state_0, state_1))
        assert np.isclose(inner, 0.0), "Z basis states not orthogonal"

    def test_x_basis_orthogonal(self, protocol):
        """Test |+⟩ and |−⟩ are orthogonal."""
        state_plus = protocol.states[(0, Basis.HADAMARD)].state_vector
        state_minus = protocol.states[(1, Basis.HADAMARD)].state_vector
        inner = np.abs(np.vdot(state_plus, state_minus))
        assert np.isclose(inner, 0.0), "X basis states not orthogonal"

    def test_bases_are_mutually_unbiased(self, protocol):
        """Test |⟨ψ_Z|ψ_X⟩|² = 1/2 for all cross-basis pairs."""
        z_states = [
            protocol.states[(0, Basis.COMPUTATIONAL)].state_vector,
            protocol.states[(1, Basis.COMPUTATIONAL)].state_vector
        ]
        x_states = [
            protocol.states[(0, Basis.HADAMARD)].state_vector,
            protocol.states[(1, Basis.HADAMARD)].state_vector
        ]

        for z in z_states:
            for x in x_states:
                overlap_sq = np.abs(np.vdot(z, x))**2
                assert np.isclose(overlap_sq, 0.5), "Bases not mutually unbiased"


class TestBB84ProtocolNoEavesdropper:
    """Test BB84 protocol without eavesdropper (ideal case)."""

    @pytest.fixture
    def protocol(self):
        return BB84Protocol(num_qubits=1000, error_estimation_fraction=0.5)

    def test_alice_prepare_generates_correct_count(self, protocol):
        """Test Alice prepares the correct number of qubits."""
        states, bits, bases = protocol.alice_prepare()
        assert len(states) == 1000
        assert len(bits) == 1000
        assert len(bases) == 1000

    def test_alice_bits_are_binary(self, protocol):
        """Test Alice's bits are all 0 or 1."""
        _, bits, _ = protocol.alice_prepare()
        assert all(b in [0, 1] for b in bits)

    def test_alice_bases_are_valid(self, protocol):
        """Test Alice's bases are all Z or X."""
        _, _, bases = protocol.alice_prepare()
        assert all(b in [Basis.COMPUTATIONAL, Basis.HADAMARD] for b in bases)

    def test_bob_measure_returns_correct_count(self, protocol):
        """Test Bob measures all received qubits."""
        states, _, _ = protocol.alice_prepare()
        results, bases = protocol.bob_measure(states)
        assert len(results) == 1000
        assert len(bases) == 1000

    def test_sifting_rate_approximately_half(self, protocol):
        """Test sifting keeps approximately 50% of bits."""
        result = protocol.run_protocol(eve_intercept=False)

        # Should be close to 0.5 (50% matching bases)
        assert 0.3 < result.sifting_rate < 0.7, \
            f"Sifting rate {result.sifting_rate} not near expected 0.5"

    def test_qber_zero_without_eavesdropper(self, protocol):
        """Test QBER is 0 when no eavesdropper (ideal channel)."""
        result = protocol.run_protocol(eve_intercept=False)

        # With no Eve and no channel noise, QBER should be 0
        assert result.qber == 0.0, f"QBER should be 0, got {result.qber}"

    def test_protocol_successful_without_eavesdropper(self, protocol):
        """Test protocol succeeds without eavesdropper."""
        result = protocol.run_protocol(eve_intercept=False)
        assert result.protocol_successful is True

    def test_eve_not_detected_without_eavesdropper(self, protocol):
        """Test Eve detection flag is False when no Eve."""
        result = protocol.run_protocol(eve_intercept=False)
        assert result.eve_detected is False

    def test_sifted_keys_match_without_eavesdropper(self, protocol):
        """Test Alice and Bob have identical sifted keys without Eve."""
        result = protocol.run_protocol(eve_intercept=False)
        assert result.sifted_key_alice == result.sifted_key_bob


class TestBB84EavesdropperDetection:
    """Test BB84 protocol detects eavesdropping."""

    @pytest.fixture
    def protocol(self):
        return BB84Protocol(num_qubits=2000, error_estimation_fraction=0.5)

    def test_intercept_resend_causes_errors(self, protocol):
        """Test intercept-resend attack introduces errors."""
        result = protocol.run_protocol(eve_intercept=True, eve_strategy='intercept_resend')

        # Eve's attack should cause ~25% QBER
        assert result.qber > 0, "Eve should introduce errors"

    def test_qber_approximately_25_percent_with_eve(self, protocol):
        """Test QBER is approximately 25% with intercept-resend attack."""
        result = protocol.run_protocol(eve_intercept=True, eve_strategy='intercept_resend')

        # Should be around 0.25 (±0.1 for statistical variation)
        assert 0.15 < result.qber < 0.35, \
            f"QBER {result.qber} not near expected 0.25"

    def test_eve_detected_with_eavesdropper(self, protocol):
        """Test Eve is detected when eavesdropping."""
        result = protocol.run_protocol(eve_intercept=True, eve_strategy='intercept_resend')

        # QBER should exceed 11% threshold
        assert result.eve_detected is True, \
            f"Eve should be detected, QBER={result.qber}"

    def test_protocol_fails_with_eavesdropper(self, protocol):
        """Test protocol is marked as failed when Eve is present."""
        result = protocol.run_protocol(eve_intercept=True, eve_strategy='intercept_resend')
        assert result.protocol_successful is False

    def test_information_leaked_estimated(self, protocol):
        """Test information leakage is estimated when Eve present."""
        result = protocol.run_protocol(eve_intercept=True, eve_strategy='intercept_resend')
        assert result.estimated_information_leaked > 0


class TestBB84SecurityAnalysis:
    """Test BB84 security analysis functions."""

    @pytest.fixture
    def protocol(self):
        return BB84Protocol(num_qubits=500)

    def test_analyze_security_returns_structure(self, protocol):
        """Test security analysis returns expected structure."""
        result = protocol.run_protocol(eve_intercept=False)
        analysis = protocol.analyze_security(result)

        assert 'protocol_summary' in analysis
        assert 'error_analysis' in analysis
        assert 'security_assessment' in analysis
        assert 'theoretical_bounds' in analysis

    def test_theoretical_bounds_documented(self, protocol):
        """Test theoretical bounds are included in analysis."""
        result = protocol.run_protocol(eve_intercept=False)
        analysis = protocol.analyze_security(result)

        bounds = analysis['theoretical_bounds']
        assert bounds['intercept_resend_qber'] == 0.25
        assert bounds['shor_preskill_bound'] == 0.146


class TestBB84SymbolicVerification:
    """Test symbolic verification of BB84 security properties."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    # --- BB84 States ---

    def test_get_bb84_states_returns_four_states(self, verifier):
        """Test QuantumVerifier returns all four BB84 states."""
        states = verifier.get_bb84_states()
        assert len(states) == 4
        assert '|0⟩' in states
        assert '|1⟩' in states
        assert '|+⟩' in states
        assert '|−⟩' in states

    def test_bb84_states_correct_dimension(self, verifier):
        """Test all BB84 states are 2-dimensional."""
        states = verifier.get_bb84_states()
        for name, state in states.items():
            assert state.shape == (2, 1), f"{name} has wrong shape"

    # --- Orthogonality ---

    def test_z_basis_orthogonality_symbolic(self, verifier):
        """Test ⟨0|1⟩ = 0 symbolically."""
        result = verifier.verify_bb84_orthogonality()
        assert result['z_basis_orthogonal'] is True
        assert result['z_inner_product'] == '0'

    def test_x_basis_orthogonality_symbolic(self, verifier):
        """Test ⟨+|−⟩ = 0 symbolically."""
        result = verifier.verify_bb84_orthogonality()
        assert result['x_basis_orthogonal'] is True
        assert result['x_inner_product'] == '0'

    # --- Mutual Unbiasedness ---

    def test_mutually_unbiased_bases(self, verifier):
        """Test Z and X bases are mutually unbiased."""
        result = verifier.verify_mutually_unbiased_bases()
        assert result['mutually_unbiased'] is True

    def test_all_cross_overlaps_equal_half(self, verifier):
        """Test |⟨ψ_Z|ψ_X⟩|² = 1/2 for all pairs."""
        result = verifier.verify_mutually_unbiased_bases()

        for overlap_name, overlap_data in result['overlaps'].items():
            assert overlap_data['equals_1/2'] is True, \
                f"{overlap_name} should equal 1/2"

    # --- No-Cloning Security ---

    def test_no_cloning_applies_to_bb84(self, verifier):
        """Test no-cloning theorem applies to BB84 states."""
        result = verifier.verify_no_cloning_security()
        assert result['no_cloning_applies'] is True

    def test_cross_basis_states_not_clonable(self, verifier):
        """Test states from different bases cannot be cloned."""
        result = verifier.verify_no_cloning_security()

        for pair_name, pair_data in result['pairs_analyzed'].items():
            # Cross-basis states should NOT be clonable
            assert pair_data['clonable'] is False, \
                f"{pair_name} should not be clonable"

    # --- Measurement Disturbance ---

    def test_wrong_basis_gives_equal_probabilities(self, verifier):
        """Test wrong-basis measurement gives 50/50 outcomes."""
        result = verifier.verify_measurement_disturbance()

        for scenario_name, scenario_data in result['scenarios'].items():
            assert scenario_data['equal_probabilities'] is True, \
                f"{scenario_name} should have equal probabilities"

    def test_wrong_basis_gives_no_information(self, verifier):
        """Test wrong-basis measurement gives no information."""
        result = verifier.verify_measurement_disturbance()

        for scenario_name, scenario_data in result['scenarios'].items():
            assert scenario_data['information_gained'] == 0, \
                f"{scenario_name} should give no information"

    # --- Complete Protocol Verification ---

    def test_verify_bb84_protocol_security(self, verifier):
        """Test complete BB84 protocol verification passes."""
        result = verifier.verify_bb84_protocol()
        assert result['security_verified'] is True

    def test_bb84_verification_returns_all_checks(self, verifier):
        """Test BB84 verification includes all security checks."""
        result = verifier.verify_bb84_protocol()

        verifications = result['verifications']
        assert 'orthogonality' in verifications
        assert 'mutual_unbiasedness' in verifications
        assert 'no_cloning' in verifications
        assert 'measurement_disturbance' in verifications

    def test_bb84_security_summary(self, verifier):
        """Test BB84 security summary is complete."""
        result = verifier.verify_bb84_protocol()

        summary = result['security_summary']
        assert summary['basis_encoding_reliable'] is True
        assert summary['bases_mutually_unbiased'] is True
        assert summary['no_cloning_protects'] is True
        assert summary['intercept_resend_qber'] == '25%'


class TestBB84ModuleFunctions:
    """Test standalone functions in bb84 module."""

    def test_get_bb84_states_symbolic(self):
        """Test symbolic state generation."""
        states = get_bb84_states_symbolic()
        assert len(states) == 4

    def test_verify_mutually_unbiased_bases_function(self):
        """Test standalone MUB verification."""
        result = verify_mutually_unbiased_bases()
        assert result['mutually_unbiased'] is True

    def test_verify_no_cloning_impossibility_function(self):
        """Test standalone no-cloning verification."""
        result = verify_no_cloning_impossibility()
        assert result['no_cloning_applies'] is True

    def test_verify_measurement_disturbance_function(self):
        """Test standalone measurement disturbance verification."""
        result = verify_measurement_disturbance()
        assert 'probabilities' in result
        assert result['information_gained'] == 0


class TestBB84Integration:
    """Integration tests for BB84 protocol."""

    def test_full_protocol_flow_no_eve(self):
        """Test complete protocol execution without Eve."""
        protocol = BB84Protocol(num_qubits=500)
        result = protocol.run_protocol(eve_intercept=False)

        # Protocol should succeed
        assert result.protocol_successful is True
        assert result.qber == 0.0
        assert result.final_key is not None
        assert len(result.final_key) > 0

    def test_full_protocol_flow_with_eve(self):
        """Test complete protocol execution with Eve."""
        protocol = BB84Protocol(num_qubits=1000)
        result = protocol.run_protocol(eve_intercept=True)

        # Protocol should detect Eve and fail
        assert result.protocol_successful is False
        assert result.eve_detected is True
        assert result.qber > 0.11  # Above threshold

    def test_symbolic_and_numerical_consistency(self):
        """Test symbolic verification matches numerical behavior."""
        # Symbolic verification
        verifier = QuantumVerifier()
        symbolic_result = verifier.verify_bb84_protocol()

        # Numerical protocol
        protocol = BB84Protocol(num_qubits=100)

        # Without Eve: QBER = 0 (symbolic says basis encoding is reliable)
        no_eve_result = protocol.run_protocol(eve_intercept=False)
        assert symbolic_result['security_summary']['basis_encoding_reliable'] is True
        assert no_eve_result.qber == 0.0

        # With Eve: QBER ≈ 25% (symbolic predicts intercept-resend causes 25%)
        with_eve_result = protocol.run_protocol(eve_intercept=True)
        assert symbolic_result['security_summary']['intercept_resend_qber'] == '25%'
        assert 0.15 < with_eve_result.qber < 0.35  # Around 25%
