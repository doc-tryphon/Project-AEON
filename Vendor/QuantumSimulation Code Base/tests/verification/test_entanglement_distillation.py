"""
TDD Test Suite for Entanglement Distillation (Step 9).

Tests verify:
1. Werner state model for noisy entanglement
2. BBPSSW distillation protocol
3. Fidelity improvement verification
4. Symbolic verification of distillation properties

Run with: pytest tests/verification/test_entanglement_distillation.py -v
"""

import pytest
import numpy as np
from sympy import sqrt, simplify, Rational, symbols, Matrix

import sys
sys.path.insert(0, 'src')

from quantum.entanglement_distillation import (
    WernerState, BBPSSWProtocol, DistillationResult,
    calculate_werner_fidelity, calculate_output_fidelity,
    distillation_threshold, get_bell_state_symbolic
)
from verification.symbolic_solver import QuantumVerifier


class TestWernerState:
    """Test Werner state model for noisy entanglement."""

    def test_werner_state_creation(self):
        """Test Werner state is created with specified fidelity."""
        werner = WernerState(fidelity=0.8)
        assert werner.fidelity == 0.8

    def test_werner_state_fidelity_bounds(self):
        """Test Werner state fidelity must be in [0.25, 1]."""
        # Valid fidelities
        WernerState(fidelity=0.25)  # Minimum (maximally mixed)
        WernerState(fidelity=0.5)
        WernerState(fidelity=1.0)  # Maximum (pure Bell state)

        # Invalid fidelities
        with pytest.raises(ValueError):
            WernerState(fidelity=0.2)  # Below minimum
        with pytest.raises(ValueError):
            WernerState(fidelity=1.1)  # Above maximum

    def test_werner_state_density_matrix_shape(self):
        """Test Werner state density matrix is 4x4."""
        werner = WernerState(fidelity=0.8)
        rho = werner.density_matrix
        assert rho.shape == (4, 4)

    def test_werner_state_trace_one(self):
        """Test Werner state has unit trace."""
        for F in [0.25, 0.5, 0.75, 1.0]:
            werner = WernerState(fidelity=F)
            trace = np.trace(werner.density_matrix)
            assert np.isclose(trace, 1.0), f"Trace not 1 for F={F}"

    def test_werner_state_positive_semidefinite(self):
        """Test Werner state is positive semidefinite."""
        for F in [0.25, 0.5, 0.75, 1.0]:
            werner = WernerState(fidelity=F)
            eigenvalues = np.linalg.eigvalsh(werner.density_matrix)
            assert all(ev >= -1e-10 for ev in eigenvalues), \
                f"Not positive semidefinite for F={F}"

    def test_werner_state_hermitian(self):
        """Test Werner state is Hermitian."""
        werner = WernerState(fidelity=0.8)
        rho = werner.density_matrix
        assert np.allclose(rho, rho.conj().T)

    def test_pure_bell_state_at_fidelity_one(self):
        """Test F=1 gives pure Bell state |Φ+⟩."""
        werner = WernerState(fidelity=1.0)
        rho = werner.density_matrix

        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        pure_state = np.outer(phi_plus, phi_plus.conj())

        assert np.allclose(rho, pure_state)

    def test_maximally_mixed_at_fidelity_quarter(self):
        """Test F=0.25 gives maximally mixed state I/4."""
        werner = WernerState(fidelity=0.25)
        rho = werner.density_matrix

        # Maximally mixed is I/4
        mixed = np.eye(4) / 4
        assert np.allclose(rho, mixed)

    def test_werner_state_formula(self):
        """Test Werner state formula: ρ = p|Φ+⟩⟨Φ+| + (1-p)/4 * I where p=(4F-1)/3."""
        F = 0.7
        werner = WernerState(fidelity=F)
        rho = werner.density_matrix

        # The Werner state is parameterized so that fidelity F = (3p+1)/4
        # Solving for p: p = (4F - 1) / 3
        p = (4 * F - 1) / 3

        # Construct expected state
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        pure_part = np.outer(phi_plus, phi_plus.conj())
        mixed_part = np.eye(4) / 4

        expected = p * pure_part + (1 - p) * mixed_part
        assert np.allclose(rho, expected)

    def test_werner_state_purity(self):
        """Test Werner state purity = Tr(ρ²)."""
        werner = WernerState(fidelity=0.8)
        rho = werner.density_matrix
        purity = np.trace(rho @ rho).real

        # For Werner state: purity = F² + (1-F)²/4 + 2F(1-F)/4
        # Simplified: (3F² - 2F + 1)/4 + F²/2... let's just verify 0 < purity ≤ 1
        assert 0 < purity <= 1


class TestBBPSSWProtocol:
    """Test BBPSSW entanglement distillation protocol."""

    @pytest.fixture
    def protocol(self):
        return BBPSSWProtocol()

    def test_protocol_exists(self, protocol):
        """Test BBPSSW protocol class exists."""
        assert protocol is not None

    def test_distill_pair_returns_result(self, protocol):
        """Test distillation returns a DistillationResult."""
        state1 = WernerState(fidelity=0.7)
        state2 = WernerState(fidelity=0.7)
        result = protocol.distill_pair(state1, state2)
        assert isinstance(result, DistillationResult)

    def test_result_contains_success_probability(self, protocol):
        """Test result includes success probability."""
        state1 = WernerState(fidelity=0.7)
        state2 = WernerState(fidelity=0.7)
        result = protocol.distill_pair(state1, state2)
        assert 0 <= result.success_probability <= 1

    def test_result_contains_output_fidelity(self, protocol):
        """Test result includes output fidelity."""
        state1 = WernerState(fidelity=0.7)
        state2 = WernerState(fidelity=0.7)
        result = protocol.distill_pair(state1, state2)
        assert 0.25 <= result.output_fidelity <= 1.0

    def test_fidelity_improvement_above_threshold(self, protocol):
        """Test fidelity improves when F > 0.5."""
        F_in = 0.7
        state1 = WernerState(fidelity=F_in)
        state2 = WernerState(fidelity=F_in)
        result = protocol.distill_pair(state1, state2)

        # Output fidelity should be higher than input
        assert result.output_fidelity > F_in, \
            f"Output {result.output_fidelity} not > input {F_in}"

    def test_no_improvement_below_threshold(self, protocol):
        """Test no fidelity improvement when F < 0.5."""
        F_in = 0.4
        state1 = WernerState(fidelity=F_in)
        state2 = WernerState(fidelity=F_in)
        result = protocol.distill_pair(state1, state2)

        # Output fidelity should NOT exceed 0.5
        assert result.output_fidelity <= 0.5

    def test_threshold_fidelity_is_half(self, protocol):
        """Test distillation threshold is F = 0.5."""
        assert distillation_threshold() == 0.5

    def test_output_fidelity_formula(self, protocol):
        """Test BBPSSW output fidelity formula: F' = (F² + (1-F)²/9) / (F² + 2F(1-F)/3 + 5(1-F)²/9)."""
        F_in = 0.7
        state1 = WernerState(fidelity=F_in)
        state2 = WernerState(fidelity=F_in)
        result = protocol.distill_pair(state1, state2)

        # Calculate expected output fidelity
        F = F_in
        numerator = F**2 + (1-F)**2 / 9
        denominator = F**2 + 2*F*(1-F)/3 + 5*(1-F)**2/9
        expected_F_out = numerator / denominator

        assert np.isclose(result.output_fidelity, expected_F_out, rtol=1e-6)

    def test_success_probability_formula(self, protocol):
        """Test BBPSSW success probability formula."""
        F_in = 0.7
        state1 = WernerState(fidelity=F_in)
        state2 = WernerState(fidelity=F_in)
        result = protocol.distill_pair(state1, state2)

        # Success probability = F² + 2F(1-F)/3 + 5(1-F)²/9
        F = F_in
        expected_p = F**2 + 2*F*(1-F)/3 + 5*(1-F)**2/9

        assert np.isclose(result.success_probability, expected_p, rtol=1e-6)

    def test_pure_state_stays_pure(self, protocol):
        """Test F=1 input stays at F=1 output."""
        state1 = WernerState(fidelity=1.0)
        state2 = WernerState(fidelity=1.0)
        result = protocol.distill_pair(state1, state2)

        assert np.isclose(result.output_fidelity, 1.0)
        assert np.isclose(result.success_probability, 1.0)


class TestIteratedDistillation:
    """Test multiple rounds of distillation."""

    @pytest.fixture
    def protocol(self):
        return BBPSSWProtocol()

    def test_iterated_distillation_increases_fidelity(self, protocol):
        """Test multiple rounds improve fidelity monotonically."""
        F = 0.7
        fidelities = [F]

        # Simulate 5 rounds (assuming perfect success each time)
        for _ in range(5):
            state1 = WernerState(fidelity=F)
            state2 = WernerState(fidelity=F)
            result = protocol.distill_pair(state1, state2)
            F = result.output_fidelity
            fidelities.append(F)

        # Fidelity should increase each round
        for i in range(len(fidelities) - 1):
            assert fidelities[i+1] >= fidelities[i], \
                f"Fidelity decreased at round {i+1}"

    def test_distillation_approaches_unity(self, protocol):
        """Test iterated distillation approaches F=1."""
        F = 0.7
        for _ in range(20):  # Many rounds
            state1 = WernerState(fidelity=F)
            state2 = WernerState(fidelity=F)
            result = protocol.distill_pair(state1, state2)
            F = result.output_fidelity

        assert F > 0.99, f"Fidelity should approach 1, got {F}"

    def test_distillation_count_to_target(self, protocol):
        """Test number of rounds needed to reach target fidelity."""
        initial_F = 0.6
        target_F = 0.95
        rounds = protocol.rounds_to_target(initial_F, target_F)

        assert rounds > 0
        assert isinstance(rounds, int)


class TestDistillationStatistics:
    """Test statistical simulation of distillation."""

    @pytest.fixture
    def protocol(self):
        return BBPSSWProtocol()

    def test_statistical_simulation(self, protocol):
        """Test statistical simulation of many distillation attempts."""
        F_in = 0.7
        num_pairs = 1000
        result = protocol.simulate_batch(F_in, num_pairs)

        assert 'successes' in result
        assert 'failures' in result
        assert 'average_output_fidelity' in result
        assert result['successes'] + result['failures'] == num_pairs

    def test_success_rate_matches_theory(self, protocol):
        """Test empirical success rate matches theoretical."""
        F_in = 0.7
        num_pairs = 10000
        result = protocol.simulate_batch(F_in, num_pairs)

        # Theoretical success probability
        F = F_in
        theoretical_p = F**2 + 2*F*(1-F)/3 + 5*(1-F)**2/9
        empirical_p = result['successes'] / num_pairs

        # Allow 5% relative error for statistical variation
        assert abs(empirical_p - theoretical_p) / theoretical_p < 0.05


class TestSymbolicDistillation:
    """Test symbolic verification of distillation."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_symbolic_bell_state(self):
        """Test symbolic Bell state |Φ+⟩ representation."""
        phi_plus = get_bell_state_symbolic('phi_plus')
        assert phi_plus is not None

    def test_symbolic_output_fidelity_formula(self, verifier):
        """Test symbolic derivation of output fidelity formula."""
        result = verifier.verify_distillation_formula()
        assert result['output_fidelity_verified'] is True

    def test_symbolic_threshold_derivation(self, verifier):
        """Test symbolic derivation that threshold is F=0.5."""
        result = verifier.verify_distillation_threshold()
        assert result['threshold'] == Rational(1, 2)
        assert result['threshold_verified'] is True

    def test_symbolic_success_probability(self, verifier):
        """Test symbolic derivation of success probability."""
        result = verifier.verify_distillation_success_probability()
        assert result['success_probability_verified'] is True

    def test_distillation_improvement_criterion(self, verifier):
        """Test symbolic proof that F' > F when F > 0.5."""
        result = verifier.verify_distillation_improvement()
        assert result['improvement_verified'] is True
        assert result['condition'] == 'F > 1/2'


class TestQuantumVerifierDistillation:
    """Test QuantumVerifier distillation methods."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_verify_distillation_protocol(self, verifier):
        """Test complete distillation protocol verification."""
        result = verifier.verify_distillation_protocol()
        assert result['protocol_verified'] is True

    def test_distillation_verifications_complete(self, verifier):
        """Test all distillation verifications are included."""
        result = verifier.verify_distillation_protocol()

        verifications = result['verifications']
        assert 'werner_state_properties' in verifications
        assert 'output_fidelity_formula' in verifications
        assert 'threshold_derivation' in verifications
        assert 'improvement_proof' in verifications

    def test_distillation_security_summary(self, verifier):
        """Test distillation verification summary."""
        result = verifier.verify_distillation_protocol()

        summary = result['summary']
        assert summary['threshold'] == '0.5'
        assert summary['protocol'] == 'BBPSSW'
        assert summary['fidelity_improvement'] is True


class TestModuleFunctions:
    """Test standalone functions in entanglement_distillation module."""

    def test_calculate_werner_fidelity(self):
        """Test Werner state fidelity calculation."""
        # For F=0.8 Werner state, fidelity with |Φ+⟩ is 0.8
        werner = WernerState(fidelity=0.8)
        fidelity = calculate_werner_fidelity(werner.density_matrix)
        assert np.isclose(fidelity, 0.8)

    def test_calculate_output_fidelity(self):
        """Test output fidelity calculation from input."""
        F_in = 0.7
        F_out = calculate_output_fidelity(F_in)

        # Expected from formula
        F = F_in
        numerator = F**2 + (1-F)**2 / 9
        denominator = F**2 + 2*F*(1-F)/3 + 5*(1-F)**2/9
        expected = numerator / denominator

        assert np.isclose(F_out, expected)


class TestDistillationIntegration:
    """Integration tests for distillation module."""

    def test_full_distillation_workflow(self):
        """Test complete distillation workflow."""
        # Create noisy states
        F_initial = 0.65
        state1 = WernerState(fidelity=F_initial)
        state2 = WernerState(fidelity=F_initial)

        # Run distillation
        protocol = BBPSSWProtocol()
        result = protocol.distill_pair(state1, state2)

        # Verify improvement
        assert result.output_fidelity > F_initial
        assert result.success_probability > 0

    def test_numerical_symbolic_consistency(self):
        """Test numerical implementation matches symbolic derivation."""
        verifier = QuantumVerifier()
        symbolic_result = verifier.verify_distillation_formula()

        # Numerical test at F = 0.75
        F = 0.75
        numerical_out = calculate_output_fidelity(F)

        # Symbolic formula evaluated
        # F' = (F² + (1-F)²/9) / (F² + 2F(1-F)/3 + 5(1-F)²/9)
        numerator = F**2 + (1-F)**2 / 9
        denominator = F**2 + 2*F*(1-F)/3 + 5*(1-F)**2/9
        symbolic_out = numerator / denominator

        assert np.isclose(numerical_out, symbolic_out)
        assert symbolic_result['output_fidelity_verified'] is True

    def test_distillation_with_entanglement_module(self):
        """Test distillation works with existing entanglement module."""
        from quantum.entanglement import BellStateGenerator

        # Get Bell state from entanglement module
        generator = BellStateGenerator()
        phi_plus = generator.create_bell_state('phi_plus')

        # Create Werner state at F=1 (should match pure Bell state)
        werner = WernerState(fidelity=1.0)

        # Verify they represent the same state
        assert np.allclose(
            np.outer(phi_plus.state_vector, phi_plus.state_vector.conj()),
            werner.density_matrix
        )
