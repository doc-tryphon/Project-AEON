"""
Verification tests for 3-qubit bit flip error correction code.

This test suite verifies:
1. Encoding preserves quantum state coefficients
2. Syndrome measurement correctly identifies errors
3. Correction achieves perfect fidelity (F = 1.0)
4. Full cycle returns original state
5. Logical error rate matches analytical formula p_L ≈ 3p²
6. Code space properties
7. Break-even threshold p < 1/3

Test tolerance: 1e-10 (10 decimal places)
"""

import pytest
import numpy as np
from src.quantum.error_correction import BitFlipCode, LogicalErrorRate, ErrorCorrectionResult


class TestBitFlipEncoding:
    """Test encoding circuit implementation."""

    def test_encoding_basis_state_zero(self):
        """Test encoding |0⟩ → |000⟩."""
        code = BitFlipCode()
        zero = np.array([1, 0], dtype=np.complex128)

        encoded = code.encode(zero)

        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)
        assert np.allclose(encoded, expected, atol=1e-10)

    def test_encoding_basis_state_one(self):
        """Test encoding |1⟩ → |111⟩."""
        code = BitFlipCode()
        one = np.array([0, 1], dtype=np.complex128)

        encoded = code.encode(one)

        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)
        assert np.allclose(encoded, expected, atol=1e-10)

    def test_encoding_plus_state(self):
        """Test encoding |+⟩ → (|000⟩ + |111⟩)/√2."""
        code = BitFlipCode()
        plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)

        encoded = code.encode(plus)

        expected = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        assert np.allclose(encoded, expected, atol=1e-10)

    def test_encoding_minus_state(self):
        """Test encoding |-⟩ → (|000⟩ - |111⟩)/√2."""
        code = BitFlipCode()
        minus = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)

        encoded = code.encode(minus)

        expected = np.array([1, 0, 0, 0, 0, 0, 0, -1], dtype=np.complex128) / np.sqrt(2)
        assert np.allclose(encoded, expected, atol=1e-10)

    def test_encoding_arbitrary_superposition(self):
        """Test encoding arbitrary state α|0⟩ + β|1⟩."""
        code = BitFlipCode()

        # α = 0.6, β = 0.8
        alpha = 0.6
        beta = 0.8
        state = np.array([alpha, beta], dtype=np.complex128)

        encoded = code.encode(state)

        # Expected: α|000⟩ + β|111⟩
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = alpha  # |000⟩
        expected[7] = beta   # |111⟩

        assert np.allclose(encoded, expected, atol=1e-10)

    def test_encoding_preserves_normalization(self):
        """Test that encoding preserves state normalization."""
        code = BitFlipCode()

        # Random normalized state
        state = np.array([0.8, 0.6], dtype=np.complex128)
        state = state / np.linalg.norm(state)

        encoded = code.encode(state)

        assert np.isclose(np.linalg.norm(encoded), 1.0, atol=1e-10)

    def test_encoding_error_handling(self):
        """Test encoding rejects invalid inputs."""
        code = BitFlipCode()

        # Wrong dimension
        with pytest.raises(ValueError, match="single-qubit"):
            code.encode(np.array([1, 0, 0, 0], dtype=np.complex128))

        # Unnormalized state
        with pytest.raises(ValueError, match="normalized"):
            code.encode(np.array([2, 0], dtype=np.complex128))


class TestBitFlipDecoding:
    """Test decoding circuit implementation."""

    def test_decoding_logical_zero(self):
        """Test decoding |000⟩ → |0⟩."""
        code = BitFlipCode()
        logical_zero = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)

        decoded = code.decode(logical_zero)

        expected = np.array([1, 0], dtype=np.complex128)
        assert np.allclose(decoded, expected, atol=1e-10)

    def test_decoding_logical_one(self):
        """Test decoding |111⟩ → |1⟩."""
        code = BitFlipCode()
        logical_one = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)

        decoded = code.decode(logical_one)

        expected = np.array([0, 1], dtype=np.complex128)
        assert np.allclose(decoded, expected, atol=1e-10)

    def test_decoding_logical_plus(self):
        """Test decoding (|000⟩ + |111⟩)/√2 → |+⟩."""
        code = BitFlipCode()
        logical_plus = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

        decoded = code.decode(logical_plus)

        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        assert np.allclose(decoded, expected, atol=1e-10)

    def test_encode_decode_cycle(self):
        """Test that decode(encode(|ψ⟩)) = |ψ⟩."""
        code = BitFlipCode()

        # Test with arbitrary state
        state = np.array([0.6, 0.8], dtype=np.complex128)

        encoded = code.encode(state)
        decoded = code.decode(encoded)

        assert np.allclose(decoded, state, atol=1e-10)


class TestErrorInjection:
    """Test error injection mechanism."""

    def test_inject_error_qubit_0(self):
        """Test bit flip on qubit 0."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩

        corrupted = code.inject_error(state, error_qubit=0)

        expected = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.complex128)  # |100⟩
        assert np.allclose(corrupted, expected, atol=1e-10)

    def test_inject_error_qubit_1(self):
        """Test bit flip on qubit 1."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩

        corrupted = code.inject_error(state, error_qubit=1)

        expected = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.complex128)  # |010⟩
        assert np.allclose(corrupted, expected, atol=1e-10)

    def test_inject_error_qubit_2(self):
        """Test bit flip on qubit 2."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩

        corrupted = code.inject_error(state, error_qubit=2)

        expected = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |001⟩
        assert np.allclose(corrupted, expected, atol=1e-10)

    def test_inject_error_invalid_qubit(self):
        """Test error handling for invalid qubit index."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)

        with pytest.raises(ValueError, match="must be 0, 1, or 2"):
            code.inject_error(state, error_qubit=3)


class TestSyndromeMeasurement:
    """Test syndrome measurement implementation."""

    def test_syndrome_no_error(self):
        """Test syndrome (0,0) for no error."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩

        syndrome = code.measure_syndrome(state)

        assert syndrome == (0, 0)

    def test_syndrome_error_qubit_0(self):
        """Test syndrome (1,0) for error on qubit 0."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=0)  # → |100⟩

        syndrome = code.measure_syndrome(corrupted)

        assert syndrome == (1, 0)

    def test_syndrome_error_qubit_1(self):
        """Test syndrome (1,1) for error on qubit 1."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=1)  # → |010⟩

        syndrome = code.measure_syndrome(corrupted)

        assert syndrome == (1, 1)

    def test_syndrome_error_qubit_2(self):
        """Test syndrome (0,1) for error on qubit 2."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=2)  # → |001⟩

        syndrome = code.measure_syndrome(corrupted)

        assert syndrome == (0, 1)

    def test_syndrome_logical_one_no_error(self):
        """Test syndrome (0,0) for |111⟩ (no error)."""
        code = BitFlipCode()
        state = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)  # |111⟩

        syndrome = code.measure_syndrome(state)

        assert syndrome == (0, 0)

    def test_syndrome_logical_one_with_error(self):
        """Test syndrome detection on |111⟩ with error."""
        code = BitFlipCode()
        state = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)  # |111⟩
        corrupted = code.inject_error(state, error_qubit=0)  # → |011⟩

        syndrome = code.measure_syndrome(corrupted)

        assert syndrome == (1, 0)  # Same syndrome as |100⟩


class TestErrorCorrection:
    """Test error correction implementation."""

    def test_correction_no_error(self):
        """Test correction with syndrome (0,0) does nothing."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)

        corrected = code.apply_correction(state, syndrome=(0, 0))

        assert np.allclose(corrected, state, atol=1e-10)

    def test_correction_qubit_0(self):
        """Test correction of error on qubit 0."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=0)  # → |100⟩

        corrected = code.apply_correction(corrupted, syndrome=(1, 0))

        # Should return to |000⟩
        assert np.allclose(corrected, state, atol=1e-10)

    def test_correction_qubit_1(self):
        """Test correction of error on qubit 1."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=1)  # → |010⟩

        corrected = code.apply_correction(corrupted, syndrome=(1, 1))

        assert np.allclose(corrected, state, atol=1e-10)

    def test_correction_qubit_2(self):
        """Test correction of error on qubit 2."""
        code = BitFlipCode()
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128)  # |000⟩
        corrupted = code.inject_error(state, error_qubit=2)  # → |001⟩

        corrected = code.apply_correction(corrupted, syndrome=(0, 1))

        assert np.allclose(corrected, state, atol=1e-10)

    def test_correction_achieves_perfect_fidelity(self):
        """Test that correction achieves F = 1.0 for single errors."""
        code = BitFlipCode()

        # Test with arbitrary superposition
        state = np.array([0.6, 0.8], dtype=np.complex128)
        encoded = code.encode(state)

        # Test correction for each possible single error
        for error_qubit in [0, 1, 2]:
            corrupted = code.inject_error(encoded, error_qubit)
            syndrome = code.measure_syndrome(corrupted)
            corrected = code.apply_correction(corrupted, syndrome)
            decoded = code.decode(corrected)

            fidelity = np.abs(np.conj(state) @ decoded)**2
            assert np.isclose(fidelity, 1.0, atol=1e-10)


class TestFullCycle:
    """Test complete error correction cycle."""

    def test_full_cycle_no_error(self):
        """Test full cycle with no error."""
        code = BitFlipCode()
        state = np.array([0.6, 0.8], dtype=np.complex128)

        result = code.full_cycle(state, error_qubit=None)

        assert result.success
        assert np.isclose(result.fidelity, 1.0, atol=1e-10)
        assert result.syndrome == (0, 0)

    def test_full_cycle_error_qubit_0(self):
        """Test full cycle with error on qubit 0."""
        code = BitFlipCode()
        state = np.array([0.6, 0.8], dtype=np.complex128)

        result = code.full_cycle(state, error_qubit=0)

        assert result.success
        assert np.isclose(result.fidelity, 1.0, atol=1e-10)
        assert result.syndrome == (1, 0)
        assert result.error_applied == 0

    def test_full_cycle_error_qubit_1(self):
        """Test full cycle with error on qubit 1."""
        code = BitFlipCode()
        state = np.array([0.6, 0.8], dtype=np.complex128)

        result = code.full_cycle(state, error_qubit=1)

        assert result.success
        assert np.isclose(result.fidelity, 1.0, atol=1e-10)
        assert result.syndrome == (1, 1)
        assert result.error_applied == 1

    def test_full_cycle_error_qubit_2(self):
        """Test full cycle with error on qubit 2."""
        code = BitFlipCode()
        state = np.array([0.6, 0.8], dtype=np.complex128)

        result = code.full_cycle(state, error_qubit=2)

        assert result.success
        assert np.isclose(result.fidelity, 1.0, atol=1e-10)
        assert result.syndrome == (0, 1)
        assert result.error_applied == 2

    def test_full_cycle_basis_states(self):
        """Test full cycle with all computational basis states."""
        code = BitFlipCode()

        for basis_state in [np.array([1, 0], dtype=np.complex128),
                           np.array([0, 1], dtype=np.complex128)]:
            for error_qubit in [0, 1, 2]:
                result = code.full_cycle(basis_state, error_qubit=error_qubit)
                assert result.success
                assert np.isclose(result.fidelity, 1.0, atol=1e-10)

    def test_full_cycle_superposition_states(self):
        """Test full cycle with superposition states."""
        code = BitFlipCode()

        # Test |+⟩ and |-⟩
        plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        minus = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)

        for state in [plus, minus]:
            for error_qubit in [0, 1, 2]:
                result = code.full_cycle(state, error_qubit=error_qubit)
                assert result.success
                assert np.isclose(result.fidelity, 1.0, atol=1e-10)


class TestCodeSpace:
    """Test code space properties."""

    def test_code_space_properties(self):
        """Test that code space has correct properties."""
        code = BitFlipCode()
        properties = code.verify_code_space()

        assert properties['logical_basis_normalized']
        assert properties['logical_basis_orthogonal']
        assert properties['distance'] == 3

    def test_stabilizer_eigenvalues(self):
        """Test stabilizer eigenvalues for code space."""
        code = BitFlipCode()
        properties = code.verify_code_space()

        # Both logical basis states should be +1 eigenstates of stabilizers
        for state_name in ['|000⟩', '|111⟩']:
            eigenvals = properties['stabilizer_eigenvalues'][state_name]
            assert np.isclose(eigenvals['S1'], 1.0, atol=1e-10)
            assert np.isclose(eigenvals['S2'], 1.0, atol=1e-10)
            assert eigenvals['in_code_space']


class TestLogicalErrorRate:
    """Test logical error rate calculations."""

    def test_logical_error_rate_formula(self):
        """Test p_L = 3p² - 2p³ formula."""
        # Test at specific values
        test_cases = [
            (0.01, 3 * 0.01**2 - 2 * 0.01**3),
            (0.05, 3 * 0.05**2 - 2 * 0.05**3),
            (0.10, 3 * 0.10**2 - 2 * 0.10**3),
        ]

        for p, expected_p_L in test_cases:
            p_L = LogicalErrorRate.calculate_logical_error_rate(p)
            assert np.isclose(p_L, expected_p_L, atol=1e-10)

    def test_logical_error_rate_small_p_approximation(self):
        """Test that p_L ≈ 3p² for small p."""
        p = 0.01
        p_L = LogicalErrorRate.calculate_logical_error_rate(p)
        p_L_approx = 3 * p**2

        # Relative error should be small
        relative_error = abs(p_L - p_L_approx) / p_L
        assert relative_error < 0.01  # Less than 1% error

    def test_improvement_factor(self):
        """Test improvement factor p_L / p."""
        # For p = 0.01, expect p_L/p ≈ 0.03
        p = 0.01
        factor = LogicalErrorRate.improvement_factor(p)

        expected = 3 * p - 2 * p**2  # p_L/p = 3p - 2p²
        assert np.isclose(factor, expected, atol=1e-10)

    def test_break_even_threshold(self):
        """Test break-even threshold is 1/2."""
        threshold = LogicalErrorRate.break_even_threshold()
        assert np.isclose(threshold, 1.0/2.0, atol=1e-10)

        # At threshold, p_L = p
        p_L = LogicalErrorRate.calculate_logical_error_rate(threshold)
        assert np.isclose(p_L, threshold, atol=1e-10)

    def test_error_suppression_regime(self):
        """Test that code suppresses errors for p < 1/2."""
        test_rates = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

        for p in test_rates:
            p_L = LogicalErrorRate.calculate_logical_error_rate(p)
            if p < 1.0/2.0:
                assert p_L < p  # Code helps
            elif p == 1.0/2.0:
                assert np.isclose(p_L, p, atol=1e-10)  # Break-even
            else:
                assert p_L > p  # Code doesn't help

    def test_performance_analysis(self):
        """Test performance analysis across multiple error rates."""
        physical_rates = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        analysis = LogicalErrorRate.analyze_performance(physical_rates)

        assert len(analysis['physical_rates']) == len(physical_rates)
        assert len(analysis['logical_rates']) == len(physical_rates)

        # Verify suppression for p < 1/2
        for i, p in enumerate(physical_rates):
            p_L = analysis['logical_rates'][i]
            suppression = analysis['suppression_achieved'][i]

            if p < 1.0/2.0:
                assert suppression
                assert p_L < p
            else:
                assert not suppression

    def test_pseudo_threshold(self):
        """Test pseudo-threshold calculation."""
        threshold_info = LogicalErrorRate.pseudo_threshold()

        p_star = threshold_info['pseudo_threshold']
        expected_p_star = (3 - np.sqrt(3)) / 6

        assert np.isclose(p_star, expected_p_star, atol=1e-10)

        # Verify derivative condition: d(p_L)/dp = 1
        # d(p_L)/dp = 6p - 6p²
        derivative = 6 * p_star - 6 * p_star**2
        assert np.isclose(derivative, 1.0, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_physical_error_rate(self):
        """Test with zero physical error rate."""
        p_L = LogicalErrorRate.calculate_logical_error_rate(0.0)
        assert np.isclose(p_L, 0.0, atol=1e-10)

    def test_unit_physical_error_rate(self):
        """Test with p = 1 (maximally noisy)."""
        p_L = LogicalErrorRate.calculate_logical_error_rate(1.0)
        expected = 3 * 1.0**2 - 2 * 1.0**3  # = 1
        assert np.isclose(p_L, expected, atol=1e-10)

    def test_encoding_with_complex_coefficients(self):
        """Test encoding with complex superposition."""
        code = BitFlipCode()

        # State with complex coefficients
        state = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)

        encoded = code.encode(state)

        # Verify normalization
        assert np.isclose(np.linalg.norm(encoded), 1.0, atol=1e-10)

        # Verify encoding structure
        assert np.isclose(encoded[0], 1/np.sqrt(2), atol=1e-10)  # α|000⟩
        assert np.isclose(encoded[7], 1j/np.sqrt(2), atol=1e-10)  # β|111⟩

    def test_multiple_errors_not_correctable(self):
        """Test that two errors cannot be corrected."""
        code = BitFlipCode()
        state = np.array([1, 0], dtype=np.complex128)
        encoded = code.encode(state)

        # Apply two errors
        corrupted = code.inject_error(encoded, error_qubit=0)
        corrupted = code.inject_error(corrupted, error_qubit=1)

        # Measure syndrome and attempt correction
        syndrome = code.measure_syndrome(corrupted)
        corrected = code.apply_correction(corrupted, syndrome)
        decoded = code.decode(corrected)

        # Fidelity should not be 1.0 (errors not correctable)
        fidelity = np.abs(np.conj(state) @ decoded)**2
        assert fidelity < 0.999  # Not perfect correction