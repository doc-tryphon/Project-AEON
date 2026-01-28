"""
TDD Test Suite for Grover's Search Algorithm (Step 11).

Tests verify:
1. Phase oracle construction (flips target state phase)
2. Diffusion operator construction (inversion about mean)
3. Optimal iteration count (π/4 * √N)
4. Quadratic speedup O(√N) vs classical O(N)
5. Amplitude amplification mechanism
6. Probabilistic success with high confidence
7. Over-iteration detection

Run with: pytest tests/verification/test_grover.py -v
"""

import pytest
import numpy as np
from sympy import sqrt, simplify, Rational, symbols, Matrix, I, pi, floor

import sys
sys.path.insert(0, 'src')

from algorithms.grover import (
    GroversAlgorithm, GroverResult,
    construct_oracle, construct_diffuser,
    calculate_optimal_iterations, calculate_success_probability
)
from verification.symbolic_solver import QuantumVerifier


class TestOracleConstruction:
    """Test phase oracle construction for Grover's algorithm."""

    def test_oracle_exists_for_target_zero(self):
        """Test oracle can be created for target index 0."""
        oracle = construct_oracle(n_qubits=2, target=0)
        assert oracle is not None

    def test_oracle_has_correct_dimension(self):
        """Test oracle matrix has dimension 2^n × 2^n."""
        for n in [1, 2, 3]:
            oracle = construct_oracle(n_qubits=n, target=0)
            expected_dim = 2**n
            assert oracle.shape == (expected_dim, expected_dim)

    def test_oracle_is_unitary(self):
        """Test oracle is unitary (U†U = I)."""
        for n in [1, 2, 3]:
            for target in [0, 1, 2**(n-1)]:
                oracle = construct_oracle(n_qubits=n, target=target)
                product = oracle.conj().T @ oracle
                assert np.allclose(product, np.eye(2**n)), \
                    f"Oracle not unitary for n={n}, target={target}"

    def test_oracle_is_hermitian(self):
        """Test oracle is Hermitian (U = U†) since it's diagonal."""
        for n in [1, 2, 3]:
            oracle = construct_oracle(n_qubits=n, target=0)
            assert np.allclose(oracle, oracle.conj().T)

    def test_oracle_is_diagonal(self):
        """Test oracle is diagonal matrix."""
        oracle = construct_oracle(n_qubits=2, target=1)
        # Check all off-diagonal elements are zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert np.abs(oracle[i, j]) < 1e-10

    def test_oracle_flips_target_phase(self):
        """Test oracle has -1 at target index, +1 elsewhere."""
        for n in [2, 3]:
            N = 2**n
            for target in [0, 1, N//2, N-1]:
                oracle = construct_oracle(n_qubits=n, target=target)
                # Check diagonal elements
                for i in range(N):
                    if i == target:
                        assert np.isclose(oracle[i, i], -1.0), \
                            f"Oracle should have -1 at target={target}"
                    else:
                        assert np.isclose(oracle[i, i], 1.0), \
                            f"Oracle should have +1 at non-target index {i}"

    def test_oracle_single_qubit(self):
        """Test oracle for n=1 (searching in 2 items)."""
        # Target 0: flip |0⟩
        oracle_0 = construct_oracle(n_qubits=1, target=0)
        expected_0 = np.diag([-1, 1])
        assert np.allclose(oracle_0, expected_0)

        # Target 1: flip |1⟩
        oracle_1 = construct_oracle(n_qubits=1, target=1)
        expected_1 = np.diag([1, -1])
        assert np.allclose(oracle_1, expected_1)

    def test_oracle_applies_phase_flip(self):
        """Test oracle flips phase of target state in superposition."""
        n = 2
        target = 2
        oracle = construct_oracle(n_qubits=n, target=target)

        # Create uniform superposition
        state = np.ones(4, dtype=complex) / 2.0  # |00⟩ + |01⟩ + |10⟩ + |11⟩

        result = oracle @ state

        # Only amplitude of |10⟩ (index 2) should be negated
        expected = np.array([1, 1, -1, 1], dtype=complex) / 2.0
        assert np.allclose(result, expected)


class TestDiffuserConstruction:
    """Test diffusion operator construction."""

    def test_diffuser_exists(self):
        """Test diffuser can be created."""
        diffuser = construct_diffuser(n_qubits=2)
        assert diffuser is not None

    def test_diffuser_has_correct_dimension(self):
        """Test diffuser matrix has dimension 2^n × 2^n."""
        for n in [1, 2, 3]:
            diffuser = construct_diffuser(n_qubits=n)
            expected_dim = 2**n
            assert diffuser.shape == (expected_dim, expected_dim)

    def test_diffuser_is_unitary(self):
        """Test diffuser is unitary."""
        for n in [1, 2, 3]:
            diffuser = construct_diffuser(n_qubits=n)
            product = diffuser.conj().T @ diffuser
            assert np.allclose(product, np.eye(2**n)), \
                f"Diffuser not unitary for n={n}"

    def test_diffuser_is_hermitian(self):
        """Test diffuser is Hermitian."""
        for n in [1, 2, 3]:
            diffuser = construct_diffuser(n_qubits=n)
            assert np.allclose(diffuser, diffuser.conj().T)

    def test_diffuser_formula_two_qubit(self):
        """Test diffuser matches 2|s⟩⟨s| - I for n=2."""
        n = 2
        N = 2**n
        diffuser = construct_diffuser(n_qubits=n)

        # |s⟩ = H^⊗n|0⟩ = uniform superposition
        s = np.ones(N) / np.sqrt(N)

        # Expected: 2|s⟩⟨s| - I
        expected = 2 * np.outer(s, s) - np.eye(N)

        assert np.allclose(diffuser, expected)

    def test_diffuser_inverts_about_mean(self):
        """Test diffuser performs inversion about mean."""
        n = 2
        diffuser = construct_diffuser(n_qubits=n)

        # Test with a simple state
        state = np.array([0.5, 0.5, -0.5, 0.5])
        result = diffuser @ state

        # Mean amplitude: (0.5 + 0.5 - 0.5 + 0.5) / 4 = 0.25
        # Inversion: 2*mean - original
        # [2*0.25 - 0.5, 2*0.25 - 0.5, 2*0.25 - (-0.5), 2*0.25 - 0.5]
        # = [0, 0, 1, 0]
        expected = np.array([0.0, 0.0, 1.0, 0.0])
        assert np.allclose(result, expected)

    def test_diffuser_single_qubit(self):
        """Test diffuser for n=1."""
        diffuser = construct_diffuser(n_qubits=1)

        # For n=1: |s⟩ = (|0⟩ + |1⟩)/√2
        s = np.array([1, 1]) / np.sqrt(2)
        expected = 2 * np.outer(s, s) - np.eye(2)

        assert np.allclose(diffuser, expected)


class TestOptimalIterations:
    """Test optimal iteration count calculation."""

    def test_iteration_count_for_n2(self):
        """Test iteration count for N=4 (n=2 qubits)."""
        k = calculate_optimal_iterations(n_qubits=2)
        # N = 4, optimal ≈ π/4 * √4 = π/4 * 2 ≈ 1.57 → 1 iteration
        assert k == 1

    def test_iteration_count_for_n3(self):
        """Test iteration count for N=8 (n=3 qubits)."""
        k = calculate_optimal_iterations(n_qubits=3)
        # N = 8, optimal ≈ π/4 * √8 ≈ 2.22 → 2 iterations
        assert k == 2

    def test_iteration_count_for_n4(self):
        """Test iteration count for N=16 (n=4 qubits)."""
        k = calculate_optimal_iterations(n_qubits=4)
        # N = 16, optimal ≈ π/4 * √16 = π/4 * 4 ≈ 3.14 → 3 iterations
        assert k == 3

    def test_iteration_count_scales_as_sqrt_n(self):
        """Test iteration count grows as O(√N)."""
        # For increasing n, iterations should grow as √(2^n)
        iterations = [calculate_optimal_iterations(n_qubits=n) for n in range(2, 8)]

        # Check they're increasing
        for i in range(len(iterations) - 1):
            assert iterations[i+1] > iterations[i]

    def test_iteration_formula(self):
        """Test iteration count matches floor(π/4 * √N)."""
        for n in [2, 3, 4, 5]:
            N = 2**n
            expected = int(np.floor(np.pi / 4 * np.sqrt(N)))
            actual = calculate_optimal_iterations(n_qubits=n)
            assert actual == expected


class TestGroverAlgorithmSmallCases:
    """Test Grover's algorithm on small databases."""

    @pytest.fixture
    def algorithm_n2(self):
        return GroversAlgorithm(n_qubits=2)

    def test_search_in_four_items_target_zero(self, algorithm_n2):
        """Test searching for index 0 in database of 4 items."""
        result = algorithm_n2.run(target=0)
        assert result.found_state == 0
        assert result.n_qubits == 2
        assert result.iterations_used > 0

    def test_search_in_four_items_target_three(self, algorithm_n2):
        """Test searching for index 3 in database of 4 items."""
        result = algorithm_n2.run(target=3)
        assert result.found_state == 3

    def test_search_all_targets_in_four_items(self, algorithm_n2):
        """Test finding each item in database of 4."""
        for target in [0, 1, 2, 3]:
            result = algorithm_n2.run(target=target)
            assert result.found_state == target, \
                f"Failed to find target {target}"

    def test_search_in_eight_items(self):
        """Test searching in database of 8 items."""
        algorithm = GroversAlgorithm(n_qubits=3)
        for target in [0, 3, 7]:
            result = algorithm.run(target=target)
            # Verify high success probability (probabilistic algorithm)
            assert result.success_probability > 0.85, \
                f"Low success probability {result.success_probability} for target {target}"

    def test_search_in_sixteen_items(self):
        """Test searching in database of 16 items."""
        algorithm = GroversAlgorithm(n_qubits=4)
        for target in [0, 5, 10, 15]:
            result = algorithm.run(target=target)
            # Verify high success probability (probabilistic algorithm)
            assert result.success_probability > 0.85, \
                f"Low success probability {result.success_probability} for target {target}"


class TestGroverResult:
    """Test GroverResult structure."""

    def test_result_contains_found_state(self):
        """Test result includes found state index."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=2)
        assert hasattr(result, 'found_state')
        assert result.found_state == 2

    def test_result_contains_iterations(self):
        """Test result includes iteration count."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=1)
        assert hasattr(result, 'iterations_used')

    def test_result_contains_success_probability(self):
        """Test result includes success probability."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=0)
        assert hasattr(result, 'success_probability')
        assert 0 <= result.success_probability <= 1

    def test_result_contains_final_state(self):
        """Test result includes final quantum state."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=1)
        assert hasattr(result, 'final_state')
        assert result.final_state is not None


class TestAmplitudeAmplification:
    """Test amplitude amplification mechanism."""

    def test_amplitude_increases_with_iterations(self):
        """Test target amplitude grows with each Grover iteration."""
        algorithm = GroversAlgorithm(n_qubits=3)
        target = 5

        # Track amplitude after each iteration
        amplitudes = algorithm.get_amplitude_evolution(target=target, max_iterations=4)

        # Amplitude should increase initially
        assert abs(amplitudes[1]) > abs(amplitudes[0])
        assert abs(amplitudes[2]) > abs(amplitudes[1])

    def test_probability_peaks_at_optimal_iterations(self):
        """Test success probability peaks near optimal iteration count."""
        algorithm = GroversAlgorithm(n_qubits=3)
        target = 2
        optimal = calculate_optimal_iterations(n_qubits=3)

        probabilities = []
        for k in range(1, 6):
            prob = algorithm.get_probability_after_k_iterations(target=target, iterations=k)
            probabilities.append(prob)

        # Probability should peak around optimal
        peak_index = probabilities.index(max(probabilities))
        assert abs(peak_index + 1 - optimal) <= 1, \
            f"Peak at iteration {peak_index+1}, expected near {optimal}"

    def test_over_iteration_decreases_probability(self):
        """Test over-iteration causes probability to decrease."""
        algorithm = GroversAlgorithm(n_qubits=3)
        target = 4
        optimal = calculate_optimal_iterations(n_qubits=3)

        prob_optimal = algorithm.get_probability_after_k_iterations(
            target=target, iterations=optimal
        )
        prob_double = algorithm.get_probability_after_k_iterations(
            target=target, iterations=2*optimal
        )

        # Doubling iterations should decrease probability
        assert prob_double < prob_optimal


class TestSuccessProbability:
    """Test probabilistic success of Grover's algorithm."""

    def test_high_success_probability_n2(self):
        """Test n=2 achieves high success probability."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=1)
        # For N=4, one iteration gives very high probability
        assert result.success_probability > 0.95

    def test_high_success_probability_n3(self):
        """Test n=3 achieves high success probability."""
        algorithm = GroversAlgorithm(n_qubits=3)
        result = algorithm.run(target=5)
        assert result.success_probability > 0.90

    def test_success_probability_formula(self):
        """Test success probability matches sin²((2k+1)θ) formula."""
        # For N items, θ = arcsin(1/√N)
        # After k iterations: P = sin²((2k+1)θ)
        n = 3
        N = 2**n
        target = 3
        k = 2

        algorithm = GroversAlgorithm(n_qubits=n)
        prob = algorithm.get_probability_after_k_iterations(target=target, iterations=k)

        # Expected probability
        theta = np.arcsin(1 / np.sqrt(N))
        expected_prob = np.sin((2*k + 1) * theta)**2

        assert np.isclose(prob, expected_prob, atol=1e-10)

    def test_initial_probability_is_one_over_n(self):
        """Test initial probability (k=0) is 1/N."""
        for n in [2, 3, 4]:
            N = 2**n
            algorithm = GroversAlgorithm(n_qubits=n)

            prob = algorithm.get_probability_after_k_iterations(target=0, iterations=0)
            expected = 1.0 / N

            assert np.isclose(prob, expected, atol=1e-10)


class TestQuadraticSpeedup:
    """Test quadratic speedup over classical search."""

    def test_grover_iterations_vs_classical(self):
        """Test Grover uses O(√N) vs classical O(N)."""
        for n in [2, 3, 4, 5, 6]:
            N = 2**n
            grover_iterations = calculate_optimal_iterations(n_qubits=n)
            classical_worst_case = N  # Need to check all items in worst case

            # Grover should be significantly fewer iterations
            assert grover_iterations < classical_worst_case

    def test_quadratic_advantage_grows(self):
        """Test advantage grows as database size increases."""
        speedups = []
        for n in [3, 4, 5, 6]:
            N = 2**n
            grover = calculate_optimal_iterations(n_qubits=n)
            classical = N / 2  # Average case for classical
            speedup = classical / grover
            speedups.append(speedup)

        # Speedup should increase with N
        for i in range(len(speedups) - 1):
            assert speedups[i+1] > speedups[i]


class TestGroverSymbolicVerification:
    """Test symbolic verification of Grover's algorithm."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_verify_oracle_properties(self, verifier):
        """Test symbolic verification of oracle properties."""
        result = verifier.verify_grover_oracle()
        assert result['oracle_unitary'] is True
        assert result['oracle_hermitian'] is True
        assert result['phase_flip_verified'] is True

    def test_verify_diffuser_properties(self, verifier):
        """Test symbolic verification of diffuser properties."""
        result = verifier.verify_grover_diffuser()
        assert result['diffuser_unitary'] is True
        assert result['diffuser_hermitian'] is True
        assert result['inversion_about_mean'] is True

    def test_verify_amplitude_amplification(self, verifier):
        """Test symbolic verification of amplitude amplification."""
        result = verifier.verify_amplitude_amplification()
        assert result['amplitude_increases'] is True

    def test_verify_optimal_iteration_formula(self, verifier):
        """Test symbolic verification of iteration formula."""
        result = verifier.verify_grover_iteration_formula()
        assert result['formula_verified'] is True
        assert result['formula'] == 'floor(π/4 * √N)'

    def test_verify_success_probability_formula(self, verifier):
        """Test symbolic verification of success probability."""
        result = verifier.verify_grover_success_probability()
        assert result['probability_formula_verified'] is True
        assert 'sin²((2k+1)θ)' in result['formula']

    def test_full_grover_protocol_verification(self, verifier):
        """Test complete Grover protocol verification."""
        result = verifier.verify_grover_protocol()
        assert result['protocol_verified'] is True

        checks = result['verifications']
        assert 'oracle' in checks
        assert 'diffuser' in checks
        assert 'amplitude_amplification' in checks
        assert 'iteration_formula' in checks
        assert 'success_probability' in checks


class TestGroverIntegration:
    """Integration tests for Grover's algorithm."""

    def test_full_algorithm_execution_n2(self):
        """Test complete algorithm for n=2."""
        algorithm = GroversAlgorithm(n_qubits=2)
        result = algorithm.run(target=2)

        assert result.found_state == 2
        assert result.iterations_used == calculate_optimal_iterations(n_qubits=2)
        assert result.success_probability > 0.9

    def test_full_algorithm_execution_n4(self):
        """Test complete algorithm for n=4."""
        algorithm = GroversAlgorithm(n_qubits=4)
        result = algorithm.run(target=10)

        # Verify high success probability (measurement is probabilistic)
        assert result.iterations_used == calculate_optimal_iterations(n_qubits=4)
        assert result.success_probability > 0.85

    def test_symbolic_numerical_consistency(self):
        """Test symbolic and numerical results agree."""
        verifier = QuantumVerifier()
        symbolic_result = verifier.verify_grover_protocol()

        algorithm = GroversAlgorithm(n_qubits=3)
        numerical_result = algorithm.run(target=3)

        # Both should confirm algorithm works
        assert symbolic_result['protocol_verified'] is True
        assert numerical_result.success_probability > 0.8

    def test_multiple_searches(self):
        """Test multiple searches with same algorithm instance."""
        algorithm = GroversAlgorithm(n_qubits=3)

        for target in [0, 2, 5, 7]:
            result = algorithm.run(target=target)
            # Verify high success probability (measurement is probabilistic)
            assert result.success_probability > 0.8

    def test_algorithm_with_different_qubit_counts(self):
        """Test algorithm scales to different database sizes."""
        for n in [2, 3, 4, 5]:
            algorithm = GroversAlgorithm(n_qubits=n)
            target = 2**(n-1)  # Middle of database
            result = algorithm.run(target=target)

            assert result.found_state == target
            assert result.success_probability > 0.75


class TestGroverEdgeCases:
    """Test edge cases and error handling."""

    def test_single_qubit_search(self):
        """Test n=1 (searching between 2 items)."""
        algorithm = GroversAlgorithm(n_qubits=1)

        # For n=1, success probability is only ~50% after 1 iteration
        # Just verify the algorithm runs without error
        result_0 = algorithm.run(target=0)
        assert result_0.n_qubits == 1
        assert 0 <= result_0.success_probability <= 1

        result_1 = algorithm.run(target=1)
        assert result_1.n_qubits == 1
        assert 0 <= result_1.success_probability <= 1

    def test_invalid_target_raises_error(self):
        """Test invalid target index raises error."""
        algorithm = GroversAlgorithm(n_qubits=2)

        with pytest.raises(ValueError):
            algorithm.run(target=4)  # Out of range for N=4

        with pytest.raises(ValueError):
            algorithm.run(target=-1)  # Negative index

    def test_zero_iterations_gives_uniform_probability(self):
        """Test k=0 iterations leaves uniform superposition."""
        algorithm = GroversAlgorithm(n_qubits=3)
        N = 8

        for target in range(N):
            prob = algorithm.get_probability_after_k_iterations(target=target, iterations=0)
            expected = 1.0 / N
            assert np.isclose(prob, expected, atol=1e-10)


class TestGroverModuleFunctions:
    """Test standalone module functions."""

    def test_construct_oracle_function(self):
        """Test standalone oracle construction."""
        oracle = construct_oracle(n_qubits=3, target=5)
        assert oracle.shape == (8, 8)
        assert np.isclose(oracle[5, 5], -1.0)

    def test_construct_diffuser_function(self):
        """Test standalone diffuser construction."""
        diffuser = construct_diffuser(n_qubits=3)
        assert diffuser.shape == (8, 8)

    def test_calculate_optimal_iterations_function(self):
        """Test standalone iteration calculation."""
        k = calculate_optimal_iterations(n_qubits=4)
        assert k == 3

    def test_calculate_success_probability_function(self):
        """Test standalone probability calculation."""
        prob = calculate_success_probability(n_qubits=3, iterations=2)
        assert 0 <= prob <= 1
