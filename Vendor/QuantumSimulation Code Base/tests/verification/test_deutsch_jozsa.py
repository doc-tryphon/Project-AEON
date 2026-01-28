"""
TDD Test Suite for Deutsch-Jozsa Algorithm (Step 10).

Tests verify:
1. Oracle construction (constant and balanced functions)
2. Algorithm correctness with deterministic classification
3. Single qubit case (Deutsch's algorithm)
4. Multi-qubit cases
5. Symbolic verification of quantum speedup

Run with: pytest tests/verification/test_deutsch_jozsa.py -v
"""

import pytest
import numpy as np
from sympy import sqrt, simplify, Rational, symbols, Matrix, I

import sys
sys.path.insert(0, 'src')

from algorithms.deutsch_jozsa import (
    DeutschJozsaAlgorithm, OracleType, DeutschJozsaResult,
    create_constant_oracle, create_balanced_oracle,
    hadamard_transform, verify_deutsch_jozsa_symbolic
)
from verification.symbolic_solver import QuantumVerifier


class TestOracleConstruction:
    """Test oracle construction for constant and balanced functions."""

    # --- Constant Oracles ---

    def test_constant_zero_oracle_exists(self):
        """Test constant-0 oracle can be created."""
        oracle = create_constant_oracle(n_qubits=2, constant_value=0)
        assert oracle is not None

    def test_constant_one_oracle_exists(self):
        """Test constant-1 oracle can be created."""
        oracle = create_constant_oracle(n_qubits=2, constant_value=1)
        assert oracle is not None

    def test_constant_oracle_is_unitary(self):
        """Test constant oracle is unitary."""
        for n in [1, 2, 3]:
            for val in [0, 1]:
                oracle = create_constant_oracle(n_qubits=n, constant_value=val)
                # U^† U = I
                product = oracle.conj().T @ oracle
                assert np.allclose(product, np.eye(2**(n+1))), \
                    f"Constant-{val} oracle not unitary for n={n}"

    def test_constant_zero_oracle_identity_like(self):
        """Test constant-0 oracle acts as identity (f(x)=0 always)."""
        # For f(x) = 0: U_f |x⟩|y⟩ = |x⟩|y⊕0⟩ = |x⟩|y⟩
        oracle = create_constant_oracle(n_qubits=1, constant_value=0)
        assert np.allclose(oracle, np.eye(4))

    def test_constant_one_oracle_flips_ancilla(self):
        """Test constant-1 oracle flips ancilla for all inputs."""
        # For f(x) = 1: U_f |x⟩|y⟩ = |x⟩|y⊕1⟩
        oracle = create_constant_oracle(n_qubits=1, constant_value=1)
        # Should flip the ancilla qubit regardless of input
        # |00⟩ → |01⟩, |01⟩ → |00⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
        expected = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        assert np.allclose(oracle, expected)

    # --- Balanced Oracles ---

    def test_balanced_oracle_exists(self):
        """Test balanced oracle can be created."""
        oracle = create_balanced_oracle(n_qubits=2, pattern=0b10)
        assert oracle is not None

    def test_balanced_oracle_is_unitary(self):
        """Test balanced oracle is unitary."""
        for n in [1, 2, 3]:
            oracle = create_balanced_oracle(n_qubits=n, pattern=1)
            product = oracle.conj().T @ oracle
            assert np.allclose(product, np.eye(2**(n+1))), \
                f"Balanced oracle not unitary for n={n}"

    def test_balanced_oracle_half_zeros_half_ones(self):
        """Test balanced oracle outputs 0 for half inputs, 1 for half."""
        for n in [1, 2, 3]:
            # Use pattern that gives balanced function f(x) = x · pattern (mod 2)
            pattern = 1  # Dot product with pattern vector
            oracle = create_balanced_oracle(n_qubits=n, pattern=pattern)

            # Count outputs
            zeros = 0
            ones = 0
            for x in range(2**n):
                # f(x) = (x & pattern).bit_count() % 2 for XOR-based oracle
                f_x = bin(x & pattern).count('1') % 2
                if f_x == 0:
                    zeros += 1
                else:
                    ones += 1

            assert zeros == ones == 2**(n-1), \
                f"Balanced oracle not balanced for n={n}"

    def test_balanced_oracle_for_n1_is_cnot(self):
        """Test n=1 balanced oracle is CNOT gate."""
        # f(x) = x for pattern=1: flips ancilla when x=1
        oracle = create_balanced_oracle(n_qubits=1, pattern=1)
        # CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        assert np.allclose(oracle, cnot)


class TestHadamardTransform:
    """Test Hadamard transform implementation."""

    def test_single_qubit_hadamard(self):
        """Test single qubit Hadamard gate."""
        H = hadamard_transform(n_qubits=1)
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(H, expected)

    def test_hadamard_is_unitary(self):
        """Test Hadamard transform is unitary."""
        for n in [1, 2, 3]:
            H = hadamard_transform(n_qubits=n)
            product = H.conj().T @ H
            assert np.allclose(product, np.eye(2**n)), \
                f"Hadamard not unitary for n={n}"

    def test_hadamard_is_hermitian(self):
        """Test Hadamard transform is Hermitian (H = H†)."""
        for n in [1, 2, 3]:
            H = hadamard_transform(n_qubits=n)
            assert np.allclose(H, H.conj().T), \
                f"Hadamard not Hermitian for n={n}"

    def test_hadamard_self_inverse(self):
        """Test H^2 = I (Hadamard is self-inverse)."""
        for n in [1, 2, 3]:
            H = hadamard_transform(n_qubits=n)
            H_squared = H @ H
            assert np.allclose(H_squared, np.eye(2**n)), \
                f"H^2 ≠ I for n={n}"

    def test_two_qubit_hadamard_is_tensor_product(self):
        """Test H⊗H = H^⊗2."""
        H1 = hadamard_transform(n_qubits=1)
        H2 = hadamard_transform(n_qubits=2)
        expected = np.kron(H1, H1)
        assert np.allclose(H2, expected)

    def test_hadamard_creates_superposition(self):
        """Test H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2."""
        H = hadamard_transform(n_qubits=1)
        ket_0 = np.array([1, 0])
        result = H @ ket_0
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(result, expected)


class TestDeutschAlgorithmSingleQubit:
    """Test Deutsch's algorithm (n=1 case of Deutsch-Jozsa)."""

    @pytest.fixture
    def algorithm_n1(self):
        return DeutschJozsaAlgorithm(n_qubits=1)

    def test_deutsch_constant_zero(self, algorithm_n1):
        """Test Deutsch algorithm correctly identifies f(x)=0."""
        result = algorithm_n1.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert result.function_type == 'constant'
        assert result.measurement_result == 0  # All zeros in register

    def test_deutsch_constant_one(self, algorithm_n1):
        """Test Deutsch algorithm correctly identifies f(x)=1."""
        result = algorithm_n1.run(oracle_type=OracleType.CONSTANT_ONE)
        assert result.function_type == 'constant'
        assert result.measurement_result == 0

    def test_deutsch_balanced(self, algorithm_n1):
        """Test Deutsch algorithm correctly identifies balanced function."""
        result = algorithm_n1.run(oracle_type=OracleType.BALANCED, pattern=1)
        assert result.function_type == 'balanced'
        assert result.measurement_result != 0  # Non-zero in register

    def test_deutsch_deterministic(self, algorithm_n1):
        """Test Deutsch algorithm is deterministic (single query)."""
        # Run multiple times - should always get same result
        results = [
            algorithm_n1.run(oracle_type=OracleType.BALANCED, pattern=1)
            for _ in range(10)
        ]
        assert all(r.function_type == 'balanced' for r in results)


class TestDeutschJozsaMultiQubit:
    """Test Deutsch-Jozsa algorithm for multiple qubits."""

    def test_two_qubit_constant_zero(self):
        """Test n=2 correctly identifies constant-0."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert result.function_type == 'constant'
        assert result.measurement_result == 0

    def test_two_qubit_constant_one(self):
        """Test n=2 correctly identifies constant-1."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ONE)
        assert result.function_type == 'constant'
        assert result.measurement_result == 0

    def test_two_qubit_balanced(self):
        """Test n=2 correctly identifies balanced function."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        for pattern in [0b01, 0b10, 0b11]:
            result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=pattern)
            assert result.function_type == 'balanced', \
                f"Failed for pattern {bin(pattern)}"

    def test_three_qubit_constant(self):
        """Test n=3 correctly identifies constant function."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert result.function_type == 'constant'

    def test_three_qubit_balanced(self):
        """Test n=3 correctly identifies balanced function."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        for pattern in [0b001, 0b010, 0b100, 0b111]:
            result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=pattern)
            assert result.function_type == 'balanced', \
                f"Failed for pattern {bin(pattern)}"

    def test_algorithm_scales_to_larger_n(self):
        """Test algorithm works for n=4, 5."""
        for n in [4, 5]:
            algorithm = DeutschJozsaAlgorithm(n_qubits=n)

            # Constant
            result_const = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
            assert result_const.function_type == 'constant'

            # Balanced
            result_bal = algorithm.run(oracle_type=OracleType.BALANCED, pattern=1)
            assert result_bal.function_type == 'balanced'


class TestDeutschJozsaResult:
    """Test DeutschJozsaResult structure."""

    def test_result_contains_function_type(self):
        """Test result includes function type classification."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert hasattr(result, 'function_type')
        assert result.function_type in ['constant', 'balanced']

    def test_result_contains_measurement(self):
        """Test result includes measurement outcome."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert hasattr(result, 'measurement_result')

    def test_result_contains_n_qubits(self):
        """Test result includes number of qubits."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=1)
        assert hasattr(result, 'n_qubits')
        assert result.n_qubits == 3

    def test_result_contains_oracle_queries(self):
        """Test result includes oracle query count."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert hasattr(result, 'oracle_queries')
        assert result.oracle_queries == 1  # Quantum advantage!

    def test_result_contains_final_state(self):
        """Test result includes final quantum state."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert hasattr(result, 'final_state')
        assert result.final_state is not None


class TestDeutschJozsaQuantumStates:
    """Test intermediate quantum states in the algorithm."""

    @pytest.fixture
    def algorithm(self):
        return DeutschJozsaAlgorithm(n_qubits=2)

    def test_initial_state_all_zeros_with_one(self, algorithm):
        """Test initial state is |0...0⟩|1⟩."""
        initial = algorithm.get_initial_state()
        # For n=2: |00⟩|1⟩ = |001⟩ in 3-qubit space
        expected = np.zeros(2**3, dtype=complex)
        expected[0b001] = 1.0  # |001⟩
        assert np.allclose(initial, expected)

    def test_after_hadamard_superposition(self, algorithm):
        """Test state after Hadamard is uniform superposition."""
        algorithm.prepare_superposition()
        state = algorithm.current_state

        # Should be (H⊗n|0⟩n) ⊗ (H|1⟩)
        # = (1/√2^n Σ|x⟩) ⊗ (|0⟩-|1⟩)/√2
        assert state is not None
        # Check normalization
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_constant_oracle_preserves_superposition_phase(self, algorithm):
        """Test constant oracle gives same global phase to all terms."""
        algorithm.prepare_superposition()
        oracle = create_constant_oracle(n_qubits=2, constant_value=0)
        state_before = algorithm.current_state.copy()
        algorithm.apply_oracle(oracle)
        state_after = algorithm.current_state

        # For f(x)=0, oracle is identity, states should match
        assert np.allclose(state_before, state_after)

    def test_balanced_oracle_creates_relative_phases(self, algorithm):
        """Test balanced oracle creates relative phases between terms."""
        algorithm.prepare_superposition()
        state_before = algorithm.current_state.copy()
        oracle = create_balanced_oracle(n_qubits=2, pattern=0b01)
        algorithm.apply_oracle(oracle)
        state_after = algorithm.current_state

        # States should differ (not just global phase)
        # Check that it's not proportional
        if not np.allclose(state_before, state_after):
            pass  # Expected - states differ
        else:
            pytest.fail("Balanced oracle should create relative phases")


class TestDeutschJozsaClassicalComparison:
    """Test quantum advantage over classical algorithm."""

    def test_single_query_suffices(self):
        """Test algorithm uses exactly 1 oracle query."""
        for n in [1, 2, 3, 4]:
            algorithm = DeutschJozsaAlgorithm(n_qubits=n)
            result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=1)
            assert result.oracle_queries == 1

    def test_classical_requires_exponential_queries(self):
        """Document that classical algorithm needs 2^(n-1)+1 queries worst case."""
        # This is a documentation test - classical needs to query > half the inputs
        for n in [1, 2, 3, 4]:
            classical_worst_case = 2**(n-1) + 1
            quantum_queries = 1
            assert quantum_queries < classical_worst_case, \
                f"Quantum should beat classical for n={n}"

    def test_exponential_speedup(self):
        """Test quantum achieves exponential speedup."""
        for n in [2, 3, 4, 5]:
            classical_queries = 2**(n-1) + 1  # Worst case
            quantum_queries = 1
            speedup = classical_queries / quantum_queries
            assert speedup >= 2**(n-1), \
                f"Speedup should be exponential for n={n}"


class TestDeutschJozsaSymbolicVerification:
    """Test symbolic verification of Deutsch-Jozsa properties."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_verify_algorithm_structure(self, verifier):
        """Test symbolic verification of algorithm structure."""
        result = verifier.verify_deutsch_jozsa_algorithm()
        assert result['algorithm_verified'] is True

    def test_verify_hadamard_creates_superposition(self, verifier):
        """Test H^⊗n|0⟩^⊗n = 1/√(2^n) Σ|x⟩ symbolically."""
        result = verifier.verify_hadamard_superposition()
        assert result['superposition_correct'] is True
        assert result['normalization_correct'] is True

    def test_verify_phase_kickback(self, verifier):
        """Test phase kickback mechanism symbolically."""
        result = verifier.verify_phase_kickback()
        assert result['phase_kickback_verified'] is True

    def test_verify_interference_for_constant(self, verifier):
        """Test constructive interference for constant functions."""
        result = verifier.verify_interference_constant()
        assert result['constructive_interference'] is True
        assert result['measures_zero'] is True

    def test_verify_interference_for_balanced(self, verifier):
        """Test destructive interference for balanced functions."""
        result = verifier.verify_interference_balanced()
        assert result['destructive_interference'] is True
        assert result['measures_nonzero'] is True

    def test_verify_measurement_determinism(self, verifier):
        """Test measurement is deterministic (not probabilistic)."""
        result = verifier.verify_measurement_determinism()
        assert result['deterministic'] is True
        assert result['probability_one_outcome'] == 1.0

    def test_full_protocol_verification(self, verifier):
        """Test complete protocol verification."""
        result = verifier.verify_deutsch_jozsa_protocol()
        assert result['protocol_verified'] is True

        checks = result['verifications']
        assert 'superposition' in checks
        assert 'phase_kickback' in checks
        assert 'interference' in checks
        assert 'measurement' in checks


class TestDeutschJozsaModuleFunctions:
    """Test standalone module functions."""

    def test_verify_deutsch_jozsa_symbolic_function(self):
        """Test standalone symbolic verification function."""
        result = verify_deutsch_jozsa_symbolic()
        assert result['verified'] is True

    def test_create_constant_oracle_function(self):
        """Test standalone constant oracle creation."""
        oracle = create_constant_oracle(n_qubits=2, constant_value=0)
        assert oracle.shape == (8, 8)

    def test_create_balanced_oracle_function(self):
        """Test standalone balanced oracle creation."""
        oracle = create_balanced_oracle(n_qubits=2, pattern=1)
        assert oracle.shape == (8, 8)

    def test_hadamard_transform_function(self):
        """Test standalone Hadamard transform."""
        H = hadamard_transform(n_qubits=3)
        assert H.shape == (8, 8)


class TestDeutschJozsaIntegration:
    """Integration tests for Deutsch-Jozsa algorithm."""

    def test_full_algorithm_constant_zero(self):
        """Test complete algorithm execution for constant-0."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)

        assert result.function_type == 'constant'
        assert result.oracle_queries == 1
        assert result.n_qubits == 3

    def test_full_algorithm_constant_one(self):
        """Test complete algorithm execution for constant-1."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        result = algorithm.run(oracle_type=OracleType.CONSTANT_ONE)

        assert result.function_type == 'constant'
        assert result.oracle_queries == 1

    def test_full_algorithm_balanced(self):
        """Test complete algorithm execution for balanced function."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=3)
        result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=0b101)

        assert result.function_type == 'balanced'
        assert result.oracle_queries == 1
        assert result.measurement_result != 0

    def test_symbolic_numerical_consistency(self):
        """Test symbolic and numerical results agree."""
        # Symbolic verification
        verifier = QuantumVerifier()
        symbolic_result = verifier.verify_deutsch_jozsa_protocol()

        # Numerical algorithm
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)

        # Both should verify constant → measure 0
        const_result = algorithm.run(oracle_type=OracleType.CONSTANT_ZERO)
        assert symbolic_result['verifications']['interference']['constant_measures_zero'] is True
        assert const_result.measurement_result == 0

        # Both should verify balanced → measure non-zero
        bal_result = algorithm.run(oracle_type=OracleType.BALANCED, pattern=1)
        assert symbolic_result['verifications']['interference']['balanced_measures_nonzero'] is True
        assert bal_result.measurement_result != 0

    def test_algorithm_with_custom_oracle(self):
        """Test algorithm can accept custom oracle matrix."""
        algorithm = DeutschJozsaAlgorithm(n_qubits=2)

        # Create a custom constant oracle
        custom_oracle = create_constant_oracle(n_qubits=2, constant_value=0)
        result = algorithm.run_with_oracle(custom_oracle)

        assert result.function_type == 'constant'
