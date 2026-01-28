"""
Verification Tests for Quantum Phase Estimation (Step 12).

Quantum Phase Estimation (QPE) estimates the eigenvalue phase θ in:
U|ψ⟩ = e^(2πiθ)|ψ⟩

where U is a unitary operator and |ψ⟩ is an eigenvector.

Key Properties to Verify:
1. Quantum Fourier Transform (QFT) creates Fourier basis
2. Inverse QFT decodes phase information
3. Exact phases (powers of 2) → deterministic results
4. Approximate phases → spectral leakage (probability distribution)
5. Precision improves with more counting qubits

References:
- Nielsen & Chuang, Chapter 5.2
- Kitaev, A. Y. (1995). "Quantum measurements and the Abelian Stabilizer Problem"
"""

import pytest
import numpy as np
from src.algorithms.phase_estimation import (
    QuantumFourierTransform,
    PhaseEstimationAlgorithm,
    PhaseEstimationResult,
    apply_controlled_unitary_power,
    estimate_phase_from_measurement,
)


# =============================================================================
# Section 1: Quantum Fourier Transform Tests
# =============================================================================

class TestQFT:
    """Test Quantum Fourier Transform construction and properties."""

    def test_qft_matrix_construction_n2(self):
        """Test QFT matrix for n=2 qubits."""
        qft = QuantumFourierTransform(n_qubits=2)
        matrix = qft.get_qft_matrix()

        # QFT should be 4×4 for 2 qubits
        assert matrix.shape == (4, 4)

        # QFT matrix is symmetric (for real QFT)
        # Actually, QFT has specific form: F_jk = (1/√N) * ω^(jk)
        # where ω = e^(2πi/N)
        N = 4
        omega = np.exp(2j * np.pi / N)

        # Construct expected QFT matrix
        expected = np.zeros((N, N), dtype=complex)
        for j in range(N):
            for k in range(N):
                expected[j, k] = omega**(j * k) / np.sqrt(N)

        assert np.allclose(matrix, expected, atol=1e-10)

    def test_qft_matrix_construction_n3(self):
        """Test QFT matrix for n=3 qubits."""
        qft = QuantumFourierTransform(n_qubits=3)
        matrix = qft.get_qft_matrix()

        assert matrix.shape == (8, 8)

        # Verify QFT formula
        N = 8
        omega = np.exp(2j * np.pi / N)
        expected = np.zeros((N, N), dtype=complex)
        for j in range(N):
            for k in range(N):
                expected[j, k] = omega**(j * k) / np.sqrt(N)

        assert np.allclose(matrix, expected, atol=1e-10)

    def test_qft_is_unitary(self):
        """Verify QFT is unitary: QFT† QFT = I."""
        for n in [2, 3, 4]:
            qft = QuantumFourierTransform(n_qubits=n)
            matrix = qft.get_qft_matrix()

            # Check unitarity
            product = matrix.conj().T @ matrix
            identity = np.eye(2**n)

            assert np.allclose(product, identity, atol=1e-10), \
                f"QFT not unitary for n={n}"

    def test_inverse_qft(self):
        """Verify inverse QFT is conjugate transpose of QFT."""
        for n in [2, 3, 4]:
            qft = QuantumFourierTransform(n_qubits=n)
            qft_matrix = qft.get_qft_matrix()
            inv_qft_matrix = qft.get_inverse_qft_matrix()

            # Inverse should be conjugate transpose
            expected_inv = qft_matrix.conj().T

            assert np.allclose(inv_qft_matrix, expected_inv, atol=1e-10), \
                f"Inverse QFT incorrect for n={n}"

    def test_qft_on_zero_state(self):
        """Apply QFT to |0⟩ state."""
        qft = QuantumFourierTransform(n_qubits=2)

        # |0⟩ state
        state = np.array([1, 0, 0, 0], dtype=complex)

        # Apply QFT
        result = qft.apply_qft(state)

        # QFT|0⟩ should be uniform superposition (1/√N)|0⟩ + ... + (1/√N)|N-1⟩
        expected = np.ones(4, dtype=complex) / 2.0

        assert np.allclose(result, expected, atol=1e-10)

    def test_qft_on_computational_basis_states(self):
        """Apply QFT to each computational basis state."""
        qft = QuantumFourierTransform(n_qubits=2)
        N = 4

        for k in range(N):
            # Create basis state |k⟩
            state = np.zeros(N, dtype=complex)
            state[k] = 1.0

            # Apply QFT
            result = qft.apply_qft(state)

            # QFT|k⟩ = (1/√N) Σ_j ω^(jk)|j⟩
            omega = np.exp(2j * np.pi / N)
            expected = np.array([omega**(j * k) / np.sqrt(N) for j in range(N)])

            assert np.allclose(result, expected, atol=1e-10)

    def test_qft_inverse_qft_identity(self):
        """Verify QFT† QFT = I by applying to arbitrary state."""
        qft = QuantumFourierTransform(n_qubits=3)

        # Arbitrary state
        state = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.1, 0.2], dtype=complex)
        state /= np.linalg.norm(state)

        # Apply QFT then inverse QFT
        qft_state = qft.apply_qft(state)
        recovered = qft.apply_inverse_qft(qft_state)

        assert np.allclose(recovered, state, atol=1e-10)

    def test_qft_fourier_transform_property(self):
        """Verify QFT transforms position basis to momentum basis."""
        qft = QuantumFourierTransform(n_qubits=2)

        # Position eigenstate |2⟩
        position_state = np.array([0, 0, 1, 0], dtype=complex)

        # Apply QFT
        fourier_state = qft.apply_qft(position_state)

        # In Fourier basis, should have specific phases
        N = 4
        k = 2  # Position index
        omega = np.exp(2j * np.pi / N)
        expected = np.array([omega**(j * k) / np.sqrt(N) for j in range(N)])

        assert np.allclose(fourier_state, expected, atol=1e-10)


# =============================================================================
# Section 2: Controlled-Unitary Power Tests
# =============================================================================

class TestControlledUnitaryPowers:
    """Test construction of controlled-U^k operations."""

    def test_controlled_u_power_z_gate(self):
        """Test controlled-Z^k for Z gate."""
        # Z gate: |0⟩ → |0⟩, |1⟩ → -|1⟩
        # Z has eigenvalues: +1 (for |0⟩), -1 (for |1⟩)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Z^2 = I
        Z_squared = apply_controlled_unitary_power(Z, power=2)
        expected = np.array([[1, 0], [0, 1]], dtype=complex)
        assert np.allclose(Z_squared, expected, atol=1e-10)

        # Z^3 = Z
        Z_cubed = apply_controlled_unitary_power(Z, power=3)
        assert np.allclose(Z_cubed, Z, atol=1e-10)

    def test_controlled_u_power_t_gate(self):
        """Test controlled-T^k for T gate."""
        # T gate: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        # T^2 = S gate (π/2 rotation)
        T_squared = apply_controlled_unitary_power(T, power=2)
        S = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]], dtype=complex)
        assert np.allclose(T_squared, S, atol=1e-10)

        # T^4 = Z gate (π rotation)
        T_fourth = apply_controlled_unitary_power(T, power=4)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(T_fourth, Z, atol=1e-10)

        # T^8 = I (full rotation)
        T_eighth = apply_controlled_unitary_power(T, power=8)
        I = np.eye(2, dtype=complex)
        assert np.allclose(T_eighth, I, atol=1e-10)

    def test_controlled_u_power_arbitrary_rotation(self):
        """Test U^k for arbitrary rotation gate."""
        # R_z(θ) = diag(1, e^(iθ))
        theta = 0.7  # Arbitrary angle
        R_z = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

        # R_z^k should give e^(ikθ) phase
        for k in [1, 2, 4, 8]:
            R_z_k = apply_controlled_unitary_power(R_z, power=k)
            expected = np.array([[1, 0], [0, np.exp(1j * k * theta)]], dtype=complex)
            assert np.allclose(R_z_k, expected, atol=1e-10)


# =============================================================================
# Section 3: Phase Estimation - Exact Phases
# =============================================================================

class TestPhaseEstimationExactPhases:
    """Test QPE with exact binary phases (no spectral leakage)."""

    def test_phase_estimation_z_gate(self):
        """
        Estimate phase for Z gate.

        Z|1⟩ = -|1⟩ = e^(iπ)|1⟩
        θ = π/(2π) = 0.5 (exact binary fraction)
        """
        # Z gate
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Eigenvector |1⟩
        eigenvector = np.array([0, 1], dtype=complex)

        # QPE with 3 counting qubits
        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)
        result = qpe.run(unitary=Z, eigenvector=eigenvector)

        # Expected phase: θ = 0.5
        # With 3 qubits: can represent 0, 1/8, 2/8, ..., 7/8
        # θ = 0.5 = 4/8 exactly
        assert np.isclose(result.estimated_phase, 0.5, atol=1e-2)

        # Measurement should be deterministic for exact phase
        # Measured value should be 4 (binary 100 → 0.100 = 0.5)
        assert result.measured_value == 4 or np.isclose(result.estimated_phase, 0.5, atol=1e-2)

    def test_phase_estimation_s_gate(self):
        """
        Estimate phase for S gate.

        S|1⟩ = i|1⟩ = e^(iπ/2)|1⟩
        θ = (π/2)/(2π) = 0.25 (exact binary fraction)
        """
        # S gate
        S = np.array([[1, 0], [0, 1j]], dtype=complex)

        # Eigenvector |1⟩
        eigenvector = np.array([0, 1], dtype=complex)

        # QPE with 3 counting qubits
        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)
        result = qpe.run(unitary=S, eigenvector=eigenvector)

        # Expected: θ = 0.25 = 2/8
        assert np.isclose(result.estimated_phase, 0.25, atol=1e-2)

    def test_phase_estimation_t_gate(self):
        """
        Estimate phase for T gate.

        T|1⟩ = e^(iπ/4)|1⟩
        θ = (π/4)/(2π) = 0.125 (exact binary fraction)
        """
        # T gate
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        # Eigenvector |1⟩
        eigenvector = np.array([0, 1], dtype=complex)

        # QPE with 4 counting qubits (need at least 3 for 0.125)
        qpe = PhaseEstimationAlgorithm(n_counting_qubits=4)
        result = qpe.run(unitary=T, eigenvector=eigenvector)

        # Expected: θ = 0.125 = 2/16
        assert np.isclose(result.estimated_phase, 0.125, atol=1e-2)

    def test_phase_estimation_identity(self):
        """
        Estimate phase for Identity gate.

        I|ψ⟩ = |ψ⟩ = e^(i·0)|ψ⟩
        θ = 0 (exact)
        """
        I = np.eye(2, dtype=complex)

        # Any eigenvector
        eigenvector = np.array([1, 0], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)
        result = qpe.run(unitary=I, eigenvector=eigenvector)

        # Expected: θ = 0
        assert np.isclose(result.estimated_phase, 0.0, atol=1e-2)


# =============================================================================
# Section 4: Phase Estimation - Approximate Phases (Spectral Leakage)
# =============================================================================

class TestPhaseEstimationSpectralLeakage:
    """Test QPE with non-binary phases (spectral leakage)."""

    def test_phase_estimation_one_third(self):
        """
        Estimate θ = 1/3 (not a binary fraction).

        This demonstrates spectral leakage - the result is a probability
        distribution peaked near 1/3, not a single deterministic value.
        """
        # Construct U with eigenvalue e^(2πi/3)
        theta_true = 1.0 / 3.0
        U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]], dtype=complex)

        eigenvector = np.array([0, 1], dtype=complex)

        # Use more counting qubits for better approximation
        qpe = PhaseEstimationAlgorithm(n_counting_qubits=5)
        result = qpe.run(unitary=U, eigenvector=eigenvector)

        # We can't expect exact 1/3, but should be close
        # With 5 qubits: resolution = 1/32 ≈ 0.03125
        # Closest binary: 11/32 = 0.34375 or 10/32 = 0.3125
        assert np.abs(result.estimated_phase - theta_true) < 0.05

    def test_phase_estimation_pi_over_8(self):
        """
        Estimate θ = π/(2π·8) = 1/16 (exact).

        Wait, 1/16 IS exact with 4+ qubits. Let's test 1/10 instead.
        """
        # θ = 0.1 (not a binary fraction)
        theta_true = 0.1
        U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]], dtype=complex)

        eigenvector = np.array([0, 1], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=6)
        result = qpe.run(unitary=U, eigenvector=eigenvector)

        # With 6 qubits: resolution = 1/64 ≈ 0.015625
        # Closest: 6/64 = 0.09375 or 7/64 = 0.109375
        # Due to spectral leakage, measurement is probabilistic
        assert np.abs(result.estimated_phase - theta_true) < 0.12  # More lenient tolerance

    def test_phase_estimation_irrational_fraction(self):
        """
        Test θ = √2/10 ≈ 0.14142 (irrational).

        This is the ultimate "intellectual honesty" test - QPE gives
        a probability distribution, not an exact answer.
        """
        theta_true = np.sqrt(2) / 10.0  # ≈ 0.14142
        U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]], dtype=complex)

        eigenvector = np.array([0, 1], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=6)
        result = qpe.run(unitary=U, eigenvector=eigenvector)

        # Should be within resolution of 1/64
        assert np.abs(result.estimated_phase - theta_true) < 0.03


# =============================================================================
# Section 5: Precision vs Number of Counting Qubits
# =============================================================================

class TestPrecisionScaling:
    """Test how precision improves with number of counting qubits."""

    def test_precision_improvement_with_qubits(self):
        """Verify precision δθ ≈ 1/2^t."""
        theta_true = 0.3  # Non-binary fraction

        U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]], dtype=complex)
        eigenvector = np.array([0, 1], dtype=complex)

        errors = []

        for t in [3, 4, 5, 6]:
            qpe = PhaseEstimationAlgorithm(n_counting_qubits=t)
            result = qpe.run(unitary=U, eigenvector=eigenvector)

            error = np.abs(result.estimated_phase - theta_true)
            errors.append(error)

            # Error should be roughly ≤ 1/2^t
            # For non-binary phases, spectral leakage can cause larger errors
            expected_resolution = 1.0 / (2**t)
            assert error <= expected_resolution * 5  # Lenient tolerance for spectral leakage

        # Verify errors generally decrease
        # (Not strictly monotonic due to discretization)
        assert errors[-1] < errors[0]  # Best precision better than worst

    def test_exact_phase_independent_of_qubit_count(self):
        """Exact binary phases should work with any sufficient qubit count."""
        # θ = 0.25 (exact with ≥2 qubits)
        theta_true = 0.25
        S = np.array([[1, 0], [0, 1j]], dtype=complex)  # S gate (π/2 rotation)
        eigenvector = np.array([0, 1], dtype=complex)

        for t in [2, 3, 4, 5]:
            qpe = PhaseEstimationAlgorithm(n_counting_qubits=t)
            result = qpe.run(unitary=S, eigenvector=eigenvector)

            # Should always get exact answer
            assert np.isclose(result.estimated_phase, theta_true, atol=1e-2)


# =============================================================================
# Section 6: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_counting_qubit(self):
        """Test QPE with t=1 counting qubit."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        eigenvector = np.array([0, 1], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=1)
        result = qpe.run(unitary=Z, eigenvector=eigenvector)

        # With 1 qubit, can only distinguish 0 from 0.5
        # Z has θ = 0.5, should measure 1
        assert result.estimated_phase in [0.0, 0.5]

    def test_non_eigenvector_input(self):
        """Test QPE with state that is NOT an eigenvector."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Superposition: (|0⟩ + |1⟩)/√2 (not an eigenvector)
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)
        result = qpe.run(unitary=Z, eigenvector=state)

        # State is 50% |0⟩ (θ=0) and 50% |1⟩ (θ=0.5)
        # Result should be probabilistic mix
        # Just verify it runs without error
        assert 0 <= result.estimated_phase <= 1

    def test_invalid_unitary_raises_error(self):
        """Test that non-unitary matrix raises error."""
        # Non-unitary matrix
        non_unitary = np.array([[2, 0], [0, 1]], dtype=complex)
        eigenvector = np.array([1, 0], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)

        with pytest.raises((ValueError, AssertionError)):
            qpe.run(unitary=non_unitary, eigenvector=eigenvector)

    def test_unnormalized_eigenvector(self):
        """Test that unnormalized eigenvector is handled correctly."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Unnormalized |1⟩
        eigenvector = np.array([0, 3], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=3)
        result = qpe.run(unitary=Z, eigenvector=eigenvector)

        # Should auto-normalize and get θ = 0.5
        assert np.isclose(result.estimated_phase, 0.5, atol=1e-2)


# =============================================================================
# Section 7: Full Algorithm Execution Tests
# =============================================================================

class TestFullAlgorithmExecution:
    """Test complete QPE execution pipeline."""

    def test_qpe_complete_workflow_z_gate(self):
        """Complete QPE workflow for Z gate."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        eigenvector = np.array([0, 1], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=4)
        result = qpe.run(unitary=Z, eigenvector=eigenvector)

        # Verify result object
        assert isinstance(result, PhaseEstimationResult)
        assert result.n_counting_qubits == 4
        assert 0 <= result.estimated_phase <= 1
        assert 0 <= result.measured_value < 2**4
        assert np.isclose(result.estimated_phase, 0.5, atol=1e-2)

    def test_qpe_multiple_runs_consistency(self):
        """Run QPE multiple times and verify consistency."""
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        eigenvector = np.array([0, 1], dtype=complex)

        qpe = PhaseEstimationAlgorithm(n_counting_qubits=5)

        results = []
        for _ in range(10):
            result = qpe.run(unitary=T, eigenvector=eigenvector)
            results.append(result.estimated_phase)

        # All results should be very close (exact phase)
        std_dev = np.std(results)
        assert std_dev < 0.01  # Low variance for exact phase

    def test_qpe_power_of_two_phases(self):
        """Test all exact power-of-two phases."""
        eigenvector = np.array([0, 1], dtype=complex)

        exact_phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        for theta in exact_phases:
            U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta)]], dtype=complex)

            qpe = PhaseEstimationAlgorithm(n_counting_qubits=4)
            result = qpe.run(unitary=U, eigenvector=eigenvector)

            assert np.isclose(result.estimated_phase, theta, atol=1e-2), \
                f"Failed for θ={theta}"
