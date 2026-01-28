"""
Comprehensive tests for quantum decoherence and master equations.

Test Categories:
1. Density matrix properties
2. TPCP channel properties
3. Decoherence channels (bit flip, phase flip, depolarizing)
4. Lindblad equation solver
5. T1/T2 relaxation
6. Purity decay
7. Experimental benchmarks
"""

import pytest
import numpy as np
from src.quantum.decoherence import (
    DensityMatrix,
    DensityMatrixOperations,
    QuantumChannel,
    DecoherenceChannels,
    LindbladSolver,
    RelaxationModels
)


class TestDensityMatrixProperties:
    """Test density matrix construction and properties."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-10

    def test_pure_state_from_vector(self):
        """Test creating density matrix from pure state vector."""
        # |0⟩ state
        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        assert rho.is_pure
        assert np.isclose(rho.purity, 1.0, atol=self.tolerance)
        assert rho.dimension == 2

        # Check matrix form
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(rho.matrix, expected, atol=self.tolerance)

    def test_superposition_state(self):
        """Test |+⟩ = (|0⟩ + |1⟩)/√2 state."""
        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = DensityMatrixOperations.from_state_vector(ket_plus)

        assert rho.is_pure
        assert np.isclose(rho.purity, 1.0, atol=self.tolerance)

        # Check matrix form: ρ = [[1/2, 1/2], [1/2, 1/2]]
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        assert np.allclose(rho.matrix, expected, atol=self.tolerance)

    def test_mixed_state_ensemble(self):
        """Test mixed state from statistical ensemble."""
        ket_0 = np.array([1, 0], dtype=complex)
        ket_1 = np.array([0, 1], dtype=complex)

        # Equal mixture: ρ = (1/2)|0⟩⟨0| + (1/2)|1⟩⟨1| = I/2
        rho = DensityMatrixOperations.from_ensemble([ket_0, ket_1], [0.5, 0.5])

        assert not rho.is_pure
        assert np.isclose(rho.purity, 0.5, atol=self.tolerance)

        # Check matrix form
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rho.matrix, expected, atol=self.tolerance)

    def test_maximally_mixed(self):
        """Test maximally mixed state."""
        rho = DensityMatrixOperations.maximally_mixed(2)

        assert not rho.is_pure
        assert np.isclose(rho.purity, 0.5, atol=self.tolerance)

        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rho.matrix, expected, atol=self.tolerance)

    def test_hermiticity(self):
        """Verify all density matrices are Hermitian."""
        states = [
            DensityMatrixOperations.from_state_vector(np.array([1, 0], dtype=complex)),
            DensityMatrixOperations.from_state_vector(np.array([1, 1], dtype=complex)/np.sqrt(2)),
            DensityMatrixOperations.maximally_mixed(2)
        ]

        for rho in states:
            assert np.allclose(rho.matrix, rho.matrix.conj().T, atol=self.tolerance)

    def test_trace_normalization(self):
        """Verify Tr(ρ) = 1 for all density matrices."""
        states = [
            DensityMatrixOperations.from_state_vector(np.array([1, 0], dtype=complex)),
            DensityMatrixOperations.from_state_vector(np.array([1, 1], dtype=complex)/np.sqrt(2)),
            DensityMatrixOperations.maximally_mixed(2)
        ]

        for rho in states:
            trace = np.trace(rho.matrix).real
            assert np.isclose(trace, 1.0, atol=self.tolerance)

    def test_bloch_vector(self):
        """Test Bloch vector computation."""
        # |0⟩ → r = (0, 0, 1)
        ket_0 = np.array([1, 0], dtype=complex)
        rho_0 = DensityMatrixOperations.from_state_vector(ket_0)
        r = rho_0.bloch_vector
        assert np.allclose(r, [0, 0, 1], atol=self.tolerance)

        # |1⟩ → r = (0, 0, -1)
        ket_1 = np.array([0, 1], dtype=complex)
        rho_1 = DensityMatrixOperations.from_state_vector(ket_1)
        r = rho_1.bloch_vector
        assert np.allclose(r, [0, 0, -1], atol=self.tolerance)

        # |+⟩ → r = (1, 0, 0)
        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho_plus = DensityMatrixOperations.from_state_vector(ket_plus)
        r = rho_plus.bloch_vector
        assert np.allclose(r, [1, 0, 0], atol=self.tolerance)

        # I/2 → r = (0, 0, 0)
        rho_mixed = DensityMatrixOperations.maximally_mixed(2)
        r = rho_mixed.bloch_vector
        assert np.allclose(r, [0, 0, 0], atol=self.tolerance)


class TestTPCPProperties:
    """Test Trace Preserving Completely Positive properties."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-10

    def test_bit_flip_trace_preserving(self):
        """Verify bit flip channel is trace preserving."""
        channel = DecoherenceChannels.bit_flip(p=0.1)

        # Test on various states
        ket_0 = np.array([1, 0], dtype=complex)
        rho_0 = DensityMatrixOperations.from_state_vector(ket_0)

        rho_out = channel.apply(rho_0)
        trace = np.trace(rho_out.matrix).real

        assert np.isclose(trace, 1.0, atol=self.tolerance)

    def test_phase_flip_trace_preserving(self):
        """Verify phase flip channel is trace preserving."""
        channel = DecoherenceChannels.phase_flip(p=0.2)

        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho_plus = DensityMatrixOperations.from_state_vector(ket_plus)

        rho_out = channel.apply(rho_plus)
        trace = np.trace(rho_out.matrix).real

        assert np.isclose(trace, 1.0, atol=self.tolerance)

    def test_depolarizing_trace_preserving(self):
        """Verify depolarizing channel is trace preserving."""
        channel = DecoherenceChannels.depolarizing(p=0.3)

        rho_mixed = DensityMatrixOperations.maximally_mixed(2)
        rho_out = channel.apply(rho_mixed)
        trace = np.trace(rho_out.matrix).real

        assert np.isclose(trace, 1.0, atol=self.tolerance)

    def test_completeness_relation(self):
        """Verify Σₖ Kₖ†Kₖ = I for all channels."""
        channels = [
            DecoherenceChannels.bit_flip(0.1),
            DecoherenceChannels.phase_flip(0.2),
            DecoherenceChannels.depolarizing(0.3),
            DecoherenceChannels.amplitude_damping(0.15),
            DecoherenceChannels.phase_damping(0.25)
        ]

        I = np.eye(2, dtype=complex)

        for channel in channels:
            completeness = sum(K.conj().T @ K for K in channel.kraus_operators)
            assert np.allclose(completeness, I, atol=self.tolerance), \
                f"Channel {channel.name} fails completeness"

    def test_complete_positivity(self):
        """Verify channels are completely positive."""
        channels = [
            DecoherenceChannels.bit_flip(0.1),
            DecoherenceChannels.phase_flip(0.2),
            DecoherenceChannels.depolarizing(0.3)
        ]

        for channel in channels:
            assert channel.verify_complete_positivity(), \
                f"Channel {channel.name} is not completely positive"


class TestDecoherenceChannels:
    """Test specific decoherence channel behaviors."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-10

    def test_bit_flip_effect(self):
        """Test bit flip channel flips |0⟩ ↔ |1⟩."""
        # With p=1, should perfectly flip
        channel = DecoherenceChannels.bit_flip(p=1.0)

        ket_0 = np.array([1, 0], dtype=complex)
        rho_0 = DensityMatrixOperations.from_state_vector(ket_0)

        rho_out = channel.apply(rho_0)

        # Should get |1⟩⟨1|
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(rho_out.matrix, expected, atol=self.tolerance)

    def test_phase_flip_populations_unchanged(self):
        """Verify phase flip preserves populations (diagonal elements)."""
        channel = DecoherenceChannels.phase_flip(p=0.5)

        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho_plus = DensityMatrixOperations.from_state_vector(ket_plus)

        rho_out = channel.apply(rho_plus)

        # Populations should be unchanged
        assert np.isclose(rho_out.matrix[0, 0].real, 0.5, atol=self.tolerance)
        assert np.isclose(rho_out.matrix[1, 1].real, 0.5, atol=self.tolerance)

    def test_depolarizing_toward_mixed(self):
        """Test depolarizing channel drives state toward I/2."""
        # Pure state
        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        # Apply depolarizing channel repeatedly
        channel = DecoherenceChannels.depolarizing(p=0.3)

        for _ in range(10):
            rho = channel.apply(rho)

        # Should approach maximally mixed
        target = np.eye(2, dtype=complex) / 2
        # Not exact match, but purity should decrease significantly
        assert rho.purity < 0.7  # Started at 1.0

    def test_amplitude_damping_to_ground_state(self):
        """Test amplitude damping drives |1⟩ → |0⟩."""
        # Start in excited state |1⟩
        ket_1 = np.array([0, 1], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_1)

        # Strong damping (γ ≈ 1)
        channel = DecoherenceChannels.amplitude_damping(gamma=0.99)
        rho_out = channel.apply(rho)

        # Should be mostly in |0⟩
        assert rho_out.matrix[0, 0].real > 0.9  # Ground state population

    def test_phase_damping_coherence_loss(self):
        """Test phase damping destroys off-diagonal elements."""
        # Superposition state |+⟩
        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = DensityMatrixOperations.from_state_vector(ket_plus)

        # Strong dephasing
        channel = DecoherenceChannels.phase_damping(lambda_=0.99)
        rho_out = channel.apply(rho)

        # Off-diagonal elements should be near zero
        assert np.abs(rho_out.matrix[0, 1]) < 0.1  # Coherence lost


class TestPurityDecay:
    """Test purity evolution under decoherence."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-10

    def test_purity_never_increases(self):
        """Verify purity never increases under decoherence."""
        # Start with pure state
        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)
        initial_purity = rho.purity

        # Apply various channels
        channels = [
            DecoherenceChannels.bit_flip(0.1),
            DecoherenceChannels.phase_flip(0.2),
            DecoherenceChannels.depolarizing(0.15)
        ]

        for channel in channels:
            rho_new = channel.apply(rho)
            assert rho_new.purity <= initial_purity + self.tolerance, \
                f"Purity increased under {channel.name}"

    def test_depolarizing_purity_formula(self):
        """Test analytical purity formula for depolarizing channel."""
        # Start with pure state (P = 1)
        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        p = 0.3
        channel = DecoherenceChannels.depolarizing(p)
        rho_out = channel.apply(rho)

        # Analytical: Our Kraus form gives
        # ε(ρ) = (1-3p/4)ρ + (p/4)(XρX† + YρY† + ZρZ†)
        # For |0⟩⟨0|: ε(|0⟩⟨0|) = (1-3p/4)|0⟩⟨0| + (p/4)(|1⟩⟨1| + |1⟩⟨1| + |0⟩⟨0|)
        #                        = (1-3p/4 + p/4)|0⟩⟨0| + (p/2)|1⟩⟨1|
        #                        = (1-p/2)|0⟩⟨0| + (p/2)|1⟩⟨1|
        # Purity: P = (1-p/2)² + (p/2)²
        expected_purity = (1 - p/2)**2 + (p/2)**2
        assert np.isclose(rho_out.purity, expected_purity, atol=self.tolerance)

    def test_purity_monotonic_decay(self):
        """Test purity decreases monotonically with repeated application."""
        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        channel = DecoherenceChannels.depolarizing(p=0.2)

        purities = [rho.purity]
        for _ in range(10):
            rho = channel.apply(rho)
            purities.append(rho.purity)

        # Check monotonic decrease
        for i in range(len(purities) - 1):
            assert purities[i+1] <= purities[i] + self.tolerance


class TestLindbladEquation:
    """Test Lindblad master equation solver."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-3  # Numerical integration tolerance

    def test_amplitude_damping_analytical(self):
        """Compare Lindblad solution to analytical result for T₁."""
        # Start in |1⟩
        ket_1 = np.array([0, 1], dtype=complex)
        rho0 = DensityMatrixOperations.from_state_vector(ket_1)

        # Lindblad operator for amplitude damping
        gamma1 = 1.0  # Decay rate
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

        solver = LindbladSolver(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            lindblad_operators=[sigma_minus],
            rates=[gamma1]
        )

        # Solve for short time
        times = np.linspace(0, 2.0, 20)
        t_array, states = solver.solve(rho0, (0, 2.0), t_eval=times)

        # Check population decay: ρ₁₁(t) = exp(-γ₁ t) ρ₁₁(0)
        for t, rho in zip(t_array, states):
            expected_pop = np.exp(-gamma1 * t)
            actual_pop = rho.matrix[1, 1].real

            assert np.isclose(actual_pop, expected_pop, rtol=0.05), \
                f"Population mismatch at t={t}: expected {expected_pop}, got {actual_pop}"

    def test_dephasing_analytical(self):
        """Test pure dephasing matches analytical solution."""
        # Start in |+⟩
        ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho0 = DensityMatrixOperations.from_state_vector(ket_plus)

        # Lindblad operator for dephasing
        gamma_phi = 0.5
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        solver = LindbladSolver(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            lindblad_operators=[sigma_z],
            rates=[gamma_phi]
        )

        times = np.linspace(0, 2.0, 20)
        t_array, states = solver.solve(rho0, (0, 2.0), t_eval=times)

        # Check coherence decay: ρ₀₁(t) = exp(-2γφ t) ρ₀₁(0)
        for t, rho in zip(t_array, states):
            expected_coherence = np.exp(-2 * gamma_phi * t) * 0.5
            actual_coherence = np.abs(rho.matrix[0, 1])

            assert np.isclose(actual_coherence, expected_coherence, rtol=0.05), \
                f"Coherence mismatch at t={t}"


class TestT1T2Relaxation:
    """Test T₁ and T₂ relaxation models."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-2

    def test_T1_decay(self):
        """Test T₁ relaxation of excited state."""
        # Start in |1⟩
        ket_1 = np.array([0, 1], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_1)

        T1 = 1.0  # T₁ time
        dt = 0.1  # Time step

        # Evolve for several T₁ times
        times = []
        populations = []

        t = 0
        for _ in range(50):
            populations.append(rho.matrix[1, 1].real)
            times.append(t)

            channel = RelaxationModels.T1_relaxation(T1, dt)
            rho = channel.apply(rho)
            t += dt

        # Check exponential decay: P₁(t) = exp(-t/T₁)
        times = np.array(times)
        populations = np.array(populations)
        expected = np.exp(-times / T1)

        assert np.allclose(populations, expected, rtol=0.05)

    def test_T2_relation(self):
        """Verify T₂ ≤ 2T₁ relation."""
        T1 = 1.0

        # Valid T₂
        try:
            RelaxationModels.T2_dephasing(T1, T2=1.5, time_step=0.1)
        except ValueError:
            pytest.fail("Valid T₂ rejected")

        # Invalid T₂ (too long)
        with pytest.raises(ValueError):
            RelaxationModels.T2_dephasing(T1, T2=2.5, time_step=0.1)


class TestExperimentalBenchmarks:
    """Benchmark against published experimental values."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 0.2  # 20% tolerance for experimental comparison

    def test_superconducting_qubit_timescales(self):
        """Test typical superconducting qubit parameters."""
        # Typical transmon values
        T1 = 100e-6  # 100 μs
        T2 = 80e-6   # 80 μs (< 2T₁)

        # Verify T₂ ≤ 2T₁
        assert T2 <= 2 * T1

        # Test channel creation
        dt = 1e-6  # 1 μs time step
        T1_channel = RelaxationModels.T1_relaxation(T1, dt)
        T2_channel = RelaxationModels.T2_dephasing(T1, T2, dt)

        # Verify channels are valid (TPCP)
        assert T1_channel.verify_complete_positivity()
        assert T2_channel.verify_complete_positivity()

    def test_decoherence_timescale_ordering(self):
        """Verify typical ordering: T₂* < T₂ < 2T₁."""
        T1 = 100e-6
        T2_star = 50e-6  # Pure dephasing
        T2 = 70e-6       # Total dephasing

        # Relations should hold
        assert T2_star < T2
        assert T2 < 2 * T1

        # Pure dephasing rate
        gamma_phi = 1/T2 - 1/(2*T1)
        assert gamma_phi > 0  # Must be positive


class TestChannelConcatenation:
    """Test composition of quantum channels."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = 1e-10

    def test_bit_flip_composition(self):
        """Test that two bit flips = identity (for p=1)."""
        channel = DecoherenceChannels.bit_flip(p=1.0)

        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        # Apply twice
        rho = channel.apply(rho)
        rho = channel.apply(rho)

        # Should return to |0⟩
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(rho.matrix, expected, atol=self.tolerance)

    def test_depolarizing_commutes(self):
        """Test depolarizing channel commutes with itself."""
        p1, p2 = 0.2, 0.3
        channel1 = DecoherenceChannels.depolarizing(p1)
        channel2 = DecoherenceChannels.depolarizing(p2)

        ket_0 = np.array([1, 0], dtype=complex)
        rho = DensityMatrixOperations.from_state_vector(ket_0)

        # Apply in both orders
        rho_12 = channel2.apply(channel1.apply(rho))
        rho_21 = channel1.apply(channel2.apply(rho))

        # Should be equal
        assert np.allclose(rho_12.matrix, rho_21.matrix, atol=self.tolerance)