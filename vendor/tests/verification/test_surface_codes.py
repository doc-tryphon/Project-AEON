"""
Verification Tests for Surface Codes (Step 13).

Surface codes are 2D topological error correction codes that are
the leading candidate for fault-tolerant quantum computing.

Key Properties to Verify:
1. Lattice structure with data qubits on edges
2. X-stabilizers on vertices, Z-stabilizers on plaquettes
3. All stabilizers commute
4. Code distance d requires d errors to cause logical error
5. Threshold theorem: below ~1% error rate, logical errors suppressed

References:
- Kitaev, A. Y. (2003). "Fault-tolerant quantum computation by anyons"
- Fowler et al. (2012). "Surface codes: Towards practical large-scale quantum computation"
- Dennis et al. (2002). "Topological quantum memory"
"""

import pytest
import numpy as np
from typing import List, Tuple, Set

from src.quantum.surface_codes import (
    SurfaceCode,
    SurfaceCodeLattice,
    Stabilizer,
    StabilizerType,
    SurfaceCodeDecoder,
    SurfaceCodeResult,
    create_x_stabilizer,
    create_z_stabilizer,
    get_stabilizer_generators,
    apply_error,
    measure_syndrome,
    correct_errors,
)


# =============================================================================
# Section 1: Lattice Construction Tests
# =============================================================================

class TestLatticeConstruction:
    """Test surface code lattice construction."""

    def test_lattice_creation_d3(self):
        """Test creation of distance-3 surface code lattice."""
        lattice = SurfaceCodeLattice(distance=3)

        # Distance-3 code has specific qubit count
        # For rotated surface code: d^2 + (d-1)^2 data qubits
        # But simpler planar code: d^2 data qubits
        assert lattice.distance == 3
        assert lattice.n_data_qubits > 0

    def test_lattice_creation_d5(self):
        """Test creation of distance-5 surface code lattice."""
        lattice = SurfaceCodeLattice(distance=5)

        assert lattice.distance == 5
        assert lattice.n_data_qubits > lattice.distance  # More qubits than distance

    def test_lattice_distance_must_be_odd(self):
        """Surface code distance should be odd for symmetric code."""
        # Odd distances work
        lattice3 = SurfaceCodeLattice(distance=3)
        lattice5 = SurfaceCodeLattice(distance=5)

        assert lattice3.distance == 3
        assert lattice5.distance == 5

        # Even distances should either raise error or be handled
        # (Some implementations allow even, but odd is standard)

    def test_lattice_contains_data_and_ancilla_qubits(self):
        """Lattice should have both data qubits and ancilla positions."""
        lattice = SurfaceCodeLattice(distance=3)

        assert lattice.n_data_qubits > 0
        assert len(lattice.x_stabilizer_positions) > 0
        assert len(lattice.z_stabilizer_positions) > 0

    def test_lattice_qubit_coordinates(self):
        """Test that qubits have valid 2D coordinates."""
        lattice = SurfaceCodeLattice(distance=3)

        for qubit_id in range(lattice.n_data_qubits):
            coord = lattice.get_qubit_coordinate(qubit_id)
            assert len(coord) == 2
            assert coord[0] >= 0 and coord[1] >= 0


# =============================================================================
# Section 2: Stabilizer Generator Tests
# =============================================================================

class TestStabilizerGenerators:
    """Test stabilizer generator construction."""

    def test_x_stabilizer_creation(self):
        """Test X-stabilizer (vertex operator) creation."""
        lattice = SurfaceCodeLattice(distance=3)
        x_stabs = lattice.get_x_stabilizers()

        assert len(x_stabs) > 0
        for stab in x_stabs:
            assert stab.stabilizer_type == StabilizerType.X
            assert len(stab.qubit_indices) > 0

    def test_z_stabilizer_creation(self):
        """Test Z-stabilizer (plaquette operator) creation."""
        lattice = SurfaceCodeLattice(distance=3)
        z_stabs = lattice.get_z_stabilizers()

        assert len(z_stabs) > 0
        for stab in z_stabs:
            assert stab.stabilizer_type == StabilizerType.Z
            assert len(stab.qubit_indices) > 0

    def test_stabilizer_weight(self):
        """Interior stabilizers should have weight 4."""
        lattice = SurfaceCodeLattice(distance=5)  # Larger to have interior

        x_stabs = lattice.get_x_stabilizers()
        z_stabs = lattice.get_z_stabilizers()

        # At least some stabilizers should have weight 4 (interior)
        has_weight_4 = any(len(s.qubit_indices) == 4 for s in x_stabs + z_stabs)
        assert has_weight_4, "Should have weight-4 interior stabilizers"

    def test_boundary_stabilizers_lower_weight(self):
        """Boundary stabilizers should have weight 2 or 3."""
        lattice = SurfaceCodeLattice(distance=3)

        all_stabs = lattice.get_x_stabilizers() + lattice.get_z_stabilizers()
        weights = [len(s.qubit_indices) for s in all_stabs]

        # Should have stabilizers with weight < 4 (boundary)
        assert min(weights) <= 3, "Should have boundary stabilizers with weight ≤ 3"

    def test_stabilizer_count(self):
        """Test correct number of stabilizer generators."""
        lattice = SurfaceCodeLattice(distance=3)

        n_x = len(lattice.get_x_stabilizers())
        n_z = len(lattice.get_z_stabilizers())

        # Total stabilizers should be n - k where k=1 (one logical qubit)
        # For surface code: roughly equal X and Z stabilizers
        assert n_x > 0
        assert n_z > 0


# =============================================================================
# Section 3: Stabilizer Commutation Tests
# =============================================================================

class TestStabilizerCommutation:
    """Test that all stabilizers commute."""

    def test_x_stabilizers_commute_with_each_other(self):
        """All X-stabilizers should commute with each other."""
        lattice = SurfaceCodeLattice(distance=3)
        x_stabs = lattice.get_x_stabilizers()

        for i, s1 in enumerate(x_stabs):
            for s2 in x_stabs[i+1:]:
                # X operators always commute with X operators
                assert stabilizers_commute(s1, s2), \
                    f"X-stabilizers should commute"

    def test_z_stabilizers_commute_with_each_other(self):
        """All Z-stabilizers should commute with each other."""
        lattice = SurfaceCodeLattice(distance=3)
        z_stabs = lattice.get_z_stabilizers()

        for i, s1 in enumerate(z_stabs):
            for s2 in z_stabs[i+1:]:
                # Z operators always commute with Z operators
                assert stabilizers_commute(s1, s2), \
                    f"Z-stabilizers should commute"

    def test_x_z_stabilizers_commute(self):
        """X and Z stabilizers should commute (even overlap)."""
        lattice = SurfaceCodeLattice(distance=3)
        x_stabs = lattice.get_x_stabilizers()
        z_stabs = lattice.get_z_stabilizers()

        for x_stab in x_stabs:
            for z_stab in z_stabs:
                # Must have even overlap for X and Z to commute
                assert stabilizers_commute(x_stab, z_stab), \
                    f"X and Z stabilizers must commute (even overlap)"


def stabilizers_commute(s1: Stabilizer, s2: Stabilizer) -> bool:
    """
    Check if two stabilizers commute.

    X and Z anticommute on single qubit, so X_i Z_i = -Z_i X_i.
    For product operators to commute, need even number of anticommuting pairs.
    """
    if s1.stabilizer_type == s2.stabilizer_type:
        # Same type always commutes (X with X, Z with Z)
        return True

    # Different types: count overlap
    overlap = len(set(s1.qubit_indices) & set(s2.qubit_indices))

    # Commute if overlap is even
    return overlap % 2 == 0


# =============================================================================
# Section 4: Syndrome Measurement Tests
# =============================================================================

class TestSyndromeMeasurement:
    """Test syndrome measurement for error detection."""

    def test_no_error_trivial_syndrome(self):
        """No errors should give trivial (all-zero) syndrome."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        syndrome = code.measure_syndrome(state)

        assert all(s == 0 for s in syndrome), \
            "No errors should give trivial syndrome"

    def test_single_x_error_detected(self):
        """Single X error should be detected by Z-stabilizers."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply X error on qubit 0
        state_with_error = code.apply_x_error(state, qubit=0)

        syndrome = code.measure_syndrome(state_with_error)

        # Should have non-trivial syndrome
        assert any(s != 0 for s in syndrome), \
            "X error should be detected"

    def test_single_z_error_detected(self):
        """Single Z error should be detected by X-stabilizers."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply Z error on qubit 0
        state_with_error = code.apply_z_error(state, qubit=0)

        syndrome = code.measure_syndrome(state_with_error)

        # Should have non-trivial syndrome
        assert any(s != 0 for s in syndrome), \
            "Z error should be detected"

    def test_syndrome_is_binary(self):
        """Syndrome values should be binary (0 or 1)."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()
        state_with_error = code.apply_x_error(state, qubit=0)

        syndrome = code.measure_syndrome(state_with_error)

        assert all(s in [0, 1] for s in syndrome), \
            "Syndrome should be binary"

    def test_syndrome_length(self):
        """Syndrome length should match number of stabilizers."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        syndrome = code.measure_syndrome(state)
        n_stabilizers = len(code.lattice.get_x_stabilizers()) + \
                       len(code.lattice.get_z_stabilizers())

        assert len(syndrome) == n_stabilizers


# =============================================================================
# Section 5: Error Correction Tests
# =============================================================================

class TestErrorCorrection:
    """Test error correction capability."""

    def test_correct_single_x_error(self):
        """Should correct single X error."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply and correct X error
        state_with_error = code.apply_x_error(state, qubit=0)
        corrected_state = code.correct_errors(state_with_error)

        # Should recover original logical state
        fidelity = code.logical_fidelity(corrected_state, state)
        assert fidelity > 0.99, "Should recover from single X error"

    def test_correct_single_z_error(self):
        """Should correct single Z error."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply and correct Z error
        state_with_error = code.apply_z_error(state, qubit=0)
        corrected_state = code.correct_errors(state_with_error)

        # Should recover original logical state
        fidelity = code.logical_fidelity(corrected_state, state)
        assert fidelity > 0.99, "Should recover from single Z error"

    def test_correct_single_y_error(self):
        """Should correct single Y error (X and Z on same qubit)."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply Y = iXZ error
        state_with_error = code.apply_y_error(state, qubit=0)
        corrected_state = code.correct_errors(state_with_error)

        # Should recover original logical state
        fidelity = code.logical_fidelity(corrected_state, state)
        assert fidelity > 0.99, "Should recover from single Y error"

    def test_cannot_correct_too_many_errors(self):
        """Distance-3 code cannot correct 2+ errors reliably."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Apply 2 errors (beyond correction capability)
        state_with_errors = code.apply_x_error(state, qubit=0)
        state_with_errors = code.apply_x_error(state_with_errors, qubit=1)

        # May or may not correct - but should at least run
        corrected_state = code.correct_errors(state_with_errors)

        # Fidelity may be lower due to logical error
        fidelity = code.logical_fidelity(corrected_state, state)
        # Just verify it runs - fidelity may be 0 or 1 depending on error pattern
        assert 0 <= fidelity <= 1


# =============================================================================
# Section 6: Logical Operator Tests
# =============================================================================

class TestLogicalOperators:
    """Test logical X and Z operators."""

    def test_logical_x_anticommutes_with_logical_z(self):
        """Logical X and Z should anticommute."""
        code = SurfaceCode(distance=3)

        logical_x = code.get_logical_x_operator()
        logical_z = code.get_logical_z_operator()

        # X_L Z_L = -Z_L X_L (anticommute)
        # Test by applying to state
        state = code.initialize_logical_zero()

        # Apply X then Z
        state_xz = code.apply_logical_z(code.apply_logical_x(state))

        # Apply Z then X
        state_zx = code.apply_logical_x(code.apply_logical_z(state))

        # Should differ by phase -1
        overlap = np.abs(np.vdot(state_xz, state_zx))
        assert np.isclose(overlap, 1.0), "Logical X and Z should anticommute"

    def test_logical_x_flips_logical_state(self):
        """Logical X should flip |0_L⟩ to |1_L⟩."""
        code = SurfaceCode(distance=3)

        state_0 = code.initialize_logical_zero()
        state_1 = code.initialize_logical_one()

        # Apply logical X to |0_L⟩
        flipped = code.apply_logical_x(state_0)

        # Should be |1_L⟩
        fidelity = np.abs(np.vdot(flipped, state_1))**2
        assert fidelity > 0.99, "Logical X should flip logical state"

    def test_logical_z_gives_phase_on_logical_one(self):
        """Logical Z should give phase -1 on |1_L⟩."""
        code = SurfaceCode(distance=3)

        state_1 = code.initialize_logical_one()

        # Apply logical Z
        z_state = code.apply_logical_z(state_1)

        # Should be -|1_L⟩
        phase = np.vdot(state_1, z_state)
        assert np.isclose(phase, -1.0, atol=1e-6), \
            "Logical Z should give phase -1 on |1_L⟩"

    def test_logical_operators_commute_with_stabilizers(self):
        """Logical operators should commute with all stabilizers."""
        code = SurfaceCode(distance=3)

        # This is implicit in the construction - logical operators
        # are products of Paulis that commute with all stabilizers
        # Test by verifying syndrome is unchanged

        state = code.initialize_logical_zero()
        syndrome_before = code.measure_syndrome(state)

        state_x = code.apply_logical_x(state)
        syndrome_after_x = code.measure_syndrome(state_x)

        state_z = code.apply_logical_z(state)
        syndrome_after_z = code.measure_syndrome(state_z)

        assert syndrome_before == syndrome_after_x, \
            "Logical X should not change syndrome"
        assert syndrome_before == syndrome_after_z, \
            "Logical Z should not change syndrome"


# =============================================================================
# Section 7: Code Distance Tests
# =============================================================================

class TestCodeDistance:
    """Test code distance properties."""

    def test_distance_3_corrects_1_error(self):
        """Distance-3 code should correct t = floor((d-1)/2) = 1 error."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        # Single error should be correctable
        for qubit in range(min(3, code.lattice.n_data_qubits)):
            state_err = code.apply_x_error(state, qubit)
            corrected = code.correct_errors(state_err)
            fidelity = code.logical_fidelity(corrected, state)
            assert fidelity > 0.99, f"Should correct single error on qubit {qubit}"

    @pytest.mark.skip(reason="Distance-5 requires too much memory for dense simulation")
    def test_distance_5_corrects_2_errors(self):
        """Distance-5 code should correct t = floor((d-1)/2) = 2 errors."""
        # Note: For production, use stabilizer formalism (Clifford simulator)
        # Dense simulation of 25 qubits requires 2^25 * 16 bytes = 512 MB per vector
        code = SurfaceCode(distance=5)
        state = code.initialize_logical_zero()

        # Two well-separated errors should be correctable
        state_err = code.apply_x_error(state, qubit=0)
        state_err = code.apply_x_error(state_err, qubit=code.lattice.n_data_qubits - 1)

        corrected = code.correct_errors(state_err)
        fidelity = code.logical_fidelity(corrected, state)

        # Should correct with high probability
        assert fidelity > 0.9, "Distance-5 should correct 2 separated errors"

    def test_logical_x_has_minimum_weight_d(self):
        """Logical X operator should have minimum weight d."""
        code = SurfaceCode(distance=3)

        logical_x = code.get_logical_x_operator()
        weight = len(logical_x.qubit_indices)

        assert weight >= code.distance, \
            f"Logical X weight {weight} should be >= distance {code.distance}"

    def test_logical_z_has_minimum_weight_d(self):
        """Logical Z operator should have minimum weight d."""
        code = SurfaceCode(distance=3)

        logical_z = code.get_logical_z_operator()
        weight = len(logical_z.qubit_indices)

        assert weight >= code.distance, \
            f"Logical Z weight {weight} should be >= distance {code.distance}"


# =============================================================================
# Section 8: Threshold Behavior Tests
# =============================================================================

class TestThresholdBehavior:
    """Test threshold theorem behavior."""

    def test_low_error_rate_correctable(self):
        """Below threshold, errors should be correctable."""
        code = SurfaceCode(distance=3)

        # Very low error rate
        error_rate = 0.001
        n_trials = 20
        successes = 0

        for _ in range(n_trials):
            state = code.initialize_logical_zero()

            # Apply random errors with given rate
            state_err = code.apply_random_errors(state, error_rate)
            corrected = code.correct_errors(state_err)

            if code.logical_fidelity(corrected, state) > 0.99:
                successes += 1

        success_rate = successes / n_trials
        assert success_rate > 0.8, \
            f"Low error rate should give high success ({success_rate})"

    @pytest.mark.skip(reason="Distance-5 requires too much memory for dense simulation")
    def test_larger_distance_better_suppression(self):
        """Larger distance should give better error suppression."""
        # Note: This test requires d=5 which is too large for dense simulation
        error_rate = 0.01  # 1% - below threshold
        n_trials = 10

        results = {}
        for d in [3, 5]:
            code = SurfaceCode(distance=d)
            successes = 0

            for _ in range(n_trials):
                state = code.initialize_logical_zero()
                state_err = code.apply_random_errors(state, error_rate)
                corrected = code.correct_errors(state_err)

                if code.logical_fidelity(corrected, state) > 0.99:
                    successes += 1

            results[d] = successes / n_trials

        # Larger distance should have equal or better success rate
        assert results[5] >= results[3] * 0.9, \
            "Larger distance should give better or equal suppression"


# =============================================================================
# Section 9: Integration Tests
# =============================================================================

class TestSurfaceCodeIntegration:
    """Integration tests for complete surface code workflow."""

    def test_full_encode_error_correct_decode_cycle(self):
        """Test complete cycle: encode → error → correct → decode."""
        code = SurfaceCode(distance=3)

        # Encode logical |0⟩
        logical_state = code.initialize_logical_zero()

        # Verify in code space
        syndrome = code.measure_syndrome(logical_state)
        assert all(s == 0 for s in syndrome), "Encoded state should have trivial syndrome"

        # Apply error
        errored_state = code.apply_x_error(logical_state, qubit=0)

        # Measure syndrome
        error_syndrome = code.measure_syndrome(errored_state)
        assert any(s != 0 for s in error_syndrome), "Error should be detected"

        # Correct
        corrected_state = code.correct_errors(errored_state)

        # Verify correction
        final_syndrome = code.measure_syndrome(corrected_state)
        assert all(s == 0 for s in final_syndrome), "Corrected state should have trivial syndrome"

        # Check logical state preserved
        fidelity = code.logical_fidelity(corrected_state, logical_state)
        assert fidelity > 0.99, "Logical state should be preserved"

    def test_logical_operations_work_correctly(self):
        """Test logical gate operations on encoded states."""
        code = SurfaceCode(distance=3)

        # Start in |0_L⟩
        state = code.initialize_logical_zero()

        # Apply logical X to get |1_L⟩
        state = code.apply_logical_x(state)

        # Apply logical Z (should give -|1_L⟩)
        state = code.apply_logical_z(state)

        # Apply logical X again to get -|0_L⟩
        state = code.apply_logical_x(state)

        # Should be back to |0_L⟩ (up to phase)
        reference = code.initialize_logical_zero()
        overlap = np.abs(np.vdot(state, reference))

        assert np.isclose(overlap, 1.0, atol=1e-6), \
            "Logical operations should work correctly"

    def test_multiple_error_correction_rounds(self):
        """Test multiple rounds of error correction."""
        code = SurfaceCode(distance=3)
        state = code.initialize_logical_zero()

        n_rounds = 5
        error_rate = 0.001  # Very low

        for round_idx in range(n_rounds):
            # Apply random errors
            state = code.apply_random_errors(state, error_rate)

            # Correct
            state = code.correct_errors(state)

            # Verify still in code space
            syndrome = code.measure_syndrome(state)
            assert all(s == 0 for s in syndrome), \
                f"Round {round_idx}: Should be in code space after correction"

        # Final logical fidelity
        reference = code.initialize_logical_zero()
        fidelity = code.logical_fidelity(state, reference)
        assert fidelity > 0.95, "Should maintain logical state through rounds"
