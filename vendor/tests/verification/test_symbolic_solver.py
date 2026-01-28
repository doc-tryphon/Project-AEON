"""
TDD Test Suite for QuantumVerifier (Symbolic Truth Engine).

These tests verify that the symbolic quantum verification engine
produces mathematically correct results using SymPy.

Run with: pytest tests/verification/test_symbolic_solver.py -v
"""

import pytest
from sympy import sqrt, Matrix, eye, zeros, ln, simplify, I, pi
from sympy.physics.quantum import TensorProduct

import sys
sys.path.insert(0, 'src')

from verification.symbolic_solver import (
    QuantumVerifier,
    # Basis states
    ket_0, ket_1, bra_0, bra_1,
    # Gates
    pauli_x, pauli_y, pauli_z, hadamard, identity_2, cnot,
    # Bell states
    bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus,
    # Quick verification functions
    quick_verify_bell_states, quick_verify_pauli_matrices, quick_verify_common_gates
)


class TestBasisStates:
    """Test that basis states are correctly defined."""

    def test_ket_0_definition(self):
        """Test |0⟩ = [1, 0]^T."""
        assert ket_0() == Matrix([1, 0])

    def test_ket_1_definition(self):
        """Test |1⟩ = [0, 1]^T."""
        assert ket_1() == Matrix([0, 1])

    def test_bra_ket_orthogonality(self):
        """Test ⟨0|1⟩ = 0 and ⟨1|0⟩ = 0."""
        assert (bra_0() * ket_1())[0, 0] == 0
        assert (bra_1() * ket_0())[0, 0] == 0

    def test_bra_ket_normalization(self):
        """Test ⟨0|0⟩ = 1 and ⟨1|1⟩ = 1."""
        assert (bra_0() * ket_0())[0, 0] == 1
        assert (bra_1() * ket_1())[0, 0] == 1


class TestFundamentalAxioms:
    """Test Section 1: Fundamental Axiom Verifiers."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    # --- Normalization Tests ---

    def test_verify_normalization_ket0(self, verifier):
        """Test |0⟩ is normalized."""
        is_norm, val = verifier.verify_normalization(ket_0())
        assert is_norm is True
        assert val == 1

    def test_verify_normalization_ket1(self, verifier):
        """Test |1⟩ is normalized."""
        is_norm, val = verifier.verify_normalization(ket_1())
        assert is_norm is True
        assert val == 1

    def test_verify_normalization_superposition(self, verifier):
        """Test (|0⟩ + |1⟩)/√2 is normalized."""
        psi = (ket_0() + ket_1()) / sqrt(2)
        is_norm, val = verifier.verify_normalization(psi)
        assert is_norm is True
        assert simplify(val) == 1

    def test_verify_normalization_unnormalized_fails(self, verifier):
        """Test unnormalized state fails verification."""
        psi = ket_0() + ket_1()  # Not normalized (norm = √2)
        is_norm, val = verifier.verify_normalization(psi)
        assert is_norm is False
        assert simplify(val) == 2

    def test_verify_normalization_bell_state(self, verifier):
        """Test Bell state |Φ+⟩ is normalized."""
        is_norm, val = verifier.verify_normalization(bell_phi_plus())
        assert is_norm is True
        assert simplify(val) == 1

    # --- Unitary Tests ---

    def test_verify_unitary_pauli_x(self, verifier):
        """Test Pauli X is unitary."""
        is_unitary, _ = verifier.verify_unitary(pauli_x())
        assert is_unitary is True

    def test_verify_unitary_pauli_y(self, verifier):
        """Test Pauli Y is unitary."""
        is_unitary, _ = verifier.verify_unitary(pauli_y())
        assert is_unitary is True

    def test_verify_unitary_pauli_z(self, verifier):
        """Test Pauli Z is unitary."""
        is_unitary, _ = verifier.verify_unitary(pauli_z())
        assert is_unitary is True

    def test_verify_unitary_hadamard(self, verifier):
        """Test Hadamard is unitary."""
        is_unitary, product = verifier.verify_unitary(hadamard())
        assert is_unitary is True
        # U†U should equal identity
        assert simplify(product - eye(2)) == zeros(2, 2)

    def test_verify_unitary_cnot(self, verifier):
        """Test CNOT is unitary."""
        is_unitary, _ = verifier.verify_unitary(cnot())
        assert is_unitary is True

    def test_verify_unitary_non_unitary_fails(self, verifier):
        """Test non-unitary matrix fails verification."""
        # A projection is not unitary
        projection = Matrix([[1, 0], [0, 0]])
        is_unitary, _ = verifier.verify_unitary(projection)
        assert is_unitary is False

    # --- Hermitian Tests ---

    def test_verify_hermitian_pauli_x(self, verifier):
        """Test Pauli X is Hermitian."""
        is_herm, _ = verifier.verify_hermitian(pauli_x())
        assert is_herm is True

    def test_verify_hermitian_pauli_y(self, verifier):
        """Test Pauli Y is Hermitian."""
        is_herm, _ = verifier.verify_hermitian(pauli_y())
        assert is_herm is True

    def test_verify_hermitian_pauli_z(self, verifier):
        """Test Pauli Z is Hermitian."""
        is_herm, _ = verifier.verify_hermitian(pauli_z())
        assert is_herm is True

    def test_verify_hermitian_identity(self, verifier):
        """Test Identity is Hermitian."""
        is_herm, _ = verifier.verify_hermitian(identity_2())
        assert is_herm is True

    def test_verify_hermitian_non_hermitian_fails(self, verifier):
        """Test non-Hermitian matrix fails verification."""
        # This matrix is not Hermitian: [[0, 1], [0, 0]]
        non_herm = Matrix([[0, 1], [0, 0]])
        is_herm, _ = verifier.verify_hermitian(non_herm)
        assert is_herm is False


class TestEntanglementVerification:
    """Test Section 2: Entanglement & Information Theory."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    # --- Partial Trace Tests ---

    def test_partial_trace_product_state(self, verifier):
        """Test partial trace of product state |00⟩⟨00| gives |0⟩⟨0|."""
        # |00⟩ = |0⟩ ⊗ |0⟩
        state = TensorProduct(ket_0(), ket_0())
        rho = state * state.adjoint()

        # Trace out system B
        rho_A = verifier.partial_trace(rho, trace_out=1)

        # Should equal |0⟩⟨0|
        expected = ket_0() * bra_0()
        assert simplify(rho_A - expected) == zeros(2, 2)

    def test_partial_trace_bell_state_gives_maximally_mixed(self, verifier):
        """Test partial trace of Bell state gives I/2."""
        bell = bell_phi_plus()
        rho = bell * bell.adjoint()

        # Trace out system B
        rho_A = verifier.partial_trace(rho, trace_out=1)

        # Should equal I/2 (maximally mixed)
        expected = eye(2) / 2
        assert simplify(rho_A - expected) == zeros(2, 2)

    # --- Von Neumann Entropy Tests ---

    def test_entropy_pure_state_is_zero(self, verifier):
        """Test S(|0⟩⟨0|) = 0 for pure state."""
        pure = ket_0() * bra_0()
        entropy = verifier.calculate_von_neumann_entropy(pure)
        assert simplify(entropy) == 0

    def test_entropy_maximally_mixed_is_ln2(self, verifier):
        """Test S(I/2) = ln(2) for maximally mixed state."""
        mixed = eye(2) / 2
        entropy = verifier.calculate_von_neumann_entropy(mixed)
        assert simplify(entropy - ln(2)) == 0

    # --- Bell State Properties Tests ---

    def test_bell_phi_plus_properties(self, verifier):
        """Test |Φ+⟩ is normalized and maximally entangled."""
        results = verifier.verify_bell_state_properties(bell_phi_plus())
        assert results['normalized'] is True
        assert results['maximally_entangled'] is True
        assert results['reduced_state_is_maximally_mixed'] is True

    def test_bell_phi_minus_properties(self, verifier):
        """Test |Φ-⟩ is normalized and maximally entangled."""
        results = verifier.verify_bell_state_properties(bell_phi_minus())
        assert results['normalized'] is True
        assert results['maximally_entangled'] is True

    def test_bell_psi_plus_properties(self, verifier):
        """Test |Ψ+⟩ is normalized and maximally entangled."""
        results = verifier.verify_bell_state_properties(bell_psi_plus())
        assert results['normalized'] is True
        assert results['maximally_entangled'] is True

    def test_bell_psi_minus_properties(self, verifier):
        """Test |Ψ-⟩ is normalized and maximally entangled."""
        results = verifier.verify_bell_state_properties(bell_psi_minus())
        assert results['normalized'] is True
        assert results['maximally_entangled'] is True


class TestCHSHInequality:
    """Test Section 3: CHSH Inequality Verification."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_chsh_optimal_settings_maximal_violation(self, verifier):
        """Test CHSH with optimal measurement settings gives S = 2√2."""
        # Optimal settings for |Φ+⟩
        # Alice: angles 0 and π/2
        # Bob: angles π/4 and 3π/4
        A = pauli_z()  # angle 0
        A_prime = pauli_x()  # angle π/2
        B = (pauli_z() + pauli_x()) / sqrt(2)  # angle π/4
        B_prime = (-pauli_z() + pauli_x()) / sqrt(2)  # angle 3π/4

        result = verifier.verify_chsh_inequality(
            alice_ops=(A, A_prime),
            bob_ops=(B, B_prime),
            state_vector=bell_phi_plus()
        )

        # S should be 2√2
        S = simplify(result['S_simplified'])
        assert simplify(S - 2*sqrt(2)) == 0 or simplify(S + 2*sqrt(2)) == 0

        # Should violate classical bound
        assert result['violates_classical_bound'] is True

    def test_chsh_z_z_measurement_no_violation(self, verifier):
        """Test CHSH with aligned measurements doesn't maximally violate."""
        # If Alice and Bob both measure in Z basis only
        A = pauli_z()
        A_prime = pauli_z()
        B = pauli_z()
        B_prime = pauli_z()

        result = verifier.verify_chsh_inequality(
            alice_ops=(A, A_prime),
            bob_ops=(B, B_prime),
            state_vector=bell_phi_plus()
        )

        # S = 1 - 1 + 1 + 1 = 2, which is the classical bound
        S_val = float(simplify(result['S_simplified']).evalf())
        assert abs(S_val) <= 2.0 + 1e-10  # Should be at or below classical bound


class TestStabilizerFormalism:
    """Test Section 4: Stabilizer Formalism."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_pauli_string_xx(self, verifier):
        """Test construction of XX Pauli string."""
        XX = verifier.get_pauli_string(['X', 'X'])
        expected = TensorProduct(pauli_x(), pauli_x())
        assert simplify(XX - expected) == zeros(4, 4)

    def test_pauli_string_zz(self, verifier):
        """Test construction of ZZ Pauli string."""
        ZZ = verifier.get_pauli_string(['Z', 'Z'])
        expected = TensorProduct(pauli_z(), pauli_z())
        assert simplify(ZZ - expected) == zeros(4, 4)

    def test_bell_phi_plus_stabilized_by_xx(self, verifier):
        """Test |Φ+⟩ is stabilized by XX."""
        XX = verifier.get_pauli_string(['X', 'X'])
        is_stab, eigenval = verifier.verify_stabilizer(XX, bell_phi_plus())
        assert is_stab is True
        assert eigenval == 1

    def test_bell_phi_plus_stabilized_by_zz(self, verifier):
        """Test |Φ+⟩ is stabilized by ZZ."""
        ZZ = verifier.get_pauli_string(['Z', 'Z'])
        is_stab, eigenval = verifier.verify_stabilizer(ZZ, bell_phi_plus())
        assert is_stab is True
        assert eigenval == 1

    def test_bell_psi_minus_eigenvalue_xx(self, verifier):
        """Test |Ψ-⟩ has eigenvalue -1 for XX."""
        XX = verifier.get_pauli_string(['X', 'X'])
        is_stab, eigenval = verifier.verify_stabilizer(XX, bell_psi_minus())
        assert is_stab is False  # Not +1 eigenstate
        assert eigenval == -1    # But is -1 eigenstate

    def test_3qubit_bit_flip_code_verification(self, verifier):
        """Test 3-qubit bit flip code stabilizers."""
        results = verifier.verify_3qubit_bit_flip_code()

        # Logical states should be stabilized
        assert results['logical_0_stabilized_by_ZZI'] is True
        assert results['logical_0_stabilized_by_IZZ'] is True
        assert results['logical_1_stabilized_by_ZZI'] is True
        assert results['logical_1_stabilized_by_IZZ'] is True

        # Error detection syndromes should distinguish errors
        syndromes = results['error_detection']

        # X1 error: ZZI = -1, IZZ = +1
        assert syndromes['X1_error']['ZZI_syndrome'] == -1
        assert syndromes['X1_error']['IZZ_syndrome'] == 1

        # X2 error: ZZI = -1, IZZ = -1
        assert syndromes['X2_error']['ZZI_syndrome'] == -1
        assert syndromes['X2_error']['IZZ_syndrome'] == -1

        # X3 error: ZZI = +1, IZZ = -1
        assert syndromes['X3_error']['ZZI_syndrome'] == 1
        assert syndromes['X3_error']['IZZ_syndrome'] == -1


class TestShor9QubitCode:
    """Test Shor's 9-qubit quantum error correcting code (Step 6)."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    # --- Logical State Tests ---

    def test_logical_states_are_512_dimensional(self, verifier):
        """Test logical states have correct dimension (2^9 = 512)."""
        logical_0, logical_1 = verifier.get_shor_9qubit_logical_states()
        assert logical_0.shape == (512, 1)
        assert logical_1.shape == (512, 1)

    def test_logical_states_normalized(self, verifier):
        """Test |0_L⟩ and |1_L⟩ are normalized."""
        logical_0, logical_1 = verifier.get_shor_9qubit_logical_states()

        is_norm_0, _ = verifier.verify_normalization(logical_0)
        is_norm_1, _ = verifier.verify_normalization(logical_1)

        assert is_norm_0 is True
        assert is_norm_1 is True

    def test_logical_states_orthogonal(self, verifier):
        """Test ⟨0_L|1_L⟩ = 0."""
        logical_0, logical_1 = verifier.get_shor_9qubit_logical_states()
        inner_product = simplify((logical_0.adjoint() * logical_1)[0, 0])
        assert inner_product == 0

    # --- Stabilizer Tests ---

    def test_stabilizer_count(self, verifier):
        """Test there are 8 stabilizer generators."""
        stabilizers = verifier.get_shor_9qubit_stabilizers()
        assert len(stabilizers) == 8

    def test_z_stabilizers_are_512x512(self, verifier):
        """Test Z-type stabilizers have correct dimension."""
        stabilizers = verifier.get_shor_9qubit_stabilizers()
        for name in ['Z1Z2', 'Z2Z3', 'Z4Z5', 'Z5Z6', 'Z7Z8', 'Z8Z9']:
            assert stabilizers[name].shape == (512, 512)

    def test_x_stabilizers_are_512x512(self, verifier):
        """Test X-type stabilizers have correct dimension."""
        stabilizers = verifier.get_shor_9qubit_stabilizers()
        for name in ['X1-6', 'X4-9']:
            assert stabilizers[name].shape == (512, 512)

    def test_all_stabilizers_stabilize_logical_0(self, verifier):
        """Test all 8 stabilizers have eigenvalue +1 on |0_L⟩."""
        logical_0, _ = verifier.get_shor_9qubit_logical_states()
        stabilizers = verifier.get_shor_9qubit_stabilizers()

        for name, stab in stabilizers.items():
            is_stab, eigenval = verifier.verify_stabilizer(stab, logical_0)
            assert is_stab is True, f"{name} should stabilize |0_L⟩"
            assert eigenval == 1, f"{name} should have eigenvalue +1 on |0_L⟩"

    def test_all_stabilizers_stabilize_logical_1(self, verifier):
        """Test all 8 stabilizers have eigenvalue +1 on |1_L⟩."""
        _, logical_1 = verifier.get_shor_9qubit_logical_states()
        stabilizers = verifier.get_shor_9qubit_stabilizers()

        for name, stab in stabilizers.items():
            is_stab, eigenval = verifier.verify_stabilizer(stab, logical_1)
            assert is_stab is True, f"{name} should stabilize |1_L⟩"
            assert eigenval == 1, f"{name} should have eigenvalue +1 on |1_L⟩"

    # --- Error Detection Tests ---

    def test_x_error_detected_by_z_stabilizers(self, verifier):
        """Test X errors are detected by Z-type stabilizers."""
        logical_0, _ = verifier.get_shor_9qubit_logical_states()
        stabilizers = verifier.get_shor_9qubit_stabilizers()

        # X error on qubit 1 should be detected by Z1Z2
        X1 = verifier.get_pauli_string(['X', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I'])
        error_state = simplify(X1 * logical_0)

        _, eigenval = verifier.verify_stabilizer(stabilizers['Z1Z2'], error_state)
        assert eigenval == -1, "Z1Z2 should detect X1 error"

    def test_z_error_detected_by_x_stabilizers(self, verifier):
        """Test Z errors are detected by X-type stabilizers."""
        logical_0, _ = verifier.get_shor_9qubit_logical_states()
        stabilizers = verifier.get_shor_9qubit_stabilizers()

        # Z error on qubit 1 should be detected by X1-6
        Z1 = verifier.get_pauli_string(['Z', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I'])
        error_state = simplify(Z1 * logical_0)

        _, eigenval = verifier.verify_stabilizer(stabilizers['X1-6'], error_state)
        assert eigenval == -1, "X1-6 should detect Z1 error"

    def test_y_error_detected_by_both_stabilizer_types(self, verifier):
        """Test Y errors are detected by both X and Z stabilizers (Y = iXZ)."""
        logical_0, _ = verifier.get_shor_9qubit_logical_states()
        stabilizers = verifier.get_shor_9qubit_stabilizers()

        # Y error on qubit 1 = iXZ, should trigger both Z1Z2 and X1-6
        Y1 = verifier.get_pauli_string(['Y', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I'])
        error_state = simplify(Y1 * logical_0)

        _, z_eigenval = verifier.verify_stabilizer(stabilizers['Z1Z2'], error_state)
        _, x_eigenval = verifier.verify_stabilizer(stabilizers['X1-6'], error_state)

        assert z_eigenval == -1, "Z1Z2 should detect Y1 error (X component)"
        assert x_eigenval == -1, "X1-6 should detect Y1 error (Z component)"

    # --- Full Verification Tests ---

    def test_verify_shor_9qubit_code_returns_valid_structure(self, verifier):
        """Test verify_shor_9qubit_code returns expected structure."""
        results = verifier.verify_shor_9qubit_code()

        assert results['code'] == 'Shor 9-qubit'
        assert results['physical_qubits'] == 9
        assert results['logical_qubits'] == 1
        assert results['distance'] == 3
        assert 'stabilizer_generators' in results
        assert 'logical_states_valid' in results
        assert 'stabilizer_verification' in results
        assert 'error_syndromes' in results
        assert 'syndrome_analysis' in results

    def test_logical_states_pass_verification(self, verifier):
        """Test logical states pass all verification checks."""
        results = verifier.verify_shor_9qubit_code()

        assert results['logical_states_valid']['|0_L⟩_normalized'] is True
        assert results['logical_states_valid']['|1_L⟩_normalized'] is True
        assert results['logical_states_valid']['orthogonal'] is True

    def test_all_stabilizers_pass_verification(self, verifier):
        """Test all stabilizers pass verification on both logical states."""
        results = verifier.verify_shor_9qubit_code()

        for stab_name, stab_data in results['stabilizer_verification'].items():
            assert stab_data['stabilizes_|0_L⟩'] is True, f"{stab_name} should stabilize |0_L⟩"
            assert stab_data['stabilizes_|1_L⟩'] is True, f"{stab_name} should stabilize |1_L⟩"
            assert stab_data['eigenvalue_|0_L⟩'] == 1, f"{stab_name} should have +1 eigenvalue on |0_L⟩"
            assert stab_data['eigenvalue_|1_L⟩'] == 1, f"{stab_name} should have +1 eigenvalue on |1_L⟩"

    def test_x_errors_have_unique_syndromes(self, verifier):
        """Test all 9 X errors have distinguishable syndromes."""
        results = verifier.verify_shor_9qubit_code()
        assert results['syndrome_analysis']['x_errors_distinguishable'] is True

    def test_z_errors_distinguishable_by_block(self, verifier):
        """
        Test Z errors are distinguishable at the BLOCK level.

        In Shor's code, Z errors within the same block have identical syndromes
        because the X-type stabilizers (X1-6, X4-9) detect phase flips between
        blocks, not within blocks. This is by design:
        - Z1, Z2, Z3 (block 1): same syndrome
        - Z4, Z5, Z6 (block 2): same syndrome
        - Z7, Z8, Z9 (block 3): same syndrome

        The 3 blocks have 3 distinct syndromes, enabling block-level correction.
        """
        results = verifier.verify_shor_9qubit_code()

        # Extract Z error syndromes
        z_syndromes = {
            name: tuple(syn.values())
            for name, syn in results['error_syndromes'].items()
            if name.startswith('Z')
        }

        # Group by block
        block_1 = [z_syndromes[f'Z{i}'] for i in [1, 2, 3]]
        block_2 = [z_syndromes[f'Z{i}'] for i in [4, 5, 6]]
        block_3 = [z_syndromes[f'Z{i}'] for i in [7, 8, 9]]

        # Within each block, all Z errors have same syndrome
        assert len(set(block_1)) == 1, "Z errors in block 1 should have same syndrome"
        assert len(set(block_2)) == 1, "Z errors in block 2 should have same syndrome"
        assert len(set(block_3)) == 1, "Z errors in block 3 should have same syndrome"

        # Between blocks, syndromes are distinct
        block_syndromes = [block_1[0], block_2[0], block_3[0]]
        assert len(set(block_syndromes)) == 3, "Each block should have a unique syndrome"

    def test_27_error_patterns_generated(self, verifier):
        """Test all 27 single-qubit Pauli errors are analyzed (9 qubits × 3 Paulis)."""
        results = verifier.verify_shor_9qubit_code()
        assert results['syndrome_analysis']['total_error_patterns'] == 27

    # --- Syndrome Table Tests ---

    def test_syndrome_table_format(self, verifier):
        """Test syndrome table generates valid output."""
        table = verifier.get_shor_syndrome_table()

        assert "Shor's 9-Qubit Code" in table
        assert "Z1Z2" in table
        assert "X1-6" in table
        assert "X1" in table  # First X error
        assert "Z9" in table  # Last Z error


class TestQuickVerificationFunctions:
    """Test convenience functions for quick verification."""

    def test_quick_verify_bell_states_all_pass(self):
        """Test all Bell states pass quick verification."""
        results = quick_verify_bell_states()

        for state_name in ['Phi+', 'Phi-', 'Psi+', 'Psi-']:
            assert results[state_name]['normalized'] is True
            assert results[state_name]['maximally_entangled'] is True

    def test_quick_verify_pauli_matrices(self):
        """Test all Pauli matrices are unitary and Hermitian."""
        results = quick_verify_pauli_matrices()

        for pauli in ['X', 'Y', 'Z']:
            assert results[f'{pauli}_hermitian'] is True
            assert results[f'{pauli}_unitary'] is True

    def test_quick_verify_common_gates(self):
        """Test common gates verification."""
        results = quick_verify_common_gates()

        # All should be unitary
        for gate_name in results:
            assert results[gate_name]['unitary'] is True

        # Pauli gates and Hadamard are also Hermitian
        assert results['Pauli_X']['hermitian'] is True
        assert results['Pauli_Y']['hermitian'] is True
        assert results['Pauli_Z']['hermitian'] is True
        assert results['Hadamard']['hermitian'] is True

        # CNOT is Hermitian too (it's its own inverse)
        assert results['CNOT']['hermitian'] is True


class TestReportGeneration:
    """Test verification report generation."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_generate_verification_report_format(self, verifier):
        """Test report generation produces valid markdown."""
        data = {
            'normalized': True,
            'unitary': True,
            'hermitian': False
        }
        references = [
            'Nielsen & Chuang, Chapter 2',
            'Preskill Lecture Notes'
        ]

        report = verifier.generate_verification_report(
            'Test System',
            data,
            references
        )

        assert '# Symbolic Verification Report: Test System' in report
        assert 'PASS' in report  # For True values
        assert 'FAIL' in report  # For False value
        assert 'Nielsen & Chuang' in report
        assert 'SymPy' in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests verifying the complete verification pipeline."""

    @pytest.fixture
    def verifier(self):
        return QuantumVerifier()

    def test_complete_bell_state_verification_pipeline(self, verifier):
        """Test complete verification pipeline for Bell states."""
        # 1. Verify all Bell states
        states = {
            'Phi+': bell_phi_plus(),
            'Phi-': bell_phi_minus(),
            'Psi+': bell_psi_plus(),
            'Psi-': bell_psi_minus()
        }

        for name, state in states.items():
            # Check normalization
            is_norm, _ = verifier.verify_normalization(state)
            assert is_norm, f"{name} should be normalized"

            # Check maximal entanglement
            is_max, entropy = verifier.verify_maximally_entangled(state)
            assert is_max, f"{name} should be maximally entangled"
            assert simplify(entropy - ln(2)) == 0, f"{name} entropy should be ln(2)"

    def test_teleportation_protocol_unitaries(self, verifier):
        """Test all gates used in teleportation are valid."""
        gates = [
            ('Hadamard', hadamard()),
            ('CNOT', cnot()),
            ('Pauli X', pauli_x()),
            ('Pauli Z', pauli_z())
        ]

        for name, gate in gates:
            is_unitary, _ = verifier.verify_unitary(gate)
            assert is_unitary, f"{name} should be unitary"

    def test_error_correction_syndrome_uniqueness(self, verifier):
        """Test that error syndromes uniquely identify errors."""
        results = verifier.verify_3qubit_bit_flip_code()
        syndromes = results['error_detection']

        # Extract syndrome pairs
        syndrome_map = {}
        for error, data in syndromes.items():
            syndrome = (data['ZZI_syndrome'], data['IZZ_syndrome'])
            assert syndrome not in syndrome_map.values(), \
                f"Syndrome {syndrome} is not unique - can't distinguish errors"
            syndrome_map[error] = syndrome

        # Verify all syndromes are different
        assert len(set(syndrome_map.values())) == 3, \
            "All three single-qubit errors should have unique syndromes"
