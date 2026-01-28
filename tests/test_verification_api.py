"""
Tests for the TutorVerificationAPI.

These tests verify that the verification API correctly wraps
the QuantumVerifier and returns properly structured results.
"""

import pytest
from sympy import Matrix, sqrt, I, simplify

from src.tutor.verification_api import (
    TutorVerificationAPI,
    VerificationResult,
    VerificationDomain,
    VerificationError,
    ParseError,
    parse_matrix_expr,
    parse_state_expr,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def api():
    """Create a fresh TutorVerificationAPI instance."""
    return TutorVerificationAPI()


# =============================================================================
# Test VerificationResult Dataclass
# =============================================================================

class TestVerificationResult:
    """Tests for the VerificationResult dataclass."""

    def test_dataclass_fields(self):
        """Test that VerificationResult has all required fields."""
        result = VerificationResult(
            verified=True,
            symbolic_proof="U^\\dagger U = I",
            explanation="The gate is unitary.",
            confidence=1.0,
            domain="unitarity",
            details={"test": "value"}
        )

        assert result.verified is True
        assert result.symbolic_proof == "U^\\dagger U = I"
        assert result.explanation == "The gate is unitary."
        assert result.confidence == 1.0
        assert result.domain == "unitarity"
        assert result.details == {"test": "value"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = VerificationResult(
            verified=True,
            symbolic_proof="proof",
            explanation="explanation",
            confidence=1.0,
            domain="general",
        )

        d = result.to_dict()
        assert d["verified"] is True
        assert d["symbolic_proof"] == "proof"
        assert d["explanation"] == "explanation"
        assert d["confidence"] == 1.0
        assert d["domain"] == "general"
        assert d["details"] == {}

    def test_default_values(self):
        """Test default values for optional fields."""
        result = VerificationResult(
            verified=False,
            symbolic_proof="",
            explanation="Failed",
        )

        assert result.confidence == 1.0
        assert result.domain == "general"
        assert result.details == {}


# =============================================================================
# Test Parser Helpers
# =============================================================================

class TestParseMatrixExpr:
    """Tests for the parse_matrix_expr helper function."""

    def test_parse_named_gate_hadamard(self):
        """Test parsing Hadamard gate by name."""
        H = parse_matrix_expr("H")
        expected = Matrix([[1, 1], [1, -1]]) / sqrt(2)
        assert simplify(H - expected) == Matrix([[0, 0], [0, 0]])

    def test_parse_named_gate_pauli_x(self):
        """Test parsing Pauli-X gate."""
        X = parse_matrix_expr("X")
        expected = Matrix([[0, 1], [1, 0]])
        assert X == expected

    def test_parse_named_gate_pauli_z(self):
        """Test parsing Pauli-Z gate."""
        Z = parse_matrix_expr("Z")
        expected = Matrix([[1, 0], [0, -1]])
        assert Z == expected

    def test_parse_named_gate_cnot(self):
        """Test parsing CNOT gate."""
        CNOT = parse_matrix_expr("CNOT")
        assert CNOT.shape == (4, 4)
        # Check it swaps |10> and |11>
        assert CNOT[2, 3] == 1
        assert CNOT[3, 2] == 1

    def test_parse_case_insensitive(self):
        """Test that gate names are case insensitive."""
        h1 = parse_matrix_expr("H")
        h2 = parse_matrix_expr("hadamard")
        assert simplify(h1 - h2) == Matrix([[0, 0], [0, 0]])

    def test_parse_invalid_raises_error(self):
        """Test that invalid expressions raise ParseError."""
        with pytest.raises(ParseError):
            parse_matrix_expr("not_a_valid_gate_123")


class TestParseStateExpr:
    """Tests for the parse_state_expr helper function."""

    def test_parse_computational_basis_0(self):
        """Test parsing |0> state."""
        state = parse_state_expr("|0>")
        expected = Matrix([[1], [0]])
        assert state == expected

    def test_parse_computational_basis_1(self):
        """Test parsing |1> state."""
        state = parse_state_expr("|1>")
        expected = Matrix([[0], [1]])
        assert state == expected

    def test_parse_plus_state(self):
        """Test parsing |+> state."""
        state = parse_state_expr("|+>")
        expected = Matrix([[1], [1]]) / sqrt(2)
        assert simplify(state - expected) == Matrix([[0], [0]])

    def test_parse_bell_phi_plus(self):
        """Test parsing Bell Phi+ state."""
        state = parse_state_expr("bell_phi_plus")
        expected = Matrix([[1], [0], [0], [1]]) / sqrt(2)
        assert simplify(state - expected) == Matrix([[0], [0], [0], [0]])

    def test_parse_invalid_raises_error(self):
        """Test that invalid state expressions raise ParseError."""
        with pytest.raises(ParseError):
            parse_state_expr("not_a_valid_state_xyz")


# =============================================================================
# Test TutorVerificationAPI - Gate Verification
# =============================================================================

class TestVerifyGate:
    """Tests for the verify_gate method."""

    def test_hadamard_is_unitary(self, api):
        """Test that Hadamard gate is verified as unitary."""
        result = api.verify_gate("H")

        assert result.verified is True
        assert result.domain == VerificationDomain.UNITARITY.value
        assert result.confidence == 1.0
        assert "unitary" in result.explanation.lower()

    def test_pauli_x_is_unitary(self, api):
        """Test that Pauli-X gate is verified as unitary."""
        result = api.verify_gate("X")
        assert result.verified is True

    def test_pauli_y_is_unitary(self, api):
        """Test that Pauli-Y gate is verified as unitary."""
        result = api.verify_gate("Y")
        assert result.verified is True

    def test_pauli_z_is_unitary(self, api):
        """Test that Pauli-Z gate is verified as unitary."""
        result = api.verify_gate("Z")
        assert result.verified is True

    def test_cnot_is_unitary(self, api):
        """Test that CNOT gate is verified as unitary."""
        result = api.verify_gate("CNOT")
        assert result.verified is True

    def test_identity_is_unitary(self, api):
        """Test that identity gate is verified as unitary."""
        result = api.verify_gate("I")
        assert result.verified is True

    def test_hadamard_is_hermitian(self, api):
        """Test that Hadamard is also Hermitian."""
        result = api.verify_gate("H", check_hermitian=True)

        assert result.verified is True
        assert result.details.get("is_hermitian") is True
        assert "Hermitian" in result.explanation

    def test_invalid_gate_returns_error(self, api):
        """Test that invalid gate input returns a failed result."""
        result = api.verify_gate("invalid_gate_xyz")

        assert result.verified is False
        assert result.confidence == 0.0
        assert "error" in result.details


# =============================================================================
# Test TutorVerificationAPI - State Verification
# =============================================================================

class TestVerifyState:
    """Tests for the verify_state method."""

    def test_zero_state_normalized(self, api):
        """Test that |0> is verified as normalized."""
        result = api.verify_state("|0>")

        assert result.verified is True
        assert result.domain == VerificationDomain.NORMALIZATION.value
        assert "normalized" in result.explanation.lower()

    def test_one_state_normalized(self, api):
        """Test that |1> is verified as normalized."""
        result = api.verify_state("|1>")
        assert result.verified is True

    def test_plus_state_normalized(self, api):
        """Test that |+> is verified as normalized."""
        result = api.verify_state("|+>")
        assert result.verified is True

    def test_minus_state_normalized(self, api):
        """Test that |-> is verified as normalized."""
        result = api.verify_state("|->")
        assert result.verified is True

    def test_bell_phi_plus_normalized(self, api):
        """Test that Bell Phi+ is verified as normalized."""
        result = api.verify_state("bell_phi_plus")
        assert result.verified is True

    def test_bell_phi_plus_entanglement(self, api):
        """Test that Bell Phi+ is verified as maximally entangled."""
        result = api.verify_state("bell_phi_plus", check_entanglement=True)

        assert result.verified is True
        assert result.domain == VerificationDomain.ENTANGLEMENT.value
        assert result.details.get("is_maximally_entangled") is True
        assert "maximally entangled" in result.explanation.lower()

    def test_invalid_state_returns_error(self, api):
        """Test that invalid state input returns a failed result."""
        result = api.verify_state("invalid_state_xyz")

        assert result.verified is False
        assert result.confidence == 0.0


# =============================================================================
# Test TutorVerificationAPI - Operator Verification
# =============================================================================

class TestVerifyOperator:
    """Tests for the verify_operator method."""

    def test_pauli_z_hermitian(self, api):
        """Test that Pauli-Z is verified as Hermitian."""
        result = api.verify_operator("Z")

        assert result.verified is True
        assert result.domain == VerificationDomain.HERMITICITY.value
        assert "Hermitian" in result.explanation

    def test_pauli_x_hermitian(self, api):
        """Test that Pauli-X is verified as Hermitian."""
        result = api.verify_operator("X")
        assert result.verified is True

    def test_pauli_y_hermitian(self, api):
        """Test that Pauli-Y is verified as Hermitian."""
        result = api.verify_operator("Y")
        assert result.verified is True

    def test_hadamard_hermitian(self, api):
        """Test that Hadamard is verified as Hermitian."""
        result = api.verify_operator("H")
        assert result.verified is True

    def test_invalid_operator_returns_error(self, api):
        """Test that invalid operator input returns a failed result."""
        result = api.verify_operator("invalid_op_xyz")

        assert result.verified is False
        assert result.confidence == 0.0


# =============================================================================
# Test TutorVerificationAPI - Claim Verification
# =============================================================================

class TestVerifyClaim:
    """Tests for the verify_claim convenience method."""

    def test_claim_hadamard_unitary(self, api):
        """Test parsing and verifying 'Hadamard is unitary'."""
        result = api.verify_claim("Hadamard is unitary")
        assert result.verified is True
        assert result.domain == VerificationDomain.UNITARITY.value

    def test_claim_x_hermitian(self, api):
        """Test parsing and verifying 'X is Hermitian'."""
        result = api.verify_claim("X is Hermitian")
        assert result.verified is True
        assert result.domain == VerificationDomain.HERMITICITY.value

    def test_claim_state_normalized(self, api):
        """Test parsing and verifying 'state is normalized'."""
        result = api.verify_claim("|0> is normalized")
        assert result.verified is True
        assert result.domain == VerificationDomain.NORMALIZATION.value

    def test_claim_unknown_returns_general(self, api):
        """Test that unknown claims return a general error result."""
        result = api.verify_claim("something completely random")

        assert result.verified is False
        assert result.domain == VerificationDomain.GENERAL.value
        assert "Could not understand" in result.explanation


# =============================================================================
# Test TutorVerificationAPI - Bell State Verification
# =============================================================================

class TestVerifyBellState:
    """Tests for the verify_bell_state method."""

    def test_bell_phi_plus_valid(self, api):
        """Test that Bell Phi+ is a valid Bell state."""
        result = api.verify_bell_state("bell_phi_plus")

        assert result.verified is True
        assert result.domain == VerificationDomain.BELL_STATE.value

    def test_bell_phi_minus_valid(self, api):
        """Test that Bell Phi- is a valid Bell state."""
        result = api.verify_bell_state("phi-")
        assert result.verified is True

    def test_bell_psi_plus_valid(self, api):
        """Test that Bell Psi+ is a valid Bell state."""
        result = api.verify_bell_state("psi+")
        assert result.verified is True

    def test_bell_psi_minus_valid(self, api):
        """Test that Bell Psi- is a valid Bell state."""
        result = api.verify_bell_state("psi-")
        assert result.verified is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests verifying end-to-end functionality."""

    def test_full_verification_flow(self, api):
        """Test a complete verification flow with multiple checks."""
        # Verify a gate
        gate_result = api.verify_gate("H", check_hermitian=True)
        assert gate_result.verified is True
        assert gate_result.details.get("is_hermitian") is True

        # Verify a state
        state_result = api.verify_state("bell_phi_plus", check_entanglement=True)
        assert state_result.verified is True
        assert state_result.details.get("is_maximally_entangled") is True

        # Verify an operator
        op_result = api.verify_operator("Z")
        assert op_result.verified is True

    def test_results_are_serializable(self, api):
        """Test that all results can be serialized to dict."""
        results = [
            api.verify_gate("H"),
            api.verify_state("|0>"),
            api.verify_operator("Z"),
            api.verify_claim("X is unitary"),
        ]

        for result in results:
            d = result.to_dict()
            assert isinstance(d, dict)
            assert "verified" in d
            assert "explanation" in d

    def test_api_reusable(self, api):
        """Test that the API can be reused for multiple verifications."""
        # Run multiple verifications on the same API instance
        results = []
        for _ in range(5):
            results.append(api.verify_gate("H"))
            results.append(api.verify_state("|0>"))

        # All should succeed
        assert all(r.verified for r in results)
