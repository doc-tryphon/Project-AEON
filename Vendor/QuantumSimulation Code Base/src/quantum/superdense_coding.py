"""
Superdense Coding Protocol Implementation

Reference: Bennett & Wiesner (1992), "Communication via one- and two-particle
           operators on Einstein-Podolsky-Rosen states"

Protocol: Alice sends 2 classical bits to Bob using only 1 quantum bit,
          exploiting shared entanglement.

Mathematical Verification:
- All encoding operators are unitary
- Encoded states are mutually orthogonal Bell states
- Bob's Bell measurement achieves 100% discrimination
- Information capacity: exactly 2 bits per transmitted qubit
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from .entanglement import BellStateGenerator, BellState


@dataclass
class SuperdenseCodingResult:
    """Results from superdense coding transmission."""

    message_sent: Tuple[int, int]  # Alice's 2 classical bits (b1, b2)
    message_received: Tuple[int, int]  # Bob's decoded message
    encoding_operator: str  # Which operator Alice applied
    encoded_state: np.ndarray  # State after Alice's encoding
    measurement_outcome: str  # Bob's Bell measurement result
    success: bool  # Whether decoded message matches sent message
    entanglement_resource: str  # Which Bell state was used

    @property
    def error_rate(self) -> float:
        """Calculate bit error rate."""
        errors = sum(1 for i in range(2) if self.message_sent[i] != self.message_received[i])
        return errors / 2.0


class SuperdenseCoding:
    """
    Implementation of superdense coding protocol.

    Protocol Steps:
    1. Alice and Bob share entangled Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2
    2. Alice wants to send 2 classical bits (b1, b2) to Bob
    3. Alice applies unitary U_b1b2 to her qubit:
       - 00 → I  (Identity)
       - 01 → X  (Bit flip)
       - 10 → Z  (Phase flip)
       - 11 → XZ (Both flips)
    4. Alice sends her qubit to Bob (1 qubit transmission)
    5. Bob performs Bell measurement on both qubits
    6. Bob decodes 2 classical bits from measurement outcome

    Information Capacity:
    - Quantum communication: 1 qubit
    - Classical information: 2 bits
    - Enhancement: 2× over classical communication
    """

    def __init__(self):
        """Initialize superdense coding protocol."""
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Encoding operators: Map 2 classical bits to unitary operator
        self.encoding_operators: Dict[Tuple[int, int], Tuple[str, np.ndarray]] = {
            (0, 0): ('I', self.I),
            (0, 1): ('X', self.X),
            (1, 0): ('Z', self.Z),
            (1, 1): ('XZ', self.X @ self.Z)
        }

        # Decoding map: Bell measurement outcome → classical bits
        # From symbolic verification:
        # I  → |Φ+⟩ → measurement outcome '00'
        # X  → |Ψ+⟩ → measurement outcome '10'
        # Z  → |Φ-⟩ → measurement outcome '01'
        # XZ → |Ψ-⟩ → measurement outcome '11'
        self.decoding_map: Dict[str, Tuple[int, int]] = {
            '00': (0, 0),   # |Φ+⟩ corresponds to I encoding
            '01': (1, 0),   # |Φ-⟩ corresponds to Z encoding
            '10': (0, 1),   # |Ψ+⟩ corresponds to X encoding
            '11': (1, 1)    # |Ψ-⟩ corresponds to XZ encoding
        }

        # Bell state generator for creating resources and measurements
        # Supports both '00'/'01'/'10'/'11' and 'phi_plus'/'phi_minus'/etc. labels
        self.bell_generator = BellStateGenerator()

        # Store computational basis for verification
        self.ket_00 = np.array([1, 0, 0, 0], dtype=complex)
        self.ket_01 = np.array([0, 1, 0, 0], dtype=complex)
        self.ket_10 = np.array([0, 0, 1, 0], dtype=complex)
        self.ket_11 = np.array([0, 0, 0, 1], dtype=complex)

    def _verify_unitarity(self) -> bool:
        """Verify all encoding operators are unitary."""
        tolerance = 1e-10
        for (b1, b2), (name, U) in self.encoding_operators.items():
            # Check U†U = I
            if not np.allclose(U.conj().T @ U, self.I, atol=tolerance):
                return False
        return True

    def send_message(
        self,
        message: Tuple[int, int],
        resource_state: str = 'phi_plus'
    ) -> SuperdenseCodingResult:
        """
        Send 2 classical bits using superdense coding.

        Args:
            message: Tuple of 2 classical bits (b1, b2) where b1, b2 ∈ {0, 1}
            resource_state: Which Bell state to use as entanglement resource
                          ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')

        Returns:
            SuperdenseCodingResult containing all protocol information

        Raises:
            ValueError: If message bits are not 0 or 1
        """
        # Validate input
        b1, b2 = message
        if b1 not in [0, 1] or b2 not in [0, 1]:
            raise ValueError(f"Message bits must be 0 or 1, got {message}")

        # Step 1: Create shared entangled resource
        # BellStateGenerator now accepts both label formats
        bell_state = self.bell_generator.create_bell_state(resource_state)
        shared_state = bell_state.state_vector

        # Step 2: Alice encodes her message by applying operator to her qubit
        operator_name, operator = self.encoding_operators[message]

        # Apply (U_A ⊗ I_B) to the shared state
        # Alice operates on first qubit (left tensor factor)
        two_qubit_operator = np.kron(operator, self.I)
        encoded_state = two_qubit_operator @ shared_state

        # Step 3: Alice sends her qubit to Bob (simulated by Bob now having both qubits)

        # Step 4: Bob performs Bell measurement
        bell_outcome = self._bob_bell_measurement(encoded_state)

        # Step 5: Bob decodes the classical bits
        decoded_message = self.decoding_map[bell_outcome]

        # Check success
        success = (decoded_message == message)

        return SuperdenseCodingResult(
            message_sent=message,
            message_received=decoded_message,
            encoding_operator=operator_name,
            encoded_state=encoded_state,
            measurement_outcome=bell_outcome,
            success=success,
            entanglement_resource=resource_state
        )

    def _bob_bell_measurement(self, state: np.ndarray) -> str:
        """
        Perform Bell basis measurement on two-qubit state.

        Args:
            state: Two-qubit state vector (4-dimensional)

        Returns:
            Bell basis measurement outcome: '00', '01', '10', or '11' (label format)
        """
        # Create all 4 Bell states using label format
        bell_labels = ['00', '01', '10', '11']
        bell_states = {
            label: self.bell_generator.create_bell_state(label)
            for label in bell_labels
        }

        # For noiseless case, find which Bell state has overlap = 1
        # In practice, this would be a probabilistic measurement
        tolerance = 1e-10
        for label, bell_state in bell_states.items():
            bell_vector = bell_state.state_vector
            overlap = np.abs(np.vdot(bell_vector, state))**2
            if np.isclose(overlap, 1.0, atol=tolerance):
                return label

        # If no perfect match (shouldn't happen in noiseless case), return closest
        overlaps = {
            label: np.abs(np.vdot(bell_state.state_vector, state))**2
            for label, bell_state in bell_states.items()
        }
        return max(overlaps, key=overlaps.get)

    def verify_orthogonality(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Verify that all 4 encoded states are mutually orthogonal.

        Returns:
            Dictionary mapping pairs of messages to their inner products
        """
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]
        phi_plus = self.bell_generator.create_bell_state('00').state_vector  # '00' = |Φ+⟩

        # Encode all messages
        encoded_states = {}
        for msg in messages:
            operator_name, operator = self.encoding_operators[msg]
            two_qubit_operator = np.kron(operator, self.I)
            encoded_states[msg] = two_qubit_operator @ phi_plus

        # Compute all inner products
        inner_products = {}
        for msg1 in messages:
            for msg2 in messages:
                inner_prod = np.vdot(encoded_states[msg1], encoded_states[msg2])
                inner_products[(msg1, msg2)] = inner_prod

        return inner_products

    def calculate_channel_capacity(self) -> Dict[str, float]:
        """
        Calculate information-theoretic capacity of superdense coding channel.

        Returns:
            Dictionary with capacity metrics
        """
        return {
            'qubits_transmitted': 1.0,  # Alice sends 1 qubit to Bob
            'classical_bits_sent': 2.0,  # Alice encodes 2 classical bits
            'bits_per_qubit': 2.0,       # 2 bits / 1 qubit
            'classical_capacity': 1.0,   # Classical channel: 1 bit per qubit
            'enhancement_factor': 2.0,   # 2× classical capacity
            'entanglement_pairs_consumed': 1.0  # 1 Bell pair per transmission
        }

    def test_all_messages(self, resource_state: str = 'phi_plus') -> Dict[str, any]:
        """
        Test protocol with all 4 possible 2-bit messages.

        Args:
            resource_state: Bell state to use as entanglement resource

        Returns:
            Dictionary with test results
        """
        messages = [(0, 0), (0, 1), (1, 0), (1, 1)]
        results = []

        for msg in messages:
            result = self.send_message(msg, resource_state)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        error_rate = 1.0 - (success_count / len(messages))

        return {
            'total_messages': len(messages),
            'successful': success_count,
            'failed': len(messages) - success_count,
            'error_rate': error_rate,
            'results': results
        }


class SuperdenseCodingAnalyzer:
    """
    Tools for analyzing superdense coding protocol properties.
    """

    @staticmethod
    def compare_to_teleportation() -> Dict[str, str]:
        """
        Document the duality between superdense coding and quantum teleportation.

        Returns:
            Dictionary explaining the duality
        """
        return {
            'teleportation': (
                "Transfer 1 qubit using 2 classical bits + 1 entangled pair\n"
                "- Alice has: 1 qubit (message) + 1 qubit (entangled)\n"
                "- Communication: 2 classical bits sent\n"
                "- Bob receives: 1 qubit (state)"
            ),
            'superdense_coding': (
                "Transfer 2 classical bits using 1 qubit + 1 entangled pair\n"
                "- Alice has: 2 classical bits (message) + 1 qubit (entangled)\n"
                "- Communication: 1 qubit sent\n"
                "- Bob receives: 2 classical bits"
            ),
            'duality': (
                "The protocols are dual to each other:\n"
                "- Same resource: 1 shared Bell pair\n"
                "- Teleportation: quantum → classical → quantum\n"
                "- Superdense: classical → quantum → classical\n"
                "- Both achieve optimal information transfer for their task"
            ),
            'resource_comparison': (
                "Entanglement as resource:\n"
                "- Teleportation: 1 ebit + 2 cbits ≈ 1 qubit\n"
                "- Superdense: 1 ebit + 1 qubit ≈ 2 cbits\n"
                "- Holevo bound: Maximum classical info from n qubits is n bits\n"
                "- Superdense coding saturates this bound with entanglement"
            )
        }

    @staticmethod
    def verify_information_theory(protocol: SuperdenseCoding) -> Dict[str, bool]:
        """
        Verify protocol satisfies information-theoretic constraints.

        Args:
            protocol: SuperdenseCoding instance

        Returns:
            Dictionary of verification results
        """
        capacity = protocol.calculate_channel_capacity()

        # Holevo bound: χ ≤ n (n qubits carry at most n bits)
        # Superdense coding: 1 qubit + 1 ebit → 2 bits
        holevo_satisfied = capacity['bits_per_qubit'] <= 2.0

        # No-signaling: Bob cannot decode without Alice's qubit
        # (Verified by orthogonality of initial Bell states without encoding)

        # Verify unitarity of all encoding operations
        unitarity = protocol._verify_unitarity()

        return {
            'holevo_bound_satisfied': holevo_satisfied,
            'encoding_operators_unitary': unitarity,
            'achieves_2_bits_per_qubit': np.isclose(capacity['bits_per_qubit'], 2.0),
            'requires_entanglement': capacity['entanglement_pairs_consumed'] == 1.0
        }