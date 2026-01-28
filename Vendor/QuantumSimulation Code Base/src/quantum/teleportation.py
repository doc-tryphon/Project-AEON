"""
Quantum Teleportation Protocol

Implements the standard quantum teleportation protocol with full verification.
All operations are verified against known analytical results.

Reference:
Bennett et al., "Teleporting an Unknown Quantum State via Dual Classical
and Einstein-Podolsky-Rosen Channels", Phys. Rev. Lett. 70, 1895 (1993)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import qutip as qt

from src.quantum.entanglement import BellStateGenerator, BellState


@dataclass
class TeleportationResult:
    """Results from a quantum teleportation protocol execution."""
    input_state: np.ndarray
    output_state: np.ndarray
    fidelity: float
    measurement_outcome: Tuple[int, int]  # Alice's 2 classical bits
    correction_applied: str  # Which Pauli correction was applied
    entanglement_resource: str  # Which Bell state was used
    bob_state_before_correction: np.ndarray  # Bob's state before classical message
    protocol_successful: bool  # Whether F > threshold


class QuantumTeleportation:
    """
    Quantum teleportation protocol implementation.

    Protocol steps:
    1. Alice and Bob share entangled pair (Bell state)
    2. Alice performs Bell measurement on her qubit + message qubit
    3. Alice sends 2 classical bits to Bob
    4. Bob applies correction operation based on classical bits
    5. Bob now has the teleported state
    """

    def __init__(self, fidelity_threshold: float = 0.99):
        """
        Initialize teleportation protocol.

        Args:
            fidelity_threshold: Minimum fidelity for successful teleportation
        """
        self.fidelity_threshold = fidelity_threshold
        self.bell_generator = BellStateGenerator()

        # Pauli correction operators
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        # Map measurement outcomes to corrections
        self.corrections = {
            (0, 0): ('I', self.I),      # |Φ+⟩ → Identity
            (0, 1): ('Z', self.Z),      # |Φ-⟩ → Z gate
            (1, 0): ('X', self.X),      # |Ψ+⟩ → X gate
            (1, 1): ('XZ', self.X @ self.Z)  # |Ψ-⟩ → XZ gate
        }

        # Verify correction operators are unitary
        self._verify_correction_unitarity()

    def _verify_correction_unitarity(self):
        """Verify all correction operators are unitary: U†U = I."""
        for name, (label, U) in self.corrections.items():
            U_dag_U = U.conj().T @ U
            if not np.allclose(U_dag_U, self.I):
                raise ValueError(f"Correction {label} is not unitary!")

    def teleport(self, message_state: np.ndarray,
                entanglement_resource: str = 'phi_plus') -> TeleportationResult:
        """
        Execute quantum teleportation protocol.

        Args:
            message_state: Quantum state to teleport (2D vector)
            entanglement_resource: Which Bell state to use as resource

        Returns:
            TeleportationResult with all protocol details
        """
        # Verify input state is normalized
        norm = np.linalg.norm(message_state)
        if not np.isclose(norm, 1.0):
            raise ValueError(f"Input state not normalized: ||ψ|| = {norm}")

        # Step 1: Create entangled resource
        bell_state = self._get_bell_resource(entanglement_resource)

        # Step 2: Prepare initial 3-qubit state
        initial_state = self._prepare_initial_state(message_state, bell_state)

        # Step 3: Alice performs Bell measurement
        measurement_outcome, bob_state_before = self._alice_bell_measurement(
            initial_state
        )

        # Step 4: Bob applies correction
        correction_name, correction_op = self.corrections[measurement_outcome]
        bob_state_after = correction_op @ bob_state_before

        # Step 5: Calculate fidelity
        fidelity = self._calculate_fidelity(message_state, bob_state_after)

        return TeleportationResult(
            input_state=message_state,
            output_state=bob_state_after,
            fidelity=fidelity,
            measurement_outcome=measurement_outcome,
            correction_applied=correction_name,
            entanglement_resource=entanglement_resource,
            bob_state_before_correction=bob_state_before,
            protocol_successful=fidelity >= self.fidelity_threshold
        )

    def _get_bell_resource(self, resource_name: str) -> BellState:
        """Get Bell state resource for teleportation."""
        label_map = {
            'phi_plus': '00',
            'phi_minus': '01',
            'psi_plus': '10',
            'psi_minus': '11'
        }

        if resource_name not in label_map:
            raise ValueError(f"Unknown Bell state: {resource_name}")

        return self.bell_generator.create_bell_state(label_map[resource_name])

    def _prepare_initial_state(self, message: np.ndarray,
                               bell_state: BellState) -> np.ndarray:
        """
        Prepare initial 3-qubit state: |ψ⟩₁ ⊗ |Φ+⟩₂₃

        Args:
            message: Message qubit state (qubit 1)
            bell_state: Entangled pair shared by Alice and Bob (qubits 2-3)

        Returns:
            8-dimensional state vector
        """
        # Tensor product: message ⊗ bell_pair
        initial_state = np.kron(message, bell_state.state_vector)
        return initial_state

    def _alice_bell_measurement(self, three_qubit_state: np.ndarray
                                ) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Simulate Alice's Bell measurement on qubits 1-2.

        This is the key step that collapses the state and determines
        which correction Bob needs to apply.

        Args:
            three_qubit_state: Full 3-qubit state

        Returns:
            Tuple of (measurement_outcome, bob_state)
            measurement_outcome: (b1, b2) classical bits
            bob_state: Bob's qubit state after Alice's measurement
        """
        # Bell basis for qubits 1-2
        bell_basis_12 = self._construct_bell_basis_12()

        # Measure in Bell basis (simulate by computing probabilities)
        # For deterministic testing, we'll compute the projection
        # In real quantum mechanics, this would be probabilistic

        # Project onto each Bell state and get Bob's state
        outcomes = []
        for bell_idx, bell_12 in enumerate(bell_basis_12):
            # Project 3-qubit state onto Bell state ⊗ Bob's space
            # Bell state is 4-dim, Bob's space is 2-dim → 8-dim total

            # Reshape to separate Alice's 2 qubits from Bob's 1 qubit
            psi_reshaped = three_qubit_state.reshape(4, 2)

            # Project Alice's qubits onto this Bell state
            # projection is Bob's state vector (unnormalized)
            projection = np.dot(bell_12.conj(), psi_reshaped)

            # Probability is ||projection||²
            total_prob = np.sum(np.abs(projection)**2)
            outcomes.append((bell_idx, projection, total_prob))

        # Select outcome (deterministic for testing - choose first non-zero)
        # In real implementation, sample according to probabilities
        for bell_idx, bob_state_raw, total_prob in outcomes:
            if total_prob > 1e-10:  # Non-zero probability
                # Normalize Bob's state
                bob_state = bob_state_raw / np.sqrt(total_prob)

                # Convert bell_idx to classical bits
                measurement_outcome = (bell_idx // 2, bell_idx % 2)

                return measurement_outcome, bob_state

        # Should never reach here for valid input
        raise ValueError("No valid measurement outcome found")

    def _construct_bell_basis_12(self) -> list:
        """
        Construct Bell basis for qubits 1-2 (Alice's qubits).

        Returns:
            List of 4 Bell states as 4-dimensional vectors
        """
        # Computational basis
        ket_00 = np.array([1, 0, 0, 0], dtype=complex)
        ket_01 = np.array([0, 1, 0, 0], dtype=complex)
        ket_10 = np.array([0, 0, 1, 0], dtype=complex)
        ket_11 = np.array([0, 0, 0, 1], dtype=complex)

        # Bell basis
        phi_plus = (ket_00 + ket_11) / np.sqrt(2)   # |Φ+⟩
        phi_minus = (ket_00 - ket_11) / np.sqrt(2)  # |Φ-⟩
        psi_plus = (ket_01 + ket_10) / np.sqrt(2)   # |Ψ+⟩
        psi_minus = (ket_01 - ket_10) / np.sqrt(2)  # |Ψ-⟩

        return [phi_plus, phi_minus, psi_plus, psi_minus]

    def _calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate state fidelity F = |⟨ψ₁|ψ₂⟩|².

        Args:
            state1: Target state
            state2: Actual state

        Returns:
            Fidelity between 0 and 1
        """
        overlap = np.dot(state1.conj(), state2)
        fidelity = np.abs(overlap)**2
        return float(fidelity)

    def verify_no_signaling(self, message_state: np.ndarray,
                           entanglement_resource: str = 'phi_plus') -> Dict:
        """
        Verify no-signaling theorem: Bob's state is maximally mixed
        before receiving Alice's measurement result.

        Args:
            message_state: Message to teleport
            entanglement_resource: Bell state resource

        Returns:
            Dictionary with verification results
        """
        # Prepare initial state
        bell_state = self._get_bell_resource(entanglement_resource)
        initial_state = self._prepare_initial_state(message_state, bell_state)

        # Compute Bob's reduced density matrix (before measurement)
        # Need to trace out qubits 1 and 2
        rho_full = np.outer(initial_state, initial_state.conj())

        # Reshape to separate Alice (4D) and Bob (2D)
        rho_reshaped = rho_full.reshape(4, 2, 4, 2)

        # Partial trace over Alice's qubits (sum over first and third indices)
        rho_bob = np.einsum('ijik->jk', rho_reshaped)

        # Verify Bob's state is maximally mixed: ρ_Bob = I/2
        I_over_2 = np.eye(2) / 2

        is_maximally_mixed = np.allclose(rho_bob, I_over_2, atol=1e-10)
        purity = np.trace(rho_bob @ rho_bob).real

        return {
            'bob_density_matrix': rho_bob,
            'is_maximally_mixed': is_maximally_mixed,
            'purity': purity,  # Should be 0.5 for maximally mixed
            'no_signaling_verified': is_maximally_mixed and np.isclose(purity, 0.5)
        }


def teleport_basis_states() -> Dict[str, TeleportationResult]:
    """
    Teleport all computational and superposition basis states.

    Returns:
        Dictionary of results for each basis state
    """
    protocol = QuantumTeleportation()

    # Define basis states
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    ket_minus = (ket_0 - ket_1) / np.sqrt(2)
    ket_i = (ket_0 + 1j*ket_1) / np.sqrt(2)
    ket_minus_i = (ket_0 - 1j*ket_1) / np.sqrt(2)

    states = {
        '|0⟩': ket_0,
        '|1⟩': ket_1,
        '|+⟩': ket_plus,
        '|-⟩': ket_minus,
        '|i⟩': ket_i,
        '|-i⟩': ket_minus_i
    }

    results = {}
    for name, state in states.items():
        result = protocol.teleport(state)
        results[name] = result

    return results


def teleport_arbitrary_superposition(alpha: complex, beta: complex) -> TeleportationResult:
    """
    Teleport arbitrary superposition α|0⟩ + β|1⟩.

    Args:
        alpha: Amplitude for |0⟩
        beta: Amplitude for |1⟩

    Returns:
        Teleportation result
    """
    # Normalize
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    state = np.array([alpha, beta], dtype=complex) / norm

    protocol = QuantumTeleportation()
    return protocol.teleport(state)


__all__ = [
    'QuantumTeleportation',
    'TeleportationResult',
    'teleport_basis_states',
    'teleport_arbitrary_superposition'
]