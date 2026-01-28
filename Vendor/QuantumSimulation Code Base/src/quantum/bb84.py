"""
BB84 Quantum Key Distribution Protocol

Implements the BB84 protocol for quantum key distribution with full
verification and eavesdropper detection. This is a prepare-and-measure
protocol that does NOT require entanglement.

Security is guaranteed by:
1. No-cloning theorem (Eve cannot copy quantum states)
2. Measurement disturbance (Eve's interception is detectable)
3. Information-theoretic security (not computational)

Reference:
Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key
distribution and coin tossing." Proceedings of IEEE International Conference
on Computers, Systems, and Signal Processing, pp. 175-179.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import secrets


class Basis(Enum):
    """Measurement/preparation basis for BB84."""
    COMPUTATIONAL = 'Z'  # {|0⟩, |1⟩}
    HADAMARD = 'X'       # {|+⟩, |-⟩}


@dataclass
class BB84State:
    """
    One of the four BB84 states.

    The four states form two mutually unbiased bases:
    - Computational (Z): |0⟩, |1⟩
    - Hadamard (X): |+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2
    """
    bit: int           # The classical bit encoded (0 or 1)
    basis: Basis       # Which basis it's encoded in
    state_vector: np.ndarray  # The quantum state
    label: str         # Human-readable label (|0⟩, |1⟩, |+⟩, |−⟩)


@dataclass
class BB84Result:
    """Results from a BB84 protocol execution."""
    # Protocol parameters
    num_qubits_sent: int

    # Raw data (before sifting)
    alice_bits: List[int]
    alice_bases: List[Basis]
    bob_bases: List[Basis]
    bob_measurements: List[int]

    # After sifting
    matching_indices: List[int]
    sifted_key_alice: List[int]
    sifted_key_bob: List[int]
    sifting_rate: float  # Fraction of bits kept after sifting

    # Error analysis
    qber: float  # Quantum Bit Error Rate
    errors_detected: int
    bits_compared: int

    # Security assessment
    eve_detected: bool
    estimated_information_leaked: float
    secure_key_rate: float

    # Final key (if protocol succeeded)
    final_key: Optional[List[int]] = None
    protocol_successful: bool = False


class BB84Protocol:
    """
    BB84 Quantum Key Distribution Protocol.

    The protocol allows Alice and Bob to establish a shared secret key
    over an insecure quantum channel, with the ability to detect any
    eavesdropping attempt by Eve.

    Protocol Steps:
    1. Alice prepares random bits in random bases
    2. Alice sends qubits to Bob over quantum channel
    3. Bob measures each qubit in a random basis
    4. Alice and Bob publicly compare bases (not bits!)
    5. They keep only bits where bases matched (sifting)
    6. They sacrifice some bits to estimate QBER
    7. If QBER < threshold, they have a secure key

    Security Threshold:
    - QBER > 11% indicates eavesdropping (theoretical limit ~14.6%)
    - In practice, any QBER > 0 in ideal simulation indicates Eve
    """

    # QBER threshold for detecting eavesdropping
    # Theoretical limit is ~14.6% (Shor-Preskill bound)
    QBER_THRESHOLD = 0.11

    def __init__(self, num_qubits: int = 100,
                 error_estimation_fraction: float = 0.5):
        """
        Initialize BB84 protocol.

        Args:
            num_qubits: Number of qubits to send in the protocol
            error_estimation_fraction: Fraction of sifted bits used for QBER estimation
        """
        self.num_qubits = num_qubits
        self.error_estimation_fraction = error_estimation_fraction

        # Define the four BB84 states
        self._init_bb84_states()

        # Verify states are properly constructed
        self._verify_bb84_states()

    def _init_bb84_states(self):
        """Initialize the four BB84 states."""
        # Computational basis
        ket_0 = np.array([1, 0], dtype=complex)
        ket_1 = np.array([0, 1], dtype=complex)

        # Hadamard basis
        ket_plus = (ket_0 + ket_1) / np.sqrt(2)
        ket_minus = (ket_0 - ket_1) / np.sqrt(2)

        self.states = {
            (0, Basis.COMPUTATIONAL): BB84State(0, Basis.COMPUTATIONAL, ket_0, '|0⟩'),
            (1, Basis.COMPUTATIONAL): BB84State(1, Basis.COMPUTATIONAL, ket_1, '|1⟩'),
            (0, Basis.HADAMARD): BB84State(0, Basis.HADAMARD, ket_plus, '|+⟩'),
            (1, Basis.HADAMARD): BB84State(1, Basis.HADAMARD, ket_minus, '|−⟩'),
        }

        # Measurement operators
        self.measurement_ops = {
            Basis.COMPUTATIONAL: {
                0: np.outer(ket_0, ket_0.conj()),  # |0⟩⟨0|
                1: np.outer(ket_1, ket_1.conj()),  # |1⟩⟨1|
            },
            Basis.HADAMARD: {
                0: np.outer(ket_plus, ket_plus.conj()),   # |+⟩⟨+|
                1: np.outer(ket_minus, ket_minus.conj()), # |−⟩⟨−|
            }
        }

    def _verify_bb84_states(self):
        """Verify BB84 states are normalized and bases are orthogonal."""
        for key, state in self.states.items():
            norm = np.linalg.norm(state.state_vector)
            if not np.isclose(norm, 1.0):
                raise ValueError(f"State {state.label} not normalized: ||ψ|| = {norm}")

        # Verify orthogonality within each basis
        for basis in [Basis.COMPUTATIONAL, Basis.HADAMARD]:
            state_0 = self.states[(0, basis)].state_vector
            state_1 = self.states[(1, basis)].state_vector
            inner = np.abs(np.vdot(state_0, state_1))
            if not np.isclose(inner, 0.0):
                raise ValueError(f"States in {basis} basis not orthogonal: ⟨ψ₀|ψ₁⟩ = {inner}")

        # Verify bases are mutually unbiased (|⟨ψ_Z|ψ_X⟩|² = 1/2)
        for bit_z in [0, 1]:
            for bit_x in [0, 1]:
                state_z = self.states[(bit_z, Basis.COMPUTATIONAL)].state_vector
                state_x = self.states[(bit_x, Basis.HADAMARD)].state_vector
                overlap_sq = np.abs(np.vdot(state_z, state_x))**2
                if not np.isclose(overlap_sq, 0.5):
                    raise ValueError(f"Bases not mutually unbiased: |⟨{bit_z}|{'+' if bit_x==0 else '-'}⟩|² = {overlap_sq}")

    def alice_prepare(self) -> Tuple[List[BB84State], List[int], List[Basis]]:
        """
        Alice prepares random qubits in random bases.

        Returns:
            Tuple of (states, bits, bases)
        """
        bits = [secrets.randbelow(2) for _ in range(self.num_qubits)]
        bases = [secrets.choice([Basis.COMPUTATIONAL, Basis.HADAMARD])
                 for _ in range(self.num_qubits)]

        states = [self.states[(bit, basis)] for bit, basis in zip(bits, bases)]

        return states, bits, bases

    def bob_measure(self, states: List[BB84State]) -> Tuple[List[int], List[Basis]]:
        """
        Bob measures each qubit in a random basis.

        Args:
            states: List of BB84 states received from Alice

        Returns:
            Tuple of (measurement_results, bases_used)
        """
        bases = [secrets.choice([Basis.COMPUTATIONAL, Basis.HADAMARD])
                 for _ in range(len(states))]

        results = []
        for state, basis in zip(states, bases):
            result = self._measure_state(state.state_vector, basis)
            results.append(result)

        return results, bases

    def _measure_state(self, state_vector: np.ndarray, basis: Basis) -> int:
        """
        Measure a quantum state in the given basis.

        Args:
            state_vector: The quantum state to measure
            basis: The basis to measure in

        Returns:
            Measurement outcome (0 or 1)
        """
        # Calculate probabilities
        proj_0 = self.measurement_ops[basis][0]
        prob_0 = np.real(np.vdot(state_vector, proj_0 @ state_vector))

        # Perform measurement
        if secrets.SystemRandom().random() < prob_0:
            return 0
        else:
            return 1

    def sift_keys(self, alice_bases: List[Basis], bob_bases: List[Basis],
                  alice_bits: List[int], bob_results: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Sifting: Keep only bits where Alice and Bob used the same basis.

        Args:
            alice_bases: Alice's preparation bases
            bob_bases: Bob's measurement bases
            alice_bits: Alice's original bits
            bob_results: Bob's measurement results

        Returns:
            Tuple of (matching_indices, alice_sifted, bob_sifted)
        """
        matching = []
        alice_sifted = []
        bob_sifted = []

        for i, (a_basis, b_basis) in enumerate(zip(alice_bases, bob_bases)):
            if a_basis == b_basis:
                matching.append(i)
                alice_sifted.append(alice_bits[i])
                bob_sifted.append(bob_results[i])

        return matching, alice_sifted, bob_sifted

    def estimate_qber(self, alice_key: List[int], bob_key: List[int],
                     sample_fraction: Optional[float] = None) -> Tuple[float, int, int]:
        """
        Estimate Quantum Bit Error Rate by comparing a sample of bits.

        These bits are sacrificed (revealed publicly) to detect eavesdropping.

        Args:
            alice_key: Alice's sifted key
            bob_key: Bob's sifted key
            sample_fraction: Fraction of bits to sample (default: self.error_estimation_fraction)

        Returns:
            Tuple of (qber, errors, bits_compared)
        """
        if sample_fraction is None:
            sample_fraction = self.error_estimation_fraction

        n = len(alice_key)
        sample_size = max(1, int(n * sample_fraction))

        # Randomly select indices to compare
        indices = list(range(n))
        secrets.SystemRandom().shuffle(indices)
        sample_indices = indices[:sample_size]

        # Count errors
        errors = sum(1 for i in sample_indices if alice_key[i] != bob_key[i])

        qber = errors / sample_size if sample_size > 0 else 0.0

        return qber, errors, sample_size

    def run_protocol(self, eve_intercept: bool = False,
                    eve_strategy: str = 'intercept_resend') -> BB84Result:
        """
        Execute the complete BB84 protocol.

        Args:
            eve_intercept: Whether Eve is eavesdropping
            eve_strategy: Eve's attack strategy ('intercept_resend' or 'none')

        Returns:
            BB84Result with complete protocol analysis
        """
        # Step 1: Alice prepares states
        states, alice_bits, alice_bases = self.alice_prepare()

        # Step 2: (Optional) Eve intercepts
        if eve_intercept and eve_strategy == 'intercept_resend':
            states = self._eve_intercept_resend(states)

        # Step 3: Bob measures
        bob_results, bob_bases = self.bob_measure(states)

        # Step 4: Sifting
        matching_idx, alice_sifted, bob_sifted = self.sift_keys(
            alice_bases, bob_bases, alice_bits, bob_results
        )

        sifting_rate = len(matching_idx) / self.num_qubits if self.num_qubits > 0 else 0

        # Step 5: Error estimation
        qber, errors, bits_compared = self.estimate_qber(alice_sifted, bob_sifted)

        # Step 6: Security assessment
        eve_detected = qber > self.QBER_THRESHOLD

        # Estimate information leaked to Eve (simplified model)
        # In ideal case with intercept-resend, Eve causes 25% QBER
        estimated_info_leaked = min(1.0, qber / 0.25) if qber > 0 else 0.0

        # Secure key rate (simplified Shannon limit)
        # R = 1 - H(QBER) - H(QBER) where H is binary entropy
        if qber < 0.5:
            h_qber = -qber * np.log2(qber + 1e-10) - (1-qber) * np.log2(1-qber + 1e-10)
            secure_key_rate = max(0, 1 - 2 * h_qber)
        else:
            secure_key_rate = 0.0

        # Step 7: Generate final key (if secure)
        protocol_successful = qber <= self.QBER_THRESHOLD and len(alice_sifted) > 0
        final_key = alice_sifted if protocol_successful else None

        return BB84Result(
            num_qubits_sent=self.num_qubits,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_measurements=bob_results,
            matching_indices=matching_idx,
            sifted_key_alice=alice_sifted,
            sifted_key_bob=bob_sifted,
            sifting_rate=sifting_rate,
            qber=qber,
            errors_detected=errors,
            bits_compared=bits_compared,
            eve_detected=eve_detected,
            estimated_information_leaked=estimated_info_leaked,
            secure_key_rate=secure_key_rate,
            final_key=final_key,
            protocol_successful=protocol_successful
        )

    def _eve_intercept_resend(self, states: List[BB84State]) -> List[BB84State]:
        """
        Eve's intercept-resend attack.

        Eve measures each qubit in a random basis and resends based on her result.
        This introduces ~25% QBER because:
        - 50% of time Eve picks wrong basis
        - When wrong basis, 50% chance of error
        - Total: 0.5 × 0.5 = 0.25 QBER

        Args:
            states: Original states from Alice

        Returns:
            States after Eve's interception (potentially modified)
        """
        intercepted_states = []

        for state in states:
            # Eve measures in random basis
            eve_basis = secrets.choice([Basis.COMPUTATIONAL, Basis.HADAMARD])
            eve_result = self._measure_state(state.state_vector, eve_basis)

            # Eve resends based on her measurement
            new_state = self.states[(eve_result, eve_basis)]
            intercepted_states.append(new_state)

        return intercepted_states

    def analyze_security(self, result: BB84Result) -> Dict[str, any]:
        """
        Detailed security analysis of protocol execution.

        Args:
            result: BB84Result from protocol execution

        Returns:
            Dictionary with detailed security metrics
        """
        analysis = {
            'protocol_summary': {
                'qubits_sent': result.num_qubits_sent,
                'sifted_key_length': len(result.sifted_key_alice),
                'sifting_efficiency': result.sifting_rate,
                'expected_sifting_rate': 0.5,  # Theoretical value
            },
            'error_analysis': {
                'qber': result.qber,
                'qber_threshold': self.QBER_THRESHOLD,
                'errors_in_sample': result.errors_detected,
                'sample_size': result.bits_compared,
            },
            'security_assessment': {
                'eve_detected': result.eve_detected,
                'information_leaked_estimate': result.estimated_information_leaked,
                'secure_key_rate': result.secure_key_rate,
                'protocol_successful': result.protocol_successful,
            },
            'theoretical_bounds': {
                'intercept_resend_qber': 0.25,  # Eve's IR attack causes 25% QBER
                'shor_preskill_bound': 0.146,   # Maximum tolerable QBER
                'bb84_security_proof': 'Unconditionally secure against individual attacks when QBER < 11%'
            }
        }

        return analysis


# =============================================================================
# Symbolic Verification Functions (for QuantumVerifier integration)
# =============================================================================

def get_bb84_states_symbolic():
    """
    Get symbolic representations of BB84 states for verification.

    Returns:
        Dictionary of state name to symbolic matrix
    """
    from sympy import Matrix, sqrt, Rational

    ket_0 = Matrix([1, 0])
    ket_1 = Matrix([0, 1])
    ket_plus = (ket_0 + ket_1) / sqrt(2)
    ket_minus = (ket_0 - ket_1) / sqrt(2)

    return {
        '|0⟩': ket_0,
        '|1⟩': ket_1,
        '|+⟩': ket_plus,
        '|−⟩': ket_minus,
    }


def verify_mutually_unbiased_bases():
    """
    Symbolically verify that Z and X bases are mutually unbiased.

    Two bases are mutually unbiased if |⟨ψ_i|φ_j⟩|² = 1/d for all i,j,
    where d is the dimension (d=2 for qubits).

    Returns:
        Dictionary with verification results
    """
    from sympy import Matrix, sqrt, simplify, Abs, Rational

    states = get_bb84_states_symbolic()

    results = {
        'definition': 'Mutually unbiased bases: |⟨ψ_Z|ψ_X⟩|² = 1/2',
        'overlaps': {}
    }

    z_states = ['|0⟩', '|1⟩']
    x_states = ['|+⟩', '|−⟩']

    all_correct = True
    for z_name in z_states:
        for x_name in x_states:
            z = states[z_name]
            x = states[x_name]

            # |⟨z|x⟩|²
            inner = (z.adjoint() * x)[0, 0]
            overlap_sq = simplify(Abs(inner)**2)

            is_correct = overlap_sq == Rational(1, 2)
            results['overlaps'][f'|⟨{z_name}|{x_name}⟩|²'] = {
                'value': str(overlap_sq),
                'equals_half': is_correct
            }
            all_correct = all_correct and is_correct

    results['mutually_unbiased'] = all_correct
    return results


def verify_no_cloning_impossibility():
    """
    Demonstrate why BB84 is secure: non-orthogonal states cannot be cloned.

    The no-cloning theorem states that there is no unitary U such that:
    U|ψ⟩|0⟩ = |ψ⟩|ψ⟩ for arbitrary |ψ⟩

    For BB84, |0⟩ and |+⟩ are non-orthogonal, so Eve cannot clone them.

    Returns:
        Dictionary with verification results
    """
    from sympy import Matrix, sqrt, simplify, Abs, Rational

    states = get_bb84_states_symbolic()

    # |0⟩ and |+⟩ are non-orthogonal
    inner = (states['|0⟩'].adjoint() * states['|+⟩'])[0, 0]
    overlap = simplify(Abs(inner))

    results = {
        'theorem': 'No-cloning theorem',
        'statement': 'Non-orthogonal quantum states cannot be perfectly cloned',
        'bb84_states': {
            '|0⟩_and_|+⟩_inner_product': str(overlap),
            'are_orthogonal': overlap == 0,
            'are_identical': overlap == 1,
        },
        'security_implication': (
            'Since |0⟩ and |+⟩ are neither orthogonal nor identical, '
            'Eve cannot clone them. Any attempt to copy introduces errors '
            'detectable by Alice and Bob through QBER estimation.'
        ),
        'eve_optimal_attack_qber': '25% (intercept-resend)'
    }

    results['no_cloning_applies'] = (overlap != 0 and overlap != 1)

    return results


def verify_measurement_disturbance():
    """
    Verify that measuring in wrong basis causes information loss.

    If Alice sends |+⟩ and Bob measures in Z basis:
    - P(0) = |⟨0|+⟩|² = 1/2
    - P(1) = |⟨1|+⟩|² = 1/2

    Bob gets no information about Alice's bit!

    Returns:
        Dictionary showing measurement probabilities
    """
    from sympy import Matrix, sqrt, simplify, Abs, Rational

    states = get_bb84_states_symbolic()

    results = {
        'scenario': 'Alice sends |+⟩ (bit=0 in X basis), Bob measures in Z basis',
        'probabilities': {},
        'interpretation': ''
    }

    # |+⟩ measured in Z basis
    plus = states['|+⟩']
    ket_0 = states['|0⟩']
    ket_1 = states['|1⟩']

    p_0 = simplify(Abs((ket_0.adjoint() * plus)[0, 0])**2)
    p_1 = simplify(Abs((ket_1.adjoint() * plus)[0, 0])**2)

    results['probabilities'] = {
        'P(measure_0)': str(p_0),
        'P(measure_1)': str(p_1),
        'sum_equals_1': simplify(p_0 + p_1) == 1
    }

    results['information_gained'] = 0 if p_0 == p_1 else 'partial'
    results['interpretation'] = (
        'When Bob measures in wrong basis, each outcome is equally likely. '
        'Bob gains no information about Alice\'s bit. This is why sifting '
        '(keeping only matching bases) is necessary.'
    )

    return results
