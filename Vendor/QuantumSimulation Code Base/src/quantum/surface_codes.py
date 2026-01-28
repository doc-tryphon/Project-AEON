"""
Surface Code Implementation (Step 13).

Surface codes are 2D topological error correction codes that are
the leading candidate for fault-tolerant quantum computing.

Architecture:
- Data qubits arranged on a 2D lattice
- X-stabilizers on vertices (detect Z errors)
- Z-stabilizers on plaquettes/faces (detect X errors)
- Logical qubits encoded in topological degrees of freedom

Key Properties:
- Code distance d: requires d errors to cause logical error
- Threshold ~1%: below this error rate, logical errors suppressed
- Scales with O(d²) physical qubits per logical qubit

References:
- Kitaev, A. Y. (2003). "Fault-tolerant quantum computation by anyons"
- Fowler et al. (2012). "Surface codes: Towards practical large-scale quantum computation"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Set, Dict, Optional
import numpy as np


class StabilizerType(Enum):
    """Type of stabilizer operator."""
    X = "X"  # Vertex stabilizer (detects Z errors)
    Z = "Z"  # Plaquette stabilizer (detects X errors)


@dataclass
class Stabilizer:
    """
    Represents a stabilizer operator.

    A stabilizer is a product of Pauli operators that stabilizes
    the code space: S|ψ⟩ = |ψ⟩ for all codewords |ψ⟩.
    """
    stabilizer_type: StabilizerType
    qubit_indices: List[int]
    position: Tuple[int, int] = (0, 0)  # Position on lattice

    def weight(self) -> int:
        """Return weight (number of qubits) of stabilizer."""
        return len(self.qubit_indices)


@dataclass
class SurfaceCodeResult:
    """Result of surface code operations."""
    success: bool
    syndrome: List[int]
    corrections_applied: List[Tuple[str, int]]  # (operator_type, qubit)
    logical_error: bool = False


class SurfaceCodeLattice:
    """
    Manages the 2D lattice structure for surface code.

    Uses the rotated surface code layout where:
    - Data qubits are on a d×d grid
    - X-stabilizers are on alternating plaquettes
    - Z-stabilizers are on the other plaquettes
    """

    def __init__(self, distance: int):
        """
        Initialize surface code lattice.

        Args:
            distance: Code distance (should be odd)
        """
        if distance < 3:
            raise ValueError("Distance must be at least 3")

        self.distance = distance

        # For rotated surface code: d² data qubits
        self.n_data_qubits = distance * distance

        # Build lattice structure
        self._build_lattice()

    def _build_lattice(self):
        """
        Build a CSS surface code lattice using checkerboard plaquettes with boundaries.

        Uses the rotated surface code architecture:
        - Data qubits on a d×d grid
        - X stabilizers on "white" plaquettes (checkerboard pattern)
        - Z stabilizers on "black" plaquettes (alternate checkerboard)
        - Boundary stabilizers (weight-2) to cover edge qubits

        Adjacent X and Z plaquettes share exactly 2 qubits, ensuring commutation.
        Boundary stabilizers ensure every qubit is covered by both types.

        Logical operators:
        - Logical X: chain along a row (commutes with Z stabilizers)
        - Logical Z: chain along a column (commutes with X stabilizers)
        """
        d = self.distance

        # Data qubit coordinates: (row, col) -> qubit_id
        self.qubit_coords: Dict[Tuple[int, int], int] = {}
        self.coord_to_qubit: Dict[Tuple[int, int], int] = {}

        qubit_id = 0
        for row in range(d):
            for col in range(d):
                self.qubit_coords[qubit_id] = (row, col)
                self.coord_to_qubit[(row, col)] = qubit_id
                qubit_id += 1

        self._x_stabilizers = []
        self._z_stabilizers = []

        # Interior plaquettes: checkerboard pattern
        # Each plaquette is centered at half-integer coordinates
        # Plaquette at (r+0.5, c+0.5) includes qubits (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        for r in range(d - 1):
            for c in range(d - 1):
                qubits = [
                    self.coord_to_qubit[(r, c)],
                    self.coord_to_qubit[(r, c + 1)],
                    self.coord_to_qubit[(r + 1, c)],
                    self.coord_to_qubit[(r + 1, c + 1)]
                ]

                # Checkerboard: (r + c) even -> X stabilizer, odd -> Z stabilizer
                if (r + c) % 2 == 0:
                    self._x_stabilizers.append(Stabilizer(
                        stabilizer_type=StabilizerType.X,
                        qubit_indices=qubits,
                        position=(r, c)
                    ))
                else:
                    self._z_stabilizers.append(Stabilizer(
                        stabilizer_type=StabilizerType.Z,
                        qubit_indices=qubits,
                        position=(r, c)
                    ))

        # Boundary stabilizers with open boundary conditions
        # Left/right boundaries are "rough" - logical X runs here, no Z boundary stabs
        # Top/bottom boundaries are "smooth" - logical Z runs here, no X boundary stabs
        #
        # We add weight-2 boundary stabilizers only where they don't interfere
        # with logical operators:
        # - Z boundary stabilizers on top and bottom edges (not left/right)
        # - X boundary stabilizers on left and right edges (not top/bottom)

        # Top boundary: weight-2 Z stabilizers (vertical pairs at top edge)
        # These are on the "smooth" boundary, complement of X checkerboard
        for c in range(d - 1):
            if ((-1) + c) % 2 == 1:  # Where there's no X plaquette above
                qubits = [
                    self.coord_to_qubit[(0, c)],
                    self.coord_to_qubit[(0, c + 1)]
                ]
                self._z_stabilizers.append(Stabilizer(
                    stabilizer_type=StabilizerType.Z,
                    qubit_indices=qubits,
                    position=(-1, c)
                ))

        # Bottom boundary: weight-2 Z stabilizers
        for c in range(d - 1):
            if ((d - 1) + c) % 2 == 1:  # Where there's no X plaquette below
                qubits = [
                    self.coord_to_qubit[(d - 1, c)],
                    self.coord_to_qubit[(d - 1, c + 1)]
                ]
                self._z_stabilizers.append(Stabilizer(
                    stabilizer_type=StabilizerType.Z,
                    qubit_indices=qubits,
                    position=(d, c)
                ))

        # Left boundary: weight-2 X stabilizers (horizontal pairs at left edge)
        # These are on the "rough" boundary, complement of Z checkerboard
        for r in range(d - 1):
            if (r + (-1)) % 2 == 0:  # Where there's no Z plaquette to left
                qubits = [
                    self.coord_to_qubit[(r, 0)],
                    self.coord_to_qubit[(r + 1, 0)]
                ]
                self._x_stabilizers.append(Stabilizer(
                    stabilizer_type=StabilizerType.X,
                    qubit_indices=qubits,
                    position=(r, -1)
                ))

        # Right boundary: weight-2 X stabilizers
        for r in range(d - 1):
            if (r + (d - 1)) % 2 == 0:  # Where there's no Z plaquette to right
                qubits = [
                    self.coord_to_qubit[(r, d - 1)],
                    self.coord_to_qubit[(r + 1, d - 1)]
                ]
                self._x_stabilizers.append(Stabilizer(
                    stabilizer_type=StabilizerType.X,
                    qubit_indices=qubits,
                    position=(r, d)
                ))

        self.x_stabilizer_positions = [(s.position[0], s.position[1]) for s in self._x_stabilizers]
        self.z_stabilizer_positions = [(s.position[0], s.position[1]) for s in self._z_stabilizers]

    def _is_valid_stabilizer_position(self, row: int, col: int) -> bool:
        """Check if position can have a stabilizer."""
        d = self.distance

        # Interior positions always valid
        if 0 < row < d and 0 < col < d:
            return True

        # Boundary positions: some are valid for open boundary conditions
        # For simplicity, include boundary stabilizers with reduced weight
        if 0 <= row <= d and 0 <= col <= d:
            # At least one adjacent data qubit exists
            adjacent = self._get_adjacent_data_qubits(row, col)
            return len(adjacent) >= 2

        return False

    def _get_adjacent_data_qubits(self, stab_row: int, stab_col: int) -> List[int]:
        """Get data qubits adjacent to a stabilizer position."""
        adjacent = []
        d = self.distance

        # Stabilizer at (r, c) is adjacent to data qubits at:
        # (r-1, c-1), (r-1, c), (r, c-1), (r, c)
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            qr, qc = stab_row + dr, stab_col + dc
            if 0 <= qr < d and 0 <= qc < d:
                qubit_id = self.coord_to_qubit[(qr, qc)]
                adjacent.append(qubit_id)

        return adjacent

    def _build_x_stabilizers(self) -> List[Stabilizer]:
        """Build X-stabilizer generators."""
        stabilizers = []

        for pos in self.x_stabilizer_positions:
            qubits = self._get_adjacent_data_qubits(pos[0], pos[1])
            if qubits:
                stab = Stabilizer(
                    stabilizer_type=StabilizerType.X,
                    qubit_indices=qubits,
                    position=pos
                )
                stabilizers.append(stab)

        return stabilizers

    def _build_z_stabilizers(self) -> List[Stabilizer]:
        """Build Z-stabilizer generators."""
        stabilizers = []

        for pos in self.z_stabilizer_positions:
            qubits = self._get_adjacent_data_qubits(pos[0], pos[1])
            if qubits:
                stab = Stabilizer(
                    stabilizer_type=StabilizerType.Z,
                    qubit_indices=qubits,
                    position=pos
                )
                stabilizers.append(stab)

        return stabilizers

    def get_x_stabilizers(self) -> List[Stabilizer]:
        """Get X-stabilizer generators."""
        return self._x_stabilizers

    def get_z_stabilizers(self) -> List[Stabilizer]:
        """Get Z-stabilizer generators."""
        return self._z_stabilizers

    def get_qubit_coordinate(self, qubit_id: int) -> Tuple[int, int]:
        """Get 2D coordinate of a data qubit."""
        return self.qubit_coords[qubit_id]


class SurfaceCodeDecoder:
    """
    Decoder for surface codes.

    Uses a simplified minimum-weight matching decoder.
    For production, would use Blossom algorithm or neural network decoder.
    """

    def __init__(self, lattice: SurfaceCodeLattice):
        """Initialize decoder with lattice structure."""
        self.lattice = lattice

    def decode(self, syndrome: List[int],
               x_stabilizers: List[Stabilizer],
               z_stabilizers: List[Stabilizer]) -> List[Tuple[str, int]]:
        """
        Decode syndrome to find correction.

        Args:
            syndrome: Measured syndrome bits
            x_stabilizers: X stabilizer list
            z_stabilizers: Z stabilizer list

        Returns:
            List of corrections (operator_type, qubit_id)
        """
        corrections = []

        # Split syndrome into X and Z parts
        n_x = len(x_stabilizers)
        x_syndrome = syndrome[:n_x]
        z_syndrome = syndrome[n_x:]

        # Find violated X stabilizers (indicate Z errors)
        violated_x = [i for i, s in enumerate(x_syndrome) if s == 1]

        # Find violated Z stabilizers (indicate X errors)
        violated_z = [i for i, s in enumerate(z_syndrome) if s == 1]

        # Simple decoder: find qubit in common between violated stabilizers
        # For single errors, this works well

        # Correct Z errors (using X stabilizers)
        if violated_x:
            correction_qubit = self._find_correction_qubit(
                violated_x, x_stabilizers)
            if correction_qubit is not None:
                corrections.append(('Z', correction_qubit))

        # Correct X errors (using Z stabilizers)
        if violated_z:
            correction_qubit = self._find_correction_qubit(
                violated_z, z_stabilizers)
            if correction_qubit is not None:
                corrections.append(('X', correction_qubit))

        return corrections

    def _find_correction_qubit(self, violated_indices: List[int],
                                stabilizers: List[Stabilizer]) -> Optional[int]:
        """Find qubit to apply correction based on violated stabilizers."""
        if not violated_indices:
            return None

        if len(violated_indices) == 1:
            # Single violated stabilizer: prefer boundary qubit (fewer connections)
            # Count how many stabilizers touch each qubit in the support
            stab = stabilizers[violated_indices[0]]
            qubit_counts = {}
            for q in stab.qubit_indices:
                count = sum(1 for s in stabilizers if q in s.qubit_indices)
                qubit_counts[q] = count

            # Pick qubit with minimum count (boundary qubit)
            best_qubit = min(stab.qubit_indices, key=lambda q: qubit_counts[q])
            return best_qubit

        # Multiple violated: find common qubit or use heuristic
        # For adjacent violated stabilizers, find common qubit
        stab1 = stabilizers[violated_indices[0]]
        stab2 = stabilizers[violated_indices[1]]

        common = set(stab1.qubit_indices) & set(stab2.qubit_indices)
        if common:
            return list(common)[0]

        # No common qubit: use first qubit of first stabilizer
        return stab1.qubit_indices[0]


class SurfaceCode:
    """
    Complete surface code implementation.

    Provides encoding, error application, syndrome measurement,
    and error correction for topological quantum error correction.
    """

    def __init__(self, distance: int):
        """
        Initialize surface code.

        Args:
            distance: Code distance (number of errors that can be detected)
        """
        self.distance = distance
        self.lattice = SurfaceCodeLattice(distance)
        self.decoder = SurfaceCodeDecoder(self.lattice)

        # Dimension of Hilbert space (2^n for n qubits)
        self.hilbert_dim = 2 ** self.lattice.n_data_qubits

        # Precompute Pauli operators
        self._build_pauli_operators()

        # Build logical operators
        self._build_logical_operators()

    def _build_pauli_operators(self):
        """Build single-qubit Pauli operators."""
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def _build_logical_operators(self):
        """Build logical X and Z operators."""
        d = self.distance

        # Logical X: chain of X operators along one edge (e.g., first row)
        self._logical_x_qubits = [self.lattice.coord_to_qubit[(0, c)] for c in range(d)]

        # Logical Z: chain of Z operators along perpendicular edge (e.g., first column)
        self._logical_z_qubits = [self.lattice.coord_to_qubit[(r, 0)] for r in range(d)]

    def _apply_pauli_string(self, state: np.ndarray,
                            pauli_ops: List[Tuple[str, int]]) -> np.ndarray:
        """Apply a product of Pauli operators to state."""
        result = state.copy()

        for op_type, qubit in pauli_ops:
            if op_type == 'X':
                result = self._apply_single_qubit_gate(result, self.X, qubit)
            elif op_type == 'Y':
                result = self._apply_single_qubit_gate(result, self.Y, qubit)
            elif op_type == 'Z':
                result = self._apply_single_qubit_gate(result, self.Z, qubit)

        return result

    def _apply_single_qubit_gate(self, state: np.ndarray,
                                  gate: np.ndarray, qubit: int) -> np.ndarray:
        """Apply single-qubit gate to specific qubit."""
        n = self.lattice.n_data_qubits

        # Build full operator: I ⊗ ... ⊗ gate ⊗ ... ⊗ I
        # Gate is on qubit `qubit` (0-indexed from left in tensor product)
        full_op = np.eye(1, dtype=complex)

        for q in range(n):
            if q == qubit:
                full_op = np.kron(full_op, gate)
            else:
                full_op = np.kron(full_op, self.I)

        return full_op @ state

    def initialize_logical_zero(self) -> np.ndarray:
        """
        Initialize logical |0⟩ state.

        The logical zero is the +1 eigenstate of all stabilizers
        and the +1 eigenstate of logical Z.
        """
        n = self.lattice.n_data_qubits

        # Start with |0...0⟩
        state = np.zeros(self.hilbert_dim, dtype=complex)
        state[0] = 1.0

        # Project onto code space (simplified: for small codes)
        # Apply stabilizer projection
        state = self._project_to_code_space(state)

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state /= norm

        return state

    def initialize_logical_one(self) -> np.ndarray:
        """
        Initialize logical |1⟩ state.

        Logical |1⟩ = X_L |0⟩_L
        """
        state = self.initialize_logical_zero()
        return self.apply_logical_x(state)

    def _project_to_code_space(self, state: np.ndarray) -> np.ndarray:
        """Project state onto +1 eigenspace of all stabilizers."""
        result = state.copy()

        # For each stabilizer, project onto +1 eigenspace
        all_stabs = self.lattice.get_x_stabilizers() + self.lattice.get_z_stabilizers()

        for stab in all_stabs:
            # Projector P = (I + S) / 2
            stab_op = self._build_stabilizer_operator(stab)
            projector = (np.eye(self.hilbert_dim) + stab_op) / 2
            result = projector @ result

            # Renormalize
            norm = np.linalg.norm(result)
            if norm > 1e-10:
                result /= norm

        return result

    def _build_stabilizer_operator(self, stab: Stabilizer) -> np.ndarray:
        """Build matrix representation of stabilizer operator."""
        n = self.lattice.n_data_qubits

        # Product of X or Z on specified qubits
        if stab.stabilizer_type == StabilizerType.X:
            pauli = self.X
        else:
            pauli = self.Z

        # Build tensor product
        ops = [(stab.stabilizer_type.value, q) for q in stab.qubit_indices]

        full_op = np.eye(1, dtype=complex)
        for q in range(n):
            if q in stab.qubit_indices:
                full_op = np.kron(full_op, pauli)
            else:
                full_op = np.kron(full_op, self.I)

        return full_op

    def apply_x_error(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply X (bit flip) error on specified qubit."""
        return self._apply_single_qubit_gate(state, self.X, qubit)

    def apply_z_error(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Z (phase flip) error on specified qubit."""
        return self._apply_single_qubit_gate(state, self.Z, qubit)

    def apply_y_error(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Y error on specified qubit."""
        return self._apply_single_qubit_gate(state, self.Y, qubit)

    def apply_random_errors(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply random Pauli errors with given rate."""
        result = state.copy()

        for qubit in range(self.lattice.n_data_qubits):
            if np.random.random() < error_rate:
                # Random Pauli error
                error_type = np.random.choice(['X', 'Y', 'Z'])
                if error_type == 'X':
                    result = self.apply_x_error(result, qubit)
                elif error_type == 'Y':
                    result = self.apply_y_error(result, qubit)
                else:
                    result = self.apply_z_error(result, qubit)

        return result

    def measure_syndrome(self, state: np.ndarray) -> List[int]:
        """
        Measure syndrome (stabilizer outcomes).

        Returns:
            List of syndrome bits (0 = +1 eigenvalue, 1 = -1 eigenvalue)
        """
        syndrome = []

        x_stabs = self.lattice.get_x_stabilizers()
        z_stabs = self.lattice.get_z_stabilizers()

        for stab in x_stabs + z_stabs:
            stab_op = self._build_stabilizer_operator(stab)

            # Expectation value ⟨ψ|S|ψ⟩
            expectation = np.real(np.vdot(state, stab_op @ state))

            # +1 → syndrome 0, -1 → syndrome 1
            syndrome_bit = 0 if expectation > 0 else 1
            syndrome.append(syndrome_bit)

        return syndrome

    def correct_errors(self, state: np.ndarray) -> np.ndarray:
        """
        Measure syndrome and apply correction.

        Returns:
            Corrected state
        """
        syndrome = self.measure_syndrome(state)

        x_stabs = self.lattice.get_x_stabilizers()
        z_stabs = self.lattice.get_z_stabilizers()

        corrections = self.decoder.decode(syndrome, x_stabs, z_stabs)

        result = state.copy()
        for op_type, qubit in corrections:
            if op_type == 'X':
                result = self.apply_x_error(result, qubit)
            elif op_type == 'Z':
                result = self.apply_z_error(result, qubit)

        return result

    def apply_logical_x(self, state: np.ndarray) -> np.ndarray:
        """Apply logical X operator."""
        result = state.copy()
        for qubit in self._logical_x_qubits:
            result = self.apply_x_error(result, qubit)
        return result

    def apply_logical_z(self, state: np.ndarray) -> np.ndarray:
        """Apply logical Z operator."""
        result = state.copy()
        for qubit in self._logical_z_qubits:
            result = self.apply_z_error(result, qubit)
        return result

    def get_logical_x_operator(self) -> Stabilizer:
        """Get logical X operator as Stabilizer object."""
        return Stabilizer(
            stabilizer_type=StabilizerType.X,
            qubit_indices=self._logical_x_qubits.copy()
        )

    def get_logical_z_operator(self) -> Stabilizer:
        """Get logical Z operator as Stabilizer object."""
        return Stabilizer(
            stabilizer_type=StabilizerType.Z,
            qubit_indices=self._logical_z_qubits.copy()
        )

    def logical_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute fidelity between two logical states.

        Returns:
            |⟨ψ1|ψ2⟩|² (fidelity)
        """
        overlap = np.vdot(state1, state2)
        return float(np.abs(overlap) ** 2)


# =============================================================================
# Helper Functions
# =============================================================================

def create_x_stabilizer(qubit_indices: List[int],
                        position: Tuple[int, int] = (0, 0)) -> Stabilizer:
    """Create an X-type stabilizer."""
    return Stabilizer(
        stabilizer_type=StabilizerType.X,
        qubit_indices=qubit_indices,
        position=position
    )


def create_z_stabilizer(qubit_indices: List[int],
                        position: Tuple[int, int] = (0, 0)) -> Stabilizer:
    """Create a Z-type stabilizer."""
    return Stabilizer(
        stabilizer_type=StabilizerType.Z,
        qubit_indices=qubit_indices,
        position=position
    )


def get_stabilizer_generators(lattice: SurfaceCodeLattice) -> List[Stabilizer]:
    """Get all stabilizer generators for a lattice."""
    return lattice.get_x_stabilizers() + lattice.get_z_stabilizers()


def apply_error(state: np.ndarray, error_type: str, qubit: int,
                code: SurfaceCode) -> np.ndarray:
    """Apply error to state."""
    if error_type == 'X':
        return code.apply_x_error(state, qubit)
    elif error_type == 'Y':
        return code.apply_y_error(state, qubit)
    elif error_type == 'Z':
        return code.apply_z_error(state, qubit)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def measure_syndrome(state: np.ndarray, code: SurfaceCode) -> List[int]:
    """Measure syndrome of state."""
    return code.measure_syndrome(state)


def correct_errors(state: np.ndarray, code: SurfaceCode) -> np.ndarray:
    """Apply error correction to state."""
    return code.correct_errors(state)
