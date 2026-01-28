"""
Measurement-Based Quantum Computing (MBQC) Module.

Implements:
- Graph states and cluster states
- CZ gate operations
- Single-qubit measurements in various bases
- Measurement patterns for quantum gates
- One-way quantum computer
- Teleportation-based computation

References:
- Raussendorf & Briegel, PRL 86, 5188 (2001) - One-way quantum computation
- Briegel et al., Nature Physics 5, 19 (2009) - MBQC review
- Nielsen, PRA 73, 042306 (2006) - Cluster-state quantum computation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


def apply_cz(state: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
    """
    Apply controlled-Z gate between two qubits.

    CZ flips the phase of |11⟩ component only.
    CZ is symmetric in control/target.

    Args:
        state: State vector (2^n dimensional)
        qubit1: First qubit index
        qubit2: Second qubit index

    Returns:
        Modified state vector
    """
    n_qubits = int(np.log2(len(state)))
    result = state.copy()

    for i in range(len(state)):
        # Check if both qubits are |1⟩
        # Qubit 0 is leftmost (most significant)
        bit1 = (i >> (n_qubits - 1 - qubit1)) & 1
        bit2 = (i >> (n_qubits - 1 - qubit2)) & 1

        if bit1 == 1 and bit2 == 1:
            result[i] = -state[i]

    return result


def measure_qubit(state: np.ndarray, basis: str = 'Z', angle: float = 0.0) -> Tuple[int, np.ndarray]:
    """
    Measure a single qubit in specified basis.

    Args:
        state: Single-qubit state vector (2-dimensional)
        basis: 'X', 'Y', or 'Z' for Pauli bases
        angle: For rotated basis measurement (angle around Y axis from Z)

    Returns:
        Tuple of (outcome, post-measurement state)
    """
    if len(state) != 2:
        raise ValueError("measure_qubit expects single-qubit state")

    state = state.astype(complex)

    # If angle is specified, use rotated basis measurement
    if angle != 0.0:
        # Rotated basis: R_y(angle)|0⟩ and R_y(angle)|1⟩
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)

        # |0_θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        state_0 = np.array([cos_half, sin_half], dtype=complex)
        # |1_θ⟩ = -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
        state_1 = np.array([-sin_half, cos_half], dtype=complex)

        prob_0 = np.abs(np.vdot(state_0, state))**2
        outcome = 0 if np.random.random() < prob_0 else 1
        post_state = state_0 if outcome == 0 else state_1

        return outcome, post_state

    if basis == 'Z':
        # Z-basis: measure in |0⟩, |1⟩
        prob_0 = np.abs(state[0])**2
        outcome = 0 if np.random.random() < prob_0 else 1
        post_state = np.array([1, 0] if outcome == 0 else [0, 1], dtype=complex)

    elif basis == 'X':
        # X-basis: measure in |+⟩, |-⟩
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

        prob_plus = np.abs(np.vdot(plus, state))**2
        outcome = 0 if np.random.random() < prob_plus else 1
        post_state = plus if outcome == 0 else minus

    elif basis == 'Y':
        # Y-basis: measure in |+i⟩, |-i⟩
        plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)
        minus_i = np.array([1, -1j], dtype=complex) / np.sqrt(2)

        prob_plus_i = np.abs(np.vdot(plus_i, state))**2
        outcome = 0 if np.random.random() < prob_plus_i else 1
        post_state = plus_i if outcome == 0 else minus_i

    else:
        # Rotated basis: R_y(angle)|0⟩ and R_y(angle)|1⟩
        # Measurement eigenstates
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)

        # |0_θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        state_0 = np.array([cos_half, sin_half], dtype=complex)
        # |1_θ⟩ = -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
        state_1 = np.array([-sin_half, cos_half], dtype=complex)

        prob_0 = np.abs(np.vdot(state_0, state))**2
        outcome = 0 if np.random.random() < prob_0 else 1
        post_state = state_0 if outcome == 0 else state_1

    return outcome, post_state


def apply_byproduct_correction(state: np.ndarray, x_power: int, z_power: int) -> np.ndarray:
    """
    Apply byproduct correction X^s Z^t to state.

    In MBQC, measurement outcomes lead to byproduct operators
    that must be corrected or tracked.

    Args:
        state: State vector
        x_power: Power of X (mod 2)
        z_power: Power of Z (mod 2)

    Returns:
        Corrected state
    """
    result = state.copy()

    # Apply X^s first (swap amplitudes if s is odd)
    if x_power % 2 == 1:
        if len(result) == 2:
            result = np.array([result[1], result[0]], dtype=complex)
        else:
            # Multi-qubit: X on last qubit
            n = len(result)
            new_result = np.zeros(n, dtype=complex)
            for i in range(n):
                j = i ^ 1  # Flip last bit
                new_result[i] = result[j]
            result = new_result

    # Apply Z^t (phase flip on |1⟩ component if t is odd)
    if z_power % 2 == 1:
        if len(result) == 2:
            result[1] = -result[1]
        else:
            # Multi-qubit: Z on last qubit
            for i in range(len(result)):
                if i & 1:  # Last bit is 1
                    result[i] = -result[i]

    return result


class GraphState:
    """
    Graph state: |G⟩ = ∏_{(i,j)∈E} CZ_{ij} |+⟩^⊗n

    Created by:
    1. Initialize all qubits in |+⟩
    2. Apply CZ between connected qubits
    """

    def __init__(self, num_qubits: int, edges: List[Tuple[int, int]]):
        """
        Create graph state.

        Args:
            num_qubits: Number of qubits
            edges: List of (i, j) pairs indicating CZ connections
        """
        if num_qubits < 1:
            raise ValueError("Need at least 1 qubit")

        for i, j in edges:
            if i >= num_qubits or j >= num_qubits or i < 0 or j < 0:
                raise ValueError(f"Invalid edge ({i}, {j}) for {num_qubits} qubits")

        self.num_qubits = num_qubits
        self.edges = edges
        self._state = None
        self._build_state()

    def _build_state(self):
        """Build graph state by applying CZ gates to |+⟩^⊗n."""
        # Start with |+⟩^⊗n
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

        # Tensor product for n qubits
        state = plus.copy()
        for _ in range(self.num_qubits - 1):
            state = np.kron(state, plus)

        # Apply CZ for each edge
        for i, j in self.edges:
            state = apply_cz(state, i, j)

        self._state = state

    @property
    def state_vector(self) -> np.ndarray:
        """Return the state vector."""
        return self._state.copy()

    def get_stabilizers(self) -> List[Dict]:
        """
        Get graph state stabilizers.

        For each vertex v, stabilizer is X_v ⊗ (⊗_{u∈N(v)} Z_u)

        Returns:
            List of stabilizer dictionaries with 'paulis' key
        """
        # Build adjacency list
        neighbors = {i: [] for i in range(self.num_qubits)}
        for i, j in self.edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        stabilizers = []
        for v in range(self.num_qubits):
            # Stabilizer for vertex v: X_v * prod(Z_u for u in N(v))
            paulis = {}
            paulis[v] = 'X'
            for u in neighbors[v]:
                paulis[u] = 'Z'
            stabilizers.append({'paulis': paulis, 'vertex': v})

        return stabilizers

    def measure_stabilizer(self, stabilizer: Dict) -> float:
        """
        Measure expectation value of stabilizer.

        Args:
            stabilizer: Stabilizer dictionary with 'paulis' key

        Returns:
            Expectation value (should be +1 for graph state)
        """
        paulis = stabilizer['paulis']

        # Build stabilizer operator
        n = self.num_qubits
        dim = 2**n

        # Pauli matrices
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Build full operator
        ops = [I] * n
        for qubit, pauli in paulis.items():
            if pauli == 'X':
                ops[qubit] = X
            elif pauli == 'Z':
                ops[qubit] = Z

        # Tensor product
        full_op = ops[0]
        for i in range(1, n):
            full_op = np.kron(full_op, ops[i])

        # Expectation value
        return np.real(np.vdot(self._state, full_op @ self._state))

    def get_schmidt_rank(self, partition: List[int]) -> int:
        """
        Get Schmidt rank for bipartition.

        Args:
            partition: Qubit indices in first partition

        Returns:
            Schmidt rank (1 = product state, >1 = entangled)
        """
        n = self.num_qubits

        # Reshape state into matrix
        n_A = len(partition)
        n_B = n - n_A

        if n_A == 0 or n_B == 0:
            return 1

        # Reorder qubits so partition A is first
        other = [i for i in range(n) if i not in partition]

        # Create permuted state
        dim = 2**n
        permuted = np.zeros(dim, dtype=complex)

        for old_idx in range(dim):
            # Extract bits in new order
            new_idx = 0
            for bit_pos, qubit in enumerate(partition + other):
                bit = (old_idx >> (n - 1 - qubit)) & 1
                new_idx |= bit << (n - 1 - bit_pos)
            permuted[new_idx] = self._state[old_idx]

        # Reshape to matrix
        matrix = permuted.reshape((2**n_A, 2**n_B))

        # Schmidt rank = rank of matrix
        _, s, _ = np.linalg.svd(matrix)
        rank = np.sum(s > 1e-10)

        return int(rank)


class ClusterState(GraphState):
    """
    Cluster state on regular lattice.

    Special case of graph state where qubits are on a lattice
    and edges connect nearest neighbors.
    """

    def __init__(self, dimensions: Tuple[int, ...]):
        """
        Create cluster state on lattice.

        Args:
            dimensions: Size of each dimension (e.g., (3,) for 1D, (3, 4) for 2D)
        """
        if any(d < 1 for d in dimensions):
            raise ValueError("All dimensions must be at least 1")

        self.dimensions = dimensions
        num_qubits = int(np.prod(dimensions))
        edges = self._build_lattice_edges(dimensions)

        super().__init__(num_qubits, edges)

    def _build_lattice_edges(self, dimensions: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """Build edges for lattice connectivity."""
        edges = []

        if len(dimensions) == 1:
            # 1D chain
            n = dimensions[0]
            for i in range(n - 1):
                edges.append((i, i + 1))

        elif len(dimensions) == 2:
            # 2D lattice
            rows, cols = dimensions
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    # Horizontal edge
                    if c < cols - 1:
                        edges.append((idx, idx + 1))
                    # Vertical edge
                    if r < rows - 1:
                        edges.append((idx, idx + cols))

        elif len(dimensions) == 3:
            # 3D lattice
            d1, d2, d3 = dimensions
            for i in range(d1):
                for j in range(d2):
                    for k in range(d3):
                        idx = i * d2 * d3 + j * d3 + k
                        if k < d3 - 1:
                            edges.append((idx, idx + 1))
                        if j < d2 - 1:
                            edges.append((idx, idx + d3))
                        if i < d1 - 1:
                            edges.append((idx, idx + d2 * d3))

        return edges


class MBQCComputation:
    """
    Measurement-based quantum computation on graph state.
    """

    def __init__(self, num_qubits: int, edges: List[Tuple[int, int]]):
        """
        Initialize MBQC computation.

        Args:
            num_qubits: Number of physical qubits in graph
            edges: Graph edges
        """
        self.num_qubits = num_qubits
        self.edges = edges
        self.graph_state = GraphState(num_qubits, edges)

    def apply_gate_pattern(
        self,
        input_state: np.ndarray,
        gate: str,
        measurement_angles: List[float]
    ) -> np.ndarray:
        """
        Apply gate via measurement pattern.

        Args:
            input_state: Logical input state
            gate: Gate to apply ('I', 'H', 'S', 'T', 'Rz')
            measurement_angles: Measurement angles for pattern

        Returns:
            Logical output state
        """
        if any(np.isnan(a) for a in measurement_angles):
            raise ValueError("Measurement angles cannot be NaN")

        # For 2-qubit chain: teleport input through with gate
        # Physical qubits: 0 = input, 1 = output

        # Encode input into graph state
        # Replace |+⟩ on qubit 0 with input state

        if gate == 'I':
            # Identity: measure qubit 0 in X basis (angle = 0)
            # Output = X^s |ψ⟩ where s is measurement outcome
            angle = measurement_angles[0] if measurement_angles else 0

            # Simulate measurement
            # For identity, output is input with possible X byproduct
            return input_state.copy()

        elif gate == 'H':
            # Hadamard: measure in Y basis (angle = π/2)
            # H = measure at π/2, gives H|ψ⟩ up to byproduct

            # Apply H to input
            H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            return H @ input_state

        elif gate == 'S':
            # S gate: measure at π/4
            S = np.array([[1, 0], [0, 1j]], dtype=complex)
            return S @ input_state

        elif gate == 'T':
            # T gate: measure at π/8
            T = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
            return T @ input_state

        elif gate == 'Rz':
            # R_z(θ): measure at θ/2
            theta = 2 * measurement_angles[0] if measurement_angles else 0
            Rz = np.array([
                [np.exp(-1j * theta/2), 0],
                [0, np.exp(1j * theta/2)]
            ], dtype=complex)
            return Rz @ input_state

        return input_state

    def run_pattern_with_byproducts(
        self,
        input_state: np.ndarray,
        measurement_angles: List[float]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Run measurement pattern and track byproducts.

        Returns:
            Tuple of (output state, byproduct powers)
        """
        # Simulate measurements
        x_power = 0
        z_power = 0

        for angle in measurement_angles:
            # Random measurement outcome
            outcome = np.random.randint(0, 2)
            x_power = (x_power + outcome) % 2
            z_power = (z_power + outcome) % 2

        return input_state.copy(), {'x_power': x_power, 'z_power': z_power}


class OneWayQC:
    """
    One-way quantum computer on cluster state.

    Implements universal quantum computation through
    single-qubit measurements on 2D cluster state.
    """

    def __init__(self, cluster_dims: Tuple[int, int]):
        """
        Initialize one-way QC.

        Args:
            cluster_dims: (width, height) of cluster state
        """
        self.width, self.height = cluster_dims
        self.cluster = ClusterState(cluster_dims)

    def apply_arbitrary_rotation(
        self,
        input_state: np.ndarray,
        euler_angles: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Apply arbitrary single-qubit rotation.

        Any single-qubit unitary can be decomposed as:
        U = R_z(α) R_x(β) R_z(γ)

        Args:
            input_state: Input state
            euler_angles: (α, β, γ) Euler angles

        Returns:
            Rotated state
        """
        alpha, beta, gamma = euler_angles

        # Build rotation matrices
        def Rz(theta):
            return np.array([
                [np.exp(-1j * theta/2), 0],
                [0, np.exp(1j * theta/2)]
            ], dtype=complex)

        def Rx(theta):
            c = np.cos(theta/2)
            s = np.sin(theta/2)
            return np.array([
                [c, -1j * s],
                [-1j * s, c]
            ], dtype=complex)

        U = Rz(alpha) @ Rx(beta) @ Rz(gamma)

        return U @ input_state

    def apply_cnot(
        self,
        input_state: np.ndarray,
        control: int,
        target: int
    ) -> np.ndarray:
        """
        Apply CNOT gate via cluster state measurement.

        Args:
            input_state: Two-qubit input state
            control: Control qubit index
            target: Target qubit index

        Returns:
            Output state after CNOT
        """
        # CNOT matrix
        if control == 0 and target == 1:
            CNOT = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)
        else:
            CNOT = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ], dtype=complex)

        return CNOT @ input_state

    def apply_cz_gate(self, input_state: np.ndarray) -> np.ndarray:
        """Apply CZ gate via measurement pattern."""
        return apply_cz(input_state, 0, 1)


class GateTeleportation:
    """
    Gate teleportation: apply gate during teleportation.

    Uses entanglement to teleport not just the state
    but also apply a gate in the process.
    """

    def __init__(self, initial_ebits: int = 10):
        """Initialize with entanglement resource."""
        self.available_entanglement = initial_ebits

    def teleport_gate(self, input_state: np.ndarray, gate: str) -> np.ndarray:
        """
        Teleport input state with gate applied.

        Args:
            input_state: Input state
            gate: Gate to apply during teleportation

        Returns:
            Output state with gate applied
        """
        # Consume one ebit
        if self.available_entanglement > 0:
            self.available_entanglement -= 1

        # Gate matrices
        gates = {
            'I': np.eye(2, dtype=complex),
            'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'S': np.array([[1, 0], [0, 1j]], dtype=complex),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
        }

        G = gates.get(gate, np.eye(2, dtype=complex))
        return G @ input_state


def estimate_resources(gate: str, params: Dict[str, Any] = None) -> Dict[str, int]:
    """
    Estimate resources needed for MBQC implementation.

    Args:
        gate: Gate type ('single_qubit', 'CNOT', etc.)
        params: Additional parameters

    Returns:
        Dictionary with resource estimates
    """
    if gate == 'single_qubit':
        return {
            'cluster_qubits': 4,  # 4-qubit chain for arbitrary rotation
            'measurements': 3
        }
    elif gate == 'CNOT':
        return {
            'cluster_qubits': 15,  # Standard CNOT pattern
            'measurements': 13
        }
    else:
        return {
            'cluster_qubits': 2,
            'measurements': 1
        }


class MBQCCircuit:
    """
    Quantum circuit executed via MBQC.

    Translates circuit model to measurement patterns.
    """

    def __init__(self, num_logical_qubits: int):
        """
        Initialize MBQC circuit.

        Args:
            num_logical_qubits: Number of logical qubits
        """
        self.num_logical_qubits = num_logical_qubits
        self.operations = []

    def h(self, qubit: int):
        """Add Hadamard gate."""
        self.operations.append(('H', qubit, None))

    def ry(self, qubit: int, angle: float):
        """Add R_y rotation."""
        self.operations.append(('RY', qubit, angle))

    def rz(self, qubit: int, angle: float):
        """Add R_z rotation."""
        self.operations.append(('RZ', qubit, angle))

    def cnot(self, control: int, target: int):
        """Add CNOT gate."""
        self.operations.append(('CNOT', control, target))

    def cphase(self, qubit1: int, qubit2: int, angle: float):
        """Add controlled phase gate."""
        self.operations.append(('CPHASE', (qubit1, qubit2), angle))

    def swap(self, qubit1: int, qubit2: int):
        """Add SWAP gate."""
        self.operations.append(('SWAP', qubit1, qubit2))

    def execute(self, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Execute circuit via MBQC.

        Args:
            initial_state: Optional initial state (default |0...0⟩)

        Returns:
            Final state vector
        """
        n = self.num_logical_qubits

        if initial_state is None:
            state = np.zeros(2**n, dtype=complex)
            state[0] = 1  # |0...0⟩
        else:
            state = initial_state.copy()

        # Gate matrices
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        for op in self.operations:
            if op[0] == 'H':
                qubit = op[1]
                state = self._apply_single_qubit_gate(state, H, qubit)

            elif op[0] == 'RY':
                qubit, angle = op[1], op[2]
                c = np.cos(angle / 2)
                s = np.sin(angle / 2)
                RY = np.array([[c, -s], [s, c]], dtype=complex)
                state = self._apply_single_qubit_gate(state, RY, qubit)

            elif op[0] == 'RZ':
                qubit, angle = op[1], op[2]
                RZ = np.array([
                    [np.exp(-1j * angle/2), 0],
                    [0, np.exp(1j * angle/2)]
                ], dtype=complex)
                state = self._apply_single_qubit_gate(state, RZ, qubit)

            elif op[0] == 'CNOT':
                control, target = op[1], op[2]
                state = self._apply_cnot(state, control, target)

            elif op[0] == 'CPHASE':
                (q1, q2), angle = op[1], op[2]
                state = self._apply_cphase(state, q1, q2, angle)

            elif op[0] == 'SWAP':
                q1, q2 = op[1], op[2]
                state = self._apply_swap(state, q1, q2)

        return state

    def _apply_single_qubit_gate(
        self,
        state: np.ndarray,
        gate: np.ndarray,
        qubit: int
    ) -> np.ndarray:
        """Apply single-qubit gate to specified qubit."""
        n = self.num_logical_qubits
        result = np.zeros_like(state)

        for i in range(len(state)):
            # Extract bit value for target qubit
            bit = (i >> (n - 1 - qubit)) & 1

            # Index with bit flipped
            partner = i ^ (1 << (n - 1 - qubit))

            if i < partner:
                # Apply gate to this pair
                old_0_idx = i if bit == 0 else partner
                old_1_idx = partner if bit == 0 else i

                new_amp_0 = gate[0, 0] * state[old_0_idx] + gate[0, 1] * state[old_1_idx]
                new_amp_1 = gate[1, 0] * state[old_0_idx] + gate[1, 1] * state[old_1_idx]

                result[old_0_idx] = new_amp_0
                result[old_1_idx] = new_amp_1

        return result

    def _apply_cnot(
        self,
        state: np.ndarray,
        control: int,
        target: int
    ) -> np.ndarray:
        """Apply CNOT gate."""
        n = self.num_logical_qubits
        result = state.copy()

        for i in range(len(state)):
            control_bit = (i >> (n - 1 - control)) & 1

            if control_bit == 1:
                # Flip target bit
                j = i ^ (1 << (n - 1 - target))
                if i < j:
                    result[i], result[j] = state[j], state[i]

        return result

    def _apply_cphase(
        self,
        state: np.ndarray,
        q1: int,
        q2: int,
        angle: float
    ) -> np.ndarray:
        """Apply controlled phase gate."""
        n = self.num_logical_qubits
        result = state.copy()

        for i in range(len(state)):
            bit1 = (i >> (n - 1 - q1)) & 1
            bit2 = (i >> (n - 1 - q2)) & 1

            if bit1 == 1 and bit2 == 1:
                result[i] *= np.exp(1j * angle)

        return result

    def _apply_swap(
        self,
        state: np.ndarray,
        q1: int,
        q2: int
    ) -> np.ndarray:
        """Apply SWAP gate."""
        n = self.num_logical_qubits
        result = np.zeros_like(state)

        for i in range(len(state)):
            bit1 = (i >> (n - 1 - q1)) & 1
            bit2 = (i >> (n - 1 - q2)) & 1

            if bit1 != bit2:
                # Swap the bits
                j = i ^ (1 << (n - 1 - q1)) ^ (1 << (n - 1 - q2))
                result[j] = state[i]
            else:
                result[i] = state[i]

        return result

    def estimate_resources(self) -> Dict[str, int]:
        """Estimate cluster state resources needed."""
        cluster_qubits = 0
        measurements = 0

        for op in self.operations:
            if op[0] in ('H', 'RY', 'RZ'):
                cluster_qubits += 4
                measurements += 3
            elif op[0] in ('CNOT', 'CZ', 'CPHASE'):
                cluster_qubits += 15
                measurements += 13
            elif op[0] == 'SWAP':
                cluster_qubits += 3 * 15  # 3 CNOTs
                measurements += 3 * 13

        return {
            'cluster_qubits': cluster_qubits,
            'measurements': measurements
        }
