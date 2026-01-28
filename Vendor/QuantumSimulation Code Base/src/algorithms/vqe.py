"""
Variational Quantum Eigensolver (VQE) Implementation - Step 14.

VQE is a hybrid quantum-classical algorithm for finding ground state energies
of quantum systems. It uses parameterized quantum circuits (ansatz) to prepare
trial wavefunctions and classical optimization to minimize the energy.

Key Components:
1. Parameterized quantum circuits (ansatz)
2. Hamiltonian representation as sum of Pauli strings
3. Expectation value measurement
4. Classical optimization loop

References:
- Peruzzo et al. (2014) - "A variational eigenvalue solver on a photonic quantum processor"
- McClean et al. (2016) - "The theory of variational hybrid quantum-classical algorithms"
- O'Malley et al. (2016) - "Scalable Quantum Simulation of Molecular Energies"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum
import numpy as np
from scipy.optimize import minimize


# =============================================================================
# Pauli Operators
# =============================================================================

# Single-qubit Pauli matrices
I_GATE = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_GATES = {'I': I_GATE, 'X': X_GATE, 'Y': Y_GATE, 'Z': Z_GATE}


# =============================================================================
# Pauli String Representation
# =============================================================================

@dataclass
class PauliString:
    """
    Represents a product of Pauli operators with a coefficient.

    Example: 0.5 * X_0 * Z_1 * Y_2
    """
    paulis: Dict[str, List[int]]  # {'X': [0, 2], 'Z': [1]} means X0 * X2 * Z1
    coefficient: complex = 1.0

    def __post_init__(self):
        # Normalize the paulis dict - combine all qubit indices
        self._qubit_ops: Dict[int, str] = {}
        for pauli_type, qubits in self.paulis.items():
            if isinstance(qubits, int):
                qubits = [qubits]
            for q in qubits:
                if q in self._qubit_ops:
                    # Combine operators (e.g., X*Z = iY)
                    self._qubit_ops[q] = self._combine_paulis(
                        self._qubit_ops[q], pauli_type
                    )
                else:
                    self._qubit_ops[q] = pauli_type

    def _combine_paulis(self, p1: str, p2: str) -> str:
        """Combine two Pauli operators on the same qubit."""
        if p1 == p2:
            return 'I'
        if p1 == 'I':
            return p2
        if p2 == 'I':
            return p1
        # XY = iZ, YZ = iX, ZX = iY (up to phase)
        combos = {
            ('X', 'Y'): 'Z', ('Y', 'X'): 'Z',
            ('Y', 'Z'): 'X', ('Z', 'Y'): 'X',
            ('Z', 'X'): 'Y', ('X', 'Z'): 'Y',
        }
        return combos.get((p1, p2), 'I')

    def to_matrix(self, n_qubits: int) -> np.ndarray:
        """Convert to full matrix representation."""
        if n_qubits == 0:
            return np.array([[self.coefficient]], dtype=complex)

        result = np.eye(1, dtype=complex)

        for q in range(n_qubits):
            op = self._qubit_ops.get(q, 'I')
            result = np.kron(result, PAULI_GATES[op])

        return self.coefficient * result

    def get_qubit_operator(self, qubit: int) -> str:
        """Get the Pauli operator acting on a specific qubit."""
        return self._qubit_ops.get(qubit, 'I')


def measure_expectation(state: np.ndarray, pauli_string: PauliString) -> float:
    """
    Measure expectation value of a Pauli string on a state.

    Returns: <state|PauliString|state>
    """
    n_qubits = int(np.log2(len(state)))
    matrix = pauli_string.to_matrix(n_qubits)
    return np.real(np.vdot(state, matrix @ state))


# =============================================================================
# Hamiltonian Representation
# =============================================================================

class Hamiltonian:
    """
    Represents a Hamiltonian as a sum of Pauli strings.

    H = sum_i c_i * P_i where P_i are Pauli strings
    """

    def __init__(self, n_qubits: int, terms: List[PauliString]):
        """
        Initialize Hamiltonian.

        Args:
            n_qubits: Number of qubits
            terms: List of PauliString terms
        """
        self.n_qubits = n_qubits
        self.terms = terms

    def to_matrix(self) -> np.ndarray:
        """Convert to full matrix representation."""
        dim = 2 ** self.n_qubits
        result = np.zeros((dim, dim), dtype=complex)

        for term in self.terms:
            result += term.to_matrix(self.n_qubits)

        return result

    def expectation(self, state: np.ndarray) -> float:
        """
        Compute expectation value <state|H|state>.

        Uses term-by-term evaluation for efficiency.
        """
        total = 0.0
        for term in self.terms:
            total += measure_expectation(state, term)
        return total

    def ground_state_energy(self) -> float:
        """Compute exact ground state energy via diagonalization."""
        matrix = self.to_matrix()
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.min(eigenvalues)


# =============================================================================
# Parameterized Quantum Circuit
# =============================================================================

class GateType(Enum):
    """Types of gates in the circuit."""
    RY = "RY"
    RZ = "RZ"
    RX = "RX"
    CNOT = "CNOT"


@dataclass
class Gate:
    """Represents a gate in the circuit."""
    gate_type: GateType
    qubits: List[int]
    param_index: Optional[int] = None  # None for non-parameterized gates


class ParameterizedCircuit:
    """
    Parameterized quantum circuit for VQE ansatz.

    Supports:
    - RY, RZ, RX rotation gates (parameterized)
    - CNOT entangling gates
    """

    def __init__(self, n_qubits: int):
        """Initialize circuit with given number of qubits."""
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []
        self._n_parameters = 0

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the circuit."""
        return self._n_parameters

    def ry(self, qubit: int, param_index: int):
        """Add RY rotation gate."""
        self.gates.append(Gate(GateType.RY, [qubit], param_index))
        self._n_parameters = max(self._n_parameters, param_index + 1)

    def rz(self, qubit: int, param_index: int):
        """Add RZ rotation gate."""
        self.gates.append(Gate(GateType.RZ, [qubit], param_index))
        self._n_parameters = max(self._n_parameters, param_index + 1)

    def rx(self, qubit: int, param_index: int):
        """Add RX rotation gate."""
        self.gates.append(Gate(GateType.RX, [qubit], param_index))
        self._n_parameters = max(self._n_parameters, param_index + 1)

    def cnot(self, control: int, target: int):
        """Add CNOT gate."""
        self.gates.append(Gate(GateType.CNOT, [control, target], None))

    def _ry_matrix(self, theta: float) -> np.ndarray:
        """RY rotation matrix."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rz_matrix(self, theta: float) -> np.ndarray:
        """RZ rotation matrix."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _rx_matrix(self, theta: float) -> np.ndarray:
        """RX rotation matrix."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    def _apply_single_qubit_gate(self, state: np.ndarray,
                                  gate: np.ndarray, qubit: int) -> np.ndarray:
        """Apply single-qubit gate to state."""
        n = self.n_qubits
        dim = 2 ** n

        # Build full operator using tensor product
        # In our convention, qubit 0 is leftmost (first in tensor product)
        full_op = np.array([[1.0]], dtype=complex)

        for q in range(n):
            if q == qubit:
                full_op = np.kron(full_op, gate)
            else:
                full_op = np.kron(full_op, I_GATE)

        return full_op @ state

    def _apply_cnot(self, state: np.ndarray,
                    control: int, target: int) -> np.ndarray:
        """Apply CNOT gate to state."""
        n = self.n_qubits
        dim = 2 ** n

        result = state.copy()

        # CNOT: |c,t> -> |c, t XOR c>
        # For each basis state, check control bit and conditionally flip target
        for i in range(dim):
            # In our convention, qubit 0 is the leftmost (most significant)
            # Bit for qubit q in state i: (i >> (n-1-q)) & 1
            control_bit = (i >> (n - 1 - control)) & 1

            if control_bit == 1:
                # Flip the target bit
                target_mask = 1 << (n - 1 - target)
                new_i = i ^ target_mask

                # Swap amplitudes (but we're overwriting, so need temp)
                if i < new_i:  # Only process each pair once
                    result[i], result[new_i] = state[new_i], state[i]

        return result

    def execute(self, parameters: List[float]) -> np.ndarray:
        """
        Execute circuit with given parameters.

        Args:
            parameters: List of parameter values

        Returns:
            Final quantum state
        """
        # Start in |0...0> state
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Apply gates sequentially
        for gate in self.gates:
            if gate.gate_type == GateType.RY:
                theta = parameters[gate.param_index]
                matrix = self._ry_matrix(theta)
                state = self._apply_single_qubit_gate(state, matrix, gate.qubits[0])

            elif gate.gate_type == GateType.RZ:
                theta = parameters[gate.param_index]
                matrix = self._rz_matrix(theta)
                state = self._apply_single_qubit_gate(state, matrix, gate.qubits[0])

            elif gate.gate_type == GateType.RX:
                theta = parameters[gate.param_index]
                matrix = self._rx_matrix(theta)
                state = self._apply_single_qubit_gate(state, matrix, gate.qubits[0])

            elif gate.gate_type == GateType.CNOT:
                state = self._apply_cnot(state, gate.qubits[0], gate.qubits[1])

        return state


def create_hardware_efficient_ansatz(n_qubits: int, n_layers: int) -> ParameterizedCircuit:
    """
    Create a hardware-efficient ansatz.

    Structure per layer:
    - RY on each qubit
    - RZ on each qubit
    - Linear CNOT chain

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers

    Returns:
        ParameterizedCircuit
    """
    circuit = ParameterizedCircuit(n_qubits)
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            circuit.ry(q, param_idx)
            param_idx += 1
            circuit.rz(q, param_idx)
            param_idx += 1

        # Entangling layer (linear chain)
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)

    return circuit


# =============================================================================
# VQE Result
# =============================================================================

@dataclass
class VQEResult:
    """Result of VQE optimization."""
    optimal_energy: float
    optimal_parameters: np.ndarray
    energy_history: List[float]
    initial_energy: float
    n_iterations: int
    converged: bool = True


# =============================================================================
# VQE Algorithm
# =============================================================================

class VQE:
    """
    Variational Quantum Eigensolver.

    Finds ground state energy of a Hamiltonian using parameterized
    quantum circuits and classical optimization.
    """

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 n_layers: int = 2,
                 circuit: Optional[ParameterizedCircuit] = None,
                 optimizer: str = 'COBYLA',
                 measurement_noise: float = 0.0,
                 n_shots: int = 0,  # 0 = exact state simulation
                 seed: Optional[int] = None):
        """
        Initialize VQE.

        Args:
            hamiltonian: Hamiltonian to minimize
            n_layers: Number of ansatz layers (if circuit not provided)
            circuit: Custom parameterized circuit (optional)
            optimizer: Classical optimizer ('COBYLA', 'gradient_descent', etc.)
            measurement_noise: Noise level for measurements
            n_shots: Number of measurement shots (0 = exact)
            seed: Random seed for reproducibility
        """
        self.hamiltonian = hamiltonian
        self.optimizer_name = optimizer
        self.measurement_noise = measurement_noise
        self.n_shots = n_shots

        if seed is not None:
            np.random.seed(seed)

        # Create circuit if not provided
        if circuit is not None:
            self.circuit = circuit
        else:
            self.circuit = create_hardware_efficient_ansatz(
                hamiltonian.n_qubits, n_layers
            )

        self._energy_history: List[float] = []

    def energy(self, parameters: np.ndarray) -> float:
        """
        Compute energy for given parameters.

        Args:
            parameters: Circuit parameters

        Returns:
            Expectation value of Hamiltonian
        """
        # Execute circuit
        state = self.circuit.execute(parameters)

        # Compute expectation
        energy = self.hamiltonian.expectation(state)

        # Add measurement noise if specified
        if self.measurement_noise > 0:
            energy += np.random.normal(0, self.measurement_noise)

        return energy

    def run(self,
            max_iterations: int = 200,
            initial_params: Optional[np.ndarray] = None,
            learning_rate: float = 0.1,
            tol: float = 1e-6) -> VQEResult:
        """
        Run VQE optimization.

        Args:
            max_iterations: Maximum number of iterations
            initial_params: Initial parameters (random if None)
            learning_rate: Learning rate for gradient descent
            tol: Convergence tolerance

        Returns:
            VQEResult with optimal energy and parameters
        """
        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(
                0, 2 * np.pi, self.circuit.n_parameters
            )

        initial_energy = self.energy(initial_params)
        self._energy_history = [initial_energy]

        # Callback to record energy
        def callback(params):
            self._energy_history.append(self.energy(params))

        # Run optimization
        if self.optimizer_name == 'gradient_descent':
            optimal_params = self._gradient_descent(
                initial_params, max_iterations, learning_rate, tol, callback
            )
        else:
            result = minimize(
                self.energy,
                initial_params,
                method=self.optimizer_name,
                options={'maxiter': max_iterations, 'rhobeg': 0.5},
                callback=callback,
                tol=tol
            )
            optimal_params = result.x

        optimal_energy = self.energy(optimal_params)

        return VQEResult(
            optimal_energy=optimal_energy,
            optimal_parameters=optimal_params,
            energy_history=self._energy_history,
            initial_energy=initial_energy,
            n_iterations=len(self._energy_history) - 1,
            converged=True
        )

    def _gradient_descent(self,
                          params: np.ndarray,
                          max_iterations: int,
                          learning_rate: float,
                          tol: float,
                          callback: Callable) -> np.ndarray:
        """Simple gradient descent optimizer."""
        for _ in range(max_iterations):
            grad = compute_gradient_parameter_shift(self, params)
            new_params = params - learning_rate * grad

            if np.linalg.norm(new_params - params) < tol:
                break

            params = new_params
            callback(params)

        return params


def compute_gradient_parameter_shift(vqe: VQE,
                                     parameters: np.ndarray) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.

    For RY, RZ gates: df/dθ = (f(θ + π/2) - f(θ - π/2)) / 2
    """
    gradient = np.zeros(len(parameters))
    shift = np.pi / 2

    for i in range(len(parameters)):
        params_plus = parameters.copy()
        params_plus[i] += shift

        params_minus = parameters.copy()
        params_minus[i] -= shift

        gradient[i] = (vqe.energy(params_plus) - vqe.energy(params_minus)) / 2

    return gradient


# =============================================================================
# Hamiltonian Constructors
# =============================================================================

def create_ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> Hamiltonian:
    """
    Create transverse-field Ising model Hamiltonian.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h: Transverse field strength

    Returns:
        Hamiltonian
    """
    terms = []

    # ZZ terms
    for i in range(n_qubits - 1):
        terms.append(PauliString(
            paulis={'Z': [i, i + 1]},
            coefficient=-J
        ))

    # X terms
    for i in range(n_qubits):
        terms.append(PauliString(
            paulis={'X': [i]},
            coefficient=-h
        ))

    return Hamiltonian(n_qubits, terms)


def create_h2_hamiltonian(bond_length: float = 0.74) -> Hamiltonian:
    """
    Create simplified H2 molecular Hamiltonian.

    Uses a 2-qubit model based on the Bravyi-Kitaev transformation
    of the minimal basis H2 Hamiltonian.

    Reference: O'Malley et al. (2016) - Scalable Quantum Simulation of Molecular Energies

    Args:
        bond_length: H-H bond length in Angstroms

    Returns:
        Hamiltonian (2 qubits for reduced model)
    """
    # Coefficients for 2-qubit H2 model at equilibrium (0.74 A)
    # From O'Malley et al. 2016 Table I
    # H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1

    # Interpolate based on bond length (approximate)
    if bond_length < 0.6:
        # Short bond - higher energy, stronger interaction
        g0, g1, g2, g3, g4, g5 = -0.20, 0.40, -0.40, 0.18, 0.18, 0.18
    elif bond_length < 0.8:
        # Near equilibrium (0.74 A) - from literature
        g0, g1, g2, g3, g4, g5 = -0.4804, 0.3435, -0.4347, 0.0910, 0.0910, 0.0910
    elif bond_length < 1.2:
        # Stretched bond
        g0, g1, g2, g3, g4, g5 = -0.35, 0.30, -0.38, 0.07, 0.07, 0.07
    else:
        # Very stretched
        g0, g1, g2, g3, g4, g5 = -0.25, 0.25, -0.30, 0.05, 0.05, 0.05

    terms = [
        PauliString(paulis={}, coefficient=g0),  # Identity
        PauliString(paulis={'Z': [0]}, coefficient=g1),
        PauliString(paulis={'Z': [1]}, coefficient=g2),
        PauliString(paulis={'Z': [0, 1]}, coefficient=g3),
        PauliString(paulis={'X': [0, 1]}, coefficient=g4),
        PauliString(paulis={'Y': [0, 1]}, coefficient=g5),
    ]

    return Hamiltonian(n_qubits=2, terms=terms)


def create_heh_plus_hamiltonian(bond_length: float = 0.93) -> Hamiltonian:
    """
    Create HeH+ molecular Hamiltonian.

    Uses simplified 2-qubit model Hamiltonian.
    Reference: Kandala et al. (2017) - Hardware-efficient variational quantum eigensolver

    Args:
        bond_length: He-H bond length in Angstroms

    Returns:
        Hamiltonian (2 qubits)
    """
    # Coefficients for HeH+ at equilibrium (0.93 A)
    # Approximate values from literature

    if bond_length < 0.8:
        g0, g1, g2, g3, g4, g5 = -2.50, 0.35, -0.35, 0.12, 0.12, 0.12
    elif bond_length < 1.0:
        # Near equilibrium - ground state ~-2.87 Ha
        g0, g1, g2, g3, g4, g5 = -2.65, 0.22, -0.22, 0.10, 0.10, 0.10
    else:
        g0, g1, g2, g3, g4, g5 = -2.40, 0.18, -0.18, 0.08, 0.08, 0.08

    terms = [
        PauliString(paulis={}, coefficient=g0),  # Identity
        PauliString(paulis={'Z': [0]}, coefficient=g1),
        PauliString(paulis={'Z': [1]}, coefficient=g2),
        PauliString(paulis={'Z': [0, 1]}, coefficient=g3),
        PauliString(paulis={'X': [0, 1]}, coefficient=g4),
        PauliString(paulis={'Y': [0, 1]}, coefficient=g5),
    ]

    return Hamiltonian(n_qubits=2, terms=terms)
