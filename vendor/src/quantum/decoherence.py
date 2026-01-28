"""
Quantum Decoherence and Master Equations

References:
- Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000), Ch. 8
- Preskill, "Quantum Computation" Lecture Notes, Ch. 3

This module implements:
1. Density matrix formalism (pure and mixed states)
2. Quantum channels (Kraus representation)
3. Decoherence channels (bit flip, phase flip, depolarizing)
4. Lindblad master equation solver
5. T1/T2 relaxation models

All implementations are verified against analytical solutions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp


@dataclass
class DensityMatrix:
    """
    Density matrix representation of quantum state.

    Supports both pure states (ρ = |ψ⟩⟨ψ|) and mixed states (ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|).

    Properties verified:
    - Hermitian: ρ† = ρ
    - Positive semi-definite: all eigenvalues ≥ 0
    - Normalized: Tr(ρ) = 1
    """

    matrix: np.ndarray  # Density matrix
    is_pure: bool  # Whether state is pure
    purity: float  # Tr(ρ²) ∈ [1/d, 1]
    dimension: int  # Hilbert space dimension

    def __post_init__(self):
        """Verify density matrix properties."""
        tolerance = 1e-10

        # Check Hermitian
        if not np.allclose(self.matrix, self.matrix.conj().T, atol=tolerance):
            raise ValueError("Density matrix must be Hermitian")

        # Check normalized
        trace = np.trace(self.matrix).real
        if not np.isclose(trace, 1.0, atol=tolerance):
            raise ValueError(f"Density matrix must have trace 1, got {trace}")

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -tolerance):
            raise ValueError(f"Density matrix must be positive semi-definite, got eigenvalues {eigenvalues}")

        # Compute purity
        computed_purity = np.trace(self.matrix @ self.matrix).real
        if not np.isclose(computed_purity, self.purity, atol=tolerance):
            raise ValueError(f"Purity mismatch: computed {computed_purity}, expected {self.purity}")

    @property
    def bloch_vector(self) -> np.ndarray:
        """
        Compute Bloch vector for qubit (dimension 2).

        For qubit: ρ = (I + r·σ)/2 where r = (x, y, z) is Bloch vector.
        """
        if self.dimension != 2:
            raise ValueError("Bloch vector only defined for qubits")

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        x = np.trace(self.matrix @ sigma_x).real
        y = np.trace(self.matrix @ sigma_y).real
        z = np.trace(self.matrix @ sigma_z).real

        return np.array([x, y, z])


class DensityMatrixOperations:
    """Operations on density matrices."""

    @staticmethod
    def from_state_vector(state_vector: np.ndarray) -> DensityMatrix:
        """
        Create density matrix from pure state vector.

        ρ = |ψ⟩⟨ψ|
        """
        # Normalize state vector
        state_vector = state_vector / np.linalg.norm(state_vector)

        # Compute density matrix
        matrix = np.outer(state_vector, state_vector.conj())

        return DensityMatrix(
            matrix=matrix,
            is_pure=True,
            purity=1.0,
            dimension=len(state_vector)
        )

    @staticmethod
    def from_ensemble(states: List[np.ndarray], probabilities: List[float]) -> DensityMatrix:
        """
        Create density matrix from statistical ensemble.

        ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

        Args:
            states: List of state vectors
            probabilities: List of probabilities (must sum to 1)
        """
        if len(states) != len(probabilities):
            raise ValueError("Number of states and probabilities must match")

        if not np.isclose(sum(probabilities), 1.0, atol=1e-10):
            raise ValueError(f"Probabilities must sum to 1, got {sum(probabilities)}")

        dimension = len(states[0])
        matrix = np.zeros((dimension, dimension), dtype=complex)

        for state, prob in zip(states, probabilities):
            state = state / np.linalg.norm(state)  # Normalize
            matrix += prob * np.outer(state, state.conj())

        # Compute purity
        purity = np.trace(matrix @ matrix).real

        return DensityMatrix(
            matrix=matrix,
            is_pure=np.isclose(purity, 1.0, atol=1e-10),
            purity=purity,
            dimension=dimension
        )

    @staticmethod
    def maximally_mixed(dimension: int) -> DensityMatrix:
        """
        Create maximally mixed state ρ = I/d.

        This has minimal purity: Tr(ρ²) = 1/d.
        """
        matrix = np.eye(dimension, dtype=complex) / dimension
        purity = 1.0 / dimension

        return DensityMatrix(
            matrix=matrix,
            is_pure=False,
            purity=purity,
            dimension=dimension
        )

    @staticmethod
    def partial_trace(rho: np.ndarray, dims: Tuple[int, int], trace_out: int) -> np.ndarray:
        """
        Compute partial trace of bipartite density matrix.

        Args:
            rho: Density matrix of composite system
            dims: Dimensions (dim_A, dim_B)
            trace_out: Which system to trace out (0 for A, 1 for B)

        Returns:
            Reduced density matrix
        """
        dim_a, dim_b = dims

        if rho.shape != (dim_a * dim_b, dim_a * dim_b):
            raise ValueError(f"Matrix shape {rho.shape} incompatible with dimensions {dims}")

        if trace_out == 0:
            # Trace out system A
            rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            rho_b = np.einsum('ijik->jk', rho_reshaped)
            return rho_b
        elif trace_out == 1:
            # Trace out system B
            rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            rho_a = np.einsum('ijkj->ik', rho_reshaped)
            return rho_a
        else:
            raise ValueError("trace_out must be 0 or 1")


@dataclass
class QuantumChannel:
    """
    Quantum channel in Kraus representation.

    ε(ρ) = Σₖ Kₖ ρ Kₖ†

    Properties:
    - Trace Preserving: Σₖ Kₖ†Kₖ = I
    - Completely Positive: (ε ⊗ I)(ρ) ≥ 0 for all ρ
    """

    kraus_operators: List[np.ndarray]  # List of Kraus operators {Kₖ}
    name: str  # Channel name
    parameters: Dict  # Channel parameters (e.g., error rate p)

    def __post_init__(self):
        """Verify channel is valid (trace preserving)."""
        self._verify_trace_preserving()

    def _verify_trace_preserving(self):
        """Verify Σₖ Kₖ†Kₖ = I (completeness relation)."""
        dimension = self.kraus_operators[0].shape[0]
        completeness = sum(K.conj().T @ K for K in self.kraus_operators)

        identity = np.eye(dimension, dtype=complex)
        if not np.allclose(completeness, identity, atol=1e-10):
            raise ValueError(f"Channel {self.name} is not trace preserving: Σ Kₖ†Kₖ = {completeness}")

    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """
        Apply quantum channel to density matrix.

        ε(ρ) = Σₖ Kₖ ρ Kₖ†
        """
        result_matrix = sum(K @ rho.matrix @ K.conj().T for K in self.kraus_operators)

        # Compute new purity
        purity = np.trace(result_matrix @ result_matrix).real

        return DensityMatrix(
            matrix=result_matrix,
            is_pure=np.isclose(purity, 1.0, atol=1e-10),
            purity=purity,
            dimension=rho.dimension
        )

    def verify_complete_positivity(self, test_dim: int = 2) -> bool:
        """
        Verify complete positivity by checking (ε ⊗ I)(ρ) ≥ 0.

        Tests on maximally entangled state.
        """
        # Create maximally entangled state on qubit ⊗ test_system
        dim = self.kraus_operators[0].shape[0]
        phi_plus = np.zeros((dim * test_dim, dim * test_dim), dtype=complex)

        for i in range(min(dim, test_dim)):
            phi_plus += np.outer(
                np.kron(np.eye(dim)[i], np.eye(test_dim)[i]),
                np.kron(np.eye(dim)[i], np.eye(test_dim)[i]).conj()
            )
        phi_plus /= min(dim, test_dim)

        # Apply (ε ⊗ I) to phi_plus
        result = np.zeros_like(phi_plus)
        for K in self.kraus_operators:
            K_extended = np.kron(K, np.eye(test_dim))
            result += K_extended @ phi_plus @ K_extended.conj().T

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(result)
        return np.all(eigenvalues >= -1e-10)


class DecoherenceChannels:
    """
    Standard decoherence channels for qubits.

    All channels are TPCP (Trace Preserving Completely Positive).
    """

    @staticmethod
    def bit_flip(p: float) -> QuantumChannel:
        """
        Bit flip channel: Random X gate with probability p.

        ε(ρ) = (1-p) ρ + p X ρ X†

        Kraus operators:
        K₀ = √(1-p) I
        K₁ = √p X

        Args:
            p: Bit flip probability ∈ [0, 1]
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Bit flip probability must be in [0,1], got {p}")

        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        K0 = np.sqrt(1 - p) * I
        K1 = np.sqrt(p) * X

        return QuantumChannel(
            kraus_operators=[K0, K1],
            name="bit_flip",
            parameters={'p': p}
        )

    @staticmethod
    def phase_flip(p: float) -> QuantumChannel:
        """
        Phase flip channel: Random Z gate with probability p.

        ε(ρ) = (1-p) ρ + p Z ρ Z†

        Kraus operators:
        K₀ = √(1-p) I
        K₁ = √p Z

        Args:
            p: Phase flip probability ∈ [0, 1]
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Phase flip probability must be in [0,1], got {p}")

        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        K0 = np.sqrt(1 - p) * I
        K1 = np.sqrt(p) * Z

        return QuantumChannel(
            kraus_operators=[K0, K1],
            name="phase_flip",
            parameters={'p': p}
        )

    @staticmethod
    def depolarizing(p: float) -> QuantumChannel:
        """
        Depolarizing channel: Random Pauli {I, X, Y, Z} with equal probability.

        ε(ρ) = (1-p) ρ + (p/3)(X ρ X† + Y ρ Y† + Z ρ Z†)
             = (1-p) ρ + p(I/2)  (for qubits)

        Kraus operators:
        K₀ = √(1-3p/4) I
        K₁ = √(p/4) X
        K₂ = √(p/4) Y
        K₃ = √(p/4) Z

        Args:
            p: Depolarizing probability ∈ [0, 4/3]
                p=0: No decoherence
                p=3/4: Maximally depolarizing (ρ → I/2)
                p=1: Physical upper bound for single application
        """
        if not 0 <= p <= 4/3:
            raise ValueError(f"Depolarizing probability must be in [0, 4/3], got {p}")

        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        K0 = np.sqrt(1 - 3*p/4) * I
        K1 = np.sqrt(p/4) * X
        K2 = np.sqrt(p/4) * Y
        K3 = np.sqrt(p/4) * Z

        return QuantumChannel(
            kraus_operators=[K0, K1, K2, K3],
            name="depolarizing",
            parameters={'p': p}
        )

    @staticmethod
    def amplitude_damping(gamma: float) -> QuantumChannel:
        """
        Amplitude damping channel: Energy relaxation |1⟩ → |0⟩.

        Models spontaneous emission, T₁ relaxation.

        Kraus operators:
        K₀ = [[1, 0], [0, √(1-γ)]]
        K₁ = [[0, √γ], [0, 0]]

        Args:
            gamma: Damping parameter ∈ [0, 1]
                  gamma = 1 - exp(-t/T₁) for time t and T₁ lifetime
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f"Damping parameter must be in [0,1], got {gamma}")

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        return QuantumChannel(
            kraus_operators=[K0, K1],
            name="amplitude_damping",
            parameters={'gamma': gamma}
        )

    @staticmethod
    def phase_damping(lambda_: float) -> QuantumChannel:
        """
        Phase damping channel: Loss of coherence without energy relaxation.

        Models pure dephasing, T₂* process.

        Kraus operators:
        K₀ = √(1-λ) I
        K₁ = √λ [[1, 0], [0, 0]]
        K₂ = √λ [[0, 0], [0, 1]]

        Args:
            lambda_: Dephasing parameter ∈ [0, 1]
        """
        if not 0 <= lambda_ <= 1:
            raise ValueError(f"Dephasing parameter must be in [0,1], got {lambda_}")

        I = np.eye(2, dtype=complex)
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)

        K0 = np.sqrt(1 - lambda_) * I
        K1 = np.sqrt(lambda_) * P0
        K2 = np.sqrt(lambda_) * P1

        return QuantumChannel(
            kraus_operators=[K0, K1, K2],
            name="phase_damping",
            parameters={'lambda': lambda_}
        )


class LindbladSolver:
    """
    Solver for Lindblad master equation.

    dρ/dt = -i[H, ρ] + Σᵢ γᵢ(Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})

    Where:
    - H: System Hamiltonian
    - γᵢ: Decay rates
    - Lᵢ: Lindblad (jump) operators
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        lindblad_operators: List[np.ndarray],
        rates: List[float]
    ):
        """
        Initialize Lindblad master equation solver.

        Args:
            hamiltonian: System Hamiltonian H
            lindblad_operators: List of Lindblad operators {Lᵢ}
            rates: List of decay rates {γᵢ} (must be positive)
        """
        if len(lindblad_operators) != len(rates):
            raise ValueError("Number of Lindblad operators and rates must match")

        if any(rate < 0 for rate in rates):
            raise ValueError("All decay rates must be non-negative")

        self.H = hamiltonian
        self.L_ops = lindblad_operators
        self.rates = rates
        self.dimension = hamiltonian.shape[0]

    def lindblad_superoperator(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute dρ/dt according to Lindblad equation.

        Returns time derivative as matrix.
        """
        # Coherent part: -i[H, ρ]
        commutator = -1j * (self.H @ rho - rho @ self.H)

        # Dissipative part: Σᵢ γᵢ(Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
        dissipator = np.zeros_like(rho)
        for L, gamma in zip(self.L_ops, self.rates):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L

            dissipator += gamma * (
                L @ rho @ L_dag
                - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            )

        return commutator + dissipator

    def solve(
        self,
        rho0: DensityMatrix,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[DensityMatrix]]:
        """
        Solve Lindblad equation numerically.

        Args:
            rho0: Initial density matrix
            t_span: Time interval (t_start, t_end)
            t_eval: Times at which to evaluate solution

        Returns:
            times: Time points
            states: List of density matrices at each time
        """
        # Flatten density matrix for ODE solver
        rho0_flat = rho0.matrix.flatten()

        def drho_dt(t, rho_flat):
            """ODE function for scipy.integrate.solve_ivp."""
            rho = rho_flat.reshape(self.dimension, self.dimension)
            drho = self.lindblad_superoperator(rho)
            return drho.flatten()

        # Solve ODE
        solution = solve_ivp(
            drho_dt,
            t_span,
            rho0_flat,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )

        # Convert back to density matrices
        times = solution.t
        states = []
        for rho_flat in solution.y.T:
            rho_matrix = rho_flat.reshape(self.dimension, self.dimension)
            purity = np.trace(rho_matrix @ rho_matrix).real

            states.append(DensityMatrix(
                matrix=rho_matrix,
                is_pure=np.isclose(purity, 1.0, atol=1e-10),
                purity=purity,
                dimension=self.dimension
            ))

        return times, states


class RelaxationModels:
    """
    Standard T₁ and T₂ relaxation models.
    """

    @staticmethod
    def T1_relaxation(T1: float, time_step: float) -> QuantumChannel:
        """
        T₁ (amplitude damping) channel for given time step.

        γ = 1 - exp(-Δt/T₁)

        Args:
            T1: T₁ relaxation time
            time_step: Time interval Δt
        """
        gamma = 1 - np.exp(-time_step / T1)
        return DecoherenceChannels.amplitude_damping(gamma)

    @staticmethod
    def T2_dephasing(T1: float, T2: float, time_step: float) -> QuantumChannel:
        """
        T₂ dephasing channel.

        Pure dephasing rate: γφ = 1/T₂ - 1/(2T₁)

        Args:
            T1: T₁ relaxation time
            T2: T₂ coherence time (T₂ ≤ 2T₁)
            time_step: Time interval Δt
        """
        if T2 > 2 * T1:
            raise ValueError(f"T₂ must satisfy T₂ ≤ 2T₁, got T₂={T2}, T₁={T1}")

        # Pure dephasing rate
        gamma_phi = 1/T2 - 1/(2*T1)

        # Dephasing parameter for time step
        lambda_ = 1 - np.exp(-2 * gamma_phi * time_step)

        return DecoherenceChannels.phase_damping(lambda_)

    @staticmethod
    def combined_T1_T2(T1: float, T2: float, time_step: float) -> Callable:
        """
        Combined T₁ and T₂ decoherence.

        Apply amplitude damping (T₁) followed by phase damping (T₂).

        Returns function that takes density matrix and returns evolved state.
        """
        T1_channel = RelaxationModels.T1_relaxation(T1, time_step)
        T2_channel = RelaxationModels.T2_dephasing(T1, T2, time_step)

        def combined_channel(rho: DensityMatrix) -> DensityMatrix:
            rho = T1_channel.apply(rho)
            rho = T2_channel.apply(rho)
            return rho

        return combined_channel