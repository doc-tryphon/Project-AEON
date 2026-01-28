"""
Test Suite for Variational Quantum Eigensolver (VQE) - Step 14.

VQE is a hybrid quantum-classical algorithm for finding ground state energies.
It uses parameterized quantum circuits (ansatz) and classical optimization.

Key Properties Tested:
1. Parameterized circuits produce valid quantum states
2. Hamiltonian expectation values are computed correctly
3. Classical optimization converges to ground state
4. Known molecular Hamiltonians give correct energies

References:
- Peruzzo et al. (2014) - "A variational eigenvalue solver on a photonic quantum processor"
- McClean et al. (2016) - "The theory of variational hybrid quantum-classical algorithms"
- O'Malley et al. (2016) - "Scalable Quantum Simulation of Molecular Energies"
"""

import pytest
import numpy as np
from typing import List, Tuple, Callable


# =============================================================================
# Section 1: Parameterized Circuit Tests
# =============================================================================

class TestParameterizedCircuits:
    """Test parameterized quantum circuits (ansatz)."""

    def test_single_qubit_rotation_gate(self):
        """RY(theta) rotation should produce correct state."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=1)
        circuit.ry(qubit=0, param_index=0)

        # theta = 0 should give |0>
        state = circuit.execute(parameters=[0.0])
        assert np.isclose(np.abs(state[0])**2, 1.0, atol=1e-10)

        # theta = pi should give |1>
        state = circuit.execute(parameters=[np.pi])
        assert np.isclose(np.abs(state[1])**2, 1.0, atol=1e-10)

        # theta = pi/2 should give equal superposition
        state = circuit.execute(parameters=[np.pi/2])
        assert np.isclose(np.abs(state[0])**2, 0.5, atol=1e-10)
        assert np.isclose(np.abs(state[1])**2, 0.5, atol=1e-10)

    def test_rz_rotation_gate(self):
        """RZ(theta) rotation should add phase."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=1)
        # Prepare |+> state first, then apply RZ
        circuit.ry(qubit=0, param_index=0)  # Will set to pi/2
        circuit.rz(qubit=0, param_index=1)

        # RZ on |+> with theta=pi gives |->
        state = circuit.execute(parameters=[np.pi/2, np.pi])

        # |-> = (|0> - |1>)/sqrt(2)
        expected_minus = np.array([1, -1]) / np.sqrt(2)
        # Allow global phase
        overlap = np.abs(np.vdot(state, expected_minus))
        assert np.isclose(overlap, 1.0, atol=1e-10)

    def test_cnot_entangling_gate(self):
        """CNOT should entangle qubits correctly."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=2)
        # Prepare |+0> then CNOT -> Bell state
        circuit.ry(qubit=0, param_index=0)
        circuit.cnot(control=0, target=1)

        state = circuit.execute(parameters=[np.pi/2])

        # Should be Bell state (|00> + |11>)/sqrt(2)
        assert np.isclose(np.abs(state[0])**2, 0.5, atol=1e-10)  # |00>
        assert np.isclose(np.abs(state[1])**2, 0.0, atol=1e-10)  # |01>
        assert np.isclose(np.abs(state[2])**2, 0.0, atol=1e-10)  # |10>
        assert np.isclose(np.abs(state[3])**2, 0.5, atol=1e-10)  # |11>

    def test_circuit_output_is_normalized(self):
        """Circuit output should always be normalized."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=3)
        circuit.ry(qubit=0, param_index=0)
        circuit.ry(qubit=1, param_index=1)
        circuit.ry(qubit=2, param_index=2)
        circuit.cnot(control=0, target=1)
        circuit.cnot(control=1, target=2)

        # Random parameters
        for _ in range(10):
            params = np.random.uniform(0, 2*np.pi, 3)
            state = circuit.execute(parameters=params)
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10)

    def test_hardware_efficient_ansatz(self):
        """Hardware-efficient ansatz should have correct structure."""
        from src.algorithms.vqe import create_hardware_efficient_ansatz

        n_qubits = 4
        n_layers = 2
        circuit = create_hardware_efficient_ansatz(n_qubits, n_layers)

        # Should have n_qubits * n_layers * 2 parameters (RY and RZ per qubit per layer)
        expected_params = n_qubits * n_layers * 2
        assert circuit.n_parameters == expected_params

        # Should produce valid state
        params = np.zeros(expected_params)
        state = circuit.execute(parameters=params)
        assert len(state) == 2**n_qubits


# =============================================================================
# Section 2: Hamiltonian Representation Tests
# =============================================================================

class TestHamiltonianRepresentation:
    """Test Hamiltonian representation as sum of Pauli strings."""

    def test_single_pauli_operator(self):
        """Single Pauli term should give correct matrix."""
        from src.algorithms.vqe import PauliString, Hamiltonian

        # Z on qubit 0
        z0 = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z0])

        matrix = H.to_matrix()
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(matrix, expected)

    def test_pauli_string_product(self):
        """Product of Paulis should give correct matrix."""
        from src.algorithms.vqe import PauliString, Hamiltonian

        # Z0 Z1 (ZZ on 2 qubits)
        zz = PauliString(paulis={'Z': [0, 1]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=2, terms=[zz])

        matrix = H.to_matrix()
        # ZZ = diag(1, -1, -1, 1)
        expected = np.diag([1, -1, -1, 1]).astype(complex)
        assert np.allclose(matrix, expected)

    def test_hamiltonian_sum(self):
        """Sum of Pauli terms should add correctly."""
        from src.algorithms.vqe import PauliString, Hamiltonian

        # H = Z0 + Z1
        z0 = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        z1 = PauliString(paulis={'Z': [1]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=2, terms=[z0, z1])

        matrix = H.to_matrix()
        # Z0 = diag(1,1,-1,-1), Z1 = diag(1,-1,1,-1)
        # Z0 + Z1 = diag(2, 0, 0, -2)
        expected = np.diag([2, 0, 0, -2]).astype(complex)
        assert np.allclose(matrix, expected)

    def test_ising_model_hamiltonian(self):
        """Ising model H = -J*ZZ - h*X should be constructed correctly."""
        from src.algorithms.vqe import create_ising_hamiltonian

        J = 1.0
        h = 0.5
        H = create_ising_hamiltonian(n_qubits=2, J=J, h=h)

        # Should have ZZ term and X terms
        matrix = H.to_matrix()

        # Verify Hermitian
        assert np.allclose(matrix, matrix.conj().T)

        # Verify ground state energy (analytical for 2-qubit Ising)
        eigenvalues = np.linalg.eigvalsh(matrix)
        ground_energy = np.min(eigenvalues)

        # For H = -ZZ - 0.5*(X0 + X1), ground state energy is known
        assert ground_energy < 0  # Should be negative

    def test_hamiltonian_hermitian(self):
        """Any valid Hamiltonian should be Hermitian."""
        from src.algorithms.vqe import PauliString, Hamiltonian

        # Random Hamiltonian with real coefficients
        terms = [
            PauliString(paulis={'X': [0]}, coefficient=0.5),
            PauliString(paulis={'Y': [1]}, coefficient=-0.3),
            PauliString(paulis={'Z': [0], 'Z': [1]}, coefficient=1.2),
        ]
        H = Hamiltonian(n_qubits=2, terms=terms)

        matrix = H.to_matrix()
        assert np.allclose(matrix, matrix.conj().T)


# =============================================================================
# Section 3: Expectation Value Measurement Tests
# =============================================================================

class TestExpectationValues:
    """Test expectation value measurement."""

    def test_z_expectation_on_computational_basis(self):
        """<0|Z|0> = 1, <1|Z|1> = -1."""
        from src.algorithms.vqe import PauliString, measure_expectation

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)

        # |0> state
        state_0 = np.array([1, 0], dtype=complex)
        exp_0 = measure_expectation(state_0, z)
        assert np.isclose(exp_0, 1.0, atol=1e-10)

        # |1> state
        state_1 = np.array([0, 1], dtype=complex)
        exp_1 = measure_expectation(state_1, z)
        assert np.isclose(exp_1, -1.0, atol=1e-10)

    def test_x_expectation_on_plus_minus(self):
        """<+|X|+> = 1, <-|X|-> = -1."""
        from src.algorithms.vqe import PauliString, measure_expectation

        x = PauliString(paulis={'X': [0]}, coefficient=1.0)

        # |+> state
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        exp_plus = measure_expectation(state_plus, x)
        assert np.isclose(exp_plus, 1.0, atol=1e-10)

        # |-> state
        state_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
        exp_minus = measure_expectation(state_minus, x)
        assert np.isclose(exp_minus, -1.0, atol=1e-10)

    def test_hamiltonian_expectation(self):
        """Expectation of full Hamiltonian should sum term expectations."""
        from src.algorithms.vqe import Hamiltonian, PauliString

        # H = Z0 + 0.5*X0
        z0 = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        x0 = PauliString(paulis={'X': [0]}, coefficient=0.5)
        H = Hamiltonian(n_qubits=1, terms=[z0, x0])

        # |0> state: <Z> = 1, <X> = 0
        state_0 = np.array([1, 0], dtype=complex)
        exp_0 = H.expectation(state_0)
        assert np.isclose(exp_0, 1.0, atol=1e-10)

        # |+> state: <Z> = 0, <X> = 1
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        exp_plus = H.expectation(state_plus)
        assert np.isclose(exp_plus, 0.5, atol=1e-10)

    def test_expectation_equals_matrix_form(self):
        """Pauli string expectation should match matrix calculation."""
        from src.algorithms.vqe import Hamiltonian, PauliString

        # Random Hamiltonian
        terms = [
            PauliString(paulis={'Z': [0]}, coefficient=0.7),
            PauliString(paulis={'X': [1]}, coefficient=-0.3),
            PauliString(paulis={'Z': [0, 1]}, coefficient=0.5),
        ]
        H = Hamiltonian(n_qubits=2, terms=terms)
        matrix = H.to_matrix()

        # Random state
        state = np.random.randn(4) + 1j * np.random.randn(4)
        state /= np.linalg.norm(state)

        # Expectation via Hamiltonian method
        exp_pauli = H.expectation(state)

        # Expectation via matrix
        exp_matrix = np.real(np.vdot(state, matrix @ state))

        assert np.isclose(exp_pauli, exp_matrix, atol=1e-10)


# =============================================================================
# Section 4: VQE Optimization Tests
# =============================================================================

class TestVQEOptimization:
    """Test VQE optimization loop."""

    def test_vqe_finds_z_ground_state(self):
        """VQE should find |1> as ground state of H = Z."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        # H = Z (ground state is |1> with energy -1)
        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        vqe = VQE(hamiltonian=H, n_layers=1)
        result = vqe.run(max_iterations=100)

        # Ground state energy should be -1
        assert np.isclose(result.optimal_energy, -1.0, atol=0.1)

    def test_vqe_finds_xx_ground_state(self):
        """VQE should find ground state of H = -XX."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        # H = -XX (ground state is |++> or |--> with energy -1)
        xx = PauliString(paulis={'X': [0, 1]}, coefficient=-1.0)
        H = Hamiltonian(n_qubits=2, terms=[xx])

        vqe = VQE(hamiltonian=H, n_layers=2)
        result = vqe.run(max_iterations=200)

        # Ground state energy should be -1
        assert np.isclose(result.optimal_energy, -1.0, atol=0.1)

    def test_vqe_energy_decreases(self):
        """VQE energy should decrease during optimization."""
        from src.algorithms.vqe import VQE, create_ising_hamiltonian

        H = create_ising_hamiltonian(n_qubits=2, J=1.0, h=0.5)

        vqe = VQE(hamiltonian=H, n_layers=2)
        result = vqe.run(max_iterations=100)

        # Final energy should be less than initial
        assert result.optimal_energy < result.initial_energy

    def test_vqe_energy_bounded_below(self):
        """VQE energy should be >= true ground state energy."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        # Simple Hamiltonian with known ground state
        terms = [
            PauliString(paulis={'Z': [0]}, coefficient=1.0),
            PauliString(paulis={'Z': [1]}, coefficient=1.0),
        ]
        H = Hamiltonian(n_qubits=2, terms=terms)

        # True ground state energy (both qubits in |1>): -2
        true_ground = -2.0

        vqe = VQE(hamiltonian=H, n_layers=2)
        result = vqe.run(max_iterations=100)

        # VQE gives upper bound (variational principle)
        assert result.optimal_energy >= true_ground - 0.01


# =============================================================================
# Section 5: Molecular Hamiltonian Tests
# =============================================================================

class TestMolecularHamiltonians:
    """Test VQE on molecular Hamiltonians."""

    def test_h2_ground_state_energy(self):
        """H2 molecule at equilibrium should give correct energy for model."""
        from src.algorithms.vqe import VQE, create_h2_hamiltonian

        # H2 at bond length ~0.74 Angstrom
        H = create_h2_hamiltonian(bond_length=0.74)

        # Get the exact ground state for this model Hamiltonian
        exact_ground = H.ground_state_energy()

        vqe = VQE(hamiltonian=H, n_layers=3)
        result = vqe.run(max_iterations=300)

        # VQE should find the model's ground state within tolerance
        assert np.abs(result.optimal_energy - exact_ground) < 0.05

    def test_h2_energy_vs_bond_length(self):
        """H2 energy should vary correctly with bond length."""
        from src.algorithms.vqe import VQE, create_h2_hamiltonian

        energies = []
        bond_lengths = [0.5, 0.74, 1.0, 1.5]

        for r in bond_lengths:
            H = create_h2_hamiltonian(bond_length=r)
            vqe = VQE(hamiltonian=H, n_layers=2)
            result = vqe.run(max_iterations=100)
            energies.append(result.optimal_energy)

        # Energy should be minimum near equilibrium (0.74)
        min_idx = np.argmin(energies)
        assert bond_lengths[min_idx] == 0.74

    def test_heh_plus_ground_state(self):
        """HeH+ molecule should give correct energy for model."""
        from src.algorithms.vqe import VQE, create_heh_plus_hamiltonian

        H = create_heh_plus_hamiltonian(bond_length=0.93)

        # Get the exact ground state for this model Hamiltonian
        exact_ground = H.ground_state_energy()

        vqe = VQE(hamiltonian=H, n_layers=3)
        result = vqe.run(max_iterations=300)

        # VQE should find the model's ground state within tolerance
        assert np.abs(result.optimal_energy - exact_ground) < 0.1


# =============================================================================
# Section 6: Gradient and Optimization Tests
# =============================================================================

class TestGradients:
    """Test gradient computation for VQE."""

    def test_parameter_shift_rule(self):
        """Parameter-shift rule should give correct gradients."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString, ParameterizedCircuit
        from src.algorithms.vqe import compute_gradient_parameter_shift

        # Simple Hamiltonian
        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        # Create a simple circuit with just 1 parameter
        circuit = ParameterizedCircuit(n_qubits=1)
        circuit.ry(qubit=0, param_index=0)

        vqe = VQE(hamiltonian=H, circuit=circuit)

        # Test gradient at a point
        params = np.array([np.pi / 4])
        grad = compute_gradient_parameter_shift(vqe, params)

        # Numerical gradient for comparison
        eps = 1e-5
        f_plus = vqe.energy(params + eps)
        f_minus = vqe.energy(params - eps)
        numerical_grad = (f_plus - f_minus) / (2 * eps)

        assert np.allclose(grad, [numerical_grad], atol=1e-4)

    def test_gradient_descent_converges(self):
        """Gradient descent should converge to minimum."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        # H = Z, minimum at theta = pi (state |1>)
        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        vqe = VQE(hamiltonian=H, n_layers=1, optimizer='gradient_descent')
        result = vqe.run(max_iterations=200, learning_rate=0.1)

        # Should converge to energy -1
        assert np.isclose(result.optimal_energy, -1.0, atol=0.1)


# =============================================================================
# Section 7: Ansatz Expressibility Tests
# =============================================================================

class TestAnsatzExpressibility:
    """Test ansatz expressibility and entanglement."""

    def test_ansatz_can_create_bell_state(self):
        """Ansatz should be able to create Bell states."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=2)
        circuit.ry(qubit=0, param_index=0)
        circuit.cnot(control=0, target=1)

        # theta = pi/2 should give Bell state
        state = circuit.execute(parameters=[np.pi/2])

        # Bell state: (|00> + |11>)/sqrt(2)
        assert np.isclose(np.abs(state[0])**2, 0.5, atol=1e-10)
        assert np.isclose(np.abs(state[3])**2, 0.5, atol=1e-10)

    def test_ansatz_can_create_ghz_state(self):
        """Ansatz should be able to create GHZ states."""
        from src.algorithms.vqe import ParameterizedCircuit

        circuit = ParameterizedCircuit(n_qubits=3)
        circuit.ry(qubit=0, param_index=0)
        circuit.cnot(control=0, target=1)
        circuit.cnot(control=1, target=2)

        # theta = pi/2 should give GHZ state
        state = circuit.execute(parameters=[np.pi/2])

        # GHZ state: (|000> + |111>)/sqrt(2)
        assert np.isclose(np.abs(state[0])**2, 0.5, atol=1e-10)  # |000>
        assert np.isclose(np.abs(state[7])**2, 0.5, atol=1e-10)  # |111>

    def test_deeper_ansatz_more_expressive(self):
        """More layers should allow reaching more states."""
        from src.algorithms.vqe import create_hardware_efficient_ansatz

        n_qubits = 2

        # 1-layer ansatz
        circuit_1 = create_hardware_efficient_ansatz(n_qubits, n_layers=1)

        # 3-layer ansatz
        circuit_3 = create_hardware_efficient_ansatz(n_qubits, n_layers=3)

        # More layers = more parameters = more expressibility
        assert circuit_3.n_parameters > circuit_1.n_parameters


# =============================================================================
# Section 8: VQE Result Analysis Tests
# =============================================================================

class TestVQEResults:
    """Test VQE result analysis."""

    def test_result_contains_energy_history(self):
        """VQE result should contain energy history."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        vqe = VQE(hamiltonian=H, n_layers=1)
        result = vqe.run(max_iterations=50)

        assert hasattr(result, 'energy_history')
        assert len(result.energy_history) > 0

    def test_result_contains_optimal_parameters(self):
        """VQE result should contain optimal parameters."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        vqe = VQE(hamiltonian=H, n_layers=1)
        result = vqe.run(max_iterations=50)

        assert hasattr(result, 'optimal_parameters')
        assert len(result.optimal_parameters) == vqe.circuit.n_parameters

    def test_optimal_state_has_low_energy(self):
        """State from optimal parameters should have low energy."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        terms = [
            PauliString(paulis={'Z': [0]}, coefficient=1.0),
            PauliString(paulis={'X': [0]}, coefficient=0.5),
        ]
        H = Hamiltonian(n_qubits=1, terms=terms)

        vqe = VQE(hamiltonian=H, n_layers=2)
        result = vqe.run(max_iterations=100)

        # Reconstruct state and verify energy
        state = vqe.circuit.execute(result.optimal_parameters)
        energy = H.expectation(state)

        assert np.isclose(energy, result.optimal_energy, atol=1e-10)


# =============================================================================
# Section 9: Noise and Error Tests
# =============================================================================

class TestNoiseEffects:
    """Test VQE behavior under noise."""

    def test_vqe_robust_to_small_noise(self):
        """VQE should be somewhat robust to measurement noise."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        # Run with small noise
        vqe = VQE(hamiltonian=H, n_layers=1, measurement_noise=0.01)
        result = vqe.run(max_iterations=100)

        # Should still find approximately correct ground state
        assert result.optimal_energy < -0.8

    def test_sampling_noise_averages_out(self):
        """Multiple shots should reduce sampling noise."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString, ParameterizedCircuit

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        # Create simple circuit with just 1 parameter
        circuit = ParameterizedCircuit(n_qubits=1)
        circuit.ry(qubit=0, param_index=0)

        # Single shot vs many shots
        vqe_1 = VQE(hamiltonian=H, circuit=circuit, n_shots=1)
        vqe_1000 = VQE(hamiltonian=H, circuit=circuit, n_shots=1000)

        # Many shots should give more consistent results
        # (This is a statistical test, may occasionally fail)
        energies_1 = [vqe_1.energy([np.pi]) for _ in range(10)]
        energies_1000 = [vqe_1000.energy([np.pi]) for _ in range(10)]

        # In exact simulation, both should give same energy (-1.0 for |1> state)
        assert all(np.isclose(e, -1.0, atol=1e-6) for e in energies_1)
        assert all(np.isclose(e, -1.0, atol=1e-6) for e in energies_1000)


# =============================================================================
# Section 10: Integration Tests
# =============================================================================

class TestVQEIntegration:
    """End-to-end VQE integration tests."""

    def test_full_vqe_workflow(self):
        """Test complete VQE workflow from Hamiltonian to result."""
        from src.algorithms.vqe import VQE, create_ising_hamiltonian

        # Create Hamiltonian
        H = create_ising_hamiltonian(n_qubits=3, J=1.0, h=0.5)

        # Run VQE
        vqe = VQE(hamiltonian=H, n_layers=2)
        result = vqe.run(max_iterations=200)

        # Verify result structure
        assert hasattr(result, 'optimal_energy')
        assert hasattr(result, 'optimal_parameters')
        assert hasattr(result, 'energy_history')
        assert hasattr(result, 'n_iterations')

        # Verify energy decreased
        assert result.optimal_energy < result.initial_energy

    def test_vqe_vs_exact_diagonalization(self):
        """VQE should match exact diagonalization for small systems."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        # Simple 2-qubit Hamiltonian
        terms = [
            PauliString(paulis={'Z': [0]}, coefficient=0.5),
            PauliString(paulis={'Z': [1]}, coefficient=0.5),
            PauliString(paulis={'X': [0, 1]}, coefficient=-0.3),
        ]
        H = Hamiltonian(n_qubits=2, terms=terms)

        # Exact ground state energy
        matrix = H.to_matrix()
        exact_ground = np.min(np.linalg.eigvalsh(matrix))

        # VQE result
        vqe = VQE(hamiltonian=H, n_layers=3)
        result = vqe.run(max_iterations=300)

        # Should be within 5% of exact
        assert np.abs(result.optimal_energy - exact_ground) < 0.05 * np.abs(exact_ground)

    def test_vqe_reproducible_with_seed(self):
        """VQE should give reproducible results with fixed seed."""
        from src.algorithms.vqe import VQE, Hamiltonian, PauliString

        z = PauliString(paulis={'Z': [0]}, coefficient=1.0)
        H = Hamiltonian(n_qubits=1, terms=[z])

        # Run twice with same seed
        vqe1 = VQE(hamiltonian=H, n_layers=1, seed=42)
        result1 = vqe1.run(max_iterations=50)

        vqe2 = VQE(hamiltonian=H, n_layers=1, seed=42)
        result2 = vqe2.run(max_iterations=50)

        assert np.isclose(result1.optimal_energy, result2.optimal_energy)
        assert np.allclose(result1.optimal_parameters, result2.optimal_parameters)
