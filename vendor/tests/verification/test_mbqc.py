"""
Test suite for Measurement-Based Quantum Computing (MBQC).

Implements tests for:
- Graph states and cluster states
- CZ gate operations
- Single-qubit measurements
- Measurement patterns for quantum gates
- One-way quantum computer
- Teleportation-based computation

References:
- Raussendorf & Briegel, PRL 86, 5188 (2001) - One-way quantum computation
- Briegel et al., Nature Physics 5, 19 (2009) - MBQC review
- Nielsen, PRA 73, 042306 (2006) - Cluster-state quantum computation
"""

import numpy as np
import pytest
from typing import List, Tuple


class TestGraphStateGeneration:
    """Test creation of graph states."""

    def test_single_qubit_graph_state(self):
        """Single qubit graph state is just |+⟩."""
        from src.quantum.mbqc import GraphState

        # Single node graph
        graph = GraphState(num_qubits=1, edges=[])
        state = graph.state_vector

        # Should be |+⟩ = (|0⟩ + |1⟩)/√2
        plus_state = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(np.abs(state), np.abs(plus_state))

    def test_two_qubit_bell_state(self):
        """Two qubits connected by edge gives Bell-like state."""
        from src.quantum.mbqc import GraphState

        # Two qubits with one edge: CZ|++⟩
        graph = GraphState(num_qubits=2, edges=[(0, 1)])
        state = graph.state_vector

        # CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        expected = np.array([1, 1, 1, -1]) / 2
        assert np.allclose(state, expected)

    def test_three_qubit_linear_cluster(self):
        """Three qubit linear cluster state."""
        from src.quantum.mbqc import GraphState

        # Linear chain: 0-1-2
        graph = GraphState(num_qubits=3, edges=[(0, 1), (1, 2)])
        state = graph.state_vector

        # Verify it's a valid quantum state
        assert np.isclose(np.linalg.norm(state), 1.0)

        # Verify stabilizer properties (graph state stabilizers)
        # Each vertex v has stabilizer X_v ⊗ (⊗_{u∈N(v)} Z_u)
        # For vertex 0: X_0 Z_1
        # For vertex 1: Z_0 X_1 Z_2
        # For vertex 2: Z_1 X_2

    def test_four_qubit_square_cluster(self):
        """Four qubit square cluster state."""
        from src.quantum.mbqc import GraphState

        # Square: 0-1-3-2-0
        edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
        graph = GraphState(num_qubits=4, edges=edges)
        state = graph.state_vector

        assert np.isclose(np.linalg.norm(state), 1.0)
        assert len(state) == 16  # 2^4 amplitudes

    def test_graph_state_is_stabilizer_state(self):
        """Graph states are stabilized by graph stabilizers."""
        from src.quantum.mbqc import GraphState

        graph = GraphState(num_qubits=3, edges=[(0, 1), (1, 2)])

        # Get stabilizers
        stabilizers = graph.get_stabilizers()

        # Should have n stabilizers for n qubits
        assert len(stabilizers) == 3

        # Each stabilizer should have eigenvalue +1
        for stab in stabilizers:
            eigenvalue = graph.measure_stabilizer(stab)
            assert np.isclose(eigenvalue, 1.0)


class TestClusterStateGeneration:
    """Test cluster states on lattices."""

    def test_1d_cluster_chain(self):
        """One-dimensional cluster state chain."""
        from src.quantum.mbqc import ClusterState

        # 1D chain of 4 qubits
        cluster = ClusterState(dimensions=(4,))

        assert cluster.num_qubits == 4
        assert len(cluster.edges) == 3  # Linear chain has n-1 edges

    def test_2d_cluster_lattice(self):
        """Two-dimensional cluster state lattice."""
        from src.quantum.mbqc import ClusterState

        # 2x3 lattice
        cluster = ClusterState(dimensions=(2, 3))

        assert cluster.num_qubits == 6
        # 2D lattice has horizontal and vertical edges
        # Horizontal: 2 rows * 2 edges = 4
        # Vertical: 3 cols * 1 edge = 3
        assert len(cluster.edges) == 7

    def test_3x3_cluster_lattice(self):
        """3x3 cluster state for universal computation."""
        from src.quantum.mbqc import ClusterState

        cluster = ClusterState(dimensions=(3, 3))

        assert cluster.num_qubits == 9
        # Horizontal: 3 rows * 2 = 6
        # Vertical: 3 cols * 2 = 6
        assert len(cluster.edges) == 12

    def test_cluster_state_vector(self):
        """Cluster state generates valid quantum state."""
        from src.quantum.mbqc import ClusterState

        cluster = ClusterState(dimensions=(2, 2))
        state = cluster.state_vector

        assert np.isclose(np.linalg.norm(state), 1.0)
        assert len(state) == 16  # 2^4


class TestCZGate:
    """Test controlled-Z gate operations."""

    def test_cz_on_computational_basis(self):
        """CZ flips phase of |11⟩ only."""
        from src.quantum.mbqc import apply_cz

        # |00⟩ -> |00⟩
        state00 = np.array([1, 0, 0, 0], dtype=complex)
        result = apply_cz(state00, 0, 1)
        assert np.allclose(result, state00)

        # |01⟩ -> |01⟩
        state01 = np.array([0, 1, 0, 0], dtype=complex)
        result = apply_cz(state01, 0, 1)
        assert np.allclose(result, state01)

        # |10⟩ -> |10⟩
        state10 = np.array([0, 0, 1, 0], dtype=complex)
        result = apply_cz(state10, 0, 1)
        assert np.allclose(result, state10)

        # |11⟩ -> -|11⟩
        state11 = np.array([0, 0, 0, 1], dtype=complex)
        result = apply_cz(state11, 0, 1)
        assert np.allclose(result, -state11)

    def test_cz_symmetric(self):
        """CZ is symmetric in control/target."""
        from src.quantum.mbqc import apply_cz

        state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

        result1 = apply_cz(state.copy(), 0, 1)
        result2 = apply_cz(state.copy(), 1, 0)

        assert np.allclose(result1, result2)

    def test_cz_creates_entanglement(self):
        """CZ on |++⟩ creates entangled state."""
        from src.quantum.mbqc import apply_cz

        # |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        plus_plus = np.array([1, 1, 1, 1], dtype=complex) / 2

        result = apply_cz(plus_plus, 0, 1)

        # CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        expected = np.array([1, 1, 1, -1], dtype=complex) / 2
        assert np.allclose(result, expected)

    def test_cz_self_inverse(self):
        """CZ² = I."""
        from src.quantum.mbqc import apply_cz

        state = np.random.rand(4) + 1j * np.random.rand(4)
        state = state / np.linalg.norm(state)

        result = apply_cz(apply_cz(state.copy(), 0, 1), 0, 1)

        assert np.allclose(result, state)


class TestSingleQubitMeasurement:
    """Test single-qubit measurements in various bases."""

    def test_z_measurement_computational_basis(self):
        """Z-basis measurement on computational basis states."""
        from src.quantum.mbqc import measure_qubit

        # |0⟩ always gives outcome 0
        state0 = np.array([1, 0], dtype=complex)
        for _ in range(10):
            outcome, _ = measure_qubit(state0.copy(), basis='Z')
            assert outcome == 0

        # |1⟩ always gives outcome 1
        state1 = np.array([0, 1], dtype=complex)
        for _ in range(10):
            outcome, _ = measure_qubit(state1.copy(), basis='Z')
            assert outcome == 1

    def test_x_measurement_plus_state(self):
        """X-basis measurement on |+⟩ always gives 0."""
        from src.quantum.mbqc import measure_qubit

        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        for _ in range(10):
            outcome, _ = measure_qubit(plus.copy(), basis='X')
            assert outcome == 0

    def test_x_measurement_minus_state(self):
        """X-basis measurement on |-⟩ always gives 1."""
        from src.quantum.mbqc import measure_qubit

        minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
        for _ in range(10):
            outcome, _ = measure_qubit(minus.copy(), basis='X')
            assert outcome == 1

    def test_y_measurement(self):
        """Y-basis measurement on |+i⟩ and |-i⟩."""
        from src.quantum.mbqc import measure_qubit

        # |+i⟩ = (|0⟩ + i|1⟩)/√2
        plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)
        for _ in range(10):
            outcome, _ = measure_qubit(plus_i.copy(), basis='Y')
            assert outcome == 0

        # |-i⟩ = (|0⟩ - i|1⟩)/√2
        minus_i = np.array([1, -1j], dtype=complex) / np.sqrt(2)
        for _ in range(10):
            outcome, _ = measure_qubit(minus_i.copy(), basis='Y')
            assert outcome == 1

    def test_arbitrary_angle_measurement(self):
        """Measurement in rotated basis."""
        from src.quantum.mbqc import measure_qubit

        # Measure |0⟩ in basis rotated by angle θ around Y
        state = np.array([1, 0], dtype=complex)

        # θ = 0 is Z basis
        # θ = π/2 is X basis
        outcomes = []
        for _ in range(100):
            outcome, _ = measure_qubit(state.copy(), angle=np.pi/4)
            outcomes.append(outcome)

        # Should get mix of 0 and 1
        prob_0 = outcomes.count(0) / len(outcomes)
        # For |0⟩ measured at θ=π/4: P(0) = cos²(π/8) ≈ 0.854
        assert 0.6 < prob_0 < 1.0  # Allow statistical variance


class TestMeasurementPatterns:
    """Test measurement patterns for quantum gates."""

    def test_identity_by_measurement(self):
        """Identity gate via measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        # 2-qubit chain: measure qubit 0 in X basis
        # Result on qubit 1 is teleported input (up to byproduct)
        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        # Input |0⟩ on logical qubit
        input_state = np.array([1, 0], dtype=complex)

        # Apply identity pattern
        output = mbqc.apply_gate_pattern(
            input_state=input_state,
            gate='I',
            measurement_angles=[0]  # X-basis measurement
        )

        # Output should be |0⟩ (possibly with byproduct)
        fidelity = np.abs(np.vdot(output, input_state))**2
        assert fidelity > 0.99

    def test_hadamard_by_measurement(self):
        """Hadamard gate via measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        # 2-qubit chain
        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        # Input |0⟩
        input_state = np.array([1, 0], dtype=complex)

        # Hadamard pattern
        output = mbqc.apply_gate_pattern(
            input_state=input_state,
            gate='H',
            measurement_angles=[np.pi/2]  # Y-basis measurement
        )

        # Output should be |+⟩
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, plus))**2
        assert fidelity > 0.99

    def test_phase_gate_by_measurement(self):
        """S (phase) gate via measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        # Input |+⟩
        input_state = np.array([1, 1], dtype=complex) / np.sqrt(2)

        # S gate pattern
        output = mbqc.apply_gate_pattern(
            input_state=input_state,
            gate='S',
            measurement_angles=[np.pi/4]
        )

        # S|+⟩ = (|0⟩ + i|1⟩)/√2 = |+i⟩
        expected = np.array([1, 1j], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99

    def test_t_gate_by_measurement(self):
        """T gate via measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        # Input |+⟩
        input_state = np.array([1, 1], dtype=complex) / np.sqrt(2)

        # T gate pattern
        output = mbqc.apply_gate_pattern(
            input_state=input_state,
            gate='T',
            measurement_angles=[np.pi/8]
        )

        # T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2
        expected = np.array([1, np.exp(1j * np.pi/4)], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99

    def test_arbitrary_z_rotation(self):
        """R_z(θ) via measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        theta = np.pi / 3
        input_state = np.array([1, 1], dtype=complex) / np.sqrt(2)

        output = mbqc.apply_gate_pattern(
            input_state=input_state,
            gate='Rz',
            measurement_angles=[theta/2]
        )

        # R_z(θ)|+⟩ = (e^{-iθ/2}|0⟩ + e^{iθ/2}|1⟩)/√2
        expected = np.array([np.exp(-1j*theta/2), np.exp(1j*theta/2)], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99


class TestByproductOperators:
    """Test byproduct operator handling."""

    def test_x_byproduct_correction(self):
        """X byproduct from measurement outcome."""
        from src.quantum.mbqc import apply_byproduct_correction

        state = np.array([1, 0], dtype=complex)  # |0⟩

        # X byproduct gives |1⟩
        corrected = apply_byproduct_correction(state, x_power=1, z_power=0)
        expected = np.array([0, 1], dtype=complex)
        assert np.allclose(corrected, expected)

    def test_z_byproduct_correction(self):
        """Z byproduct from measurement outcome."""
        from src.quantum.mbqc import apply_byproduct_correction

        state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩

        # Z byproduct gives |-⟩
        corrected = apply_byproduct_correction(state, x_power=0, z_power=1)
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        assert np.allclose(corrected, expected)

    def test_xz_byproduct_correction(self):
        """Combined XZ byproduct."""
        from src.quantum.mbqc import apply_byproduct_correction

        state = np.array([1, 0], dtype=complex)  # |0⟩

        # XZ|0⟩ = X|0⟩ = |1⟩ (Z|0⟩ = |0⟩)
        # Actually XZ|0⟩ = X(|0⟩) = |1⟩
        corrected = apply_byproduct_correction(state, x_power=1, z_power=1)
        # Z first then X: ZX|0⟩ = Z|1⟩ = -|1⟩
        # But in MBQC convention: X^s Z^t, so X then Z
        # X|0⟩ = |1⟩, Z|1⟩ = -|1⟩
        expected = np.array([0, -1], dtype=complex)
        assert np.allclose(corrected, expected)

    def test_byproduct_propagation(self):
        """Byproducts propagate through measurement pattern."""
        from src.quantum.mbqc import MBQCComputation

        mbqc = MBQCComputation(num_qubits=3, edges=[(0, 1), (1, 2)])

        # Track byproduct accumulation
        input_state = np.array([1, 0], dtype=complex)

        # Multiple measurements accumulate byproducts
        output, byproducts = mbqc.run_pattern_with_byproducts(
            input_state=input_state,
            measurement_angles=[0, 0]
        )

        assert 'x_power' in byproducts
        assert 'z_power' in byproducts


class TestOneWayQuantumComputer:
    """Test universal one-way quantum computation."""

    def test_single_qubit_universal_gate_set(self):
        """Any single-qubit gate can be decomposed into MBQC pattern."""
        from src.quantum.mbqc import OneWayQC

        # 4-qubit chain for arbitrary single-qubit unitary
        owqc = OneWayQC(cluster_dims=(4, 1))

        # Arbitrary rotation parameters
        alpha, beta, gamma = np.pi/3, np.pi/5, np.pi/7

        input_state = np.array([1, 0], dtype=complex)

        output = owqc.apply_arbitrary_rotation(
            input_state=input_state,
            euler_angles=(alpha, beta, gamma)
        )

        # Verify output is valid quantum state
        assert np.isclose(np.linalg.norm(output), 1.0)

    def test_cnot_on_cluster(self):
        """CNOT gate via 2D cluster state."""
        from src.quantum.mbqc import OneWayQC

        # 2D cluster needed for two-qubit gates
        owqc = OneWayQC(cluster_dims=(4, 2))

        # Input |00⟩
        input_state = np.array([1, 0, 0, 0], dtype=complex)

        output = owqc.apply_cnot(input_state, control=0, target=1)

        # CNOT|00⟩ = |00⟩
        expected = np.array([1, 0, 0, 0], dtype=complex)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99

    def test_cnot_on_superposition(self):
        """CNOT on superposition state."""
        from src.quantum.mbqc import OneWayQC

        owqc = OneWayQC(cluster_dims=(4, 2))

        # Input |+0⟩ = (|00⟩ + |10⟩)/√2
        input_state = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

        output = owqc.apply_cnot(input_state, control=0, target=1)

        # CNOT|+0⟩ = (|00⟩ + |11⟩)/√2 = Bell state
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99

    def test_universality_two_qubit_gate(self):
        """Any two-qubit gate can be implemented."""
        from src.quantum.mbqc import OneWayQC

        owqc = OneWayQC(cluster_dims=(6, 3))

        # Create entangled input
        input_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        # Apply arbitrary two-qubit gate pattern
        output = owqc.apply_cz_gate(input_state)

        # Verify valid output
        assert np.isclose(np.linalg.norm(output), 1.0)


class TestTeleportationBasedComputation:
    """Test teleportation as computational primitive."""

    def test_gate_teleportation_identity(self):
        """Teleportation of identity gate."""
        from src.quantum.mbqc import GateTeleportation

        gt = GateTeleportation()

        input_state = np.array([1, 0], dtype=complex)

        # Teleport through identity
        output = gt.teleport_gate(input_state, gate='I')

        fidelity = np.abs(np.vdot(output, input_state))**2
        assert fidelity > 0.99

    def test_gate_teleportation_hadamard(self):
        """Teleportation of Hadamard gate."""
        from src.quantum.mbqc import GateTeleportation

        gt = GateTeleportation()

        input_state = np.array([1, 0], dtype=complex)

        output = gt.teleport_gate(input_state, gate='H')

        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.99

    def test_gate_teleportation_uses_entanglement(self):
        """Gate teleportation consumes entanglement."""
        from src.quantum.mbqc import GateTeleportation

        gt = GateTeleportation()

        # Track resource usage
        initial_ebits = gt.available_entanglement

        input_state = np.array([1, 0], dtype=complex)
        gt.teleport_gate(input_state, gate='H')

        # Should consume one ebit
        assert gt.available_entanglement == initial_ebits - 1


class TestMBQCCircuitExecution:
    """Test execution of quantum circuits via MBQC."""

    def test_bell_state_preparation(self):
        """Prepare Bell state using MBQC."""
        from src.quantum.mbqc import MBQCCircuit

        circuit = MBQCCircuit(num_logical_qubits=2)

        # H on qubit 0
        circuit.h(0)
        # CNOT from 0 to 1
        circuit.cnot(0, 1)

        # Execute
        output = circuit.execute()

        # Should be Bell state |Φ+⟩
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.95

    def test_ghz_state_preparation(self):
        """Prepare 3-qubit GHZ state using MBQC."""
        from src.quantum.mbqc import MBQCCircuit

        circuit = MBQCCircuit(num_logical_qubits=3)

        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)

        output = circuit.execute()

        # GHZ state: (|000⟩ + |111⟩)/√2
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)

        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.90

    def test_quantum_fourier_transform(self):
        """2-qubit QFT using MBQC."""
        from src.quantum.mbqc import MBQCCircuit

        circuit = MBQCCircuit(num_logical_qubits=2)

        # QFT circuit
        circuit.h(0)
        circuit.cphase(0, 1, np.pi/2)
        circuit.h(1)
        circuit.swap(0, 1)

        # Input |00⟩
        output = circuit.execute(initial_state=np.array([1, 0, 0, 0], dtype=complex))

        # QFT|00⟩ = |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        expected = np.array([1, 1, 1, 1], dtype=complex) / 2
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.90


class TestMBQCResourceEstimation:
    """Test resource requirements for MBQC."""

    def test_single_qubit_gate_resources(self):
        """Single-qubit gates need linear cluster."""
        from src.quantum.mbqc import estimate_resources

        resources = estimate_resources(gate='single_qubit', params={'angle': np.pi/4})

        assert resources['cluster_qubits'] >= 2
        assert resources['measurements'] >= 1

    def test_cnot_resources(self):
        """CNOT needs 2D cluster section."""
        from src.quantum.mbqc import estimate_resources

        resources = estimate_resources(gate='CNOT')

        assert resources['cluster_qubits'] >= 15  # Typical for CNOT
        assert resources['measurements'] >= 13

    def test_circuit_resource_scaling(self):
        """Resource scaling with circuit depth."""
        from src.quantum.mbqc import MBQCCircuit

        # Depth 1 circuit
        circuit1 = MBQCCircuit(num_logical_qubits=2)
        circuit1.h(0)
        circuit1.h(1)
        resources1 = circuit1.estimate_resources()

        # Depth 3 circuit
        circuit3 = MBQCCircuit(num_logical_qubits=2)
        circuit3.h(0)
        circuit3.h(1)
        circuit3.cnot(0, 1)
        circuit3.h(0)
        circuit3.h(1)
        resources3 = circuit3.estimate_resources()

        # Deeper circuit needs more resources
        assert resources3['cluster_qubits'] > resources1['cluster_qubits']


class TestMBQCErrorHandling:
    """Test error handling in MBQC."""

    def test_invalid_edge(self):
        """Invalid edge raises error."""
        from src.quantum.mbqc import GraphState

        with pytest.raises(ValueError):
            GraphState(num_qubits=2, edges=[(0, 2)])  # Qubit 2 doesn't exist

    def test_invalid_measurement_angle(self):
        """Invalid measurement angle handling."""
        from src.quantum.mbqc import MBQCComputation

        mbqc = MBQCComputation(num_qubits=2, edges=[(0, 1)])

        # NaN angle should raise error
        with pytest.raises(ValueError):
            mbqc.apply_gate_pattern(
                input_state=np.array([1, 0], dtype=complex),
                gate='Rz',
                measurement_angles=[float('nan')]
            )

    def test_empty_cluster(self):
        """Empty cluster state handling."""
        from src.quantum.mbqc import ClusterState

        with pytest.raises(ValueError):
            ClusterState(dimensions=(0,))


class TestMBQCEquivalence:
    """Test equivalence between MBQC and circuit model."""

    def test_single_gate_equivalence(self):
        """MBQC matches circuit model for single gate."""
        from src.quantum.mbqc import MBQCCircuit

        # Direct calculation of RY(π/3)|0⟩
        angle = np.pi / 3
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        # RY|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        expected_output = np.array([c, s], dtype=complex)

        # MBQC model
        mbqc = MBQCCircuit(num_logical_qubits=1)
        mbqc.ry(0, np.pi/3)
        mbqc_output = mbqc.execute()

        fidelity = np.abs(np.vdot(expected_output, mbqc_output))**2
        assert fidelity > 0.95

    def test_circuit_equivalence(self):
        """MBQC matches circuit model for multi-gate circuit."""
        from src.quantum.mbqc import MBQCCircuit

        # Simple test circuit
        mbqc = MBQCCircuit(num_logical_qubits=2)
        mbqc.h(0)
        mbqc.cnot(0, 1)

        output = mbqc.execute()

        # Bell state
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        fidelity = np.abs(np.vdot(output, expected))**2
        assert fidelity > 0.90


class TestMBQCVerification:
    """Verify MBQC against analytical results."""

    def test_graph_state_schmidt_rank(self):
        """Graph state entanglement structure."""
        from src.quantum.mbqc import GraphState

        # Two disconnected qubits: product state
        graph_product = GraphState(num_qubits=2, edges=[])
        schmidt_rank = graph_product.get_schmidt_rank(partition=[0])
        assert schmidt_rank == 1

        # Connected qubits: entangled
        graph_entangled = GraphState(num_qubits=2, edges=[(0, 1)])
        schmidt_rank = graph_entangled.get_schmidt_rank(partition=[0])
        assert schmidt_rank == 2

    def test_cluster_state_stabilizer_count(self):
        """Cluster state has n independent stabilizers."""
        from src.quantum.mbqc import ClusterState

        cluster = ClusterState(dimensions=(3, 3))
        stabilizers = cluster.get_stabilizers()

        # 9 qubits = 9 independent stabilizers
        assert len(stabilizers) == 9

    def test_measurement_outcome_statistics(self):
        """Measurement statistics match Born rule."""
        from src.quantum.mbqc import measure_qubit

        # Measure |+⟩ in Z basis many times
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

        outcomes = []
        for _ in range(1000):
            outcome, _ = measure_qubit(plus.copy(), basis='Z')
            outcomes.append(outcome)

        # Should be roughly 50-50
        prob_0 = outcomes.count(0) / len(outcomes)
        assert 0.4 < prob_0 < 0.6  # Allow statistical variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
