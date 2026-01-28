"""
TensorFlow Quantum implementation for wormhole quantum circuits.

This module provides a TensorFlow Quantum backend for quantum wormhole simulations,
enabling hybrid quantum-classical machine learning workflows and advanced
optimization of wormhole parameters.
"""

import numpy as np
import tensorflow as tf
import cirq
import sympy
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import warnings

try:
    import tensorflow_quantum as tfq
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False
    warnings.warn("TensorFlow Quantum not available. Falling back to classical simulation.")

from src.physics.constants import HBAR, C, PLANCK_LENGTH
from src.physics.spacetime_metrics import SpacetimeMetric


class TFQWormholeCircuit:
    """TensorFlow Quantum implementation of wormhole quantum circuits."""
    
    def __init__(self, num_qubits: int, geometry_params: Dict, use_gpu: bool = True):
        """Initialize TensorFlow Quantum wormhole circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            geometry_params: Wormhole geometry parameters
            use_gpu: Whether to use GPU acceleration if available
        """
        
        if not TFQ_AVAILABLE:
            raise ImportError("TensorFlow Quantum not available. Please install: pip install tensorflow-quantum")
        
        self.num_qubits = num_qubits
        self.geometry = geometry_params
        self.use_gpu = use_gpu and tf.config.list_physical_devices('GPU')
        
        # Create qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
        
        # Initialize circuit components
        self._setup_wormhole_circuit()
        self._setup_measurement_operators()
        self._setup_hamiltonian()
        
        # TensorFlow setup
        if self.use_gpu:
            self.device = '/GPU:0'
        else:
            self.device = '/CPU:0'
    
    def _setup_wormhole_circuit(self):
        """Set up the quantum circuit for wormhole simulation."""
        
        # Main traversal circuit
        self.traversal_circuit = cirq.Circuit()
        
        # Create Bell state for entanglement (EPR pair across wormhole mouths)
        if self.num_qubits >= 2:
            self.traversal_circuit.append([
                cirq.H(self.qubits[0]),  # Superposition
                cirq.CNOT(self.qubits[0], self.qubits[1])  # Entanglement
            ])
        
        # Add rotation gates parameterized by wormhole geometry
        throat_radius = self.geometry.get('throat_radius', 1e3)
        traversal_prob = self.geometry.get('traversal_probability', 0.8)
        
        # Rotation angle based on wormhole geometry
        # Larger throat radius -> higher traversal probability -> smaller rotation
        self.theta = sympy.Symbol('theta')
        rotation_angle = np.arccos(np.sqrt(traversal_prob)) * (PLANCK_LENGTH / throat_radius)
        
        # Add parameterized rotations for quantum interference effects
        for i in range(min(self.num_qubits, 4)):  # Limit to avoid exponential scaling
            self.traversal_circuit.append(cirq.ry(self.theta)(self.qubits[i]))
        
        # Add controlled phase gates for exotic matter interactions
        if self.num_qubits >= 4:
            self.traversal_circuit.append([
                cirq.CZ(self.qubits[i], self.qubits[i+1]) 
                for i in range(0, min(self.num_qubits-1, 3), 2)
            ])
        
        # Store the parameter value
        self.theta_value = rotation_angle
    
    def _setup_measurement_operators(self):
        """Set up measurement operators for observables."""
        
        # Pauli operators for measurements
        self.pauli_x = [cirq.X(q) for q in self.qubits]
        self.pauli_y = [cirq.Y(q) for q in self.qubits]
        self.pauli_z = [cirq.Z(q) for q in self.qubits]
        
        # Observables we want to measure
        self.observables = []
        
        # Single-qubit Z measurements (computational basis)
        for i, qubit in enumerate(self.qubits):
            self.observables.append(cirq.Z(qubit))
        
        # Two-qubit correlations (for entanglement)
        if self.num_qubits >= 2:
            for i in range(min(self.num_qubits-1, 3)):
                self.observables.append(cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i+1]))
        
        # Convert to TensorFlow Quantum format
        self.tfq_observables = tfq.convert_to_tensor([self.observables])
    
    def _setup_hamiltonian(self):
        """Set up Hamiltonian for time evolution."""
        
        # Wormhole Hamiltonian terms
        hamiltonian_ops = []
        
        # Kinetic term (hopping between mouths)
        if self.num_qubits >= 2:
            hamiltonian_ops.extend([
                cirq.X(self.qubits[0]) * cirq.X(self.qubits[1]),
                cirq.Y(self.qubits[0]) * cirq.Y(self.qubits[1])
            ])
        
        # Interaction with exotic matter (Z-Z coupling)
        exotic_coupling = -0.1  # Negative for attractive interaction
        for i in range(min(self.num_qubits-1, 3)):
            hamiltonian_ops.append(
                exotic_coupling * cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i+1])
            )
        
        # Single-qubit terms (local fields)
        local_field = 0.05
        for qubit in self.qubits:
            hamiltonian_ops.append(local_field * cirq.Z(qubit))
        
        self.hamiltonian = sum(hamiltonian_ops)
    
    def create_traversal_state(self) -> tf.Tensor:
        """Create quantum state representing particle traversal through wormhole.
        
        Returns:
            TensorFlow tensor representing the quantum state
        """
        
        # Convert circuit to tensor with parameter values
        circuit_tensor = tfq.convert_to_tensor([self.traversal_circuit])
        
        # Set parameter values
        param_values = tf.constant([[self.theta_value]], dtype=tf.float32)
        param_symbols = [self.theta]
        
        # Simulate the circuit state
        with tf.device(self.device):
            # Use TFQ simulator to get state vector
            simulator = tfq.layers.State()
            state_vector = simulator(circuit_tensor, symbol_names=param_symbols, 
                                   symbol_values=param_values)
        
        return state_vector
    
    def measure_observables(self, num_shots: int = 1000) -> Dict[str, float]:
        """Measure quantum observables from the circuit.
        
        Args:
            num_shots: Number of measurement shots
            
        Returns:
            Dictionary of observable measurements
        """
        
        # Convert circuit to tensor
        circuit_tensor = tfq.convert_to_tensor([self.traversal_circuit])
        param_values = tf.constant([[self.theta_value]], dtype=tf.float32)
        param_symbols = [self.theta]
        
        with tf.device(self.device):
            # Expectation values
            expectation_layer = tfq.layers.Expectation()
            expectations = expectation_layer(
                circuit_tensor, 
                operators=self.tfq_observables,
                symbol_names=param_symbols,
                symbol_values=param_values
            )
        
        # Convert to dictionary
        results = {}
        for i, obs in enumerate(self.observables):
            if len(self.observables) > i:
                results[f"observable_{i}"] = float(expectations[0, i])
        
        return results
    
    def compute_entanglement_entropy(self) -> float:
        """Compute entanglement entropy of the quantum state.
        
        Returns:
            Von Neumann entropy
        """
        
        if self.num_qubits < 2:
            return 0.0
        
        # Get state vector
        state = self.create_traversal_state()
        
        # Compute reduced density matrix for first half of qubits
        # This is a simplified calculation - full implementation would require
        # partial trace operations that are complex in TensorFlow
        
        # For now, estimate from measurement correlations
        measurements = self.measure_observables()
        
        # Simple estimate: higher correlations -> higher entanglement
        correlations = [v for k, v in measurements.items() if "observable" in k and len(k.split('_')) > 2]
        
        if correlations:
            correlation_strength = np.mean(np.abs(correlations))
            # Convert correlation to entropy estimate (rough approximation)
            entropy = -correlation_strength * np.log2(correlation_strength + 1e-10)
        else:
            entropy = 0.0
        
        return float(entropy)
    
    def time_evolve(self, time_steps: int, dt: float) -> List[Dict[str, float]]:
        """Time evolve the quantum state through the wormhole.
        
        Args:
            time_steps: Number of evolution steps
            dt: Time step size
            
        Returns:
            List of measurement results at each time step
        """
        
        results = []
        
        # Create time evolution circuit
        evolution_circuit = cirq.Circuit()
        
        # Add initial state preparation
        evolution_circuit += self.traversal_circuit
        
        for step in range(time_steps):
            # Add time evolution operator
            # Use Trotter approximation for Hamiltonian evolution
            
            # For simplicity, add rotation gates that approximate time evolution
            evolution_angle = dt * 0.1  # Scaled for stability
            
            for i, qubit in enumerate(self.qubits):
                evolution_circuit.append(cirq.rz(evolution_angle)(qubit))
            
            # Add entangling gates for interaction terms
            if self.num_qubits >= 2:
                for i in range(min(self.num_qubits-1, 3)):
                    evolution_circuit.append(cirq.CZ(self.qubits[i], self.qubits[i+1]))
            
            # Measure at this time step
            temp_circuit = evolution_circuit.copy()
            
            # Convert and simulate
            circuit_tensor = tfq.convert_to_tensor([temp_circuit])
            
            with tf.device(self.device):
                # Sample from circuit
                sampler = tfq.layers.Sample()
                samples = sampler(circuit_tensor, repetitions=100)
            
            # Convert samples to measurement statistics
            measurements = {}
            for i in range(self.num_qubits):
                bit_samples = samples[0, :, i]
                measurements[f"qubit_{i}"] = float(tf.reduce_mean(tf.cast(bit_samples, tf.float32)))
            
            measurements["time"] = step * dt
            measurements["entanglement_entropy"] = self.compute_entanglement_entropy()
            
            results.append(measurements)
        
        return results
    
    def construct_hamiltonian(self) -> Any:
        """Return the Hamiltonian for compatibility with existing code."""
        return self.hamiltonian
    
    def get_circuit_depth(self) -> int:
        """Get the depth of the quantum circuit."""
        return len(self.traversal_circuit)
    
    def get_gate_count(self) -> int:
        """Get the total number of gates in the circuit."""
        return sum(1 for _ in self.traversal_circuit.all_operations())
    
    def optimize_parameters(self, target_state: Optional[np.ndarray] = None, 
                          learning_rate: float = 0.01, steps: int = 100) -> Dict[str, float]:
        """Optimize circuit parameters using TensorFlow optimization.
        
        Args:
            target_state: Target quantum state (if None, optimize for max entanglement)
            learning_rate: Learning rate for optimization
            steps: Number of optimization steps
            
        Returns:
            Dictionary with optimization results
        """
        
        # Define trainable parameter
        theta_var = tf.Variable(self.theta_value, dtype=tf.float32, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        def loss_function():
            # Update circuit with current parameter value
            circuit_tensor = tfq.convert_to_tensor([self.traversal_circuit])
            param_values = tf.expand_dims(theta_var, 0)
            param_values = tf.expand_dims(param_values, 0)
            
            # Compute expectation values
            expectation_layer = tfq.layers.Expectation()
            expectations = expectation_layer(
                circuit_tensor,
                operators=self.tfq_observables,
                symbol_names=[self.theta],
                symbol_values=param_values
            )
            
            if target_state is not None:
                # Loss based on target state fidelity (simplified)
                loss = tf.reduce_mean(tf.square(expectations - target_state))
            else:
                # Optimize for maximum entanglement (maximize correlation)
                correlations = expectations[0, min(len(self.observables)-1, self.num_qubits):]
                loss = -tf.reduce_mean(tf.abs(correlations))  # Negative for maximization
            
            return loss
        
        # Optimization loop
        loss_history = []
        
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = loss_function()
            
            gradients = tape.gradient(loss_value, [theta_var])
            optimizer.apply_gradients(zip(gradients, [theta_var]))
            
            loss_history.append(float(loss_value))
            
            if step % 20 == 0:
                print(f"Optimization step {step}, Loss: {loss_value:.6f}, θ: {theta_var.numpy():.6f}")
        
        # Update the circuit parameter
        self.theta_value = float(theta_var.numpy())
        
        return {
            "optimized_theta": self.theta_value,
            "final_loss": loss_history[-1],
            "loss_history": loss_history
        }


class HybridQuantumClassicalSystem:
    """Hybrid system combining TensorFlow Quantum circuits with classical ML."""
    
    def __init__(self, num_qubits: int, classical_hidden_dims: List[int] = [64, 32]):
        """Initialize hybrid quantum-classical system.
        
        Args:
            num_qubits: Number of qubits in quantum circuit
            classical_hidden_dims: Hidden layer dimensions for classical NN
        """
        
        self.num_qubits = num_qubits
        self.quantum_circuit = None
        
        # Build classical neural network
        self.classical_model = tf.keras.Sequential([
            tf.keras.layers.Dense(classical_hidden_dims[0], activation='relu', 
                                input_shape=(num_qubits,)),
            tf.keras.layers.Dense(classical_hidden_dims[1], activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output: traversability score
        ])
    
    def build_hybrid_model(self, geometry_params: Dict) -> tf.keras.Model:
        """Build hybrid quantum-classical model.
        
        Args:
            geometry_params: Wormhole geometry parameters
            
        Returns:
            Compiled Keras model
        """
        
        # Create quantum circuit
        self.quantum_circuit = TFQWormholeCircuit(self.num_qubits, geometry_params)
        
        # Input layer (classical parameters)
        classical_input = tf.keras.Input(shape=(len(geometry_params),), name='classical_input')
        
        # Quantum layer
        circuit_tensor = tfq.convert_to_tensor([self.quantum_circuit.traversal_circuit])
        quantum_layer = tfq.layers.Expectation()(
            circuit_tensor, 
            operators=self.quantum_circuit.tfq_observables
        )
        
        # Combine quantum and classical
        combined = tf.keras.layers.Concatenate()([classical_input, quantum_layer])
        
        # Classical post-processing
        output = self.classical_model(combined)
        
        # Build model
        model = tf.keras.Model(inputs=classical_input, outputs=output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def create_tfq_backend(config: Dict) -> TFQWormholeCircuit:
    """Factory function to create TensorFlow Quantum backend.
    
    Args:
        config: Configuration parameters
        
    Returns:
        TFQ quantum circuit instance
    """
    
    num_qubits = config.get('num_qubits', 4)
    geometry_params = {
        'throat_radius': config.get('throat_radius', 1e3),
        'traversal_probability': config.get('traversal_probability', 0.8)
    }
    
    return TFQWormholeCircuit(num_qubits, geometry_params)