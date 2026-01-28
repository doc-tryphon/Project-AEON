"""
Quantum machine learning algorithms for enhanced wormhole prediction.

This module implements quantum machine learning algorithms that leverage
quantum circuits for enhanced prediction capabilities, including quantum
neural networks, variational quantum classifiers, and quantum feature maps.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from src.quantum.wormhole_circuit import WormholeQuantumCircuit


class QuantumLayer(nn.Module):
    """Quantum circuit layer for hybrid quantum-classical neural network."""
    
    def __init__(self, n_qubits: int, n_params: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.params = nn.Parameter(torch.randn(n_params))
        
        # Initialize quantum circuit
        self.circuit = WormholeQuantumCircuit(
            num_qubits=n_qubits,
            geometry_params={'throat_radius': 1.0, 'mass': 1.0}
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum circuit transformation.
        
        Args:
            x: Input tensor (batch_size, n_features)
            
        Returns:
            Quantum circuit output (batch_size, 2**n_qubits)
        """
        batch_size = x.shape[0]
        device = x.device
        output = torch.zeros((batch_size, 2**self.n_qubits), device=device)
        
        for i in range(batch_size):
            # Encode classical data into quantum state
            quantum_state = self._encode_input(x[i])
            
            # Apply parameterized quantum circuit
            quantum_state = self._apply_quantum_circuit(quantum_state)
            
            # Measure quantum state
            output[i] = self._measure_state(quantum_state)
            
        return output
        
    def _encode_input(self, x: torch.Tensor) -> np.ndarray:
        """Encode classical input into quantum state."""
        # Amplitude encoding
        amplitudes = x.cpu().numpy()
        normalized = amplitudes / np.linalg.norm(amplitudes)
        return self.circuit.prepare_state(normalized)
        
    def _apply_quantum_circuit(self, state: np.ndarray) -> np.ndarray:
        """Apply parameterized quantum circuit."""
        params_dict = {f'theta_{i}': p.item() for i, p in enumerate(self.params)}
        return self.circuit.apply_gates(state, params_dict)
        
    def _measure_state(self, state: np.ndarray) -> torch.Tensor:
        """Measure quantum state to get classical output."""
        probabilities = np.abs(state)**2
        return torch.from_numpy(probabilities).float()


class QuantumNeuralNetwork(nn.Module):
    """Hybrid quantum-classical neural network."""
    
    def __init__(self,
                input_size: int,
                n_qubits: int,
                n_quantum_layers: int = 2,
                hidden_size: int = 64):
        super().__init__()
        
        self.input_size = input_size
        self.n_qubits = n_qubits
        
        # Classical pre-processing
        self.pre_processing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2**n_qubits)
        )
        
        # Quantum layers
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(n_qubits, n_params=3*n_qubits)
            for _ in range(n_quantum_layers)
        ])
        
        # Classical post-processing
        self.post_processing = nn.Sequential(
            nn.Linear(2**n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid network.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Network output (batch_size, 1)
        """
        # Classical pre-processing
        x = self.pre_processing(x)
        
        # Quantum circuit layers
        for quantum_layer in self.quantum_layers:
            x = quantum_layer(x)
            
        # Classical post-processing
        return self.post_processing(x)
        
    def quantum_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate quantum expectation values.
        
        Args:
            x: Input tensor
            
        Returns:
            Expectation values for quantum observables
        """
        with torch.no_grad():
            # Get quantum state after circuit
            pre_quantum = self.pre_processing(x)
            quantum_state = self.quantum_layers[-1](pre_quantum)
            
            # Calculate expectations for Pauli observables
            expectations = []
            for obs in ['X', 'Y', 'Z']:
                exp_val = self._measure_observable(quantum_state, obs)
                expectations.append(exp_val)
                
        return torch.stack(expectations, dim=1)
        
    def _measure_observable(self, 
                         state: torch.Tensor,
                         observable: str) -> torch.Tensor:
        """Measure quantum observable expectation value."""
        if observable == 'X':
            matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        elif observable == 'Y':
            matrix = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        else:  # Z
            matrix = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
            
        # Convert to density matrix
        state = state.view(-1, 2**self.n_qubits, 1)
        density = torch.bmm(state, state.transpose(1,2))
        
        # Calculate expectation value
        expectation = torch.real(torch.trace(torch.mm(density[0], matrix)))
        return expectation.unsqueeze(0)
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Quantum computing imports
import qutip as qt
from qutip import Qobj, tensor, basis, qeye, sigmaz, sigmax, sigmay
import cirq
import sympy

# Classical ML for comparison and hybrid approaches
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# TensorFlow Quantum (if available)
try:
    import tensorflow as tf
    import tensorflow_quantum as tfq
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False
    print("TensorFlow Quantum not available. Using qutip-based implementations.")

from src.quantum.wormhole_circuit import QuantumWormholeCircuit
from src.quantum.entanglement_dynamics import WormholeEntanglementDynamics


@dataclass
class QuantumFeatureMap:
    """Quantum feature map for encoding classical data into quantum states."""
    
    feature_dimension: int
    qubit_count: int
    encoding_type: str  # 'angle', 'amplitude', 'iqp'
    entangling_layers: int
    
    def encode_data(self, data: np.ndarray) -> List[Qobj]:
        """Encode classical data points into quantum states."""
        
        if self.encoding_type == 'angle':
            return self._angle_encoding(data)
        elif self.encoding_type == 'amplitude':
            return self._amplitude_encoding(data)
        elif self.encoding_type == 'iqp':
            return self._iqp_encoding(data)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _angle_encoding(self, data: np.ndarray) -> List[Qobj]:
        """Encode data using angle encoding (rotation gates)."""
        
        encoded_states = []
        
        for sample in data:
            # Start with |0⟩^n state
            state = tensor([basis(2, 0) for _ in range(self.qubit_count)])
            
            # Apply rotation gates
            for i in range(min(len(sample), self.qubit_count)):
                # Create rotation operator for qubit i
                rotation_ops = []
                for j in range(self.qubit_count):
                    if j == i:
                        # R_Y(θ) rotation where θ = π * normalized_feature
                        angle = np.pi * sample[i]
                        ry = qt.ry(angle)
                        rotation_ops.append(ry)
                    else:
                        rotation_ops.append(qeye(2))
                
                rotation_gate = tensor(rotation_ops)
                state = rotation_gate * state
            
            # Apply entangling layers
            for layer in range(self.entangling_layers):
                state = self._apply_entangling_layer(state)
            
            encoded_states.append(state)
        
        return encoded_states
    
    def _amplitude_encoding(self, data: np.ndarray) -> List[Qobj]:
        """Encode data using amplitude encoding."""
        
        encoded_states = []
        
        for sample in data:
            # Normalize the data to unit vector
            sample_normalized = sample / np.linalg.norm(sample)
            
            # Pad or truncate to fit 2^n dimensions
            target_dim = 2**self.qubit_count
            
            if len(sample_normalized) > target_dim:
                amplitudes = sample_normalized[:target_dim]
            else:
                amplitudes = np.pad(sample_normalized, 
                                  (0, target_dim - len(sample_normalized)))
            
            # Create quantum state with these amplitudes
            state = Qobj(amplitudes.reshape(-1, 1))
            state = state.unit()  # Ensure normalization
            
            encoded_states.append(state)
        
        return encoded_states
    
    def _iqp_encoding(self, data: np.ndarray) -> List[Qobj]:
        """Encode data using Instantaneous Quantum Polynomial (IQP) circuits."""
        
        encoded_states = []
        
        for sample in data:
            # Start with |+⟩^n state (Hadamard on all qubits)
            plus_state = tensor([(basis(2, 0) + basis(2, 1)).unit() 
                               for _ in range(self.qubit_count)])
            
            state = plus_state
            
            # Apply diagonal unitaries based on data
            for i in range(min(len(sample), self.qubit_count)):
                # Z-rotation based on feature value
                angle = 2 * np.pi * sample[i]
                
                rotation_ops = []
                for j in range(self.qubit_count):
                    if j == i:
                        rz = qt.rz(angle)
                        rotation_ops.append(rz)
                    else:
                        rotation_ops.append(qeye(2))
                
                rotation_gate = tensor(rotation_ops)
                state = rotation_gate * state
            
            # Apply ZZ interactions
            for i in range(self.qubit_count - 1):
                for j in range(i + 1, self.qubit_count):
                    # ZZ interaction strength based on feature correlation
                    if i < len(sample) and j < len(sample):
                        interaction_strength = sample[i] * sample[j]
                        
                        zz_ops = []
                        for k in range(self.qubit_count):
                            if k == i or k == j:
                                zz_ops.append(sigmaz())
                            else:
                                zz_ops.append(qeye(2))
                        
                        zz_gate = tensor(zz_ops)
                        zz_evolution = (-1j * interaction_strength * zz_gate).expm()
                        state = zz_evolution * state
            
            encoded_states.append(state)
        
        return encoded_states
    
    def _apply_entangling_layer(self, state: Qobj) -> Qobj:
        """Apply entangling layer (CNOT gates)."""
        
        # Apply CNOT gates between adjacent qubits
        for i in range(self.qubit_count - 1):
            cnot_ops = []
            
            for j in range(self.qubit_count):
                if j == i:
                    # Control qubit
                    cnot_ops.append(basis(2, 0) * basis(2, 0).dag() + 
                                   basis(2, 1) * basis(2, 1).dag())
                elif j == i + 1:
                    # Target qubit  
                    cnot_ops.append(qeye(2))
                else:
                    cnot_ops.append(qeye(2))
            
            # This is a simplified CNOT - full implementation would be more complex
            # For now, just apply some entangling operation
            entangling_ops = []
            for j in range(self.qubit_count):
                if j == i or j == i + 1:
                    entangling_ops.append((sigmaz() + sigmax()).unit())
                else:
                    entangling_ops.append(qeye(2))
            
            entangling_gate = tensor(entangling_ops)
            state = entangling_gate * state
        
        return state


class QuantumNeuralNetwork:
    """Quantum neural network using parameterized quantum circuits."""
    
    def __init__(self, num_qubits: int, num_layers: int, 
                 feature_map: QuantumFeatureMap):
        """Initialize quantum neural network.
        
        Args:
            num_qubits: Number of qubits in the circuit
            num_layers: Number of variational layers
            feature_map: Quantum feature map for data encoding
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.feature_map = feature_map
        
        # Initialize random parameters
        self.num_params = num_qubits * num_layers * 3  # 3 rotation angles per qubit per layer
        self.parameters = np.random.uniform(0, 2*np.pi, self.num_params)
        
        self.training_history = []
        
    def create_ansatz(self, parameters: np.ndarray) -> Qobj:
        """Create parameterized quantum circuit (ansatz).
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Quantum circuit as unitary operator
        """
        
        # Start with identity
        circuit = tensor([qeye(2) for _ in range(self.num_qubits)])
        
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Parameterized single-qubit rotations
            for qubit in range(self.num_qubits):
                
                # Three rotation angles per qubit
                rx_angle = parameters[param_idx]
                ry_angle = parameters[param_idx + 1] 
                rz_angle = parameters[param_idx + 2]
                param_idx += 3
                
                # Create rotation operators
                rx_ops = []
                ry_ops = []
                rz_ops = []
                
                for i in range(self.num_qubits):
                    if i == qubit:
                        rx_ops.append(qt.rx(rx_angle))
                        ry_ops.append(qt.ry(ry_angle))
                        rz_ops.append(qt.rz(rz_angle))
                    else:
                        rx_ops.append(qeye(2))
                        ry_ops.append(qeye(2))
                        rz_ops.append(qeye(2))
                
                rx_gate = tensor(rx_ops)
                ry_gate = tensor(ry_ops)
                rz_gate = tensor(rz_ops)
                
                # Apply rotations
                circuit = rz_gate * ry_gate * rx_gate * circuit
            
            # Entangling gates (simplified - ring connectivity)
            if layer < self.num_layers - 1:  # Don't entangle on last layer
                for i in range(self.num_qubits - 1):
                    # Simple entangling gate between qubits i and i+1
                    entangling_ops = []
                    for j in range(self.num_qubits):
                        if j == i:
                            entangling_ops.append(sigmaz())
                        elif j == i + 1:
                            entangling_ops.append(sigmaz())
                        else:
                            entangling_ops.append(qeye(2))
                    
                    zz_gate = tensor(entangling_ops)
                    entangling_evolution = (-1j * 0.1 * zz_gate).expm()  # Small coupling
                    circuit = entangling_evolution * circuit
        
        return circuit
    
    def forward_pass(self, encoded_states: List[Qobj], 
                    parameters: np.ndarray) -> np.ndarray:
        """Forward pass through quantum neural network.
        
        Args:
            encoded_states: List of quantum states (encoded data)
            parameters: Circuit parameters
            
        Returns:
            Measurement outcomes
        """
        
        ansatz = self.create_ansatz(parameters)
        
        results = []
        
        for state in encoded_states:
            # Apply ansatz to encoded state
            output_state = ansatz * state
            
            # Measurement - expectation value of Pauli-Z on first qubit
            measurement_ops = [sigmaz()] + [qeye(2) for _ in range(self.num_qubits - 1)]
            measurement_operator = tensor(measurement_ops)
            
            expectation_value = qt.expect(measurement_operator, output_state)
            results.append(expectation_value)
        
        return np.array(results)
    
    def compute_gradients(self, encoded_states: List[Qobj], 
                         targets: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Compute gradients using parameter shift rule.
        
        Args:
            encoded_states: Encoded training data
            targets: Target values
            parameters: Current parameters
            
        Returns:
            Parameter gradients
        """
        
        gradients = np.zeros_like(parameters)
        
        # Parameter shift rule: ∂⟨O⟩/∂θ = [⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)] / 2
        shift = np.pi / 2
        
        for i in range(len(parameters)):
            # Forward pass with positive shift
            params_plus = parameters.copy()
            params_plus[i] += shift
            outputs_plus = self.forward_pass(encoded_states, params_plus)
            
            # Forward pass with negative shift  
            params_minus = parameters.copy()
            params_minus[i] -= shift
            outputs_minus = self.forward_pass(encoded_states, params_minus)
            
            # Gradient of loss function
            loss_grad_plus = 2 * np.mean((outputs_plus - targets) * outputs_plus)
            loss_grad_minus = 2 * np.mean((outputs_minus - targets) * outputs_minus)
            
            gradients[i] = (loss_grad_plus - loss_grad_minus) / 2
        
        return gradients
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100, learning_rate: float = 0.1,
              batch_size: Optional[int] = None) -> Dict:
        """Train quantum neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets (converted to -1/+1)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Batch size (None for full batch)
            
        Returns:
            Training history
        """
        
        # Convert labels to -1/+1
        y_binary = 2 * y_train - 1
        
        # Encode training data
        print("Encoding training data...")
        encoded_train = self.feature_map.encode_data(X_train)
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            
            # Batch processing
            if batch_size is None or batch_size >= len(encoded_train):
                batch_indices = list(range(len(encoded_train)))
            else:
                batch_indices = np.random.choice(len(encoded_train), batch_size, replace=False)
            
            batch_states = [encoded_train[i] for i in batch_indices]
            batch_targets = y_binary[batch_indices]
            
            # Forward pass
            outputs = self.forward_pass(batch_states, self.parameters)
            
            # Compute loss (MSE)
            loss = np.mean((outputs - batch_targets)**2)
            
            # Compute accuracy
            predictions = np.sign(outputs)
            accuracy = np.mean(predictions == batch_targets)
            
            # Compute gradients
            gradients = self.compute_gradients(batch_states, batch_targets, self.parameters)
            
            # Update parameters
            self.parameters -= learning_rate * gradients
            
            # Record history
            losses.append(loss)
            accuracies.append(accuracy)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.training_history = {
            'losses': losses,
            'accuracies': accuracies
        }
        
        return self.training_history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Binary predictions (0/1)
        """
        
        # Encode test data
        encoded_test = self.feature_map.encode_data(X_test)
        
        # Forward pass
        outputs = self.forward_pass(encoded_test, self.parameters)
        
        # Convert to binary predictions
        binary_predictions = (np.sign(outputs) + 1) / 2  # Convert -1/+1 to 0/1
        
        return binary_predictions.astype(int)


class VariationalQuantumClassifier:
    """Variational Quantum Classifier using quantum kernel methods."""
    
    def __init__(self, num_qubits: int, feature_map: QuantumFeatureMap):
        """Initialize VQC.
        
        Args:
            num_qubits: Number of qubits
            feature_map: Quantum feature map
        """
        self.num_qubits = num_qubits
        self.feature_map = feature_map
        self.quantum_kernel_matrix = None
        self.classical_svm = None
        
    def compute_quantum_kernel(self, X1: np.ndarray, 
                             X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute quantum kernel matrix.
        
        Args:
            X1: First set of data points
            X2: Second set of data points (if None, use X1)
            
        Returns:
            Quantum kernel matrix
        """
        
        if X2 is None:
            X2 = X1
        
        # Encode data
        encoded_X1 = self.feature_map.encode_data(X1)
        encoded_X2 = self.feature_map.encode_data(X2)
        
        # Compute kernel matrix
        kernel_matrix = np.zeros((len(encoded_X1), len(encoded_X2)))
        
        for i, state1 in enumerate(encoded_X1):
            for j, state2 in enumerate(encoded_X2):
                # Quantum kernel: |⟨φ(x1)|φ(x2)⟩|²
                overlap = state1.dag() * state2
                kernel_value = abs(overlap.tr())**2
                kernel_matrix[i, j] = kernel_value
        
        return kernel_matrix
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train VQC using quantum kernel.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training results
        """
        
        print("Computing quantum kernel matrix...")
        
        # Compute quantum kernel matrix for training data
        self.quantum_kernel_matrix = self.compute_quantum_kernel(X_train)
        
        # Train classical SVM with quantum kernel
        self.classical_svm = SVC(kernel='precomputed')
        self.classical_svm.fit(self.quantum_kernel_matrix, y_train)
        
        return {
            'kernel_matrix_shape': self.quantum_kernel_matrix.shape,
            'support_vectors': len(self.classical_svm.support_)
        }
    
    def predict(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Make predictions using trained VQC.
        
        Args:
            X_test: Test features  
            X_train: Original training features (needed for kernel computation)
            
        Returns:
            Predictions
        """
        
        if self.classical_svm is None:
            raise ValueError("Model must be trained first")
        
        # Compute kernel matrix between test and train data
        test_kernel_matrix = self.compute_quantum_kernel(X_test, X_train)
        
        # Make predictions
        predictions = self.classical_svm.predict(test_kernel_matrix)
        
        return predictions


class QuantumFeatureSelection:
    """Quantum-enhanced feature selection using quantum mutual information."""
    
    def __init__(self, num_qubits: int):
        """Initialize quantum feature selection.
        
        Args:
            num_qubits: Number of qubits for quantum circuits
        """
        self.num_qubits = num_qubits
        self.selected_features = None
        
    def quantum_mutual_information(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """Compute quantum mutual information between two feature sets.
        
        Args:
            X1: First feature set
            X2: Second feature set
            
        Returns:
            Quantum mutual information
        """
        
        # Create joint quantum state encoding both feature sets
        joint_states = []
        
        for x1, x2 in zip(X1, X2):
            # Simple encoding: angle encoding for each feature set
            state1 = basis(2, 0)
            state2 = basis(2, 0)
            
            # Apply rotations based on feature values
            angle1 = np.pi * np.mean(x1)  # Average feature value
            angle2 = np.pi * np.mean(x2)
            
            state1 = qt.ry(angle1) * state1
            state2 = qt.ry(angle2) * state2
            
            # Create joint state
            joint_state = tensor(state1, state2)
            joint_states.append(joint_state)
        
        # Compute quantum mutual information using entanglement entropy
        total_mi = 0.0
        
        for joint_state in joint_states:
            # Reduced density matrices
            rho1 = joint_state.ptrace(0)  # Trace out second subsystem
            rho2 = joint_state.ptrace(1)  # Trace out first subsystem  
            
            # Von Neumann entropies
            S1 = qt.entropy_vn(rho1)
            S2 = qt.entropy_vn(rho2)
            S12 = qt.entropy_vn(joint_state)
            
            # Quantum mutual information: I(1:2) = S(1) + S(2) - S(1,2)
            mi = S1 + S2 - S12
            total_mi += mi
        
        return total_mi / len(joint_states)
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       k_features: int) -> List[int]:
        """Select top k features using quantum mutual information.
        
        Args:
            X: Feature matrix
            y: Target vector  
            k_features: Number of features to select
            
        Returns:
            Indices of selected features
        """
        
        num_features = X.shape[1]
        
        # Compute mutual information for each feature with target
        feature_scores = []
        
        for i in range(num_features):
            feature_i = X[:, i:i+1]
            target_expanded = y.reshape(-1, 1)
            
            mi_score = self.quantum_mutual_information(feature_i, target_expanded)
            feature_scores.append((i, mi_score))
        
        # Sort features by mutual information score
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top k features
        selected_indices = [idx for idx, _ in feature_scores[:k_features]]
        self.selected_features = selected_indices
        
        return selected_indices


class HybridQuantumClassical:
    """Hybrid quantum-classical machine learning model."""
    
    def __init__(self, quantum_component: QuantumNeuralNetwork,
                 classical_component: Any):
        """Initialize hybrid model.
        
        Args:
            quantum_component: Quantum neural network
            classical_component: Classical ML model
        """
        self.quantum_component = quantum_component
        self.classical_component = classical_component
        
    def extract_quantum_features(self, X: np.ndarray) -> np.ndarray:
        """Extract quantum features for classical processing.
        
        Args:
            X: Input data
            
        Returns:
            Quantum-derived features
        """
        
        # Encode data into quantum states
        encoded_states = self.quantum_component.feature_map.encode_data(X)
        
        # Extract quantum features using various measurements
        quantum_features = []
        
        for state in encoded_states:
            features = []
            
            # Pauli expectation values
            for i in range(self.quantum_component.num_qubits):
                # Measurement operators for qubit i
                pauli_ops = []
                
                for op in [sigmax(), sigmay(), sigmaz()]:
                    ops = []
                    for j in range(self.quantum_component.num_qubits):
                        if j == i:
                            ops.append(op)
                        else:
                            ops.append(qeye(2))
                    
                    measurement_op = tensor(ops)
                    expectation = qt.expect(measurement_op, state)
                    features.append(expectation)
            
            # Entanglement measures (simplified)
            if self.quantum_component.num_qubits >= 2:
                # Compute entanglement entropy between first and rest
                rho_reduced = state.ptrace([0])
                entropy = qt.entropy_vn(rho_reduced)
                features.append(entropy)
            
            quantum_features.append(features)
        
        return np.array(quantum_features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train hybrid model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training results
        """
        
        # Extract quantum features
        quantum_features = self.extract_quantum_features(X_train)
        
        # Combine with original features
        hybrid_features = np.hstack([X_train, quantum_features])
        
        # Train classical component on hybrid features
        if hasattr(self.classical_component, 'fit'):
            self.classical_component.fit(hybrid_features, y_train)
        
        return {
            'quantum_features_shape': quantum_features.shape,
            'hybrid_features_shape': hybrid_features.shape
        }
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using hybrid model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        
        # Extract quantum features
        quantum_features = self.extract_quantum_features(X_test)
        
        # Combine with original features
        hybrid_features = np.hstack([X_test, quantum_features])
        
        # Make predictions using classical component
        if hasattr(self.classical_component, 'predict'):
            return self.classical_component.predict(hybrid_features)
        else:
            raise ValueError("Classical component does not have predict method")


class QuantumEnsemble:
    """Ensemble of quantum classifiers."""
    
    def __init__(self, quantum_models: List[Union[QuantumNeuralNetwork, 
                                                VariationalQuantumClassifier]]):
        """Initialize quantum ensemble.
        
        Args:
            quantum_models: List of quantum models
        """
        self.quantum_models = quantum_models
        self.model_weights = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_split: float = 0.2) -> Dict:
        """Train ensemble of quantum models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Validation split ratio
            
        Returns:
            Training results
        """
        
        # Split data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        model_scores = []
        
        # Train each model
        for i, model in enumerate(self.quantum_models):
            print(f"Training quantum model {i+1}/{len(self.quantum_models)}")
            
            if isinstance(model, QuantumNeuralNetwork):
                model.train(X_train_split, y_train_split, epochs=50)
                predictions = model.predict(X_val)
            elif isinstance(model, VariationalQuantumClassifier):
                model.train(X_train_split, y_train_split)
                predictions = model.predict(X_val, X_train_split)
            
            # Compute validation accuracy
            accuracy = accuracy_score(y_val, predictions)
            model_scores.append(accuracy)
        
        # Compute model weights based on performance
        model_scores = np.array(model_scores)
        self.model_weights = model_scores / np.sum(model_scores)
        
        return {
            'individual_scores': model_scores.tolist(),
            'model_weights': self.model_weights.tolist()
        }
    
    def predict(self, X_test: np.ndarray, X_train: np.ndarray = None) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X_test: Test features
            X_train: Training features (needed for some models)
            
        Returns:
            Ensemble predictions
        """
        
        if self.model_weights is None:
            raise ValueError("Ensemble must be trained first")
        
        predictions_list = []
        
        for model in self.quantum_models:
            if isinstance(model, QuantumNeuralNetwork):
                pred = model.predict(X_test)
            elif isinstance(model, VariationalQuantumClassifier):
                if X_train is None:
                    raise ValueError("X_train required for VQC predictions")
                pred = model.predict(X_test, X_train)
            
            predictions_list.append(pred)
        
        # Weighted voting
        predictions_array = np.array(predictions_list)
        weighted_predictions = np.average(predictions_array, axis=0, weights=self.model_weights)
        
        # Convert to binary predictions
        return (weighted_predictions > 0.5).astype(int)


def compare_quantum_classical(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            num_qubits: int = 4) -> Dict:
    """Compare quantum and classical machine learning approaches.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        num_qubits: Number of qubits for quantum models
        
    Returns:
        Comparison results
    """
    
    results = {}
    
    # Classical baselines
    print("Training classical baselines...")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    results['classical'] = {
        'random_forest_accuracy': rf_accuracy,
        'svm_accuracy': svm_accuracy
    }
    
    # Quantum models
    print("Training quantum models...")
    
    try:
        # Quantum Neural Network
        feature_map = QuantumFeatureMap(
            feature_dimension=X_train.shape[1],
            qubit_count=num_qubits,
            encoding_type='angle',
            entangling_layers=2
        )
        
        qnn = QuantumNeuralNetwork(num_qubits, num_layers=3, feature_map=feature_map)
        qnn.train(X_train, y_train, epochs=50, learning_rate=0.1)
        qnn_pred = qnn.predict(X_test)
        qnn_accuracy = accuracy_score(y_test, qnn_pred)
        
        # Variational Quantum Classifier
        vqc = VariationalQuantumClassifier(num_qubits, feature_map)
        vqc.train(X_train, y_train)
        vqc_pred = vqc.predict(X_test, X_train)
        vqc_accuracy = accuracy_score(y_test, vqc_pred)
        
        results['quantum'] = {
            'qnn_accuracy': qnn_accuracy,
            'vqc_accuracy': vqc_accuracy
        }
        
        # Hybrid approach
        print("Training hybrid model...")
        
        from sklearn.linear_model import LogisticRegression
        
        hybrid = HybridQuantumClassical(qnn, LogisticRegression())
        hybrid.train(X_train, y_train)
        hybrid_pred = hybrid.predict(X_test)
        hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
        
        results['hybrid'] = {
            'accuracy': hybrid_accuracy
        }
        
    except Exception as e:
        print(f"Error in quantum models: {e}")
        results['quantum'] = {'error': str(e)}
        results['hybrid'] = {'error': str(e)}
    
    return results


def create_wormhole_quantum_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create quantum-enhanced dataset for wormhole stability prediction.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Features and labels
    """
    
    np.random.seed(42)
    
    # Generate base physics features
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Wormhole parameters
        throat_radius = np.random.uniform(1e3, 1e6)
        exotic_density = np.random.uniform(-1e20, -1e10)
        shape_param = np.random.uniform(0.1, 10.0)
        quantum_corr = np.random.uniform(0.0, 1.0)
        
        # Additional quantum features
        entanglement_entropy = np.random.exponential(2.0)
        vacuum_fluctuations = np.random.normal(0, 1e-10)
        
        feature_vector = [
            throat_radius / 1e6,  # Normalized
            exotic_density / -1e20,
            shape_param / 10.0,
            quantum_corr,
            entanglement_entropy / 5.0,
            vacuum_fluctuations * 1e10
        ]
        
        # Stability based on quantum-influenced criteria
        stability_score = (
            0.3 * (1 - shape_param / throat_radius * 1e-3) +
            0.3 * min(1, abs(exotic_density) / 1e15) +
            0.2 * (1 - abs(quantum_corr - 0.1) / 0.9) +
            0.2 * min(1, entanglement_entropy / 3.0)
        )
        
        # Add quantum enhancement based on feature correlations
        quantum_enhancement = np.sin(np.pi * feature_vector[0] * feature_vector[1]) * 0.1
        stability_score += quantum_enhancement
        
        label = 1 if stability_score > 0.5 else 0
        
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)


def visualize_quantum_ml_results(results: Dict, save_path: Optional[str] = None):
    """Visualize quantum ML results.
    
    Args:
        results: Results from compare_quantum_classical
        save_path: Path to save plot
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    methods = []
    accuracies = []
    
    if 'classical' in results:
        methods.extend(['Random Forest', 'SVM'])
        accuracies.extend([
            results['classical']['random_forest_accuracy'],
            results['classical']['svm_accuracy']
        ])
    
    if 'quantum' in results and 'error' not in results['quantum']:
        methods.extend(['QNN', 'VQC'])
        accuracies.extend([
            results['quantum']['qnn_accuracy'],
            results['quantum']['vqc_accuracy']
        ])
    
    if 'hybrid' in results and 'error' not in results['hybrid']:
        methods.append('Hybrid')
        accuracies.append(results['hybrid']['accuracy'])
    
    colors = ['blue', 'blue', 'red', 'red', 'green'][:len(methods)]
    
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Quantum vs Classical ML Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Quantum advantage analysis
    if ('classical' in results and 'quantum' in results and 
        'error' not in results['quantum']):
        
        classical_best = max(results['classical']['random_forest_accuracy'],
                           results['classical']['svm_accuracy'])
        quantum_best = max(results['quantum']['qnn_accuracy'],
                          results['quantum']['vqc_accuracy'])
        
        quantum_advantage = quantum_best - classical_best
        
        ax2.bar(['Classical Best', 'Quantum Best'], [classical_best, quantum_best],
               color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Quantum Advantage Analysis')
        ax2.set_ylim(0, 1)
        
        ax2.text(0.5, max(classical_best, quantum_best) + 0.05,
                f'Quantum Advantage: {quantum_advantage:+.3f}',
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Integration with existing wormhole simulation
def integrate_quantum_ml_with_simulation(wormhole_circuit: QuantumWormholeCircuit,
                                       stability_data: np.ndarray) -> QuantumNeuralNetwork:
    """Integrate quantum ML with wormhole quantum circuits.
    
    Args:
        wormhole_circuit: Quantum wormhole circuit
        stability_data: Historical stability data
        
    Returns:
        Trained quantum neural network
    """
    
    # Extract quantum features from wormhole circuit
    quantum_features = []
    
    # Use the quantum circuit to generate feature representations
    for i in range(len(stability_data)):
        # Create quantum state from stability data
        state = wormhole_circuit.encode_geometry()
        
        # Extract features using measurements
        feature_vector = []
        
        # Measure various observables
        observables = [wormhole_circuit.total_sx, wormhole_circuit.total_sy, wormhole_circuit.total_sz]
        
        for obs in observables:
            expectation = qt.expect(obs, state)
            feature_vector.append(expectation.real)
        
        quantum_features.append(feature_vector)
    
    quantum_features = np.array(quantum_features)
    
    # Create labels (simplified - would use actual stability assessment)
    labels = np.random.randint(0, 2, len(stability_data))
    
    # Create and train quantum neural network
    feature_map = QuantumFeatureMap(
        feature_dimension=quantum_features.shape[1],
        qubit_count=wormhole_circuit.num_qubits,
        encoding_type='angle',
        entangling_layers=2
    )
    
    qnn = QuantumNeuralNetwork(
        num_qubits=wormhole_circuit.num_qubits,
        num_layers=3,
        feature_map=feature_map
    )
    
    qnn.train(quantum_features, labels, epochs=100)
    
    return qnn