"""
Hybrid Quantum-AI System for Wormhole Simulations.

This module provides a hybrid approach combining QuTiP quantum circuits with
TensorFlow/Keras for AI-driven parameter optimization, when TensorFlow Quantum
is not available.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import warnings
import logging

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    warnings.warn("QuTiP not available.")

from src.physics.constants import HBAR, C, PLANCK_LENGTH

logger = logging.getLogger(__name__)


class HybridQuantumAICircuit:
    """Hybrid quantum circuit using QuTiP for quantum operations and TF for AI."""
    
    def __init__(self, num_qubits: int, geometry_params: Dict):
        """Initialize hybrid quantum-AI circuit.
        
        Args:
            num_qubits: Number of qubits
            geometry_params: Wormhole geometry parameters
        """
        
        if not QUTIP_AVAILABLE:
            raise ImportError("QuTiP not available. Please install: pip install qutip")
        
        self.num_qubits = num_qubits
        self.geometry = geometry_params
        self.hilbert_dim = 2**num_qubits
        
        # Quantum circuit parameters (trainable with TensorFlow)
        self.theta = tf.Variable(np.pi/4, dtype=tf.float32, name='rotation_angle')
        self.phi = tf.Variable(0.0, dtype=tf.float32, name='phase_angle')
        
        # Initialize quantum operators
        self._setup_quantum_operators()
        self._setup_ai_components()
        
        logger.info(f"Initialized hybrid quantum-AI circuit with {num_qubits} qubits")
    
    def _setup_quantum_operators(self):
        """Set up QuTiP quantum operators."""
        
        # Single qubit operators
        self.sigma_x = [qt.tensor(*[qt.sigmax() if i == j else qt.qeye(2) 
                                   for j in range(self.num_qubits)]) 
                       for i in range(self.num_qubits)]
        
        self.sigma_y = [qt.tensor(*[qt.sigmay() if i == j else qt.qeye(2) 
                                   for j in range(self.num_qubits)]) 
                       for i in range(self.num_qubits)]
        
        self.sigma_z = [qt.tensor(*[qt.sigmaz() if i == j else qt.qeye(2) 
                                   for j in range(self.num_qubits)]) 
                       for i in range(self.num_qubits)]
        
        # Two-qubit operators for entanglement
        if self.num_qubits >= 2:
            self.cnot_ops = []
            for i in range(self.num_qubits - 1):
                # Create CNOT gate between qubit i (control) and i+1 (target)
                if self.num_qubits == 2:
                    cnot = qt.gates.cnot()
                else:
                    # For multi-qubit systems, embed CNOT in larger Hilbert space
                    ops = []
                    for j in range(self.num_qubits):
                        if j == i:
                            # Control qubit - use projection operators
                            proj_0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
                            proj_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
                            ops.append(proj_0)  # Will be modified below
                        elif j == i + 1:
                            # Target qubit - X gate
                            ops.append(qt.sigmax())
                        else:
                            # Identity on other qubits
                            ops.append(qt.qeye(2))
                    
                    # Build CNOT manually: |0><0| ⊗ I + |1><1| ⊗ X
                    proj_0_ops = ops.copy()
                    proj_0_ops[i] = qt.basis(2, 0) * qt.basis(2, 0).dag()
                    proj_0_ops[i + 1] = qt.qeye(2)
                    
                    proj_1_ops = ops.copy()
                    proj_1_ops[i] = qt.basis(2, 1) * qt.basis(2, 1).dag()
                    proj_1_ops[i + 1] = qt.sigmax()
                    
                    cnot = qt.tensor(*proj_0_ops) + qt.tensor(*proj_1_ops)
                
                self.cnot_ops.append(cnot)
    
    def _setup_ai_components(self):
        """Set up AI/ML components."""
        
        # Parameter optimization network
        self.param_optimizer = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),  # 4 inputs: throat radius, mass, etc.
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')  # Output: theta, phi
        ])
        
        # Quantum state classifier (for traversability prediction)
        self.state_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(2**self.num_qubits,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Traversability probability
        ])
    
    def create_traversal_state(self, use_ai_params: bool = False) -> qt.Qobj:
        """Create quantum state for wormhole traversal.
        
        Args:
            use_ai_params: Whether to use AI-optimized parameters
            
        Returns:
            QuTiP quantum state
        """
        
        if use_ai_params:
            # Use AI to predict optimal parameters
            geometry_features = np.array([[
                self.geometry.get('throat_radius', 1e3),
                self.geometry.get('mass', 1e30),
                self.geometry.get('traversal_probability', 0.8),
                self.geometry.get('exotic_matter_density', -1e-3)
            ]], dtype=np.float32)
            
            # Normalize features
            geometry_features = geometry_features / np.array([1e4, 1e31, 1.0, 1e-2])
            
            predicted_params = self.param_optimizer(geometry_features)
            theta_opt = float(predicted_params[0, 0].numpy())
            phi_opt = float(predicted_params[0, 1].numpy())
        else:
            theta_opt = float(self.theta.numpy())
            phi_opt = float(self.phi.numpy())
        
        # Create initial state (|0...0⟩) with proper tensor structure
        state = qt.tensor(*[qt.basis(2, 0) for _ in range(self.num_qubits)])
        
        # Apply Hadamard to create superposition
        if self.num_qubits >= 1:
            h_op = qt.tensor(*[qt.gates.snot() if i == 0 else qt.qeye(2) 
                              for i in range(self.num_qubits)])
            state = h_op * state
        
        # Apply rotation based on wormhole geometry
        if self.num_qubits >= 1:
            rot_op = (-1j * theta_opt * self.sigma_z[0]).expm()
            state = rot_op * state
        
        # Create entanglement between mouth regions
        if self.num_qubits >= 2 and len(self.cnot_ops) > 0:
            cnot = self.cnot_ops[0]
            state = cnot * state
        
        # Apply phase rotation
        if self.num_qubits >= 2:
            phase_op = (-1j * phi_opt * self.sigma_z[1]).expm()
            state = phase_op * state
        
        return state
    
    def measure_observables(self, state: Optional[qt.Qobj] = None) -> Dict[str, float]:
        """Measure quantum observables.
        
        Args:
            state: Quantum state to measure (if None, create new state)
            
        Returns:
            Dictionary of measurement results
        """
        
        if state is None:
            state = self.create_traversal_state()
        
        results = {}
        
        # Single-qubit measurements
        for i in range(min(self.num_qubits, 4)):  # Limit for performance
            results[f'sigma_x_{i}'] = qt.expect(self.sigma_x[i], state)
            results[f'sigma_y_{i}'] = qt.expect(self.sigma_y[i], state)
            results[f'sigma_z_{i}'] = qt.expect(self.sigma_z[i], state)
        
        # Two-qubit correlations
        if self.num_qubits >= 2:
            for i in range(min(self.num_qubits - 1, 2)):
                correlation = qt.expect(self.sigma_z[i] * self.sigma_z[i+1], state)
                results[f'correlation_z_{i}_{i+1}'] = correlation
        
        return results
    
    def compute_entanglement_entropy(self, state: Optional[qt.Qobj] = None) -> float:
        """Compute entanglement entropy.
        
        Args:
            state: Quantum state (if None, create new state)
            
        Returns:
            Von Neumann entropy
        """
        
        if state is None:
            state = self.create_traversal_state()
        
        if self.num_qubits < 2:
            return 0.0
        
        # Compute reduced density matrix for first half of qubits
        subsystem_dims = [2] * self.num_qubits
        keep_indices = list(range(self.num_qubits // 2))
        
        try:
            rho_reduced = state.ptrace(keep_indices)
            entropy = qt.entropy_vn(rho_reduced)
            return entropy
        except Exception as e:
            logger.warning(f"Could not compute entropy: {e}")
            return 0.0
    
    def compute_concurrence(self, state: Optional[qt.Qobj] = None) -> float:
        """Compute concurrence for two-qubit entanglement.
        
        Args:
            state: Quantum state (if None, create new state)
            
        Returns:
            Concurrence value
        """
        
        if self.num_qubits < 2:
            return 0.0
        
        if state is None:
            state = self.create_traversal_state()
        
        try:
            # For 2-qubit states, compute concurrence directly
            if self.num_qubits == 2:
                return qt.concurrence(state)
            else:
                # For multi-qubit states, trace out other qubits
                rho_2qubit = state.ptrace([0, 1])
                return qt.concurrence(rho_2qubit)
        except Exception as e:
            logger.warning(f"Could not compute concurrence: {e}")
            return 0.0
    
    def time_evolve(self, time_steps: int, dt: float, 
                   hamiltonian: Optional[qt.Qobj] = None) -> List[Dict[str, Any]]:
        """Time evolve quantum state.
        
        Args:
            time_steps: Number of evolution steps
            dt: Time step
            hamiltonian: Hamiltonian operator (if None, use default)
            
        Returns:
            List of measurement results at each time
        """
        
        if hamiltonian is None:
            # Create default wormhole Hamiltonian
            H = 0.1 * self.sigma_z[0]  # Single-qubit field
            
            if self.num_qubits >= 2:
                # Add interaction terms
                H += 0.05 * (self.sigma_x[0] * self.sigma_x[1] + 
                            self.sigma_y[0] * self.sigma_y[1])
                H += -0.02 * self.sigma_z[0] * self.sigma_z[1]  # Exotic matter coupling
            
            hamiltonian = H
        
        # Initial state
        state = self.create_traversal_state()
        
        results = []
        
        for step in range(time_steps):
            t = step * dt
            
            # Time evolution operator
            U = (-1j * hamiltonian * dt).expm()
            state = U * state
            
            # Measurements
            measurements = self.measure_observables(state)
            measurements['time'] = t
            measurements['entropy'] = self.compute_entanglement_entropy(state)
            measurements['concurrence'] = self.compute_concurrence(state)
            
            results.append(measurements)
        
        return results
    
    def predict_traversability(self, state: Optional[qt.Qobj] = None) -> float:
        """Predict wormhole traversability using AI.
        
        Args:
            state: Quantum state (if None, create new state)
            
        Returns:
            Traversability probability
        """
        
        if state is None:
            state = self.create_traversal_state(use_ai_params=True)
        
        # Convert quantum state to classical features
        state_vector = state.full().flatten()
        state_probs = np.abs(state_vector)**2  # Probability amplitudes
        
        # Predict traversability
        traversability = self.state_classifier(state_probs.reshape(1, -1))
        
        return float(traversability[0, 0])
    
    def optimize_parameters(self, target_metrics: Dict[str, float], 
                          learning_rate: float = 0.01, 
                          steps: int = 100) -> Dict[str, Any]:
        """Optimize circuit parameters for target metrics.
        
        Args:
            target_metrics: Target values for optimization
            learning_rate: Learning rate
            steps: Number of optimization steps
            
        Returns:
            Optimization results
        """
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        def loss_function():
            # Create state with current parameters
            state = self.create_traversal_state()
            
            # Compute current metrics
            current_entropy = self.compute_entanglement_entropy(state)
            current_concurrence = self.compute_concurrence(state)
            
            # Loss based on target metrics
            loss = 0.0
            if 'entropy' in target_metrics:
                loss += tf.square(current_entropy - target_metrics['entropy'])
            if 'concurrence' in target_metrics:
                loss += tf.square(current_concurrence - target_metrics['concurrence'])
            
            return loss
        
        loss_history = []
        
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_function()
            
            gradients = tape.gradient(loss, [self.theta, self.phi])
            optimizer.apply_gradients(zip(gradients, [self.theta, self.phi]))
            
            loss_history.append(float(loss))
            
            if step % 20 == 0:
                print(f"Step {step}, Loss: {loss:.6f}, θ: {self.theta.numpy():.6f}, φ: {self.phi.numpy():.6f}")
        
        return {
            'optimized_theta': float(self.theta.numpy()),
            'optimized_phi': float(self.phi.numpy()),
            'final_loss': loss_history[-1],
            'loss_history': loss_history
        }
    
    def train_ai_components(self, training_data: List[Dict], epochs: int = 50):
        """Train AI components on simulation data.
        
        Args:
            training_data: List of training examples
            epochs: Training epochs
        """
        
        # Prepare training data
        geometry_features = []
        optimal_params = []
        state_features = []
        traversability_labels = []
        
        for example in training_data:
            geometry_features.append([
                example['throat_radius'],
                example['mass'],
                example['traversal_probability'],
                example['exotic_matter_density']
            ])
            
            optimal_params.append([
                example['optimal_theta'],
                example['optimal_phi']
            ])
            
            if 'state_vector' in example:
                state_probs = np.abs(example['state_vector'])**2
                state_features.append(state_probs)
                traversability_labels.append(example['traversability'])
        
        # Train parameter optimizer
        if geometry_features and optimal_params:
            X_geom = np.array(geometry_features, dtype=np.float32)
            y_params = np.array(optimal_params, dtype=np.float32)
            
            # Normalize features
            X_geom = X_geom / np.array([1e4, 1e31, 1.0, 1e-2])
            
            self.param_optimizer.compile(optimizer='adam', loss='mse')
            self.param_optimizer.fit(X_geom, y_params, epochs=epochs, verbose=0)
        
        # Train state classifier
        if state_features and traversability_labels:
            X_states = np.array(state_features, dtype=np.float32)
            y_trav = np.array(traversability_labels, dtype=np.float32)
            
            self.state_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.state_classifier.fit(X_states, y_trav, epochs=epochs, verbose=0)
        
        logger.info(f"AI components trained on {len(training_data)} examples")
    
    def construct_hamiltonian(self) -> qt.Qobj:
        """Construct Hamiltonian for compatibility."""
        H = 0.1 * self.sigma_z[0]
        if self.num_qubits >= 2:
            H += 0.05 * (self.sigma_x[0] * self.sigma_x[1] + self.sigma_y[0] * self.sigma_y[1])
        return H


def create_hybrid_backend(config: Dict) -> HybridQuantumAICircuit:
    """Factory function to create hybrid quantum-AI backend.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Hybrid quantum-AI circuit
    """
    
    num_qubits = config.get('num_qubits', 4)
    geometry_params = {
        'throat_radius': config.get('throat_radius', 1e3),
        'traversal_probability': config.get('traversal_probability', 0.8),
        'mass': config.get('mass', 1e30),
        'exotic_matter_density': config.get('exotic_matter_density', -1e-3)
    }
    
    return HybridQuantumAICircuit(num_qubits, geometry_params)