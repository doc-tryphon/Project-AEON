"""
Quantum Superposition Attention Mechanism.

This module implements attention mechanisms that can maintain multiple contradictory
interpretations simultaneously until forced collapse. Built on the existing
quantum simulation framework.
"""

import numpy as np
import torch
import torch.nn as nn
import qutip as qt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Import from existing quantum framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum.entanglement_dynamics import EntanglementDynamics
from quantum.hybrid_quantum_ai import HybridQuantumAICircuit
from ai.quantum_ml import QuantumLayer
from physics.constants import HBAR, C


@dataclass
class AttentionState:
    """Represents a quantum superposition attention state."""
    
    quantum_state: qt.Qobj
    classical_weights: torch.Tensor
    coherence_time: float
    collapse_threshold: float
    interpretations: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validate attention state consistency."""
        if len(self.interpretations) != self.quantum_state.dims[0][0]:
            raise ValueError("Number of interpretations must match quantum state dimension")


class QuantumAttentionHead(nn.Module):
    """Single quantum attention head that maintains superposed interpretations."""
    
    def __init__(self, 
                 d_model: int,
                 num_interpretations: int = 4,
                 coherence_time: float = 1e-3,
                 collapse_threshold: float = 0.8):
        """Initialize quantum attention head.
        
        Args:
            d_model: Model dimension
            num_interpretations: Number of simultaneous interpretations
            coherence_time: Quantum coherence time before natural collapse
            collapse_threshold: Probability threshold for forced collapse
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_interpretations = num_interpretations
        self.coherence_time = coherence_time
        self.collapse_threshold = collapse_threshold
        
        # Classical components
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # Quantum components
        self.quantum_circuit = HybridQuantumAICircuit(
            num_qubits=int(np.ceil(np.log2(num_interpretations))),
            geometry_params={'throat_radius': 1.0, 'mass': 1.0}
        )
        
        # Entanglement dynamics for interpretation correlations
        self.entanglement = EntanglementDynamics(
            num_qubits=int(np.ceil(np.log2(num_interpretations)))
        )
        
        # Learnable quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(num_interpretations, 2) * np.pi
        )
        
        # Interpretation collapse detector
        self.collapse_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def create_superposition_state(self, 
                                  query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor) -> AttentionState:
        """Create quantum superposition of attention interpretations.
        
        Args:
            query, key, value: Standard attention inputs
            
        Returns:
            AttentionState with superposed interpretations
        """
        batch_size, seq_len, _ = query.shape
        
        # Classical attention weights for each interpretation
        interpretations = []
        quantum_amplitudes = []
        
        for i in range(self.num_interpretations):
            # Parameterized interpretation
            theta, phi = self.quantum_params[i]
            
            # Modified attention computation for this interpretation
            q_i = self.query_projection(query) * torch.cos(theta)
            k_i = self.key_projection(key) * torch.sin(theta) * torch.exp(1j * phi)
            
            # Compute attention weights for this interpretation
            weights_i = torch.softmax(
                torch.matmul(q_i, k_i.transpose(-2, -1)) / np.sqrt(self.d_model),
                dim=-1
            )
            
            # Create interpretation dictionary
            interpretation = {
                'weights': weights_i,
                'theta': theta.item(),
                'phi': phi.item(),
                'confidence': torch.mean(torch.max(weights_i, dim=-1)[0]).item()
            }
            interpretations.append(interpretation)
            
            # Quantum amplitude based on confidence and parameters
            amplitude = np.sqrt(interpretation['confidence']) * np.exp(1j * phi.item())
            quantum_amplitudes.append(amplitude)
        
        # Normalize quantum amplitudes
        quantum_amplitudes = np.array(quantum_amplitudes)
        quantum_amplitudes /= np.linalg.norm(quantum_amplitudes)
        
        # Create quantum state in computational basis
        quantum_state = qt.Qobj(quantum_amplitudes[:2**self.quantum_circuit.num_qubits])
        
        # Combine classical weights as superposition
        classical_weights = torch.stack([interp['weights'] for interp in interpretations])
        classical_weights = torch.sum(
            classical_weights * torch.tensor([abs(amp)**2 for amp in quantum_amplitudes]).unsqueeze(-1).unsqueeze(-1),
            dim=0
        )
        
        return AttentionState(
            quantum_state=quantum_state,
            classical_weights=classical_weights,
            coherence_time=self.coherence_time,
            collapse_threshold=self.collapse_threshold,
            interpretations=interpretations
        )
    
    def check_collapse_condition(self, attention_state: AttentionState) -> bool:
        """Check if quantum attention should collapse to single interpretation.
        
        Args:
            attention_state: Current attention state
            
        Returns:
            True if collapse should occur
        """
        # Check if any interpretation dominates
        confidences = torch.tensor([interp['confidence'] for interp in attention_state.interpretations])
        max_confidence = torch.max(confidences)
        
        if max_confidence > self.collapse_threshold:
            return True
        
        # Check for contradictory interpretations
        for i, interp_i in enumerate(attention_state.interpretations):
            for j, interp_j in enumerate(attention_state.interpretations[i+1:], i+1):
                # Compute overlap between attention patterns
                overlap = torch.sum(interp_i['weights'] * interp_j['weights'])
                if overlap < 0.1:  # Highly contradictory
                    return True
        
        return False
    
    def collapse_attention(self, attention_state: AttentionState) -> Tuple[torch.Tensor, Dict]:
        """Collapse quantum attention to single interpretation.
        
        Args:
            attention_state: Superposed attention state
            
        Returns:
            Collapsed attention weights and metadata
        """
        # Measure quantum state to select interpretation
        probabilities = np.abs(attention_state.quantum_state.data.toarray().flatten())**2
        selected_idx = np.random.choice(len(probabilities), p=probabilities)
        
        if selected_idx < len(attention_state.interpretations):
            selected_interpretation = attention_state.interpretations[selected_idx]
            
            collapse_metadata = {
                'selected_interpretation': selected_idx,
                'collapse_reason': 'quantum_measurement',
                'pre_collapse_entropy': qt.entropy_vn(attention_state.quantum_state),
                'selected_confidence': selected_interpretation['confidence']
            }
            
            return selected_interpretation['weights'], collapse_metadata
        else:
            # Fallback to highest confidence interpretation
            confidences = [interp['confidence'] for interp in attention_state.interpretations]
            best_idx = np.argmax(confidences)
            
            collapse_metadata = {
                'selected_interpretation': best_idx,
                'collapse_reason': 'confidence_fallback',
                'pre_collapse_entropy': qt.entropy_vn(attention_state.quantum_state),
                'selected_confidence': attention_state.interpretations[best_idx]['confidence']
            }
            
            return attention_state.interpretations[best_idx]['weights'], collapse_metadata
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                force_collapse: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with quantum superposition attention.
        
        Args:
            query, key, value: Standard attention inputs
            force_collapse: Whether to force collapse to single interpretation
            
        Returns:
            Attention output and metadata
        """
        # Create superposition state
        attention_state = self.create_superposition_state(query, key, value)
        
        # Check if collapse should occur
        should_collapse = force_collapse or self.check_collapse_condition(attention_state)
        
        if should_collapse:
            # Collapse to single interpretation
            attention_weights, metadata = self.collapse_attention(attention_state)
            metadata['superposition_maintained'] = False
        else:
            # Maintain superposition - use weighted combination
            attention_weights = attention_state.classical_weights
            metadata = {
                'superposition_maintained': True,
                'num_active_interpretations': len(attention_state.interpretations),
                'quantum_entropy': qt.entropy_vn(attention_state.quantum_state),
                'coherence_time_remaining': self.coherence_time
            }
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, metadata


class MultiHeadQuantumAttention(nn.Module):
    """Multi-head quantum attention with superposition capabilities."""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 num_interpretations: int = 4,
                 coherence_time: float = 1e-3):
        """Initialize multi-head quantum attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_interpretations: Interpretations per head
            coherence_time: Quantum coherence time
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Create quantum attention heads
        self.quantum_heads = nn.ModuleList([
            QuantumAttentionHead(
                d_model=self.head_dim,
                num_interpretations=num_interpretations,
                coherence_time=coherence_time
            ) for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Global collapse detector
        self.global_collapse_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                force_collapse: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Multi-head quantum attention forward pass.
        
        Args:
            query, key, value: Input tensors [batch, seq, d_model]
            force_collapse: Force all heads to collapse
            
        Returns:
            Attention output and aggregated metadata
        """
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        head_outputs = []
        head_metadata = []
        
        # Process each head
        for i, head in enumerate(self.quantum_heads):
            head_output, metadata = head(
                q[:, :, i, :],
                k[:, :, i, :],
                v[:, :, i, :],
                force_collapse=force_collapse
            )
            head_outputs.append(head_output)
            head_metadata.append(metadata)
        
        # Concatenate head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)
        
        # Apply output projection
        output = self.output_projection(multi_head_output)
        
        # Aggregate metadata
        aggregated_metadata = {
            'heads_in_superposition': sum(1 for m in head_metadata if m.get('superposition_maintained', False)),
            'total_heads': len(head_metadata),
            'average_entropy': np.mean([m.get('quantum_entropy', 0) for m in head_metadata]),
            'head_metadata': head_metadata
        }
        
        return output, aggregated_metadata


if __name__ == "__main__":
    # Test quantum attention mechanism
    print("Testing Quantum Superposition Attention...")
    
    # Create test inputs
    batch_size, seq_len, d_model = 2, 10, 256
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize quantum attention
    qa = MultiHeadQuantumAttention(d_model=d_model, num_heads=8)
    
    # Test forward pass maintaining superposition
    print("Testing superposition mode...")
    output, metadata = qa(query, key, value, force_collapse=False)
    print(f"Heads in superposition: {metadata['heads_in_superposition']}/{metadata['total_heads']}")
    print(f"Average quantum entropy: {metadata['average_entropy']:.4f}")
    
    # Test forced collapse
    print("Testing collapse mode...")
    output, metadata = qa(query, key, value, force_collapse=True)
    print(f"Heads in superposition: {metadata['heads_in_superposition']}/{metadata['total_heads']}")
    print(f"Average quantum entropy: {metadata['average_entropy']:.4f}")
    
    print("Quantum attention test completed!")
