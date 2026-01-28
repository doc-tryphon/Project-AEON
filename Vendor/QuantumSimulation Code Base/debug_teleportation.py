"""Debug teleportation protocol."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from quantum.teleportation import QuantumTeleportation
from quantum.entanglement import BellStateGenerator

# Test with |0> state
protocol = QuantumTeleportation()
message = np.array([1, 0], dtype=complex)

# Get Bell state
bell_gen = BellStateGenerator()
bell_state = bell_gen.create_bell_state('00')  # |Φ+>

print("Message state:", message)
print("Bell state:", bell_state.state_vector)

# Initial 3-qubit state
initial = np.kron(message, bell_state.state_vector)
print("\nInitial 3-qubit state:")
print(initial)
print("Norm:", np.linalg.norm(initial))

# Expected expansion for |0>⊗|Φ+>:
# |0> ⊗ (|00> + |11>)/√2 = (|000> + |011>)/√2
expected = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=complex) / np.sqrt(2)
print("\nExpected:")
print(expected)
print("Match?", np.allclose(initial, expected))

# Now rewrite in Bell basis
# |000> = (|Φ+>|0> + |Φ->|0> + |Ψ+>|0> + |Ψ->|0>)/2 ...
# This is complex - let me trace through the measurement

print("\n" + "="*50)
print("Bell Measurement Simulation")
print("="*50)

# Reshape to Alice (4D) and Bob (2D)
psi_reshaped = initial.reshape(4, 2)
print("\nReshaped state (Alice x Bob):")
print(psi_reshaped)

# Bell basis for Alice's qubits
ket_00 = np.array([1, 0, 0, 0], dtype=complex)
ket_01 = np.array([0, 1, 0, 0], dtype=complex)
ket_10 = np.array([0, 0, 1, 0], dtype=complex)
ket_11 = np.array([0, 0, 0, 1], dtype=complex)

phi_plus = (ket_00 + ket_11) / np.sqrt(2)
phi_minus = (ket_00 - ket_11) / np.sqrt(2)
psi_plus = (ket_01 + ket_10) / np.sqrt(2)
psi_minus = (ket_01 - ket_10) / np.sqrt(2)

bell_basis = [phi_plus, phi_minus, psi_plus, psi_minus]
names = ['Phi+', 'Phi-', 'Psi+', 'Psi-']

print("\nProjecting onto Bell states:")
for i, (bell, name) in enumerate(zip(bell_basis, names)):
    projection = np.dot(bell.conj(), psi_reshaped)
    prob = np.sum(np.abs(projection)**2)
    print(f"\n{name}:")
    print(f"  Projection: {projection}")
    print(f"  Probability: {prob:.6f}")
    if prob > 1e-10:
        bob_state = projection / np.sqrt(prob)
        print(f"  Bob's normalized state: {bob_state}")
        print(f"  Bob's norm: {np.linalg.norm(bob_state):.6f}")