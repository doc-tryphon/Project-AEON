"""Test different Bell state resources."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from quantum.teleportation import QuantumTeleportation

protocol = QuantumTeleportation()
test_state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>

print("Testing teleportation with different Bell state resources")
print("Input state:", test_state)
print()

for resource in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
    result = protocol.teleport(test_state, resource)
    print(f"{resource}:")
    print(f"  Measurement: {result.measurement_outcome}")
    print(f"  Correction: {result.correction_applied}")
    print(f"  Bob before correction: {result.bob_state_before_correction}")
    print(f"  Bob after correction: {result.output_state}")
    print(f"  Fidelity: {result.fidelity:.6f}")
    print(f"  Success: {result.protocol_successful}")
    print()