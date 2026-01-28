"""Debug CHSH measurement."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from quantum.entanglement import BellStateGenerator, BellMeasurement

# Create Bell state
gen = BellStateGenerator()
phi_plus = gen.create_bell_state('00')

print(f"Phi+ state vector: {phi_plus.state_vector}")
print()

# Create measurement apparatus
meas = BellMeasurement()

# Test angles
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

print("Testing correlations for Phi+:")
print("For Phi+, E(a,b) should equal cos(a-b)\n")

for a in [0, np.pi/2]:
    for b in [np.pi/4, -np.pi/4]:
        correlation = meas.measure_correlation(phi_plus, a, b)
        expected = np.cos(a - b)
        print(f"E({a:.3f}, {b:.3f}) = {correlation:.6f}, expected cos({a-b:.3f}) = {expected:.6f}")

print("\nCHSH test:")
result = meas.chsh_inequality_test(phi_plus)
print(f"S = {result['S']}")
print(f"E_ab = {result['E_ab']}")
print(f"E_ab_prime = {result['E_ab_prime']}")
print(f"E_a_prime_b = {result['E_a_prime_b']}")
print(f"E_a_prime_b_prime = {result['E_a_prime_b_prime']}")

print(f"\nExpected S = 2√2 = {2 * np.sqrt(2):.6f}")

# Manual calculation
print("\n--- Manual Calculation ---")
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

psi = phi_plus.state_vector
print(f"State: {psi}")

# Test simple correlation: σₓ ⊗ σₓ
op = np.kron(sigma_x, sigma_x)
exp_val = psi.conj() @ op @ psi
print(f"\n<Phi+|sigma_x x sigma_x|Phi+> = {exp_val.real:.6f} (should be 1.0)")

# Test sigma_z x sigma_z
op = np.kron(sigma_z, sigma_z)
exp_val = psi.conj() @ op @ psi
print(f"<Phi+|sigma_z x sigma_z|Phi+> = {exp_val.real:.6f} (should be 1.0)")