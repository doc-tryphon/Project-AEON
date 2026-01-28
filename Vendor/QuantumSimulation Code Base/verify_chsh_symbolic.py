"""
Symbolic derivation of CHSH inequality using SymPy.

We'll derive the correct measurement angles and verify that S = 2√2 for Bell states.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("CHSH Inequality Symbolic Derivation")
print("=" * 60)

# Define Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
print("\nBell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
print("In computational basis: [1/√2, 0, 0, 1/√2]")

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

print("\nPauli matrices:")
print("σₓ =", sigma_x)
print("σᵧ =", sigma_y)
print("σᵤ =", sigma_z)

# Bell state
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

print("\n" + "=" * 60)
print("Testing correlation function E(a, b)")
print("=" * 60)

def measure_correlation(state, theta_a, theta_b):
    """
    Measure correlation E(a,b) = ⟨ψ|(σₐ ⊗ σᵦ)|ψ⟩
    where σₐ = cos(a)σₓ + sin(a)σᵤ
    """
    sigma_a = np.cos(theta_a) * sigma_x + np.sin(theta_a) * sigma_z
    sigma_b = np.cos(theta_b) * sigma_x + np.sin(theta_b) * sigma_z

    operator = np.kron(sigma_a, sigma_b)
    expectation = state.conj() @ operator @ state
    return expectation.real

# For |Φ+⟩, the correlation should be: E(a,b) = cos(a-b)
print("\nFor |Φ+⟩, analytical result: E(a,b) = cos(a-b)")
print("\nVerifying this:")

test_angles = [
    (0, 0),           # cos(0) = 1
    (0, np.pi/4),     # cos(-π/4) = 1/√2
    (np.pi/4, 0),     # cos(π/4) = 1/√2
    (np.pi/2, 0),     # cos(π/2) = 0
    (0, np.pi/2),     # cos(-π/2) = 0
    (np.pi/4, np.pi/4), # cos(0) = 1
]

for a, b in test_angles:
    E_ab = measure_correlation(phi_plus, a, b)
    expected = np.cos(a - b)
    error = abs(E_ab - expected)
    print(f"E({a:.3f}, {b:.3f}) = {E_ab:.6f}, cos({a-b:.3f}) = {expected:.6f}, error = {error:.2e}")

print("\n" + "=" * 60)
print("CHSH Inequality: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')")
print("=" * 60)

print("\nClassical bound: |S| ≤ 2")
print("Quantum (Tsirelson) bound: |S| ≤ 2√2 ≈ 2.828")

print("\n--- Testing different angle combinations ---")

# Standard CHSH angles from literature
angle_sets = [
    {
        'name': 'Standard 1: a=0, a\'=π/2, b=π/4, b\'=-π/4',
        'a': 0, 'a_prime': np.pi/2, 'b': np.pi/4, 'b_prime': -np.pi/4
    },
    {
        'name': 'Standard 2: a=0, a\'=π/4, b=π/8, b\'=-π/8',
        'a': 0, 'a_prime': np.pi/4, 'b': np.pi/8, 'b_prime': -np.pi/8
    },
    {
        'name': 'Optimal: a=0, a\'=π/2, b=π/4, b\'=3π/4',
        'a': 0, 'a_prime': np.pi/2, 'b': np.pi/4, 'b_prime': 3*np.pi/4
    },
]

for angle_set in angle_sets:
    print(f"\n{angle_set['name']}")
    a = angle_set['a']
    a_prime = angle_set['a_prime']
    b = angle_set['b']
    b_prime = angle_set['b_prime']

    E_ab = measure_correlation(phi_plus, a, b)
    E_ab_prime = measure_correlation(phi_plus, a, b_prime)
    E_a_prime_b = measure_correlation(phi_plus, a_prime, b)
    E_a_prime_b_prime = measure_correlation(phi_plus, a_prime, b_prime)

    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

    print(f"  E(a,b) = {E_ab:.6f}")
    print(f"  E(a,b') = {E_ab_prime:.6f}")
    print(f"  E(a',b) = {E_a_prime_b:.6f}")
    print(f"  E(a',b') = {E_a_prime_b_prime:.6f}")
    print(f"  S = {S:.6f}")
    print(f"  Violates classical? {abs(S) > 2}")
    print(f"  Achieves 2√2? {np.isclose(abs(S), 2*np.sqrt(2), rtol=1e-6)}")

print("\n" + "=" * 60)
print("Analytical Calculation for Optimal Angles")
print("=" * 60)

# For E(a,b) = cos(a-b), let's compute S analytically
print("\nFor a=0, a'=π/2, b=π/4, b'=3π/4:")
print("  E(0, π/4) = cos(-π/4) = 1/√2")
print("  E(0, 3π/4) = cos(-3π/4) = -1/√2")
print("  E(π/2, π/4) = cos(π/4) = 1/√2")
print("  E(π/2, 3π/4) = cos(-π/4) = 1/√2")
print("  S = 1/√2 - (-1/√2) + 1/√2 + 1/√2")
print("    = 1/√2 + 1/√2 + 1/√2 + 1/√2")
print("    = 4/√2 = 2√2 ≈ 2.828")

print(f"\nNumerical: 2√2 = {2*np.sqrt(2):.6f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("\nThe optimal CHSH angles are:")
print("  Alice: a = 0, a' = π/2")
print("  Bob: b = π/4, b' = 3π/4")
print("\nThis gives S = 2√2, achieving maximal quantum violation.")