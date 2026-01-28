"""Test different CHSH angle combinations to find optimal."""
import numpy as np

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# Bell state |Phi+> = (|00> + |11>)/sqrt(2)
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

def measure_E(state, a, b):
    """Measure E(a,b) = <psi|(cos(a)sx + sin(a)sz) x (cos(b)sx + sin(b)sz)|psi>"""
    sa = np.cos(a) * sx + np.sin(a) * sz
    sb = np.cos(b) * sx + np.sin(b) * sz
    op = np.kron(sa, sb)
    return (state.conj() @ op @ state).real

print("Testing CHSH angle combinations:")
print("=" * 50)

# Test angle set 3: a=0, a'=pi/2, b=pi/4, b'=3pi/4
a = 0
a_prime = np.pi/2
b = np.pi/4
b_prime = 3*np.pi/4

E_ab = measure_E(phi_plus, a, b)
E_ab_p = measure_E(phi_plus, a, b_prime)
E_ap_b = measure_E(phi_plus, a_prime, b)
E_ap_bp = measure_E(phi_plus, a_prime, b_prime)

S = E_ab - E_ab_p + E_ap_b + E_ap_bp

print(f"\nAngle set: a=0, a'=pi/2, b=pi/4, b'=3pi/4")
print(f"E(a,b) = {E_ab:.6f}")
print(f"E(a,b') = {E_ab_p:.6f}")
print(f"E(a',b) = {E_ap_b:.6f}")
print(f"E(a',b') = {E_ap_bp:.6f}")
print(f"S = {S:.6f}")
print(f"Target: 2*sqrt(2) = {2*np.sqrt(2):.6f}")
print(f"Match? {np.isclose(S, 2*np.sqrt(2))}")