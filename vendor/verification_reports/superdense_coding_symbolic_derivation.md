# Superdense Coding - Symbolic Derivation

**Date**: 2025-09-30
**Status**: In Progress
**Reference**: Bennett & Wiesner (1992), "Communication via one- and two-particle operators on Einstein-Podolsky-Rosen states"

## Protocol Overview

**Goal**: Alice sends 2 classical bits to Bob using only 1 quantum bit, exploiting shared entanglement.

**Resources**:
- Shared entangled Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2
- Alice has qubit A, Bob has qubit B
- Classical bit string b₁b₂ ∈ {00, 01, 10, 11} to transmit

**Protocol Steps**:
1. Alice applies unitary Uᵦ₁ᵦ₂ to her qubit based on classical bits
2. Alice sends her qubit to Bob (1 qubit transmission)
3. Bob performs Bell measurement on both qubits
4. Bob decodes 2 classical bits from measurement outcome

## Part 1: Alice's Encoding Operators

Alice encodes 2 classical bits by applying one of 4 Pauli operators to her qubit:

- **00** → I (Identity)
- **01** → X (Bit flip)
- **10** → Z (Phase flip)
- **11** → XZ (Both flips)

### Symbolic Derivation: Orthogonality of Encoded States

**Initial shared state**: |Φ+⟩ = (|00⟩ + |11⟩)/√2

We will prove that Alice's 4 operations create 4 orthogonal Bell states, which Bob can perfectly distinguish.

## Symbolic Verification Results

**Verification Script**: `verify_superdense_symbolic.py`

### Part 1: Encoded States

Alice's encoding operators transform the shared |Φ+⟩ state as follows:

- **I ⊗ I |Φ+⟩ = |Φ+⟩** = (|00⟩ + |11⟩)/√2
- **X ⊗ I |Φ+⟩ = |Ψ+⟩** = (|01⟩ + |10⟩)/√2
- **Z ⊗ I |Φ+⟩ = |Φ-⟩** = (|00⟩ - |11⟩)/√2
- **XZ ⊗ I |Φ+⟩ = |Ψ-⟩** = (|01⟩ - |10⟩)/√2

**Result**: ✓ All 4 operators map |Φ+⟩ to the 4 Bell basis states

### Part 2: Orthogonality Matrix

Inner products ⟨ψᵢ|ψⱼ⟩:

```
         00(I)   01(X)   10(Z)   11(XZ)
00(I)      1       0       0        0
01(X)      0       1       0        0
10(Z)      0       0       1        0
11(XZ)     0       0       0        1
```

**Result**: ✓ All 4 encoded states are mutually orthogonal

### Part 3: Bob's Measurement Probabilities

For each encoded state, Bob's Bell measurement gives:

**Message 00 (I encoding → |Φ+⟩)**:
- P(|Φ+⟩) = 1, P(|Φ-⟩) = 0, P(|Ψ+⟩) = 0, P(|Ψ-⟩) = 0

**Message 01 (X encoding → |Ψ+⟩)**:
- P(|Φ+⟩) = 0, P(|Φ-⟩) = 0, P(|Ψ+⟩) = 1, P(|Ψ-⟩) = 0

**Message 10 (Z encoding → |Φ-⟩)**:
- P(|Φ+⟩) = 0, P(|Φ-⟩) = 1, P(|Ψ+⟩) = 0, P(|Ψ-⟩) = 0

**Message 11 (XZ encoding → |Ψ-⟩)**:
- P(|Φ+⟩) = 0, P(|Φ-⟩) = 0, P(|Ψ+⟩) = 0, P(|Ψ-⟩) = 1

**Result**: ✓ Bob can perfectly distinguish all 4 encoded states (100% success rate)

### Part 4: Information Capacity

- **Alice's encoding**: 2 classical bits → 1 of 4 orthogonal quantum states
- **Bob's decoding**: Perfect measurement of 4 orthogonal states
- **Quantum communication**: 1 qubit transmitted from Alice to Bob
- **Classical information**: log₂(4) = 2 bits extracted by Bob
- **Information capacity**: **2 bits per qubit**
- **Enhancement**: 2× classical capacity (due to shared entanglement)

**Result**: ✓ Protocol achieves exactly 2 classical bits per transmitted qubit

## Implementation Verification

**Implementation**: `src/quantum/superdense_coding.py`
**Tests**: `tests/verification/test_superdense_coding.py`

### Test Results (27 tests, all passing):

1. **Operator Properties** (6 tests):
   - ✓ All encoding operators are unitary
   - ✓ Pauli matrices are Hermitian
   - ✓ Pauli eigenvalues are ±1
   - ✓ Commutation relations: [X,Z] = -2iY
   - ✓ Encoding map is bijective

2. **Message Transmission** (6 tests):
   - ✓ Message 00: I → |Φ+⟩ → decoded as 00
   - ✓ Message 01: X → |Ψ+⟩ → decoded as 01
   - ✓ Message 10: Z → |Φ-⟩ → decoded as 10
   - ✓ Message 11: XZ → |Ψ-⟩ → decoded as 11
   - ✓ All 4 messages: 100% success rate
   - ✓ All encoded states properly normalized

3. **Bell State Resources** (2 tests):
   - ✓ Standard resource |Φ+⟩: 100% success
   - ✓ Other Bell states: protocol executes correctly

4. **Orthogonality** (2 tests):
   - ✓ All 4 encoded states mutually orthogonal
   - ✓ Encoded states form complete Bell basis

5. **Information Theory** (3 tests):
   - ✓ Channel capacity: 2 bits per qubit
   - ✓ Holevo bound satisfied
   - ✓ Requires exactly 1 Bell pair

6. **Duality with Teleportation** (2 tests):
   - ✓ Duality properly documented
   - ✓ Resource symmetry verified

7. **Edge Cases** (4 tests):
   - ✓ Invalid message bits rejected
   - ✓ Perfect entanglement verified
   - ✓ Measurement outcomes deterministic
   - ✓ Protocol reversible (lossless)

8. **Protocol Properties** (2 tests):
   - ✓ All operations unitary
   - ✓ Communication complexity: 1 qubit

## Mathematical Summary

### Theorem (Superdense Coding)

Given a shared maximally entangled state |Φ+⟩ = (|00⟩ + |11⟩)/√2:

1. Alice can encode 2 classical bits by applying one of 4 local unitaries {I, X, Z, XZ}
2. These operations transform |Φ+⟩ into 4 orthogonal Bell states
3. Bob can perfectly distinguish these states via Bell measurement
4. Information capacity: 2 classical bits per transmitted qubit

**Proof**: Verified symbolically in `verify_superdense_symbolic.py` ✓

### Protocol Duality

**Quantum Teleportation**:
- Input: 1 qubit + 1 Bell pair
- Communication: 2 classical bits
- Output: 1 qubit (fidelity F = 1)

**Superdense Coding**:
- Input: 2 classical bits + 1 Bell pair
- Communication: 1 qubit
- Output: 2 classical bits (error rate = 0)

Both protocols consume exactly 1 Bell pair and achieve optimal information transfer for their respective tasks.

## Conclusion

✓ **All symbolic derivations verified**
✓ **All implementation tests passing (27/27)**
✓ **Information capacity proven: 2 bits per qubit**
✓ **100% success rate achieved in noiseless case**
✓ **Protocol duality with teleportation documented**

The superdense coding protocol has been rigorously verified both symbolically and experimentally.
