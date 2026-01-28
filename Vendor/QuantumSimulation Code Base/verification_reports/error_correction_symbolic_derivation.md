# 3-Qubit Bit Flip Code - Symbolic Derivation

**Date**: 2025-09-30
**Status**: In Progress
**References**:
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000), Ch. 10
- Preskill, "Quantum Computation" Lecture Notes, Ch. 7
- Shor, P. W. (1995). "Scheme for reducing decoherence in quantum computer memory"

## Motivation: Why Quantum Error Correction?

**Classical Error Correction**: Use redundancy (repetition codes, parity checks)

**Quantum Challenges**:
1. **No-cloning theorem**: Cannot copy |ψ⟩ to create |ψ⟩|ψ⟩|ψ⟩
2. **Measurement collapses**: Cannot measure |ψ⟩ directly to check for errors
3. **Continuous errors**: Errors are not discrete (α|0⟩ + β|1⟩ can be rotated by any angle)

**Quantum Solution**:
- Use **entanglement** instead of copying
- **Syndrome measurement**: Extract error information without measuring logical state
- **Stabilizer formalism**: Continuous errors discretized by measurement

## Part 1: The 3-Qubit Bit Flip Code

### Logical Basis States

**Encoding**: Map 1 logical qubit to 3 physical qubits

```
|0⟩_L = |000⟩  (logical zero)
|1⟩_L = |111⟩  (logical one)
```

**General logical state**:
```
|ψ⟩_L = α|0⟩_L + β|1⟩_L
      = α|000⟩ + β|111⟩
```

**Properties**:
- Superposition preserved: coefficients α, β unchanged
- No cloning: We don't have |ψ⟩|ψ⟩|ψ⟩, we have entangled state
- Redundancy: Majority vote can detect/correct single bit flip

### Code Space

**Code Space** C: 2-dimensional subspace of 8-dimensional Hilbert space

```
C = span{|000⟩, |111⟩} ⊂ ℂ⁸
```

**Orthonormal basis**:
```
⟨000|111⟩ = 0  ✓
⟨000|000⟩ = 1  ✓
⟨111|111⟩ = 1  ✓
```

## Part 2: Multi-Qubit Gates

### CNOT Gate (Controlled-NOT)

**Definition**: Flips target qubit if control qubit is |1⟩

**Matrix** (control = qubit 0, target = qubit 1):
```
CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X

     = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]]
```

**Action**:
```
CNOT|00⟩ = |00⟩
CNOT|01⟩ = |01⟩
CNOT|10⟩ = |11⟩  (flip target)
CNOT|11⟩ = |10⟩  (flip target)
```

**Properties**:
```
CNOT† = CNOT  (self-adjoint)
CNOT² = I     (involutory)
```

### Toffoli Gate (CCNOT)

**Definition**: Flips target if both controls are |1⟩

**Matrix**:
```
Toffoli = |00⟩⟨00| ⊗ I + |01⟩⟨01| ⊗ I + |10⟩⟨10| ⊗ I + |11⟩⟨11| ⊗ X

8×8 matrix with X gate acting on target only when both controls = 1
```

**Action**:
```
Toffoli|110⟩ = |111⟩  (both controls = 1, flip target)
Toffoli|010⟩ = |010⟩  (control 0 = 0, no flip)
Toffoli|100⟩ = |100⟩  (control 1 = 0, no flip)
```

**Universal gate**: Toffoli + Hadamard = universal quantum computation

## Part 3: Encoding Circuit

### Circuit Diagram (ASCII)

```
Input:  |ψ⟩ = α|0⟩ + β|1⟩  (logical qubit)
        |0⟩                 (ancilla)
        |0⟩                 (ancilla)

        |ψ⟩ ──●────●──  (control for both CNOTs)
              │    │
        |0⟩ ──⊕────│──  (target of CNOT₁₂)
                   │
        |0⟩ ───────⊕──  (target of CNOT₁₃)

Output: |ψ⟩_L = α|000⟩ + β|111⟩
```

### Symbolic Derivation

**Initial state**: |ψ,0,0⟩ = (α|0⟩ + β|1⟩) ⊗ |0⟩ ⊗ |0⟩ = α|000⟩ + β|100⟩

**After CNOT₁₂** (control = qubit 0, target = qubit 1):
```
CNOT₁₂(α|000⟩ + β|100⟩) = α|000⟩ + β|110⟩
```

**After CNOT₁₃** (control = qubit 0, target = qubit 2):
```
CNOT₁₃(α|000⟩ + β|110⟩) = α|000⟩ + β|111⟩ ✓
```

**Result**: Successfully encodes |ψ⟩ → |ψ⟩_L without cloning!

### Matrix Form

**Encoding operator** U_enc:
```
U_enc = CNOT₁₃ · CNOT₁₂

U_enc |ψ,0,0⟩ = |ψ⟩_L
```

**Verification**:
```
U_enc|000⟩ = |000⟩ ✓
U_enc|100⟩ = |111⟩ ✓
```

## Part 4: Error Models

### Single Bit Flip Errors

**Error operators** {E₀, E₁, E₂, E₃}:

```
E₀ = I ⊗ I ⊗ I     (no error)
E₁ = X ⊗ I ⊗ I     (flip qubit 0)
E₂ = I ⊗ X ⊗ I     (flip qubit 1)
E₃ = I ⊗ I ⊗ X     (flip qubit 2)
```

**Error channel** (single bit flip with probability p):
```
ε(ρ) = (1-p)³ E₀ ρ E₀† + p(1-p)² (E₁ ρ E₁† + E₂ ρ E₂† + E₃ ρ E₃†) + O(p²)
```

For small p, dominant terms are no error and single bit flips.

### Effect on Encoded States

**No error**: |000⟩ → |000⟩, |111⟩ → |111⟩

**Error on qubit 0**:
```
E₁|000⟩ = X₀|000⟩ = |100⟩
E₁|111⟩ = X₀|111⟩ = |011⟩
```

**Error on qubit 1**:
```
E₂|000⟩ = X₁|000⟩ = |010⟩
E₂|111⟩ = X₁|111⟩ = |101⟩
```

**Error on qubit 2**:
```
E₃|000⟩ = X₂|000⟩ = |001⟩
E₃|111⟩ = X₂|111⟩ = |110⟩
```

**Key observation**: All error states orthogonal to code space and to each other!

## Part 5: Syndrome Measurement

### Stabilizer Operators

**Stabilizers**: Operators that fix code space

```
S₁ = Z₀Z₁  (parity of qubits 0 and 1)
S₂ = Z₁Z₂  (parity of qubits 1 and 2)
```

**Code space stabilized**:
```
S₁|000⟩ = Z₀Z₁|000⟩ = |000⟩  (eigenvalue +1)
S₁|111⟩ = Z₀Z₁|111⟩ = |111⟩  (eigenvalue +1)

S₂|000⟩ = +|000⟩
S₂|111⟩ = +|111⟩
```

**Error states NOT stabilized**:
```
S₁|100⟩ = Z₀Z₁|100⟩ = -|100⟩  (eigenvalue -1)
S₁|010⟩ = Z₀Z₁|010⟩ = -|010⟩  (eigenvalue -1)
S₁|001⟩ = Z₀Z₁|001⟩ = +|001⟩  (eigenvalue +1)
```

### Syndrome Table

Measure eigenvalues (s₁, s₂) of (S₁, S₂):

| Error | State after | S₁ = Z₀Z₁ | S₂ = Z₁Z₂ | Syndrome (s₁,s₂) |
|-------|-------------|-----------|-----------|------------------|
| E₀ (none) | \|000⟩/\|111⟩ | +1 | +1 | (0, 0) |
| E₁ (X₀) | \|100⟩/\|011⟩ | -1 | +1 | (1, 0) |
| E₂ (X₁) | \|010⟩/\|101⟩ | -1 | -1 | (1, 1) |
| E₃ (X₂) | \|001⟩/\|110⟩ | +1 | -1 | (0, 1) |

**Convention**: Encode +1 → 0, -1 → 1 for syndrome bits

### Non-Demolition Property

**Key property**: Syndrome measurement does NOT collapse logical state!

**Proof**:
```
|ψ⟩_L = α|000⟩ + β|111⟩

After error E₁:
|ψ_error⟩ = α|100⟩ + β|011⟩

Syndrome measurement projects onto S₁ = -1, S₂ = +1:
Result: |ψ_error⟩ (unchanged!) + syndrome (1, 0)

Coefficients α, β still unknown! Logical state preserved.
```

**Mathematics**: S₁ and S₂ commute with logical operators (Z_L, X_L), so measuring stabilizers doesn't reveal logical state.

## Part 6: Correction Circuit

### Correction Map

**Syndrome** → **Recovery operation**:

| Syndrome (s₁,s₂) | Error detected | Correction R |
|------------------|----------------|--------------|
| (0, 0) | None | I (do nothing) |
| (1, 0) | Qubit 0 flipped | X₀ |
| (1, 1) | Qubit 1 flipped | X₁ |
| (0, 1) | Qubit 2 flipped | X₂ |

**Recovery operator** R_s based on syndrome s:
```
R_(0,0) = I ⊗ I ⊗ I
R_(1,0) = X ⊗ I ⊗ I
R_(1,1) = I ⊗ X ⊗ I
R_(0,1) = I ⊗ I ⊗ X
```

### Correction Verification

**No error**:
```
R_(0,0) |000⟩ = |000⟩ ✓
R_(0,0) |111⟩ = |111⟩ ✓
```

**Error on qubit 0**:
```
R_(1,0) |100⟩ = X₀|100⟩ = |000⟩ ✓
R_(1,0) |011⟩ = X₀|011⟩ = |111⟩ ✓
```

**Error on qubit 1**:
```
R_(1,1) |010⟩ = X₁|010⟩ = |000⟩ ✓
R_(1,1) |101⟩ = X₁|101⟩ = |111⟩ ✓
```

**Error on qubit 2**:
```
R_(0,1) |001⟩ = X₂|001⟩ = |000⟩ ✓
R_(0,1) |110⟩ = X₂|110⟩ = |111⟩ ✓
```

**Result**: Perfect correction for all single bit flip errors!

## Part 7: Decoding Circuit

### Decoding = Reverse Encoding

**Decoding operator**: U_dec = U_enc† = (CNOT₁₃ · CNOT₁₂)†

Since CNOT† = CNOT:
```
U_dec = CNOT₁₂ · CNOT₁₃  (reverse order)
```

**Effect**:
```
U_dec |000⟩ = |000⟩ → extract |0⟩ from qubit 0
U_dec |111⟩ = |100⟩ → extract |1⟩ from qubit 0
```

**Full cycle verification**:
```
U_dec · U_enc |ψ,0,0⟩ = |ψ,0,0⟩ ✓
```

## Part 8: Logical Error Rate Analysis

### Physical vs Logical Error Rates

**Physical error rate**: p (per qubit)

**Logical error**: Uncorrectable error (2 or 3 bit flips)

**Error probabilities** (independent errors):
```
P(0 errors) = (1-p)³
P(1 error)  = 3p(1-p)²     (correctable)
P(2 errors) = 3p²(1-p)     (NOT correctable)
P(3 errors) = p³           (NOT correctable)
```

**Logical error rate**:
```
p_L = P(2 errors) + P(3 errors)
    = 3p²(1-p) + p³
    = 3p² - 3p³ + p³
    = 3p² - 2p³
    ≈ 3p²  (for small p)
```

### Break-Even Point

**Question**: When does error correction help?

**Condition**: p_L < p (logical error rate < physical error rate)

```
3p² - 2p³ < p
3p - 2p² < 1
p < 1/3
```

**Result**: Code helps when physical error rate p < 33%

### Improvement Factor

For p = 0.1:
```
p_L = 3(0.1)² - 2(0.1)³ = 0.03 - 0.002 = 0.028

Improvement: p/p_L = 0.1/0.028 ≈ 3.6× better!
```

For p = 0.01:
```
p_L ≈ 3(0.01)² = 0.0003

Improvement: p/p_L = 0.01/0.0003 ≈ 33× better!
```

## Part 9: Limitations

### What This Code Does NOT Protect Against

**1. Phase Flip Errors** (Z errors):
```
Z|+⟩ = Z(|0⟩ + |1⟩)/√2 = (|0⟩ - |1⟩)/√2 = |-⟩
```
Our code uses Z as stabilizer, so phase flips commute with error detection!

**2. Multiple Simultaneous Errors**:
- 2 bit flips: Misdiagnosed (appears as different single error)
- 3 bit flips: Flips logical qubit

**3. Measurement Errors**:
- Assumes syndrome measurement is perfect
- In reality, measurement errors require fault-tolerant syndrome extraction

**4. Overhead**:
- 3× qubit overhead (3 physical qubits per logical qubit)
- 2 syndrome measurements per error correction cycle

### Extensions

**Shor 9-Qubit Code**: Protects against both X and Z errors
**Steane 7-Qubit Code**: [[7,1,3]] code, more efficient
**Surface Codes**: Topological codes, scale to large systems

## Summary

✓ **Encoding**: |ψ⟩ → α|000⟩ + β|111⟩ (no cloning!)
✓ **Syndrome measurement**: Detects error location without measuring logical state
✓ **Correction**: Applies X gate to error qubit based on syndrome
✓ **Decoding**: Extracts logical qubit
✓ **Performance**: p_L ≈ 3p² << p for small p
✓ **Limitation**: Only protects against bit flips, requires p < 1/3

This is the foundation for all quantum error correction!