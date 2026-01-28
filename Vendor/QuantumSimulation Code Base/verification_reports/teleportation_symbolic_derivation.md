# Quantum Teleportation: Symbolic Derivation

**Date:** 2025-09-29
**Status:** In Progress
**Reference:** Bennett et al., Phys. Rev. Lett. 70, 1895 (1993)

## Protocol Overview

Quantum teleportation transfers an unknown quantum state |ψ⟩ from Alice to Bob using:
1. Shared entanglement (Bell state)
2. Classical communication (2 bits)
3. Local operations at Bob's site

**Key Result:** Perfect state transfer with F = 1 (ideal case)

## Step 1: Initial State

Alice wants to teleport: |ψ⟩₁ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1

Alice and Bob share entangled pair: |Φ+⟩₂₃ = (|00⟩ + |11⟩)/√2

**Total initial state:**
```
|Ψ_init⟩ = |ψ⟩₁ ⊗ |Φ+⟩₂₃
         = (α|0⟩₁ + β|1⟩₁) ⊗ (|00⟩₂₃ + |11⟩₂₃)/√2
         = 1/√2 [α|0⟩₁(|00⟩₂₃ + |11⟩₂₃) + β|1⟩₁(|00⟩₂₃ + |11⟩₂₃)]
         = 1/√2 [α|000⟩ + α|011⟩ + β|100⟩ + β|111⟩]
```

## Step 2: Rewrite in Bell Basis

The Bell basis for qubits 1-2:
- |Φ+⟩₁₂ = (|00⟩ + |11⟩)/√2
- |Φ-⟩₁₂ = (|00⟩ - |11⟩)/√2
- |Ψ+⟩₁₂ = (|01⟩ + |10⟩)/√2
- |Ψ-⟩₁₂ = (|01⟩ - |10⟩)/√2

**Express computational basis in Bell basis:**
```
|00⟩₁₂ = 1/√2 (|Φ+⟩₁₂ + |Φ-⟩₁₂)
|01⟩₁₂ = 1/√2 (|Ψ+⟩₁₂ + |Ψ-⟩₁₂)
|10⟩₁₂ = 1/√2 (|Ψ+⟩₁₂ - |Ψ-⟩₁₂)
|11⟩₁₂ = 1/√2 (|Φ+⟩₁₂ - |Φ-⟩₁₂)
```

**Substitute into initial state:**
```
|Ψ_init⟩ = 1/√2 [α|000⟩ + α|011⟩ + β|100⟩ + β|111⟩]
```

Expanding |00⟩₁₂, |01⟩₁₂, |10⟩₁₂, |11⟩₁₂ in Bell basis:

```
|Ψ_init⟩ = 1/2 [
    |Φ+⟩₁₂ ⊗ (α|0⟩₃ + β|1⟩₃) +
    |Φ-⟩₁₂ ⊗ (α|0⟩₃ - β|1⟩₃) +
    |Ψ+⟩₁₂ ⊗ (α|1⟩₃ + β|0⟩₃) +
    |Ψ-⟩₁₂ ⊗ (α|1⟩₃ - β|0⟩₃)
]
```

## Step 3: Measurement Outcomes and Bob's States

After Alice's Bell measurement, the system collapses to one of four outcomes:

| Alice's Result | Probability | Bob's State | Correction |
|----------------|-------------|-------------|------------|
| \|Φ+⟩₁₂ (00)  | 1/4 | α\|0⟩ + β\|1⟩ | I (identity) |
| \|Φ-⟩₁₂ (01)  | 1/4 | α\|0⟩ - β\|1⟩ | Z |
| \|Ψ+⟩₁₂ (10)  | 1/4 | α\|1⟩ + β\|0⟩ | X |
| \|Ψ-⟩₁₂ (11)  | 1/4 | α\|1⟩ - β\|0⟩ | XZ |

## Step 4: Correction Operations

Bob applies correction based on Alice's 2 classical bits:

**Pauli Corrections:**
- I = [[1, 0], [0, 1]]     - No change
- Z = [[1, 0], [0, -1]]    - Phase flip
- X = [[0, 1], [1, 0]]     - Bit flip
- XZ = [[0, -1], [1, 0]]   - Both flips

**Verification of corrections:**

```
I (α|0⟩ + β|1⟩) = α|0⟩ + β|1⟩  ✓
Z (α|0⟩ - β|1⟩) = α|0⟩ + β|1⟩  ✓
X (α|1⟩ + β|0⟩) = α|0⟩ + β|1⟩  ✓
XZ (α|1⟩ - β|0⟩) = α|0⟩ + β|1⟩  ✓
```

## Step 5: Fidelity Calculation

State fidelity: F = |⟨ψ_target|ψ_final⟩|²

For perfect teleportation:
```
|ψ_final⟩ = α|0⟩ + β|1⟩ = |ψ_target⟩
F = |⟨ψ_target|ψ_target⟩|² = 1
```

## Step 6: No-Signaling Verification

**Before Alice's measurement result arrives, Bob's reduced state is:**

ρ_Bob = Tr₁₂(|Ψ_init⟩⟨Ψ_init|)

Computing the partial trace:
```
ρ_Bob = 1/2 (|0⟩⟨0| + |1⟩⟨1|) = I/2
```

This is maximally mixed - Bob has no information about |ψ⟩ until he receives Alice's measurement result. This verifies the no-signaling theorem.

## Mathematical Properties to Verify

1. **Unitarity of gates:** U†U = I for all correction operators
2. **Probability conservation:** Σ P(i) = 1 (all outcomes sum to 1)
3. **Fidelity preservation:** F = 1 for all input states
4. **No-signaling:** Bob's reduced density matrix is maximally mixed
5. **Linearity:** Protocol works for any superposition

## Next: SymPy Verification

We will verify all of the above symbolically using SymPy MCP to ensure:
- Bell basis transformations are correct
- Correction operators restore the original state
- Fidelity F = 1 in all cases
- No information leaks before classical communication