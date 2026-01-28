# Quantum Decoherence and Master Equations - Symbolic Derivation

**Date**: 2025-09-30
**Status**: In Progress
**References**:
- Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000), Ch. 8
- Preskill, "Quantum Computation" Lecture Notes, Ch. 3

## Motivation

**Ideal Quantum Systems** (Steps 1-3): Pure states |ψ⟩, unitary evolution U

**Realistic Quantum Systems**: Interact with environment → decoherence → mixed states

**Why This Matters**:
- All real qubits experience decoherence
- Sets fundamental limits on quantum computation
- T1, T2 times are key hardware metrics
- Error correction must overcome decoherence

## Part 1: Density Matrix Formalism

### Pure States

**State Vector**: |ψ⟩ ∈ ℋ (Hilbert space)
**Density Matrix**: ρ = |ψ⟩⟨ψ|

**Properties**:
```
Tr(ρ) = 1           (normalization)
ρ† = ρ              (Hermitian)
ρ² = ρ              (idempotent for pure states)
Tr(ρ²) = 1          (purity = 1)
eigenvalues: {1, 0}  (one eigenvalue = 1, rest = 0)
```

**Example** (qubit in |0⟩):
```
|ψ⟩ = |0⟩ = [1, 0]ᵀ

ρ = |0⟩⟨0| = [1, 0]ᵀ [1, 0] = [[1, 0],
                                 [0, 0]]

Tr(ρ) = 1 + 0 = 1 ✓
Tr(ρ²) = Tr([[1, 0], [0, 0]]) = 1 ✓
```

### Mixed States

**Statistical Ensemble**: {|ψᵢ⟩, pᵢ} where pᵢ is probability of state |ψᵢ⟩

**Density Matrix**: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

**Properties**:
```
Tr(ρ) = 1           (normalization)
ρ† = ρ              (Hermitian)
ρ² ≠ ρ              (NOT idempotent)
Tr(ρ²) < 1          (purity < 1)
all eigenvalues ∈ [0,1]
```

**Example** (maximally mixed qubit):
```
ρ = (1/2)|0⟩⟨0| + (1/2)|1⟩⟨1|
  = (1/2)[[1, 0], [0, 0]] + (1/2)[[0, 0], [0, 1]]
  = [[1/2, 0],
     [0, 1/2]]
  = I/2

Tr(ρ) = 1/2 + 1/2 = 1 ✓
Tr(ρ²) = Tr([[1/4, 0], [0, 1/4]]) = 1/2 < 1 ✓
```

**Purity**: P = Tr(ρ²)
- P = 1: Pure state
- P < 1: Mixed state
- P = 1/d: Maximally mixed (d = dimension)

### Why Density Matrices?

**Problem**: Cannot represent mixed states with state vectors
**Solution**: Density matrices encompass both pure and mixed states

**Operational Interpretation**:
- ρ describes our knowledge about the quantum state
- If we know |ψ⟩ exactly: ρ = |ψ⟩⟨ψ| (pure)
- If uncertain: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| (mixed)

## Part 2: Quantum Channels (Kraus Representation)

### Definition

**Quantum Channel**: ε: ρ → ρ' (maps density matrices to density matrices)

**Kraus Representation**:
```
ε(ρ) = Σₖ Kₖ ρ Kₖ†
```

Where {Kₖ} are **Kraus operators** satisfying:
```
Σₖ Kₖ†Kₖ = I  (completeness relation)
```

### TPCP Properties

A valid quantum channel must be:

**1. Trace Preserving (TP)**:
```
Tr(ε(ρ)) = Tr(ρ) = 1
```

**Proof**:
```
Tr(ε(ρ)) = Tr(Σₖ Kₖ ρ Kₖ†)
         = Σₖ Tr(Kₖ ρ Kₖ†)
         = Σₖ Tr(Kₖ†Kₖ ρ)    (cyclic property)
         = Tr((Σₖ Kₖ†Kₖ) ρ)
         = Tr(I ρ)
         = Tr(ρ) ✓
```

**2. Completely Positive (CP)**:
```
(ε ⊗ I)(ρ) is positive for all ρ
```

Where I is the identity channel on an auxiliary system.

**Physical Meaning**:
- TP: Probability conserved
- CP: No unphysical correlations when system is entangled with auxiliary system

## Part 3: Decoherence Channels

### Bit Flip Channel

**Physical Process**: Random X gate applied with probability p

**Kraus Operators**:
```
K₀ = √(1-p) I
K₁ = √p X
```

**Channel Action**:
```
ε_BF(ρ) = (1-p) ρ + p X ρ X†
```

**Verification**:
```
K₀†K₀ + K₁†K₁ = (1-p)I + p·I = I ✓
```

**Effect on Bloch Vector**: r = (x, y, z)
```
x → (1-2p)x
y → (1-2p)y
z → z  (no change in z-direction)
```

### Phase Flip Channel

**Physical Process**: Random Z gate applied with probability p

**Kraus Operators**:
```
K₀ = √(1-p) I
K₁ = √p Z
```

**Channel Action**:
```
ε_PF(ρ) = (1-p) ρ + p Z ρ Z†
```

**Effect on Bloch Vector**:
```
x → (1-2p)x
y → (1-2p)y
z → z  (coherences decay, population unchanged)
```

### Depolarizing Channel

**Physical Process**: Random Pauli {I, X, Y, Z} with probability p/4 each

**Kraus Operators**:
```
K₀ = √(1-3p/4) I
K₁ = √(p/4) X
K₂ = √(p/4) Y
K₃ = √(p/4) Z
```

**Channel Action**:
```
ε_D(ρ) = (1-p) ρ + (p/3)(X ρ X† + Y ρ Y† + Z ρ Z†)
```

**Alternative Form** (for qubits):
```
ε_D(ρ) = (1-p) ρ + p(I/2)
```

**Effect on Bloch Vector**:
```
r → (1-p)r  (isotropic shrinking toward origin)
```

**Physical Meaning**: State becomes more mixed, approaches maximally mixed state I/2

## Part 4: Lindblad Master Equation

### Motivation

**Open System**: Quantum system S coupled to environment E

**Total Evolution**: Unitary on S ⊗ E
**Reduced Evolution**: Non-unitary on S alone

**Goal**: Find equation for ρ_S(t) without tracking environment

### Lindblad Equation

**General Form**:
```
dρ/dt = -i[H, ρ] + Σᵢ γᵢ(Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
```

Where:
- H: System Hamiltonian (coherent evolution)
- γᵢ: Decay rates (positive)
- Lᵢ: Lindblad operators (jump operators)
- {A, B} = AB + BA (anticommutator)

**Components**:
```
Coherent part:    -i[H, ρ]
Dissipative part: Σᵢ γᵢ(Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
```

### Properties

**1. Trace Preservation**:
```
d/dt Tr(ρ) = 0
```

**Proof**:
```
d/dt Tr(ρ) = Tr(-i[H,ρ]) + Σᵢ γᵢ Tr(Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
           = 0 + Σᵢ γᵢ(Tr(Lᵢ ρ Lᵢ†) - ½Tr(Lᵢ†Lᵢ ρ) - ½Tr(ρ Lᵢ†Lᵢ))
           = Σᵢ γᵢ(Tr(Lᵢ†Lᵢ ρ) - Tr(Lᵢ†Lᵢ ρ))
           = 0 ✓
```

**2. Complete Positivity**: Guaranteed by Lindblad form

**3. Hermiticity Preservation**: ρ†(t) = ρ(t) for all t

### Physical Interpretation

**Lindblad Operators**: Represent different decay processes
- L = σ₋ = |0⟩⟨1|: Energy relaxation (|1⟩ → |0⟩)
- L = σ₊ = |1⟩⟨0|: Energy excitation
- L = σz: Dephasing

**Decay Rates** γᵢ: Inverse timescales
- γ₁ = 1/T₁: Energy relaxation rate
- γ₂ = 1/T₂: Dephasing rate

## Part 5: T1 and T2 Relaxation

### Amplitude Damping (T1)

**Physical Process**: Energy relaxation |1⟩ → |0⟩ (spontaneous emission)

**Lindblad Operator**: L = √γ₁ σ₋ where σ₋ = |0⟩⟨1|

**Master Equation** (zero temperature, H = 0):
```
dρ/dt = γ₁(σ₋ ρ σ₊ - ½{σ₊σ₋, ρ})
```

**Matrix Elements**:
```
dρ₀₀/dt = γ₁ ρ₁₁           (population flows 1→0)
dρ₁₁/dt = -γ₁ ρ₁₁          (|1⟩ decays)
dρ₀₁/dt = -(γ₁/2) ρ₀₁      (coherence decays)
dρ₁₀/dt = -(γ₁/2) ρ₁₀
```

**Analytical Solution**:
```
ρ₀₀(t) = 1 - e^(-γ₁t) ρ₁₁(0)
ρ₁₁(t) = e^(-γ₁t) ρ₁₁(0)
ρ₀₁(t) = e^(-γ₁t/2) ρ₀₁(0)
```

**T1 Time**: Characteristic decay time for population
```
T₁ = 1/γ₁
```

**Kraus Operators** (discrete time step Δt):
```
K₀ = [[1, 0], [0, √(1-p)]]
K₁ = [[0, √p], [0, 0]]

where p = 1 - e^(-Δt/T₁)
```

### Phase Damping (T2)

**Physical Process**: Loss of phase coherence (no energy change)

**Lindblad Operator**: L = √γφ σz

**Master Equation**:
```
dρ/dt = γφ(σz ρ σz - ρ)
```

**Matrix Elements**:
```
dρ₀₀/dt = 0              (populations unchanged)
dρ₁₁/dt = 0
dρ₀₁/dt = -2γφ ρ₀₁       (coherence decays)
dρ₁₀/dt = -2γφ ρ₁₀
```

**Analytical Solution**:
```
ρ₀₀(t) = ρ₀₀(0)  (constant)
ρ₁₁(t) = ρ₁₁(0)
ρ₀₁(t) = e^(-2γφt) ρ₀₁(0)
```

**T2 Time**: Characteristic decay time for coherence
```
1/T₂ = 1/(2T₁) + 1/T₂*

where T₂* is pure dephasing time
```

**Relation**: T₂ ≤ 2T₁ (always)

### Typical Experimental Values

**Superconducting Qubits** (transmon):
- T₁ ~ 50-200 μs
- T₂ ~ 30-150 μs
- T₂ < 2T₁ (limited by 1/f noise)

**Trapped Ions**:
- T₁ ~ minutes to hours
- T₂ ~ seconds
- T₂ ≪ T₁ (magnetic field noise)

**Quantum Dots**:
- T₁ ~ 1-10 μs
- T₂ ~ 1-5 μs

## Part 6: Purity Evolution

### Purity Decay

**Purity**: P(t) = Tr(ρ²(t))

**Properties**:
- P = 1: Pure state
- P < 1: Mixed state
- dP/dt ≤ 0: Purity never increases under decoherence

**Example**: Depolarizing channel
```
ρ(t) = (1-p(t)) ρ(0) + p(t) I/2

P(t) = Tr(ρ²(t))
     = (1-p(t))² Tr(ρ²(0)) + p(t)² Tr((I/2)²) + cross terms
     = (1-p(t))² + p(t)²/4  (for initially pure state)

As p → 1: P → 1/2 (maximally mixed)
```

**Monotonic Decay**: P(t) decreases from 1 to 1/d (maximally mixed)

## Part 7: Verification Strategy

### Symbolic Tests

1. **Completeness**: Σₖ Kₖ†Kₖ = I
2. **Trace Preservation**: Tr(ε(ρ)) = 1
3. **Hermiticity**: ε(ρ)† = ε(ρ)
4. **Positive Semi-Definite**: All eigenvalues ≥ 0

### Numerical Tests

1. **Analytical Solutions**: Compare to known exponential decays
2. **Purity Decay**: Verify P(t) decreases monotonically
3. **Steady States**: ε(ρ_ss) = ρ_ss (fixed points)
4. **Channel Concatenation**: ε₂∘ε₁ is also TPCP

### Experimental Benchmarks

1. **T₁ Measurement**: Prepare |1⟩, measure population vs time
2. **T₂ Measurement**: Ramsey experiment, measure coherence vs time
3. **Compare to Literature**: Superconducting qubits, trapped ions

## Summary

✓ Density matrices represent both pure and mixed states
✓ Quantum channels = Kraus operators (TPCP maps)
✓ Lindblad equation governs open system dynamics
✓ T₁ (amplitude damping) and T₂ (phase damping) are key timescales
✓ Decoherence drives pure → mixed, purity decays
✓ All real qubits experience these effects

**Next**: Implement these concepts with full numerical verification.