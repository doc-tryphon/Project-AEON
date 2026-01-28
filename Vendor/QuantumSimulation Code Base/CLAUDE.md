# CLAUDE.md - Experimental Quantum Physics Framework

## Project Overview

This is a **rigorously verified** quantum physics simulation framework focused on experimentally observable phenomena with known analytical solutions.

## Core Philosophy

**Every equation in this codebase must be:**
1. Derived symbolically using SymPy
2. Verified against published textbooks/papers
3. Tested against experimental data where available
4. Documented with paper references

**We do NOT simulate:**
- Speculative theoretical physics (wormholes, exotic matter)
- Unverified quantum gravity theories
- Phenomena with no experimental evidence

**We DO simulate:**
- Quantum entanglement (Bell states, EPR)
- Quantum decoherence and measurement
- Quantum interference experiments
- Quantum teleportation protocols
- Quantum error correction codes
- Cavity QED (Jaynes-Cummings model)

## Project Structure

```
QuantumSimulation/
├── src/
│   ├── quantum/
│   │   ├── entanglement.py        # Bell states, EPR pairs
│   │   ├── decoherence.py         # Decoherence models
│   │   ├── teleportation.py       # Quantum teleportation
│   │   └── error_correction.py    # QEC codes
│   ├── verification/
│   │   ├── symbolic_solver.py     # SymPy verification
│   │   ├── experimental_data.py   # Published benchmarks
│   │   └── test_framework.py      # Validation tests
│   ├── physics/
│   │   ├── constants.py           # Physical constants
│   │   └── measurements.py        # Observable operators
│   └── visualization/
│       └── bloch_sphere.py        # Quantum state visualization
├── experiments/                    # Experimental benchmarks
├── verification_reports/           # SymPy verification outputs
├── tests/                         # Unit tests
├── legacy_wormhole_simulation/    # Archived unverified code
└── main.py                        # Entry point
```

## Verification Workflow

For every quantum system:

1. **Mathematical Derivation** (SymPy)
   - Define Hilbert space symbolically
   - Derive operators and eigenstates
   - Verify commutation relations
   - Check unitarity/hermiticity

2. **Code Implementation**
   - Implement numerical version
   - Test against symbolic results
   - Verify edge cases

3. **Experimental Validation**
   - Compare to published data
   - Document deviations
   - Understand approximations

## Core Components

### 1. Quantum Entanglement (`src/quantum/entanglement.py`)
- **Bell States**: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
- **Bell Inequality Tests**: CHSH inequality
- **EPR Correlations**: Spin measurements
- **Verification**: Known maximally entangled states

### 2. Quantum Measurement (`src/quantum/measurements.py`)
- **Projection Operators**: Von Neumann measurement
- **POVM**: Generalized measurements
- **Weak Measurements**: Post-selection
- **Verification**: Measurement postulates

### 3. Quantum Decoherence (`src/quantum/decoherence.py`)
- **Pure to Mixed State Evolution**
- **Master Equations**: Lindblad form
- **Decoherence Times**: T1, T2 relaxation
- **Verification**: Known decay rates

### 4. Quantum Teleportation (`src/quantum/teleportation.py`)
- **Standard Protocol**: Alice → Bob teleportation
- **Fidelity Calculations**: State transfer quality
- **Resource States**: Entanglement consumption
- **Verification**: Known protocol success rates

### 5. Quantum Error Correction (`src/quantum/error_correction.py`)
- **3-Qubit Bit Flip Code**
- **Shor's 9-Qubit Code**
- **Surface Codes**: Topological protection
- **Verification**: Known threshold theorems

## Working with the Codebase

### Development Principles

1. **Symbolic First**: Derive with SymPy before coding
2. **Test Driven**: Write tests from analytical solutions
3. **Document Sources**: Cite papers/textbooks
4. **Experimental Grounding**: Link to real experiments
5. **No Speculation**: If it's not verified, it doesn't go in

### Adding New Systems

```python
# 1. Symbolic derivation
verification_report = verify_system_symbolically(hamiltonian)

# 2. Numerical implementation
simulator = QuantumSimulator(system_params)

# 3. Test against known results
assert np.allclose(simulator.eigenvalues(), analytical_eigenvalues)

# 4. Compare to experiment
experimental_data = load_experiment('Nature_2023_123')
fidelity = compare_to_experiment(simulator, experimental_data)
```

### Verification Reports

Every major component has a verification report:
```
verification_reports/
├── bell_states_verification.md       # Mathematical derivation
├── teleportation_protocol.md         # Protocol correctness
└── decoherence_master_equation.md    # Lindblad form validation
```

## Key Dependencies

- **NumPy/SciPy**: Numerical linear algebra
- **QuTiP**: Quantum toolbox (verified against analytical solutions)
- **SymPy**: Symbolic mathematics and verification
- **pytest**: Testing framework
- **Matplotlib**: Visualization

## Testing Strategy

```bash
# Run all verification tests
pytest tests/verification/

# Run against experimental benchmarks
pytest tests/experimental/

# Generate verification reports
python scripts/generate_verification_reports.py
```

## References

This codebase implements systems from:

- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Preskill, "Lecture Notes on Quantum Computation"
- Wiseman & Milburn, "Quantum Measurement and Control"
- Experimental papers (documented per module)

## Migration from Legacy Code

The previous "wormhole simulation" has been archived to `legacy_wormhole_simulation/`.

**Reason for pivot:**
- Wormholes are unverified theoretical constructs
- No experimental data to validate against
- Hand-coded equations without symbolic verification
- Cannot teach real physics from speculative phenomena

**New focus:**
- Experimentally observed quantum phenomena
- Known analytical solutions
- Published experimental benchmarks
- Rigorous mathematical verification

---

## Important Instructions

**NEVER:**
- Add unverified theoretical physics
- Implement equations without symbolic derivation
- Skip experimental validation where possible
- Use numerical values without source citation

**ALWAYS:**
- Derive symbolically first (SymPy)
- Test against known analytical solutions
- Document paper/textbook sources
- Compare to experimental data when available
- Question consensus if math doesn't check out