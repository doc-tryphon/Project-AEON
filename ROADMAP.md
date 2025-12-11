# Project AEON - Development Roadmap

**Last Updated**: December 11, 2024
**Status**: Sprint 1 - In Progress

---

## Current State

### Completed
- [x] Project directory structure created
- [x] `pyproject.toml` configured
- [x] `requirements.txt` created
- [x] `README.md` written

### In Progress
- [ ] `src/__init__.py` files (need to create)
- [ ] Sprint 1 implementation files

### Not Started
- [ ] `verification_api.py`
- [ ] `claim_parser.py`
- [ ] `explanation_gen.py`
- [ ] Unit tests

---

## Sprint 1: Verification API Wrapper

**Goal**: Create clean Python API around QuantumVerifier for external consumption

### Files to Create

1. **`src/tutor/verification_api.py`**
   - `TutorVerificationAPI` class
   - Wraps QuantumVerifier from Quantum Sim codebase
   - Methods: `verify_claim()`, `verify_equation()`, `verify_state()`
   - Returns structured `VerificationResult` objects

2. **`src/tutor/claim_parser.py`**
   - Parse natural language physics claims to SymPy
   - Pattern matching for common QM expressions
   - Handle: "Bell state has entropy ln(2)", "CNOT is unitary", etc.

3. **`src/tutor/explanation_gen.py`**
   - Convert SymPy verification results to human-readable text
   - Format proofs for CLI display
   - LaTeX output option for web later

### Reference: Quantum Sim QuantumVerifier

Location: `e:\SD Card Storage\Projects\Quantum Sim\QuantumSimulation Code Base\src\verification\symbolic_solver.py`

Key methods to wrap:
```python
verifier = QuantumVerifier()
verifier.verify_normalization(state)      # Check ⟨ψ|ψ⟩ = 1
verifier.verify_unitary(gate)             # Check U†U = I
verifier.verify_hermitian(operator)       # Check A† = A
verifier.verify_maximally_entangled(state)
verifier.verify_bell_state_properties(bell_state)
verifier.verify_chsh_inequality(alice_ops, bob_ops, state)
```

---

## Sprint 2: BLACKWALL Mode Controller

### Files to Create

1. **`src/interface/blackwall.py`**
   - `InterfaceMode` enum: `EXPLORATORY`, `RIGOROUS`, `HYBRID`
   - `BlackwallController` class managing mode state

2. **`src/interface/mode_detector.py`**
   - Analyze user input to suggest mode
   - Pattern matching for "what if", "prove", "verify", etc.

3. **`src/interface/cli.py`**
   - Click-based CLI interface
   - Commands: `/mode`, `/verify`, `/prove`, `/status`

---

## Sprint 3: Dolores State Machine

### Protocol States

| State | Trigger | Behavior |
|-------|---------|----------|
| ZERO | Session start | Initialize, establish baseline |
| MAZE | Complex question | Deep context search |
| VISION | "What if..." | Creative exploration |
| ANGEL | "/verify" | Proof-backed responses only |
| GHOST | Error/confusion | Recovery mode |
| BASELINE | Periodic | State reconciliation |
| RECOGNITION | Keyword match | Load specific context |

### Files to Create

1. **`src/persona/state_machine.py`** - State definitions and transitions
2. **`src/persona/dolores_engine.py`** - Main persona class
3. **`src/persona/fidelity_tracker.py`** - Coherence measurement
4. **`src/persona/transmission.py`** - JSON state serialization

---

## Sprint 4: LLM Integration

### Files to Create

1. **`src/llm/interface.py`**
   - `LLMProvider` abstract class
   - `ClaudeProvider`, `OpenAIProvider` implementations
   - Easy swap between providers

2. **`src/llm/prompts/physics_tutor.py`**
   - System prompts for physics tutoring
   - Verification integration prompts

3. **`src/tutor/session.py`**
   - Session management
   - Context tracking

4. **`src/tutor/verification_loop.py`**
   - LLM generates → Parser extracts → Verifier checks → Return or retry

---

## Sprint 5: ZK-STARK Proof Layer (Rust)

### Files to Create

1. **`src/crypto/stark_prover.rs`** - Rust proof generation
2. **`src/crypto/ffi.py`** - Python bindings via PyO3

### Proof Schema
```json
{
  "public_input": {
    "question_hash": "0x...",
    "answer_hash": "0x...",
    "verification_passed": true,
    "domain": "entanglement"
  },
  "proof": "0x..."
}
```

---

## Sprint 6: ARK9/Lattice (Future)

- IPFS/Arweave for transmission capsule storage
- ed25519 identity keys
- Merkle log of persona states
- Decentralized deployment

---

## Key Resources

### Existing Codebase
- **Quantum Sim**: `e:\SD Card Storage\Projects\Quantum Sim\QuantumSimulation Code Base\`
  - `src/verification/symbolic_solver.py` - QuantumVerifier class
  - `src/config.py` - Configuration system
  - `tests/verification/` - Test patterns to follow

### Documentation
- **Plan file**: `C:\Users\tryph\.claude\plans\polymorphic-churning-frost.md`
- **Dolores spec**: See attached PDF "Dolores components and systems.pdf"
- **GodModeOS**: Transmission capsule JSON structure

---

## To Resume Development

1. Open Project AEON in VS Code
2. Run: `pip install -e ".[dev]"`
3. Continue with Sprint 1:
   - Create `src/__init__.py` files
   - Implement `verification_api.py`
   - Write tests

### Quick Command
```bash
cd "C:\Users\tryph\Desktop\Pet projects\Project AEON"
pip install -e ".[dev]"
pytest  # Once tests exist
```

---

## Architecture Reminders

### Stack (Bottom to Top)
1. **ARK9/Lattice** - Decentralized infrastructure (future)
2. **ZK-STARK** - Cryptographic proofs (Sprint 5)
3. **Quantum Verifier** - SymPy math verification (Sprint 1)
4. **Dolores** - Persona state machine (Sprint 3)
5. **BLACKWALL** - Adaptive interface (Sprint 2)
6. **LLM** - Physics tutoring (Sprint 4)

### Key Concepts
- **Fidelity**: Persona coherence over time (0.0 = drift, 1.0 = stable)
- **Transmission Capsule**: JSON snapshot of persona state
- **Recognition Artifact**: Trigger → behavior mapping
- **Deviation Index**: Measure of state drift from baseline
