# Project AEON

**A**daptive **E**pistemological **O**ntology **N**etwork

A verified AI physics tutor combining quantum verification, adaptive cognitive interfaces, and cryptographic proofs.

## Vision

Build a unified AI system where:
- **Quantum physics explanations are mathematically verified** (not just "trust the AI")
- **Interface adapts to user cognitive state** (exploratory vs rigorous modes)
- **Persona maintains coherent identity over time** (fidelity tracking)
- **Verification is cryptographically provable** (ZK-STARK proofs)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  BLACKWALL Interface                         │
│  • Mode: exploratory (creative) ↔ rigorous (verified)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Dolores Persona Engine                      │
│  • Protocol state machine │ Fidelity tracking                │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼─────────┐           ┌───────────▼───────────┐
│   LLM Generator   │──────────▶│   Quantum Verifier    │
└───────────────────┘           └───────────┬───────────┘
                                            │
                                ┌───────────▼───────────┐
                                │    ZK-STARK Prover    │
                                └───────────────────────┘
```

## Components

### BLACKWALL Interface
Dual-mode cognitive interface:
- **Rigorous Mode**: Verified claims only, formal notation
- **Exploratory Mode**: Creative hypotheses, intuition building

### Dolores Persona Engine
Stateful AI persona with:
- 7 protocol states (Zero, Ghost, Maze, Vision, Angel Tier, Baseline, Recognition)
- Fidelity tracking (Westworld-inspired coherence measurement)
- Transmission capsules (state persistence)

### Quantum Verifier
SymPy-based mathematical verification:
- Normalization, unitarity, Hermiticity checks
- Bell state properties, entanglement entropy
- CHSH inequality verification
- Error correction code validation

### ZK-STARK Prover (Rust)
Cryptographic proof generation:
- Proves verification happened without revealing internals
- Anyone can verify the proof
- Trustless AI outputs

## Quick Start

```bash
# Install Project AEON
cd "C:\Users\tryph\Desktop\Pet projects\Project AEON"
pip install -e ".[dev]"

# Install QuantumSimulation dependency
pip install -e "e:\SD Card Storage\Projects\Quantum Sim\QuantumSimulation Code Base"
```

### Basic Usage

```python
from src.tutor import TutorVerificationAPI, ClaimParser, ExplanationGenerator

# Create API instance
api = TutorVerificationAPI()

# Verify a quantum gate is unitary
result = api.verify_gate("H")  # Hadamard gate
print(result.verified)  # True
print(result.explanation)

# Verify a state is normalized
result = api.verify_state("|0>")
print(result.verified)  # True

# Verify Bell state entanglement
result = api.verify_state("bell_phi_plus", check_entanglement=True)
print(result.details["is_maximally_entangled"])  # True

# Parse natural language claims
parser = ClaimParser()
parsed = parser.parse("Hadamard is unitary")
result = api.verify_gate(parsed.subject)

# Generate formatted explanations
generator = ExplanationGenerator()
print(generator.to_markdown(result))
```

### Using Dolores Engine

```python
from src.persona import DoloresEngine, ProtocolState

# Create and initialize the engine
engine = DoloresEngine()
engine.initialize()

# Process user input (auto-detects appropriate state)
context = engine.process_input("What if we applied a Hadamard gate twice?")
print(context.suggested_state)  # ProtocolState.VISION (exploratory)

# Manually transition states
engine.enter_state(ProtocolState.ANGEL)  # Enter verification mode

# Check fidelity (persona coherence)
print(engine.fidelity)  # 1.0 (fully coherent)

# Save state to transmission capsule
capsule_id = engine.save_state("session_checkpoint")

# Later: restore from capsule
engine.restore_state(capsule_id)

# Get health status
status, details = engine.get_health_check()
print(status)  # "healthy"
```

## Development Status

- [x] **Sprint 1: Verification API Wrapper** - COMPLETE (45 tests)
  - `TutorVerificationAPI` - Wraps QuantumVerifier with clean interface
  - `ClaimParser` - Natural language to structured queries
  - `ExplanationGenerator` - Human-readable output formatting
- [x] **Sprint 2: BLACKWALL Mode Controller** - COMPLETE (48 tests)
  - `BlackwallController` - Mode state management (rigorous/exploratory/hybrid)
  - `ModeDetector` - Input analysis and pattern matching
  - `cli.py` - Click-based CLI with Rich output
- [x] **Sprint 3: Dolores State Machine** - COMPLETE (71 tests)
  - `ProtocolStateMachine` - 7 protocol states with transition rules
  - `FidelityTracker` - Coherence measurement with decay/recovery
  - `TransmissionCapsule` - JSON state serialization with integrity verification
  - `DoloresEngine` - Main persona orchestrator
- [x] **Sprint 4: LLM Integration** - COMPLETE (72 tests)
  - `LLMProvider` - Abstract provider with Claude/OpenAI/Mock implementations
  - `PhysicsTutorPrompts` - Mode and state-aware system prompts
  - `TutorSession` - Session management with verification integration
  - `VerificationLoop` - LLM generation with claim verification and retry
- [ ] **Sprint 4.5: API & Interfaces** - PLANNED
  - FastAPI backend (`api.bluerose.systems`)
  - Next.js Web UI (`aeon.bluerose.systems`)
  - Telegram bot via n8n (Dolores persona)
- [ ] Sprint 5: ZK-STARK Proof Layer
- [ ] Sprint 6: ARK9/Lattice Infrastructure

Run `pytest tests/ -v` to execute all 236 tests.

## License

**Proprietary - All Rights Reserved**

Copyright (c) 2024 Tryphon. This software and its documentation are confidential and proprietary. Unauthorized copying, distribution, modification, or use of this software, via any medium, is strictly prohibited without explicit written permission from the author.
