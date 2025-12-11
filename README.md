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
# Clone and install
git clone https://github.com/youruser/project-aeon.git
cd project-aeon
pip install -e ".[dev]"

# Run CLI
aeon --help

# Run tests
pytest
```

## Development Status

- [x] Sprint 1: Verification API Wrapper
- [ ] Sprint 2: BLACKWALL Mode Controller
- [ ] Sprint 3: Dolores State Machine
- [ ] Sprint 4: LLM Integration
- [ ] Sprint 5: ZK-STARK Proof Layer
- [ ] Sprint 6: ARK9/Lattice Infrastructure

## License

MIT
