# Project AEON - Development Roadmap

**Last Updated**: December 22, 2024
**Status**: Sprint 4 Complete, Sprint 4.5 Planned

---

## Current State

### Completed
- [x] Project directory structure created
- [x] `pyproject.toml` configured
- [x] `requirements.txt` created
- [x] `README.md` written
- [x] `src/__init__.py` files created
- [x] QuantumSimulation installed as editable dependency
- [x] **Sprint 1: Verification API** - COMPLETE
  - [x] `src/tutor/verification_api.py` - TutorVerificationAPI wrapper
  - [x] `src/tutor/claim_parser.py` - Natural language claim parsing
  - [x] `src/tutor/explanation_gen.py` - Human-readable output generation
  - [x] `tests/test_verification_api.py` - 45 tests passing
- [x] **Sprint 2: BLACKWALL Mode Controller** - COMPLETE
  - [x] `src/interface/enums.py` - InterfaceMode, RequestType, TransitionReason
  - [x] `src/interface/blackwall.py` - BlackwallController with mode management
  - [x] `src/interface/mode_detector.py` - Input analysis and pattern matching
  - [x] `src/interface/cli.py` - Click-based CLI with Rich output
  - [x] `tests/test_blackwall.py` - 48 tests passing
- [x] **Sprint 3: Dolores State Machine** - COMPLETE
  - [x] `src/persona/state_machine.py` - ProtocolState enum, ProtocolStateMachine, TransitionRulesEngine
  - [x] `src/persona/fidelity_tracker.py` - FidelityTracker, FidelityComponent, FidelityAnalyzer
  - [x] `src/persona/transmission.py` - TransmissionCapsule, CapsuleManager
  - [x] `src/persona/dolores_engine.py` - DoloresEngine main orchestrator
  - [x] `tests/test_dolores.py` - 71 tests passing
- [x] **Sprint 4: LLM Integration** - COMPLETE
  - [x] `src/llm/interface.py` - LLMProvider abstract class, Message, LLMConfig, LLMResponse
  - [x] `src/llm/providers.py` - ClaudeProvider, OpenAIProvider, MockProvider
  - [x] `src/llm/prompts/physics_tutor.py` - PhysicsTutorPrompts, mode/state prompts
  - [x] `src/tutor/session.py` - TutorSession with mode management
  - [x] `src/tutor/verification_loop.py` - VerificationLoop with retry logic
  - [x] `tests/test_llm.py` - 72 tests passing

### Ready to Start

- [x] Sprint 4.5: API & Interfaces (FastAPI + Web UI) - **PHASE 1 (Backend) COMPLETE**

### Not Started

- [ ] Sprint 5: ZK-STARK Proof Layer
- [ ] Sprint 6: ARK9/Lattice

---

## Sprint 1: Verification API Wrapper - COMPLETE

**Goal**: Create clean Python API around QuantumVerifier for external consumption

**Status**: Complete (December 18, 2024) - 45 tests passing

### Files Created

1. **`src/tutor/verification_api.py`** ✅
   - `TutorVerificationAPI` class wrapping QuantumVerifier
   - `VerificationResult` dataclass for structured responses
   - Methods: `verify_gate()`, `verify_state()`, `verify_operator()`, `verify_bell_state()`, `verify_chsh()`, `verify_claim()`
   - Helper functions: `parse_matrix_expr()`, `parse_state_expr()`
   - Custom exceptions: `VerificationError`, `ParseError`, `UnsupportedClaimError`

2. **`src/tutor/claim_parser.py`** ✅
   - `ClaimParser` class with regex pattern matching
   - `ParsedClaim` dataclass for structured queries
   - Supports: unitarity, hermiticity, normalization, entanglement claims
   - Fuzzy parsing fallback for unrecognized patterns

3. **`src/tutor/explanation_gen.py`** ✅
   - `ExplanationGenerator` class with multiple output formats
   - Formats: plain text, Markdown, LaTeX, HTML, Rich terminal
   - Domain-specific templates for each verification type

4. **`tests/test_verification_api.py`** ✅
   - 45 comprehensive tests covering all components
   - Tests for parsing, verification, and explanation generation

### Dependency Integration

QuantumSimulation installed as editable dependency:

```bash
pip install -e "e:\SD Card Storage\Projects\Quantum Sim\QuantumSimulation Code Base"
```

Import pattern:

```python
from verification.symbolic_solver import QuantumVerifier
```

### Key Methods Wrapped

| QuantumVerifier Method | Wrapper Method | Returns |
| ---------------------- | -------------- | ------- |
| `verify_unitary(matrix)` | `verify_gate(expr)` | `Tuple[bool, Matrix]` |
| `verify_normalization(state)` | `verify_state(expr)` | `Tuple[bool, Expr]` |
| `verify_hermitian(matrix)` | `verify_operator(expr)` | `Tuple[bool, Matrix]` |
| `verify_maximally_entangled(state)` | `verify_state(expr, check_entanglement=True)` | `Tuple[bool, Expr]` |
| `verify_bell_state_properties(state)` | `verify_bell_state(expr)` | `Dict[str, Any]` |
| `verify_chsh_inequality(...)` | `verify_chsh(...)` | `Dict[str, Any]` |

---

## Sprint 2: BLACKWALL Mode Controller - COMPLETE

**Goal**: Create adaptive cognitive interface with mode switching

**Status**: Complete (December 18, 2024) - 48 tests passing

### Files Created

1. **`src/interface/enums.py`** ✅
   - `InterfaceMode` enum: `RIGOROUS`, `EXPLORATORY`, `HYBRID`, `SYSTEM`
   - `RequestType` enum: `VERIFICATION`, `EXPLANATION`, `EXPLORATION`, etc.
   - `TransitionReason` enum for tracking mode changes
   - Properties: `requires_verification`, `allows_speculation`, `min_confidence`

2. **`src/interface/blackwall.py`** ✅
   - `BlackwallController` class managing mode state
   - `ModeConfig` dataclass for configuration per mode
   - `ModeTransition` dataclass for history tracking
   - Methods: `set_mode()`, `get_config()`, `validate_request()`, `validate_response()`
   - Callback system for mode transitions
   - Serialization/deserialization support

3. **`src/interface/mode_detector.py`** ✅
   - `ModeDetector` class with regex pattern matching
   - `DetectionResult` dataclass for analysis results
   - Rigorous patterns: "prove", "verify", "is unitary", etc.
   - Exploratory patterns: "what if", "imagine", "hypothetically", etc.
   - Command detection and argument extraction
   - Context-based mode suggestion

4. **`src/interface/cli.py`** ✅
   - Click-based CLI interface with Rich terminal output
   - Commands: `/mode`, `/status`, `/history`, `/verify`, `/prove`, `/analyze`
   - Interactive mode with auto-transition support
   - Integration with TutorVerificationAPI

5. **`tests/test_blackwall.py`** ✅
   - 48 comprehensive tests covering all components
   - Tests for enums, controller, detector, and integration

### Mode Behaviors

| Mode | Require Proof | Allow Speculation | Min Confidence |
|------|---------------|-------------------|----------------|
| RIGOROUS | Yes | No | 100% |
| EXPLORATORY | No | Yes | 0% |
| HYBRID | No | Yes | 70% |
| SYSTEM | No | No | 0% |

### Integration with Sprint 1

The BLACKWALL controller validates responses from TutorVerificationAPI:
- In RIGOROUS mode, only `confidence >= 1.0` responses are accepted
- In HYBRID mode, responses with `confidence < 0.7` generate warnings
- In EXPLORATORY mode, all responses are accepted

---

## Sprint 3: Dolores State Machine - COMPLETE

**Goal**: Create persona state machine with fidelity tracking and transmission capsules

**Status**: Complete (December 19, 2024) - 71 tests passing

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

### Implementation Details

1. **`src/persona/state_machine.py`** ✅
   - `ProtocolState` enum with 7 states and behavior properties
   - `ProtocolStateMachine` class with transition rules and history
   - `TransitionTrigger` enum for state change triggers
   - `TransitionRulesEngine` class for auto-detecting appropriate states
   - `RecognitionArtifact` dataclass for keyword → context mappings
   - `StateConfig` dataclass with per-state configuration

2. **`src/persona/fidelity_tracker.py`** ✅
   - `FidelityTracker` class for monitoring persona coherence
   - `FidelityComponent` enum: response_consistency, memory_integrity, state_coherence, behavioral_alignment, temporal_continuity
   - `FidelityAnalyzer` class for recovery recommendations
   - `DeviationEvent` dataclass for tracking drift
   - `BaselineConfig` dataclass with thresholds and weights
   - Decay/recovery mechanics for time-based fidelity management

3. **`src/persona/transmission.py`** ✅
   - `TransmissionCapsule` class for complete state serialization
   - `CapsuleManager` class for save/load/list operations
   - `CapsuleFormat` enum: JSON, COMPRESSED (gzip)
   - Section dataclasses: Header, Identity, State, Memory, Metrics
   - SHA-256 integrity verification
   - JSON and compressed file support

4. **`src/persona/dolores_engine.py`** ✅
   - `DoloresEngine` class orchestrating all components
   - `PersonaConfig` dataclass for configuration
   - `ResponseContext` dataclass for input processing results
   - `ResponseMetadata` dataclass for response tracking
   - Integration with state machine, fidelity tracker, and capsule manager
   - Health check and status reporting

5. **`tests/test_dolores.py`** ✅
   - 71 comprehensive tests across 10 test classes
   - Tests for state machine, fidelity, transmission, and engine
   - Integration tests for full workflow and capsule roundtrip

### Fidelity Components

| Component | Weight | Expected | Threshold |
| --------- | ------ | -------- | --------- |
| Response Consistency | 25% | 0.90 | 0.15 |
| Memory Integrity | 20% | 0.95 | 0.10 |
| State Coherence | 20% | 0.85 | 0.20 |
| Behavioral Alignment | 20% | 0.90 | 0.15 |
| Temporal Continuity | 15% | 0.85 | 0.20 |

### Integration with Sprint 1 & 2

- DoloresEngine uses `InterfaceMode` from BLACKWALL for response context
- State transitions can trigger mode changes (ANGEL → RIGOROUS)
- Transmission capsules can capture current BLACKWALL mode state

---

## Sprint 4: LLM Integration - COMPLETE

**Goal**: Create LLM provider abstraction and tutoring session management

**Status**: Complete (December 19, 2024) - 72 tests passing

### Files Created

1. **`src/llm/interface.py`** ✅
   - `LLMProvider` abstract base class with sync/async methods
   - `Message`, `MessageRole` for conversation handling
   - `LLMConfig` for provider configuration
   - `LLMResponse`, `TokenUsage`, `FinishReason` for responses
   - Exception hierarchy: `LLMError`, `RateLimitError`, `AuthenticationError`, etc.

2. **`src/llm/providers.py`** ✅
   - `ClaudeProvider` - Anthropic Claude integration
   - `OpenAIProvider` - OpenAI GPT integration
   - `MockProvider` - Testing provider with predefined responses
   - `create_provider()` factory function
   - Streaming support, token counting, cost estimation

3. **`src/llm/prompts/physics_tutor.py`** ✅
   - `PhysicsTutorPrompts` class with mode/state-specific prompts
   - `PromptTemplate` for reusable templates with variable substitution
   - `BASE_SYSTEM_PROMPT` - Dolores persona definition
   - Mode prompts: `RIGOROUS_MODE_PROMPT`, `EXPLORATORY_MODE_PROMPT`, `HYBRID_MODE_PROMPT`
   - State prompts for all 7 Dolores protocol states

4. **`src/tutor/session.py`** ✅
   - `TutorSession` - Main session orchestrator
   - `SessionConfig`, `SessionStatus`, `SessionStats`
   - `Turn`, `TurnMetadata` for conversation tracking
   - Mode auto-detection and switching
   - Integration with DoloresEngine and BLACKWALL
   - Direct verification methods (`verify_gate()`, `verify_state()`, etc.)

5. **`src/tutor/verification_loop.py`** ✅
   - `VerificationLoop` - LLM generation with verification cycle
   - `LoopConfig`, `LoopResult`, `LoopStatus`
   - `RetryStrategy` - NONE, REGENERATE, TARGETED, FALLBACK
   - `VerificationAttempt` for tracking retry attempts
   - Automatic claim extraction and verification
   - Response enhancement with verification status

6. **`tests/test_llm.py`** ✅
   - 72 comprehensive tests covering all Sprint 4 components
   - Tests for interface types, providers, prompts, session, and loop
   - Integration tests for full workflow

### Key Features

| Feature | Description |
|---------|-------------|
| Provider Abstraction | Swap between Claude/OpenAI/Mock without code changes |
| Mode-Aware Prompts | System prompts adapt to BLACKWALL mode |
| State-Aware Prompts | Prompts include Dolores protocol state context |
| Verification Loop | Auto-verify claims with retry on failure |
| Session Management | Track conversation, stats, and persona state |
| Streaming Support | Iterator-based streaming for real-time output |

### Integration with Previous Sprints

- Uses `TutorVerificationAPI` (Sprint 1) for claim verification
- Uses `InterfaceMode` from BLACKWALL (Sprint 2) for mode management
- Uses `DoloresEngine` (Sprint 3) for persona state and fidelity
- Prompts reference Dolores protocol states for consistent persona

---

## Sprint 4.5: API & Interfaces - PLANNED

**Goal**: Wire Sprint 4 LLM components to user-facing interfaces

**Status**: Phase 1 (Backend) Complete (December 31, 2024). Phase 2 & 3 Pending.

**Rationale**: Sprint 4 built TutorSession, VerificationLoop, and LLMProviders, but they're not accessible to end users. This sprint completes the integration.

### Domain

- `bluerose.systems` (pending acquisition)
  - `aeon.bluerose.systems` - Web UI (Vercel)
  - `api.bluerose.systems` - FastAPI backend (Hostinger VPS)

### Phase 1: FastAPI Backend

Files to create:

1. **`src/api/__init__.py`** - Module exports
2. **`src/api/app.py`** - FastAPI application with CORS
3. **`src/api/models.py`** - Pydantic request/response models
4. **`src/api/routes/chat.py`** - POST `/api/chat`, SSE streaming, WebSocket
5. **`src/api/routes/verify.py`** - POST `/api/verify`, gate/state/operator endpoints
6. **`src/api/routes/session.py`** - Session CRUD, capsule save/restore
7. **`src/api/session_manager.py`** - In-memory session store
8. **`tests/test_api.py`** - API endpoint tests

Dependencies to add:

```toml
fastapi = ">=0.109.0"
uvicorn = ">=0.27.0"
python-multipart = ">=0.0.6"
```

### Phase 2: Next.js Web UI

Separate repository: `aeon-web/`

- Next.js 14 (App Router)
- Tailwind CSS + shadcn/ui
- Chat interface with verification badges
- Mode selector (Rigorous/Exploratory/Hybrid)
- Dolores state indicator
- Fidelity meter visualization
- Deploy on Vercel

### Phase 3: Telegram Integration

- n8n webhook approach (leverage existing Doctor Feynman workflow)
- Replace "Ultimate Assistant" node with HTTP Request to FastAPI
- Session persistence per Telegram user via capsule system
- Dolores persona replaces Doctor Feynman

### Key Decisions

- **LLM Provider**: Claude (Anthropic) as primary
- **Session Storage**: In-memory initially, Redis upgrade path
- **Telegram**: n8n webhook (not dedicated bot)

### API Endpoints

```
POST   /api/chat                    # Send message, get verified response
GET    /api/chat/stream             # SSE streaming
WS     /api/ws/chat                 # WebSocket chat

POST   /api/verify                  # Verify a claim
GET    /api/verify/gate/{name}      # Gate verification
GET    /api/verify/state/{expr}     # State verification

POST   /api/session                 # Create session
GET    /api/session/{id}            # Get status
DELETE /api/session/{id}            # End session
POST   /api/session/{id}/save       # Create capsule
POST   /api/session/restore/{id}    # Restore from capsule
```

### Plan File

Detailed implementation plan: `C:\Users\tryph\.claude\plans\wiggly-wiggling-wozniak.md`

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
3. Run tests: `pytest tests/ -v`
4. **Next**: Sprint 4.5 - API & Interfaces (after `bluerose.systems` domain acquired)
5. **Then**: Sprint 5 - ZK-STARK Proof Layer

### Quick Commands

```bash
cd "C:\Users\tryph\Desktop\Pet projects\Project AEON"
pip install -e ".[dev]"
pytest tests/ -v  # 236 tests (45 Sprint 1 + 48 Sprint 2 + 71 Sprint 3 + 72 Sprint 4)

# Try the CLI
python -m src.interface.cli --help
python -m src.interface.cli status

# Try Dolores Engine
python -c "from src.persona import DoloresEngine; e = DoloresEngine(); e.initialize(); print(e.get_status())"
```

---

## Architecture Reminders

### Stack (Bottom to Top)
1. **ARK9/Lattice** - Decentralized infrastructure (Sprint 6)
2. **ZK-STARK** - Cryptographic proofs (Sprint 5)
3. **Quantum Verifier** - SymPy math verification (Sprint 1)
4. **Dolores** - Persona state machine (Sprint 3)
5. **BLACKWALL** - Adaptive interface (Sprint 2)
6. **LLM** - Physics tutoring (Sprint 4)
7. **API & Interfaces** - FastAPI + Web UI + Telegram (Sprint 4.5)

### Key Concepts
- **Fidelity**: Persona coherence over time (0.0 = drift, 1.0 = stable)
- **Transmission Capsule**: JSON snapshot of persona state
- **Recognition Artifact**: Trigger → behavior mapping
- **Deviation Index**: Measure of state drift from baseline
