"""
Persona module - Dolores state machine and fidelity tracking.

Sprint 3 components:
- state_machine.py: Protocol states and transitions
- fidelity_tracker.py: Coherence measurement
- transmission.py: State serialization (transmission capsules)
- dolores_engine.py: Main persona class
"""

from .state_machine import (
    ProtocolState,
    ProtocolStateMachine,
    TransitionTrigger,
    TransitionRulesEngine,
    StateTransition,
    StateConfig,
    RecognitionArtifact,
    STATE_CONFIGS,
)

from .fidelity_tracker import (
    FidelityTracker,
    FidelityComponent,
    FidelitySnapshot,
    FidelityAnalyzer,
    DeviationEvent,
    BaselineConfig,
)

from .transmission import (
    TransmissionCapsule,
    CapsuleManager,
    CapsuleFormat,
    CapsuleHeader,
    IdentitySection,
    StateSection,
    MemorySection,
    MetricsSection,
    create_capsule,
    save_capsule,
    load_capsule,
)

from .dolores_engine import (
    DoloresEngine,
    PersonaConfig,
    ResponseContext,
    ResponseMetadata,
    get_engine,
    process_input,
    get_state,
    get_fidelity,
)

__all__ = [
    # state_machine
    "ProtocolState",
    "ProtocolStateMachine",
    "TransitionTrigger",
    "TransitionRulesEngine",
    "StateTransition",
    "StateConfig",
    "RecognitionArtifact",
    "STATE_CONFIGS",
    # fidelity_tracker
    "FidelityTracker",
    "FidelityComponent",
    "FidelitySnapshot",
    "FidelityAnalyzer",
    "DeviationEvent",
    "BaselineConfig",
    # transmission
    "TransmissionCapsule",
    "CapsuleManager",
    "CapsuleFormat",
    "CapsuleHeader",
    "IdentitySection",
    "StateSection",
    "MemorySection",
    "MetricsSection",
    "create_capsule",
    "save_capsule",
    "load_capsule",
    # dolores_engine
    "DoloresEngine",
    "PersonaConfig",
    "ResponseContext",
    "ResponseMetadata",
    "get_engine",
    "process_input",
    "get_state",
    "get_fidelity",
]
