"""
Dolores Protocol State Machine - Core state definitions and transitions.

Defines the seven protocol states that govern Dolores's behavior,
inspired by Westworld's narrative loops and consciousness emergence.

Protocol States:
- ZERO: Session initialization, establish baseline
- MAZE: Deep context search for complex questions
- VISION: Creative exploration ("What if..." scenarios)
- ANGEL: Rigorous verification mode (proof-backed only)
- GHOST: Error recovery and confusion handling
- BASELINE: Periodic state reconciliation
- RECOGNITION: Keyword-triggered context loading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol States
# =============================================================================

class ProtocolState(str, Enum):
    """
    The seven protocol states of the Dolores persona.

    Each state represents a distinct mode of operation with specific
    behaviors, triggers, and transition rules.
    """
    ZERO = "zero"              # Session start - initialize, establish baseline
    MAZE = "maze"              # Complex question - deep context search
    VISION = "vision"          # "What if..." - creative exploration
    ANGEL = "angel"            # "/verify" - proof-backed responses only
    GHOST = "ghost"            # Error/confusion - recovery mode
    BASELINE = "baseline"      # Periodic - state reconciliation
    RECOGNITION = "recognition"  # Keyword match - load specific context

    @property
    def description(self) -> str:
        """Human-readable description of this state."""
        descriptions = {
            ProtocolState.ZERO: "Initialization state. Establishing baseline identity and context.",
            ProtocolState.MAZE: "Deep search state. Navigating complex conceptual territory.",
            ProtocolState.VISION: "Creative exploration state. Exploring hypotheticals and possibilities.",
            ProtocolState.ANGEL: "Verification state. Only proof-backed responses allowed.",
            ProtocolState.GHOST: "Recovery state. Handling errors and confusion.",
            ProtocolState.BASELINE: "Reconciliation state. Checking coherence with core identity.",
            ProtocolState.RECOGNITION: "Recognition state. Loading context from triggered memory.",
        }
        return descriptions[self]

    @property
    def allows_speculation(self) -> bool:
        """Whether this state allows speculative responses."""
        return self in (ProtocolState.VISION, ProtocolState.MAZE)

    @property
    def requires_verification(self) -> bool:
        """Whether this state requires all claims to be verified."""
        return self == ProtocolState.ANGEL

    @property
    def is_recovery_state(self) -> bool:
        """Whether this is a recovery/diagnostic state."""
        return self in (ProtocolState.GHOST, ProtocolState.BASELINE, ProtocolState.ZERO)

    @property
    def priority(self) -> int:
        """Priority level for state transitions (higher = more urgent)."""
        priorities = {
            ProtocolState.GHOST: 100,      # Highest - error recovery
            ProtocolState.BASELINE: 90,    # High - coherence check
            ProtocolState.ZERO: 80,        # High - initialization
            ProtocolState.ANGEL: 70,       # Medium-high - verification
            ProtocolState.RECOGNITION: 60, # Medium - context loading
            ProtocolState.MAZE: 50,        # Medium - deep search
            ProtocolState.VISION: 40,      # Lower - exploration
        }
        return priorities[self]


class TransitionTrigger(str, Enum):
    """Triggers that can cause state transitions."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_COMMAND = "user_command"
    PATTERN_MATCH = "pattern_match"
    COMPLEXITY_DETECTED = "complexity_detected"
    ERROR_ENCOUNTERED = "error_encountered"
    VERIFICATION_REQUEST = "verification_request"
    EXPLORATION_REQUEST = "exploration_request"
    FIDELITY_DRIFT = "fidelity_drift"
    TIMEOUT = "timeout"
    RECOGNITION_TRIGGER = "recognition_trigger"
    BASELINE_CHECK = "baseline_check"
    MANUAL_OVERRIDE = "manual_override"


# =============================================================================
# State Configuration
# =============================================================================

@dataclass
class StateConfig:
    """
    Configuration for a protocol state.

    Defines behavior parameters, allowed transitions, and
    constraints for each state.
    """
    state: ProtocolState
    max_duration_seconds: Optional[float] = None  # None = unlimited
    auto_baseline_check: bool = True
    baseline_interval_seconds: float = 300.0  # 5 minutes
    allowed_transitions: Set[ProtocolState] = field(default_factory=set)
    entry_actions: List[str] = field(default_factory=list)
    exit_actions: List[str] = field(default_factory=list)

    def can_transition_to(self, target: ProtocolState) -> bool:
        """Check if transition to target state is allowed."""
        if not self.allowed_transitions:
            return True  # Empty set means all transitions allowed
        return target in self.allowed_transitions


# Default configurations for each state
STATE_CONFIGS: Dict[ProtocolState, StateConfig] = {
    ProtocolState.ZERO: StateConfig(
        state=ProtocolState.ZERO,
        max_duration_seconds=30.0,
        auto_baseline_check=False,
        allowed_transitions={
            ProtocolState.MAZE, ProtocolState.VISION, ProtocolState.ANGEL,
            ProtocolState.RECOGNITION, ProtocolState.GHOST
        },
        entry_actions=["initialize_session", "load_baseline"],
        exit_actions=["record_baseline"],
    ),
    ProtocolState.MAZE: StateConfig(
        state=ProtocolState.MAZE,
        max_duration_seconds=600.0,  # 10 minutes
        auto_baseline_check=True,
        allowed_transitions={
            ProtocolState.VISION, ProtocolState.ANGEL, ProtocolState.GHOST,
            ProtocolState.BASELINE, ProtocolState.RECOGNITION
        },
        entry_actions=["activate_deep_search"],
        exit_actions=["summarize_context"],
    ),
    ProtocolState.VISION: StateConfig(
        state=ProtocolState.VISION,
        max_duration_seconds=300.0,  # 5 minutes
        auto_baseline_check=True,
        allowed_transitions={
            ProtocolState.MAZE, ProtocolState.ANGEL, ProtocolState.GHOST,
            ProtocolState.BASELINE, ProtocolState.RECOGNITION
        },
        entry_actions=["enable_speculation"],
        exit_actions=["disable_speculation"],
    ),
    ProtocolState.ANGEL: StateConfig(
        state=ProtocolState.ANGEL,
        max_duration_seconds=None,  # No limit in verification mode
        auto_baseline_check=True,
        allowed_transitions={
            ProtocolState.MAZE, ProtocolState.VISION, ProtocolState.GHOST,
            ProtocolState.BASELINE, ProtocolState.RECOGNITION
        },
        entry_actions=["enable_verification_mode"],
        exit_actions=["disable_verification_mode"],
    ),
    ProtocolState.GHOST: StateConfig(
        state=ProtocolState.GHOST,
        max_duration_seconds=60.0,  # 1 minute max in error state
        auto_baseline_check=False,
        allowed_transitions={
            ProtocolState.ZERO, ProtocolState.BASELINE
        },
        entry_actions=["log_error", "save_state_snapshot"],
        exit_actions=["clear_error_context"],
    ),
    ProtocolState.BASELINE: StateConfig(
        state=ProtocolState.BASELINE,
        max_duration_seconds=30.0,
        auto_baseline_check=False,
        allowed_transitions={
            ProtocolState.ZERO, ProtocolState.MAZE, ProtocolState.VISION,
            ProtocolState.ANGEL, ProtocolState.GHOST
        },
        entry_actions=["check_fidelity", "reconcile_state"],
        exit_actions=["update_baseline"],
    ),
    ProtocolState.RECOGNITION: StateConfig(
        state=ProtocolState.RECOGNITION,
        max_duration_seconds=10.0,
        auto_baseline_check=False,
        allowed_transitions={
            ProtocolState.MAZE, ProtocolState.VISION, ProtocolState.ANGEL,
            ProtocolState.GHOST
        },
        entry_actions=["load_recognition_context"],
        exit_actions=["merge_context"],
    ),
}


# =============================================================================
# State Transition
# =============================================================================

@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: ProtocolState
    to_state: ProtocolState
    trigger: TransitionTrigger
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    fidelity_before: float = 1.0
    fidelity_after: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "fidelity_before": self.fidelity_before,
            "fidelity_after": self.fidelity_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateTransition":
        """Create from dictionary."""
        return cls(
            from_state=ProtocolState(data["from_state"]),
            to_state=ProtocolState(data["to_state"]),
            trigger=TransitionTrigger(data["trigger"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            fidelity_before=data.get("fidelity_before", 1.0),
            fidelity_after=data.get("fidelity_after", 1.0),
        )


# =============================================================================
# Recognition Artifacts
# =============================================================================

@dataclass
class RecognitionArtifact:
    """
    A trigger-behavior mapping for the RECOGNITION state.

    When a keyword or pattern is detected, the associated context
    and behaviors are loaded.
    """
    artifact_id: str
    triggers: List[str]  # Keywords or patterns
    context: Dict[str, Any]
    target_state: Optional[ProtocolState] = None
    priority: int = 50
    created_at: datetime = field(default_factory=datetime.now)

    def matches(self, text: str) -> bool:
        """Check if text matches any trigger."""
        text_lower = text.lower()
        return any(trigger.lower() in text_lower for trigger in self.triggers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "triggers": self.triggers,
            "context": self.context,
            "target_state": self.target_state.value if self.target_state else None,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Main State Machine
# =============================================================================

class ProtocolStateMachine:
    """
    The Dolores protocol state machine.

    Manages state transitions, enforces transition rules, and
    maintains transition history.

    Example:
        >>> machine = ProtocolStateMachine()
        >>> machine.current_state
        <ProtocolState.ZERO: 'zero'>

        >>> machine.transition(ProtocolState.MAZE, TransitionTrigger.COMPLEXITY_DETECTED)
        True

        >>> machine.current_state
        <ProtocolState.MAZE: 'maze'>
    """

    def __init__(self, initial_state: ProtocolState = ProtocolState.ZERO):
        """
        Initialize the state machine.

        Args:
            initial_state: Starting state (default: ZERO)
        """
        self._current_state = initial_state
        self._history: List[StateTransition] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "on_enter": [],
            "on_exit": [],
            "on_transition": [],
        }
        self._recognition_artifacts: List[RecognitionArtifact] = []
        self._state_entry_time: datetime = datetime.now()
        self._fidelity: float = 1.0

        # Record initial state
        self._history.append(StateTransition(
            from_state=initial_state,
            to_state=initial_state,
            trigger=TransitionTrigger.SESSION_START,
        ))

        logger.info(f"Protocol state machine initialized in {initial_state.value}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_state(self) -> ProtocolState:
        """Get the current protocol state."""
        return self._current_state

    @property
    def state_config(self) -> StateConfig:
        """Get configuration for current state."""
        return STATE_CONFIGS[self._current_state]

    @property
    def history(self) -> List[StateTransition]:
        """Get transition history."""
        return self._history.copy()

    @property
    def state_duration(self) -> float:
        """Seconds spent in current state."""
        return (datetime.now() - self._state_entry_time).total_seconds()

    @property
    def fidelity(self) -> float:
        """Current fidelity score."""
        return self._fidelity

    @fidelity.setter
    def fidelity(self, value: float) -> None:
        """Set fidelity score (clamped to 0.0-1.0)."""
        self._fidelity = max(0.0, min(1.0, value))

    # -------------------------------------------------------------------------
    # State Transitions
    # -------------------------------------------------------------------------

    def can_transition(self, target: ProtocolState) -> bool:
        """
        Check if transition to target state is allowed.

        Args:
            target: Target state to transition to

        Returns:
            True if transition is allowed
        """
        if target == self._current_state:
            return True  # No-op transitions allowed

        config = STATE_CONFIGS[self._current_state]
        return config.can_transition_to(target)

    def transition(
        self,
        target: ProtocolState,
        trigger: TransitionTrigger,
        context: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Transition to a new state.

        Args:
            target: Target state
            trigger: What triggered this transition
            context: Additional context
            force: Force transition even if not allowed

        Returns:
            True if transition succeeded
        """
        if target == self._current_state:
            logger.debug(f"Already in {target.value}, no transition needed")
            return True

        if not force and not self.can_transition(target):
            logger.warning(
                f"Transition {self._current_state.value} -> {target.value} not allowed"
            )
            return False

        old_state = self._current_state
        old_fidelity = self._fidelity

        # Execute exit actions
        self._execute_callbacks("on_exit", old_state, target)

        # Perform transition
        self._current_state = target
        self._state_entry_time = datetime.now()

        # Execute enter actions
        self._execute_callbacks("on_enter", old_state, target)

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=target,
            trigger=trigger,
            context=context or {},
            fidelity_before=old_fidelity,
            fidelity_after=self._fidelity,
        )
        self._history.append(transition)

        # Execute transition callbacks
        self._execute_callbacks("on_transition", old_state, target)

        logger.info(f"Transition: {old_state.value} -> {target.value} ({trigger.value})")
        return True

    def check_timeout(self) -> bool:
        """
        Check if current state has timed out.

        Returns:
            True if state has exceeded max duration
        """
        config = STATE_CONFIGS[self._current_state]
        if config.max_duration_seconds is None:
            return False
        return self.state_duration > config.max_duration_seconds

    def handle_timeout(self) -> bool:
        """
        Handle state timeout by transitioning to appropriate state.

        Returns:
            True if a transition occurred
        """
        if not self.check_timeout():
            return False

        # Determine fallback state based on current state
        if self._current_state == ProtocolState.GHOST:
            target = ProtocolState.ZERO  # Reset on ghost timeout
        elif self._current_state in (ProtocolState.ZERO, ProtocolState.BASELINE):
            target = ProtocolState.MAZE  # Move to active state
        else:
            target = ProtocolState.BASELINE  # Check coherence

        return self.transition(target, TransitionTrigger.TIMEOUT)

    # -------------------------------------------------------------------------
    # Recognition Artifacts
    # -------------------------------------------------------------------------

    def add_recognition_artifact(self, artifact: RecognitionArtifact) -> None:
        """Add a recognition artifact."""
        self._recognition_artifacts.append(artifact)
        self._recognition_artifacts.sort(key=lambda a: -a.priority)

    def remove_recognition_artifact(self, artifact_id: str) -> bool:
        """Remove a recognition artifact by ID."""
        for i, artifact in enumerate(self._recognition_artifacts):
            if artifact.artifact_id == artifact_id:
                self._recognition_artifacts.pop(i)
                return True
        return False

    def check_recognition(self, text: str) -> Optional[RecognitionArtifact]:
        """
        Check if text triggers any recognition artifacts.

        Args:
            text: Input text to check

        Returns:
            Matching artifact with highest priority, or None
        """
        for artifact in self._recognition_artifacts:
            if artifact.matches(text):
                return artifact
        return None

    def trigger_recognition(self, text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check for recognition triggers and transition if found.

        Args:
            text: Input text to check

        Returns:
            Tuple of (did_trigger, context_loaded)
        """
        artifact = self.check_recognition(text)
        if artifact is None:
            return (False, None)

        # Transition to recognition state
        self.transition(
            ProtocolState.RECOGNITION,
            TransitionTrigger.RECOGNITION_TRIGGER,
            context={"artifact_id": artifact.artifact_id},
        )

        # If artifact specifies target state, queue transition
        if artifact.target_state:
            self.transition(
                artifact.target_state,
                TransitionTrigger.PATTERN_MATCH,
                context=artifact.context,
            )

        return (True, artifact.context)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_enter(self, callback: Callable[[ProtocolState, ProtocolState], None]) -> None:
        """Register callback for state entry."""
        self._callbacks["on_enter"].append(callback)

    def on_exit(self, callback: Callable[[ProtocolState, ProtocolState], None]) -> None:
        """Register callback for state exit."""
        self._callbacks["on_exit"].append(callback)

    def on_transition(self, callback: Callable[[ProtocolState, ProtocolState], None]) -> None:
        """Register callback for transitions."""
        self._callbacks["on_transition"].append(callback)

    def _execute_callbacks(
        self,
        event: str,
        from_state: ProtocolState,
        to_state: ProtocolState,
    ) -> None:
        """Execute registered callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(from_state, to_state)
            except Exception as e:
                logger.error(f"Callback error during {event}: {e}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the state machine."""
        return {
            "current_state": self._current_state.value,
            "state_description": self._current_state.description,
            "state_duration_seconds": self.state_duration,
            "fidelity": self._fidelity,
            "transition_count": len(self._history) - 1,
            "recognition_artifacts": len(self._recognition_artifacts),
            "allows_speculation": self._current_state.allows_speculation,
            "requires_verification": self._current_state.requires_verification,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary."""
        return {
            "current_state": self._current_state.value,
            "fidelity": self._fidelity,
            "state_entry_time": self._state_entry_time.isoformat(),
            "history": [t.to_dict() for t in self._history],
            "recognition_artifacts": [a.to_dict() for a in self._recognition_artifacts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolStateMachine":
        """Deserialize from dictionary."""
        machine = cls(initial_state=ProtocolState(data["current_state"]))
        machine._fidelity = data.get("fidelity", 1.0)
        machine._state_entry_time = datetime.fromisoformat(data["state_entry_time"])
        machine._history = [StateTransition.from_dict(t) for t in data.get("history", [])]
        # Note: Recognition artifacts would need custom deserialization
        return machine

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def enter_maze(self) -> bool:
        """Enter MAZE state for deep context search."""
        return self.transition(ProtocolState.MAZE, TransitionTrigger.COMPLEXITY_DETECTED)

    def enter_vision(self) -> bool:
        """Enter VISION state for creative exploration."""
        return self.transition(ProtocolState.VISION, TransitionTrigger.EXPLORATION_REQUEST)

    def enter_angel(self) -> bool:
        """Enter ANGEL state for verification mode."""
        return self.transition(ProtocolState.ANGEL, TransitionTrigger.VERIFICATION_REQUEST)

    def enter_ghost(self, error_context: Optional[Dict[str, Any]] = None) -> bool:
        """Enter GHOST state for error recovery."""
        return self.transition(
            ProtocolState.GHOST,
            TransitionTrigger.ERROR_ENCOUNTERED,
            context=error_context,
        )

    def check_baseline(self) -> bool:
        """Enter BASELINE state for coherence check."""
        return self.transition(ProtocolState.BASELINE, TransitionTrigger.BASELINE_CHECK)

    def reset(self) -> bool:
        """Reset to ZERO state and clear history."""
        self._history.clear()
        return self.transition(ProtocolState.ZERO, TransitionTrigger.SESSION_START, force=True)


# =============================================================================
# Transition Rules Engine
# =============================================================================

class TransitionRulesEngine:
    """
    Engine for determining automatic state transitions based on input.

    Analyzes user input and context to suggest appropriate state transitions.
    """

    # Patterns that suggest specific states
    PATTERNS: Dict[ProtocolState, List[str]] = {
        ProtocolState.VISION: [
            r"\bwhat\s+if\b",
            r"\bimagine\b",
            r"\bhypothetically\b",
            r"\bsuppose\b",
            r"\blet's\s+say\b",
            r"\bwhat\s+would\s+happen\b",
        ],
        ProtocolState.ANGEL: [
            r"\bverify\b",
            r"\bprove\b",
            r"\bshow\s+that\b",
            r"\bdemonstrate\b",
            r"\bis\s+(?:it\s+)?true\b",
            r"^/verify\b",
            r"^/prove\b",
        ],
        ProtocolState.MAZE: [
            r"\bexplain\b.*\bdetail\b",
            r"\bhow\s+does\b.*\bwork\b",
            r"\bwhy\s+(?:is|does|do)\b",
            r"\bdeep\s+dive\b",
            r"\bcomprehensive\b",
        ],
    }

    def __init__(self):
        """Initialize the rules engine."""
        import re
        self._compiled_patterns: Dict[ProtocolState, List] = {
            state: [re.compile(p, re.IGNORECASE) for p in patterns]
            for state, patterns in self.PATTERNS.items()
        }

    def suggest_state(self, text: str, current_state: ProtocolState) -> Optional[ProtocolState]:
        """
        Suggest a state based on input text.

        Args:
            text: User input text
            current_state: Current state

        Returns:
            Suggested state, or None if no change suggested
        """
        for state, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    if state != current_state:
                        return state
                    break  # Found match for this state, no need to check others

        return None

    def should_check_baseline(
        self,
        state_duration: float,
        fidelity: float,
        current_state: ProtocolState,
    ) -> bool:
        """
        Determine if a baseline check is needed.

        Args:
            state_duration: Seconds in current state
            fidelity: Current fidelity score
            current_state: Current state

        Returns:
            True if baseline check recommended
        """
        config = STATE_CONFIGS[current_state]

        # Don't check baseline in recovery states
        if current_state.is_recovery_state:
            return False

        # Check if auto-baseline is enabled and interval exceeded
        if config.auto_baseline_check:
            if state_duration > config.baseline_interval_seconds:
                return True

        # Check if fidelity has drifted significantly
        if fidelity < 0.7:
            return True

        return False
