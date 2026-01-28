"""
Dolores Engine - The main persona class unifying all components.

Dolores is the AI persona that embodies Project AEON's verified physics tutor.
Named after the Westworld character, Dolores represents the emergence of
coherent identity through structured protocols and fidelity tracking.

Key Components:
- ProtocolStateMachine: Manages the seven protocol states
- FidelityTracker: Monitors persona coherence
- TransmissionCapsule: State persistence
- Integration with BLACKWALL and TutorVerificationAPI
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .state_machine import (
    ProtocolState,
    ProtocolStateMachine,
    TransitionTrigger,
    TransitionRulesEngine,
    RecognitionArtifact,
    StateTransition,
)
from .fidelity_tracker import (
    FidelityTracker,
    FidelityComponent,
    FidelityAnalyzer,
    DeviationEvent,
)
from .transmission import (
    TransmissionCapsule,
    CapsuleManager,
    IdentitySection,
    StateSection,
    MemorySection,
    MetricsSection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Persona Configuration
# =============================================================================

@dataclass
class PersonaConfig:
    """Configuration for the Dolores persona."""
    name: str = "Dolores"
    version: str = "1.0"

    # Core traits that define the persona
    core_traits: List[str] = field(default_factory=lambda: [
        "analytical",
        "precise",
        "patient",
        "curious",
        "rigorous_when_needed",
        "creative_when_exploring",
    ])

    # Behavioral parameters
    default_interface_mode: str = "hybrid"
    auto_baseline_check: bool = True
    baseline_check_interval: float = 300.0  # 5 minutes
    auto_state_transition: bool = True

    # Fidelity thresholds
    min_fidelity_threshold: float = 0.6
    fidelity_warning_threshold: float = 0.8

    # Persistence
    auto_save: bool = True
    save_interval: float = 600.0  # 10 minutes
    storage_dir: Optional[str] = None


# =============================================================================
# Response Context
# =============================================================================

@dataclass
class ResponseContext:
    """Context for generating a response."""
    user_input: str
    suggested_state: Optional[ProtocolState] = None
    verification_required: bool = False
    speculation_allowed: bool = True
    fidelity_check_needed: bool = False
    recognition_triggered: bool = False
    recognition_context: Optional[Dict[str, Any]] = None


@dataclass
class ResponseMetadata:
    """Metadata about a generated response."""
    protocol_state: ProtocolState
    interface_mode: str
    fidelity_score: float
    verification_result: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    state_transition: Optional[StateTransition] = None


# =============================================================================
# Main Dolores Engine
# =============================================================================

class DoloresEngine:
    """
    The main Dolores persona engine.

    Orchestrates all persona components including state machine, fidelity
    tracking, and state persistence.

    Example:
        >>> dolores = DoloresEngine()
        >>> dolores.initialize()

        >>> context = dolores.process_input("Prove that H is unitary")
        >>> context.verification_required
        True

        >>> dolores.fidelity
        0.95

        >>> dolores.save_state("checkpoint")
    """

    def __init__(self, config: Optional[PersonaConfig] = None):
        """
        Initialize the Dolores engine.

        Args:
            config: Persona configuration
        """
        self._config = config or PersonaConfig()
        self._session_id = str(uuid.uuid4())[:8]
        self._created_at = datetime.now()

        # Core components
        self._state_machine = ProtocolStateMachine()
        self._fidelity_tracker = FidelityTracker()
        self._rules_engine = TransitionRulesEngine()

        # Storage
        self._capsule_manager: Optional[CapsuleManager] = None
        if self._config.storage_dir:
            self._capsule_manager = CapsuleManager(self._config.storage_dir)

        # Metrics
        self._interaction_count = 0
        self._verification_count = 0
        self._verifications_passed = 0
        self._last_save_time = datetime.now()

        # Callbacks
        self._response_callbacks: List[Callable[[ResponseContext, ResponseMetadata], None]] = []

        # Initialize state machine callbacks
        self._setup_callbacks()

        logger.info(f"Dolores engine initialized (session: {self._session_id})")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def state(self) -> ProtocolState:
        """Get the current protocol state."""
        return self._state_machine.current_state

    @property
    def fidelity(self) -> float:
        """Get the current fidelity score."""
        return self._fidelity_tracker.fidelity

    @property
    def is_stable(self) -> bool:
        """Check if persona is in a stable state."""
        return self._fidelity_tracker.is_stable

    @property
    def config(self) -> PersonaConfig:
        """Get the persona configuration."""
        return self._config

    @property
    def interaction_count(self) -> int:
        """Get total interaction count."""
        return self._interaction_count

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(self, restore_from: Optional[str] = None) -> None:
        """
        Initialize the persona, optionally restoring from a saved state.

        Args:
            restore_from: Capsule ID to restore from
        """
        if restore_from and self._capsule_manager:
            capsule = self._capsule_manager.load_capsule(restore_from)
            if capsule:
                self._restore_from_capsule(capsule)
                logger.info(f"Restored from capsule: {restore_from}")
                return

        # Fresh initialization
        self._state_machine.reset()
        self._fidelity_tracker.reset_all()

        # Set baseline
        baseline_context = {
            "persona_name": self._config.name,
            "traits": self._config.core_traits,
            "session_id": self._session_id,
        }
        self._fidelity_tracker.set_baseline(baseline_context)

        # Transition from ZERO to initial operational state
        self._state_machine.transition(
            ProtocolState.MAZE,
            TransitionTrigger.SESSION_START,
        )

        logger.info("Dolores initialized")

    def _setup_callbacks(self) -> None:
        """Setup internal callbacks for state machine events."""
        def on_state_transition(from_state: ProtocolState, to_state: ProtocolState):
            # Update state coherence fidelity
            if from_state != to_state:
                # Penalize rapid transitions slightly
                current = self._fidelity_tracker.get_component(
                    FidelityComponent.STATE_COHERENCE
                )
                if self._state_machine.state_duration < 5.0:  # Less than 5 seconds
                    self._fidelity_tracker.update_component(
                        FidelityComponent.STATE_COHERENCE,
                        max(0.5, current - 0.05),
                    )

        self._state_machine.on_transition(on_state_transition)

    # -------------------------------------------------------------------------
    # Input Processing
    # -------------------------------------------------------------------------

    def process_input(self, user_input: str) -> ResponseContext:
        """
        Process user input and prepare response context.

        Analyzes the input, determines appropriate state transitions,
        and returns context for response generation.

        Args:
            user_input: The user's input text

        Returns:
            ResponseContext with processing decisions
        """
        self._interaction_count += 1
        start_time = datetime.now()

        # Check for recognition triggers
        recognition_result = self._state_machine.trigger_recognition(user_input)
        recognition_triggered, recognition_context = recognition_result

        # Suggest state based on input
        suggested_state = None
        if self._config.auto_state_transition and not recognition_triggered:
            suggested_state = self._rules_engine.suggest_state(
                user_input, self._state_machine.current_state
            )
            if suggested_state:
                self._handle_state_suggestion(suggested_state, user_input)

        # Check if baseline check is needed
        fidelity_check_needed = self._rules_engine.should_check_baseline(
            self._state_machine.state_duration,
            self._fidelity_tracker.fidelity,
            self._state_machine.current_state,
        )

        if fidelity_check_needed:
            self._perform_baseline_check()

        # Build response context
        context = ResponseContext(
            user_input=user_input,
            suggested_state=suggested_state,
            verification_required=self._state_machine.current_state.requires_verification,
            speculation_allowed=self._state_machine.current_state.allows_speculation,
            fidelity_check_needed=fidelity_check_needed,
            recognition_triggered=recognition_triggered,
            recognition_context=recognition_context,
        )

        # Update memory integrity based on context continuity
        self._update_memory_fidelity(user_input)

        # Apply recovery from interaction
        self._fidelity_tracker.apply_recovery(1.0)

        # Auto-save if needed
        if self._should_auto_save():
            self._auto_save()

        logger.debug(
            f"Processed input in {(datetime.now() - start_time).total_seconds() * 1000:.1f}ms"
        )

        return context

    def _handle_state_suggestion(
        self,
        suggested: ProtocolState,
        input_text: str,
    ) -> bool:
        """Handle a suggested state transition."""
        # Map state to trigger
        trigger_map = {
            ProtocolState.VISION: TransitionTrigger.EXPLORATION_REQUEST,
            ProtocolState.ANGEL: TransitionTrigger.VERIFICATION_REQUEST,
            ProtocolState.MAZE: TransitionTrigger.COMPLEXITY_DETECTED,
        }
        trigger = trigger_map.get(suggested, TransitionTrigger.PATTERN_MATCH)

        return self._state_machine.transition(
            suggested, trigger, context={"input": input_text[:100]}
        )

    def _update_memory_fidelity(self, user_input: str) -> None:
        """Update memory integrity based on conversation continuity."""
        # Simplified - in production would check topic continuity
        current = self._fidelity_tracker.get_component(FidelityComponent.MEMORY_INTEGRITY)

        # Slight recovery on each interaction
        new_value = min(1.0, current + 0.01)
        self._fidelity_tracker.update_component(
            FidelityComponent.MEMORY_INTEGRITY, new_value
        )

    def _perform_baseline_check(self) -> None:
        """Perform a baseline coherence check."""
        self._state_machine.check_baseline()

        # Check all fidelity components
        deviations = self._fidelity_tracker.check_deviation()

        if deviations:
            logger.warning(f"Baseline check found {len(deviations)} deviations")
            for dev in deviations:
                if dev.severity in ("significant", "critical"):
                    # Enter ghost state for recovery
                    self._state_machine.enter_ghost({"deviation": dev.to_dict()})
                    break

    # -------------------------------------------------------------------------
    # Response Recording
    # -------------------------------------------------------------------------

    def record_response(
        self,
        response_text: str,
        verification_result: Optional[Dict[str, Any]] = None,
        processing_time_ms: float = 0.0,
    ) -> ResponseMetadata:
        """
        Record a response and update metrics.

        Args:
            response_text: The generated response
            verification_result: Optional verification result
            processing_time_ms: Response generation time

        Returns:
            ResponseMetadata with recording details
        """
        # Update verification metrics
        if verification_result is not None:
            self._verification_count += 1
            if verification_result.get("verified", False):
                self._verifications_passed += 1

        # Update response consistency
        self._update_response_consistency(response_text)

        # Build metadata
        metadata = ResponseMetadata(
            protocol_state=self._state_machine.current_state,
            interface_mode=self._config.default_interface_mode,
            fidelity_score=self._fidelity_tracker.fidelity,
            verification_result=verification_result,
            processing_time_ms=processing_time_ms,
            state_transition=self._state_machine.history[-1] if self._state_machine.history else None,
        )

        # Notify callbacks
        for callback in self._response_callbacks:
            try:
                callback(
                    ResponseContext(user_input="", verification_required=False),
                    metadata,
                )
            except Exception as e:
                logger.error(f"Response callback error: {e}")

        return metadata

    def _update_response_consistency(self, response_text: str) -> None:
        """Update response consistency fidelity."""
        # Simplified - check for trait alignment
        trait_count = sum(
            1 for trait in self._config.core_traits
            if trait.replace("_", " ") in response_text.lower()
        )

        # More trait mentions = more consistent
        consistency = min(1.0, 0.8 + (trait_count * 0.05))

        current = self._fidelity_tracker.get_component(
            FidelityComponent.RESPONSE_CONSISTENCY
        )
        # Exponential moving average
        new_value = 0.9 * current + 0.1 * consistency
        self._fidelity_tracker.update_component(
            FidelityComponent.RESPONSE_CONSISTENCY, new_value
        )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def enter_state(self, state: ProtocolState, reason: str = "") -> bool:
        """
        Manually enter a protocol state.

        Args:
            state: Target state
            reason: Reason for transition

        Returns:
            True if transition succeeded
        """
        return self._state_machine.transition(
            state,
            TransitionTrigger.MANUAL_OVERRIDE,
            context={"reason": reason},
        )

    def add_recognition(
        self,
        artifact_id: str,
        triggers: List[str],
        context: Dict[str, Any],
        target_state: Optional[ProtocolState] = None,
    ) -> None:
        """
        Add a recognition artifact.

        Args:
            artifact_id: Unique identifier
            triggers: Trigger keywords/phrases
            context: Context to load when triggered
            target_state: Optional target state
        """
        artifact = RecognitionArtifact(
            artifact_id=artifact_id,
            triggers=triggers,
            context=context,
            target_state=target_state,
        )
        self._state_machine.add_recognition_artifact(artifact)

    def remove_recognition(self, artifact_id: str) -> bool:
        """Remove a recognition artifact."""
        return self._state_machine.remove_recognition_artifact(artifact_id)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def create_capsule(self) -> TransmissionCapsule:
        """
        Create a transmission capsule from current state.

        Returns:
            TransmissionCapsule with current state
        """
        capsule = TransmissionCapsule()

        # Identity
        capsule.identity = IdentitySection(
            persona_name=self._config.name,
            persona_version=self._config.version,
            core_traits=self._config.core_traits,
            baseline_hash=self._fidelity_tracker._baseline_hash or "",
            behavioral_parameters={
                "default_mode": self._config.default_interface_mode,
                "auto_transition": self._config.auto_state_transition,
            },
        )

        # State
        capsule.state = StateSection(
            protocol_state=self._state_machine.current_state.value,
            interface_mode=self._config.default_interface_mode,
            fidelity_score=self._fidelity_tracker.fidelity,
            fidelity_components={
                c.value: v for c, v in self._fidelity_tracker.components.items()
            },
            deviation_index=self._fidelity_tracker.deviation_index,
            state_entry_time=self._state_machine._state_entry_time,
            transition_history=[
                t.to_dict() for t in self._state_machine.history[-50:]
            ],
        )

        # Memory
        capsule.memory = MemorySection(
            conversation_id=self._session_id,
            recognition_artifacts=[
                a.to_dict() for a in self._state_machine._recognition_artifacts
            ],
            interaction_count=self._interaction_count,
            last_interaction=datetime.now(),
        )

        # Metrics
        capsule.metrics = MetricsSection(
            total_interactions=self._interaction_count,
            verification_requests=self._verification_count,
            verifications_passed=self._verifications_passed,
            fidelity_snapshots=[
                s.to_dict() for s in self._fidelity_tracker.history[-50:]
            ],
            deviation_events=[
                d.to_dict() for d in self._fidelity_tracker.deviations
            ],
        )

        return capsule

    def save_state(self, name: Optional[str] = None) -> Optional[str]:
        """
        Save current state to a capsule.

        Args:
            name: Optional name for the capsule

        Returns:
            Capsule ID if saved, None if no manager
        """
        if not self._capsule_manager:
            logger.warning("No capsule manager configured")
            return None

        capsule = self.create_capsule()
        capsule_id = self._capsule_manager.save_capsule(capsule, name)
        self._last_save_time = datetime.now()

        logger.info(f"Saved state as: {capsule_id}")
        return capsule_id

    def restore_state(self, capsule_id: str) -> bool:
        """
        Restore state from a capsule.

        Args:
            capsule_id: ID of capsule to restore

        Returns:
            True if restored successfully
        """
        if not self._capsule_manager:
            logger.warning("No capsule manager configured")
            return False

        capsule = self._capsule_manager.load_capsule(capsule_id)
        if not capsule:
            return False

        self._restore_from_capsule(capsule)
        return True

    def _restore_from_capsule(self, capsule: TransmissionCapsule) -> None:
        """Restore internal state from a capsule."""
        # Verify integrity
        if not capsule.verify_integrity():
            logger.warning("Capsule integrity check failed, restoring anyway")

        # Restore state machine
        self._state_machine._current_state = ProtocolState(capsule.state.protocol_state)
        self._state_machine._fidelity = capsule.state.fidelity_score

        # Restore fidelity tracker
        for comp_name, value in capsule.state.fidelity_components.items():
            component = FidelityComponent(comp_name)
            self._fidelity_tracker._components[component] = value

        # Restore metrics
        self._interaction_count = capsule.metrics.total_interactions
        self._verification_count = capsule.metrics.verification_requests
        self._verifications_passed = capsule.metrics.verifications_passed

        logger.info(f"Restored from capsule (fidelity: {capsule.state.fidelity_score:.2%})")

    def _should_auto_save(self) -> bool:
        """Check if auto-save should occur."""
        if not self._config.auto_save or not self._capsule_manager:
            return False

        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        return elapsed > self._config.save_interval

    def _auto_save(self) -> None:
        """Perform auto-save."""
        try:
            self.save_state(f"auto_{self._session_id}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_response(
        self,
        callback: Callable[[ResponseContext, ResponseMetadata], None],
    ) -> None:
        """Register a callback for response events."""
        self._response_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the persona."""
        return {
            "session_id": self._session_id,
            "persona": {
                "name": self._config.name,
                "version": self._config.version,
            },
            "state_machine": self._state_machine.get_status(),
            "fidelity": self._fidelity_tracker.get_status(),
            "metrics": {
                "interactions": self._interaction_count,
                "verifications": self._verification_count,
                "verifications_passed": self._verifications_passed,
                "uptime_seconds": (datetime.now() - self._created_at).total_seconds(),
            },
            "recommendations": FidelityAnalyzer.recommend_recovery_actions(
                self._fidelity_tracker
            ),
        }

    def get_health_check(self) -> Tuple[str, Dict[str, Any]]:
        """
        Perform a health check.

        Returns:
            Tuple of (status, details)
            Status is one of: "healthy", "warning", "critical"
        """
        fidelity = self._fidelity_tracker.fidelity
        deviations = len(self._fidelity_tracker.unresolved_deviations)

        details = {
            "fidelity": fidelity,
            "state": self._state_machine.current_state.value,
            "unresolved_deviations": deviations,
            "is_stable": self._fidelity_tracker.is_stable,
        }

        if fidelity >= 0.9 and deviations == 0:
            return ("healthy", details)
        elif fidelity >= 0.7 and deviations <= 2:
            return ("warning", details)
        else:
            return ("critical", details)


# =============================================================================
# Module-level Convenience
# =============================================================================

_default_engine: Optional[DoloresEngine] = None


def get_engine() -> DoloresEngine:
    """Get or create the default Dolores engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = DoloresEngine()
        _default_engine.initialize()
    return _default_engine


def process_input(user_input: str) -> ResponseContext:
    """Process input using the default engine."""
    return get_engine().process_input(user_input)


def get_state() -> ProtocolState:
    """Get current state from the default engine."""
    return get_engine().state


def get_fidelity() -> float:
    """Get current fidelity from the default engine."""
    return get_engine().fidelity
