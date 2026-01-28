"""
BLACKWALL Mode Controller - Adaptive cognitive interface for Project AEON.

The BLACKWALL controller manages the operational mode of the AI system,
switching between rigorous (proof-backed) and exploratory (creative) modes
based on user commands and query analysis.

Named after the boundary concept - a firewall between verified truth and
speculative exploration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .enums import InterfaceMode, RequestType, TransitionReason
from src.llm.interface import Message, MessageRole


@dataclass
class BlackwallResponse:
    content: str
    metadata: Dict


# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Mode Configuration
# =============================================================================

@dataclass
class ModeConfig:
    """
    Configuration for an interface mode.

    Stores the behavior parameters for each mode, including
    verification requirements and response formatting options.
    """
    mode: InterfaceMode
    require_proof: bool = False
    allow_speculation: bool = True
    min_confidence: float = 0.0
    max_retries: int = 3
    show_symbolic: bool = True
    format_preference: str = "markdown"

    # Response filtering
    reject_unverified: bool = False
    warn_low_confidence: bool = True
    confidence_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "require_proof": self.require_proof,
            "allow_speculation": self.allow_speculation,
            "min_confidence": self.min_confidence,
            "max_retries": self.max_retries,
            "show_symbolic": self.show_symbolic,
            "format_preference": self.format_preference,
            "reject_unverified": self.reject_unverified,
            "warn_low_confidence": self.warn_low_confidence,
            "confidence_threshold": self.confidence_threshold,
        }


# Default configurations for each mode
MODE_CONFIGS: Dict[InterfaceMode, ModeConfig] = {
    InterfaceMode.RIGOROUS: ModeConfig(
        mode=InterfaceMode.RIGOROUS,
        require_proof=True,
        allow_speculation=False,
        min_confidence=1.0,
        max_retries=5,
        show_symbolic=True,
        format_preference="latex",
        reject_unverified=True,
        warn_low_confidence=True,
        confidence_threshold=1.0,
    ),
    InterfaceMode.EXPLORATORY: ModeConfig(
        mode=InterfaceMode.EXPLORATORY,
        require_proof=False,
        allow_speculation=True,
        min_confidence=0.0,
        max_retries=1,
        show_symbolic=False,
        format_preference="plain",
        reject_unverified=False,
        warn_low_confidence=False,
        confidence_threshold=0.0,
    ),
    InterfaceMode.HYBRID: ModeConfig(
        mode=InterfaceMode.HYBRID,
        require_proof=False,
        allow_speculation=True,
        min_confidence=0.5,
        max_retries=3,
        show_symbolic=True,
        format_preference="markdown",
        reject_unverified=False,
        warn_low_confidence=True,
        confidence_threshold=0.7,
    ),
    InterfaceMode.SYSTEM: ModeConfig(
        mode=InterfaceMode.SYSTEM,
        require_proof=False,
        allow_speculation=False,
        min_confidence=0.0,
        max_retries=1,
        show_symbolic=False,
        format_preference="plain",
        reject_unverified=False,
        warn_low_confidence=False,
        confidence_threshold=0.0,
    ),
}


# =============================================================================
# Transition History
# =============================================================================

@dataclass
class ModeTransition:
    """Record of a mode transition."""
    from_mode: InterfaceMode
    to_mode: InterfaceMode
    reason: TransitionReason
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "reason": self.reason.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


# =============================================================================
# Main Controller Class
# =============================================================================

class BlackwallController:
    """
    Controller for the BLACKWALL adaptive cognitive interface.

    Manages mode state, handles transitions, and provides configuration
    for how the AI system should respond to different types of queries.

    Example:
        >>> controller = BlackwallController()
        >>> controller.current_mode
        <InterfaceMode.HYBRID: 'hybrid'>

        >>> controller.set_mode(InterfaceMode.RIGOROUS)
        >>> controller.get_config().require_proof
        True

        >>> controller.validate_response(result)
        True  # If result.confidence >= 1.0
    """

    def __init__(
        self,
        default_mode: InterfaceMode = InterfaceMode.HYBRID,
        auto_transition: bool = True,
        llm: Optional[Any] = None,
        verifier: Optional[Any] = None,
        persona_engine: Optional[Any] = None,
    ):
        """
        Initialize the BLACKWALL controller.

        Args:
            default_mode: Initial operational mode
            auto_transition: Whether to automatically suggest mode changes
            llm: LLM Provider instance
            verifier: TutorVerificationAPI instance
            persona_engine: DoloresEngine instance
        """
        self._current_mode = default_mode
        self._auto_transition = auto_transition
        self._history: List[ModeTransition] = []
        self._callbacks: List[Callable[[InterfaceMode, InterfaceMode], None]] = []
        self._custom_configs: Dict[InterfaceMode, ModeConfig] = {}
        
        # Dependencies
        self.llm = llm
        self.verifier = verifier
        self.persona = persona_engine

        # Record initial state
        self._history.append(ModeTransition(
            from_mode=default_mode,
            to_mode=default_mode,
            reason=TransitionReason.SESSION_START,
            context={"auto_transition": auto_transition},
        ))

        logger.info(f"BLACKWALL initialized in {default_mode.value} mode")

    def process_input(self, message: str) -> Any:
        """
        Main orchestration loop:
        1. Contextualize (add inputs to prompt)
        2. Verify (if claims detected)
        3. Generate (via LLM/Persona)
        """
        # 1. Verification Step (Connectivity to Sprint 1)
        verified_claims = []
        if self.verifier and self._current_mode != InterfaceMode.SYSTEM:
            # Simple heuristic: treat the whole message as a potential claim to verify
            # Real implementation would use ClaimParser from Sprint 1
            from src.tutor.verification_api import VerificationResult
            
            # TODO: Use ClaimParser here. For now, we try to verify atomic claims if they look like math.
            if "unitary" in message or "gate" in message or "|" in message:
                try:
                    # We pass the raw message as a claim for now
                    # Ideally: claims = claim_parser.parse(message)
                    pass
                except Exception as e:
                    logger.warning(f"Verification attempt failed: {e}")

        # 2. Persona/LLM Step (Connectivity to Sprint 4)
        response_text = "Analysis complete."
        if self.llm:
            # Construct a system prompt based on mode
            system_prompt = self._get_system_prompt()
            # Send to LLM
            # Send to LLM
            try:
                # Basic context message
                messages = [Message.user(message)]
                
                # Call LLM
                response = self.llm.complete(
                    messages=messages,
                    system=system_prompt,
                    temperature=0.7 if self._current_mode == InterfaceMode.EXPLORATORY else 0.2
                )
                response_text = response.content
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                response_text = f"Error communicating with {self.llm.provider_name}: {str(e)}"
        
        # 3. Encapsulate Result
        # We need a structured object to return
        return BlackwallResponse(
            content=response_text,
            metadata={"verified_claims": verified_claims}
        )



    def _get_system_prompt(self) -> str:
        base = "You are Dolores, an AI interface inside the BLACKWALL system."
        if self._current_mode == InterfaceMode.RIGOROUS:
            return base + " You must verify all claims mathematically."
        return base + " You are in exploratory mode."

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_mode(self) -> InterfaceMode:
        """Get the current interface mode."""
        return self._current_mode

    @property
    def auto_transition(self) -> bool:
        """Whether automatic mode transitions are enabled."""
        return self._auto_transition

    @auto_transition.setter
    def auto_transition(self, value: bool) -> None:
        """Enable or disable automatic mode transitions."""
        self._auto_transition = value

    @property
    def history(self) -> List[ModeTransition]:
        """Get the mode transition history."""
        return self._history.copy()

    # -------------------------------------------------------------------------
    # Mode Management
    # -------------------------------------------------------------------------

    def set_mode(
        self,
        mode: InterfaceMode,
        reason: TransitionReason = TransitionReason.USER_COMMAND,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the current interface mode.

        Args:
            mode: The new mode to set
            reason: Why the transition is happening
            context: Additional context about the transition
        """
        if mode == self._current_mode:
            logger.debug(f"Already in {mode.value} mode, no transition needed")
            return

        old_mode = self._current_mode
        self._current_mode = mode

        # Record transition
        transition = ModeTransition(
            from_mode=old_mode,
            to_mode=mode,
            reason=reason,
            context=context or {},
        )
        self._history.append(transition)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_mode, mode)
            except Exception as e:
                logger.error(f"Callback error during mode transition: {e}")

        logger.info(f"Mode transition: {old_mode.value} -> {mode.value} ({reason.value})")

    def suggest_mode(self, request_type: RequestType) -> InterfaceMode:
        """
        Suggest an appropriate mode for a request type.

        Args:
            request_type: The type of request being made

        Returns:
            Suggested InterfaceMode
        """
        return request_type.suggested_mode

    def maybe_transition(
        self,
        request_type: RequestType,
        force: bool = False,
    ) -> Tuple[bool, Optional[InterfaceMode]]:
        """
        Maybe transition to a new mode based on request type.

        Only transitions if auto_transition is enabled or force is True.

        Args:
            request_type: The type of incoming request
            force: Force transition even if auto_transition is disabled

        Returns:
            Tuple of (did_transition, new_mode or None)
        """
        if not self._auto_transition and not force:
            return (False, None)

        suggested = self.suggest_mode(request_type)

        if suggested != self._current_mode:
            self.set_mode(
                suggested,
                reason=TransitionReason.AUTO_DETECTED,
                context={"request_type": request_type.value},
            )
            return (True, suggested)

        return (False, None)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def get_config(self) -> ModeConfig:
        """
        Get the configuration for the current mode.

        Returns:
            ModeConfig for the current mode
        """
        if self._current_mode in self._custom_configs:
            return self._custom_configs[self._current_mode]
        return MODE_CONFIGS[self._current_mode]

    def set_custom_config(self, mode: InterfaceMode, config: ModeConfig) -> None:
        """
        Set a custom configuration for a mode.

        Args:
            mode: The mode to configure
            config: The custom configuration
        """
        self._custom_configs[mode] = config

    def reset_config(self, mode: Optional[InterfaceMode] = None) -> None:
        """
        Reset configuration to defaults.

        Args:
            mode: Specific mode to reset, or None to reset all
        """
        if mode is not None:
            self._custom_configs.pop(mode, None)
        else:
            self._custom_configs.clear()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_request(self, request_type: str) -> bool:
        """
        Validate if a request type is allowed in the current mode.

        Args:
            request_type: String name of the request type

        Returns:
            True if the request is allowed
        """
        config = self.get_config()

        # In RIGOROUS mode, only verification and calculation allowed
        if self._current_mode == InterfaceMode.RIGOROUS:
            allowed = {"verification", "calculation", "definition", "command"}
            return request_type.lower() in allowed

        # In SYSTEM mode, only commands allowed
        if self._current_mode == InterfaceMode.SYSTEM:
            return request_type.lower() == "command"

        # EXPLORATORY and HYBRID allow everything
        return True

    def validate_response(
        self,
        verified: bool,
        confidence: float,
        domain: str = "general",
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if a response meets the current mode's requirements.

        Args:
            verified: Whether the claim was verified
            confidence: Confidence score (0.0 to 1.0)
            domain: Verification domain

        Returns:
            Tuple of (is_valid, warning_message or None)
        """
        config = self.get_config()
        warning = None

        # Check verification requirement
        if config.reject_unverified and not verified:
            return (False, f"Response rejected: Unverified claim in {self._current_mode.value} mode")

        # Check confidence threshold
        if confidence < config.min_confidence:
            if config.reject_unverified:
                return (False, f"Response rejected: Confidence {confidence:.2f} below threshold {config.min_confidence}")
            elif config.warn_low_confidence:
                warning = f"Low confidence ({confidence:.2f}) in {self._current_mode.value} mode"

        return (True, warning)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_transition(
        self,
        callback: Callable[[InterfaceMode, InterfaceMode], None],
    ) -> None:
        """
        Register a callback for mode transitions.

        Args:
            callback: Function called with (old_mode, new_mode)
        """
        self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[InterfaceMode, InterfaceMode], None],
    ) -> bool:
        """
        Remove a transition callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was found and removed
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # -------------------------------------------------------------------------
    # Status and Serialization
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the controller.

        Returns:
            Dictionary with status information
        """
        config = self.get_config()
        return {
            "current_mode": self._current_mode.value,
            "mode_description": self._current_mode.description,
            "auto_transition": self._auto_transition,
            "config": config.to_dict(),
            "transition_count": len(self._history) - 1,  # Exclude initial
            "last_transition": self._history[-1].to_dict() if self._history else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the controller state.

        Returns:
            Dictionary representation of the controller
        """
        return {
            "current_mode": self._current_mode.value,
            "auto_transition": self._auto_transition,
            "history": [t.to_dict() for t in self._history],
            "custom_configs": {
                mode.value: config.to_dict()
                for mode, config in self._custom_configs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackwallController":
        """
        Deserialize a controller from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New BlackwallController instance
        """
        controller = cls(
            default_mode=InterfaceMode(data["current_mode"]),
            auto_transition=data.get("auto_transition", True),
        )

        # Restore custom configs
        for mode_str, config_data in data.get("custom_configs", {}).items():
            mode = InterfaceMode(mode_str)
            config = ModeConfig(mode=mode, **{
                k: v for k, v in config_data.items() if k != "mode"
            })
            controller._custom_configs[mode] = config

        return controller

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def enter_rigorous_mode(self) -> None:
        """Enter rigorous (proof-required) mode."""
        self.set_mode(InterfaceMode.RIGOROUS)

    def enter_exploratory_mode(self) -> None:
        """Enter exploratory (creative) mode."""
        self.set_mode(InterfaceMode.EXPLORATORY)

    def enter_hybrid_mode(self) -> None:
        """Enter hybrid (balanced) mode."""
        self.set_mode(InterfaceMode.HYBRID)

    def reset(self, default_mode: InterfaceMode = InterfaceMode.HYBRID) -> None:
        """
        Reset the controller to initial state.

        Args:
            default_mode: Mode to reset to
        """
        self._current_mode = default_mode
        self._history.clear()
        self._custom_configs.clear()
        self._history.append(ModeTransition(
            from_mode=default_mode,
            to_mode=default_mode,
            reason=TransitionReason.SESSION_START,
        ))
        logger.info(f"BLACKWALL reset to {default_mode.value} mode")


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_controller: Optional[BlackwallController] = None


def get_controller() -> BlackwallController:
    """Get or create the default BLACKWALL controller."""
    global _default_controller
    if _default_controller is None:
        _default_controller = BlackwallController()
    return _default_controller


def set_mode(mode: InterfaceMode) -> None:
    """Set the mode on the default controller."""
    get_controller().set_mode(mode)


def get_mode() -> InterfaceMode:
    """Get the current mode from the default controller."""
    return get_controller().current_mode


def get_config() -> ModeConfig:
    """Get the current config from the default controller."""
    return get_controller().get_config()
