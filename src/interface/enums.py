"""
BLACKWALL Interface Enums - Mode definitions for the adaptive cognitive interface.

Defines the operational modes that control how AEON responds to queries:
- RIGOROUS: Verified claims only, formal notation, proof-backed responses
- EXPLORATORY: Creative hypotheses, intuition building, speculative reasoning
- HYBRID: Balanced approach, adapts based on context
- SYSTEM: Internal mode for system operations and diagnostics
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any


class InterfaceMode(str, Enum):
    """
    Operational modes for the BLACKWALL adaptive interface.

    Each mode affects how the system processes and responds to queries,
    controlling the balance between verified/formal responses and
    creative/exploratory thinking.

    Attributes:
        RIGOROUS: Only verified claims, formal notation, requires symbolic proofs
        EXPLORATORY: Creative hypotheses allowed, intuition building mode
        HYBRID: Adaptive mode that balances rigor and exploration
        SYSTEM: Internal mode for diagnostics and system operations
    """
    RIGOROUS = "rigorous"
    EXPLORATORY = "exploratory"
    HYBRID = "hybrid"
    SYSTEM = "system"

    @property
    def requires_verification(self) -> bool:
        """Whether this mode requires all claims to be verified."""
        return self in (InterfaceMode.RIGOROUS,)

    @property
    def allows_speculation(self) -> bool:
        """Whether this mode allows speculative/creative responses."""
        return self in (InterfaceMode.EXPLORATORY, InterfaceMode.HYBRID)

    @property
    def min_confidence(self) -> float:
        """Minimum confidence threshold for claims in this mode."""
        thresholds = {
            InterfaceMode.RIGOROUS: 1.0,      # Only symbolic proofs
            InterfaceMode.EXPLORATORY: 0.0,   # Any confidence accepted
            InterfaceMode.HYBRID: 0.7,        # Moderate threshold
            InterfaceMode.SYSTEM: 0.0,        # No restriction
        }
        return thresholds[self]

    @property
    def description(self) -> str:
        """Human-readable description of this mode."""
        descriptions = {
            InterfaceMode.RIGOROUS: "Verified claims only. All responses backed by symbolic proofs.",
            InterfaceMode.EXPLORATORY: "Creative exploration mode. Hypotheses and intuition allowed.",
            InterfaceMode.HYBRID: "Balanced mode. Adapts rigor based on query context.",
            InterfaceMode.SYSTEM: "System diagnostics mode. Internal operations only.",
        }
        return descriptions[self]

    def to_dict(self) -> Dict[str, Any]:
        """Convert mode to dictionary for serialization."""
        return {
            "mode": self.value,
            "requires_verification": self.requires_verification,
            "allows_speculation": self.allows_speculation,
            "min_confidence": self.min_confidence,
            "description": self.description,
        }


class RequestType(str, Enum):
    """
    Types of requests that can be processed by BLACKWALL.

    Used by the mode detector to classify incoming queries and
    determine appropriate handling.
    """
    VERIFICATION = "verification"      # "Is X unitary?", "Prove that..."
    EXPLANATION = "explanation"        # "Why is...", "How does..."
    EXPLORATION = "exploration"        # "What if...", "Could we..."
    CALCULATION = "calculation"        # "Calculate...", "Compute..."
    DEFINITION = "definition"          # "What is...", "Define..."
    COMMAND = "command"               # "/mode", "/verify", etc.
    UNKNOWN = "unknown"

    @property
    def suggested_mode(self) -> InterfaceMode:
        """Suggest an appropriate interface mode for this request type."""
        suggestions = {
            RequestType.VERIFICATION: InterfaceMode.RIGOROUS,
            RequestType.EXPLANATION: InterfaceMode.HYBRID,
            RequestType.EXPLORATION: InterfaceMode.EXPLORATORY,
            RequestType.CALCULATION: InterfaceMode.RIGOROUS,
            RequestType.DEFINITION: InterfaceMode.HYBRID,
            RequestType.COMMAND: InterfaceMode.SYSTEM,
            RequestType.UNKNOWN: InterfaceMode.HYBRID,
        }
        return suggestions[self]


class TransitionReason(str, Enum):
    """
    Reasons for mode transitions in BLACKWALL.

    Tracks why the interface mode changed, useful for logging
    and debugging persona behavior.
    """
    USER_COMMAND = "user_command"           # Explicit /mode command
    AUTO_DETECTED = "auto_detected"         # Mode detector suggestion
    VERIFICATION_REQUIRED = "verification_required"  # Query needs proof
    EXPLORATION_DETECTED = "exploration_detected"    # "What if" detected
    CONTEXT_SWITCH = "context_switch"       # Topic changed
    SESSION_START = "session_start"         # New session initialized
    FIDELITY_DRIFT = "fidelity_drift"       # Persona coherence issue
