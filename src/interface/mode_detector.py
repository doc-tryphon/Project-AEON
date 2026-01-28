"""
Mode Detector - Analyze user input to suggest appropriate interface mode.

This module provides pattern matching and heuristics to classify user
queries and suggest whether AEON should operate in RIGOROUS, EXPLORATORY,
or HYBRID mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

from .enums import InterfaceMode, RequestType


# =============================================================================
# Detection Patterns
# =============================================================================

# Patterns that suggest RIGOROUS mode (verification needed)
RIGOROUS_PATTERNS: List[Tuple[str, float]] = [
    # Verification requests
    (r"\bprove\s+that\b", 1.0),
    (r"\bverify\s+(?:that|if)?\b", 1.0),
    (r"\bis\s+(?:it\s+)?(?:true|correct)\s+that\b", 0.9),
    (r"\bshow\s+that\b", 0.9),
    (r"\bdemonstrate\b", 0.8),

    # Mathematical assertions
    (r"\bis\s+(?:a\s+)?unitary\b", 1.0),
    (r"\bis\s+(?:a\s+)?hermitian\b", 1.0),
    (r"\bis\s+normalized\b", 1.0),
    (r"\bis\s+(?:maximally\s+)?entangled\b", 0.9),
    (r"\bsatisfies?\b", 0.8),

    # Formal language
    (r"\biff\b", 0.9),
    (r"\bif\s+and\s+only\s+if\b", 0.9),
    (r"\btherefore\b", 0.7),
    (r"\bthus\b", 0.6),
    (r"\bhence\b", 0.6),
    (r"\bQ\.?E\.?D\.?\b", 1.0),

    # Calculation requests
    (r"\bcalculate\b", 0.8),
    (r"\bcompute\b", 0.8),
    (r"\bevaluate\b", 0.7),
    (r"\bfind\s+the\s+(?:value|result)\b", 0.7),

    # Commands
    (r"^/verify\b", 1.0),
    (r"^/prove\b", 1.0),
    (r"^/check\b", 0.9),
]

# Patterns that suggest EXPLORATORY mode (creative exploration)
EXPLORATORY_PATTERNS: List[Tuple[str, float]] = [
    # Hypotheticals
    (r"\bwhat\s+if\b", 1.0),
    (r"\bwhat\s+would\s+happen\b", 0.9),
    (r"\bimagine\b", 0.9),
    (r"\bsuppose\b", 0.8),
    (r"\bhypothetically\b", 1.0),
    (r"\blet's\s+say\b", 0.8),
    (r"\bassume\s+that\b", 0.7),

    # Creative requests
    (r"\bcould\s+(?:we|you)\b", 0.7),
    (r"\bwhat\s+about\b", 0.6),
    (r"\bhow\s+might\b", 0.8),
    (r"\bexplore\b", 0.9),
    (r"\bbrainstorm\b", 1.0),
    (r"\bspeculate\b", 1.0),

    # Intuition building
    (r"\bintuitively\b", 0.8),
    (r"\bget\s+(?:a\s+)?sense\b", 0.7),
    (r"\bunderstand\s+(?:the\s+)?(?:big\s+)?picture\b", 0.7),
    (r"\bin\s+simple\s+terms\b", 0.6),
    (r"\beli5\b", 0.8),  # Explain like I'm 5

    # Analogies
    (r"\blike\s+(?:a|an)\b", 0.5),
    (r"\banalog(?:y|ous)\b", 0.7),
    (r"\bsimilar\s+to\b", 0.5),

    # Commands
    (r"^/explore\b", 1.0),
    (r"^/imagine\b", 1.0),
    (r"^/brainstorm\b", 1.0),
]

# Patterns for detecting request types
REQUEST_TYPE_PATTERNS: Dict[RequestType, List[Tuple[str, float]]] = {
    RequestType.VERIFICATION: [
        (r"\bverify\b", 1.0),
        (r"\bprove\b", 1.0),
        (r"\bcheck\s+(?:if|that|whether)\b", 0.9),
        (r"\bis\s+(?:it\s+)?true\b", 0.8),
        (r"\bconfirm\b", 0.7),
    ],
    RequestType.EXPLANATION: [
        (r"\bwhy\s+(?:is|does|do|are|did|can|would)\b", 1.0),
        (r"\bhow\s+(?:does|do|is|are|can|would)\b", 0.9),
        (r"\bexplain\b", 1.0),
        (r"\bdescribe\b", 0.7),
        (r"\bwalk\s+(?:me\s+)?through\b", 0.8),
        (r"\bhelp\s+(?:me\s+)?understand\b", 0.9),
    ],
    RequestType.EXPLORATION: [
        (r"\bwhat\s+if\b", 1.0),
        (r"\bwhat\s+would\b", 0.9),
        (r"\bcould\s+(?:we|you|it)\b", 0.7),
        (r"\bimagine\b", 0.9),
        (r"\bexplore\b", 0.8),
        (r"\bwhat\s+about\b", 0.6),
    ],
    RequestType.CALCULATION: [
        (r"\bcalculate\b", 1.0),
        (r"\bcompute\b", 1.0),
        (r"\bevaluate\b", 0.8),
        (r"\bfind\s+(?:the\s+)?(?:value|result|answer)\b", 0.9),
        (r"\bsolve\b", 0.8),
        (r"\bwork\s+out\b", 0.7),
    ],
    RequestType.DEFINITION: [
        (r"\bwhat\s+is\s+(?:a|an|the)?\b", 0.9),
        (r"\bdefine\b", 1.0),
        (r"\bdefinition\s+of\b", 1.0),
        (r"\btell\s+me\s+about\b", 0.6),
        (r"\bwhat\s+(?:does|do)\s+.+\s+mean\b", 0.8),
    ],
    RequestType.COMMAND: [
        (r"^/\w+", 1.0),  # Any slash command
    ],
}


# =============================================================================
# Detection Result
# =============================================================================

@dataclass
class DetectionResult:
    """
    Result of analyzing user input for mode detection.

    Attributes:
        suggested_mode: The recommended interface mode
        request_type: The detected type of request
        confidence: Confidence in the detection (0.0 to 1.0)
        matched_patterns: Patterns that matched
        reasoning: Human-readable explanation
    """
    suggested_mode: InterfaceMode
    request_type: RequestType
    confidence: float
    matched_patterns: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suggested_mode": self.suggested_mode.value,
            "request_type": self.request_type.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "reasoning": self.reasoning,
        }


# =============================================================================
# Main Detector Class
# =============================================================================

class ModeDetector:
    """
    Detector for analyzing user input and suggesting interface modes.

    Uses pattern matching and heuristics to classify queries and
    recommend the appropriate BLACKWALL mode.

    Example:
        >>> detector = ModeDetector()
        >>> result = detector.analyze("Prove that the Hadamard gate is unitary")
        >>> result.suggested_mode
        <InterfaceMode.RIGOROUS: 'rigorous'>

        >>> result = detector.analyze("What if we used a different basis?")
        >>> result.suggested_mode
        <InterfaceMode.EXPLORATORY: 'exploratory'>
    """

    def __init__(self):
        """Initialize the detector with compiled patterns."""
        self._rigorous_patterns: List[Tuple[Pattern, float]] = [
            (re.compile(p, re.IGNORECASE), w) for p, w in RIGOROUS_PATTERNS
        ]
        self._exploratory_patterns: List[Tuple[Pattern, float]] = [
            (re.compile(p, re.IGNORECASE), w) for p, w in EXPLORATORY_PATTERNS
        ]
        self._request_patterns: Dict[RequestType, List[Tuple[Pattern, float]]] = {
            rt: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
            for rt, patterns in REQUEST_TYPE_PATTERNS.items()
        }

    def analyze(self, text: str) -> DetectionResult:
        """
        Analyze user input and suggest an interface mode.

        Args:
            text: The user's input text

        Returns:
            DetectionResult with mode suggestion and confidence
        """
        text = text.strip()
        if not text:
            return DetectionResult(
                suggested_mode=InterfaceMode.HYBRID,
                request_type=RequestType.UNKNOWN,
                confidence=0.0,
                reasoning="Empty input",
            )

        # Detect request type first
        request_type = self._detect_request_type(text)

        # Calculate mode scores
        rigorous_score, rigorous_matches = self._score_patterns(
            text, self._rigorous_patterns
        )
        exploratory_score, exploratory_matches = self._score_patterns(
            text, self._exploratory_patterns
        )

        # Determine mode based on scores
        matched_patterns = []
        if rigorous_score > exploratory_score + 0.2:  # Strong rigorous signal
            mode = InterfaceMode.RIGOROUS
            confidence = min(rigorous_score, 1.0)
            matched_patterns = rigorous_matches
            reasoning = f"Detected verification/proof language (score: {rigorous_score:.2f})"
        elif exploratory_score > rigorous_score + 0.2:  # Strong exploratory signal
            mode = InterfaceMode.EXPLORATORY
            confidence = min(exploratory_score, 1.0)
            matched_patterns = exploratory_matches
            reasoning = f"Detected hypothetical/creative language (score: {exploratory_score:.2f})"
        else:  # Mixed or unclear signals
            mode = InterfaceMode.HYBRID
            confidence = 1.0 - abs(rigorous_score - exploratory_score)
            matched_patterns = rigorous_matches + exploratory_matches
            reasoning = f"Mixed signals (rigorous: {rigorous_score:.2f}, exploratory: {exploratory_score:.2f})"

        # Override with request type suggestion if confident
        if request_type != RequestType.UNKNOWN:
            type_suggested = request_type.suggested_mode
            if type_suggested == InterfaceMode.RIGOROUS and rigorous_score > 0.3:
                mode = InterfaceMode.RIGOROUS
            elif type_suggested == InterfaceMode.EXPLORATORY and exploratory_score > 0.3:
                mode = InterfaceMode.EXPLORATORY

        return DetectionResult(
            suggested_mode=mode,
            request_type=request_type,
            confidence=confidence,
            matched_patterns=matched_patterns,
            reasoning=reasoning,
        )

    def _detect_request_type(self, text: str) -> RequestType:
        """Detect the type of request from the input text."""
        best_type = RequestType.UNKNOWN
        best_score = 0.0

        for req_type, patterns in self._request_patterns.items():
            score, _ = self._score_patterns(text, patterns)
            if score > best_score:
                best_score = score
                best_type = req_type

        return best_type if best_score > 0.3 else RequestType.UNKNOWN

    def _score_patterns(
        self,
        text: str,
        patterns: List[Tuple[Pattern, float]],
    ) -> Tuple[float, List[str]]:
        """
        Score text against a list of patterns.

        Returns:
            Tuple of (total_score, list of matched pattern strings)
        """
        total_score = 0.0
        matched = []

        for pattern, weight in patterns:
            if pattern.search(text):
                total_score += weight
                matched.append(pattern.pattern)

        return (total_score, matched)

    def is_command(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the input is a slash command.

        Args:
            text: The user's input

        Returns:
            Tuple of (is_command, command_name or None)
        """
        text = text.strip()
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command = parts[0][1:]  # Remove the slash
            return (True, command)
        return (False, None)

    def extract_command_args(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract command and arguments from a slash command.

        Args:
            text: The command text (e.g., "/mode rigorous")

        Returns:
            Tuple of (command_name, [args])
        """
        text = text.strip()
        if not text.startswith("/"):
            return ("", [])

        parts = text.split()
        command = parts[0][1:]  # Remove the slash
        args = parts[1:] if len(parts) > 1 else []
        return (command, args)

    def suggest_mode_from_context(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None,
    ) -> InterfaceMode:
        """
        Suggest a mode based on multiple pieces of context.

        Useful for analyzing conversation history to determine
        the overall mode tendency.

        Args:
            texts: List of text snippets (e.g., recent messages)
            weights: Optional weights for each text (default: newer = higher)

        Returns:
            Suggested InterfaceMode
        """
        if not texts:
            return InterfaceMode.HYBRID

        if weights is None:
            # Weight more recent texts higher
            n = len(texts)
            weights = [(i + 1) / n for i in range(n)]

        rigorous_total = 0.0
        exploratory_total = 0.0
        total_weight = sum(weights)

        for text, weight in zip(texts, weights):
            result = self.analyze(text)
            if result.suggested_mode == InterfaceMode.RIGOROUS:
                rigorous_total += weight * result.confidence
            elif result.suggested_mode == InterfaceMode.EXPLORATORY:
                exploratory_total += weight * result.confidence

        rigorous_avg = rigorous_total / total_weight
        exploratory_avg = exploratory_total / total_weight

        if rigorous_avg > exploratory_avg + 0.15:
            return InterfaceMode.RIGOROUS
        elif exploratory_avg > rigorous_avg + 0.15:
            return InterfaceMode.EXPLORATORY
        else:
            return InterfaceMode.HYBRID


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_detector: Optional[ModeDetector] = None


def get_detector() -> ModeDetector:
    """Get or create the default mode detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ModeDetector()
    return _default_detector


def analyze(text: str) -> DetectionResult:
    """Analyze text using the default detector."""
    return get_detector().analyze(text)


def suggest_mode(text: str) -> InterfaceMode:
    """Suggest a mode for the given text."""
    return get_detector().analyze(text).suggested_mode


def is_command(text: str) -> Tuple[bool, Optional[str]]:
    """Check if text is a slash command."""
    return get_detector().is_command(text)
