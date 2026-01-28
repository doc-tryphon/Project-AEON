"""
Claim Parser - Parse natural language physics claims to structured verification queries.

This module provides parsing capabilities to convert natural language claims
about quantum physics into structured queries that can be processed by the
TutorVerificationAPI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from sympy import Matrix, sqrt, I
from sympy.parsing.sympy_parser import parse_expr


# =============================================================================
# Enums and Types
# =============================================================================

class ClaimType(str, Enum):
    """Types of physics claims that can be parsed and verified."""
    UNITARITY = "unitarity"
    HERMITICITY = "hermiticity"
    NORMALIZATION = "normalization"
    ENTANGLEMENT = "entanglement"
    BELL_STATE = "bell_state"
    CHSH = "chsh"
    EQUATION = "equation"
    UNKNOWN = "unknown"


class VerificationMethod(str, Enum):
    """Verification methods available in TutorVerificationAPI."""
    VERIFY_GATE = "verify_gate"
    VERIFY_STATE = "verify_state"
    VERIFY_OPERATOR = "verify_operator"
    VERIFY_BELL_STATE = "verify_bell_state"
    VERIFY_CHSH = "verify_chsh"
    VERIFY_CLAIM = "verify_claim"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParsedClaim:
    """
    Result of parsing a natural language claim.

    Attributes:
        claim_type: The type of claim (unitarity, hermiticity, etc.)
        method: The verification method to call
        subject: The subject of the claim (gate name, state name, etc.)
        parameters: Additional parameters for the verification
        original_text: The original claim text
        confidence: Parsing confidence (1.0 = exact match, <1.0 = fuzzy)
    """
    claim_type: ClaimType
    method: VerificationMethod
    subject: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_type": self.claim_type.value,
            "method": self.method.value,
            "subject": self.subject,
            "parameters": self.parameters,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

# Named quantum objects for recognition
KNOWN_GATES = {
    "hadamard", "h", "pauli-x", "pauli-y", "pauli-z", "x", "y", "z",
    "cnot", "cx", "cz", "swap", "toffoli", "ccx", "s", "t",
    "identity", "i", "rx", "ry", "rz", "phase"
}

KNOWN_STATES = {
    "|0>", "|1>", "|+>", "|->", "|0", "|1", "|+", "|-",
    "zero", "one", "plus", "minus",
    "bell", "phi+", "phi-", "psi+", "psi-",
    "bell_phi_plus", "bell_phi_minus", "bell_psi_plus", "bell_psi_minus",
    "ghz", "w state", "cat state"
}

KNOWN_OPERATORS = {
    "pauli-x", "pauli-y", "pauli-z", "x", "y", "z",
    "sigma_x", "sigma_y", "sigma_z",
    "hamiltonian", "observable"
}

# Claim patterns: regex -> (ClaimType, VerificationMethod, subject_group)
CLAIM_PATTERNS: List[Tuple[str, ClaimType, VerificationMethod, int]] = [
    # Unitarity patterns
    (r"(.+?)\s+is\s+unitary", ClaimType.UNITARITY, VerificationMethod.VERIFY_GATE, 1),
    (r"(.+?)\s+gate\s+is\s+unitary", ClaimType.UNITARITY, VerificationMethod.VERIFY_GATE, 1),
    (r"the\s+(.+?)\s+gate\s+is\s+unitary", ClaimType.UNITARITY, VerificationMethod.VERIFY_GATE, 1),
    (r"verify\s+(.+?)\s+is\s+unitary", ClaimType.UNITARITY, VerificationMethod.VERIFY_GATE, 1),
    (r"check\s+if\s+(.+?)\s+is\s+unitary", ClaimType.UNITARITY, VerificationMethod.VERIFY_GATE, 1),

    # Hermiticity patterns
    (r"(.+?)\s+is\s+hermitian", ClaimType.HERMITICITY, VerificationMethod.VERIFY_OPERATOR, 1),
    (r"(.+?)\s+is\s+self-adjoint", ClaimType.HERMITICITY, VerificationMethod.VERIFY_OPERATOR, 1),
    (r"(.+?)\s+is\s+an?\s+observable", ClaimType.HERMITICITY, VerificationMethod.VERIFY_OPERATOR, 1),
    (r"verify\s+(.+?)\s+is\s+hermitian", ClaimType.HERMITICITY, VerificationMethod.VERIFY_OPERATOR, 1),

    # Normalization patterns
    (r"(.+?)\s+is\s+normalized", ClaimType.NORMALIZATION, VerificationMethod.VERIFY_STATE, 1),
    (r"state\s+(.+?)\s+is\s+normalized", ClaimType.NORMALIZATION, VerificationMethod.VERIFY_STATE, 1),
    (r"the\s+state\s+(.+?)\s+is\s+normalized", ClaimType.NORMALIZATION, VerificationMethod.VERIFY_STATE, 1),
    (r"(.+?)\s+has\s+unit\s+norm", ClaimType.NORMALIZATION, VerificationMethod.VERIFY_STATE, 1),
    (r"verify\s+(.+?)\s+is\s+normalized", ClaimType.NORMALIZATION, VerificationMethod.VERIFY_STATE, 1),

    # Entanglement patterns
    (r"(.+?)\s+is\s+(?:maximally\s+)?entangled", ClaimType.ENTANGLEMENT, VerificationMethod.VERIFY_STATE, 1),
    (r"(.+?)\s+is\s+a\s+bell\s+state", ClaimType.BELL_STATE, VerificationMethod.VERIFY_BELL_STATE, 1),
    (r"bell\s+state\s+(.+?)\s+is\s+(?:maximally\s+)?entangled", ClaimType.BELL_STATE, VerificationMethod.VERIFY_BELL_STATE, 1),
    (r"verify\s+(.+?)\s+is\s+entangled", ClaimType.ENTANGLEMENT, VerificationMethod.VERIFY_STATE, 1),

    # CHSH patterns
    (r"chsh\s+(?:inequality\s+)?violation", ClaimType.CHSH, VerificationMethod.VERIFY_CHSH, 0),
    (r"verify\s+chsh", ClaimType.CHSH, VerificationMethod.VERIFY_CHSH, 0),
    (r"bell\s+(?:inequality\s+)?violation", ClaimType.CHSH, VerificationMethod.VERIFY_CHSH, 0),
]


# =============================================================================
# Main Parser Class
# =============================================================================

class ClaimParser:
    """
    Parser for natural language physics claims.

    Converts natural language claims into structured ParsedClaim objects
    that can be dispatched to the appropriate verification methods.

    Example:
        >>> parser = ClaimParser()
        >>> result = parser.parse("Hadamard is unitary")
        >>> print(result.claim_type)  # ClaimType.UNITARITY
        >>> print(result.subject)  # "Hadamard"
    """

    def __init__(self):
        """Initialize the parser with default patterns."""
        self._patterns = CLAIM_PATTERNS
        self._compiled_patterns: List[Tuple[re.Pattern, ClaimType, VerificationMethod, int]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), claim_type, method, group)
            for pattern, claim_type, method, group in self._patterns
        ]

    def parse(self, claim: str) -> ParsedClaim:
        """
        Parse a natural language claim into a structured query.

        Args:
            claim: Natural language claim string

        Returns:
            ParsedClaim object with parsed information

        Example:
            >>> parser.parse("The Hadamard gate is unitary")
            ParsedClaim(claim_type=UNITARITY, method=VERIFY_GATE, subject="Hadamard")
        """
        claim = claim.strip()

        # Try each pattern
        for pattern, claim_type, method, subject_group in self._compiled_patterns:
            match = pattern.search(claim)
            if match:
                subject = match.group(subject_group).strip() if subject_group > 0 else ""
                subject = self._normalize_subject(subject, claim_type)

                return ParsedClaim(
                    claim_type=claim_type,
                    method=method,
                    subject=subject,
                    parameters=self._extract_parameters(claim, claim_type),
                    original_text=claim,
                    confidence=1.0
                )

        # Fallback: try to extract subject and guess type
        return self._fuzzy_parse(claim)

    def _normalize_subject(self, subject: str, claim_type: ClaimType) -> str:
        """
        Normalize the subject string for consistency.

        Handles various formats and aliases for quantum objects.
        """
        subject = subject.strip().lower()

        # Gate normalizations
        gate_aliases = {
            "hadamard": "H",
            "h": "H",
            "pauli-x": "X",
            "pauli x": "X",
            "x": "X",
            "pauli-y": "Y",
            "pauli y": "Y",
            "y": "Y",
            "pauli-z": "Z",
            "pauli z": "Z",
            "z": "Z",
            "identity": "I",
            "i": "I",
            "cnot": "CNOT",
            "cx": "CNOT",
            "s gate": "S",
            "t gate": "T",
        }

        # State normalizations
        state_aliases = {
            "zero": "|0>",
            "one": "|1>",
            "plus": "|+>",
            "minus": "|->",
            "phi+": "bell_phi_plus",
            "phi-": "bell_phi_minus",
            "psi+": "bell_psi_plus",
            "psi-": "bell_psi_minus",
            "bell phi+": "bell_phi_plus",
            "bell phi-": "bell_phi_minus",
            "bell psi+": "bell_psi_plus",
            "bell psi-": "bell_psi_minus",
        }

        if claim_type in (ClaimType.UNITARITY,):
            return gate_aliases.get(subject, subject.upper() if len(subject) <= 4 else subject)
        elif claim_type in (ClaimType.NORMALIZATION, ClaimType.ENTANGLEMENT, ClaimType.BELL_STATE):
            return state_aliases.get(subject, subject)
        elif claim_type == ClaimType.HERMITICITY:
            return gate_aliases.get(subject, subject.upper() if len(subject) <= 4 else subject)

        return subject

    def _extract_parameters(self, claim: str, claim_type: ClaimType) -> Dict[str, Any]:
        """Extract additional parameters from the claim."""
        params = {}

        if claim_type == ClaimType.ENTANGLEMENT:
            # Check if entanglement check is requested
            params["check_entanglement"] = True

        return params

    def _fuzzy_parse(self, claim: str) -> ParsedClaim:
        """
        Attempt fuzzy parsing when exact patterns don't match.

        Uses keyword detection and heuristics to guess the claim type.
        """
        claim_lower = claim.lower()

        # Detect claim type by keywords
        if any(word in claim_lower for word in ["unitary", "u†u", "u*u"]):
            # Try to extract a subject
            subject = self._extract_subject(claim)
            return ParsedClaim(
                claim_type=ClaimType.UNITARITY,
                method=VerificationMethod.VERIFY_GATE,
                subject=subject or "I",
                original_text=claim,
                confidence=0.7
            )

        elif any(word in claim_lower for word in ["hermitian", "self-adjoint", "observable"]):
            subject = self._extract_subject(claim)
            return ParsedClaim(
                claim_type=ClaimType.HERMITICITY,
                method=VerificationMethod.VERIFY_OPERATOR,
                subject=subject or "I",
                original_text=claim,
                confidence=0.7
            )

        elif any(word in claim_lower for word in ["normalized", "norm", "unit norm"]):
            subject = self._extract_subject(claim)
            return ParsedClaim(
                claim_type=ClaimType.NORMALIZATION,
                method=VerificationMethod.VERIFY_STATE,
                subject=subject or "|0>",
                original_text=claim,
                confidence=0.7
            )

        elif any(word in claim_lower for word in ["entangled", "entanglement", "bell"]):
            subject = self._extract_subject(claim)
            return ParsedClaim(
                claim_type=ClaimType.ENTANGLEMENT,
                method=VerificationMethod.VERIFY_STATE,
                subject=subject or "bell_phi_plus",
                parameters={"check_entanglement": True},
                original_text=claim,
                confidence=0.7
            )

        elif any(word in claim_lower for word in ["chsh", "violation"]):
            return ParsedClaim(
                claim_type=ClaimType.CHSH,
                method=VerificationMethod.VERIFY_CHSH,
                subject="",
                original_text=claim,
                confidence=0.7
            )

        # Unknown claim type
        return ParsedClaim(
            claim_type=ClaimType.UNKNOWN,
            method=VerificationMethod.VERIFY_CLAIM,
            subject=claim,
            original_text=claim,
            confidence=0.3
        )

    def _extract_subject(self, claim: str) -> str:
        """
        Extract the subject (gate/state/operator name) from a claim.

        Uses heuristics to identify quantum object names.
        """
        claim_lower = claim.lower()

        # Check for known gates
        for gate in KNOWN_GATES:
            if gate in claim_lower:
                return gate

        # Check for known states
        for state in KNOWN_STATES:
            if state in claim_lower:
                return state

        # Check for ket notation
        ket_match = re.search(r'\|[0-9+\-a-zA-Z]+\>?', claim)
        if ket_match:
            return ket_match.group()

        # Try to find a capitalized word (likely a name)
        cap_match = re.search(r'\b([A-Z][a-zA-Z]*)\b', claim)
        if cap_match:
            return cap_match.group(1)

        return ""

    def add_pattern(
        self,
        pattern: str,
        claim_type: ClaimType,
        method: VerificationMethod,
        subject_group: int = 1
    ) -> None:
        """
        Add a custom pattern to the parser.

        Args:
            pattern: Regex pattern string
            claim_type: The claim type for matches
            method: The verification method to use
            subject_group: The regex group containing the subject
        """
        self._patterns.append((pattern, claim_type, method, subject_group))
        self._compile_patterns()

    def parse_batch(self, claims: List[str]) -> List[ParsedClaim]:
        """
        Parse multiple claims at once.

        Args:
            claims: List of claim strings

        Returns:
            List of ParsedClaim objects
        """
        return [self.parse(claim) for claim in claims]

    def parse_multiple(self, text: str) -> List[ParsedClaim]:
        """
        Extract and parse multiple claims from a text.

        Searches for patterns in the text that match known claim types.

        Args:
            text: Text potentially containing multiple claims

        Returns:
            List of ParsedClaim objects found in the text
        """
        claims = []
        text_lower = text.lower()

        # Search for each pattern in the text
        for regex, claim_type, method, subject_group in self._compiled_patterns:
            for match in regex.finditer(text_lower):
                try:
                    subject = match.group(subject_group).strip()
                    # Try to find the original case from the text
                    start, end = match.span(subject_group)
                    original_subject = text[start:end].strip()

                    claim = ParsedClaim(
                        original_text=match.group(0),
                        claim_type=claim_type,
                        subject=original_subject if original_subject else subject,
                        method=method,
                        confidence=0.9,
                        parameters={"match_type": "pattern"},
                    )
                    claims.append(claim)
                except (IndexError, AttributeError):
                    continue

        return claims


# =============================================================================
# Module-level convenience functions
# =============================================================================

def parse_claim(claim: str) -> ParsedClaim:
    """Parse a single claim using the default parser."""
    return ClaimParser().parse(claim)


def parse_claims(claims: List[str]) -> List[ParsedClaim]:
    """Parse multiple claims using the default parser."""
    return ClaimParser().parse_batch(claims)
