"""
Tutor module - Verification API and explanation generation.

Sprint 1 components:
- verification_api.py: Wrapper for QuantumVerifier
- claim_parser.py: Natural language to SymPy
- explanation_gen.py: SymPy results to human-readable

Sprint 4 components:
- session.py: Session management with LLM integration
- verification_loop.py: LLM generation with verification cycle
"""

from .verification_api import (
    TutorVerificationAPI,
    VerificationResult,
    VerificationDomain,
    VerificationError,
    ParseError,
    UnsupportedClaimError,
    InvalidInputError,
    create_api,
    verify_unitary,
    verify_normalized,
    verify_hermitian,
)

from .claim_parser import (
    ClaimParser,
    ParsedClaim,
    ClaimType,
    VerificationMethod,
    parse_claim,
    parse_claims,
)

from .explanation_gen import (
    ExplanationGenerator,
    OutputFormat,
    explain,
    explain_markdown,
    explain_latex,
)

from .session import (
    TutorSession,
    SessionConfig,
    SessionStatus,
    Turn,
    TurnMetadata,
    SessionStats,
    create_session,
    get_session,
    process,
    verify,
    get_status,
)

from .verification_loop import (
    VerificationLoop,
    LoopConfig,
    LoopResult,
    LoopStatus,
    RetryStrategy,
    VerificationAttempt,
    create_loop,
    run_verified,
    verify_response,
)

__all__ = [
    # verification_api
    "TutorVerificationAPI",
    "VerificationResult",
    "VerificationDomain",
    "VerificationError",
    "ParseError",
    "UnsupportedClaimError",
    "InvalidInputError",
    "create_api",
    "verify_unitary",
    "verify_normalized",
    "verify_hermitian",
    # claim_parser
    "ClaimParser",
    "ParsedClaim",
    "ClaimType",
    "VerificationMethod",
    "parse_claim",
    "parse_claims",
    # explanation_gen
    "ExplanationGenerator",
    "OutputFormat",
    "explain",
    "explain_markdown",
    "explain_latex",
    # session (Sprint 4)
    "TutorSession",
    "SessionConfig",
    "SessionStatus",
    "Turn",
    "TurnMetadata",
    "SessionStats",
    "create_session",
    "get_session",
    "process",
    "verify",
    "get_status",
    # verification_loop (Sprint 4)
    "VerificationLoop",
    "LoopConfig",
    "LoopResult",
    "LoopStatus",
    "RetryStrategy",
    "VerificationAttempt",
    "create_loop",
    "run_verified",
    "verify_response",
]
