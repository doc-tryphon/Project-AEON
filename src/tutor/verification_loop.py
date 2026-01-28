"""
Verification Loop - LLM generation with verification and retry logic.

Implements the core loop where:
1. LLM generates a response
2. Claims are extracted and verified
3. If verification fails, LLM retries with feedback
4. Final response is returned with verification status
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm.interface import (
    LLMProvider,
    LLMResponse,
    Message,
    LLMError,
)
from ..llm.prompts.physics_tutor import PhysicsTutorPrompts
from .verification_api import TutorVerificationAPI, VerificationResult
from .claim_parser import ClaimParser, ParsedClaim


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class LoopStatus(str, Enum):
    """Status of the verification loop."""
    SUCCESS = "success"                    # All claims verified
    PARTIAL = "partial"                    # Some claims verified
    FAILED = "failed"                      # Verification failed after retries
    NO_CLAIMS = "no_claims"                # No verifiable claims found
    ERROR = "error"                        # System error occurred


class RetryStrategy(str, Enum):
    """Strategy for handling verification failures."""
    NONE = "none"                          # Don't retry, return as-is
    REGENERATE = "regenerate"              # Regenerate entire response
    TARGETED = "targeted"                  # Ask LLM to fix specific claims
    FALLBACK = "fallback"                  # Use fallback response


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LoopConfig:
    """Configuration for the verification loop."""
    max_retries: int = 2
    retry_strategy: RetryStrategy = RetryStrategy.TARGETED
    extract_claims: bool = True
    require_all_verified: bool = False
    include_proof_in_response: bool = True
    confidence_threshold: float = 1.0


@dataclass
class VerificationAttempt:
    """Record of a single verification attempt."""
    attempt_number: int
    response_text: str
    claims: List[ParsedClaim]
    results: List[VerificationResult]
    all_verified: bool
    partial_verified: bool
    
    @property
    def verified_count(self) -> int:
        """Number of verified claims."""
        return sum(1 for r in self.results if r.verified)
    
    @property
    def total_claims(self) -> int:
        """Total number of claims."""
        return len(self.claims)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attempt_number": self.attempt_number,
            "response_text": self.response_text[:500],  # Truncate for logging
            "claim_count": len(self.claims),
            "verified_count": self.verified_count,
            "all_verified": self.all_verified,
            "partial_verified": self.partial_verified,
        }


@dataclass
class LoopResult:
    """Result of the verification loop."""
    status: LoopStatus
    final_response: str
    attempts: List[VerificationAttempt]
    verification_results: List[VerificationResult]
    
    @property
    def total_attempts(self) -> int:
        """Total number of attempts made."""
        return len(self.attempts)
    
    @property
    def all_verified(self) -> bool:
        """Check if all claims were verified."""
        return all(r.verified for r in self.verification_results)
    
    @property
    def any_verified(self) -> bool:
        """Check if any claims were verified."""
        return any(r.verified for r in self.verification_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "total_attempts": self.total_attempts,
            "verification_count": len(self.verification_results),
            "verified_count": sum(1 for r in self.verification_results if r.verified),
            "attempts": [a.to_dict() for a in self.attempts],
        }


# =============================================================================
# Main Verification Loop
# =============================================================================

class VerificationLoop:
    """
    Implements LLM generation with verification and retry logic.
    
    The loop:
    1. Generates a response using the LLM
    2. Extracts verifiable claims from the response
    3. Verifies each claim using TutorVerificationAPI
    4. If verification fails, retries with feedback
    5. Returns final response with verification status
    
    Example:
        >>> from src.llm.providers import create_provider
        >>> 
        >>> provider = create_provider("anthropic")
        >>> loop = VerificationLoop(provider)
        >>> 
        >>> result = loop.run(
        ...     "Prove that the Hadamard gate is unitary",
        ...     system="You are a physics tutor."
        ... )
        >>> 
        >>> print(result.status)  # LoopStatus.SUCCESS
        >>> print(result.final_response)
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[LoopConfig] = None,
    ):
        """
        Initialize the verification loop.
        
        Args:
            llm_provider: LLM provider for generation
            config: Loop configuration
        """
        self._provider = llm_provider
        self._config = config or LoopConfig()
        
        # Components
        self._verification_api = TutorVerificationAPI()
        self._claim_parser = ClaimParser()
        self._prompts = PhysicsTutorPrompts()
        
        # Callbacks
        self._on_attempt: List[Callable[[VerificationAttempt], None]] = []
        self._on_verification: List[Callable[[VerificationResult], None]] = []
        
        logger.debug("VerificationLoop initialized")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def config(self) -> LoopConfig:
        """Get loop configuration."""
        return self._config
    
    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    
    def run(
        self,
        user_input: str,
        system: Optional[str] = None,
        context: Optional[List[Message]] = None,
    ) -> LoopResult:
        """
        Run the verification loop.
        
        Args:
            user_input: User's query
            system: System prompt
            context: Previous conversation context
            
        Returns:
            LoopResult with final response and verification status
        """
        attempts: List[VerificationAttempt] = []
        context = context or []
        
        # Build initial messages
        messages = context + [Message.user(user_input)]
        
        # Initial generation
        try:
            response_text = self._generate(messages, system)
        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            return LoopResult(
                status=LoopStatus.ERROR,
                final_response=f"Error generating response: {str(e)}",
                attempts=[],
                verification_results=[],
            )
        
        # Extract and verify claims
        attempt = self._create_attempt(1, response_text)
        attempts.append(attempt)
        self._notify_attempt(attempt)
        
        # Check if no claims found
        if not attempt.claims:
            return LoopResult(
                status=LoopStatus.NO_CLAIMS,
                final_response=response_text,
                attempts=attempts,
                verification_results=[],
            )
        
        # Check if all verified on first try
        if attempt.all_verified:
            return LoopResult(
                status=LoopStatus.SUCCESS,
                final_response=self._enhance_response(response_text, attempt.results),
                attempts=attempts,
                verification_results=attempt.results,
            )
        
        # Retry loop
        if self._config.retry_strategy != RetryStrategy.NONE:
            for retry_num in range(self._config.max_retries):
                # Generate retry based on strategy
                retry_response = self._generate_retry(
                    user_input,
                    system,
                    context,
                    attempts[-1],
                )
                
                if retry_response is None:
                    break
                
                # Create new attempt
                attempt = self._create_attempt(len(attempts) + 1, retry_response)
                attempts.append(attempt)
                self._notify_attempt(attempt)
                
                # Check if successful
                if attempt.all_verified:
                    return LoopResult(
                        status=LoopStatus.SUCCESS,
                        final_response=self._enhance_response(
                            retry_response, attempt.results
                        ),
                        attempts=attempts,
                        verification_results=attempt.results,
                    )
        
        # Return final result
        final_attempt = attempts[-1]
        
        if self._config.require_all_verified:
            status = LoopStatus.FAILED
        elif final_attempt.partial_verified:
            status = LoopStatus.PARTIAL
        else:
            status = LoopStatus.FAILED
        
        return LoopResult(
            status=status,
            final_response=self._enhance_response(
                final_attempt.response_text,
                final_attempt.results,
            ),
            attempts=attempts,
            verification_results=final_attempt.results,
        )
    
    # -------------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------------
    
    def _generate(
        self,
        messages: List[Message],
        system: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM."""
        response = self._provider.complete(messages, system=system)
        return response.content
    
    def _generate_retry(
        self,
        user_input: str,
        system: Optional[str],
        context: List[Message],
        previous_attempt: VerificationAttempt,
    ) -> Optional[str]:
        """Generate a retry response based on strategy."""
        if self._config.retry_strategy == RetryStrategy.REGENERATE:
            return self._retry_regenerate(user_input, system, context, previous_attempt)
        elif self._config.retry_strategy == RetryStrategy.TARGETED:
            return self._retry_targeted(user_input, system, context, previous_attempt)
        elif self._config.retry_strategy == RetryStrategy.FALLBACK:
            return self._retry_fallback(previous_attempt)
        else:
            return None
    
    def _retry_regenerate(
        self,
        user_input: str,
        system: Optional[str],
        context: List[Message],
        previous: VerificationAttempt,
    ) -> str:
        """Regenerate the entire response with feedback."""
        # Add feedback about failed verifications
        failed_claims = [
            (c, r) for c, r in zip(previous.claims, previous.results)
            if not r.verified
        ]
        
        feedback_parts = ["Some claims in your previous response failed verification:"]
        for claim, result in failed_claims:
            feedback_parts.append(f"- Claim: \"{claim.original_text}\"")
            feedback_parts.append(f"  Error: {result.explanation}")
        
        feedback_parts.append(
            "\nPlease regenerate your response with correct, verifiable claims."
        )
        
        feedback = "\n".join(feedback_parts)
        
        # Add feedback to context
        messages = context + [
            Message.user(user_input),
            Message.assistant(previous.response_text),
            Message.user(feedback),
        ]
        
        return self._generate(messages, system)
    
    def _retry_targeted(
        self,
        user_input: str,
        system: Optional[str],
        context: List[Message],
        previous: VerificationAttempt,
    ) -> str:
        """Ask LLM to fix specific failed claims."""
        failed_claims = [
            (c, r) for c, r in zip(previous.claims, previous.results)
            if not r.verified
        ]
        
        if not failed_claims:
            return previous.response_text
        
        # Build targeted fix request
        fix_parts = [
            "The following claims need correction based on verification:"
        ]
        
        for claim, result in failed_claims:
            fix_parts.append(f"\n**Claim**: \"{claim.original_text}\"")
            fix_parts.append(f"**Issue**: {result.explanation}")
            if result.symbolic_proof:
                fix_parts.append(f"**Proof details**: {result.symbolic_proof}")
        
        fix_parts.append(
            "\nPlease provide a corrected version of your response that "
            "addresses these verification failures while maintaining accuracy."
        )
        
        fix_request = "\n".join(fix_parts)
        
        messages = context + [
            Message.user(user_input),
            Message.assistant(previous.response_text),
            Message.user(fix_request),
        ]
        
        return self._generate(messages, system)
    
    def _retry_fallback(self, previous: VerificationAttempt) -> str:
        """Generate a fallback response using verified claims only."""
        verified = [
            (c, r) for c, r in zip(previous.claims, previous.results)
            if r.verified
        ]
        
        if not verified:
            return (
                "I apologize, but I was unable to verify the claims in my response. "
                "Let me provide what I can verify:\n\n"
                "Please rephrase your question or ask about specific quantum concepts "
                "that can be mathematically verified."
            )
        
        # Build response from verified claims only
        parts = [
            "Based on verified mathematical proofs, here is what I can confirm:"
        ]
        
        for claim, result in verified:
            parts.append(f"\n**{claim.original_text}**")
            parts.append(f"✓ {result.explanation}")
            if result.symbolic_proof and self._config.include_proof_in_response:
                parts.append(f"Proof: ${result.symbolic_proof}$")
        
        return "\n".join(parts)
    
    # -------------------------------------------------------------------------
    # Verification Methods
    # -------------------------------------------------------------------------
    
    def _create_attempt(
        self,
        attempt_number: int,
        response_text: str,
    ) -> VerificationAttempt:
        """Create a verification attempt from a response."""
        # Extract claims
        if self._config.extract_claims:
            claims = self._claim_parser.parse_multiple(response_text)
        else:
            claims = []
        
        # Verify each claim
        results = []
        for claim in claims:
            result = self._verify_claim(claim)
            results.append(result)
            self._notify_verification(result)
        
        # Determine status
        all_verified = len(claims) > 0 and all(r.verified for r in results)
        partial_verified = any(r.verified for r in results)
        
        return VerificationAttempt(
            attempt_number=attempt_number,
            response_text=response_text,
            claims=claims,
            results=results,
            all_verified=all_verified,
            partial_verified=partial_verified,
        )
    
    def _verify_claim(self, claim: ParsedClaim) -> VerificationResult:
        """Verify a single claim."""
        try:
            # Route to appropriate verification method
            if claim.claim_type.value == "unitarity":
                return self._verification_api.verify_gate(claim.subject)
            elif claim.claim_type.value == "hermiticity":
                return self._verification_api.verify_operator(claim.subject)
            elif claim.claim_type.value == "normalization":
                return self._verification_api.verify_state(claim.subject)
            elif claim.claim_type.value == "entanglement":
                return self._verification_api.verify_state(
                    claim.subject, check_entanglement=True
                )
            else:
                return self._verification_api.verify_claim(claim.original_text)
                
        except Exception as e:
            logger.error(f"Verification error for claim '{claim.original_text}': {e}")
            return VerificationResult(
                verified=False,
                symbolic_proof="",
                explanation=f"Verification error: {str(e)}",
                confidence=0.0,
                domain="error",
                details={"error": str(e)},
            )
    
    # -------------------------------------------------------------------------
    # Response Enhancement
    # -------------------------------------------------------------------------
    
    def _enhance_response(
        self,
        response: str,
        results: List[VerificationResult],
    ) -> str:
        """Enhance response with verification status."""
        if not results or not self._config.include_proof_in_response:
            return response
        
        # Add verification summary
        verified_count = sum(1 for r in results if r.verified)
        total_count = len(results)
        
        if verified_count == total_count:
            summary = f"\n\n---\n✓ All {total_count} claims verified with mathematical proof."
        elif verified_count > 0:
            summary = f"\n\n---\n⚠ {verified_count}/{total_count} claims verified."
        else:
            summary = f"\n\n---\n✗ Verification failed for all claims."
        
        return response + summary
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_attempt(self, callback: Callable[[VerificationAttempt], None]) -> None:
        """Register callback for attempt events."""
        self._on_attempt.append(callback)
    
    def on_verification(self, callback: Callable[[VerificationResult], None]) -> None:
        """Register callback for verification events."""
        self._on_verification.append(callback)
    
    def _notify_attempt(self, attempt: VerificationAttempt) -> None:
        """Notify callbacks of an attempt."""
        for callback in self._on_attempt:
            try:
                callback(attempt)
            except Exception as e:
                logger.error(f"Attempt callback error: {e}")
    
    def _notify_verification(self, result: VerificationResult) -> None:
        """Notify callbacks of a verification."""
        for callback in self._on_verification:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Verification callback error: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_loop(
    llm_provider: LLMProvider,
    config: Optional[LoopConfig] = None,
) -> VerificationLoop:
    """Create a verification loop."""
    return VerificationLoop(llm_provider, config)


def run_verified(
    llm_provider: LLMProvider,
    prompt: str,
    system: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[str, LoopStatus]:
    """
    Run a single verified generation.
    
    Args:
        llm_provider: LLM provider
        prompt: User prompt
        system: System prompt
        max_retries: Maximum retry attempts
        
    Returns:
        Tuple of (response text, loop status)
    """
    config = LoopConfig(max_retries=max_retries)
    loop = VerificationLoop(llm_provider, config)
    result = loop.run(prompt, system=system)
    return result.final_response, result.status


def verify_response(
    response: str,
) -> List[VerificationResult]:
    """
    Verify claims in an existing response.
    
    Args:
        response: Response text to verify
        
    Returns:
        List of verification results
    """
    parser = ClaimParser()
    api = TutorVerificationAPI()
    
    claims = parser.parse_multiple(response)
    results = []
    
    for claim in claims:
        try:
            if claim.claim_type.value == "unitarity":
                result = api.verify_gate(claim.subject)
            elif claim.claim_type.value == "hermiticity":
                result = api.verify_operator(claim.subject)
            elif claim.claim_type.value == "normalization":
                result = api.verify_state(claim.subject)
            elif claim.claim_type.value == "entanglement":
                result = api.verify_state(claim.subject, check_entanglement=True)
            else:
                result = api.verify_claim(claim.original_text)
            
            results.append(result)
        except Exception as e:
            results.append(VerificationResult(
                verified=False,
                symbolic_proof="",
                explanation=f"Error: {str(e)}",
                confidence=0.0,
                domain="error",
            ))
    
    return results
