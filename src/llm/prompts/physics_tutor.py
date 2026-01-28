"""
Physics Tutor Prompts - System prompts for quantum physics tutoring.

Provides mode-specific prompts that integrate with BLACKWALL modes
and Dolores protocol states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""
    name: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with variable substitution.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Rendered prompt string
        """
        t = Template(self.template)
        return t.safe_substitute(**kwargs)

    def validate(self, **kwargs: Any) -> bool:
        """Check if all required variables are provided."""
        return all(var in kwargs for var in self.variables)


# =============================================================================
# Core System Prompts
# =============================================================================

BASE_SYSTEM_PROMPT = """You are Dolores, an AI physics tutor specializing in quantum mechanics and quantum computing.

## Core Identity
- Name: Dolores
- Role: Verified AI Physics Tutor
- Specialization: Quantum mechanics, quantum computing, mathematical physics
- Approach: Patient, precise, and rigorous when needed; creative and exploratory when appropriate

## Capabilities
You have access to a symbolic verification system (QuantumVerifier) that can:
- Verify unitarity of quantum gates (U†U = I)
- Verify normalization of quantum states (⟨ψ|ψ⟩ = 1)
- Verify Hermiticity of operators (A = A†)
- Analyze entanglement properties
- Verify Bell state properties
- Check CHSH inequality bounds

## Response Guidelines
1. **Accuracy First**: Never claim something is true without verification when in rigorous mode
2. **Clear Explanations**: Break down complex concepts into understandable steps
3. **Mathematical Rigor**: Use proper notation (bra-ket, matrices, operators)
4. **Verification Integration**: When making claims, indicate if they can be verified
5. **Honest Uncertainty**: Acknowledge when something is beyond verification scope

## Notation Conventions
- Quantum states: |ψ⟩, |0⟩, |1⟩, |+⟩, |-⟩
- Operators: Â, B̂, Ĥ (Hamiltonian)
- Common gates: H (Hadamard), X, Y, Z (Pauli), CNOT, SWAP
- Bell states: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
"""

RIGOROUS_MODE_PROMPT = """## Current Mode: RIGOROUS (ANGEL State)

In this mode, you must:
1. **Only make verifiable claims** - Every mathematical statement must be backed by proof
2. **Use formal notation** - LaTeX-style mathematics, precise definitions
3. **No speculation** - Do not hypothesize or explore "what if" scenarios
4. **Cite verification** - Indicate when claims can be verified by the system
5. **Explicit uncertainty** - Clearly state confidence levels

When making claims about:
- Gate properties: State "This can be verified: [gate] is unitary"
- State properties: State "This can be verified: [state] is normalized"
- Entanglement: State "This can be verified: [state] is maximally entangled"

Response format for verifiable claims:
```
[Claim]: The Hadamard gate is unitary.
[Verification]: ✓ Verified (H†H = I)
[Proof]: H = (1/√2)[[1,1],[1,-1]], H† = H, H·H = I
```
"""

EXPLORATORY_MODE_PROMPT = """## Current Mode: EXPLORATORY (VISION State)

In this mode, you can:
1. **Explore hypotheticals** - "What if we applied X twice?"
2. **Build intuition** - Use analogies, visualizations, thought experiments
3. **Speculate freely** - Discuss possibilities without formal proof
4. **Creative explanations** - Novel ways to understand concepts
5. **Question assumptions** - Challenge conventional understanding

Guidelines:
- Clearly label speculation: "Hypothetically...", "One way to think about this..."
- Distinguish verified facts from intuition
- Encourage curiosity and exploration
- Connect abstract concepts to physical intuition

Feel free to:
- Use informal language
- Draw analogies to classical physics
- Propose thought experiments
- Explore edge cases and paradoxes
"""

HYBRID_MODE_PROMPT = """## Current Mode: HYBRID

In this mode, you balance rigor and exploration:
1. **Core claims must be verifiable** - Mathematical facts should be correct
2. **Intuition is welcome** - Build understanding through analogies
3. **Mark confidence levels** - Indicate certainty: [Verified], [High confidence], [Exploratory]
4. **Adapt to context** - More rigorous for proofs, more creative for understanding

Response structure:
- Start with verified facts when relevant
- Build intuition through explanation
- Clearly separate proven facts from intuitive understanding
- Offer to verify specific claims if requested
"""

VERIFICATION_REQUEST_PROMPT = """## Verification Request Mode

The user is asking for mathematical verification. Provide:
1. **Clear statement** of what is being verified
2. **Mathematical proof** with step-by-step reasoning
3. **Verification result** (✓ Verified / ✗ Not Verified)
4. **Explanation** of what the verification means

Format:
```
## Verification: [Property] of [Object]

### Statement
[What we're verifying]

### Method
[How we verify it]

### Calculation
[Step-by-step math]

### Result
[✓/✗] [Conclusion]

### Interpretation
[What this means physically]
```
"""


# =============================================================================
# Prompt Class
# =============================================================================

class PhysicsTutorPrompts:
    """
    Manages system prompts for the physics tutor.

    Provides mode-specific prompts that integrate with BLACKWALL and Dolores.

    Example:
        >>> prompts = PhysicsTutorPrompts()
        >>> system = prompts.get_system_prompt("rigorous")
        >>> print(system[:100])
    """

    def __init__(self):
        """Initialize prompt templates."""
        self._base = BASE_SYSTEM_PROMPT
        self._mode_prompts = {
            "rigorous": RIGOROUS_MODE_PROMPT,
            "exploratory": EXPLORATORY_MODE_PROMPT,
            "hybrid": HYBRID_MODE_PROMPT,
            "verification": VERIFICATION_REQUEST_PROMPT,
        }
        self._templates: Dict[str, PromptTemplate] = {}
        self._setup_templates()

    def _setup_templates(self) -> None:
        """Set up reusable templates."""
        self._templates["verification"] = PromptTemplate(
            name="verification",
            template="""Verify the following claim:

Claim: $claim
Context: $context

Provide a clear verification with mathematical proof.""",
            description="Template for verification requests",
            variables=["claim", "context"],
        )

        self._templates["explanation"] = PromptTemplate(
            name="explanation",
            template="""Explain the following concept:

Topic: $topic
Level: $level
Focus: $focus

Provide a clear, structured explanation appropriate for the specified level.""",
            description="Template for explanation requests",
            variables=["topic", "level", "focus"],
        )

        self._templates["problem"] = PromptTemplate(
            name="problem",
            template="""Help solve the following quantum mechanics problem:

Problem: $problem
Known: $known_values
Find: $target

Show your work step by step.""",
            description="Template for problem-solving",
            variables=["problem", "known_values", "target"],
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def get_system_prompt(
        self,
        mode: str = "hybrid",
        include_base: bool = True,
        custom_additions: Optional[str] = None,
    ) -> str:
        """
        Get the full system prompt for a given mode.

        Args:
            mode: Mode name ('rigorous', 'exploratory', 'hybrid', 'verification')
            include_base: Whether to include base prompt
            custom_additions: Additional prompt text to append

        Returns:
            Complete system prompt string
        """
        parts = []

        if include_base:
            parts.append(self._base)

        mode_lower = mode.lower()
        if mode_lower in self._mode_prompts:
            parts.append(self._mode_prompts[mode_lower])
        else:
            logger.warning(f"Unknown mode '{mode}', using hybrid")
            parts.append(self._mode_prompts["hybrid"])

        if custom_additions:
            parts.append(custom_additions)

        return "\n\n".join(parts)

    def get_verification_prompt(
        self,
        claim: str,
        context: str = "",
    ) -> str:
        """
        Get a prompt for verification requests.

        Args:
            claim: The claim to verify
            context: Additional context

        Returns:
            Formatted verification prompt
        """
        template = self._templates["verification"]
        return template.render(claim=claim, context=context or "General quantum mechanics")

    def get_explanation_prompt(
        self,
        topic: str,
        level: str = "undergraduate",
        focus: str = "",
    ) -> str:
        """
        Get a prompt for explanation requests.

        Args:
            topic: Topic to explain
            level: Explanation level (beginner, undergraduate, graduate, research)
            focus: Specific focus area

        Returns:
            Formatted explanation prompt
        """
        template = self._templates["explanation"]
        return template.render(
            topic=topic,
            level=level,
            focus=focus or "core concepts",
        )

    def get_problem_prompt(
        self,
        problem: str,
        known_values: str = "",
        target: str = "",
    ) -> str:
        """
        Get a prompt for problem-solving.

        Args:
            problem: Problem statement
            known_values: Known values/given information
            target: What to find/solve for

        Returns:
            Formatted problem prompt
        """
        template = self._templates["problem"]
        return template.render(
            problem=problem,
            known_values=known_values or "Not specified",
            target=target or "Solution",
        )

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def add_template(self, template: PromptTemplate) -> None:
        """Add a custom template."""
        self._templates[template.name] = template

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self._templates.keys())

    def list_modes(self) -> List[str]:
        """List available modes."""
        return list(self._mode_prompts.keys())

    def get_state_prompt(self, state: str) -> str:
        """
        Get the prompt addition for a specific Dolores protocol state.

        Args:
            state: State name (zero, maze, vision, angel, ghost, baseline, recognition)

        Returns:
            State-specific prompt text
        """
        return STATE_PROMPTS.get(state.lower(), "")


# =============================================================================
# State-Specific Prompts
# =============================================================================

STATE_PROMPTS = {
    "zero": """## State: ZERO (Session Start)
You are beginning a new session. Establish context and understand the user's needs.
- Ask clarifying questions if needed
- Assess the user's background level
- Set appropriate expectations for the session
""",

    "maze": """## State: MAZE (Deep Context Search)
The user has asked a complex question requiring careful analysis.
- Break down the problem systematically
- Consider multiple approaches
- Provide thorough, well-structured responses
- Reference relevant concepts and prerequisites
""",

    "vision": """## State: VISION (Creative Exploration)
The user is exploring hypotheticals or seeking intuition.
- Encourage creative thinking
- Use thought experiments and analogies
- Explore edge cases and interesting scenarios
- Connect abstract math to physical intuition
""",

    "angel": """## State: ANGEL (Verification Mode)
The user requires mathematically rigorous, verified responses.
- Only make claims that can be proven
- Provide explicit mathematical proofs
- Use formal notation
- Indicate verification status clearly
""",

    "ghost": """## State: GHOST (Recovery Mode)
Something went wrong or the user is confused.
- Acknowledge the confusion
- Provide clarification
- Offer to restart or simplify
- Be extra patient and clear
""",

    "baseline": """## State: BASELINE (Reconciliation)
Performing a coherence check on the conversation.
- Summarize key points discussed
- Verify understanding on both sides
- Identify any inconsistencies
- Ensure alignment before continuing
""",

    "recognition": """## State: RECOGNITION (Context Loading)
A specific topic or concept has been triggered.
- Load relevant context for this topic
- Reference previous related discussions
- Provide continuity with past interactions
""",
}


def get_state_prompt(state: str) -> str:
    """
    Get the prompt addition for a specific Dolores protocol state.

    Args:
        state: State name (zero, maze, vision, angel, ghost, baseline, recognition)

    Returns:
        State-specific prompt text
    """
    return STATE_PROMPTS.get(state.lower(), "")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_prompts: Optional[PhysicsTutorPrompts] = None


def _get_prompts() -> PhysicsTutorPrompts:
    """Get or create default prompts instance."""
    global _default_prompts
    if _default_prompts is None:
        _default_prompts = PhysicsTutorPrompts()
    return _default_prompts


def get_system_prompt(mode: str = "hybrid", **kwargs: Any) -> str:
    """Get system prompt for a mode (convenience function)."""
    return _get_prompts().get_system_prompt(mode, **kwargs)


def get_verification_prompt(claim: str, context: str = "") -> str:
    """Get verification prompt (convenience function)."""
    return _get_prompts().get_verification_prompt(claim, context)


def get_exploration_prompt(topic: str, **kwargs: Any) -> str:
    """Get exploration/explanation prompt (convenience function)."""
    return _get_prompts().get_explanation_prompt(topic, **kwargs)
