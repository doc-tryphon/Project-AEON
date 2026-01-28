"""
LLM prompts module - System prompts for different contexts.

Contains specialized prompts for:
- Physics tutoring
- Verification mode
- Exploratory mode
"""

from .physics_tutor import (
    PhysicsTutorPrompts,
    get_system_prompt,
    get_verification_prompt,
    get_exploration_prompt,
    PromptTemplate,
)

__all__ = [
    "PhysicsTutorPrompts",
    "get_system_prompt",
    "get_verification_prompt",
    "get_exploration_prompt",
    "PromptTemplate",
]
