"""
LLM module - Language model providers and integration.

Sprint 4 components:
- interface.py: Abstract LLMProvider and common types
- providers.py: Claude and OpenAI implementations
- prompts/: System prompts for different contexts
"""

from .interface import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)

from .providers import (
    ClaudeProvider,
    OpenAIProvider,
    MockProvider,
    create_provider,
    get_available_providers,
)

__all__ = [
    # interface
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "MessageRole",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    # providers
    "ClaudeProvider",
    "OpenAIProvider",
    "MockProvider",
    "create_provider",
    "get_available_providers",
]
