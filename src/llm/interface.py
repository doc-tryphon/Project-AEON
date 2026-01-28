"""
LLM Interface - Abstract base class and common types for LLM providers.

Provides a unified interface for interacting with different LLM providers
(Claude, OpenAI, etc.) with support for streaming, token counting, and
error handling.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Authentication failed (invalid API key)."""
    pass


class ModelNotFoundError(LLMError):
    """Requested model not available."""
    pass


class ContextLengthError(LLMError):
    """Input exceeds model's context length."""
    def __init__(self, message: str, max_tokens: Optional[int] = None):
        super().__init__(message)
        self.max_tokens = max_tokens


class ContentFilterError(LLMError):
    """Content was filtered by safety systems."""
    pass


# =============================================================================
# Enums
# =============================================================================

class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason the model stopped generating."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost_estimate(self) -> Optional[float]:
        """Estimate cost (requires model-specific pricing)."""
        # Override in subclass or set externally
        return None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    finish_reason: FinishReason
    model: str
    usage: Optional[TokenUsage] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def stopped_naturally(self) -> bool:
        """Check if generation stopped naturally (not truncated)."""
        return self.finish_reason == FinishReason.STOP

    @property
    def was_truncated(self) -> bool:
        """Check if response was truncated due to length."""
        return self.finish_reason == FinishReason.LENGTH

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    # Model selection
    model: str = "claude-sonnet-4-20250514"
    
    # Generation parameters
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    
    # Behavior
    stream: bool = False
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Advanced
    stop_sequences: List[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "timeout": self.timeout,
        }


# =============================================================================
# Abstract Base Class
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Defines the interface that all LLM providers must implement.

    Example:
        >>> provider = ClaudeProvider(api_key="...")
        >>> response = provider.complete([Message.user("Hello!")])
        >>> print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the provider.

        Args:
            config: Provider configuration
        """
        self._config = config or LLMConfig()
        self._initialized = False
        self._client: Any = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def config(self) -> LLMConfig:
        """Get current configuration."""
        return self._config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'anthropic', 'openai')."""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider (create client, validate API key).

        Raises:
            AuthenticationError: If API key is invalid
            LLMError: If initialization fails
        """
        pass

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history
            system: System prompt (optional, overrides config)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If generation fails
            RateLimitError: If rate limited
            ContextLengthError: If input too long
        """
        pass

    @abstractmethod
    async def complete_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Async version of complete().

        Args:
            messages: Conversation history
            system: System prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        pass

    # -------------------------------------------------------------------------
    # Streaming Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream completion tokens.

        Args:
            messages: Conversation history
            system: System prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as strings
        """
        pass

    @abstractmethod
    async def stream_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async streaming completion.

        Args:
            messages: Conversation history
            system: System prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as strings
        """
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    def estimate_cost(self, usage: TokenUsage) -> Optional[float]:
        """
        Estimate cost for token usage.

        Args:
            usage: Token usage statistics

        Returns:
            Estimated cost in USD, or None if pricing unavailable
        """
        # Override in subclass with model-specific pricing
        return None

    def validate_messages(self, messages: List[Message]) -> bool:
        """
        Validate message list before sending.

        Args:
            messages: Messages to validate

        Returns:
            True if valid

        Raises:
            ValueError: If messages are invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        for i, msg in enumerate(messages):
            if not msg.content:
                raise ValueError(f"Message {i} has empty content")
            if msg.role not in MessageRole:
                raise ValueError(f"Message {i} has invalid role: {msg.role}")

        return True

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "LLMProvider":
        """Enter context manager."""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the provider and clean up resources."""
        self._client = None
        self._initialized = False

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def quick_complete(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Quick completion with just a prompt string.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            Generated text
        """
        messages = [Message.user(prompt)]
        response = self.complete(messages, system=system)
        return response.content

    def chat(
        self,
        user_message: str,
        history: Optional[List[Message]] = None,
        system: Optional[str] = None,
    ) -> tuple[str, List[Message]]:
        """
        Simple chat interface.

        Args:
            user_message: New user message
            history: Previous conversation history
            system: System prompt

        Returns:
            Tuple of (response text, updated history)
        """
        history = history or []
        history.append(Message.user(user_message))

        response = self.complete(history, system=system)
        history.append(Message.assistant(response.content))

        return response.content, history


# =============================================================================
# Callback Types
# =============================================================================

StreamCallback = Callable[[str], None]
CompletionCallback = Callable[[LLMResponse], None]
ErrorCallback = Callable[[LLMError], None]


# =============================================================================
# Factory Function
# =============================================================================

def create_config(
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> LLMConfig:
    """
    Create an LLM configuration.

    Args:
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration options

    Returns:
        LLMConfig instance
    """
    return LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
