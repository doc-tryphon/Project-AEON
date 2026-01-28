"""
LLM Providers - Concrete implementations for Claude and OpenAI.

Provides ready-to-use LLM providers with proper error handling,
retry logic, and token counting.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .interface import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    FinishReason,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ContextLengthError,
    ContentFilterError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Claude Provider
# =============================================================================

class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models.

    Example:
        >>> provider = ClaudeProvider(api_key="sk-ant-...")
        >>> provider.initialize()
        >>> response = provider.complete([Message.user("Hello!")])
    """

    # Model pricing (USD per 1M tokens) - as of Dec 2024
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    AVAILABLE_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        """
        Initialize Claude provider.

        Args:
            config: LLM configuration
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        super().__init__(config)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()

    def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if self._initialized:
            return

        if not self._api_key:
            raise AuthenticationError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
            self._initialized = True
            logger.info(f"Claude provider initialized with model {self._config.model}")
        except ImportError:
            raise LLMError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Anthropic client: {e}")

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages handled separately in Claude API
                continue
            result.append({
                "role": msg.role.value if msg.role != MessageRole.TOOL else "user",
                "content": msg.content,
            })
        return result

    def _extract_system(self, messages: List[Message], system: Optional[str]) -> Optional[str]:
        """Extract system message from messages or use provided one."""
        if system:
            return system
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return None

    def _handle_error(self, e: Exception) -> None:
        """Convert Anthropic exceptions to our exception types."""
        import anthropic

        if isinstance(e, anthropic.RateLimitError):
            raise RateLimitError(str(e))
        elif isinstance(e, anthropic.AuthenticationError):
            raise AuthenticationError(str(e))
        elif isinstance(e, anthropic.NotFoundError):
            raise ModelNotFoundError(str(e))
        elif isinstance(e, anthropic.BadRequestError):
            if "context_length" in str(e).lower():
                raise ContextLengthError(str(e))
            raise LLMError(str(e))
        else:
            raise LLMError(str(e))

    def complete(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using Claude."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages)
        system_prompt = self._extract_system(messages, system)

        try:
            params = {
                "model": kwargs.get("model", self._config.model),
                "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
                "temperature": kwargs.get("temperature", self._config.temperature),
                "system": system_prompt or "",
                "messages": converted,
            }
            
            stop_seq = kwargs.get("stop_sequences", self._config.stop_sequences)
            if stop_seq:
                params["stop_sequences"] = stop_seq
                
            response = self._client.messages.create(**params)

            # Map stop reason
            finish_reason = FinishReason.STOP
            if response.stop_reason == "max_tokens":
                finish_reason = FinishReason.LENGTH
            elif response.stop_reason == "tool_use":
                finish_reason = FinishReason.TOOL_CALL

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                finish_reason=finish_reason,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                ),
                metadata={"id": response.id},
            )

        except Exception as e:
            self._handle_error(e)

    async def complete_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion using Claude."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages)
        system_prompt = self._extract_system(messages, system)

        try:
            params = {
                "model": kwargs.get("model", self._config.model),
                "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
                "temperature": kwargs.get("temperature", self._config.temperature),
                "system": system_prompt or "",
                "messages": converted,
            }

            stop_seq = kwargs.get("stop_sequences", self._config.stop_sequences)
            if stop_seq:
                params["stop_sequences"] = stop_seq

            response = await self._async_client.messages.create(**params)

            finish_reason = FinishReason.STOP
            if response.stop_reason == "max_tokens":
                finish_reason = FinishReason.LENGTH

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                finish_reason=finish_reason,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                ),
            )

        except Exception as e:
            self._handle_error(e)

    def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream completion tokens from Claude."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages)
        system_prompt = self._extract_system(messages, system)

        try:
            with self._client.messages.stream(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                system=system_prompt or "",
                messages=converted,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_error(e)

    async def stream_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming from Claude."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages)
        system_prompt = self._extract_system(messages, system)

        try:
            async with self._async_client.messages.stream(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                system=system_prompt or "",
                messages=converted,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_error(e)

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        if not self._initialized:
            self.initialize()

        try:
            # Use the client's count_tokens method
            return self._client.count_tokens(text)
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def estimate_cost(self, usage: TokenUsage) -> Optional[float]:
        """Estimate cost based on model pricing."""
        pricing = self.PRICING.get(self._config.model)
        if not pricing:
            return None

        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(LLMProvider):
    """
    OpenAI provider.

    Supports GPT-4, GPT-4 Turbo, GPT-3.5, and other OpenAI models.

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> provider.initialize()
        >>> response = provider.complete([Message.user("Hello!")])
    """

    PRICING = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    AVAILABLE_MODELS = [
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        super().__init__(config)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Default to GPT-4o if no model specified
        if self._config.model.startswith("claude"):
            self._config.model = "gpt-4o"

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        if not self._api_key:
            raise AuthenticationError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        try:
            import openai
            base_url = os.environ.get("OPENAI_BASE_URL")
            self._client = openai.OpenAI(api_key=self._api_key, base_url=base_url)
            self._async_client = openai.AsyncOpenAI(api_key=self._api_key, base_url=base_url)
            self._initialized = True
            logger.info(f"OpenAI provider initialized with model {self._config.model} (URL: {base_url or 'default'})")
        except ImportError:
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI client: {e}")

    def _convert_messages(
        self, messages: List[Message], system: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Convert Message objects to OpenAI format."""
        result = []
        
        # Add system message if provided
        if system:
            result.append({"role": "system", "content": system})
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and not system:
                result.append({"role": "system", "content": msg.content})
            elif msg.role != MessageRole.SYSTEM:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return result

    def _handle_error(self, e: Exception) -> None:
        """Convert OpenAI exceptions to our exception types."""
        import openai

        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(str(e))
        elif isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(str(e))
        elif isinstance(e, openai.NotFoundError):
            raise ModelNotFoundError(str(e))
        elif isinstance(e, openai.BadRequestError):
            if "context_length" in str(e).lower():
                raise ContextLengthError(str(e))
            raise LLMError(str(e))
        else:
            raise LLMError(str(e))

    def complete(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages, system)

        try:
            response = self._client.chat.completions.create(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                messages=converted,
                stop=kwargs.get("stop_sequences", self._config.stop_sequences) or None,
            )

            choice = response.choices[0]
            finish_reason = FinishReason.STOP
            if choice.finish_reason == "length":
                finish_reason = FinishReason.LENGTH
            elif choice.finish_reason == "content_filter":
                finish_reason = FinishReason.CONTENT_FILTER
            elif choice.finish_reason == "tool_calls":
                finish_reason = FinishReason.TOOL_CALL

            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=finish_reason,
                model=response.model,
                usage=usage,
                metadata={"id": response.id},
            )

        except Exception as e:
            self._handle_error(e)

    async def complete_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion using OpenAI."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages, system)

        try:
            response = await self._async_client.chat.completions.create(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                messages=converted,
            )

            choice = response.choices[0]
            finish_reason = FinishReason.STOP
            if choice.finish_reason == "length":
                finish_reason = FinishReason.LENGTH

            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=finish_reason,
                model=response.model,
                usage=usage,
            )

        except Exception as e:
            self._handle_error(e)

    def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream completion tokens from OpenAI."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages, system)

        try:
            stream = self._client.chat.completions.create(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                messages=converted,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)

    async def stream_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming from OpenAI."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)
        converted = self._convert_messages(messages, system)

        try:
            stream = await self._async_client.chat.completions.create(
                model=kwargs.get("model", self._config.model),
                max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
                temperature=kwargs.get("temperature", self._config.temperature),
                messages=converted,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self._config.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    def estimate_cost(self, usage: TokenUsage) -> Optional[float]:
        """Estimate cost based on model pricing."""
        pricing = self.PRICING.get(self._config.model)
        if not pricing:
            return None

        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)


# =============================================================================
# Mock Provider (for testing)
# =============================================================================

class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses without making API calls.

    Example:
        >>> provider = MockProvider(responses=["Hello!", "How can I help?"])
        >>> provider.complete([Message.user("Hi")])
        LLMResponse(content="Hello!", ...)
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        responses: Optional[List[str]] = None,
        delay: float = 0.0,
    ):
        """
        Initialize mock provider.

        Args:
            config: LLM configuration
            responses: List of responses to return (cycles through)
            delay: Simulated delay in seconds
        """
        super().__init__(config)
        self._responses = responses or ["This is a mock response."]
        self._response_index = 0
        self._delay = delay
        self._call_history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def available_models(self) -> List[str]:
        return ["mock-model"]

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get history of calls made to the provider."""
        return self._call_history.copy()

    def initialize(self) -> None:
        """Initialize mock provider (no-op)."""
        self._initialized = True
        logger.info("Mock provider initialized")

    def _get_next_response(self) -> str:
        """Get the next response from the list."""
        response = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1
        return response

    def complete(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return mock completion."""
        if not self._initialized:
            self.initialize()

        self.validate_messages(messages)

        # Record call
        self._call_history.append({
            "messages": [m.to_dict() for m in messages],
            "system": system,
            "kwargs": kwargs,
        })

        # Simulate delay
        if self._delay > 0:
            time.sleep(self._delay)

        content = self._get_next_response()
        tokens = len(content.split())

        return LLMResponse(
            content=content,
            finish_reason=FinishReason.STOP,
            model="mock-model",
            usage=TokenUsage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=tokens,
                total_tokens=sum(len(m.content.split()) for m in messages) + tokens,
            ),
        )

    async def complete_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async mock completion."""
        import asyncio
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self.complete(messages, system, **kwargs)

    def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream mock tokens."""
        if not self._initialized:
            self.initialize()

        content = self._get_next_response()
        words = content.split()

        for word in words:
            if self._delay > 0:
                time.sleep(self._delay / len(words))
            yield word + " "

    async def stream_async(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async mock streaming."""
        import asyncio

        content = self._get_next_response()
        words = content.split()

        for word in words:
            if self._delay > 0:
                await asyncio.sleep(self._delay / len(words))
            yield word + " "

    def count_tokens(self, text: str) -> int:
        """Simple word-based token count."""
        return len(text.split())

    def set_responses(self, responses: List[str]) -> None:
        """Set new responses for the mock."""
        self._responses = responses
        self._response_index = 0

    def reset(self) -> None:
        """Reset call history and response index."""
        self._call_history.clear()
        self._response_index = 0


# =============================================================================
# Factory Functions
# =============================================================================

PROVIDERS: Dict[str, type] = {
    "anthropic": ClaudeProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "gpt": OpenAIProvider,
    "mock": MockProvider,
}


def create_provider(
    provider: str = "anthropic",
    config: Optional[LLMConfig] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Create an LLM provider by name.

    Args:
        provider: Provider name ('anthropic', 'openai', 'mock')
        config: LLM configuration
        api_key: API key for the provider
        **kwargs: Additional provider-specific arguments

    Returns:
        Initialized LLMProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    provider_lower = provider.lower()

    if provider_lower not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider}. Available: {available}"
        )

    provider_class = PROVIDERS[provider_lower]

    if provider_class == MockProvider:
        return provider_class(config=config, **kwargs)
    else:
        return provider_class(config=config, api_key=api_key, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available provider names."""
    return list(PROVIDERS.keys())
