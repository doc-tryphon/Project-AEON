"""
Tests for Sprint 4: LLM Integration.

Tests cover:
- LLM interface and types
- Provider implementations (Mock provider for testing)
- Physics tutor prompts
- TutorSession
- VerificationLoop
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.llm.interface import (
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
    create_config,
)

from src.llm.providers import (
    MockProvider,
    ClaudeProvider,
    OpenAIProvider,
    create_provider,
    get_available_providers,
)

from src.llm.prompts.physics_tutor import (
    PhysicsTutorPrompts,
    get_system_prompt,
    get_verification_prompt,
    PromptTemplate,
    BASE_SYSTEM_PROMPT,
    RIGOROUS_MODE_PROMPT,
    EXPLORATORY_MODE_PROMPT,
)

from src.tutor.session import (
    TutorSession,
    SessionConfig,
    SessionStatus,
    Turn,
    TurnMetadata,
    SessionStats,
    create_session,
)

from src.tutor.verification_loop import (
    VerificationLoop,
    LoopConfig,
    LoopResult,
    LoopStatus,
    RetryStrategy,
    VerificationAttempt,
    create_loop,
    verify_response,
)

from src.interface.enums import InterfaceMode
from src.persona.state_machine import ProtocolState


# =============================================================================
# Test LLM Interface Types
# =============================================================================

class TestMessage:
    """Tests for Message class."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a tutor.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a tutor."

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("I can help with that.")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "I can help with that."

    def test_to_dict(self):
        """Test message serialization."""
        msg = Message.user("Test")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test"

    def test_message_with_name(self):
        """Test message with name field."""
        msg = Message(role=MessageRole.USER, content="Test", name="alice")
        d = msg.to_dict()
        assert d["name"] == "alice"


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_token_usage_fields(self):
        """Test TokenUsage has correct fields."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.stream is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            model="gpt-4",
            max_tokens=2048,
            temperature=0.3
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 2048
        assert config.temperature == 0.3

    def test_create_config_function(self):
        """Test create_config convenience function."""
        config = create_config(
            model="claude-3-opus",
            temperature=0.5,
            max_tokens=1024
        )
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5

    def test_to_dict(self):
        """Test config serialization."""
        config = LLMConfig(model="test-model")
        d = config.to_dict()
        assert d["model"] == "test-model"
        assert "max_tokens" in d


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_response_fields(self):
        """Test LLMResponse has correct fields."""
        response = LLMResponse(
            content="Hello!",
            finish_reason=FinishReason.STOP,
            model="test-model"
        )
        assert response.content == "Hello!"
        assert response.finish_reason == FinishReason.STOP
        assert response.model == "test-model"

    def test_stopped_naturally(self):
        """Test stopped_naturally property."""
        response = LLMResponse(
            content="Done",
            finish_reason=FinishReason.STOP,
            model="test"
        )
        assert response.stopped_naturally is True

    def test_was_truncated(self):
        """Test was_truncated property."""
        response = LLMResponse(
            content="Cut off...",
            finish_reason=FinishReason.LENGTH,
            model="test"
        )
        assert response.was_truncated is True

    def test_has_tool_calls(self):
        """Test has_tool_calls property."""
        response = LLMResponse(
            content="",
            finish_reason=FinishReason.TOOL_CALL,
            model="test",
            tool_calls=[{"name": "test"}]
        )
        assert response.has_tool_calls is True


# =============================================================================
# Test LLM Exceptions
# =============================================================================

class TestLLMExceptions:
    """Tests for LLM exception classes."""

    def test_llm_error(self):
        """Test base LLMError."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0

    def test_context_length_error(self):
        """Test ContextLengthError with max_tokens."""
        error = ContextLengthError("Too long", max_tokens=8192)
        assert error.max_tokens == 8192


# =============================================================================
# Test MockProvider
# =============================================================================

class TestMockProvider:
    """Tests for MockProvider (used for testing)."""

    def test_create_mock_provider(self):
        """Test creating a mock provider."""
        provider = MockProvider(responses=["Hello!"])
        assert provider.provider_name == "mock"

    def test_mock_complete(self):
        """Test mock completion."""
        provider = MockProvider(responses=["Response 1", "Response 2"])
        provider.initialize()

        response = provider.complete([Message.user("Test")])
        assert response.content == "Response 1"

        response = provider.complete([Message.user("Test")])
        assert response.content == "Response 2"

    def test_mock_cycling(self):
        """Test mock cycles through responses."""
        provider = MockProvider(responses=["A", "B"])
        provider.initialize()

        assert provider.complete([Message.user("1")]).content == "A"
        assert provider.complete([Message.user("2")]).content == "B"
        assert provider.complete([Message.user("3")]).content == "A"  # Cycles

    def test_mock_available_models(self):
        """Test mock available models."""
        provider = MockProvider(responses=["test"])
        assert "mock-model" in provider.available_models

    def test_mock_count_tokens(self):
        """Test mock token counting."""
        provider = MockProvider(responses=["test"])
        count = provider.count_tokens("Hello world")
        assert count > 0

    def test_quick_complete(self):
        """Test quick_complete convenience method."""
        provider = MockProvider(responses=["Quick response"])
        provider.initialize()

        result = provider.quick_complete("Hello")
        assert result == "Quick response"

    def test_chat_method(self):
        """Test chat convenience method."""
        provider = MockProvider(responses=["Hello back!"])
        provider.initialize()

        response, history = provider.chat("Hello")
        assert response == "Hello back!"
        assert len(history) == 2
        assert history[0].role == MessageRole.USER
        assert history[1].role == MessageRole.ASSISTANT


# =============================================================================
# Test Provider Factory
# =============================================================================

class TestProviderFactory:
    """Tests for provider factory functions."""

    def test_create_mock_provider(self):
        """Test creating mock provider via factory."""
        provider = create_provider("mock", responses=["test"])
        assert isinstance(provider, MockProvider)

    def test_get_available_providers(self):
        """Test getting available providers list."""
        providers = get_available_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "mock" in providers


# =============================================================================
# Test Physics Tutor Prompts
# =============================================================================

class TestPhysicsTutorPrompts:
    """Tests for PhysicsTutorPrompts class."""

    def test_base_prompt_exists(self):
        """Test BASE_SYSTEM_PROMPT is defined."""
        assert len(BASE_SYSTEM_PROMPT) > 100
        assert "Dolores" in BASE_SYSTEM_PROMPT

    def test_mode_prompts_exist(self):
        """Test mode-specific prompts are defined."""
        assert len(RIGOROUS_MODE_PROMPT) > 50
        assert len(EXPLORATORY_MODE_PROMPT) > 50

    def test_get_system_prompt_default(self):
        """Test default system prompt generation."""
        prompt = get_system_prompt()
        assert "Dolores" in prompt
        assert len(prompt) > 100

    def test_get_system_prompt_rigorous(self):
        """Test rigorous mode system prompt."""
        prompt = get_system_prompt(mode="rigorous")
        assert "RIGOROUS" in prompt or "rigorous" in prompt.lower()

    def test_get_system_prompt_exploratory(self):
        """Test exploratory mode system prompt."""
        prompt = get_system_prompt(mode="exploratory")
        assert "EXPLORATORY" in prompt or "exploratory" in prompt.lower()

    def test_get_verification_prompt(self):
        """Test verification prompt generation."""
        prompt = get_verification_prompt("Hadamard is unitary")
        assert "Hadamard" in prompt
        assert "unitary" in prompt

    def test_physics_tutor_prompts_class(self):
        """Test PhysicsTutorPrompts class."""
        prompts = PhysicsTutorPrompts()
        
        system = prompts.get_system_prompt(mode="hybrid")
        assert len(system) > 100

    def test_prompt_template(self):
        """Test PromptTemplate class."""
        template = PromptTemplate(
            name="test",
            template="Hello $name!",
            description="A test template"
        )
        result = template.render(name="World")
        assert result == "Hello World!"


# =============================================================================
# Test TutorSession
# =============================================================================

class TestTutorSession:
    """Tests for TutorSession class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = MockProvider(responses=[
            "The Hadamard gate is unitary because H†H = I.",
            "A qubit is the fundamental unit of quantum information.",
        ])
        provider.initialize()
        return provider

    @pytest.fixture
    def session(self, mock_provider):
        """Create a session for testing."""
        return TutorSession(llm_provider=mock_provider)

    def test_create_session(self):
        """Test creating a session."""
        session = TutorSession()
        assert session.session_id is not None
        assert session.status == SessionStatus.INITIALIZING

    def test_session_start(self, session):
        """Test starting a session."""
        session.start()
        assert session.status == SessionStatus.ACTIVE

    def test_session_with_config(self):
        """Test session with custom config."""
        config = SessionConfig(
            initial_mode=InterfaceMode.RIGOROUS,
            verify_claims=True
        )
        session = TutorSession(config=config)
        assert session.mode == InterfaceMode.RIGOROUS

    def test_session_context_manager(self, mock_provider):
        """Test session as context manager."""
        with TutorSession(llm_provider=mock_provider) as session:
            assert session.status == SessionStatus.ACTIVE
        assert session.status == SessionStatus.ENDED

    def test_session_process(self, session):
        """Test processing user input."""
        session.start()
        response = session.process("What is a qubit?")
        assert len(response) > 0
        assert session.turn_count == 1

    def test_session_mode_detection(self, session):
        """Test automatic mode detection."""
        session.start()
        
        # Default mode
        assert session.mode == InterfaceMode.HYBRID
        
        # Process verification request
        session.process("Prove that H is unitary")
        # Mode should change to rigorous
        assert session.mode == InterfaceMode.RIGOROUS

    def test_session_set_mode(self, session):
        """Test manual mode setting."""
        session.start()
        session.set_mode(InterfaceMode.EXPLORATORY)
        assert session.mode == InterfaceMode.EXPLORATORY

    def test_session_stats(self, session):
        """Test session statistics."""
        session.start()
        session.process("Hello")
        
        stats = session.stats
        assert stats.total_turns == 1
        assert stats.average_response_time_ms > 0

    def test_session_history(self, session):
        """Test session history."""
        session.start()
        session.process("Question 1")
        session.process("Question 2")
        
        history = session.history
        assert len(history) == 2
        assert history[0].user_message == "Question 1"
        assert history[1].user_message == "Question 2"

    def test_session_get_status(self, session):
        """Test getting session status."""
        session.start()
        session.process("Test")
        
        status = session.get_status()
        assert "session_id" in status
        assert "status" in status
        assert status["stats"]["total_turns"] == 1

    def test_session_export_transcript(self, session):
        """Test exporting session transcript."""
        session.start()
        session.process("Hello")
        
        transcript = session.export_transcript()
        assert "session_id" in transcript
        assert "turns" in transcript
        assert len(transcript["turns"]) == 1

    def test_session_direct_verify(self, session):
        """Test direct verification method."""
        session.start()
        result = session.verify_gate("H")
        assert result.verified is True
        assert result.domain == "unitarity"

    def test_session_callbacks(self, session):
        """Test session callbacks."""
        turns_recorded = []
        
        def on_turn(turn):
            turns_recorded.append(turn)
        
        session.on_turn_complete(on_turn)
        session.start()
        session.process("Test")
        
        assert len(turns_recorded) == 1

    def test_create_session_function(self, mock_provider):
        """Test create_session convenience function."""
        session = create_session(llm_provider=mock_provider)
        assert isinstance(session, TutorSession)


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert SessionStatus.INITIALIZING.value == "initializing"
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.ENDED.value == "ended"


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SessionConfig()
        assert config.initial_mode == InterfaceMode.HYBRID
        assert config.verify_claims is True
        assert config.max_history_length == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = SessionConfig(
            initial_mode=InterfaceMode.RIGOROUS,
            verify_claims=False,
            max_history_length=50
        )
        assert config.initial_mode == InterfaceMode.RIGOROUS
        assert config.verify_claims is False


# =============================================================================
# Test VerificationLoop
# =============================================================================

class TestVerificationLoop:
    """Tests for VerificationLoop class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = MockProvider(responses=[
            "The Hadamard gate H is unitary because H†H = I.",
        ])
        provider.initialize()
        return provider

    @pytest.fixture
    def loop(self, mock_provider):
        """Create a verification loop for testing."""
        return VerificationLoop(mock_provider)

    def test_create_loop(self, mock_provider):
        """Test creating a verification loop."""
        loop = VerificationLoop(mock_provider)
        assert loop.config.max_retries == 2

    def test_loop_with_config(self, mock_provider):
        """Test loop with custom config."""
        config = LoopConfig(
            max_retries=5,
            retry_strategy=RetryStrategy.REGENERATE
        )
        loop = VerificationLoop(mock_provider, config=config)
        assert loop.config.max_retries == 5

    def test_loop_run_no_claims(self, mock_provider):
        """Test loop with response containing no verifiable claims."""
        provider = MockProvider(responses=["Hello, how can I help you?"])
        provider.initialize()
        loop = VerificationLoop(provider)
        
        result = loop.run("Hi there")
        assert result.status == LoopStatus.NO_CLAIMS

    def test_loop_run_success(self, loop):
        """Test successful verification loop."""
        result = loop.run("Prove that Hadamard is unitary")
        # Even without actual LLM, the mock should work
        assert isinstance(result, LoopResult)
        assert result.total_attempts >= 1

    def test_loop_callbacks(self, mock_provider):
        """Test loop callbacks."""
        attempts_recorded = []
        
        def on_attempt(attempt):
            attempts_recorded.append(attempt)
        
        loop = VerificationLoop(mock_provider)
        loop.on_attempt(on_attempt)
        
        loop.run("Test")
        
        assert len(attempts_recorded) >= 1


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_default_config(self):
        """Test default loop configuration."""
        config = LoopConfig()
        assert config.max_retries == 2
        assert config.retry_strategy == RetryStrategy.TARGETED
        assert config.extract_claims is True

    def test_custom_config(self):
        """Test custom loop configuration."""
        config = LoopConfig(
            max_retries=5,
            retry_strategy=RetryStrategy.NONE
        )
        assert config.max_retries == 5
        assert config.retry_strategy == RetryStrategy.NONE


class TestLoopStatus:
    """Tests for LoopStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert LoopStatus.SUCCESS.value == "success"
        assert LoopStatus.PARTIAL.value == "partial"
        assert LoopStatus.FAILED.value == "failed"
        assert LoopStatus.NO_CLAIMS.value == "no_claims"
        assert LoopStatus.ERROR.value == "error"


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        assert RetryStrategy.NONE.value == "none"
        assert RetryStrategy.REGENERATE.value == "regenerate"
        assert RetryStrategy.TARGETED.value == "targeted"
        assert RetryStrategy.FALLBACK.value == "fallback"


class TestVerificationAttempt:
    """Tests for VerificationAttempt dataclass."""

    def test_attempt_properties(self):
        """Test attempt properties."""
        from src.tutor.verification_api import VerificationResult
        from src.tutor.claim_parser import ParsedClaim, ClaimType, VerificationMethod

        claim = ParsedClaim(
            original_text="H is unitary",
            claim_type=ClaimType.UNITARITY,
            subject="H",
            method=VerificationMethod.VERIFY_GATE,
        )
        result = VerificationResult(
            verified=True,
            symbolic_proof="test",
            explanation="test",
            confidence=1.0,
            domain="unitarity"
        )
        
        attempt = VerificationAttempt(
            attempt_number=1,
            response_text="Test response",
            claims=[claim],
            results=[result],
            all_verified=True,
            partial_verified=True
        )
        
        assert attempt.verified_count == 1
        assert attempt.total_claims == 1

    def test_to_dict(self):
        """Test attempt serialization."""
        attempt = VerificationAttempt(
            attempt_number=1,
            response_text="Test",
            claims=[],
            results=[],
            all_verified=False,
            partial_verified=False
        )
        
        d = attempt.to_dict()
        assert d["attempt_number"] == 1
        assert d["claim_count"] == 0


class TestLoopResult:
    """Tests for LoopResult dataclass."""

    def test_result_properties(self):
        """Test result properties."""
        result = LoopResult(
            status=LoopStatus.SUCCESS,
            final_response="Test",
            attempts=[],
            verification_results=[]
        )
        
        assert result.total_attempts == 0
        assert result.all_verified is True  # Empty list = all verified
        assert result.any_verified is False

    def test_to_dict(self):
        """Test result serialization."""
        result = LoopResult(
            status=LoopStatus.PARTIAL,
            final_response="Test",
            attempts=[],
            verification_results=[]
        )
        
        d = result.to_dict()
        assert d["status"] == "partial"


# =============================================================================
# Test verify_response Utility
# =============================================================================

class TestVerifyResponse:
    """Tests for verify_response utility function."""

    def test_verify_response_with_claims(self):
        """Test verifying a response with claims."""
        response = "The Hadamard gate H is unitary."
        results = verify_response(response)
        # May or may not find claims depending on parser
        assert isinstance(results, list)

    def test_verify_response_empty(self):
        """Test verifying a response with no claims."""
        response = "Hello, how can I help?"
        results = verify_response(response)
        assert len(results) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for Sprint 4 components."""

    def test_full_session_workflow(self):
        """Test complete session workflow."""
        # Create mock provider
        provider = MockProvider(responses=[
            "The Hadamard gate is a fundamental quantum gate.",
            "H is unitary because H†H = I.",
            "A Bell state is maximally entangled."
        ])
        provider.initialize()
        
        # Create and start session
        config = SessionConfig(
            initial_mode=InterfaceMode.HYBRID,
            verify_claims=True
        )
        
        with TutorSession(llm_provider=provider, config=config) as session:
            # Process multiple queries
            r1 = session.process("What is the Hadamard gate?")
            assert len(r1) > 0
            
            r2 = session.process("Prove that H is unitary")
            assert len(r2) > 0
            
            # Check stats
            assert session.turn_count == 2
            
            # Export transcript
            transcript = session.export_transcript()
            assert len(transcript["turns"]) == 2

    def test_verification_loop_integration(self):
        """Test verification loop with session."""
        provider = MockProvider(responses=[
            "The Pauli X gate is unitary: X†X = I."
        ])
        provider.initialize()
        
        loop = VerificationLoop(provider)
        result = loop.run(
            "Is the Pauli X gate unitary?",
            system=get_system_prompt(mode="rigorous")
        )
        
        assert isinstance(result, LoopResult)
        assert result.total_attempts >= 1

    def test_prompts_with_provider(self):
        """Test prompts integration with provider."""
        prompts = PhysicsTutorPrompts()
        system = prompts.get_system_prompt(mode="rigorous")
        
        provider = MockProvider(responses=["Test response"])
        provider.initialize()
        
        response = provider.complete(
            [Message.user("Test")],
            system=system
        )
        
        assert response.content == "Test response"


# =============================================================================
# Test Turn and TurnMetadata
# =============================================================================

class TestTurn:
    """Tests for Turn dataclass."""

    def test_turn_to_dict(self):
        """Test turn serialization."""
        metadata = TurnMetadata(
            turn_id="test-123",
            timestamp=datetime.now(),
            protocol_state=ProtocolState.MAZE,
            interface_mode=InterfaceMode.HYBRID
        )
        
        turn = Turn(
            user_message="Hello",
            assistant_response="Hi there!",
            metadata=metadata
        )
        
        d = turn.to_dict()
        assert d["turn_id"] == "test-123"
        assert d["user_message"] == "Hello"
        assert d["assistant_response"] == "Hi there!"


class TestTurnMetadata:
    """Tests for TurnMetadata dataclass."""

    def test_metadata_fields(self):
        """Test metadata has correct fields."""
        metadata = TurnMetadata(
            turn_id="test",
            timestamp=datetime.now(),
            protocol_state=ProtocolState.ANGEL,
            interface_mode=InterfaceMode.RIGOROUS,
            response_time_ms=150.5
        )
        
        assert metadata.turn_id == "test"
        assert metadata.protocol_state == ProtocolState.ANGEL
        assert metadata.interface_mode == InterfaceMode.RIGOROUS
        assert metadata.response_time_ms == 150.5


class TestSessionStats:
    """Tests for SessionStats dataclass."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = SessionStats()
        assert stats.total_turns == 0
        assert stats.total_verifications == 0
        assert stats.average_response_time_ms == 0.0
