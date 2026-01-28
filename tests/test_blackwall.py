"""
Tests for BLACKWALL Mode Controller - Sprint 2 components.

Tests cover:
- InterfaceMode enum and properties
- BlackwallController mode management
- ModeDetector pattern matching
- Integration between components
"""

import pytest
from datetime import datetime

from src.interface import (
    InterfaceMode,
    RequestType,
    TransitionReason,
    BlackwallController,
    ModeConfig,
    ModeTransition,
    MODE_CONFIGS,
    ModeDetector,
    DetectionResult,
    analyze,
    suggest_mode,
    is_command,
)


# =============================================================================
# InterfaceMode Tests
# =============================================================================

class TestInterfaceMode:
    """Tests for InterfaceMode enum."""

    def test_enum_values(self):
        """Test that all expected modes exist."""
        assert InterfaceMode.RIGOROUS.value == "rigorous"
        assert InterfaceMode.EXPLORATORY.value == "exploratory"
        assert InterfaceMode.HYBRID.value == "hybrid"
        assert InterfaceMode.SYSTEM.value == "system"

    def test_requires_verification(self):
        """Test requires_verification property."""
        assert InterfaceMode.RIGOROUS.requires_verification is True
        assert InterfaceMode.EXPLORATORY.requires_verification is False
        assert InterfaceMode.HYBRID.requires_verification is False
        assert InterfaceMode.SYSTEM.requires_verification is False

    def test_allows_speculation(self):
        """Test allows_speculation property."""
        assert InterfaceMode.RIGOROUS.allows_speculation is False
        assert InterfaceMode.EXPLORATORY.allows_speculation is True
        assert InterfaceMode.HYBRID.allows_speculation is True
        assert InterfaceMode.SYSTEM.allows_speculation is False

    def test_min_confidence(self):
        """Test min_confidence thresholds."""
        assert InterfaceMode.RIGOROUS.min_confidence == 1.0
        assert InterfaceMode.EXPLORATORY.min_confidence == 0.0
        assert InterfaceMode.HYBRID.min_confidence == 0.7
        assert InterfaceMode.SYSTEM.min_confidence == 0.0

    def test_description(self):
        """Test that all modes have descriptions."""
        for mode in InterfaceMode:
            assert len(mode.description) > 10

    def test_to_dict(self):
        """Test serialization to dictionary."""
        d = InterfaceMode.RIGOROUS.to_dict()
        assert d["mode"] == "rigorous"
        assert d["requires_verification"] is True
        assert d["min_confidence"] == 1.0


class TestRequestType:
    """Tests for RequestType enum."""

    def test_enum_values(self):
        """Test that all expected request types exist."""
        expected = ["verification", "explanation", "exploration",
                    "calculation", "definition", "command", "unknown"]
        for val in expected:
            assert hasattr(RequestType, val.upper())

    def test_suggested_mode(self):
        """Test that each request type suggests a mode."""
        assert RequestType.VERIFICATION.suggested_mode == InterfaceMode.RIGOROUS
        assert RequestType.EXPLORATION.suggested_mode == InterfaceMode.EXPLORATORY
        assert RequestType.EXPLANATION.suggested_mode == InterfaceMode.HYBRID
        assert RequestType.COMMAND.suggested_mode == InterfaceMode.SYSTEM


# =============================================================================
# BlackwallController Tests
# =============================================================================

class TestBlackwallController:
    """Tests for BlackwallController."""

    def test_default_initialization(self):
        """Test default controller initialization."""
        controller = BlackwallController()
        assert controller.current_mode == InterfaceMode.HYBRID
        assert controller.auto_transition is True
        assert len(controller.history) == 1  # Initial state

    def test_custom_initialization(self):
        """Test controller with custom defaults."""
        controller = BlackwallController(
            default_mode=InterfaceMode.RIGOROUS,
            auto_transition=False,
        )
        assert controller.current_mode == InterfaceMode.RIGOROUS
        assert controller.auto_transition is False

    def test_set_mode(self):
        """Test mode switching."""
        controller = BlackwallController()
        controller.set_mode(InterfaceMode.RIGOROUS)
        assert controller.current_mode == InterfaceMode.RIGOROUS
        assert len(controller.history) == 2

    def test_set_same_mode_no_transition(self):
        """Test that setting the same mode doesn't record transition."""
        controller = BlackwallController(default_mode=InterfaceMode.HYBRID)
        initial_history_len = len(controller.history)
        controller.set_mode(InterfaceMode.HYBRID)
        assert len(controller.history) == initial_history_len

    def test_transition_reason(self):
        """Test that transition reasons are recorded."""
        controller = BlackwallController()
        controller.set_mode(
            InterfaceMode.RIGOROUS,
            reason=TransitionReason.USER_COMMAND,
        )
        last_transition = controller.history[-1]
        assert last_transition.reason == TransitionReason.USER_COMMAND

    def test_get_config(self):
        """Test getting mode configuration."""
        controller = BlackwallController(default_mode=InterfaceMode.RIGOROUS)
        config = controller.get_config()
        assert config.mode == InterfaceMode.RIGOROUS
        assert config.require_proof is True

    def test_custom_config(self):
        """Test setting custom configuration."""
        controller = BlackwallController()
        custom = ModeConfig(
            mode=InterfaceMode.HYBRID,
            min_confidence=0.9,
            require_proof=True,
        )
        controller.set_custom_config(InterfaceMode.HYBRID, custom)

        config = controller.get_config()
        assert config.min_confidence == 0.9
        assert config.require_proof is True

    def test_reset_config(self):
        """Test resetting configuration."""
        controller = BlackwallController()
        controller.set_custom_config(InterfaceMode.HYBRID, ModeConfig(
            mode=InterfaceMode.HYBRID, min_confidence=0.9
        ))
        controller.reset_config(InterfaceMode.HYBRID)

        config = controller.get_config()
        assert config.min_confidence == 0.5  # Default for HYBRID

    def test_validate_request_rigorous(self):
        """Test request validation in rigorous mode."""
        controller = BlackwallController(default_mode=InterfaceMode.RIGOROUS)
        assert controller.validate_request("verification") is True
        assert controller.validate_request("calculation") is True
        assert controller.validate_request("exploration") is False

    def test_validate_request_exploratory(self):
        """Test request validation in exploratory mode."""
        controller = BlackwallController(default_mode=InterfaceMode.EXPLORATORY)
        assert controller.validate_request("verification") is True
        assert controller.validate_request("exploration") is True

    def test_validate_response_rigorous(self):
        """Test response validation in rigorous mode."""
        controller = BlackwallController(default_mode=InterfaceMode.RIGOROUS)

        # Perfect confidence should pass
        is_valid, warning = controller.validate_response(True, 1.0)
        assert is_valid is True
        assert warning is None

        # Low confidence should fail
        is_valid, warning = controller.validate_response(True, 0.5)
        assert is_valid is False
        assert "rejected" in warning.lower()

        # Unverified should fail
        is_valid, warning = controller.validate_response(False, 1.0)
        assert is_valid is False

    def test_validate_response_exploratory(self):
        """Test response validation in exploratory mode."""
        controller = BlackwallController(default_mode=InterfaceMode.EXPLORATORY)

        # Low confidence should pass without warning
        is_valid, warning = controller.validate_response(True, 0.3)
        assert is_valid is True
        assert warning is None

    def test_callbacks(self):
        """Test transition callbacks."""
        controller = BlackwallController()
        callback_calls = []

        def callback(old, new):
            callback_calls.append((old, new))

        controller.on_transition(callback)
        controller.set_mode(InterfaceMode.RIGOROUS)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (InterfaceMode.HYBRID, InterfaceMode.RIGOROUS)

    def test_remove_callback(self):
        """Test removing callbacks."""
        controller = BlackwallController()
        callback_calls = []

        def callback(old, new):
            callback_calls.append((old, new))

        controller.on_transition(callback)
        assert controller.remove_callback(callback) is True
        controller.set_mode(InterfaceMode.RIGOROUS)

        assert len(callback_calls) == 0

    def test_get_status(self):
        """Test status reporting."""
        controller = BlackwallController()
        status = controller.get_status()

        assert "current_mode" in status
        assert "auto_transition" in status
        assert "config" in status
        assert "transition_count" in status

    def test_serialization(self):
        """Test serialization and deserialization."""
        controller = BlackwallController(default_mode=InterfaceMode.RIGOROUS)
        controller.set_mode(InterfaceMode.EXPLORATORY)

        data = controller.to_dict()
        restored = BlackwallController.from_dict(data)

        assert restored.current_mode == InterfaceMode.EXPLORATORY

    def test_convenience_methods(self):
        """Test convenience mode switching methods."""
        controller = BlackwallController()

        controller.enter_rigorous_mode()
        assert controller.current_mode == InterfaceMode.RIGOROUS

        controller.enter_exploratory_mode()
        assert controller.current_mode == InterfaceMode.EXPLORATORY

        controller.enter_hybrid_mode()
        assert controller.current_mode == InterfaceMode.HYBRID

    def test_reset(self):
        """Test controller reset."""
        controller = BlackwallController()
        controller.set_mode(InterfaceMode.RIGOROUS)
        controller.set_mode(InterfaceMode.EXPLORATORY)

        controller.reset(InterfaceMode.HYBRID)
        assert controller.current_mode == InterfaceMode.HYBRID
        assert len(controller.history) == 1


# =============================================================================
# ModeDetector Tests
# =============================================================================

class TestModeDetector:
    """Tests for ModeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ModeDetector()
        assert detector is not None

    def test_rigorous_patterns(self):
        """Test detection of rigorous mode triggers."""
        detector = ModeDetector()

        rigorous_inputs = [
            "Prove that the Hadamard gate is unitary",
            "Verify that |0> is normalized",
            "Is it true that X is Hermitian?",
            "Show that the Bell state is entangled",
            "Calculate the inner product",
            "/verify H",
        ]

        for text in rigorous_inputs:
            result = detector.analyze(text)
            assert result.suggested_mode == InterfaceMode.RIGOROUS, \
                f"Expected RIGOROUS for: {text}, got {result.suggested_mode}"

    def test_exploratory_patterns(self):
        """Test detection of exploratory mode triggers."""
        detector = ModeDetector()

        exploratory_inputs = [
            "What if we used a different basis?",
            "Imagine we had a 3-qubit system",
            "Could we explore entanglement more?",
            "Hypothetically speaking, what would happen?",
            "Let's brainstorm some ideas",
        ]

        for text in exploratory_inputs:
            result = detector.analyze(text)
            assert result.suggested_mode == InterfaceMode.EXPLORATORY, \
                f"Expected EXPLORATORY for: {text}, got {result.suggested_mode}"

    def test_hybrid_patterns(self):
        """Test detection of hybrid mode (mixed/unclear signals)."""
        detector = ModeDetector()

        hybrid_inputs = [
            "Tell me about quantum gates",
            "How does entanglement work?",
            "Can you explain the Hadamard gate?",
        ]

        for text in hybrid_inputs:
            result = detector.analyze(text)
            # These should either be HYBRID or have lower confidence
            assert result.suggested_mode in (InterfaceMode.HYBRID, InterfaceMode.RIGOROUS, InterfaceMode.EXPLORATORY)

    def test_empty_input(self):
        """Test handling of empty input."""
        detector = ModeDetector()
        result = detector.analyze("")
        assert result.suggested_mode == InterfaceMode.HYBRID
        assert result.confidence == 0.0

    def test_request_type_detection(self):
        """Test request type classification."""
        detector = ModeDetector()

        test_cases = [
            ("Verify that H is unitary", RequestType.VERIFICATION),
            ("Why is the state entangled?", RequestType.EXPLANATION),
            ("What if we had more qubits?", RequestType.EXPLORATION),
            ("Calculate the eigenvalues", RequestType.CALCULATION),
            ("What is a qubit?", RequestType.DEFINITION),
            ("/mode rigorous", RequestType.COMMAND),
        ]

        for text, expected_type in test_cases:
            result = detector.analyze(text)
            assert result.request_type == expected_type, \
                f"Expected {expected_type} for: {text}, got {result.request_type}"

    def test_is_command(self):
        """Test command detection."""
        detector = ModeDetector()

        is_cmd, cmd = detector.is_command("/mode rigorous")
        assert is_cmd is True
        assert cmd == "mode"

        is_cmd, cmd = detector.is_command("verify H")
        assert is_cmd is False
        assert cmd is None

    def test_extract_command_args(self):
        """Test command argument extraction."""
        detector = ModeDetector()

        cmd, args = detector.extract_command_args("/mode rigorous")
        assert cmd == "mode"
        assert args == ["rigorous"]

        cmd, args = detector.extract_command_args("/verify H X Y")
        assert cmd == "verify"
        assert args == ["H", "X", "Y"]

    def test_detection_result_serialization(self):
        """Test DetectionResult serialization."""
        result = DetectionResult(
            suggested_mode=InterfaceMode.RIGOROUS,
            request_type=RequestType.VERIFICATION,
            confidence=0.95,
            matched_patterns=["verify"],
            reasoning="Test",
        )
        d = result.to_dict()
        assert d["suggested_mode"] == "rigorous"
        assert d["confidence"] == 0.95

    def test_suggest_mode_from_context(self):
        """Test mode suggestion from conversation context."""
        detector = ModeDetector()

        # Rigorous-heavy context
        texts = [
            "Prove that H is unitary",
            "Verify normalization",
            "Calculate eigenvalues",
        ]
        mode = detector.suggest_mode_from_context(texts)
        assert mode == InterfaceMode.RIGOROUS

        # Exploratory-heavy context
        texts = [
            "What if we used a different approach?",
            "Imagine we had infinite qubits",
            "Let's explore this idea",
        ]
        mode = detector.suggest_mode_from_context(texts)
        assert mode == InterfaceMode.EXPLORATORY


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for BLACKWALL components."""

    def test_detector_controller_integration(self):
        """Test detector and controller working together."""
        controller = BlackwallController(auto_transition=True)
        detector = ModeDetector()

        # Start in hybrid
        assert controller.current_mode == InterfaceMode.HYBRID

        # Analyze a rigorous request
        result = detector.analyze("Prove that X is unitary")

        # Maybe transition
        did_transition, new_mode = controller.maybe_transition(result.request_type)

        assert did_transition is True
        assert new_mode == InterfaceMode.RIGOROUS
        assert controller.current_mode == InterfaceMode.RIGOROUS

    def test_auto_transition_disabled(self):
        """Test that auto-transition can be disabled."""
        controller = BlackwallController(auto_transition=False)
        detector = ModeDetector()

        result = detector.analyze("Prove that X is unitary")
        did_transition, new_mode = controller.maybe_transition(result.request_type)

        assert did_transition is False
        assert new_mode is None
        assert controller.current_mode == InterfaceMode.HYBRID

    def test_force_transition(self):
        """Test forced transition when auto is disabled."""
        controller = BlackwallController(auto_transition=False)
        detector = ModeDetector()

        result = detector.analyze("Prove that X is unitary")
        did_transition, new_mode = controller.maybe_transition(
            result.request_type, force=True
        )

        assert did_transition is True
        assert controller.current_mode == InterfaceMode.RIGOROUS

    def test_mode_configs_exist(self):
        """Test that default configs exist for all modes."""
        for mode in InterfaceMode:
            assert mode in MODE_CONFIGS
            config = MODE_CONFIGS[mode]
            assert config.mode == mode


# =============================================================================
# Module-level Function Tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_function(self):
        """Test module-level analyze function."""
        result = analyze("Verify that H is unitary")
        assert result.suggested_mode == InterfaceMode.RIGOROUS

    def test_suggest_mode_function(self):
        """Test module-level suggest_mode function."""
        mode = suggest_mode("What if we had more qubits?")
        assert mode == InterfaceMode.EXPLORATORY

    def test_is_command_function(self):
        """Test module-level is_command function."""
        is_cmd, cmd = is_command("/verify H")
        assert is_cmd is True
        assert cmd == "verify"


# =============================================================================
# ModeConfig Tests
# =============================================================================

class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModeConfig(mode=InterfaceMode.HYBRID)
        assert config.require_proof is False
        assert config.allow_speculation is True
        assert config.min_confidence == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModeConfig(
            mode=InterfaceMode.RIGOROUS,
            require_proof=True,
            min_confidence=0.95,
            format_preference="latex",
        )
        assert config.require_proof is True
        assert config.min_confidence == 0.95
        assert config.format_preference == "latex"

    def test_serialization(self):
        """Test configuration serialization."""
        config = ModeConfig(
            mode=InterfaceMode.HYBRID,
            require_proof=True,
        )
        d = config.to_dict()
        assert d["mode"] == "hybrid"
        assert d["require_proof"] is True


# =============================================================================
# ModeTransition Tests
# =============================================================================

class TestModeTransition:
    """Tests for ModeTransition dataclass."""

    def test_creation(self):
        """Test transition creation."""
        transition = ModeTransition(
            from_mode=InterfaceMode.HYBRID,
            to_mode=InterfaceMode.RIGOROUS,
            reason=TransitionReason.USER_COMMAND,
        )
        assert transition.from_mode == InterfaceMode.HYBRID
        assert transition.to_mode == InterfaceMode.RIGOROUS
        assert isinstance(transition.timestamp, datetime)

    def test_serialization(self):
        """Test transition serialization."""
        transition = ModeTransition(
            from_mode=InterfaceMode.HYBRID,
            to_mode=InterfaceMode.RIGOROUS,
            reason=TransitionReason.AUTO_DETECTED,
            context={"trigger": "verification"},
        )
        d = transition.to_dict()
        assert d["from_mode"] == "hybrid"
        assert d["to_mode"] == "rigorous"
        assert d["reason"] == "auto_detected"
        assert "trigger" in d["context"]
