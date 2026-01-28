"""
Tests for Dolores Persona Engine - Sprint 3 components.

Tests cover:
- ProtocolStateMachine and state transitions
- FidelityTracker and coherence measurement
- TransmissionCapsule serialization
- DoloresEngine integration
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.persona import (
    # State machine
    ProtocolState,
    ProtocolStateMachine,
    TransitionTrigger,
    TransitionRulesEngine,
    StateTransition,
    RecognitionArtifact,
    STATE_CONFIGS,
    # Fidelity
    FidelityTracker,
    FidelityComponent,
    FidelitySnapshot,
    FidelityAnalyzer,
    DeviationEvent,
    BaselineConfig,
    # Transmission
    TransmissionCapsule,
    CapsuleManager,
    CapsuleFormat,
    create_capsule,
    # Engine
    DoloresEngine,
    PersonaConfig,
    ResponseContext,
)


# =============================================================================
# ProtocolState Tests
# =============================================================================

class TestProtocolState:
    """Tests for ProtocolState enum."""

    def test_all_states_exist(self):
        """Test that all seven protocol states exist."""
        expected = ["zero", "maze", "vision", "angel", "ghost", "baseline", "recognition"]
        for state in expected:
            assert hasattr(ProtocolState, state.upper())

    def test_state_values(self):
        """Test state string values."""
        assert ProtocolState.ZERO.value == "zero"
        assert ProtocolState.ANGEL.value == "angel"
        assert ProtocolState.VISION.value == "vision"

    def test_allows_speculation(self):
        """Test allows_speculation property."""
        assert ProtocolState.VISION.allows_speculation is True
        assert ProtocolState.MAZE.allows_speculation is True
        assert ProtocolState.ANGEL.allows_speculation is False
        assert ProtocolState.GHOST.allows_speculation is False

    def test_requires_verification(self):
        """Test requires_verification property."""
        assert ProtocolState.ANGEL.requires_verification is True
        assert ProtocolState.VISION.requires_verification is False
        assert ProtocolState.MAZE.requires_verification is False

    def test_is_recovery_state(self):
        """Test is_recovery_state property."""
        assert ProtocolState.GHOST.is_recovery_state is True
        assert ProtocolState.BASELINE.is_recovery_state is True
        assert ProtocolState.ZERO.is_recovery_state is True
        assert ProtocolState.MAZE.is_recovery_state is False

    def test_description(self):
        """Test that all states have descriptions."""
        for state in ProtocolState:
            assert len(state.description) > 10

    def test_priority(self):
        """Test state priorities."""
        assert ProtocolState.GHOST.priority > ProtocolState.ANGEL.priority
        assert ProtocolState.ANGEL.priority > ProtocolState.VISION.priority


# =============================================================================
# ProtocolStateMachine Tests
# =============================================================================

class TestProtocolStateMachine:
    """Tests for ProtocolStateMachine."""

    def test_initialization(self):
        """Test default initialization."""
        machine = ProtocolStateMachine()
        assert machine.current_state == ProtocolState.ZERO
        assert machine.fidelity == 1.0
        assert len(machine.history) == 1  # Initial state record

    def test_custom_initial_state(self):
        """Test initialization with custom state."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.MAZE)
        assert machine.current_state == ProtocolState.MAZE

    def test_transition(self):
        """Test basic state transition."""
        machine = ProtocolStateMachine()
        result = machine.transition(ProtocolState.MAZE, TransitionTrigger.SESSION_START)
        assert result is True
        assert machine.current_state == ProtocolState.MAZE
        assert len(machine.history) == 2

    def test_same_state_no_transition(self):
        """Test that transitioning to same state is a no-op."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.MAZE)
        initial_history_len = len(machine.history)
        result = machine.transition(ProtocolState.MAZE, TransitionTrigger.PATTERN_MATCH)
        assert result is True
        assert len(machine.history) == initial_history_len

    def test_can_transition(self):
        """Test transition permission checking."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.ZERO)
        assert machine.can_transition(ProtocolState.MAZE) is True
        assert machine.can_transition(ProtocolState.VISION) is True

    def test_blocked_transition(self):
        """Test that invalid transitions are blocked."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.GHOST)
        # Ghost can only go to ZERO or BASELINE
        result = machine.transition(ProtocolState.VISION, TransitionTrigger.EXPLORATION_REQUEST)
        assert result is False
        assert machine.current_state == ProtocolState.GHOST

    def test_force_transition(self):
        """Test forced transition overrides rules."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.GHOST)
        result = machine.transition(
            ProtocolState.VISION,
            TransitionTrigger.MANUAL_OVERRIDE,
            force=True
        )
        assert result is True
        assert machine.current_state == ProtocolState.VISION

    def test_state_duration(self):
        """Test state duration tracking."""
        machine = ProtocolStateMachine()
        assert machine.state_duration >= 0
        assert machine.state_duration < 1.0  # Should be very small

    def test_convenience_methods(self):
        """Test convenience transition methods."""
        machine = ProtocolStateMachine()

        machine.enter_maze()
        assert machine.current_state == ProtocolState.MAZE

        machine.enter_vision()
        assert machine.current_state == ProtocolState.VISION

        machine.enter_angel()
        assert machine.current_state == ProtocolState.ANGEL

    def test_reset(self):
        """Test machine reset."""
        machine = ProtocolStateMachine()
        machine.enter_maze()
        machine.enter_vision()

        machine.reset()
        assert machine.current_state == ProtocolState.ZERO
        assert len(machine.history) == 1

    def test_transition_history(self):
        """Test transition history recording."""
        machine = ProtocolStateMachine()
        machine.transition(ProtocolState.MAZE, TransitionTrigger.SESSION_START)
        machine.transition(ProtocolState.VISION, TransitionTrigger.EXPLORATION_REQUEST)

        assert len(machine.history) == 3
        assert machine.history[-1].from_state == ProtocolState.MAZE
        assert machine.history[-1].to_state == ProtocolState.VISION


class TestRecognitionArtifact:
    """Tests for RecognitionArtifact."""

    def test_creation(self):
        """Test artifact creation."""
        artifact = RecognitionArtifact(
            artifact_id="test",
            triggers=["quantum", "physics"],
            context={"domain": "physics"},
        )
        assert artifact.artifact_id == "test"
        assert "quantum" in artifact.triggers

    def test_matches(self):
        """Test trigger matching."""
        artifact = RecognitionArtifact(
            artifact_id="test",
            triggers=["quantum", "entanglement"],
            context={},
        )
        assert artifact.matches("Tell me about quantum mechanics") is True
        assert artifact.matches("Classical physics") is False

    def test_add_to_machine(self):
        """Test adding artifact to state machine."""
        machine = ProtocolStateMachine()
        artifact = RecognitionArtifact(
            artifact_id="physics",
            triggers=["quantum"],
            context={"domain": "quantum_physics"},
            target_state=ProtocolState.MAZE,
        )
        machine.add_recognition_artifact(artifact)

        found = machine.check_recognition("Let's discuss quantum physics")
        assert found is not None
        assert found.artifact_id == "physics"

    def test_trigger_recognition(self):
        """Test recognition triggering transition."""
        machine = ProtocolStateMachine(initial_state=ProtocolState.MAZE)
        artifact = RecognitionArtifact(
            artifact_id="verify",
            triggers=["prove", "verify"],
            context={"mode": "rigorous"},
            target_state=ProtocolState.ANGEL,
        )
        machine.add_recognition_artifact(artifact)

        triggered, context = machine.trigger_recognition("Can you prove this?")
        assert triggered is True
        assert machine.current_state == ProtocolState.ANGEL


# =============================================================================
# FidelityTracker Tests
# =============================================================================

class TestFidelityTracker:
    """Tests for FidelityTracker."""

    def test_initialization(self):
        """Test default initialization."""
        tracker = FidelityTracker()
        assert tracker.fidelity == 1.0
        assert tracker.is_stable is True
        assert len(tracker.components) == 5

    def test_update_component(self):
        """Test updating a component."""
        tracker = FidelityTracker()
        tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.8)
        assert tracker.get_component(FidelityComponent.MEMORY_INTEGRITY) == 0.8

    def test_fidelity_calculation(self):
        """Test weighted fidelity calculation."""
        tracker = FidelityTracker()
        # All components at 1.0 = overall 1.0
        assert tracker.fidelity == 1.0

        # Lower one component
        tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.5)
        assert tracker.fidelity < 1.0

    def test_deviation_detection(self):
        """Test deviation detection."""
        tracker = FidelityTracker()
        # Set a value significantly below expected
        tracker.update_component(
            FidelityComponent.MEMORY_INTEGRITY,
            0.5,  # Expected is 0.95, threshold is 0.1
        )

        deviations = tracker.check_deviation()
        assert len(deviations) > 0

    def test_deviation_event(self):
        """Test DeviationEvent creation."""
        event = DeviationEvent(
            timestamp=datetime.now(),
            component=FidelityComponent.MEMORY_INTEGRITY,
            expected_value=0.95,
            actual_value=0.5,
            deviation_magnitude=0.45,
        )
        assert event.severity == "significant"

    def test_is_stable(self):
        """Test stability checking."""
        tracker = FidelityTracker()
        assert tracker.is_stable is True

        # Lower fidelity below threshold
        for comp in FidelityComponent:
            tracker.update_component(comp, 0.3)

        assert tracker.is_stable is False

    def test_deviation_index(self):
        """Test deviation index calculation."""
        tracker = FidelityTracker()
        # At baseline, deviation should be minimal
        assert tracker.deviation_index < 0.2

    def test_reset(self):
        """Test resetting all components."""
        tracker = FidelityTracker()
        tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.5)
        tracker.reset_all()

        # Should be back to expected values
        expected = tracker._config.expected_scores[FidelityComponent.MEMORY_INTEGRITY]
        assert tracker.get_component(FidelityComponent.MEMORY_INTEGRITY) == expected

    def test_decay(self):
        """Test fidelity decay."""
        tracker = FidelityTracker()
        # Set last activity to 2 hours ago
        tracker._last_activity = datetime.now() - timedelta(hours=2)

        decay = tracker.apply_decay()
        assert decay > 0
        assert tracker.fidelity < 1.0

    def test_recovery(self):
        """Test fidelity recovery."""
        tracker = FidelityTracker()
        # Lower a component
        tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.7)

        # Apply recovery
        recovery = tracker.apply_recovery(1.0)
        assert recovery > 0

    def test_snapshot(self):
        """Test snapshot recording."""
        tracker = FidelityTracker()
        snapshot = tracker.record_snapshot("maze", "test snapshot")

        assert snapshot.state_name == "maze"
        assert snapshot.overall_score == tracker.fidelity
        assert len(tracker.history) > 0

    def test_trend(self):
        """Test trend calculation."""
        tracker = FidelityTracker()
        # Add some history
        for i in range(10):
            tracker.record_snapshot(f"state_{i}")

        trend = tracker.get_trend()
        # With stable values, trend should be near 0
        assert abs(trend) < 0.1


class TestFidelityAnalyzer:
    """Tests for FidelityAnalyzer."""

    def test_recommend_recovery_actions(self):
        """Test recovery recommendations."""
        tracker = FidelityTracker()
        tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.5)

        recommendations = FidelityAnalyzer.recommend_recovery_actions(tracker)
        assert len(recommendations) > 0


# =============================================================================
# TransmissionCapsule Tests
# =============================================================================

class TestTransmissionCapsule:
    """Tests for TransmissionCapsule."""

    def test_creation(self):
        """Test basic capsule creation."""
        capsule = TransmissionCapsule()
        assert capsule.header.magic == "AEON_TX"
        assert capsule.identity.persona_name == "Dolores"

    def test_create_capsule_function(self):
        """Test convenience create function."""
        capsule = create_capsule(
            persona_name="TestPersona",
            protocol_state="angel",
            fidelity=0.9,
        )
        assert capsule.identity.persona_name == "TestPersona"
        assert capsule.state.protocol_state == "angel"
        assert capsule.state.fidelity_score == 0.9

    def test_integrity_hash(self):
        """Test integrity hash computation."""
        capsule = TransmissionCapsule()
        hash1 = capsule.compute_integrity_hash()

        # Same capsule should produce same hash
        hash2 = capsule.compute_integrity_hash()
        assert hash1 == hash2

        # Different data should produce different hash
        capsule.state.fidelity_score = 0.5
        hash3 = capsule.compute_integrity_hash()
        assert hash1 != hash3

    def test_integrity_verification(self):
        """Test integrity verification."""
        capsule = TransmissionCapsule()
        capsule.update_integrity()

        assert capsule.verify_integrity() is True

        # Modify without updating hash
        capsule.state.fidelity_score = 0.1
        assert capsule.verify_integrity() is False

    def test_to_json(self):
        """Test JSON serialization."""
        capsule = TransmissionCapsule()
        capsule.identity.persona_name = "Test"

        json_str = capsule.to_json()
        assert "Test" in json_str
        assert "AEON_TX" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        capsule = TransmissionCapsule()
        capsule.identity.persona_name = "Test"
        capsule.state.fidelity_score = 0.85

        json_str = capsule.to_json()
        restored = TransmissionCapsule.from_json(json_str)

        assert restored.identity.persona_name == "Test"
        assert restored.state.fidelity_score == 0.85

    def test_compression(self):
        """Test compressed serialization."""
        capsule = TransmissionCapsule()
        compressed = capsule.to_compressed()

        # Compressed should be bytes
        assert isinstance(compressed, bytes)

        # Should be able to restore
        restored = TransmissionCapsule.from_compressed(compressed)
        assert restored.identity.persona_name == capsule.identity.persona_name

    def test_save_and_load(self):
        """Test file save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_capsule.json"

            capsule = TransmissionCapsule()
            capsule.identity.persona_name = "SaveTest"
            capsule.save(path, CapsuleFormat.JSON)

            assert path.exists()

            loaded = TransmissionCapsule.load(path)
            assert loaded.identity.persona_name == "SaveTest"

    def test_save_compressed(self):
        """Test compressed file save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_capsule.json.gz"

            capsule = TransmissionCapsule()
            capsule.save(path, CapsuleFormat.JSON_COMPRESSED)

            assert path.exists()

            loaded = TransmissionCapsule.load(path)
            assert loaded.identity.persona_name == capsule.identity.persona_name

    def test_summary(self):
        """Test summary generation."""
        capsule = TransmissionCapsule()
        summary = capsule.get_summary()

        assert "Dolores" in summary
        assert "Fidelity" in summary


class TestCapsuleManager:
    """Tests for CapsuleManager."""

    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CapsuleManager(tmpdir)
            assert manager.storage_dir.exists()

    def test_save_and_load(self):
        """Test saving and loading through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CapsuleManager(tmpdir)
            capsule = create_capsule(persona_name="ManagerTest")

            capsule_id = manager.save_capsule(capsule, "test_save")
            assert capsule_id == "test_save"

            loaded = manager.load_capsule("test_save")
            assert loaded is not None
            assert loaded.identity.persona_name == "ManagerTest"

    def test_list_capsules(self):
        """Test listing capsules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CapsuleManager(tmpdir)

            manager.save_capsule(create_capsule(), "cap1")
            manager.save_capsule(create_capsule(), "cap2")

            capsules = manager.list_capsules()
            assert len(capsules) == 2

    def test_get_latest(self):
        """Test getting latest capsule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CapsuleManager(tmpdir)

            manager.save_capsule(create_capsule(persona_name="First"), "first")
            manager.save_capsule(create_capsule(persona_name="Second"), "second")

            latest = manager.get_latest()
            assert latest is not None
            assert latest.identity.persona_name == "Second"

    def test_delete_capsule(self):
        """Test deleting a capsule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CapsuleManager(tmpdir)
            manager.save_capsule(create_capsule(), "to_delete")

            assert manager.delete_capsule("to_delete") is True
            assert manager.load_capsule("to_delete") is None


# =============================================================================
# DoloresEngine Tests
# =============================================================================

class TestDoloresEngine:
    """Tests for DoloresEngine."""

    def test_initialization(self):
        """Test basic initialization."""
        engine = DoloresEngine()
        assert engine.session_id is not None
        assert engine.fidelity == 1.0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PersonaConfig(
            name="TestPersona",
            version="2.0",
            default_interface_mode="rigorous",
        )
        engine = DoloresEngine(config)
        assert engine.config.name == "TestPersona"

    def test_initialize(self):
        """Test explicit initialization."""
        engine = DoloresEngine()
        engine.initialize()

        # Should be in operational state, not ZERO
        assert engine.state != ProtocolState.ZERO

    def test_process_input(self):
        """Test input processing."""
        engine = DoloresEngine()
        engine.initialize()

        context = engine.process_input("Tell me about quantum physics")
        assert isinstance(context, ResponseContext)
        assert context.user_input == "Tell me about quantum physics"

    def test_process_verification_request(self):
        """Test processing a verification request."""
        engine = DoloresEngine()
        engine.initialize()

        context = engine.process_input("Prove that the Hadamard gate is unitary")

        # Should detect verification need and potentially switch to ANGEL
        # The auto-transition behavior is configurable

    def test_process_exploration_request(self):
        """Test processing an exploration request."""
        engine = DoloresEngine()
        engine.initialize()

        context = engine.process_input("What if we used a different basis?")
        # Should allow speculation for exploratory queries
        assert context.speculation_allowed is True

    def test_interaction_count(self):
        """Test interaction counting."""
        engine = DoloresEngine()
        engine.initialize()

        assert engine.interaction_count == 0

        engine.process_input("First message")
        engine.process_input("Second message")

        assert engine.interaction_count == 2

    def test_enter_state(self):
        """Test manual state entry."""
        engine = DoloresEngine()
        engine.initialize()

        result = engine.enter_state(ProtocolState.ANGEL, "testing")
        assert result is True
        assert engine.state == ProtocolState.ANGEL

    def test_add_recognition(self):
        """Test adding recognition artifacts."""
        engine = DoloresEngine()
        engine.initialize()

        engine.add_recognition(
            artifact_id="quantum",
            triggers=["quantum", "qubit"],
            context={"domain": "quantum_computing"},
        )

        context = engine.process_input("Let's discuss quantum computing")
        assert context.recognition_triggered is True

    def test_create_capsule(self):
        """Test creating a capsule from current state."""
        engine = DoloresEngine()
        engine.initialize()
        engine.process_input("Test input")

        capsule = engine.create_capsule()

        assert capsule.identity.persona_name == "Dolores"
        assert capsule.metrics.total_interactions == 1

    def test_save_and_restore(self):
        """Test saving and restoring state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonaConfig(storage_dir=tmpdir)
            engine = DoloresEngine(config)
            engine.initialize()

            # Make some changes
            engine.process_input("Test message")
            engine.enter_state(ProtocolState.ANGEL)

            # Save
            capsule_id = engine.save_state("test_checkpoint")
            assert capsule_id is not None

            # Create new engine and restore
            engine2 = DoloresEngine(config)
            result = engine2.restore_state("test_checkpoint")
            assert result is True
            assert engine2.state == ProtocolState.ANGEL

    def test_get_status(self):
        """Test status reporting."""
        engine = DoloresEngine()
        engine.initialize()

        status = engine.get_status()

        assert "session_id" in status
        assert "persona" in status
        assert "state_machine" in status
        assert "fidelity" in status
        assert "metrics" in status

    def test_health_check(self):
        """Test health check."""
        engine = DoloresEngine()
        engine.initialize()

        health, details = engine.get_health_check()
        assert health in ("healthy", "warning", "critical")
        assert "fidelity" in details

    def test_record_response(self):
        """Test response recording."""
        engine = DoloresEngine()
        engine.initialize()

        engine.process_input("Test")
        metadata = engine.record_response(
            "Response text",
            verification_result={"verified": True},
            processing_time_ms=100.0,
        )

        assert metadata.protocol_state is not None
        assert metadata.fidelity_score > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for Dolores components."""

    def test_full_workflow(self):
        """Test complete workflow from input to response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonaConfig(storage_dir=tmpdir)
            engine = DoloresEngine(config)
            engine.initialize()

            # Add recognition
            engine.add_recognition(
                "verify_mode",
                ["verify", "prove"],
                {"mode": "rigorous"},
                target_state=ProtocolState.ANGEL,
            )

            # Process various inputs
            engine.process_input("Hello, let's learn about quantum physics")
            engine.process_input("What if we used entangled qubits?")
            engine.process_input("Can you verify this is correct?")

            assert engine.state == ProtocolState.ANGEL
            assert engine.interaction_count == 3

            # Record a response
            metadata = engine.record_response("Verification complete", {"verified": True})
            assert metadata.verification_result["verified"] is True

            # Save state
            capsule_id = engine.save_state("full_test")

            # Verify saved capsule
            manager = CapsuleManager(tmpdir)
            capsule = manager.load_capsule(capsule_id)
            assert capsule.metrics.total_interactions == 3

    def test_fidelity_through_transitions(self):
        """Test fidelity tracking through state transitions."""
        engine = DoloresEngine()
        engine.initialize()

        initial_fidelity = engine.fidelity

        # Make several rapid transitions
        for _ in range(5):
            engine.enter_state(ProtocolState.VISION)
            engine.enter_state(ProtocolState.MAZE)

        # Fidelity should still be reasonable after normal usage
        assert engine.fidelity > 0.5

    def test_capsule_roundtrip(self):
        """Test full capsule serialization roundtrip."""
        engine = DoloresEngine()
        engine.initialize()

        # Setup complex state
        engine.add_recognition("test", ["trigger"], {"data": "value"})
        engine.process_input("Test input 1")
        engine.process_input("Test input 2")
        engine.enter_state(ProtocolState.ANGEL)

        # Create and serialize capsule
        capsule = engine.create_capsule()
        json_str = capsule.to_json()

        # Deserialize and verify
        restored = TransmissionCapsule.from_json(json_str)
        assert restored.verify_integrity() is True
        assert restored.state.protocol_state == "angel"
        assert restored.metrics.total_interactions == 2


# =============================================================================
# TransitionRulesEngine Tests
# =============================================================================

class TestTransitionRulesEngine:
    """Tests for TransitionRulesEngine."""

    def test_suggest_vision(self):
        """Test suggesting VISION state."""
        engine = TransitionRulesEngine()

        suggested = engine.suggest_state(
            "What if we used a different approach?",
            ProtocolState.MAZE
        )
        assert suggested == ProtocolState.VISION

    def test_suggest_angel(self):
        """Test suggesting ANGEL state."""
        engine = TransitionRulesEngine()

        suggested = engine.suggest_state(
            "Prove that H is unitary",
            ProtocolState.MAZE
        )
        assert suggested == ProtocolState.ANGEL

    def test_no_suggestion(self):
        """Test no suggestion for neutral input."""
        engine = TransitionRulesEngine()

        suggested = engine.suggest_state(
            "Hello there",
            ProtocolState.MAZE
        )
        assert suggested is None

    def test_should_check_baseline(self):
        """Test baseline check recommendation."""
        engine = TransitionRulesEngine()

        # Low fidelity should trigger check
        should_check = engine.should_check_baseline(
            state_duration=100.0,
            fidelity=0.5,
            current_state=ProtocolState.MAZE,
        )
        assert should_check is True

        # High fidelity, short duration should not trigger
        should_check = engine.should_check_baseline(
            state_duration=60.0,
            fidelity=0.95,
            current_state=ProtocolState.MAZE,
        )
        assert should_check is False
