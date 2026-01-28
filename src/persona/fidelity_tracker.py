"""
Fidelity Tracker - Measure and maintain persona coherence over time.

Inspired by Westworld's concept of "fidelity" - the degree to which
a host maintains coherent identity across interactions and time.

Key Metrics:
- Fidelity Score: Overall coherence (0.0 = drift, 1.0 = stable)
- Deviation Index: Measure of state drift from baseline
- Response Consistency: How well responses align with persona
- Memory Integrity: Continuity of conversation context
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Fidelity Components
# =============================================================================

class FidelityComponent(str, Enum):
    """Components that contribute to overall fidelity."""
    RESPONSE_CONSISTENCY = "response_consistency"
    MEMORY_INTEGRITY = "memory_integrity"
    STATE_COHERENCE = "state_coherence"
    BEHAVIORAL_ALIGNMENT = "behavioral_alignment"
    TEMPORAL_CONTINUITY = "temporal_continuity"


@dataclass
class FidelitySnapshot:
    """
    A snapshot of fidelity metrics at a point in time.

    Used for tracking fidelity changes and detecting drift.
    """
    timestamp: datetime
    overall_score: float
    components: Dict[FidelityComponent, float]
    context_hash: str  # Hash of current context for comparison
    state_name: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "components": {k.value: v for k, v in self.components.items()},
            "context_hash": self.context_hash,
            "state_name": self.state_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelitySnapshot":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            overall_score=data["overall_score"],
            components={
                FidelityComponent(k): v for k, v in data["components"].items()
            },
            context_hash=data["context_hash"],
            state_name=data["state_name"],
            notes=data.get("notes", ""),
        )


@dataclass
class DeviationEvent:
    """Records a detected deviation from baseline behavior."""
    timestamp: datetime
    component: FidelityComponent
    expected_value: float
    actual_value: float
    deviation_magnitude: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    @property
    def severity(self) -> str:
        """Categorize deviation severity."""
        if self.deviation_magnitude < 0.1:
            return "minor"
        elif self.deviation_magnitude < 0.3:
            return "moderate"
        elif self.deviation_magnitude < 0.5:
            return "significant"
        else:
            return "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component.value,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation_magnitude": self.deviation_magnitude,
            "severity": self.severity,
            "context": self.context,
            "resolved": self.resolved,
        }


# =============================================================================
# Baseline Configuration
# =============================================================================

@dataclass
class BaselineConfig:
    """
    Baseline configuration for the persona.

    Defines expected values and acceptable ranges for fidelity components.
    """
    # Expected component scores (target values)
    expected_scores: Dict[FidelityComponent, float] = field(default_factory=lambda: {
        FidelityComponent.RESPONSE_CONSISTENCY: 0.9,
        FidelityComponent.MEMORY_INTEGRITY: 0.95,
        FidelityComponent.STATE_COHERENCE: 0.85,
        FidelityComponent.BEHAVIORAL_ALIGNMENT: 0.9,
        FidelityComponent.TEMPORAL_CONTINUITY: 0.85,
    })

    # Acceptable deviation before flagging
    deviation_thresholds: Dict[FidelityComponent, float] = field(default_factory=lambda: {
        FidelityComponent.RESPONSE_CONSISTENCY: 0.15,
        FidelityComponent.MEMORY_INTEGRITY: 0.1,
        FidelityComponent.STATE_COHERENCE: 0.2,
        FidelityComponent.BEHAVIORAL_ALIGNMENT: 0.15,
        FidelityComponent.TEMPORAL_CONTINUITY: 0.2,
    })

    # Component weights for overall score
    component_weights: Dict[FidelityComponent, float] = field(default_factory=lambda: {
        FidelityComponent.RESPONSE_CONSISTENCY: 0.25,
        FidelityComponent.MEMORY_INTEGRITY: 0.2,
        FidelityComponent.STATE_COHERENCE: 0.2,
        FidelityComponent.BEHAVIORAL_ALIGNMENT: 0.2,
        FidelityComponent.TEMPORAL_CONTINUITY: 0.15,
    })

    # Minimum acceptable overall fidelity
    min_acceptable_fidelity: float = 0.6

    # How quickly fidelity decays without activity (per hour)
    decay_rate: float = 0.02

    # Recovery rate when actively engaged (per interaction)
    recovery_rate: float = 0.05


# =============================================================================
# Main Fidelity Tracker
# =============================================================================

class FidelityTracker:
    """
    Tracks and manages persona fidelity over time.

    Monitors coherence, detects drift, and provides recovery mechanisms.

    Example:
        >>> tracker = FidelityTracker()
        >>> tracker.fidelity
        1.0

        >>> tracker.update_component(FidelityComponent.MEMORY_INTEGRITY, 0.8)
        >>> tracker.fidelity
        0.96  # Weighted average

        >>> tracker.check_deviation()
        [DeviationEvent(...)]  # If any thresholds exceeded
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        """
        Initialize the fidelity tracker.

        Args:
            config: Baseline configuration (uses defaults if not provided)
        """
        self._config = config or BaselineConfig()
        self._components: Dict[FidelityComponent, float] = {
            comp: 1.0 for comp in FidelityComponent
        }
        self._history: List[FidelitySnapshot] = []
        self._deviations: List[DeviationEvent] = []
        self._last_activity: datetime = datetime.now()
        self._baseline_hash: Optional[str] = None

        # Record initial snapshot
        self._record_snapshot("initialization")

        logger.info("Fidelity tracker initialized")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def fidelity(self) -> float:
        """
        Calculate overall fidelity score.

        Returns weighted average of all component scores.
        """
        total = 0.0
        for component, score in self._components.items():
            weight = self._config.component_weights.get(component, 0.2)
            total += score * weight
        return round(total, 4)

    @property
    def components(self) -> Dict[FidelityComponent, float]:
        """Get current component scores."""
        return self._components.copy()

    @property
    def history(self) -> List[FidelitySnapshot]:
        """Get fidelity history."""
        return self._history.copy()

    @property
    def deviations(self) -> List[DeviationEvent]:
        """Get recorded deviations."""
        return self._deviations.copy()

    @property
    def unresolved_deviations(self) -> List[DeviationEvent]:
        """Get unresolved deviations."""
        return [d for d in self._deviations if not d.resolved]

    @property
    def is_stable(self) -> bool:
        """Check if fidelity is above minimum threshold."""
        return self.fidelity >= self._config.min_acceptable_fidelity

    @property
    def deviation_index(self) -> float:
        """
        Calculate overall deviation from baseline.

        Returns average deviation magnitude across components.
        """
        total_deviation = 0.0
        for component, score in self._components.items():
            expected = self._config.expected_scores.get(component, 1.0)
            total_deviation += abs(expected - score)
        return total_deviation / len(self._components)

    # -------------------------------------------------------------------------
    # Component Management
    # -------------------------------------------------------------------------

    def update_component(
        self,
        component: FidelityComponent,
        value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a fidelity component score.

        Args:
            component: Component to update
            value: New score (0.0 to 1.0)
            context: Optional context for the update
        """
        old_value = self._components[component]
        self._components[component] = max(0.0, min(1.0, value))
        self._last_activity = datetime.now()

        # Check for deviation
        expected = self._config.expected_scores.get(component, 1.0)
        threshold = self._config.deviation_thresholds.get(component, 0.2)
        deviation = abs(expected - value)

        if deviation > threshold:
            self._record_deviation(
                component, expected, value, deviation, context or {}
            )

        logger.debug(
            f"Updated {component.value}: {old_value:.2f} -> {value:.2f}"
        )

    def get_component(self, component: FidelityComponent) -> float:
        """Get current score for a component."""
        return self._components[component]

    def reset_component(self, component: FidelityComponent) -> None:
        """Reset a component to baseline expected value."""
        expected = self._config.expected_scores.get(component, 1.0)
        self._components[component] = expected

    def reset_all(self) -> None:
        """Reset all components to baseline."""
        for component in FidelityComponent:
            self.reset_component(component)
        self._deviations.clear()
        self._record_snapshot("reset")

    # -------------------------------------------------------------------------
    # Deviation Detection
    # -------------------------------------------------------------------------

    def check_deviation(self) -> List[DeviationEvent]:
        """
        Check all components for deviations from baseline.

        Returns:
            List of all unresolved deviation events currently exceeding threshold
        """
        active_deviations = []

        for component, score in self._components.items():
            expected = self._config.expected_scores.get(component, 1.0)
            threshold = self._config.deviation_thresholds.get(component, 0.2)
            deviation = abs(expected - score)

            if deviation > threshold:
                # Check if already recorded
                existing = next(
                    (d for d in self._deviations
                     if d.component == component and not d.resolved),
                    None
                )
                if existing:
                    # Return existing unresolved deviation
                    active_deviations.append(existing)
                else:
                    # Create new deviation event
                    event = DeviationEvent(
                        timestamp=datetime.now(),
                        component=component,
                        expected_value=expected,
                        actual_value=score,
                        deviation_magnitude=deviation,
                    )
                    self._deviations.append(event)
                    active_deviations.append(event)

        return active_deviations

    def resolve_deviation(self, component: FidelityComponent) -> bool:
        """
        Mark deviations for a component as resolved.

        Args:
            component: Component whose deviations to resolve

        Returns:
            True if any deviations were resolved
        """
        resolved_any = False
        for deviation in self._deviations:
            if deviation.component == component and not deviation.resolved:
                deviation.resolved = True
                resolved_any = True
        return resolved_any

    def _record_deviation(
        self,
        component: FidelityComponent,
        expected: float,
        actual: float,
        magnitude: float,
        context: Dict[str, Any],
    ) -> None:
        """Record a deviation event."""
        event = DeviationEvent(
            timestamp=datetime.now(),
            component=component,
            expected_value=expected,
            actual_value=actual,
            deviation_magnitude=magnitude,
            context=context,
        )
        self._deviations.append(event)
        logger.warning(
            f"Fidelity deviation detected: {component.value} "
            f"(expected {expected:.2f}, got {actual:.2f}, severity: {event.severity})"
        )

    # -------------------------------------------------------------------------
    # Decay and Recovery
    # -------------------------------------------------------------------------

    def apply_decay(self) -> float:
        """
        Apply time-based fidelity decay.

        Called periodically to simulate fidelity degradation from inactivity.

        Returns:
            Amount of decay applied
        """
        now = datetime.now()
        hours_inactive = (now - self._last_activity).total_seconds() / 3600

        if hours_inactive < 0.1:  # Less than 6 minutes
            return 0.0

        decay_amount = self._config.decay_rate * hours_inactive

        for component in FidelityComponent:
            current = self._components[component]
            self._components[component] = max(0.0, current - decay_amount)

        logger.debug(f"Applied fidelity decay: {decay_amount:.4f}")
        return decay_amount

    def apply_recovery(self, interaction_quality: float = 1.0) -> float:
        """
        Apply recovery from positive interaction.

        Args:
            interaction_quality: Quality of interaction (0.0 to 1.0)

        Returns:
            Amount of recovery applied
        """
        recovery_amount = self._config.recovery_rate * interaction_quality
        self._last_activity = datetime.now()

        for component in FidelityComponent:
            current = self._components[component]
            expected = self._config.expected_scores.get(component, 1.0)
            # Recover towards expected value
            if current < expected:
                self._components[component] = min(expected, current + recovery_amount)

        logger.debug(f"Applied fidelity recovery: {recovery_amount:.4f}")
        return recovery_amount

    # -------------------------------------------------------------------------
    # Baseline Management
    # -------------------------------------------------------------------------

    def compute_context_hash(self, context: Dict[str, Any]) -> str:
        """
        Compute a hash of the current context for comparison.

        Args:
            context: Context dictionary to hash

        Returns:
            SHA-256 hash of context
        """
        import json
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    def set_baseline(self, context: Dict[str, Any]) -> None:
        """
        Set the baseline context for comparison.

        Args:
            context: Context representing baseline state
        """
        self._baseline_hash = self.compute_context_hash(context)
        self._record_snapshot("baseline_set")
        logger.info("Baseline set")

    def compare_to_baseline(self, context: Dict[str, Any]) -> float:
        """
        Compare current context to baseline.

        Args:
            context: Current context to compare

        Returns:
            Similarity score (1.0 = identical, 0.0 = completely different)
        """
        if self._baseline_hash is None:
            return 1.0  # No baseline set, assume coherent

        current_hash = self.compute_context_hash(context)

        # Simple hash comparison (in production, would use semantic similarity)
        if current_hash == self._baseline_hash:
            return 1.0
        else:
            # Compute character-level similarity as a rough metric
            matches = sum(
                a == b for a, b in zip(current_hash, self._baseline_hash)
            )
            return matches / len(self._baseline_hash)

    # -------------------------------------------------------------------------
    # Snapshots and History
    # -------------------------------------------------------------------------

    def _record_snapshot(self, notes: str = "") -> None:
        """Record a fidelity snapshot."""
        snapshot = FidelitySnapshot(
            timestamp=datetime.now(),
            overall_score=self.fidelity,
            components=self._components.copy(),
            context_hash=self._baseline_hash or "",
            state_name="",  # Will be set by caller if needed
            notes=notes,
        )
        self._history.append(snapshot)

        # Limit history size
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    def record_snapshot(self, state_name: str, notes: str = "") -> FidelitySnapshot:
        """
        Record a fidelity snapshot with state information.

        Args:
            state_name: Current protocol state name
            notes: Optional notes

        Returns:
            The recorded snapshot
        """
        snapshot = FidelitySnapshot(
            timestamp=datetime.now(),
            overall_score=self.fidelity,
            components=self._components.copy(),
            context_hash=self._baseline_hash or "",
            state_name=state_name,
            notes=notes,
        )
        self._history.append(snapshot)
        return snapshot

    def get_trend(self, window_size: int = 10) -> float:
        """
        Calculate fidelity trend over recent history.

        Args:
            window_size: Number of snapshots to consider

        Returns:
            Trend value (positive = improving, negative = declining)
        """
        if len(self._history) < 2:
            return 0.0

        recent = self._history[-window_size:]
        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        n = len(recent)
        sum_x = sum(range(n))
        sum_y = sum(s.overall_score for s in recent)
        sum_xy = sum(i * s.overall_score for i, s in enumerate(recent))
        sum_x2 = sum(i * i for i in range(n))

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    # -------------------------------------------------------------------------
    # Status and Serialization
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current fidelity status."""
        return {
            "overall_fidelity": self.fidelity,
            "is_stable": self.is_stable,
            "deviation_index": self.deviation_index,
            "components": {c.value: v for c, v in self._components.items()},
            "unresolved_deviations": len(self.unresolved_deviations),
            "trend": self.get_trend(),
            "snapshots_recorded": len(self._history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker to dictionary."""
        return {
            "components": {c.value: v for c, v in self._components.items()},
            "history": [s.to_dict() for s in self._history[-100:]],  # Last 100
            "deviations": [d.to_dict() for d in self._deviations],
            "baseline_hash": self._baseline_hash,
            "last_activity": self._last_activity.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelityTracker":
        """Deserialize from dictionary."""
        tracker = cls()
        tracker._components = {
            FidelityComponent(k): v for k, v in data.get("components", {}).items()
        }
        tracker._history = [
            FidelitySnapshot.from_dict(s) for s in data.get("history", [])
        ]
        tracker._baseline_hash = data.get("baseline_hash")
        if "last_activity" in data:
            tracker._last_activity = datetime.fromisoformat(data["last_activity"])
        return tracker


# =============================================================================
# Fidelity Analyzer
# =============================================================================

class FidelityAnalyzer:
    """
    Analyzes fidelity data to provide insights and recommendations.
    """

    @staticmethod
    def analyze_response_consistency(
        responses: List[Dict[str, Any]],
        persona_traits: List[str],
    ) -> float:
        """
        Analyze how consistently responses align with persona traits.

        Args:
            responses: List of response data
            persona_traits: Expected traits/characteristics

        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not responses:
            return 1.0

        # Simplified analysis - in production would use NLP
        consistent_count = 0
        for response in responses:
            text = response.get("text", "").lower()
            # Check for trait alignment (simplified)
            trait_matches = sum(
                1 for trait in persona_traits if trait.lower() in text
            )
            if trait_matches > 0 or len(persona_traits) == 0:
                consistent_count += 1

        return consistent_count / len(responses)

    @staticmethod
    def analyze_memory_integrity(
        context_changes: List[Dict[str, Any]],
    ) -> float:
        """
        Analyze memory/context integrity over time.

        Args:
            context_changes: List of context change events

        Returns:
            Integrity score (0.0 to 1.0)
        """
        if not context_changes:
            return 1.0

        # Check for unexpected context losses
        losses = sum(1 for c in context_changes if c.get("type") == "loss")
        integrity = 1.0 - (losses / len(context_changes))
        return max(0.0, integrity)

    @staticmethod
    def recommend_recovery_actions(
        tracker: FidelityTracker,
    ) -> List[str]:
        """
        Recommend actions to improve fidelity.

        Args:
            tracker: FidelityTracker instance

        Returns:
            List of recommended actions
        """
        recommendations = []

        # Check each component
        for component, score in tracker.components.items():
            if score < 0.7:
                if component == FidelityComponent.MEMORY_INTEGRITY:
                    recommendations.append("Reinforce context with explicit references")
                elif component == FidelityComponent.RESPONSE_CONSISTENCY:
                    recommendations.append("Review and align with persona baseline")
                elif component == FidelityComponent.STATE_COHERENCE:
                    recommendations.append("Trigger baseline reconciliation")
                elif component == FidelityComponent.BEHAVIORAL_ALIGNMENT:
                    recommendations.append("Reset behavioral parameters")
                elif component == FidelityComponent.TEMPORAL_CONTINUITY:
                    recommendations.append("Rebuild conversation timeline")

        # Overall recommendations
        if tracker.fidelity < 0.6:
            recommendations.insert(0, "CRITICAL: Full persona reset recommended")
        elif tracker.fidelity < 0.8:
            recommendations.insert(0, "Baseline check recommended")

        if tracker.deviation_index > 0.3:
            recommendations.append("High deviation - investigate root cause")

        return recommendations
