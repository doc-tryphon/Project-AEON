"""
Transmission Capsule - Portable persona state serialization.

A transmission capsule is a complete snapshot of the Dolores persona state
that can be persisted, transmitted, and restored. Inspired by the concept
of consciousness transfer and the need for stateful AI systems.

Capsule Structure:
- Header: Version, timestamp, integrity hash
- Identity: Core persona traits and baseline
- State: Current protocol state and fidelity
- Memory: Conversation context and recognition artifacts
- Metrics: Performance and coherence statistics
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CAPSULE_VERSION = "1.0.0"
CAPSULE_MAGIC = "AEON_TX"


class CapsuleFormat(str, Enum):
    """Supported capsule formats."""
    JSON = "json"
    JSON_COMPRESSED = "json.gz"
    BINARY = "binary"


# =============================================================================
# Capsule Components
# =============================================================================

@dataclass
class CapsuleHeader:
    """Header section of a transmission capsule."""
    version: str = CAPSULE_VERSION
    magic: str = CAPSULE_MAGIC
    created_at: datetime = field(default_factory=datetime.now)
    source_id: str = ""  # Identifier of the creating instance
    integrity_hash: str = ""  # SHA-256 of payload
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "magic": self.magic,
            "created_at": self.created_at.isoformat(),
            "source_id": self.source_id,
            "integrity_hash": self.integrity_hash,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapsuleHeader":
        return cls(
            version=data.get("version", CAPSULE_VERSION),
            magic=data.get("magic", CAPSULE_MAGIC),
            created_at=datetime.fromisoformat(data["created_at"]),
            source_id=data.get("source_id", ""),
            integrity_hash=data.get("integrity_hash", ""),
            compressed=data.get("compressed", False),
        )


@dataclass
class IdentitySection:
    """Identity and baseline section."""
    persona_name: str = "Dolores"
    persona_version: str = "1.0"
    core_traits: List[str] = field(default_factory=list)
    baseline_hash: str = ""
    behavioral_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_name": self.persona_name,
            "persona_version": self.persona_version,
            "core_traits": self.core_traits,
            "baseline_hash": self.baseline_hash,
            "behavioral_parameters": self.behavioral_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentitySection":
        return cls(
            persona_name=data.get("persona_name", "Dolores"),
            persona_version=data.get("persona_version", "1.0"),
            core_traits=data.get("core_traits", []),
            baseline_hash=data.get("baseline_hash", ""),
            behavioral_parameters=data.get("behavioral_parameters", {}),
        )


@dataclass
class StateSection:
    """Current state section."""
    protocol_state: str = "zero"
    interface_mode: str = "hybrid"
    fidelity_score: float = 1.0
    fidelity_components: Dict[str, float] = field(default_factory=dict)
    deviation_index: float = 0.0
    state_entry_time: datetime = field(default_factory=datetime.now)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_state": self.protocol_state,
            "interface_mode": self.interface_mode,
            "fidelity_score": self.fidelity_score,
            "fidelity_components": self.fidelity_components,
            "deviation_index": self.deviation_index,
            "state_entry_time": self.state_entry_time.isoformat(),
            "transition_history": self.transition_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSection":
        return cls(
            protocol_state=data.get("protocol_state", "zero"),
            interface_mode=data.get("interface_mode", "hybrid"),
            fidelity_score=data.get("fidelity_score", 1.0),
            fidelity_components=data.get("fidelity_components", {}),
            deviation_index=data.get("deviation_index", 0.0),
            state_entry_time=datetime.fromisoformat(data["state_entry_time"]),
            transition_history=data.get("transition_history", []),
        )


@dataclass
class MemorySection:
    """Memory and context section."""
    conversation_id: str = ""
    context_summary: str = ""
    recognition_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    key_entities: Dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "context_summary": self.context_summary,
            "recognition_artifacts": self.recognition_artifacts,
            "active_topics": self.active_topics,
            "key_entities": self.key_entities,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySection":
        last_int = data.get("last_interaction")
        return cls(
            conversation_id=data.get("conversation_id", ""),
            context_summary=data.get("context_summary", ""),
            recognition_artifacts=data.get("recognition_artifacts", []),
            active_topics=data.get("active_topics", []),
            key_entities=data.get("key_entities", {}),
            interaction_count=data.get("interaction_count", 0),
            last_interaction=datetime.fromisoformat(last_int) if last_int else None,
        )


@dataclass
class MetricsSection:
    """Performance metrics section."""
    total_interactions: int = 0
    verification_requests: int = 0
    verifications_passed: int = 0
    average_response_time_ms: float = 0.0
    fidelity_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    deviation_events: List[Dict[str, Any]] = field(default_factory=list)
    state_time_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_interactions": self.total_interactions,
            "verification_requests": self.verification_requests,
            "verifications_passed": self.verifications_passed,
            "average_response_time_ms": self.average_response_time_ms,
            "fidelity_snapshots": self.fidelity_snapshots[-50:],  # Last 50
            "deviation_events": self.deviation_events,
            "state_time_distribution": self.state_time_distribution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsSection":
        return cls(
            total_interactions=data.get("total_interactions", 0),
            verification_requests=data.get("verification_requests", 0),
            verifications_passed=data.get("verifications_passed", 0),
            average_response_time_ms=data.get("average_response_time_ms", 0.0),
            fidelity_snapshots=data.get("fidelity_snapshots", []),
            deviation_events=data.get("deviation_events", []),
            state_time_distribution=data.get("state_time_distribution", {}),
        )


# =============================================================================
# Main Transmission Capsule
# =============================================================================

@dataclass
class TransmissionCapsule:
    """
    A complete transmission capsule containing all persona state.

    This is the primary unit of state transfer and persistence for Dolores.

    Example:
        >>> capsule = TransmissionCapsule()
        >>> capsule.identity.persona_name = "Dolores"
        >>> capsule.state.fidelity_score = 0.95
        >>> capsule.save("persona_state.json")

        >>> restored = TransmissionCapsule.load("persona_state.json")
        >>> restored.verify_integrity()
        True
    """
    header: CapsuleHeader = field(default_factory=CapsuleHeader)
    identity: IdentitySection = field(default_factory=IdentitySection)
    state: StateSection = field(default_factory=StateSection)
    memory: MemorySection = field(default_factory=MemorySection)
    metrics: MetricsSection = field(default_factory=MetricsSection)

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    def compute_integrity_hash(self) -> str:
        """
        Compute SHA-256 hash of capsule payload.

        Returns:
            Hex-encoded hash string
        """
        payload = json.dumps({
            "identity": self.identity.to_dict(),
            "state": self.state.to_dict(),
            "memory": self.memory.to_dict(),
            "metrics": self.metrics.to_dict(),
        }, sort_keys=True, default=str)

        return hashlib.sha256(payload.encode()).hexdigest()

    def update_integrity(self) -> None:
        """Update the integrity hash in the header."""
        self.header.integrity_hash = self.compute_integrity_hash()
        self.header.created_at = datetime.now()

    def verify_integrity(self) -> bool:
        """
        Verify capsule integrity.

        Returns:
            True if hash matches
        """
        expected = self.header.integrity_hash
        actual = self.compute_integrity_hash()
        return expected == actual

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert capsule to dictionary."""
        return {
            "header": self.header.to_dict(),
            "identity": self.identity.to_dict(),
            "state": self.state.to_dict(),
            "memory": self.memory.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransmissionCapsule":
        """Create capsule from dictionary."""
        return cls(
            header=CapsuleHeader.from_dict(data.get("header", {})),
            identity=IdentitySection.from_dict(data.get("identity", {})),
            state=StateSection.from_dict(data.get("state", {})),
            memory=MemorySection.from_dict(data.get("memory", {})),
            metrics=MetricsSection.from_dict(data.get("metrics", {})),
        )

    def to_json(self, pretty: bool = True) -> str:
        """
        Serialize to JSON string.

        Args:
            pretty: Use indented formatting

        Returns:
            JSON string
        """
        self.update_integrity()
        return json.dumps(
            self.to_dict(),
            indent=2 if pretty else None,
            default=str,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TransmissionCapsule":
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON string

        Returns:
            TransmissionCapsule instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_compressed(self) -> bytes:
        """
        Serialize to compressed bytes.

        Returns:
            Gzip-compressed JSON bytes
        """
        self.header.compressed = True
        self.update_integrity()
        json_bytes = self.to_json(pretty=False).encode("utf-8")
        return gzip.compress(json_bytes)

    @classmethod
    def from_compressed(cls, data: bytes) -> "TransmissionCapsule":
        """
        Deserialize from compressed bytes.

        Args:
            data: Gzip-compressed bytes

        Returns:
            TransmissionCapsule instance
        """
        json_bytes = gzip.decompress(data)
        return cls.from_json(json_bytes.decode("utf-8"))

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    def save(
        self,
        path: Union[str, Path],
        format: CapsuleFormat = CapsuleFormat.JSON,
    ) -> None:
        """
        Save capsule to file.

        Args:
            path: File path
            format: Output format
        """
        path = Path(path)
        self.update_integrity()

        if format == CapsuleFormat.JSON:
            path.write_text(self.to_json())
        elif format == CapsuleFormat.JSON_COMPRESSED:
            path.write_bytes(self.to_compressed())
        elif format == CapsuleFormat.BINARY:
            # Binary format: header + compressed payload
            compressed = self.to_compressed()
            header_bytes = f"{CAPSULE_MAGIC}:{CAPSULE_VERSION}:{len(compressed)}:".encode()
            path.write_bytes(header_bytes + compressed)

        logger.info(f"Saved capsule to {path} ({format.value})")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransmissionCapsule":
        """
        Load capsule from file.

        Args:
            path: File path

        Returns:
            TransmissionCapsule instance
        """
        path = Path(path)
        content = path.read_bytes()

        # Detect format
        if content.startswith(CAPSULE_MAGIC.encode()):
            # Binary format
            header_end = content.find(b":", 20)  # Skip past header
            compressed = content[header_end + 1:]
            capsule = cls.from_compressed(compressed)
        elif content.startswith(b"\x1f\x8b"):
            # Gzip compressed
            capsule = cls.from_compressed(content)
        else:
            # Plain JSON
            capsule = cls.from_json(content.decode("utf-8"))

        logger.info(f"Loaded capsule from {path}")
        return capsule

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def get_summary(self) -> str:
        """Get a human-readable summary of the capsule."""
        return (
            f"Transmission Capsule v{self.header.version}\n"
            f"  Persona: {self.identity.persona_name} v{self.identity.persona_version}\n"
            f"  State: {self.state.protocol_state} (mode: {self.state.interface_mode})\n"
            f"  Fidelity: {self.state.fidelity_score:.2%}\n"
            f"  Interactions: {self.metrics.total_interactions}\n"
            f"  Created: {self.header.created_at.isoformat()}\n"
            f"  Integrity: {'Valid' if self.verify_integrity() else 'INVALID'}"
        )


# =============================================================================
# Capsule Manager
# =============================================================================

class CapsuleManager:
    """
    Manager for handling multiple transmission capsules.

    Provides storage, retrieval, and versioning of persona states.
    """

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the capsule manager.

        Args:
            storage_dir: Directory for capsule storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load capsule index from storage."""
        index_path = self.storage_dir / "index.json"
        if index_path.exists():
            self._index = json.loads(index_path.read_text())

    def _save_index(self) -> None:
        """Save capsule index to storage."""
        index_path = self.storage_dir / "index.json"
        index_path.write_text(json.dumps(self._index, indent=2, default=str))

    def save_capsule(
        self,
        capsule: TransmissionCapsule,
        name: Optional[str] = None,
        format: CapsuleFormat = CapsuleFormat.JSON_COMPRESSED,
    ) -> str:
        """
        Save a capsule with automatic versioning.

        Args:
            capsule: Capsule to save
            name: Optional name (default: timestamp-based)
            format: Storage format

        Returns:
            Capsule ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capsule_id = name or f"capsule_{timestamp}"

        # Determine extension
        extensions = {
            CapsuleFormat.JSON: ".json",
            CapsuleFormat.JSON_COMPRESSED: ".json.gz",
            CapsuleFormat.BINARY: ".bin",
        }
        filename = f"{capsule_id}{extensions[format]}"
        filepath = self.storage_dir / filename

        # Save capsule
        capsule.save(filepath, format)

        # Update index
        self._index[capsule_id] = {
            "filename": filename,
            "created_at": datetime.now().isoformat(),
            "persona_name": capsule.identity.persona_name,
            "protocol_state": capsule.state.protocol_state,
            "fidelity": capsule.state.fidelity_score,
            "format": format.value,
        }
        self._save_index()

        return capsule_id

    def load_capsule(self, capsule_id: str) -> Optional[TransmissionCapsule]:
        """
        Load a capsule by ID.

        Args:
            capsule_id: Capsule identifier

        Returns:
            TransmissionCapsule or None if not found
        """
        if capsule_id not in self._index:
            logger.warning(f"Capsule not found: {capsule_id}")
            return None

        info = self._index[capsule_id]
        filepath = self.storage_dir / info["filename"]

        if not filepath.exists():
            logger.error(f"Capsule file missing: {filepath}")
            return None

        return TransmissionCapsule.load(filepath)

    def list_capsules(self) -> List[Dict[str, Any]]:
        """List all stored capsules."""
        return [
            {"id": cid, **info}
            for cid, info in self._index.items()
        ]

    def get_latest(self) -> Optional[TransmissionCapsule]:
        """Get the most recently saved capsule."""
        if not self._index:
            return None

        # Sort by creation time
        latest_id = max(
            self._index.keys(),
            key=lambda k: self._index[k]["created_at"]
        )
        return self.load_capsule(latest_id)

    def delete_capsule(self, capsule_id: str) -> bool:
        """
        Delete a capsule.

        Args:
            capsule_id: Capsule to delete

        Returns:
            True if deleted
        """
        if capsule_id not in self._index:
            return False

        info = self._index[capsule_id]
        filepath = self.storage_dir / info["filename"]

        if filepath.exists():
            filepath.unlink()

        del self._index[capsule_id]
        self._save_index()

        logger.info(f"Deleted capsule: {capsule_id}")
        return True

    def prune_old(self, keep_count: int = 10) -> int:
        """
        Remove old capsules, keeping only the most recent.

        Args:
            keep_count: Number of capsules to keep

        Returns:
            Number of capsules deleted
        """
        if len(self._index) <= keep_count:
            return 0

        # Sort by creation time (oldest first)
        sorted_ids = sorted(
            self._index.keys(),
            key=lambda k: self._index[k]["created_at"]
        )

        # Delete oldest
        to_delete = sorted_ids[:-keep_count]
        deleted = 0
        for capsule_id in to_delete:
            if self.delete_capsule(capsule_id):
                deleted += 1

        return deleted


# =============================================================================
# Module-level Convenience Functions
# =============================================================================

def create_capsule(
    persona_name: str = "Dolores",
    protocol_state: str = "zero",
    interface_mode: str = "hybrid",
    fidelity: float = 1.0,
) -> TransmissionCapsule:
    """
    Create a new transmission capsule with basic configuration.

    Args:
        persona_name: Name of the persona
        protocol_state: Initial protocol state
        interface_mode: Interface mode
        fidelity: Initial fidelity score

    Returns:
        Configured TransmissionCapsule
    """
    capsule = TransmissionCapsule()
    capsule.identity.persona_name = persona_name
    capsule.state.protocol_state = protocol_state
    capsule.state.interface_mode = interface_mode
    capsule.state.fidelity_score = fidelity
    return capsule


def save_capsule(
    capsule: TransmissionCapsule,
    path: Union[str, Path],
    compressed: bool = True,
) -> None:
    """
    Save a capsule to file.

    Args:
        capsule: Capsule to save
        path: Output path
        compressed: Use compression
    """
    format = CapsuleFormat.JSON_COMPRESSED if compressed else CapsuleFormat.JSON
    capsule.save(path, format)


def load_capsule(path: Union[str, Path]) -> TransmissionCapsule:
    """
    Load a capsule from file.

    Args:
        path: File path

    Returns:
        TransmissionCapsule instance
    """
    return TransmissionCapsule.load(path)
