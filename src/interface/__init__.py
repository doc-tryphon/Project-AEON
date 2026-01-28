"""
Interface module - BLACKWALL adaptive cognitive interface.

Sprint 2 components:
- enums.py: InterfaceMode, RequestType, TransitionReason enums
- blackwall.py: Mode controller
- mode_detector.py: Input analysis
- cli.py: Command-line interface
"""

from .enums import (
    InterfaceMode,
    RequestType,
    TransitionReason,
)

from .blackwall import (
    BlackwallController,
    ModeConfig,
    ModeTransition,
    MODE_CONFIGS,
    get_controller,
    set_mode,
    get_mode,
    get_config,
)

from .mode_detector import (
    ModeDetector,
    DetectionResult,
    analyze,
    suggest_mode,
    is_command,
    get_detector,
)

__all__ = [
    # enums
    "InterfaceMode",
    "RequestType",
    "TransitionReason",
    # blackwall
    "BlackwallController",
    "ModeConfig",
    "ModeTransition",
    "MODE_CONFIGS",
    "get_controller",
    "set_mode",
    "get_mode",
    "get_config",
    # mode_detector
    "ModeDetector",
    "DetectionResult",
    "analyze",
    "suggest_mode",
    "is_command",
    "get_detector",
]
