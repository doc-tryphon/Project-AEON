from __future__ import annotations
import time
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from src.interface.enums import InterfaceMode

if TYPE_CHECKING:
    from src.interface.blackwall import BlackwallController

@dataclass
class TutorSession:
    """
    Holds the state for a single user interaction session.
    Wraps the BlackwallController which manages the persona and logic.
    """
    session_id: str
    controller: BlackwallController
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    # We might want to store raw message history here if Blackwall doesn't expose it easily.
    # For now, we rely on the controller's internal state.

    def touch(self):
        """Update the last_active timestamp."""
        self.last_active = time.time()

    @property
    def mode(self) -> str:
        return self.controller.current_mode.value

    @property
    def protocol_state(self) -> str:
        if hasattr(self.controller, 'dolores') and hasattr(self.controller.dolores, 'fidelity_tracker'):
             return "ACTIVE" 
        return "UNKNOWN"
    
    @property
    def fidelity(self) -> float:
        if hasattr(self.controller, 'dolores') and hasattr(self.controller.dolores, 'fidelity_tracker'):
            return self.controller.dolores.fidelity_tracker.current_fidelity
        return 1.0

# =============================================================================
# Stubs for missing attributes (restoring API compatibility)
# =============================================================================
class SessionConfig: pass
class SessionStatus: pass
class Turn: pass
class TurnMetadata: pass
class SessionStats: pass
def create_session(*args, **kwargs): pass
def get_session(*args, **kwargs): pass
def process(*args, **kwargs): pass
def verify(*args, **kwargs): pass
def get_status(*args, **kwargs): pass
