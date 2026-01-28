import uuid
import time
import os
from typing import Dict, Optional
from src.interface.blackwall import BlackwallController
from src.tutor.session import TutorSession

class SessionManager:
    """
    Manages active TutorSessions in memory.
    """
    def __init__(self):
        self._sessions: Dict[str, TutorSession] = {}

    def create_session(self, session_id: Optional[str] = None) -> TutorSession:
        """
        Creates a new session with a fresh BlackwallController.
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Dependency Injection (similar to CLI setup)
        # 1. LLM Provider
        # TODO: Allow configuration via env vars or request
        api_key = os.getenv("ANTHROPIC_API_KEY", "mock-key")
        # Defaulting to Mock if key is missing/mock, or Anthropic otherwise
        # For production flexibility, we should inspect environment more robustly
        
        from src.llm.providers import LLMProvider, MockProvider, ClaudeProvider, OpenAIProvider
        from src.llm.interface import LLMConfig

        provider_type = os.getenv("LLM_PROVIDER", "anthropic").lower()
        provider: LLMProvider

        if provider_type == "openrouter" or provider_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("LLM_MODEL", "gpt-4o")
            if not api_key:
                 provider = MockProvider() # Fallback
            else:
                 config = LLMConfig(model=model, max_tokens=1000)
                 provider = OpenAIProvider(api_key=api_key, config=config)
        elif not api_key or api_key.startswith("mock"):
             provider = MockProvider()
        else:
             provider = ClaudeProvider(api_key=api_key)

        # 2. Key components
        from src.tutor.verification_api import TutorVerificationAPI
        from src.persona.dolores_engine import DoloresEngine
        
        verifier = TutorVerificationAPI() 
        # Note: In a real scenario, we might persistence for Dolores state here
        persona_engine = DoloresEngine()
        
        controller = BlackwallController(
            llm=provider,
            verifier=verifier,
            persona_engine=persona_engine
        )
        
        session = TutorSession(session_id=session_id, controller=controller)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[TutorSession]:
        """
        Retrieves an active session.
        """
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Removes a session.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
        
    def cleanup_expired(self, timeout_seconds: int = 3600):
        """
        Removes sessions inactive for longer than timeout.
        """
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_active > timeout_seconds]
        for sid in expired:
            del self._sessions[sid]

# Global instance
session_manager = SessionManager()
