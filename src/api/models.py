from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class DisplayStatus(str, Enum):
    """
    Status signals for the Frontend CRT Monitor to determine text color/alert level.
    """
    VERIFIED = "VERIFIED"               # CRT-GREEN (Safe/Confirmed)
    UNVERIFIED = "UNVERIFIED"           # CRT-AMBER (Info/Speculation)
    PROTOCOL_BREACH = "PROTOCOL_BREACH" # CRT-RED (Danger/Violation)
    SYSTEM_ERROR = "SYSTEM_ERROR"       # CRT-RED (System Failure)

class VerificationResultModel(BaseModel):
    """
    Data model for verification results, mirroring the internal VerificationResult.
    """
    claim: str = Field(..., description="The mathematical or physical claim being verified.")
    verified: bool = Field(..., description="Whether the claim was verified as true.")
    proof: Optional[str] = Field(None, description="Symbolic proof or explanation.")
    confidence: float = Field(1.0, description="Confidence score (0.0 to 1.0).")
    error: Optional[str] = Field(None, description="Error message if verification failed.")
    display_status: Optional[DisplayStatus] = Field(None, description="Visual status signal.")

class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """
    message: str = Field(..., description="User's input message.")
    session_id: Optional[str] = Field(None, description="Unique session identifier. If None, a new session is created.")
    mode: Optional[str] = Field(None, description="Optional mode override (rigorous/exploratory).")

class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.
    """
    response: str = Field(..., description="The AI's text response.")
    session_id: str = Field(..., description="Session identifier to maintain context.")
    verified_claims: List[VerificationResultModel] = Field(default_factory=list, description="List of claims verified in this turn.")
    protocol_state: str = Field(..., description="Current state of the Dolores protocol (e.g., ANGEL, CHIMERA).")
    fidelity: float = Field(..., description="Current fidelity score of the persona.")
    mode: str = Field(..., description="Current operating mode.")
    display_status: DisplayStatus = Field(DisplayStatus.UNVERIFIED, description="Visual status signal for CRT monitor color coding.")

class SessionInfo(BaseModel):
    """
    Metadata about an active session.
    """
    session_id: str
    created_at: float
    last_active: float
    mode: str
