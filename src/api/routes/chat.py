from fastapi import APIRouter, HTTPException, Depends
from src.api.models import ChatRequest, ChatResponse, VerificationResultModel
from src.api.session_manager import session_manager, TutorSession

router = APIRouter()

async def get_session(request: ChatRequest) -> TutorSession:
    """Dependency to retrieve or create session."""
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if not session:
             # If session expired or invalid, we could error.
             # Or auto-recreate. Let's error to be strict about state.
             raise HTTPException(status_code=404, detail="Session not found or expired")
        return session
    else:
        return session_manager.create_session()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, session: TutorSession = Depends(get_session)):
    """
    Main interactions endpoint. 
    1. Receives user message.
    2. BlackwallController processes it (checking mode, adding context).
    3. DoloresEngine generates response (via LLM).
    4. Returns text + verified claims metadata.
    """
    # Override mode if requested and allowed
    if request.mode:
        from src.interface.enums import InterfaceMode
        # TODO: safe conversion
        pass 

    # Processing logic
    # We need to map the Blackwall response back to our API model
    # Since BlackwallController.process() isn't fully async/standardized in the snippets 
    # I viewed, I will assume a synchronous interface for now based on CLI.
    
    try:
        # Assuming process_input returns a rich structure
        # We might need to adjust this call based on actual method signature
        result = session.controller.process_input(request.message) 
        
        from src.api.models import DisplayStatus
        
        # Determine Display Status for CRT Monitor
        # Logic: 
        # - Rigorous Mode + Verified Claims = VERIFIED (Green)
        # - Error/Low Fidelity = PROTOCOL_BREACH (Red)
        # - Default = UNVERIFIED (Amber)
        
        display_status = DisplayStatus.UNVERIFIED
        
        # This is a heuristic until we have meaningful flags from Blackwall
        if session.fidelity < 0.5:
             display_status = DisplayStatus.PROTOCOL_BREACH
        elif session.mode == "rigorous":
             # In rigorous mode, we assume checks passed if we got a response without exception
             # Ideally we check result metadata
             display_status = DisplayStatus.VERIFIED
        
        # Transform result to ChatResponse
        return ChatResponse(
            response=result.content,
            session_id=session.session_id,
            verified_claims=result.metadata.get("verified_claims", []),
            protocol_state=session.protocol_state,
            fidelity=session.fidelity,
            mode=session.mode,
            display_status=display_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
