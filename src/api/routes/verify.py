from fastapi import APIRouter, HTTPException
from src.api.models import VerificationResultModel
from src.tutor.verification_api import TutorVerificationAPI, VerificationResult

router = APIRouter()
verifier = TutorVerificationAPI()

@router.get("/verify/gate/{expression}", response_model=VerificationResultModel)
async def verify_gate(expression: str):
    """Verify if a matrix/gate is unitary."""
    result = verifier.verify_gate(expression)
    return _map_result(expression, result)

@router.get("/verify/state/{expression}", response_model=VerificationResultModel)
async def verify_state(expression: str):
    """Verify if a state is valid (normalized)."""
    result = verifier.verify_state(expression)
    return _map_result(expression, result)

def _map_result(claim: str, internal_result: VerificationResult) -> VerificationResultModel:
    from src.api.models import DisplayStatus
    
    status = DisplayStatus.UNVERIFIED
    if internal_result.verified:
        status = DisplayStatus.VERIFIED
    elif internal_result.confidence < 1.0: # Numerical/approximate
        status = DisplayStatus.UNVERIFIED
    else: # Explicit failure
        status = DisplayStatus.PROTOCOL_BREACH

    return VerificationResultModel(
        claim=claim,
        verified=internal_result.verified,
        proof=internal_result.symbolic_proof,
        confidence=internal_result.confidence,
        error=internal_result.explanation if not internal_result.verified else None,
        display_status=status
    )
