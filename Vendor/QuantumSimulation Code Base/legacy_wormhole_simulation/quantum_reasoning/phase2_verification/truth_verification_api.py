"""
FastAPI server for Truth Verification Network.

This server provides HTTP endpoints for the multi-agent truth verification
system with quantum-inspired entanglement between reasoning chains.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import asyncio
import logging
from datetime import datetime
import sys
import os

# Add path to truth verification module
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from truth_verification_network import (
        TruthVerificationNetwork, 
        LLMAgent, 
        VerificationResult,
        VerificationReport,
        ReasoningChain
    )
    VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Truth verification not available: {e}")
    VERIFICATION_AVAILABLE = False


# Request/Response Models
class TruthVerificationRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    num_agents: int = 3
    consistency_threshold: float = 0.7
    agent_models: Optional[List[str]] = None


class VerificationAgentInfo(BaseModel):
    agent_id: str
    model_name: str
    confidence: float
    response: str
    reasoning_steps: List[str]


class TruthVerificationResponse(BaseModel):
    query: str
    primary_response: str
    verification_result: str
    consistency_score: float
    confidence_score: float
    entanglement_strength: float
    agents_used: List[VerificationAgentInfo]
    recommendations: List[str]
    cross_examination_summary: str
    processing_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Truth Verification API",
    description="API for multi-agent truth verification with quantum entanglement",
    version="1.0.0"
)

# Global verification network
verification_network = None


def initialize_verification_network():
    """Initialize the truth verification network."""
    global verification_network
    
    if not VERIFICATION_AVAILABLE:
        return False
    
    try:
        # Initialize agents with your local LLMs
        # These should match your Ollama models
        agents = [
            LLMAgent("llama_agent", "llama2:7b", "http://localhost:11434/api/generate"),
            LLMAgent("codellama_agent", "codellama:7b", "http://localhost:11434/api/generate"),
            LLMAgent("mistral_agent", "mistral:7b", "http://localhost:11434/api/generate"),
            # Add more agents based on your available models
            # LLMAgent("gpt4all_agent", "gpt4all-falcon:latest", "http://localhost:11434/api/generate"),
        ]
        
        verification_network = TruthVerificationNetwork(agents)
        logging.info(f"Truth verification network initialized with {len(agents)} agents")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize verification network: {e}")
        return False


def format_cross_examination_summary(cross_examinations: List[Dict[str, Any]]) -> str:
    """Format cross-examination results into readable summary.
    
    Args:
        cross_examinations: List of cross-examination results
        
    Returns:
        Formatted summary string
    """
    if not cross_examinations:
        return "No cross-examinations performed."
    
    summary_parts = ["## Cross-Examination Results\n"]
    
    for i, exam in enumerate(cross_examinations, 1):
        examiner = exam.get('examiner_agent', f'Agent_{i}')
        agreement = exam.get('agreement_level', 0.0)
        concerns = exam.get('concerns', [])
        confidence = exam.get('examiner_confidence', 0.0)
        entanglement = exam.get('entanglement_strength', 0.0)
        
        summary_parts.append(f"**{examiner}** (Confidence: {confidence:.2f}, Entanglement: {entanglement:.2f})")
        summary_parts.append(f"- Agreement Level: {agreement:.2f}")
        
        if concerns and len(concerns) > 0 and concerns[0]:
            summary_parts.append("- Concerns:")
            for concern in concerns[:3]:  # Limit to first 3 concerns
                if concern.strip():
                    summary_parts.append(f"  • {concern.strip()}")
        else:
            summary_parts.append("- No significant concerns raised")
        
        summary_parts.append("")
    
    return "\n".join(summary_parts)


def create_mock_verification_response(request: TruthVerificationRequest) -> Dict[str, Any]:
    """Create mock verification response when real verification is unavailable.
    
    Args:
        request: Verification request
        
    Returns:
        Mock response data
    """
    mock_agents = [
        {
            "agent_id": "mock_agent_1",
            "model_name": "mock_llama",
            "confidence": 0.8,
            "response": f"Mock analysis of: {request.query}",
            "reasoning_steps": [
                "Analyzed query from perspective 1",
                "Considered multiple interpretations",
                "Reached conclusion with high confidence"
            ]
        },
        {
            "agent_id": "mock_agent_2", 
            "model_name": "mock_mistral",
            "confidence": 0.75,
            "response": f"Alternative analysis of: {request.query}",
            "reasoning_steps": [
                "Examined query from different angle",
                "Cross-referenced with known information",
                "Moderate confidence in conclusion"
            ]
        }
    ]
    
    mock_summary = """## Mock Cross-Examination Results

**mock_agent_2** (Confidence: 0.75, Entanglement: 0.6)
- Agreement Level: 0.80
- Concerns: None significant
- Note: This is a simulated verification process

To enable full truth verification:
1. Ensure Ollama is running with your LLM models
2. Install required dependencies
3. Restart the server"""
    
    return {
        "primary_response": f"Mock verified response for: {request.query}",
        "verification_result": "uncertain",
        "consistency_score": 0.75,
        "confidence_score": 0.8,
        "entanglement_strength": 0.6,
        "agents_used": mock_agents,
        "recommendations": [
            "Mock mode active - install dependencies for full verification",
            "Query processed with simulated multi-agent analysis"
        ],
        "cross_examination_summary": mock_summary
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    success = initialize_verification_network()
    if not success:
        logging.warning("Truth verification initialization failed - API will run in mock mode")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Truth Verification API",
        "version": "1.0.0", 
        "status": "operational",
        "verification_available": VERIFICATION_AVAILABLE,
        "endpoints": {
            "/truth-verification": "POST - Verify query with multi-agent network",
            "/health": "GET - Health check",
            "/status": "GET - System status",
            "/agents": "GET - List available agents"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "verification_available": VERIFICATION_AVAILABLE,
        "network_initialized": verification_network is not None
    }


@app.get("/status")
async def system_status():
    """System status endpoint."""
    agent_count = len(verification_network.agents) if verification_network else 0
    
    return {
        "verification_available": VERIFICATION_AVAILABLE,
        "network_initialized": verification_network is not None,
        "agent_count": agent_count,
        "system_info": {
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
    }


@app.get("/agents")
async def list_agents():
    """List available verification agents."""
    if not verification_network:
        return {"agents": [], "message": "Verification network not initialized"}
    
    agents_info = []
    for agent in verification_network.agents:
        agents_info.append({
            "agent_id": agent.agent_id,
            "model_name": agent.model_name,
            "api_endpoint": agent.api_endpoint,
            "capabilities": agent.capabilities
        })
    
    return {"agents": agents_info, "total_count": len(agents_info)}


@app.post("/truth-verification", response_model=TruthVerificationResponse)
async def verify_truth(request: TruthVerificationRequest):
    """Verify query using multi-agent truth verification network.
    
    Args:
        request: Truth verification request
        
    Returns:
        Truth verification response
    """
    start_time = datetime.now()
    
    try:
        # Check if verification network is available
        if not VERIFICATION_AVAILABLE or verification_network is None:
            # Mock response when verification is not available
            mock_data = create_mock_verification_response(request)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TruthVerificationResponse(
                query=request.query,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                success=True,
                **mock_data
            )
        
        # Real verification processing
        report = await verification_network.verify_query(
            query=request.query,
            context=request.context,
            num_agents=request.num_agents
        )
        
        # Extract agent information
        agents_info = []
        
        # Add primary agent
        agents_info.append(VerificationAgentInfo(
            agent_id=report.primary_chain.agent_id,
            model_name=next(
                (a.model_name for a in verification_network.agents if a.agent_id == report.primary_chain.agent_id),
                "unknown"
            ),
            confidence=report.primary_chain.confidence,
            response=report.primary_chain.response,
            reasoning_steps=report.primary_chain.reasoning_steps
        ))
        
        # Add examining agents
        for exam in report.cross_examinations:
            examiner_id = exam.get('examiner_agent', 'unknown')
            agents_info.append(VerificationAgentInfo(
                agent_id=examiner_id,
                model_name=next(
                    (a.model_name for a in verification_network.agents if a.agent_id == examiner_id),
                    "unknown"
                ),
                confidence=exam.get('examiner_confidence', 0.0),
                response=exam.get('examiner_response', ''),
                reasoning_steps=[]  # Not available in cross-examination format
            ))
        
        # Format cross-examination summary
        cross_exam_summary = format_cross_examination_summary(report.cross_examinations)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TruthVerificationResponse(
            query=request.query,
            primary_response=report.primary_chain.response,
            verification_result=report.overall_result.value,
            consistency_score=report.consistency_score,
            confidence_score=report.primary_chain.confidence,
            entanglement_strength=report.quantum_entanglement_strength,
            agents_used=agents_info,
            recommendations=report.recommendations,
            cross_examination_summary=cross_exam_summary,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TruthVerificationResponse(
            query=request.query,
            primary_response="Error processing verification request",
            verification_result="error",
            consistency_score=0.0,
            confidence_score=0.0,
            entanglement_strength=0.0,
            agents_used=[],
            recommendations=["Error occurred during verification"],
            cross_examination_summary="Verification failed due to error",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=str(e)
        )


@app.post("/test")
async def test_verification():
    """Test endpoint for truth verification functionality."""
    test_request = TruthVerificationRequest(
        query="What is the capital of France and why is it economically important?",
        context="Consider both historical and modern perspectives",
        num_agents=3,
        consistency_threshold=0.7
    )
    
    return await verify_truth(test_request)


@app.post("/verify-high-stakes")
async def verify_high_stakes_query(request: TruthVerificationRequest):
    """Specialized endpoint for high-stakes queries requiring maximum verification.
    
    Args:
        request: Verification request
        
    Returns:
        Enhanced verification response
    """
    # Use all available agents and lower consistency threshold for high-stakes queries
    enhanced_request = TruthVerificationRequest(
        query=request.query,
        context=f"HIGH-STAKES QUERY: {request.context}",
        num_agents=None,  # Use all agents
        consistency_threshold=0.8,  # Higher threshold
        agent_models=request.agent_models
    )
    
    response = await verify_truth(enhanced_request)
    
    # Add high-stakes warning to recommendations
    if response.success:
        response.recommendations.insert(0, 
            "HIGH-STAKES VERIFICATION: This query was processed with maximum verification protocols"
        )
    
    return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Truth Verification API server...")
    print(f"Verification Available: {VERIFICATION_AVAILABLE}")
    
    uvicorn.run(
        app,
        host="localhost", 
        port=8002,
        log_level="info"
    )
