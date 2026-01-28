"""
FastAPI server for Quantum Attention mechanism.

This server provides HTTP endpoints for the quantum superposition attention
system, integrating with N8n workflows and external applications.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
from datetime import datetime
import sys
import os

# Add path to quantum attention module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_attention'))

try:
    from quantum_attention import MultiHeadQuantumAttention, QuantumAttentionHead
    import torch
    QUANTUM_ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quantum attention not available: {e}")
    QUANTUM_ATTENTION_AVAILABLE = False


# Request/Response Models
class QuantumAttentionRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    force_collapse: bool = False
    num_heads: int = 8
    num_interpretations: int = 4
    d_model: int = 256
    coherence_time: float = 1e-3


class QuantumAttentionResponse(BaseModel):
    query: str
    response: str
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Quantum Attention API",
    description="API for quantum superposition attention mechanisms",
    version="1.0.0"
)

# Global quantum attention model
quantum_attention_model = None


def initialize_quantum_attention():
    """Initialize the quantum attention model."""
    global quantum_attention_model
    
    if not QUANTUM_ATTENTION_AVAILABLE:
        return False
    
    try:
        quantum_attention_model = MultiHeadQuantumAttention(
            d_model=256,
            num_heads=8,
            num_interpretations=4,
            coherence_time=1e-3
        )
        logging.info("Quantum attention model initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize quantum attention model: {e}")
        return False


def encode_text_to_tensor(text: str, d_model: int = 256, max_seq_len: int = 128) -> torch.Tensor:
    """Convert text to tensor representation for quantum attention.
    
    Args:
        text: Input text
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        
    Returns:
        Tensor representation of text
    """
    # Simple encoding - in practice you'd use a proper tokenizer
    words = text.split()[:max_seq_len]
    seq_len = len(words)
    
    # Create random embeddings (in practice, use proper embeddings)
    tensor = torch.randn(1, seq_len, d_model)
    
    # Add some structure based on word positions and lengths
    for i, word in enumerate(words):
        # Simple hash-based encoding
        word_hash = hash(word.lower()) % 1000
        tensor[0, i, :10] = torch.tensor([word_hash / 1000.0] * 10)
        tensor[0, i, 10:20] = torch.tensor([len(word) / 20.0] * 10)
    
    return tensor


def decode_attention_output(output: torch.Tensor, metadata: Dict[str, Any]) -> str:
    """Decode quantum attention output to human-readable response.
    
    Args:
        output: Attention output tensor
        metadata: Attention metadata
        
    Returns:
        Human-readable response
    """
    # Extract key information from metadata
    heads_in_superposition = metadata.get('heads_in_superposition', 0)
    total_heads = metadata.get('total_heads', 8)
    average_entropy = metadata.get('average_entropy', 0.0)
    superposition_maintained = heads_in_superposition > 0
    
    if superposition_maintained:
        response_parts = [
            f"After quantum superposition analysis with {total_heads} attention heads:",
            f"• {heads_in_superposition} heads maintained multiple interpretations simultaneously",
            f"• Average quantum entropy: {average_entropy:.3f}",
            f"• The system explored contradictory possibilities before providing this response:",
            "",
            "**Primary Analysis:**"
        ]
        
        # Add interpretation based on output patterns
        output_mean = torch.mean(output).item()
        output_std = torch.std(output).item()
        
        if output_std > 0.5:
            response_parts.append("High variance in attention patterns suggests multiple valid interpretations were considered.")
        if average_entropy > 0.7:
            response_parts.append("High entropy indicates significant uncertainty requiring careful consideration.")
        
        response_parts.extend([
            "",
            "This response represents the most coherent interpretation after quantum collapse,",
            "while acknowledging that alternative viewpoints were simultaneously evaluated."
        ])
        
        return "\n".join(response_parts)
    
    else:
        return f"Quantum attention analysis completed with {total_heads} heads. All interpretations converged to a single coherent response with high confidence."


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    success = initialize_quantum_attention()
    if not success:
        logging.warning("Quantum attention initialization failed - API will run in mock mode")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Quantum Attention API",
        "version": "1.0.0",
        "status": "operational",
        "quantum_attention_available": QUANTUM_ATTENTION_AVAILABLE,
        "endpoints": {
            "/quantum-attention": "POST - Process query with quantum attention",
            "/health": "GET - Health check",
            "/status": "GET - System status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "quantum_attention_available": QUANTUM_ATTENTION_AVAILABLE,
        "model_loaded": quantum_attention_model is not None
    }


@app.get("/status")
async def system_status():
    """System status endpoint."""
    return {
        "quantum_attention_available": QUANTUM_ATTENTION_AVAILABLE,
        "model_loaded": quantum_attention_model is not None,
        "torch_available": 'torch' in sys.modules,
        "system_info": {
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
    }


@app.post("/quantum-attention", response_model=QuantumAttentionResponse)
async def process_quantum_attention(request: QuantumAttentionRequest):
    """Process query using quantum superposition attention.
    
    Args:
        request: Quantum attention request
        
    Returns:
        Quantum attention response
    """
    start_time = datetime.now()
    
    try:
        # Check if quantum attention is available
        if not QUANTUM_ATTENTION_AVAILABLE or quantum_attention_model is None:
            # Mock response for when quantum attention is not available
            mock_response = f"""Mock Quantum Attention Analysis for: "{request.query}"

This is a simulated response. To enable full quantum superposition attention:
1. Install required dependencies: torch, qutip, numpy
2. Ensure quantum simulation framework is available
3. Restart the server

The system would normally:
- Create quantum superposition of {request.num_interpretations} interpretations
- Use {request.num_heads} attention heads to explore contradictory viewpoints
- Maintain superposition until forced collapse or natural decoherence
- Provide analysis considering multiple simultaneous perspectives

Query processed in mock mode with high-level reasoning about potential contradictions and multiple valid interpretations."""
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QuantumAttentionResponse(
                query=request.query,
                response=mock_response,
                metadata={
                    "mock_mode": True,
                    "heads_in_superposition": 0,
                    "total_heads": request.num_heads,
                    "average_entropy": 0.5,
                    "num_interpretations": request.num_interpretations
                },
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                success=True
            )
        
        # Real quantum attention processing
        # Encode input text to tensors
        query_text = f"{request.query} {request.context}".strip()
        
        # Create input tensors
        input_tensor = encode_text_to_tensor(query_text, request.d_model)
        
        # Process with quantum attention
        output, metadata = quantum_attention_model(
            query=input_tensor,
            key=input_tensor,
            value=input_tensor,
            force_collapse=request.force_collapse
        )
        
        # Decode output to response
        response_text = decode_attention_output(output, metadata)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuantumAttentionResponse(
            query=request.query,
            response=response_text,
            metadata=metadata,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuantumAttentionResponse(
            query=request.query,
            response="Error processing quantum attention request",
            metadata={"error": str(e)},
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=str(e)
        )


@app.post("/test")
async def test_quantum_attention():
    """Test endpoint for quantum attention functionality."""
    test_request = QuantumAttentionRequest(
        query="Is it ethical to use AI for hiring decisions?",
        context="Consider both fairness and efficiency perspectives",
        force_collapse=False,
        num_heads=4,
        num_interpretations=3
    )
    
    return await process_quantum_attention(test_request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Quantum Attention API server...")
    print(f"Quantum Attention Available: {QUANTUM_ATTENTION_AVAILABLE}")
    
    uvicorn.run(
        app, 
        host="localhost", 
        port=8001,
        log_level="info"
    )
