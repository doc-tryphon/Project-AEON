import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources (e.g., check LLM connection)
    print("Project AEON API starting up...")
    yield
    # Shutdown: Cleanup resources
    print("Project AEON API shutting down...")

app = FastAPI(
    title="Project AEON API",
    description="API interface for the Quantum logic and Dolores persona",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
origins_str = os.getenv("AEON_CORS_ORIGINS", "http://localhost:3000,https://bluerose.systems")
origins = [origin.strip() for origin in origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Project AEON API is operational", "status": "running"}

# Import and include routers
from .routes import chat, verify
# Session routes not implemented yet, skipping for now
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(verify.router, prefix="/api", tags=["Verify"])
# app.include_router(session.router, prefix="/api", tags=["Session"])
