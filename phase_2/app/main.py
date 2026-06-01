"""FastAPI application: Main entry point for the Phase 2 Agent API.

Provides the /chat endpoint and application lifecycle management.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent.core import AgentCore
from .models.schemas import ChatRequest, ChatResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent instance
_agent: AgentCore | None = None


def get_config_path() -> str:
    """Get the configuration file path."""
    return os.environ.get("PHASE2_CONFIG", "config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and cleanup resources."""
    global _agent

    config_path = get_config_path()
    logger.info(f"Starting Phase 2 Agent with config: {config_path}")

    try:
        _agent = AgentCore(config_path=config_path)
        logger.info("Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Phase 2 Agent.")
    _agent = None


# Create FastAPI app
app = FastAPI(
    title="Phase 2: Agentic AI on RAG",
    description=(
        "Intelligent Agent that uses Phase 1 RAG system as a tool "
        "to answer product-related queries with evidence-based responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for UI/Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint for the Agent.

    Receives user messages, processes them through the agent pipeline,
    and returns a response with optional product recommendations.

    The agent will:
    1. Check conversation history (short-term memory)
    2. Decide if RAG tools are needed
    3. Query the Phase 1 RAG system if needed
    4. Generate an evidence-based response

    Args:
        request: ChatRequest with chat_id and messages.

    Returns:
        ChatResponse with message and optional product keys.

    Raises:
        HTTPException: If agent is not initialized or processing fails.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please try again later.",
        )

    try:
        # Convert messages to dict format
        user_messages = [
            {"type": msg.type, "content": msg.content}
            for msg in request.messages
        ]

        # Process through the agent
        response = await _agent.process_message(
            chat_id=request.chat_id,
            user_messages=user_messages,
        )

        logger.info(
            f"Chat {request.chat_id}: processed {len(request.messages)} "
            f"message(s), response length: "
            f"{len(response.message) if response.message else 0}"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing chat {request.chat_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": _agent is not None,
    }


@app.get("/sessions")
async def list_sessions():
    """List active conversation sessions (for debugging)."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    sessions = _agent.memory.get_active_sessions()
    return {"active_sessions": sessions, "count": len(sessions)}
