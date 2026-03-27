"""FastAPI application for the Voice Agent with RAG for Customer Service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from voice_agent.agent import AgentResponse, VoiceAgent
from voice_agent.config import settings

# ---------------------------------------------------------------------------
# Application lifespan – build/load the vector index at startup
# ---------------------------------------------------------------------------

_agent: VoiceAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the knowledge base when the server starts."""
    global _agent
    _agent = VoiceAgent()
    _agent.load_knowledge_base()
    yield


app = FastAPI(
    title="Voice Agent – Customer Service",
    description=(
        "A voice-enabled customer-service agent that uses OpenAI Whisper for "
        "speech-to-text, GPT for intent detection and response generation, "
        "a FAISS-backed RAG pipeline for grounded answers, and OpenAI TTS to "
        "speak the reply back to the caller."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_agent() -> VoiceAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")
    return _agent


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TextRequest(BaseModel):
    text: str
    synthesize_reply: bool = False


class TextResponse(BaseModel):
    transcript: str
    intent: str
    reply_text: str
    context_used: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post(
    "/voice",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio reply from the agent.",
        }
    },
    tags=["voice"],
    summary="Submit audio and receive an audio reply",
)
async def voice_endpoint(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/m4a/…)"),
) -> Response:
    """Process a caller's audio file and return an MP3 audio response.

    The pipeline:
    1. Transcribes the uploaded audio with OpenAI Whisper (STT).
    2. Detects intent with GPT.
    3. Retrieves grounding context from the FAISS knowledge base (RAG).
    4. Generates a natural-language reply with GPT.
    5. Converts the reply to speech with OpenAI TTS.
    """
    agent = _get_agent()
    audio_bytes = await audio.read()
    try:
        result: AgentResponse = agent.handle_audio(
            audio_bytes,
            filename=audio.filename or "audio.wav",
            synthesize_reply=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=result.reply_audio,
        media_type="audio/mpeg",
        headers={
            "X-Transcript": result.transcript,
            "X-Intent": result.intent,
        },
    )


@app.post(
    "/voice/text",
    response_model=TextResponse,
    tags=["voice"],
    summary="Submit text and receive a text reply (no audio I/O)",
)
def text_endpoint(request: TextRequest) -> TextResponse:
    """Process a plain-text query and return a JSON response.

    Identical pipeline to ``/voice`` but without STT / TTS, making it
    suitable for testing and chat-based interfaces.
    """
    agent = _get_agent()
    try:
        result: AgentResponse = agent.handle_text(
            request.text,
            synthesize_reply=request.synthesize_reply,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TextResponse(
        transcript=result.transcript,
        intent=result.intent,
        reply_text=result.reply_text,
        context_used=result.context_used,
    )


@app.post(
    "/voice/text/audio",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio reply from the agent.",
        }
    },
    tags=["voice"],
    summary="Submit text and receive an audio reply",
)
def text_to_audio_endpoint(request: TextRequest) -> Response:
    """Process a plain-text query and return an MP3 audio response.

    Useful for hybrid deployments where the caller types but receives
    a spoken reply.
    """
    agent = _get_agent()
    try:
        result: AgentResponse = agent.handle_text(request.text, synthesize_reply=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=result.reply_audio,
        media_type="audio/mpeg",
        headers={
            "X-Transcript": result.transcript,
            "X-Intent": result.intent,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
