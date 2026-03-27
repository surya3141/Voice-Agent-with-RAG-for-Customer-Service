"""Integration tests for FastAPI routes (no real API calls)."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(
    transcript: str = "Hello",
    intent: str = "faq",
    reply_text: str = "We are open 9-5.",
    reply_audio: bytes = b"mp3-bytes",
    context: str = "Some context.",
):
    from voice_agent.agent import AgentResponse

    agent = MagicMock()
    agent.handle_audio.return_value = AgentResponse(
        transcript=transcript,
        intent=intent,
        reply_text=reply_text,
        reply_audio=reply_audio,
        context_used=context,
    )
    agent.handle_text.return_value = AgentResponse(
        transcript=transcript,
        intent=intent,
        reply_text=reply_text,
        reply_audio=reply_audio,
        context_used=context,
    )
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Create a TestClient with a pre-loaded mock agent (no lifespan)."""
    import app as app_module
    from fastapi import FastAPI

    mock = _mock_agent()

    # Patch the module-level _agent so the routes use the mock,
    # and replace lifespan with a no-op so no real VoiceAgent is created.
    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield

    with patch.object(app_module, "_agent", mock):
        with patch.object(app_module.app.router, "lifespan_context", _noop_lifespan):
            with TestClient(app_module.app) as c:
                yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestTextEndpoint:
    def test_text_returns_json(self, client: TestClient):
        resp = client.post("/voice/text", json={"text": "What are your hours?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data
        assert "reply_text" in data
        assert "transcript" in data

    def test_text_missing_body_returns_422(self, client: TestClient):
        resp = client.post("/voice/text", json={})
        assert resp.status_code == 422


class TestVoiceEndpoint:
    def test_voice_returns_audio(self, client: TestClient):
        audio = io.BytesIO(b"fake-audio")
        resp = client.post(
            "/voice", files={"audio": ("test.wav", audio, "audio/wav")}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert resp.content == b"mp3-bytes"

    def test_voice_sets_intent_header(self, client: TestClient):
        audio = io.BytesIO(b"fake-audio")
        resp = client.post(
            "/voice", files={"audio": ("test.wav", audio, "audio/wav")}
        )
        assert "x-intent" in resp.headers


class TestTextToAudioEndpoint:
    def test_text_to_audio_returns_mp3(self, client: TestClient):
        resp = client.post(
            "/voice/text/audio", json={"text": "What are your hours?"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
