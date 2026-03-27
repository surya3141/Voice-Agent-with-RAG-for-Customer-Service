"""Unit tests for the VoiceAgent orchestration pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voice_agent.agent import AgentResponse, VoiceAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    transcript: str = "Where is my order?",
    intent: str = "order_status",
    reply_text: str = "Your order is on its way!",
    context: str = "Order #123 is shipped.",
    reply_audio: bytes = b"fake-mp3",
) -> VoiceAgent:
    stt = MagicMock()
    stt.transcribe.return_value = transcript

    tts = MagicMock()
    tts.synthesize.return_value = reply_audio

    llm = MagicMock()
    llm.detect_intent.return_value = intent
    llm.generate_response.return_value = reply_text

    rag = MagicMock()
    rag.get_context.return_value = context
    rag.load_or_build_index.return_value = None

    return VoiceAgent(stt=stt, tts=tts, llm=llm, rag=rag)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandleText:
    def test_returns_agent_response(self):
        agent = _make_agent()
        result = agent.handle_text("Where is my order?")
        assert isinstance(result, AgentResponse)

    def test_transcript_equals_input(self):
        agent = _make_agent(transcript="Where is my order?")
        result = agent.handle_text("Where is my order?")
        assert result.transcript == "Where is my order?"

    def test_intent_populated(self):
        agent = _make_agent(intent="order_status")
        result = agent.handle_text("Where is my order?")
        assert result.intent == "order_status"

    def test_reply_text_populated(self):
        agent = _make_agent(reply_text="Your order ships tomorrow.")
        result = agent.handle_text("Where is my order?")
        assert result.reply_text == "Your order ships tomorrow."

    def test_no_audio_by_default(self):
        agent = _make_agent()
        result = agent.handle_text("Hello")
        assert result.reply_audio == b""

    def test_audio_when_synthesize_requested(self):
        agent = _make_agent(reply_audio=b"mp3-bytes")
        result = agent.handle_text("Hello", synthesize_reply=True)
        assert result.reply_audio == b"mp3-bytes"

    def test_tts_not_called_without_synthesize(self):
        agent = _make_agent()
        agent.handle_text("Hello", synthesize_reply=False)
        agent.tts.synthesize.assert_not_called()

    def test_context_used_populated(self):
        agent = _make_agent(context="Some context passage.")
        result = agent.handle_text("What are your hours?")
        assert result.context_used == "Some context passage."


class TestHandleAudio:
    def test_transcribes_audio(self):
        agent = _make_agent(transcript="Hello, I need help.")
        result = agent.handle_audio(b"raw-audio")
        agent.stt.transcribe.assert_called_once()
        assert result.transcript == "Hello, I need help."

    def test_synthesizes_audio_by_default(self):
        agent = _make_agent(reply_audio=b"mp3")
        result = agent.handle_audio(b"raw-audio", synthesize_reply=True)
        agent.tts.synthesize.assert_called_once()
        assert result.reply_audio == b"mp3"

    def test_filename_passed_to_stt(self):
        agent = _make_agent()
        agent.handle_audio(b"audio", filename="call.mp3")
        agent.stt.transcribe.assert_called_once_with(b"audio", filename="call.mp3")


class TestLoadKnowledgeBase:
    def test_delegates_to_rag(self):
        agent = _make_agent()
        agent.load_knowledge_base("some/data/dir")
        agent.rag.load_or_build_index.assert_called_once_with("some/data/dir")
