"""Tests for the VoiceAgent orchestrator."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from src.config import AppConfig
from src.agent.voice_agent import VoiceAgent


@pytest.fixture
def app_config():
    """Return an AppConfig with dummy API keys."""
    config = AppConfig()
    config.deepgram.api_key = "test-deepgram-key"
    config.elevenlabs.api_key = "test-elevenlabs-key"
    config.groq.api_key = "test-groq-key"
    return config


@pytest.fixture
def mock_agent(app_config):
    """Return a VoiceAgent with all components mocked."""
    agent = VoiceAgent(app_config)

    # Mock STT
    agent._stt = MagicMock()
    agent._stt.call_count = 0
    agent._stt.total_duration_seconds = 0.0

    # Mock TTS
    agent._tts = MagicMock()
    agent._tts.call_count = 0
    agent._tts.total_characters_synthesized = 0
    agent._tts.synthesize.return_value = b"audio-response"

    # Mock LLM
    agent._llm = MagicMock()
    agent._llm.call_count = 0
    agent._llm.total_input_tokens = 0
    agent._llm.total_output_tokens = 0

    # Mock RAG pipeline
    agent._rag_pipeline = MagicMock()
    agent._rag_pipeline.query.return_value = {
        "answer": "Mocked response",
        "sources": [{"document": "doc1", "metadata": {}, "distance": 0.1}],
        "intent": "faq",
    }

    # Mock knowledge store
    agent._knowledge_store = MagicMock()

    return agent


def test_voice_agent_initialisation_default_config():
    """VoiceAgent can be created with default config."""
    agent = VoiceAgent()
    assert agent._config is not None
    assert agent._stt is None
    assert agent._tts is None
    assert agent._llm is None
    assert agent._rag_pipeline is None


def test_process_text_with_mocked_components(mock_agent):
    """process_text returns expected keys with mocked components."""
    result = mock_agent.process_text("What is the return policy?")

    assert "response_text" in result
    assert result["response_text"] == "Mocked response"
    assert "response_audio" in result
    assert "intent" in result
    assert result["intent"] == "faq"
    assert "sources" in result


def test_get_usage_stats_returns_expected_keys(mock_agent):
    """get_usage_stats returns a dict with stt, tts, and llm sections."""
    stats = mock_agent.get_usage_stats()

    assert "stt" in stats
    assert "call_count" in stats["stt"]
    assert "total_duration_seconds" in stats["stt"]

    assert "tts" in stats
    assert "call_count" in stats["tts"]
    assert "total_characters_synthesized" in stats["tts"]

    assert "llm" in stats
    assert "call_count" in stats["llm"]
    assert "total_input_tokens" in stats["llm"]
    assert "total_output_tokens" in stats["llm"]


def test_process_text_handles_rag_failure(mock_agent):
    """process_text handles RAG pipeline failures gracefully."""
    mock_agent._rag_pipeline.query.side_effect = RuntimeError("LLM error")

    result = mock_agent.process_text("What is the return policy?")
    assert "response_text" in result
    assert "Knowledge retrieval failed" in result["response_text"]
    assert result["response_audio"] is None


def test_process_text_without_initialisation():
    """process_text raises RuntimeError when agent is not initialised."""
    agent = VoiceAgent()
    with pytest.raises(RuntimeError, match="not been initialised"):
        agent.process_text("Hello")
