"""Tests for the DeepgramSTT speech-to-text module."""

import pytest
from unittest.mock import patch, MagicMock

from src.config import DeepgramConfig
from src.stt.deepgram_stt import DeepgramSTT, STTError


@pytest.fixture
def deepgram_config():
    """Return a DeepgramConfig with a dummy API key."""
    return DeepgramConfig(api_key="test-api-key", model="nova-2", language="en-US")


@pytest.fixture
def mock_deepgram_client():
    """Return a fully mocked Deepgram client."""
    client = MagicMock()
    # Build the chained call structure:
    # client.listen.rest.v("1").transcribe_file(payload, options)
    mock_transcribe = MagicMock()
    client.listen.rest.v.return_value.transcribe_file = mock_transcribe
    return client, mock_transcribe


def _make_mock_response(transcript="Hello world", confidence=0.98,
                        words=None, duration=1.5):
    """Helper to build a mock Deepgram response object."""
    if words is None:
        words = [
            MagicMock(word="Hello", start=0.0, end=0.5, confidence=0.99),
            MagicMock(word="world", start=0.6, end=1.0, confidence=0.97),
        ]
    alternative = MagicMock()
    alternative.transcript = transcript
    alternative.confidence = confidence
    alternative.words = words

    channel = MagicMock()
    channel.alternatives = [alternative]

    results = MagicMock()
    results.channels = [channel]

    response = MagicMock()
    response.results = results
    response.metadata = MagicMock(duration=duration)
    return response


def test_stt_error_raised_without_api_key():
    """STTError is raised when no API key is provided."""
    config = DeepgramConfig(api_key="")
    with pytest.raises(STTError, match="Deepgram API key is required"):
        DeepgramSTT(config)


@patch("src.stt.deepgram_stt.DeepgramClient")
def test_transcribe_bytes_returns_expected_keys(mock_client_cls, deepgram_config):
    """transcribe_bytes returns a dict with transcript, confidence, words, duration."""
    mock_response = _make_mock_response()
    mock_instance = MagicMock()
    mock_instance.listen.rest.v.return_value.transcribe_file.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    stt = DeepgramSTT(deepgram_config)
    result = stt.transcribe_bytes(b"fake audio data", mimetype="audio/wav")

    assert "transcript" in result
    assert result["transcript"] == "Hello world"
    assert "confidence" in result
    assert "words" in result
    assert "duration" in result


@patch("src.stt.deepgram_stt.DeepgramClient")
def test_call_count_increments(mock_client_cls, deepgram_config):
    """call_count increments after each successful transcription."""
    mock_response = _make_mock_response()
    mock_instance = MagicMock()
    mock_instance.listen.rest.v.return_value.transcribe_file.return_value = mock_response
    mock_client_cls.return_value = mock_instance

    stt = DeepgramSTT(deepgram_config)
    assert stt.call_count == 0

    stt.transcribe_bytes(b"audio1")
    assert stt.call_count == 1

    stt.transcribe_bytes(b"audio2")
    assert stt.call_count == 2


@patch("src.stt.deepgram_stt.DeepgramClient")
def test_total_duration_accumulates(mock_client_cls, deepgram_config):
    """total_duration_seconds accumulates across multiple transcriptions."""
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    stt = DeepgramSTT(deepgram_config)
    assert stt.total_duration_seconds == 0.0

    resp1 = _make_mock_response(duration=2.5)
    resp2 = _make_mock_response(duration=3.0)
    mock_instance.listen.rest.v.return_value.transcribe_file.side_effect = [resp1, resp2]

    stt.transcribe_bytes(b"audio1")
    assert stt.total_duration_seconds == pytest.approx(2.5)

    stt.transcribe_bytes(b"audio2")
    assert stt.total_duration_seconds == pytest.approx(5.5)


@patch("src.stt.deepgram_stt.DeepgramClient")
def test_transcribe_bytes_raises_on_client_failure(mock_client_cls, deepgram_config):
    """STTError is raised when the Deepgram client throws an exception."""
    mock_instance = MagicMock()
    mock_instance.listen.rest.v.return_value.transcribe_file.side_effect = RuntimeError("API down")
    mock_client_cls.return_value = mock_instance

    stt = DeepgramSTT(deepgram_config)
    with pytest.raises(STTError, match="Deepgram transcription failed"):
        stt.transcribe_bytes(b"audio")
