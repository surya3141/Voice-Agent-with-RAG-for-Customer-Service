"""Tests for the ElevenLabsTTS text-to-speech module."""

import pytest
from unittest.mock import patch, MagicMock

from src.config import ElevenLabsConfig
from src.tts.elevenlabs_tts import ElevenLabsTTS, TTSError, MAX_CHUNK_SIZE


@pytest.fixture
def elevenlabs_config():
    """Return an ElevenLabsConfig with a dummy API key."""
    return ElevenLabsConfig(
        api_key="test-api-key",
        voice_id="test-voice",
        model_id="eleven_monolingual_v1",
    )


def test_tts_error_raised_without_api_key():
    """TTSError is raised when no API key is configured."""
    config = ElevenLabsConfig(api_key="")
    with pytest.raises(TTSError, match="ElevenLabs API key is required"):
        ElevenLabsTTS(config)


@patch("src.tts.elevenlabs_tts.ElevenLabs")
def test_synthesize_returns_audio_bytes(mock_eleven_cls, elevenlabs_config):
    """synthesize returns audio bytes from the mocked client."""
    mock_instance = MagicMock()
    mock_instance.text_to_speech.convert.return_value = b"fake-audio-data"
    mock_eleven_cls.return_value = mock_instance

    tts = ElevenLabsTTS(elevenlabs_config)
    result = tts.synthesize("Hello world")

    assert isinstance(result, bytes)
    assert result == b"fake-audio-data"


@patch("src.tts.elevenlabs_tts.ElevenLabs")
def test_synthesize_to_file_creates_file(mock_eleven_cls, elevenlabs_config, tmp_path):
    """synthesize_to_file writes audio bytes to the specified path."""
    mock_instance = MagicMock()
    mock_instance.text_to_speech.convert.return_value = b"audio-bytes"
    mock_eleven_cls.return_value = mock_instance

    tts = ElevenLabsTTS(elevenlabs_config)
    output_file = tmp_path / "output.mp3"
    tts.synthesize_to_file("Test text", str(output_file))

    assert output_file.exists()
    assert output_file.read_bytes() == b"audio-bytes"


@patch("src.tts.elevenlabs_tts.ElevenLabs")
def test_call_count_and_characters_tracked(mock_eleven_cls, elevenlabs_config):
    """call_count and total_characters_synthesized are tracked correctly."""
    mock_instance = MagicMock()
    mock_instance.text_to_speech.convert.return_value = b"audio"
    mock_eleven_cls.return_value = mock_instance

    tts = ElevenLabsTTS(elevenlabs_config)
    assert tts.call_count == 0
    assert tts.total_characters_synthesized == 0

    tts.synthesize("Hello")  # 5 chars
    assert tts.call_count == 1
    assert tts.total_characters_synthesized == 5

    tts.synthesize("World!")  # 6 chars
    assert tts.call_count == 2
    assert tts.total_characters_synthesized == 11


@patch("src.tts.elevenlabs_tts.ElevenLabs")
def test_text_chunking_for_long_text(mock_eleven_cls, elevenlabs_config):
    """Long text (>5000 chars) is split into chunks before synthesis."""
    mock_instance = MagicMock()
    mock_instance.text_to_speech.convert.return_value = b"chunk-audio"
    mock_eleven_cls.return_value = mock_instance

    tts = ElevenLabsTTS(elevenlabs_config)
    long_text = "A" * (MAX_CHUNK_SIZE + 100)
    tts.synthesize(long_text)

    # The text should have been split so convert is called more than once
    assert mock_instance.text_to_speech.convert.call_count >= 2


@patch("src.tts.elevenlabs_tts.ElevenLabs")
def test_tts_error_on_synthesis_failure(mock_eleven_cls, elevenlabs_config):
    """TTSError is raised when the ElevenLabs client throws an exception."""
    mock_instance = MagicMock()
    mock_instance.text_to_speech.convert.side_effect = RuntimeError("Service unavailable")
    mock_eleven_cls.return_value = mock_instance

    tts = ElevenLabsTTS(elevenlabs_config)
    with pytest.raises(TTSError, match="ElevenLabs synthesis failed"):
        tts.synthesize("Hello")
