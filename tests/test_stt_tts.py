"""Unit tests for SpeechToText and TextToSpeech modules."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from voice_agent.stt import SpeechToText
from voice_agent.tts import TextToSpeech


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------


class TestSpeechToText:
    def _make_stt(self, transcript: str) -> SpeechToText:
        client = MagicMock()
        client.audio.transcriptions.create.return_value = transcript
        return SpeechToText(client=client)

    def test_transcribe_returns_text(self):
        stt = self._make_stt("Hello, I need help with my order.")
        result = stt.transcribe(b"fake-audio-bytes")
        assert result == "Hello, I need help with my order."

    def test_transcribe_strips_whitespace(self):
        stt = self._make_stt("  Hello world.  ")
        result = stt.transcribe(b"fake-audio-bytes")
        assert result == "Hello world."

    def test_transcribe_passes_filename(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = "text"
        stt = SpeechToText(client=client)
        stt.transcribe(b"bytes", filename="recording.mp3")

        call_kwargs = client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["file"].name == "recording.mp3"

    def test_transcribe_empty_response(self):
        stt = self._make_stt("")
        result = stt.transcribe(b"bytes")
        assert result == ""


# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------


class TestTextToSpeech:
    def _make_tts(self, audio_bytes: bytes = b"fake-mp3") -> TextToSpeech:
        client = MagicMock()
        response = MagicMock()
        response.read.return_value = audio_bytes
        client.audio.speech.create.return_value = response
        return TextToSpeech(client=client)

    def test_synthesize_returns_bytes(self):
        tts = self._make_tts(b"mp3-data")
        result = tts.synthesize("Hello, how can I help you?")
        assert result == b"mp3-data"

    def test_synthesize_calls_api_with_text(self):
        client = MagicMock()
        response = MagicMock()
        response.read.return_value = b"audio"
        client.audio.speech.create.return_value = response
        tts = TextToSpeech(client=client)

        tts.synthesize("Test message")
        call_kwargs = client.audio.speech.create.call_args
        assert call_kwargs.kwargs["input"] == "Test message"
        assert call_kwargs.kwargs["response_format"] == "mp3"
