"""Speech-to-Text module using OpenAI Whisper."""

from __future__ import annotations

import io

from openai import OpenAI

from voice_agent.config import settings


class SpeechToText:
    """Transcribes audio bytes to text using OpenAI Whisper."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=settings.openai_api_key)

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe raw audio bytes and return the transcript as a string.

        Args:
            audio_bytes: Raw audio file content (wav, mp3, m4a, etc.).
            filename: Filename hint that tells Whisper the audio format.

        Returns:
            Transcribed text string.
        """
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        response = self._client.audio.transcriptions.create(
            model=settings.stt_model,
            file=audio_file,
            response_format="text",
        )
        # The API returns a str when response_format="text"
        return str(response).strip()
