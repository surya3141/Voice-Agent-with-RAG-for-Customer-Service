"""Text-to-Speech module using OpenAI TTS."""

from __future__ import annotations

from openai import OpenAI

from voice_agent.config import settings


class TextToSpeech:
    """Converts text to audio bytes using OpenAI TTS."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=settings.openai_api_key)

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech and return raw audio bytes (MP3).

        Args:
            text: The text to synthesize.

        Returns:
            Raw MP3 audio bytes.
        """
        response = self._client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,  # type: ignore[arg-type]
            input=text,
            response_format="mp3",
        )
        return response.read()
