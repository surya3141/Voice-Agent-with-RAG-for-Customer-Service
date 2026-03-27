"""Text-to-Speech module using ElevenLabs SDK."""

import logging
from pathlib import Path
from typing import List

from elevenlabs import ElevenLabs

from src.config import ElevenLabsConfig

logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 5000


class TTSError(Exception):
    """Custom exception for Text-to-Speech errors."""


class ElevenLabsTTS:
    """Text-to-Speech synthesis using the ElevenLabs API.

    Args:
        config: ElevenLabs configuration containing API key and voice settings.
    """

    def __init__(self, config: ElevenLabsConfig) -> None:
        if not config.api_key:
            raise TTSError("ElevenLabs API key is required")
        self._config = config
        self._client = ElevenLabs(api_key=config.api_key)
        self.call_count = 0
        self.total_characters_synthesized = 0
        logger.info("ElevenLabsTTS initialized with voice_id=%s, model_id=%s",
                     config.voice_id, config.model_id)

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Split text into chunks of at most MAX_CHUNK_SIZE characters.

        Splits on sentence boundaries (period, exclamation mark, question mark
        or space) when possible.
        """
        chunks = []
        remaining = text
        while remaining is not None and len(remaining) > MAX_CHUNK_SIZE:
            candidate = remaining[:MAX_CHUNK_SIZE]
            split_pos = -1
            for sep in (".", "!", "?"):
                pos = candidate.rfind(sep)
                if pos > split_pos:
                    split_pos = pos
            if split_pos == -1:
                split_pos = candidate.rfind(" ")
            if split_pos == -1:
                split_pos = MAX_CHUNK_SIZE
            else:
                split_pos += 1
            chunk = remaining[:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_pos:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _synthesize_chunk(self, text: str) -> bytes:
        """Synthesize a single text chunk via the ElevenLabs API.

        Args:
            text: Text to synthesize (must be <= MAX_CHUNK_SIZE).

        Returns:
            Audio bytes for the synthesized text.
        """
        response = self._client.text_to_speech.convert(
            voice_id=self._config.voice_id,
            text=text,
            model_id=self._config.model_id,
        )
        if isinstance(response, bytes):
            return response
        audio_parts = []
        for part in response:
            audio_parts.append(part)
        return b"".join(audio_parts)

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio bytes.

        Long texts (>5000 characters) are automatically chunked and
        synthesized in segments, then concatenated.

        Args:
            text: The text to convert to speech.

        Returns:
            Audio bytes.
        """
        if not text or not text.strip():
            raise TTSError("No text provided for synthesis")
        logger.info("Synthesizing %d characters of text", len(text))
        chunks = self._chunk_text(text)
        if len(chunks) > 1:
            logger.info("Text split into %d chunks for synthesis", len(chunks))
        audio_segments = []
        for i, chunk in enumerate(chunks):
            logger.debug("Synthesizing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            try:
                segment = self._synthesize_chunk(chunk)
                audio_segments.append(segment)
            except Exception as exc:
                raise TTSError("ElevenLabs synthesis failed: " + str(exc)) from exc
        audio_bytes = b"".join(audio_segments)
        self.call_count += 1
        self.total_characters_synthesized += len(text)
        logger.info("Synthesis complete: %d bytes of audio produced", len(audio_bytes))
        return audio_bytes

    def synthesize_to_file(self, text: str, output_path: str) -> None:
        """Convert text to speech and save to a file.

        Args:
            text: The text to convert to speech.
            output_path: Path where the audio file will be saved.
        """
        logger.info("Synthesizing text to file: %s", output_path)
        try:
            audio_bytes = self.synthesize(text)
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio_bytes)
            logger.info("Audio saved to %s (%d bytes)", str(path), len(audio_bytes))
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError("Failed to save audio to '" + str(output_path) + "': " + str(exc)) from exc
