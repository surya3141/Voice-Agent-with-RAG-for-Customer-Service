"""Speech-to-Text module using Deepgram SDK."""

import logging
from pathlib import Path

from deepgram import DeepgramClient, PrerecordedOptions, FileSource

from src.config import DeepgramConfig

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Custom exception for Speech-to-Text errors."""


class DeepgramSTT:
    """Speech-to-Text transcription using the Deepgram API.

    Args:
        config: Deepgram configuration containing API key and model settings.
    """

    def __init__(self, config: DeepgramConfig) -> None:
        if not config.api_key:
            raise STTError("Deepgram API key is required")
        self._config = config
        self._client = DeepgramClient(config.api_key)
        self.call_count = 0
        self.total_duration_seconds = 0.0
        logger.info("DeepgramSTT initialized with model=%s, language=%s",
                     config.model, config.language)

    def _build_options(self) -> PrerecordedOptions:
        """Build Deepgram transcription options from config."""
        return PrerecordedOptions(
            model=self._config.model,
            language=self._config.language,
            smart_format=True,
        )

    def _parse_response(self, response) -> dict:
        """Parse a Deepgram prerecorded response into a standardised dict.

        Returns:
            dict with keys: transcript, confidence, words, duration
        """
        try:
            result = response.results
            channel = result.channels[0]
            alternative = channel.alternatives[0]
            transcript = alternative.transcript
            confidence = alternative.confidence
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                for w in alternative.words
            ]
            duration = 0.0
            if hasattr(response, "metadata"):
                duration = response.metadata.duration
            self.call_count += 1
            self.total_duration_seconds += duration
            logger.info(
                "Transcription complete: %d words, duration=%.2fs, confidence=%.3f",
                len(words), duration, confidence,
            )
            return {
                "transcript": transcript,
                "confidence": confidence,
                "words": words,
                "duration": duration,
            }
        except (AttributeError, IndexError, TypeError) as exc:
            raise STTError("Failed to parse Deepgram response: " + str(exc)) from exc

    def transcribe_file(self, audio_path: str) -> dict:
        """Transcribe an audio file via Deepgram's prerecorded API.

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            dict with transcription results.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise STTError("Audio file not found: " + str(audio_path))
        logger.info("Transcribing file: %s", audio_path)
        audio_data = path.read_bytes()
        payload = {"buffer": audio_data}
        options = self._build_options()
        try:
            response = self._client.listen.rest.v("1").transcribe_file(payload, options)
            return self._parse_response(response)
        except Exception as exc:
            raise STTError("Deepgram transcription failed for file '" + str(audio_path) + "': " + str(exc)) from exc

    def transcribe_bytes(self, audio_data: bytes, mimetype: str = "audio/wav") -> dict:
        """Transcribe raw audio bytes via Deepgram's prerecorded API.

        Args:
            audio_data: Raw audio bytes to transcribe.
            mimetype: MIME type of the audio data.

        Returns:
            dict with transcription results.
        """
        if not audio_data:
            raise STTError("No audio data provided")
        logger.info("Transcribing %d bytes of %s audio", len(audio_data), mimetype)
        payload = {"buffer": audio_data, "mimetype": mimetype}
        options = self._build_options()
        try:
            response = self._client.listen.rest.v("1").transcribe_file(payload, options)
            return self._parse_response(response)
        except Exception as exc:
            raise STTError("Deepgram transcription failed: " + str(exc)) from exc
