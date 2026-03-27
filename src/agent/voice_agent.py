"""Main voice agent orchestrator.

Coordinates STT, TTS, RAG pipeline, LLM, and knowledge store
components to process audio and text customer service queries.
"""

import logging

from src.config import AppConfig
from src.stt.deepgram_stt import DeepgramSTT
from src.tts.elevenlabs_tts import ElevenLabsTTS
from src.llm.groq_llm import GroqLLM
from src.rag.vector_store import KnowledgeStore
from src.rag.pipeline import RAGPipeline
from src.security.data_masking import DataMasker

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Voice agent that orchestrates STT, TTS, RAG, and LLM components.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig = None) -> None:
        self._config = config or AppConfig()
        self._stt = None
        self._tts = None
        self._llm = None
        self._knowledge_store = None
        self._rag_pipeline = None
        self._data_masker = DataMasker()
        logger.info("VoiceAgent created (components not yet initialised)")

    def initialize(self) -> None:
        """Initialise all sub-components and load knowledge data.

        Creates STT, TTS, LLM, KnowledgeStore and RAGPipeline instances,
        then loads FAQ, order, and appointment data from config.data_dir.
        """
        logger.info("Initialising VoiceAgent components")
        self._stt = DeepgramSTT(self._config.deepgram)
        self._tts = ElevenLabsTTS(self._config.elevenlabs)
        self._llm = GroqLLM(self._config.groq)
        self._knowledge_store = KnowledgeStore(persist_directory=str(self._config.chroma_dir))
        self._rag_pipeline = RAGPipeline(self._knowledge_store, self._llm)
        self._load_data(self._config.data_dir)
        logger.info("VoiceAgent fully initialised")

    def _load_data(self, data_dir) -> None:
        """Load knowledge-base data files from *data_dir*."""
        faqs_path = data_dir / "faqs.json"
        orders_path = data_dir / "orders.json"
        appointments_path = data_dir / "appointments.json"
        for label, path, loader in [
            ("FAQs", faqs_path, self._knowledge_store.load_faqs),
            ("orders", orders_path, self._knowledge_store.load_orders),
            ("appointments", appointments_path, self._knowledge_store.load_appointments),
        ]:
            if path.is_file():
                try:
                    count = loader(str(path))
                    logger.info("Loaded %d %s documents", count, label)
                except Exception as exc:
                    logger.error("Failed to load %s from %s: %s", label, path, exc)
            else:
                logger.warning("Data file not found: %s", path)

    def process_audio(self, audio_data: bytes, mimetype: str = "audio/wav") -> dict:
        """Process an audio query end-to-end.

        Steps:
            1. Transcribe audio via STT.
            2. Mask sensitive data in the transcript.
            3. Query the RAG pipeline.
        """
        if self._stt is None or self._rag_pipeline is None:
            raise RuntimeError("VoiceAgent has not been initialised. Call initialize() first.")
        try:
            stt_result = self._stt.transcribe_bytes(audio_data, mimetype=mimetype)
        except Exception as exc:
            logger.error("STT failed: %s", exc)
            return self._error_result("Speech-to-text failed: " + str(exc))

        transcript = stt_result.get("transcript", "").strip()
        stt_confidence = stt_result.get("confidence", 0.0)
        if not transcript:
            logger.warning("Empty transcript received from STT")
            return self._error_result("Could not understand audio input.")

        masked_transcript = self._data_masker.mask_text(transcript)
        try:
            rag_result = self._rag_pipeline.query(masked_transcript)
        except Exception as exc:
            logger.error("RAG pipeline failed: %s", exc)
            return self._error_result("Knowledge retrieval failed: " + str(exc))

        response_text = rag_result.get("answer", "")
        intent = rag_result.get("intent", "")
        sources = rag_result.get("sources", [])
        response_audio = self._synthesize_safely(response_text)
        return {
            "transcript": transcript,
            "masked_transcript": masked_transcript,
            "response_text": response_text,
            "response_audio": response_audio,
            "intent": intent,
            "sources": sources,
            "stt_confidence": stt_confidence,
        }

    def process_text(self, text: str) -> dict:
        """Process a text query (skips STT).

        Args:
            text: Customer query text.

        Returns:
            Dict with keys: response_text, response_audio, intent, sources.
        """
        if self._rag_pipeline is None:
            raise RuntimeError("VoiceAgent has not been initialised. Call initialize() first.")
        try:
            rag_result = self._rag_pipeline.query(text)
        except Exception as exc:
            logger.error("RAG pipeline failed: %s", exc)
            return {
                "response_text": "Knowledge retrieval failed: " + str(exc),
                "response_audio": None,
                "intent": "",
                "sources": [],
            }
        response_text = rag_result.get("answer", "")
        intent = rag_result.get("intent", "")
        sources = rag_result.get("sources", [])
        response_audio = self._synthesize_safely(response_text)
        return {
            "response_text": response_text,
            "response_audio": response_audio,
            "intent": intent,
            "sources": sources,
        }

    def get_usage_stats(self) -> dict:
        """Return aggregated API usage statistics from all components.

        Returns:
            Dict with per-service usage data.
        """
        stats = {}
        stats["stt"] = {
            "call_count": self._stt.call_count,
            "total_duration_seconds": self._stt.total_duration_seconds,
        }
        stats["tts"] = {
            "call_count": self._tts.call_count,
            "total_characters_synthesized": self._tts.total_characters_synthesized,
        }
        stats["llm"] = {
            "call_count": self._llm.call_count,
            "total_input_tokens": self._llm.total_input_tokens,
            "total_output_tokens": self._llm.total_output_tokens,
        }
        return stats

    def _synthesize_safely(self, text: str):
        """Attempt TTS synthesis, returning *None* on failure."""
        if self._tts is None or not text or not text.strip():
            return None
        try:
            return self._tts.synthesize(text)
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return None

    @staticmethod
    def _error_result(message: str) -> dict:
        """Build an error result dict for ``process_audio``."""
        return {
            "transcript": "",
            "masked_transcript": "",
            "response_text": message,
            "response_audio": None,
            "intent": "",
            "sources": [],
            "stt_confidence": 0.0,
        }
