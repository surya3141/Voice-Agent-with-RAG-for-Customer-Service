"""RAG pipeline combining vector retrieval with LLM generation."""

import logging

from src.llm.groq_llm import GroqLLM
from src.rag.vector_store import KnowledgeStore

logger = logging.getLogger(__name__)

CUSTOMER_SERVICE_SYSTEM_PROMPT = (
    "You are a helpful and friendly customer service agent for TechPulse Electronics. "
    "Use only the provided context to answer the customer's question. If the context does "
    "not contain enough information to answer, let the customer know and offer to help in "
    "another way."
)

_ORDER_KEYWORDS = frozenset({
    "order", "tracking", "shipped", "delivery", "shipment", "package",
})
_APPOINTMENT_KEYWORDS = frozenset({
    "appointment", "schedule", "reschedule", "cancel appointment",
    "booking", "consultation", "demo", "repair appointment", "visit",
})


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for customer service.

    Combines semantic search over a KnowledgeStore with LLM generation.
    """

    def __init__(self, knowledge_store: KnowledgeStore, llm: GroqLLM) -> None:
        """Initialise the RAG pipeline.

        Args:
            knowledge_store: The vector store used for document retrieval.
            llm: The LLM client used for response generation.
        """
        self._store = knowledge_store
        self._llm = llm
        logger.info("RAG pipeline initialised")

    def query(self, user_query: str, context_type: str = "auto") -> dict:
        """Answer a user query using retrieval-augmented generation.

        Args:
            user_query: The customer's question or request.
            context_type: Collection to search. "auto" detects the intent.

        Returns:
            Dict with keys: answer, sources, intent.
        """
        intent = self.detect_intent(user_query)
        logger.info("Detected intent '%s' for query: %s", intent, user_query)

        if context_type == "auto":
            all_results = self._store.search_all(user_query)
            sources = []
            for collection_results in all_results.values():
                sources.extend(collection_results)
        else:
            sources = self._store.search(user_query, collection_name=context_type)

        context = self._build_context(sources)
        answer = self._llm.generate_with_context(
            query=user_query,
            context=context,
            system_prompt=CUSTOMER_SERVICE_SYSTEM_PROMPT,
        )
        logger.info("Generated answer for query (intent=%s, sources=%d)", intent, len(sources))
        return {"answer": answer, "sources": sources, "intent": intent}

    def detect_intent(self, query: str) -> str:
        """Detect the intent of a user query using keyword matching.

        Args:
            query: The user's question or request.

        Returns:
            One of "order_status", "appointment", or "faq".
        """
        query_lower = query.lower()
        cleaned = query_lower.replace("?", "").replace("!", "").replace(".", "").replace(",", "")
        tokens = set(cleaned.split())
        if tokens & _ORDER_KEYWORDS:
            return "order_status"
        for phrase in _APPOINTMENT_KEYWORDS:
            if " " in phrase:
                if phrase in query_lower:
                    return "appointment"
            elif phrase in tokens:
                return "appointment"
        return "faq"

    @staticmethod
    def _build_context(sources: list) -> str:
        """Build a context string from a list of retrieved source documents."""
        if not sources:
            return "No relevant context found."
        parts = []
        for idx, source in enumerate(sources, start=1):
            parts.append("[" + str(idx) + "] " + source.get("document", ""))
        return "\n\n".join(parts)
