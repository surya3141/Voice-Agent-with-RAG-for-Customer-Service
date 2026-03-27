"""LLM module: intent detection and response generation using OpenAI chat models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from voice_agent.config import settings

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------

INTENTS = [
    "faq",
    "order_status",
    "schedule_appointment",
    "cancel_appointment",
    "product_info",
    "complaint",
    "billing",
    "general",
]

_INTENT_DETECTION_SYSTEM = """\
You are an intent classifier for a customer-service voice agent.
Given the user's message, respond with EXACTLY ONE intent label from the list below \
and nothing else.

Intent labels:
- faq              : general frequently asked questions
- order_status     : asking about an order, shipment, or delivery
- schedule_appointment : wants to book or schedule an appointment
- cancel_appointment   : wants to cancel or reschedule an appointment
- product_info     : asking about product features, availability, or price
- complaint        : reporting a problem or expressing dissatisfaction
- billing          : questions about invoices, charges, refunds, or payments
- general          : anything else that does not fit the above

Respond with exactly one label, lowercase, no punctuation.\
"""

_RESPONSE_SYSTEM = """\
You are a friendly and professional customer-service voice agent.
Use the context passages provided to answer the customer's question accurately \
and concisely (2–4 sentences).
If the context does not contain the answer, politely say you will escalate the issue \
to a human agent.
Always be empathetic and helpful.\
"""


class LLMClient:
    """Wraps OpenAI chat completions for intent detection and response generation."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_intent(self, user_message: str) -> str:
        """Classify the user's message into one of the predefined intents.

        Args:
            user_message: Raw transcription of what the user said.

        Returns:
            Intent label string (e.g. ``"order_status"``).
        """
        response: ChatCompletion = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _INTENT_DETECTION_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=20,
        )
        label = response.choices[0].message.content or "general"
        label = label.strip().lower()
        # Validate against known intents; fall back to "general"
        return label if label in INTENTS else "general"

    def generate_response(self, user_message: str, context: str) -> str:
        """Generate a natural-language response grounded in *context*.

        Args:
            user_message: The user's question or statement.
            context: Retrieved knowledge-base passages relevant to the question.

        Returns:
            Agent reply as a plain string.
        """
        user_content = (
            f"Context:\n{context}\n\nCustomer question: {user_message}"
            if context.strip()
            else f"Customer question: {user_message}"
        )

        response: ChatCompletion = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _RESPONSE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=256,
        )
        return (response.choices[0].message.content or "").strip()
