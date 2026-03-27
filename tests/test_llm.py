"""Unit tests for the LLMClient module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voice_agent.llm import INTENTS, LLMClient


def _make_llm(response_text: str) -> LLMClient:
    """Return an LLMClient backed by a mock OpenAI client."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create.return_value = completion
    return LLMClient(client=client)


class TestDetectIntent:
    def test_known_intent_returned(self):
        llm = _make_llm("order_status")
        assert llm.detect_intent("Where is my order?") == "order_status"

    def test_unknown_intent_falls_back_to_general(self):
        llm = _make_llm("unknown_xyz")
        assert llm.detect_intent("Something weird") == "general"

    def test_intent_whitespace_stripped(self):
        llm = _make_llm("  faq  ")
        assert llm.detect_intent("What are your hours?") == "faq"

    def test_intent_case_normalised(self):
        llm = _make_llm("FAQ")
        assert llm.detect_intent("What are your hours?") == "faq"

    def test_none_response_falls_back_to_general(self):
        llm = _make_llm(None)
        assert llm.detect_intent("Hello") == "general"

    @pytest.mark.parametrize("intent", INTENTS)
    def test_all_known_intents_accepted(self, intent: str):
        llm = _make_llm(intent)
        assert llm.detect_intent("some query") == intent


class TestGenerateResponse:
    def test_returns_string(self):
        llm = _make_llm("Your order is on its way!")
        result = llm.generate_response("Where is my order?", "Order #123 is shipped.")
        assert isinstance(result, str)
        assert result == "Your order is on its way!"

    def test_empty_context_still_works(self):
        llm = _make_llm("I can help you with that.")
        result = llm.generate_response("What time do you open?", "")
        assert result == "I can help you with that."

    def test_whitespace_stripped_from_reply(self):
        llm = _make_llm("  Reply.  ")
        result = llm.generate_response("question", "context")
        assert result == "Reply."
