"""Tests for the KnowledgeStore and RAGPipeline modules."""

import os
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from src.rag.vector_store import KnowledgeStore, VectorStoreError
from src.rag.pipeline import RAGPipeline

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
FAQS_PATH = os.path.join(PROJECT_ROOT, "data", "faqs.json")


@pytest.fixture
def knowledge_store():
    """Create an ephemeral KnowledgeStore for testing."""
    return KnowledgeStore()


@pytest.fixture
def loaded_store(knowledge_store):
    """Return a KnowledgeStore with FAQs loaded."""
    knowledge_store.load_faqs(FAQS_PATH)
    return knowledge_store


def test_knowledge_store_initialisation():
    """KnowledgeStore can be initialised in ephemeral mode."""
    store = KnowledgeStore()
    assert store._client is not None
    assert store._collection_names == []


def test_load_faqs_correct_count(knowledge_store):
    """load_faqs loads the correct number of FAQ documents."""
    count = knowledge_store.load_faqs(FAQS_PATH)
    assert count == 18


def test_search_returns_results_with_expected_keys(loaded_store):
    """search returns results with document, metadata, and distance keys."""
    results = loaded_store.search("return policy", collection_name="faqs", n_results=3)
    assert len(results) > 0
    for result in results:
        assert "document" in result
        assert "metadata" in result
        assert "distance" in result


def test_search_all_returns_results_for_loaded_collections(loaded_store):
    """search_all returns a dict with results for each loaded collection."""
    results = loaded_store.search_all("return policy")
    assert isinstance(results, dict)
    assert "faqs" in results
    assert len(results["faqs"]) > 0


def test_search_all_multiple_collections(knowledge_store):
    """search_all covers all loaded collections."""
    knowledge_store.load_faqs(FAQS_PATH)
    orders_path = os.path.join(PROJECT_ROOT, "data", "orders.json")
    if os.path.exists(orders_path):
        knowledge_store.load_orders(orders_path)
    results = knowledge_store.search_all("order status")
    assert "faqs" in results


# --- RAGPipeline tests ---

@pytest.fixture
def rag_pipeline(loaded_store):
    """Create a RAGPipeline with a loaded store and mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.generate_with_context.return_value = "Mocked LLM answer"
    return RAGPipeline(loaded_store, mock_llm)


@pytest.mark.parametrize("query,expected_intent", [
    ("What is your return policy?", "faq"),
    ("Where is my order ORD-10001?", "order_status"),
    ("Schedule an appointment", "appointment"),
    ("Hello", "faq"),
])
def test_detect_intent(rag_pipeline, query, expected_intent):
    """detect_intent correctly classifies queries by intent."""
    assert rag_pipeline.detect_intent(query) == expected_intent


def test_rag_query_with_mocked_llm(rag_pipeline):
    """RAGPipeline.query returns an answer from the mocked LLM."""
    result = rag_pipeline.query("What is the return policy?")
    assert "answer" in result
    assert result["answer"] == "Mocked LLM answer"
    assert "sources" in result
    assert "intent" in result
