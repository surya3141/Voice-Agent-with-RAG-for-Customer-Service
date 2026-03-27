"""Unit tests for the RAGPipeline module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from langchain.schema import Document
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 16


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing (no API calls)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(i % 100) * 0.01] * _EMBEDDING_DIM for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [0.01] * _EMBEDDING_DIM


def _make_fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


def _write_faq_json(directory: str) -> None:
    faqs = [
        {"question": "What are your hours?", "answer": "9 AM to 5 PM."},
        {"question": "How do I return an item?", "answer": "Within 30 days."},
    ]
    with open(os.path.join(directory, "faqs.json"), "w") as f:
        json.dump(faqs, f)


def _write_txt(directory: str) -> None:
    Path(os.path.join(directory, "extra.txt")).write_text(
        "We offer free shipping on orders over $50.", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRAGPipelineLoading:
    def test_build_index_from_json(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_faq_json(tmpdir)
            pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
            # Should not raise
            pipeline.build_index(data_dir=tmpdir)

    def test_build_index_from_txt(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_txt(tmpdir)
            pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
            pipeline.build_index(data_dir=tmpdir)

    def test_build_index_empty_dir_raises(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
            with pytest.raises(ValueError, match="No documents found"):
                pipeline.build_index(data_dir=tmpdir)

    def test_retrieve_without_index_raises(self):
        from voice_agent.rag import RAGPipeline

        pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
        with pytest.raises(RuntimeError, match="not initialised"):
            pipeline.retrieve("query")

    def test_get_context_without_index_raises(self):
        from voice_agent.rag import RAGPipeline

        pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
        with pytest.raises(RuntimeError, match="not initialised"):
            pipeline.get_context("query")

    def test_retrieve_returns_documents(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_faq_json(tmpdir)
            pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
            pipeline.build_index(data_dir=tmpdir)
            docs = pipeline.retrieve("hours", k=1)
            assert isinstance(docs, list)
            assert len(docs) >= 1
            assert isinstance(docs[0], Document)

    def test_get_context_returns_string(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_faq_json(tmpdir)
            pipeline = RAGPipeline(embeddings=_make_fake_embeddings())
            pipeline.build_index(data_dir=tmpdir)
            context = pipeline.get_context("return policy")
            assert isinstance(context, str)
            assert len(context) > 0

    def test_save_and_load_index(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as data_dir:
            with tempfile.TemporaryDirectory() as index_dir:
                _write_faq_json(data_dir)
                embeddings = _make_fake_embeddings()
                pipeline = RAGPipeline(embeddings=embeddings)
                pipeline.build_index(data_dir=data_dir)
                pipeline.save_index(path=index_dir)

                # Load into a fresh pipeline
                pipeline2 = RAGPipeline(embeddings=embeddings)
                pipeline2.load_index(path=index_dir)
                docs = pipeline2.retrieve("hours", k=1)
                assert len(docs) >= 1


class TestRAGDocumentLoading:
    def test_load_json_faq_format(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text(
                json.dumps(
                    [{"question": "Q?", "answer": "A."}]
                ),
                encoding="utf-8",
            )
            docs = RAGPipeline._load_json(path)
            assert len(docs) == 1
            assert "Q: Q?" in docs[0].page_content
            assert "A: A." in docs[0].page_content

    def test_load_json_string_list(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text(json.dumps(["Item one.", "Item two."]), encoding="utf-8")
            docs = RAGPipeline._load_json(path)
            assert len(docs) == 2

    def test_load_json_dict(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text(json.dumps({"key": "value"}), encoding="utf-8")
            docs = RAGPipeline._load_json(path)
            assert len(docs) == 1

    def test_load_txt(self):
        from voice_agent.rag import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "info.txt"
            path.write_text("Some plain text.", encoding="utf-8")
            docs = RAGPipeline._load_txt(path)
            assert len(docs) == 1
            assert docs[0].page_content == "Some plain text."
