"""RAG pipeline: document loading, chunking, embedding, and retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from voice_agent.config import settings


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline backed by a FAISS vector store."""

    def __init__(self, embeddings: OpenAIEmbeddings | None = None) -> None:
        self._embeddings = embeddings or OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._vectorstore: FAISS | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

    # ------------------------------------------------------------------
    # Building / loading the index
    # ------------------------------------------------------------------

    def build_index(self, data_dir: str | None = None) -> None:
        """Load knowledge-base files, embed them, and build a FAISS index.

        Supported file types inside *data_dir*:
        - ``*.json`` – list of objects with ``"question"``/``"answer"`` keys (FAQs)
          or any list of strings/dicts that will be serialised to text.
        - ``*.txt`` – plain text, split into chunks.

        Args:
            data_dir: Directory containing knowledge-base files.
                      Defaults to ``settings.data_dir``.
        """
        data_dir = data_dir or settings.data_dir
        docs: List[Document] = []

        for path in Path(data_dir).iterdir():
            if path.suffix == ".json":
                docs.extend(self._load_json(path))
            elif path.suffix == ".txt":
                docs.extend(self._load_txt(path))

        if not docs:
            raise ValueError(
                f"No documents found in '{data_dir}'. "
                "Add .json or .txt knowledge-base files."
            )

        chunks = self._splitter.split_documents(docs)
        self._vectorstore = FAISS.from_documents(chunks, self._embeddings)

    def save_index(self, path: str | None = None) -> None:
        """Persist the FAISS index to disk."""
        self._require_index()
        path = path or settings.faiss_index_path
        self._vectorstore.save_local(path)  # type: ignore[union-attr]

    def load_index(self, path: str | None = None) -> None:
        """Load a previously persisted FAISS index from disk."""
        path = path or settings.faiss_index_path
        self._vectorstore = FAISS.load_local(
            path,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def load_or_build_index(self, data_dir: str | None = None) -> None:
        """Load index from disk if it exists, otherwise build and save it."""
        index_path = settings.faiss_index_path
        if os.path.isdir(index_path):
            self.load_index(index_path)
        else:
            self.build_index(data_dir)
            self.save_index(index_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve the *k* most relevant documents for *query*.

        Args:
            query: The user's question or input text.
            k: Number of documents to retrieve.

        Returns:
            List of :class:`~langchain.schema.Document` objects.
        """
        self._require_index()
        return self._vectorstore.similarity_search(query, k=k)  # type: ignore[union-attr]

    def get_context(self, query: str, k: int = 4) -> str:
        """Return retrieved document content joined as a single context string.

        Args:
            query: The user's question.
            k: Number of documents to retrieve.

        Returns:
            Newline-separated context passages.
        """
        docs = self.retrieve(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_index(self) -> None:
        if self._vectorstore is None:
            raise RuntimeError(
                "Vector store is not initialised. "
                "Call build_index() or load_index() first."
            )

    @staticmethod
    def _load_json(path: Path) -> List[Document]:
        """Load a JSON file as :class:`~langchain.schema.Document` objects."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        docs: List[Document] = []
        source = path.name

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # FAQ-style: {"question": "...", "answer": "..."}
                    if "question" in item and "answer" in item:
                        content = f"Q: {item['question']}\nA: {item['answer']}"
                    else:
                        content = json.dumps(item, ensure_ascii=False)
                    metadata = {k: v for k, v in item.items() if isinstance(v, str)}
                    metadata["source"] = source
                    docs.append(Document(page_content=content, metadata=metadata))
                elif isinstance(item, str):
                    docs.append(
                        Document(page_content=item, metadata={"source": source})
                    )
        elif isinstance(data, dict):
            docs.append(
                Document(
                    page_content=json.dumps(data, ensure_ascii=False),
                    metadata={"source": source},
                )
            )
        return docs

    @staticmethod
    def _load_txt(path: Path) -> List[Document]:
        """Load a plain-text file as a single :class:`~langchain.schema.Document`."""
        text = path.read_text(encoding="utf-8").strip()
        return [Document(page_content=text, metadata={"source": path.name})]
