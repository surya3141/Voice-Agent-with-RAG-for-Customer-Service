"""Vector store module using ChromaDB for knowledge retrieval."""

import json
import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""


class KnowledgeStore:
    """ChromaDB-backed vector store for customer service knowledge.

    Stores FAQs, orders, and appointments in separate collections.
    """

    def __init__(self, persist_directory: str = None) -> None:
        """Initialise the knowledge store.

        Args:
            persist_directory: Path for persistent storage. Uses ephemeral
                storage when None.
        """
        try:
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
                logger.info("Initialised persistent ChromaDB at %s", persist_directory)
            else:
                self._client = chromadb.Client()
                logger.info("Initialised ephemeral ChromaDB client")
        except chromadb.errors.ChromaError as exc:
            raise VectorStoreError("Failed to initialise ChromaDB client: " + str(exc)) from exc
        self._collection_names = []

    def load_faqs(self, faqs_path: str) -> int:
        """Load FAQ data from a JSON file into the 'faqs' collection.

        Args:
            faqs_path: Path to the FAQs JSON file.

        Returns:
            Number of FAQs loaded.
        """
        data = self._read_json(faqs_path)
        documents = []
        metadatas = []
        ids = []
        for faq in data:
            doc_text = "Question: " + faq["question"] + "\nAnswer: " + faq["answer"]
            documents.append(doc_text)
            metadatas.append({
                "id": str(faq.get("id", "")),
                "category": faq.get("category", "general"),
            })
            ids.append("faq-" + str(faq.get("id", "")))
        self._upsert_collection("faqs", documents, metadatas, ids)
        logger.info("Loaded %d FAQs from %s", len(documents), faqs_path)
        return len(documents)

    def load_orders(self, orders_path: str) -> int:
        """Load order data from a JSON file into the 'orders' collection.

        Args:
            orders_path: Path to the orders JSON file.

        Returns:
            Number of orders loaded.
        """
        data = self._read_json(orders_path)
        documents = []
        metadatas = []
        ids = []
        for order in data:
            items_summary = ", ".join(str(i) for i in order.get("items", []))
            doc_text = (
                "Order " + order.get("order_id", "") +
                " for " + order.get("customer_name", "") +
                " (" + order.get("customer_email", "") +
                "). Items: " + items_summary +
                ". Total: $" + str(order.get("total", "")) +
                ". Status: " + order.get("status", "") + "."
            )
            if order.get("tracking_number"):
                doc_text += " Tracking: " + order["tracking_number"]
            if order.get("estimated_delivery"):
                doc_text += " Estimated delivery: " + order["estimated_delivery"]
            documents.append(doc_text)
            metadatas.append({
                "order_id": order.get("order_id", ""),
                "status": order.get("status", ""),
                "customer_name": order.get("customer_name", ""),
                "customer_email": order.get("customer_email", ""),
            })
            ids.append("order-" + order.get("order_id", ""))
        self._upsert_collection("orders", documents, metadatas, ids)
        logger.info("Loaded %d orders from %s", len(documents), orders_path)
        return len(documents)

    def load_appointments(self, appointments_path: str) -> int:
        """Load appointment data from a JSON file into the 'appointments' collection.

        Args:
            appointments_path: Path to the appointments JSON file.

        Returns:
            Number of appointments loaded.
        """
        data = self._read_json(appointments_path)
        documents = []
        metadatas = []
        ids = []
        for appt in data:
            doc_text = (
                "Appointment " + appt.get("appointment_id", "") +
                " for " + appt.get("customer_name", "") +
                " (" + appt.get("customer_email", "") +
                "). Service: " + appt.get("service_type", "") +
                ". Scheduled: " + appt.get("scheduled_date", "") +
                " at " + appt.get("scheduled_time", "") +
                " (" + str(appt.get("duration_minutes", "")) +
                " minutes). Status: " + appt.get("status", "") + "."
            )
            if appt.get("notes"):
                doc_text += " Notes: " + appt["notes"]
            documents.append(doc_text)
            metadatas.append({
                "appointment_id": appt.get("appointment_id", ""),
                "status": appt.get("status", ""),
                "customer_name": appt.get("customer_name", ""),
                "customer_email": appt.get("customer_email", ""),
            })
            ids.append("appt-" + appt.get("appointment_id", ""))
        self._upsert_collection("appointments", documents, metadatas, ids)
        logger.info("Loaded %d appointments from %s", len(documents), appointments_path)
        return len(documents)

    def search(self, query: str, collection_name: str = "faqs", n_results: int = 3) -> list:
        """Search a specific collection for documents matching the query.

        Args:
            query: The search query text.
            collection_name: Name of the collection to search.
            n_results: Maximum number of results to return.

        Returns:
            A list of dicts with keys: document, metadata, distance.
        """
        try:
            collection = self._client.get_collection(collection_name)
            results = collection.query(query_texts=[query], n_results=n_results)
        except chromadb.errors.ChromaError as exc:
            raise VectorStoreError("Search failed on collection '" + collection_name + "': " + str(exc)) from exc
        except ValueError as exc:
            raise VectorStoreError("Collection '" + collection_name + "' not found: " + str(exc)) from exc

        output = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append({"document": doc, "metadata": meta, "distance": dist})
        logger.debug("Search '%s' in '%s' returned %d results", query, collection_name, len(output))
        return output

    def search_all(self, query: str, n_results: int = 3) -> dict:
        """Search all loaded collections for documents matching the query.

        Args:
            query: The search query text.
            n_results: Maximum number of results to return per collection.

        Returns:
            Dict mapping collection names to lists of results.
        """
        results = {}
        for name in self._collection_names:
            try:
                results[name] = self.search(query, name, n_results)
            except VectorStoreError:
                logger.warning("Skipping collection '%s' during search_all", name)
        return results

    def _read_json(self, path: str) -> list:
        """Read and parse a JSON file.

        Raises:
            VectorStoreError: If reading or parsing fails.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise VectorStoreError("File not found: " + str(path))
        try:
            with open(file_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise VectorStoreError("Invalid JSON in " + str(path) + ": " + str(exc)) from exc
        except OSError as exc:
            raise VectorStoreError("Cannot read file " + str(path) + ": " + str(exc)) from exc
        if not isinstance(data, list):
            raise VectorStoreError("Expected a JSON array in " + str(path))
        return data

    def _upsert_collection(self, name: str, documents: list, metadatas: list, ids: list) -> None:
        """Create or update a collection with the given documents.

        Raises:
            VectorStoreError: If the upsert fails.
        """
        try:
            collection = self._client.get_or_create_collection(name)
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            if name not in self._collection_names:
                self._collection_names.append(name)
        except chromadb.errors.ChromaError as exc:
            raise VectorStoreError("Failed to upsert collection '" + name + "': " + str(exc)) from exc
