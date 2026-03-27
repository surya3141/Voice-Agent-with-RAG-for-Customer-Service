"""Evaluation and testing pipeline for the TechPulse RAG system.

Provides systematic evaluation of FAQ, order, and appointment query handling
by running test cases through the RAG pipeline and measuring intent accuracy.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Run evaluation test cases against a RAG pipeline and report accuracy."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        """Initialise the evaluation pipeline.

        Args:
            pipeline: A fully initialised ``RAGPipeline`` instance.
        """
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # Core evaluation runner
    # ------------------------------------------------------------------
    def _run_evaluation(
        self, test_cases: list[dict[str, str]], category_label: str
    ) -> dict[str, object]:
        """Run a batch of test cases and measure intent-detection accuracy.

        Args:
            test_cases: List of dicts, each with ``"query"`` and
                ``"expected_category"`` keys.
            category_label: Human-readable label for the test category.

        Returns:
            Dict containing ``total_cases``, ``correct_intents``,
            ``intent_accuracy``, and ``results`` (per-case details).
        """
        results: list[dict[str, object]] = []
        correct = 0

        for i, case in enumerate(test_cases, 1):
            query: str = case["query"]
            expected: str = case["expected_category"]

            logger.info(
                "[%s %d/%d] Query: %s",
                category_label,
                i,
                len(test_cases),
                query,
            )

            try:
                response = self.pipeline.query(query)
                detected_intent: str = response.get("intent", "")
                answer: str = response.get("answer", "")
                sources: list[dict[str, object]] = response.get("sources", [])
                is_correct = detected_intent == expected

                if is_correct:
                    correct += 1

                result: dict[str, object] = {
                    "query": query,
                    "expected_intent": expected,
                    "detected_intent": detected_intent,
                    "correct": is_correct,
                    "answer_preview": answer[:200],
                    "num_sources": len(sources),
                }
                logger.info(
                    "  → intent=%s (expected=%s) correct=%s",
                    detected_intent,
                    expected,
                    is_correct,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("  → Error processing query: %s", exc)
                result = {
                    "query": query,
                    "expected_intent": expected,
                    "detected_intent": "error",
                    "correct": False,
                    "error": str(exc),
                    "num_sources": 0,
                }

            results.append(result)

        total = len(test_cases)
        accuracy = (correct / total * 100.0) if total else 0.0

        return {
            "total_cases": total,
            "correct_intents": correct,
            "intent_accuracy": accuracy,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Category-specific evaluations
    # ------------------------------------------------------------------
    def run_faq_evaluation(self, test_cases: list[dict[str, str]]) -> dict[str, object]:
        """Evaluate FAQ query handling.

        Args:
            test_cases: List of dicts with ``"query"`` and
                ``"expected_category"`` keys.

        Returns:
            Evaluation results dict with accuracy metrics and per-case details.
        """
        logger.info("Starting FAQ evaluation with %d cases", len(test_cases))
        return self._run_evaluation(test_cases, "FAQ")

    def run_order_evaluation(self, test_cases: list[dict[str, str]]) -> dict[str, object]:
        """Evaluate order-related query handling.

        Args:
            test_cases: List of dicts with ``"query"`` and
                ``"expected_category"`` keys.

        Returns:
            Evaluation results dict with accuracy metrics and per-case details.
        """
        logger.info("Starting order evaluation with %d cases", len(test_cases))
        return self._run_evaluation(test_cases, "Order")

    def run_appointment_evaluation(
        self, test_cases: list[dict[str, str]]
    ) -> dict[str, object]:
        """Evaluate appointment-related query handling.

        Args:
            test_cases: List of dicts with ``"query"`` and
                ``"expected_category"`` keys.

        Returns:
            Evaluation results dict with accuracy metrics and per-case details.
        """
        logger.info("Starting appointment evaluation with %d cases", len(test_cases))
        return self._run_evaluation(test_cases, "Appointment")

    # ------------------------------------------------------------------
    # Default test cases
    # ------------------------------------------------------------------
    @staticmethod
    def get_default_test_cases() -> dict[str, list[dict[str, str]]]:
        """Return default test cases for all categories.

        Returns:
            Dict with keys ``"faq"``, ``"order"``, ``"appointment"``, each
            containing a list of test-case dicts.
        """
        return {
            "faq": [
                {"query": "What is your return policy?", "expected_category": "faq"},
                {"query": "How do I track my order?", "expected_category": "faq"},
                {"query": "What payment methods do you accept?", "expected_category": "faq"},
                {"query": "Do you offer a warranty on your products?", "expected_category": "faq"},
                {"query": "What are your business hours?", "expected_category": "faq"},
                {"query": "How do I contact customer support?", "expected_category": "faq"},
                {"query": "What is your privacy policy?", "expected_category": "faq"},
            ],
            "order": [
                {"query": "What is the status of order ORD-10001?", "expected_category": "order_status"},
                {"query": "Where is my package?", "expected_category": "order_status"},
                {"query": "Has my order been shipped yet?", "expected_category": "order_status"},
                {"query": "I need tracking information for my delivery", "expected_category": "order_status"},
                {"query": "When will my order ORD-10003 arrive?", "expected_category": "order_status"},
                {"query": "Can I check my shipment status?", "expected_category": "order_status"},
            ],
            "appointment": [
                {"query": "Can I schedule a repair?", "expected_category": "appointment"},
                {"query": "When is my next appointment?", "expected_category": "appointment"},
                {"query": "I need to book a tech support session", "expected_category": "appointment"},
                {"query": "How do I reschedule my appointment?", "expected_category": "appointment"},
                {"query": "I want to cancel my booking", "expected_category": "appointment"},
                {"query": "Is there an available slot for a consultation?", "expected_category": "appointment"},
            ],
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def run_full_evaluation(self) -> dict[str, object]:
        """Run all evaluations using the default test cases.

        Returns:
            Combined results dict with keys ``"faq"``, ``"order"``,
            ``"appointment"``, ``"overall_accuracy"``, and ``"timestamp"``.
        """
        test_cases = self.get_default_test_cases()

        logger.info("Running full evaluation suite")

        faq_results = self.run_faq_evaluation(test_cases["faq"])
        order_results = self.run_order_evaluation(test_cases["order"])
        appointment_results = self.run_appointment_evaluation(test_cases["appointment"])

        total_cases = (
            faq_results["total_cases"]
            + order_results["total_cases"]
            + appointment_results["total_cases"]
        )
        total_correct = (
            faq_results["correct_intents"]
            + order_results["correct_intents"]
            + appointment_results["correct_intents"]
        )
        overall_accuracy = (total_correct / total_cases * 100.0) if total_cases else 0.0

        return {
            "faq": faq_results,
            "order": order_results,
            "appointment": appointment_results,
            "overall_accuracy": overall_accuracy,
            "total_cases": total_cases,
            "total_correct": total_correct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    @staticmethod
    def generate_report(results: dict[str, object]) -> str:
        """Generate a human-readable text report from evaluation results.

        Args:
            results: Combined results dict produced by ``run_full_evaluation``.

        Returns:
            Multi-line string report.
        """
        lines: list[str] = [
            "=" * 60,
            "  TechPulse RAG Evaluation Report",
            "=" * 60,
            f"  Timestamp : {results.get('timestamp', 'N/A')}",
            f"  Total Cases : {results.get('total_cases', 0)}",
            f"  Total Correct : {results.get('total_correct', 0)}",
            f"  Overall Accuracy : {results.get('overall_accuracy', 0):.1f}%",
            "-" * 60,
        ]

        for category in ("faq", "order", "appointment"):
            cat_results = results.get(category)
            if not isinstance(cat_results, dict):
                continue
            lines.append(f"\n  [{category.upper()}]")
            lines.append(
                f"    Cases : {cat_results.get('total_cases', 0)}  |  "
                f"Correct : {cat_results.get('correct_intents', 0)}  |  "
                f"Accuracy : {cat_results.get('intent_accuracy', 0):.1f}%"
            )

            per_case: list[dict[str, object]] = cat_results.get("results", [])
            for r in per_case:
                status = "✓" if r.get("correct") else "✗"
                lines.append(
                    f"    {status} Q: {r.get('query', '')}"
                    f"  → {r.get('detected_intent', '')} "
                    f"(expected {r.get('expected_intent', '')})"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.config import get_config, DATA_DIR
    from src.llm.groq_llm import GroqLLM, LLMError
    from src.rag.vector_store import KnowledgeStore, VectorStoreError

    config = get_config()

    if not config.groq.api_key:
        logger.error("GROQ_API_KEY is not set. Cannot run evaluation.")
        sys.exit(1)

    try:
        store = KnowledgeStore()
        store.load_faqs(str(DATA_DIR / "faqs.json"))
        store.load_orders(str(DATA_DIR / "orders.json"))
        store.load_appointments(str(DATA_DIR / "appointments.json"))
    except (VectorStoreError, FileNotFoundError) as exc:
        logger.error("Failed to load knowledge store: %s", exc)
        sys.exit(1)

    try:
        llm = GroqLLM(config.groq)
    except LLMError as exc:
        logger.error("Failed to initialise LLM: %s", exc)
        sys.exit(1)

    pipeline = RAGPipeline(knowledge_store=store, llm=llm)
    evaluator = EvaluationPipeline(pipeline)

    eval_results = evaluator.run_full_evaluation()
    report = evaluator.generate_report(eval_results)
    print(report)
