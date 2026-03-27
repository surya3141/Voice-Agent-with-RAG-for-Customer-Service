"""API cost tracking and ROI analysis module.

Tracks per-service usage costs for Deepgram (STT), Groq (LLM),
ElevenLabs (TTS), and Twilio (telephony), and provides alerts
when spending exceeds configurable thresholds.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class APIUsageRecord:
    """A single API usage record.

    Attributes:
        service: The service name (e.g. ``"deepgram"``).
        operation: Short description of the operation.
        units: Number of billable units consumed.
        cost: Dollar cost for this record.
        timestamp: When the usage occurred (UTC).
    """

    service: str
    operation: str
    units: float
    cost: float
    timestamp: datetime


class CostTracker:
    """Tracks API usage costs and generates ROI analysis.

    Args:
        alert_threshold: Dollar amount that triggers a spending alert.
    """

    # Pricing constants
    DEEPGRAM_PRICE_PER_MINUTE: float = 0.0043
    GROQ_PRICE_PER_1K_TOKENS: float = 0.0003
    ELEVENLABS_PRICE_PER_1K_CHARS: float = 0.30
    TWILIO_PRICE_PER_MINUTE: float = 0.0085

    def __init__(self, alert_threshold: float = 50.0) -> None:
        self.alert_threshold = alert_threshold
        self.records: list[APIUsageRecord] = []
        logger.info(
            "CostTracker initialised with alert threshold $%.2f",
            alert_threshold,
        )

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_stt_usage(self, duration_seconds: float) -> APIUsageRecord:
        """Record Deepgram STT usage.

        Args:
            duration_seconds: Audio duration in seconds.

        Returns:
            The created :class:`APIUsageRecord`.
        """
        minutes = duration_seconds / 60.0
        cost = minutes * self.DEEPGRAM_PRICE_PER_MINUTE
        record = APIUsageRecord(
            service="deepgram",
            operation="stt_transcription",
            units=minutes,
            cost=cost,
            timestamp=datetime.now(timezone.utc),
        )
        self.records.append(record)
        logger.debug(
            "Recorded STT usage: %.2f min, $%.6f", minutes, cost
        )
        return record

    def record_llm_usage(
        self, input_tokens: int, output_tokens: int
    ) -> APIUsageRecord:
        """Record Groq LLM usage.

        Args:
            input_tokens: Number of prompt tokens.
            output_tokens: Number of completion tokens.

        Returns:
            The created :class:`APIUsageRecord`.
        """
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000.0) * self.GROQ_PRICE_PER_1K_TOKENS
        record = APIUsageRecord(
            service="groq",
            operation="llm_completion",
            units=total_tokens,
            cost=cost,
            timestamp=datetime.now(timezone.utc),
        )
        self.records.append(record)
        logger.debug(
            "Recorded LLM usage: %d tokens, $%.6f", total_tokens, cost
        )
        return record

    def record_tts_usage(self, characters: int) -> APIUsageRecord:
        """Record ElevenLabs TTS usage.

        Args:
            characters: Number of characters synthesised.

        Returns:
            The created :class:`APIUsageRecord`.
        """
        cost = (characters / 1000.0) * self.ELEVENLABS_PRICE_PER_1K_CHARS
        record = APIUsageRecord(
            service="elevenlabs",
            operation="tts_synthesis",
            units=characters,
            cost=cost,
            timestamp=datetime.now(timezone.utc),
        )
        self.records.append(record)
        logger.debug(
            "Recorded TTS usage: %d chars, $%.6f", characters, cost
        )
        return record

    def record_telephony_usage(
        self, duration_seconds: float
    ) -> APIUsageRecord:
        """Record Twilio telephony usage.

        Args:
            duration_seconds: Call duration in seconds.

        Returns:
            The created :class:`APIUsageRecord`.
        """
        minutes = duration_seconds / 60.0
        cost = minutes * self.TWILIO_PRICE_PER_MINUTE
        record = APIUsageRecord(
            service="twilio",
            operation="voice_call",
            units=minutes,
            cost=cost,
            timestamp=datetime.now(timezone.utc),
        )
        self.records.append(record)
        logger.debug(
            "Recorded telephony usage: %.2f min, $%.6f", minutes, cost
        )
        return record

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return the total cost across all recorded usage.

        Returns:
            Total cost in dollars.
        """
        return sum(record.cost for record in self.records)

    def get_cost_breakdown(self) -> dict[str, float]:
        """Return costs grouped by service name.

        Returns:
            Dict mapping service name to total cost.
        """
        breakdown: dict[str, float] = {}
        for record in self.records:
            breakdown[record.service] = (
                breakdown.get(record.service, 0.0) + record.cost
            )
        return breakdown

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def check_alerts(self) -> list[str]:
        """Check whether spending exceeds the alert threshold.

        Returns:
            List of alert message strings (empty if within budget).
        """
        alerts: list[str] = []
        total = self.get_total_cost()

        if total >= self.alert_threshold:
            alerts.append(
                f"ALERT: Total API cost ${total:.2f} has exceeded the "
                f"${self.alert_threshold:.2f} threshold."
            )

        breakdown = self.get_cost_breakdown()
        for service, cost in breakdown.items():
            service_threshold = self.alert_threshold * 0.5
            if cost >= service_threshold:
                alerts.append(
                    f"WARNING: {service} cost ${cost:.2f} exceeds 50% of "
                    f"total threshold (${service_threshold:.2f})."
                )

        if alerts:
            for alert in alerts:
                logger.warning(alert)

        return alerts

    # ------------------------------------------------------------------
    # ROI analysis
    # ------------------------------------------------------------------

    def generate_roi_report(
        self,
        calls_handled: int,
        avg_human_cost_per_call: float = 5.0,
    ) -> dict:
        """Generate a return-on-investment report.

        Compares the AI agent's cost per call against an estimated
        human-agent cost per call.

        Args:
            calls_handled: Number of calls handled by the AI agent.
            avg_human_cost_per_call: Average cost of a human-handled call.

        Returns:
            Dict with keys: ``total_cost``, ``calls_handled``,
            ``cost_per_call``, ``human_equivalent_cost``, ``savings``,
            ``roi_percentage``.
        """
        total_cost = self.get_total_cost()
        cost_per_call = total_cost / calls_handled if calls_handled else 0.0
        human_equivalent_cost = calls_handled * avg_human_cost_per_call
        savings = human_equivalent_cost - total_cost
        roi_percentage = (
            (savings / human_equivalent_cost * 100.0)
            if human_equivalent_cost > 0
            else 0.0
        )

        report = {
            "total_cost": total_cost,
            "calls_handled": calls_handled,
            "cost_per_call": cost_per_call,
            "human_equivalent_cost": human_equivalent_cost,
            "savings": savings,
            "roi_percentage": roi_percentage,
        }

        logger.info(
            "ROI report: %d calls, $%.2f total, $%.2f/call, "
            "%.1f%% ROI",
            calls_handled,
            total_cost,
            cost_per_call,
            roi_percentage,
        )
        return report
