"""Twilio telephony integration for voice call handling."""

import logging

from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse

from src.config import TwilioConfig

logger = logging.getLogger(__name__)


class TelephonyError(Exception):
    """Custom exception for telephony operations."""


class TwilioHandler:
    """Manages Twilio voice call operations.

    Provides helpers for creating outbound calls, generating TwiML
    responses, and parsing incoming webhook data.

    Args:
        config: Twilio configuration with account SID, auth token,
            and phone number.
    """

    def __init__(self, config: TwilioConfig) -> None:
        self._config = config
        self._client: Client | None = None
        logger.info("TwilioHandler created for number %s", config.phone_number)

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def create_client(self) -> Client:
        """Create and return a Twilio REST client.

        Returns:
            An authenticated :class:`twilio.rest.Client` instance.

        Raises:
            TelephonyError: If credentials are missing or client
                creation fails.
        """
        if not self._config.account_sid or not self._config.auth_token:
            raise TelephonyError(
                "Twilio account SID and auth token are required"
            )

        try:
            self._client = Client(
                self._config.account_sid, self._config.auth_token
            )
            logger.info("Twilio client created successfully")
            return self._client
        except Exception as exc:
            raise TelephonyError(
                f"Failed to create Twilio client: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Outbound calls
    # ------------------------------------------------------------------

    def make_call(self, to_number: str, callback_url: str) -> str:
        """Initiate an outbound voice call.

        Args:
            to_number: The destination phone number (E.164 format).
            callback_url: URL that Twilio will request for call instructions.

        Returns:
            The call SID string.

        Raises:
            TelephonyError: If the call cannot be initiated.
        """
        if self._client is None:
            self.create_client()

        if not self._config.phone_number:
            raise TelephonyError("Twilio phone number is not configured")

        logger.info(
            "Initiating call from %s to %s",
            self._config.phone_number,
            to_number,
        )

        try:
            call = self._client.calls.create(  # type: ignore[union-attr]
                to=to_number,
                from_=self._config.phone_number,
                url=callback_url,
            )
            logger.info("Call initiated: SID=%s", call.sid)
            return call.sid
        except Exception as exc:
            raise TelephonyError(
                f"Failed to initiate call to {to_number}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # TwiML generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_twiml_response(message: str) -> str:
        """Generate a TwiML XML response with a Say verb.

        Args:
            message: The text to be spoken to the caller.

        Returns:
            TwiML XML string.
        """
        response = VoiceResponse()
        response.say(message)
        logger.debug("Generated TwiML Say response")
        return str(response)

    @staticmethod
    def generate_gather_twiml(
        prompt: str, action_url: str, timeout: int = 5
    ) -> str:
        """Generate TwiML with Gather for speech input.

        The response will listen for speech, then POST the result to
        *action_url*.  If no input is received within *timeout* seconds
        the prompt is repeated.

        Args:
            prompt: Text spoken to the caller before gathering input.
            action_url: URL to POST gathered speech results to.
            timeout: Seconds to wait for speech input.

        Returns:
            TwiML XML string.
        """
        response = VoiceResponse()
        gather: Gather = response.gather(
            input="speech",
            action=action_url,
            timeout=timeout,
        )
        gather.say(prompt)
        # If no input, repeat
        response.say(prompt)
        logger.debug("Generated TwiML Gather response")
        return str(response)

    # ------------------------------------------------------------------
    # Webhook handling
    # ------------------------------------------------------------------

    @staticmethod
    def handle_incoming_webhook(form_data: dict) -> dict:
        """Parse incoming Twilio webhook data.

        Args:
            form_data: Dictionary of form fields from the webhook POST.

        Returns:
            Dict with keys: ``caller``, ``called``, ``call_sid``,
            ``speech_result``.

        Raises:
            TelephonyError: If required fields are missing.
        """
        try:
            result = {
                "caller": form_data.get("From", ""),
                "called": form_data.get("To", ""),
                "call_sid": form_data.get("CallSid", ""),
                "speech_result": form_data.get("SpeechResult"),
            }
            logger.info(
                "Parsed webhook: call_sid=%s, caller=%s",
                result["call_sid"],
                result["caller"],
            )
            return result
        except (AttributeError, TypeError) as exc:
            raise TelephonyError(
                f"Failed to parse webhook data: {exc}"
            ) from exc
