"""Role-Based Access Control module.

Provides role definitions, token creation / validation using
HMAC-SHA256 signed base64-encoded JSON tokens, and permission
checking for the voice agent system.
"""

import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

_TOKEN_LIFETIME_MINUTES = 30


class Role(Enum):
    """Available user roles."""
    ADMIN = "admin"
    SUPPORT = "support"
    VIEWER = "viewer"


@dataclass
class User:
    """Represents an authenticated user.

    Attributes:
        username: Unique user identifier.
        role: The user's assigned role.
        created_at: Timestamp of account creation.
    """
    username: str
    role: Role
    created_at: datetime = None


_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"*"},
    Role.SUPPORT: {"faqs", "dashboard", "appointments", "orders", "calls"},
    Role.VIEWER: {"dashboard", "faqs"},
}


class RBACManager:
    """Manages authentication tokens and role-based permissions.

    Tokens are base64-encoded JSON payloads signed with HMAC-SHA256
    using the provided *secret_key*.

    Args:
        secret_key: Secret used for signing and verifying tokens.
    """

    def __init__(self, secret_key: str) -> None:
        self._secret_key = secret_key
        logger.info("RBACManager initialised")

    def create_token(self, user: User) -> str:
        """Create a signed, base64-encoded JSON token for *user*.

        The token contains the username, role, and an expiry timestamp
        set to 30 minutes from now.

        Args:
            user: The user to create a token for.

        Returns:
            The base64-encoded signed token string.
        """
        expiry = datetime.now(timezone.utc) + timedelta(minutes=_TOKEN_LIFETIME_MINUTES)
        payload = {
            "username": user.username,
            "role": user.role.value,
            "expiry": expiry.isoformat(),
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = self._sign(payload_bytes)
        token_data = json.dumps({
            "payload": base64.urlsafe_b64encode(payload_bytes).decode("ascii"),
            "signature": signature,
        }, sort_keys=True).encode("utf-8")
        token = base64.urlsafe_b64encode(token_data).decode("ascii")
        logger.info("Token created for user '%s' (role=%s)", user.username, user.role.value)
        return token

    def validate_token(self, token: str) -> User | None:
        """Validate a token and return the corresponding User.

        Returns None when the token is malformed, the signature is
        invalid, or the token has expired.

        Args:
            token: The token string to validate.

        Returns:
            The User if valid, None otherwise.
        """
        try:
            outer = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
            payload_b64 = outer["payload"]
            provided_sig = outer["signature"]
            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
            expected_sig = self._sign(payload_bytes)
            if not hmac.compare_digest(provided_sig, expected_sig):
                logger.warning("Token signature mismatch")
                return None
            payload = json.loads(payload_bytes)
            expiry = datetime.fromisoformat(payload["expiry"])
            if datetime.now(timezone.utc) > expiry:
                logger.warning("Token expired for user '%s'", payload.get("username"))
                return None
            user = User(
                username=payload["username"],
                role=Role(payload["role"]),
                created_at=datetime.now(timezone.utc),
            )
            logger.debug("Token validated for user '%s'", user.username)
            return user
        except (KeyError, ValueError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Token validation failed: %s", exc)
            return None

    @staticmethod
    def check_permission(user: User, resource: str) -> bool:
        """Check whether *user* has access to *resource*.

        Args:
            user: The user requesting access.
            resource: The resource identifier (e.g. "orders").

        Returns:
            True if the user has access, False otherwise.
        """
        allowed = _PERMISSIONS.get(user.role, set())
        has_access = "*" in allowed or resource in allowed
        logger.debug(
            "Permission check: user='%s' role='%s' resource='%s' -> %s",
            user.username, user.role.value, resource, has_access,
        )
        return has_access

    def _sign(self, data: bytes) -> str:
        """Return the hex-encoded HMAC-SHA256 signature of *data*."""
        return hmac.new(
            self._secret_key.encode("utf-8"),
            data,
            hashlib.sha256,
        ).hexdigest()
