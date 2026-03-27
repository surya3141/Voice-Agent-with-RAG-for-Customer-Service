"""Tests for DataMasker and RBAC modules."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.security.data_masking import DataMasker
from src.security.rbac import RBACManager, Role, User


# --- DataMasker tests ---

@pytest.fixture
def masker():
    """Return a DataMasker instance."""
    return DataMasker()


@pytest.mark.parametrize("cc_number", [
    "4111111111111111",
    "4111-1111-1111-1111",
    "4111 1111 1111 1111",
])
def test_mask_credit_card(masker, cc_number):
    """DataMasker masks credit card numbers in various formats."""
    result = masker.mask_text(f"Card: {cc_number}")
    assert "4111" not in result.split("****-****-****-")[0]
    assert "****" in result


def test_mask_ssn(masker):
    """DataMasker masks Social Security Numbers."""
    result = masker.mask_text("SSN: 123-45-6789")
    assert "123-45" not in result
    assert "***-**-6789" in result


def test_mask_email(masker):
    """DataMasker masks email addresses."""
    result = masker.mask_text("Email: john.doe@example.com")
    assert "john.doe@example.com" not in result
    assert "j***@example.com" in result


@pytest.mark.parametrize("phone", [
    "555-123-4567",
    "(555) 123-4567",
    "+1-555-123-4567",
])
def test_mask_phone(masker, phone):
    """DataMasker masks phone numbers in various formats."""
    result = masker.mask_text(f"Call: {phone}")
    assert "4567" in result
    assert "(***) ***-4567" in result


def test_mask_dict_recursive(masker):
    """DataMasker.mask_dict recursively masks sensitive data."""
    data = {
        "customer": {
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
        },
        "payment": {
            "card": "4111111111111111",
        },
        "notes": "SSN is 123-45-6789",
    }
    masked = masker.mask_dict(data)
    assert "john.doe@example.com" not in masked["customer"]["email"]
    assert "4111111111111111" not in masked["payment"]["card"]
    assert "123-45" not in masked["notes"]


# --- RBAC tests ---

@pytest.fixture
def rbac():
    """Return an RBACManager with a test secret key."""
    return RBACManager(secret_key="test-secret-key-12345")


@pytest.fixture
def admin_user():
    """Return an admin User."""
    return User(username="admin_user", role=Role.ADMIN)


@pytest.fixture
def support_user():
    """Return a support User."""
    return User(username="support_user", role=Role.SUPPORT)


@pytest.fixture
def viewer_user():
    """Return a viewer User."""
    return User(username="viewer_user", role=Role.VIEWER)


def test_create_and_validate_token(rbac, admin_user):
    """RBACManager can create and validate a token."""
    token = rbac.create_token(admin_user)
    assert isinstance(token, str)
    assert len(token) > 0

    validated_user = rbac.validate_token(token)
    assert validated_user is not None
    assert validated_user.username == "admin_user"
    assert validated_user.role == Role.ADMIN


def test_token_expiry(rbac, admin_user):
    """Expired tokens are rejected by validate_token."""
    token = rbac.create_token(admin_user)
    # Mock datetime.now to return a time 31 minutes in the future
    future_time = datetime.now(timezone.utc) + timedelta(minutes=31)
    with patch("src.security.rbac.datetime") as mock_dt:
        mock_dt.now.return_value = future_time
        mock_dt.fromisoformat = datetime.fromisoformat
        result = rbac.validate_token(token)
    assert result is None


def test_admin_permissions(rbac, admin_user):
    """Admin role has access to all resources."""
    assert RBACManager.check_permission(admin_user, "orders") is True
    assert RBACManager.check_permission(admin_user, "dashboard") is True
    assert RBACManager.check_permission(admin_user, "calls") is True
    assert RBACManager.check_permission(admin_user, "faqs") is True
    assert RBACManager.check_permission(admin_user, "appointments") is True
    assert RBACManager.check_permission(admin_user, "anything_else") is True


def test_support_permissions(rbac, support_user):
    """Support role has access to specific resources only."""
    assert RBACManager.check_permission(support_user, "orders") is True
    assert RBACManager.check_permission(support_user, "dashboard") is True
    assert RBACManager.check_permission(support_user, "calls") is True
    assert RBACManager.check_permission(support_user, "faqs") is True
    assert RBACManager.check_permission(support_user, "appointments") is True
    assert RBACManager.check_permission(support_user, "admin_panel") is False


def test_viewer_permissions(rbac, viewer_user):
    """Viewer role has access to dashboard and faqs only."""
    assert RBACManager.check_permission(viewer_user, "dashboard") is True
    assert RBACManager.check_permission(viewer_user, "faqs") is True
    assert RBACManager.check_permission(viewer_user, "orders") is False
    assert RBACManager.check_permission(viewer_user, "calls") is False
    assert RBACManager.check_permission(viewer_user, "appointments") is False


def test_tampered_token_rejected(rbac, admin_user):
    """Tampered tokens are rejected by validate_token."""
    token = rbac.create_token(admin_user)
    # Tamper with the token by modifying a character
    tampered = token[:-5] + "XXXXX"
    result = rbac.validate_token(tampered)
    assert result is None
