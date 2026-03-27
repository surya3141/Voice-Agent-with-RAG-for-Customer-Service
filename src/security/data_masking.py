"""Data masking module for sensitive information protection.

Detects and masks credit card numbers, SSNs, email addresses,
and phone numbers in text using regular expressions.
"""

import re

_CREDIT_CARD_RE = re.compile(r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b")
_SSN_RE = re.compile(r"\b(\d{3})-(\d{2})-(\d{4})\b")
_EMAIL_RE = re.compile(r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")
_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b")


class DataMasker:
    """Detects and masks sensitive data in text and dictionaries."""

    def mask_text(self, text: str) -> str:
        """Mask all sensitive data in the given text."""
        text = self._mask_credit_cards(text)
        text = self._mask_ssns(text)
        text = self._mask_emails(text)
        text = self._mask_phones(text)
        return text

    def mask_dict(self, data: dict) -> dict:
        """Recursively mask sensitive data in dictionary values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask_text(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.mask_dict(item) if isinstance(item, dict)
                    else self.mask_text(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    @staticmethod
    def _mask_credit_cards(text: str) -> str:
        def _replace(match):
            return "****-****-****-" + match.group(4)
        return _CREDIT_CARD_RE.sub(_replace, text)

    @staticmethod
    def _mask_ssns(text: str) -> str:
        def _replace(match):
            return "***-**-" + match.group(3)
        return _SSN_RE.sub(_replace, text)

    @staticmethod
    def _mask_emails(text: str) -> str:
        def _replace(match):
            name = match.group(1)
            domain = match.group(2)
            return name[0] + "***@" + domain
        return _EMAIL_RE.sub(_replace, text)

    @staticmethod
    def _mask_phones(text: str) -> str:
        def _replace(match):
            return "(***) ***-" + match.group(3)
        return _PHONE_RE.sub(_replace, text)
