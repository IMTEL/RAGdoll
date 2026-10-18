"""Helpers for keeping secrets out of log output."""

import re


_MASK = "[REDACTED]"

_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"\brgd_[A-Za-z0-9_-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    # Long base64/hex-like runs (32+ chars with no spaces).
    re.compile(r"[A-Za-z0-9+/_-]{32,}={0,2}"),
]


def redact(text: str) -> str:
    """Mask anything that looks like an API key or token."""
    for pattern in _PATTERNS:
        text = pattern.sub(_MASK, text)
    return text
