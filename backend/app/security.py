"""At-rest encryption for integration API keys (DESIGN §5.7).

Keys are encrypted with Fernet (AES-128-CBC + HMAC). The Fernet key is derived
deterministically from the app ``secret_key`` via SHA-256 → urlsafe-base64, so rotating the
secret invalidates stored ciphertexts (by design).
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


def _fernet() -> Fernet:
    secret = get_settings().secret_key.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a credential for storage; returns urlsafe ciphertext text."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored credential. Raises ``InvalidToken`` if the secret changed."""
    return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")


def try_decrypt_secret(ciphertext: str) -> str | None:
    """Decrypt, returning None instead of raising on an invalid/rotated token."""
    try:
        return decrypt_secret(ciphertext)
    except (InvalidToken, ValueError):
        return None


def mask_secret(plaintext: str) -> str:
    """Return a masked preview safe to show in the UI (never the full key)."""
    if len(plaintext) <= 4:
        return "••••"
    return f"••••{plaintext[-4:]}"
