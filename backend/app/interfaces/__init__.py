"""Pluggable infrastructure seams (DESIGN §4.1a).

These interfaces keep v1 single-user/local while leaving the cloud/multi-user door open
without a rewrite:

- :mod:`app.interfaces.storage` — ``StorageProvider`` (LocalFsStorage now, object store later)
- :mod:`app.interfaces.auth` — ``AuthProvider`` (NoAuthProvider now, token/OIDC later)
- :mod:`app.interfaces.queue` — ``JobQueue`` (in-process FIFO now, broker later)
"""
