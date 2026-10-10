"""Timing and byte accounting for the crypto layer.

`metered_suite` wraps a Suite so every KEM / signature call is timed into a Meter without
the protocol code knowing about benchmarking. Callers time the symmetric path and the
total protocol cost themselves by wrapping ClientCrypto / ServerCrypto calls in
`meter.time(...)`.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from pqfl.crypto.primitives import KeyPair, Suite

# Columns produced by a metered suite + harness-side timers, in CSV order.
TIME_KEYS = ("keygen", "encaps", "decaps", "sign", "verify",
             "sym_encrypt", "sym_decrypt", "crypto_total")
BYTE_KEYS = ("key", "kem_ct", "sig", "payload", "wire_total")


class Meter:
    """Accumulates seconds per operation and bytes per category until reset()."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.bytes: dict[str, int] = defaultdict(int)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.seconds.clear()
        self.bytes.clear()
        self.counts.clear()

    @contextmanager
    def time(self, *keys: str) -> Iterator[None]:
        """Time the block once and add the elapsed seconds to every key given."""
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            for k in keys:
                self.seconds[k] += dt
                self.counts[k] += 1

    def add_bytes(self, key: str, n: int) -> None:
        self.bytes[key] += n


class _MeteredKEM:
    def __init__(self, inner, meter: Meter) -> None:
        self._inner, self._meter, self.name = inner, meter, inner.name

    def keygen(self) -> KeyPair:
        with self._meter.time("keygen"):
            return self._inner.keygen()

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        with self._meter.time("encaps"):
            return self._inner.encaps(public)

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        with self._meter.time("decaps"):
            return self._inner.decaps(secret, ciphertext)


class _MeteredSigner:
    def __init__(self, inner, meter: Meter) -> None:
        self._inner, self._meter, self.name = inner, meter, inner.name

    def keygen(self) -> KeyPair:
        # Identity keys are generated once at setup; timed separately from per-round keygen.
        with self._meter.time("sig_keygen"):
            return self._inner.keygen()

    def sign(self, secret: bytes, message: bytes) -> bytes:
        with self._meter.time("sign"):
            return self._inner.sign(secret, message)

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool:
        with self._meter.time("verify"):
            return self._inner.verify(public, message, signature)


def metered_suite(suite: Suite, meter: Meter) -> Suite:
    """Same primitives and name (so derived keys are unchanged), with every call timed."""
    return Suite(suite.name, _MeteredKEM(suite.kem, meter), _MeteredSigner(suite.sig, meter))
