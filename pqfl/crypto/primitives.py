"""Uniform wrappers over the KEM, signature, KDF and AEAD primitives.

Keys cross these interfaces as raw bytes so the classical and PQC suites are
interchangeable in the protocol layer and their wire sizes are directly comparable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import oqs
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_LEN = 32    # AES-256 / HKDF-SHA256 output
NONCE_LEN = 12  # 96-bit GCM nonce


@dataclass(frozen=True)
class KeyPair:
    public: bytes
    secret: bytes = field(repr=False)


class KEM(Protocol):
    name: str

    def keygen(self) -> KeyPair: ...
    def encaps(self, public: bytes) -> tuple[bytes, bytes]: ...  # (ciphertext, shared_secret)
    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes: ...


class Signer(Protocol):
    name: str

    def keygen(self) -> KeyPair: ...
    def sign(self, secret: bytes, message: bytes) -> bytes: ...
    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool: ...


# --- Post-quantum (liboqs) -------------------------------------------------------

class MLKEM:
    def __init__(self, alg: str = "ML-KEM-768") -> None:
        self.name = alg
        with oqs.KeyEncapsulation(alg) as kem:
            self._pk_len = kem.details["length_public_key"]
            self._ct_len = kem.details["length_ciphertext"]

    def keygen(self) -> KeyPair:
        with oqs.KeyEncapsulation(self.name) as kem:
            pk = kem.generate_keypair()
            return KeyPair(pk, kem.export_secret_key())

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        # liboqs-python does not length-check its inputs, so do it here.
        if len(public) != self._pk_len:
            raise ValueError(f"{self.name} public key must be {self._pk_len} bytes, got {len(public)}")
        with oqs.KeyEncapsulation(self.name) as kem:
            return kem.encap_secret(public)

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        if len(ciphertext) != self._ct_len:
            raise ValueError(f"{self.name} ciphertext must be {self._ct_len} bytes, got {len(ciphertext)}")
        with oqs.KeyEncapsulation(self.name, secret_key=secret) as kem:
            return kem.decap_secret(ciphertext)


class MLDSA:
    def __init__(self, alg: str = "ML-DSA-65") -> None:
        self.name = alg

    def keygen(self) -> KeyPair:
        with oqs.Signature(self.name) as sig:
            pk = sig.generate_keypair()
            return KeyPair(pk, sig.export_secret_key())

    def sign(self, secret: bytes, message: bytes) -> bytes:
        with oqs.Signature(self.name, secret_key=secret) as sig:
            return sig.sign(message)

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool:
        with oqs.Signature(self.name) as sig:
            return bool(sig.verify(message, signature, public))


# --- Classical baseline ----------------------------------------------------------

class X25519KEM:
    """X25519 used as a KEM (DHKEM-style): the ciphertext is an ephemeral public key."""

    name = "X25519"

    def keygen(self) -> KeyPair:
        sk = X25519PrivateKey.generate()
        return KeyPair(sk.public_key().public_bytes_raw(), sk.private_bytes_raw())

    def encaps(self, public: bytes) -> tuple[bytes, bytes]:
        eph = X25519PrivateKey.generate()
        # exchange() raises ValueError on low-order points (all-zero shared secret).
        ss = eph.exchange(X25519PublicKey.from_public_bytes(public))
        return eph.public_key().public_bytes_raw(), ss

    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes:
        sk = X25519PrivateKey.from_private_bytes(secret)
        return sk.exchange(X25519PublicKey.from_public_bytes(ciphertext))


class Ed25519Signer:
    name = "Ed25519"

    def keygen(self) -> KeyPair:
        sk = Ed25519PrivateKey.generate()
        return KeyPair(sk.public_key().public_bytes_raw(), sk.private_bytes_raw())

    def sign(self, secret: bytes, message: bytes) -> bytes:
        return Ed25519PrivateKey.from_private_bytes(secret).sign(message)

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
            return True
        except (InvalidSignature, ValueError):
            return False


@dataclass(frozen=True)
class Suite:
    name: str
    kem: KEM
    sig: Signer


CLASSICAL_SUITE = Suite("x25519+ed25519", X25519KEM(), Ed25519Signer())
PQC_SUITE = Suite("mlkem768+mldsa65", MLKEM("ML-KEM-768"), MLDSA("ML-DSA-65"))


# --- Symmetric layer -------------------------------------------------------------

def hkdf_sha256(ikm: bytes, info: bytes, salt: bytes | None = None, length: int = KEY_LEN) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
    """AES-256-GCM with a fresh random 96-bit nonce. Returns (nonce, ciphertext||tag)."""
    nonce = os.urandom(NONCE_LEN)
    return nonce, AESGCM(key).encrypt(nonce, plaintext, aad)


def aead_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    """Raises cryptography.exceptions.InvalidTag on any authentication failure."""
    return AESGCM(key).decrypt(nonce, ciphertext, aad)
