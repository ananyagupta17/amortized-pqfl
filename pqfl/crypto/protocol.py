"""Client/server session protocol for encrypting FL model updates.

Session setup (handshake), once per key epoch:
  1. Client generates a fresh KEM keypair and signs (client_id, epoch, kem_pk) with its
     long-term identity key.                                       -> HandshakeInit
  2. Server verifies the signature against the client's registered key, encapsulates
     to kem_pk, and signs the response bound to a hash of the init. -> HandshakeResponse
  3. Client verifies the server signature and decapsulates. Both sides derive
     session_key = HKDF-SHA256(ss, info = label | suite | client_id | epoch | H(transcript)).

Per round:
  round_key = HKDF-SHA256(session_key, info = label | round)
  ciphertext = AES-256-GCM(round_key, random 96-bit nonce, update,
                           aad = label | mode | client_id | round | epoch)

Key epochs follow a fixed schedule, epoch = round // rekey_interval, so clients that skip
rounds (partial participation) still agree with the server on which key is current.

The server never trusts the header's round/epoch: it builds the AAD from its own view of
the current round and the expected epoch, so a replayed or reordered update fails
authentication even if an attacker rewrites the header. It also remembers the last round
it accepted from each client and rejects anything at or before it.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, replace
from enum import Enum

from cryptography.exceptions import InvalidTag

from pqfl.crypto.primitives import (
    CLASSICAL_SUITE,
    PQC_SUITE,
    KeyPair,
    Suite,
    aead_decrypt,
    aead_encrypt,
    hkdf_sha256,
)

_LABEL_INIT = b"pqfl/v1/handshake-init"
_LABEL_RESP = b"pqfl/v1/handshake-resp"
_LABEL_SESSION = b"pqfl/v1/session-key"
_LABEL_ROUND = b"pqfl/v1/round-key"
_LABEL_AAD = b"pqfl/v1/update-aad"


# --- Errors ----------------------------------------------------------------------

class CryptoError(Exception):
    """Base class for every rejection raised by the protocol."""


class AuthenticationError(CryptoError):
    """A signature or AEAD tag did not verify."""


class ReplayError(CryptoError):
    """A message for a round or epoch that was already used or has passed."""


class EpochError(CryptoError):
    """A message under a key epoch that does not match the round's schedule."""


class SessionError(CryptoError):
    """No usable session: the handshake has not been run or is incomplete."""


# --- Configuration ---------------------------------------------------------------

class Mode(str, Enum):
    CLASSICAL = "classical"   # X25519 + Ed25519 once, or once per N rounds; AES-GCM per round
    NAIVE_PQC = "naive_pqc"   # full ML-KEM + ML-DSA handshake every round
    AMORTIZED = "amortized"   # ML-KEM + ML-DSA once per N rounds, AES-GCM per round


@dataclass(frozen=True)
class CryptoConfig:
    mode: Mode
    rekey_interval: int | None = None  # rounds per key epoch; None = never re-key

    def __post_init__(self) -> None:
        mode = Mode(self.mode)
        object.__setattr__(self, "mode", mode)
        n = self.rekey_interval
        if mode is Mode.CLASSICAL and n is not None and (not isinstance(n, int) or n < 1):
            raise ValueError("classical rekey_interval must be None (key once) or an integer >= 1")
        if mode is Mode.NAIVE_PQC:
            if n not in (None, 1):
                raise ValueError("naive_pqc re-keys every round; rekey_interval must be 1")
            object.__setattr__(self, "rekey_interval", 1)
        if mode is Mode.AMORTIZED and (not isinstance(n, int) or n < 1):
            raise ValueError("amortized mode needs an integer rekey_interval >= 1")

    @property
    def suite(self) -> Suite:
        return CLASSICAL_SUITE if self.mode is Mode.CLASSICAL else PQC_SUITE

    def epoch_for_round(self, round_num: int) -> int:
        if round_num < 0:
            raise ValueError("round numbers start at 0")
        return 0 if self.rekey_interval is None else round_num // self.rekey_interval


def generate_identity(config: CryptoConfig) -> KeyPair:
    """Long-term signing keypair for a client or the server, in the config's suite."""
    return config.suite.sig.keygen()


# --- Wire messages ---------------------------------------------------------------

def _u64(x: int) -> bytes:
    return struct.pack(">Q", x)


def _encode(*fields: bytes) -> bytes:
    """Length-prefixed concatenation, so field boundaries are unambiguous."""
    return b"".join(struct.pack(">I", len(f)) + f for f in fields)


@dataclass(frozen=True)
class HandshakeInit:
    client_id: str
    epoch: int
    kem_public: bytes
    signature: bytes

    def to_bytes(self) -> bytes:
        return _encode(self.client_id.encode(), _u64(self.epoch), self.kem_public, self.signature)

    @property
    def nbytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class HandshakeResponse:
    client_id: str
    epoch: int
    kem_ciphertext: bytes
    signature: bytes

    def to_bytes(self) -> bytes:
        return _encode(self.client_id.encode(), _u64(self.epoch), self.kem_ciphertext, self.signature)

    @property
    def nbytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class EncryptedUpdate:
    client_id: str
    round: int
    epoch: int
    nonce: bytes
    ciphertext: bytes  # includes the 16-byte GCM tag

    def to_bytes(self) -> bytes:
        return _encode(self.client_id.encode(), _u64(self.round), _u64(self.epoch),
                       self.nonce, self.ciphertext)

    @property
    def nbytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class Session:
    epoch: int
    key: bytes = field(repr=False)


def _init_payload(suite: Suite, init: HandshakeInit) -> bytes:
    return _encode(_LABEL_INIT, suite.name.encode(), init.client_id.encode(),
                   _u64(init.epoch), init.kem_public)


def _resp_payload(suite: Suite, init: HandshakeInit, resp: HandshakeResponse) -> bytes:
    init_hash = hashlib.sha256(init.to_bytes()).digest()
    return _encode(_LABEL_RESP, suite.name.encode(), init_hash, resp.client_id.encode(),
                   _u64(resp.epoch), resp.kem_ciphertext)


def _session_key(suite: Suite, shared_secret: bytes, init: HandshakeInit,
                 resp: HandshakeResponse) -> bytes:
    transcript = hashlib.sha256(init.to_bytes() + resp.to_bytes()).digest()
    info = _encode(_LABEL_SESSION, suite.name.encode(), init.client_id.encode(),
                   _u64(init.epoch), transcript)
    return hkdf_sha256(shared_secret, info)


def round_key(session_key: bytes, round_num: int) -> bytes:
    return hkdf_sha256(session_key, _encode(_LABEL_ROUND, _u64(round_num)))


def update_aad(mode: Mode, client_id: str, round_num: int, epoch: int) -> bytes:
    return _encode(_LABEL_AAD, mode.value.encode(), client_id.encode(),
                   _u64(round_num), _u64(epoch))


def seal_update(mode: Mode, session: Session, client_id: str, round_num: int,
                plaintext: bytes) -> EncryptedUpdate:
    aad = update_aad(mode, client_id, round_num, session.epoch)
    nonce, ct = aead_encrypt(round_key(session.key, round_num), plaintext, aad)
    return EncryptedUpdate(client_id, round_num, session.epoch, nonce, ct)


# --- Parties ---------------------------------------------------------------------

class ClientCrypto:
    def __init__(self, client_id: str, config: CryptoConfig, identity: KeyPair,
                 server_public: bytes, suite: Suite | None = None) -> None:
        self.client_id = client_id
        self.config = config
        # `suite` overrides config.suite with an equivalent one, e.g. an instrumented wrapper.
        self.suite = suite or config.suite
        self._identity = identity
        self._server_public = server_public
        self._session: Session | None = None
        self._pending: tuple[HandshakeInit, bytes] | None = None  # (init, kem secret)

    @property
    def session(self) -> Session | None:
        return self._session

    def needs_handshake(self, round_num: int) -> bool:
        epoch = self.config.epoch_for_round(round_num)
        return self._session is None or self._session.epoch != epoch

    def start_handshake(self, round_num: int) -> HandshakeInit:
        suite = self.suite
        kem_kp = suite.kem.keygen()
        init = HandshakeInit(self.client_id, self.config.epoch_for_round(round_num),
                             kem_kp.public, b"")
        init = replace(init, signature=suite.sig.sign(self._identity.secret,
                                                      _init_payload(suite, init)))
        self._pending = (init, kem_kp.secret)
        return init

    def finish_handshake(self, resp: HandshakeResponse) -> None:
        if self._pending is None:
            raise SessionError("no handshake in progress")
        init, kem_secret = self._pending
        if resp.client_id != init.client_id or resp.epoch != init.epoch:
            raise AuthenticationError("handshake response does not match the pending init")
        suite = self.suite
        if not suite.sig.verify(self._server_public, _resp_payload(suite, init, resp),
                                resp.signature):
            raise AuthenticationError("server signature on handshake response is invalid")
        ss = suite.kem.decaps(kem_secret, resp.kem_ciphertext)
        self._session = Session(init.epoch, _session_key(suite, ss, init, resp))
        self._pending = None

    def encrypt(self, round_num: int, plaintext: bytes) -> EncryptedUpdate:
        if self.needs_handshake(round_num):
            raise SessionError(f"client {self.client_id!r} has no session for epoch "
                               f"{self.config.epoch_for_round(round_num)}; run the handshake")
        assert self._session is not None
        return seal_update(self.config.mode, self._session, self.client_id, round_num, plaintext)


class ServerCrypto:
    def __init__(self, config: CryptoConfig, identity: KeyPair, suite: Suite | None = None) -> None:
        self.config = config
        self.suite = suite or config.suite
        self._identity = identity
        self._client_keys: dict[str, bytes] = {}
        self._sessions: dict[str, Session] = {}
        self._last_round: dict[str, int] = {}

    @property
    def public_key(self) -> bytes:
        return self._identity.public

    def register_client(self, client_id: str, public_key: bytes) -> None:
        """Pin a client's long-term signing key (assumed distributed out of band)."""
        self._client_keys[client_id] = public_key

    def accept_handshake(self, round_num: int, init: HandshakeInit) -> HandshakeResponse:
        suite = self.suite
        client_pk = self._client_keys.get(init.client_id)
        if client_pk is None:
            raise AuthenticationError(f"unknown client {init.client_id!r}")
        expected = self.config.epoch_for_round(round_num)
        if init.epoch != expected:
            raise EpochError(f"handshake for epoch {init.epoch}, round {round_num} "
                             f"requires epoch {expected}")
        current = self._sessions.get(init.client_id)
        if current is not None and init.epoch <= current.epoch:
            raise ReplayError(f"client {init.client_id!r} already keyed epoch {current.epoch}")
        if not suite.sig.verify(client_pk, _init_payload(suite, init), init.signature):
            raise AuthenticationError("client signature on handshake init is invalid")

        try:
            ct, ss = suite.kem.encaps(init.kem_public)
        except ValueError as e:
            raise AuthenticationError(f"malformed KEM public key: {e}") from e
        resp = HandshakeResponse(init.client_id, init.epoch, ct, b"")
        resp = replace(resp, signature=suite.sig.sign(self._identity.secret,
                                                      _resp_payload(suite, init, resp)))
        self._sessions[init.client_id] = Session(init.epoch, _session_key(suite, ss, init, resp))
        return resp

    def decrypt(self, round_num: int, update: EncryptedUpdate) -> bytes:
        """Open `update` as a submission for the server's current round `round_num`."""
        cid = update.client_id
        session = self._sessions.get(cid)
        if session is None:
            raise SessionError(f"no session for client {cid!r}")
        expected_epoch = self.config.epoch_for_round(round_num)
        if session.epoch != expected_epoch:
            raise EpochError(f"client {cid!r} is keyed for epoch {session.epoch}, "
                             f"round {round_num} requires epoch {expected_epoch}")
        last = self._last_round.get(cid)
        if update.round != round_num or (last is not None and round_num <= last):
            raise ReplayError(f"update for round {update.round} from client {cid!r} "
                              f"rejected at round {round_num} (last accepted: {last})")
        if update.epoch != expected_epoch:
            raise EpochError(f"update claims epoch {update.epoch}, expected {expected_epoch}")

        # AAD comes from the server's own state, not the header.
        aad = update_aad(self.config.mode, cid, round_num, expected_epoch)
        try:
            plaintext = aead_decrypt(round_key(session.key, round_num), update.nonce,
                                     update.ciphertext, aad)
        except (InvalidTag, ValueError) as e:  # ValueError: malformed nonce length
            raise AuthenticationError(f"update from client {cid!r} failed authentication") from e
        self._last_round[cid] = round_num
        return plaintext
