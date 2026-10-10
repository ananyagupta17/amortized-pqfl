import copy
import os
from dataclasses import replace

import pytest

from pqfl.crypto import (
    AuthenticationError,
    ClientCrypto,
    CryptoConfig,
    EpochError,
    Mode,
    ReplayError,
    ServerCrypto,
    SessionError,
    generate_identity,
)
from pqfl.crypto.protocol import _resp_payload, seal_update

N = 3  # re-key interval for amortized mode

CONFIGS = {
    "classical": CryptoConfig(Mode.CLASSICAL),
    "naive_pqc": CryptoConfig(Mode.NAIVE_PQC),
    "amortized": CryptoConfig(Mode.AMORTIZED, rekey_interval=N),
    "classical_rekey": CryptoConfig(Mode.CLASSICAL, rekey_interval=N),
}
REKEYING = ["amortized", "classical_rekey"]  # configs with a re-key interval N > 1


def make_parties(config, client_ids=("c0", "c1")):
    server = ServerCrypto(config, generate_identity(config))
    clients = {}
    for cid in client_ids:
        identity = generate_identity(config)
        server.register_client(cid, identity.public)
        clients[cid] = ClientCrypto(cid, config, identity, server.public_key)
    return server, clients


def handshake(client, server, round_num):
    client.finish_handshake(server.accept_handshake(round_num, client.start_handshake(round_num)))


def ensure_session(client, server, round_num):
    if client.needs_handshake(round_num):
        handshake(client, server, round_num)


def flip(b: bytes, i: int = 0) -> bytes:
    return b[:i] + bytes([b[i] ^ 0x01]) + b[i + 1:]


@pytest.fixture(params=list(CONFIGS), ids=list(CONFIGS))
def config(request):
    return CONFIGS[request.param]


# --- Round trip ------------------------------------------------------------------

def test_roundtrip_and_rekey_schedule(config):
    server, clients = make_parties(config)
    rounds = 7
    handshakes = {cid: 0 for cid in clients}
    for r in range(rounds):
        for cid, client in clients.items():
            if client.needs_handshake(r):
                handshake(client, server, r)
                handshakes[cid] += 1
            update = os.urandom(1000)
            msg = client.encrypt(r, update)
            assert len(msg.nonce) == 12
            assert server.decrypt(r, msg) == update
            assert msg.epoch == config.epoch_for_round(r)

    n = config.rekey_interval
    expected = 1 if n is None else -(-rounds // n)
    assert all(h == expected for h in handshakes.values())


def test_nonces_unique_and_ciphertexts_differ(config):
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    ensure_session(client, server, 0)
    msgs = [client.encrypt(0, b"same plaintext") for _ in range(50)]
    assert len({m.nonce for m in msgs}) == 50
    assert len({m.ciphertext for m in msgs}) == 50


@pytest.mark.parametrize("name", REKEYING)
def test_partial_participation_late_joiner(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    ensure_session(client, server, 7)  # first appears in round 7 -> epoch 2
    assert client.session.epoch == 2
    assert server.decrypt(7, client.encrypt(7, b"hi")) == b"hi"


def test_encrypt_without_session_raises(config):
    _, clients = make_parties(config, ("c0",))
    with pytest.raises(SessionError):
        clients["c0"].encrypt(0, b"x")


# --- Tampering -------------------------------------------------------------------

@pytest.mark.parametrize("field", ["ciphertext_first", "tag_last", "nonce", "truncated", "client_id"])
def test_tampered_ciphertext_rejected(config, field):
    server, clients = make_parties(config)
    for c in clients.values():
        ensure_session(c, server, 0)
    msg = clients["c0"].encrypt(0, b"model update bytes")

    tampered = {
        "ciphertext_first": lambda m: replace(m, ciphertext=flip(m.ciphertext, 0)),
        "tag_last": lambda m: replace(m, ciphertext=flip(m.ciphertext, len(m.ciphertext) - 1)),
        "nonce": lambda m: replace(m, nonce=flip(m.nonce, 5)),
        "truncated": lambda m: replace(m, ciphertext=m.ciphertext[:-1]),
        # Attribute c0's update to c1: c1's session key and the AAD both differ.
        "client_id": lambda m: replace(m, client_id="c1"),
    }[field](msg)

    with pytest.raises(AuthenticationError):
        server.decrypt(0, tampered)
    # The untouched message is still accepted afterwards (a rejection doesn't burn the round).
    assert server.decrypt(0, msg) == b"model update bytes"


# --- Replay / reordering ---------------------------------------------------------

def test_replayed_update_from_old_round_rejected(config):
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    ensure_session(client, server, 2)
    old = client.encrypt(2, b"round-2 update")
    assert server.decrypt(2, old) == b"round-2 update"

    # Duplicate within the same round.
    with pytest.raises(ReplayError):
        server.decrypt(2, old)

    ensure_session(client, server, 3)
    epoch3 = config.epoch_for_round(3)

    # Replayed verbatim into round 3.
    with pytest.raises(ReplayError):
        server.decrypt(3, old)
    # Header rewritten to look current: caught by the AEAD (round is in the key and AAD).
    with pytest.raises(AuthenticationError):
        server.decrypt(3, replace(old, round=3, epoch=epoch3))

    # The legitimate round-3 update still goes through.
    assert server.decrypt(3, client.encrypt(3, b"round-3 update")) == b"round-3 update"


@pytest.mark.parametrize("name", ["classical", *REKEYING])  # naive_pqc can't pre-encrypt: new key each round
def test_reordered_update_rejected(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    ensure_session(client, server, 0)
    early = client.encrypt(1, b"round-1 update")  # delivered during round 0
    with pytest.raises(ReplayError):
        server.decrypt(0, early)
    with pytest.raises(AuthenticationError):
        server.decrypt(0, replace(early, round=0))


def test_rollback_to_earlier_round_rejected():
    config = CONFIGS["classical"]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    ensure_session(client, server, 0)
    server.decrypt(5, client.encrypt(5, b"r5"))
    # A fresh, correctly-encrypted update for an earlier round is still refused.
    with pytest.raises(ReplayError):
        server.decrypt(4, client.encrypt(4, b"r4"))


# --- Key epochs ------------------------------------------------------------------

@pytest.fixture(params=REKEYING)
def rekeyed(request):
    """Client c0 keyed epoch 0 at round 0, then re-keyed to epoch 1 at round N."""
    config = CONFIGS[request.param]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    handshake(client, server, 0)
    stale = copy.deepcopy(client)  # keeps the epoch-0 session
    handshake(client, server, N)
    return config, server, client, stale


def test_old_epoch_key_rejected_after_rekey(rekeyed):
    config, server, client, stale = rekeyed
    assert stale.session.epoch == 0 and client.session.epoch == 1

    # Old-epoch key, honestly labelled epoch 0.
    with pytest.raises(EpochError):
        server.decrypt(N, seal_update(config.mode, stale.session, "c0", N, b"x"))
    # Old-epoch key, header and AAD relabelled as epoch 1.
    forged = replace(stale.session, epoch=1)
    with pytest.raises(AuthenticationError):
        server.decrypt(N, seal_update(config.mode, forged, "c0", N, b"x"))
    # Stale client object refuses to encrypt at all.
    with pytest.raises(SessionError):
        stale.encrypt(N, b"x")

    assert server.decrypt(N, client.encrypt(N, b"ok")) == b"ok"


def test_update_with_wrong_epoch_header_rejected(rekeyed):
    _, server, client, _ = rekeyed
    msg = client.encrypt(N, b"x")
    for bad_epoch in (0, 2):
        with pytest.raises(EpochError):
            server.decrypt(N, replace(msg, epoch=bad_epoch))


@pytest.mark.parametrize("name", REKEYING)
def test_server_rejects_update_when_client_skipped_rekey(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    handshake(client, server, 0)
    # Server is at round N (epoch 1) but this client never re-keyed.
    msg = seal_update(config.mode, client.session, "c0", N, b"x")
    with pytest.raises(EpochError):
        server.decrypt(N, msg)


@pytest.mark.parametrize("name", REKEYING)
def test_handshake_for_wrong_epoch_rejected(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    init = client.start_handshake(0)  # signed for epoch 0
    with pytest.raises(EpochError):
        server.accept_handshake(N, init)  # server is in epoch 1


@pytest.mark.parametrize("name", REKEYING)
def test_replayed_handshake_rejected(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    init = client.start_handshake(0)
    client.finish_handshake(server.accept_handshake(0, init))
    # Replaying the epoch-0 init must not overwrite the established session.
    with pytest.raises(ReplayError):
        server.accept_handshake(1, init)
    assert server.decrypt(1, client.encrypt(1, b"still works")) == b"still works"


# --- Signature forgery -----------------------------------------------------------

def test_impostor_signing_with_own_key_rejected(config):
    server, _ = make_parties(config, ("victim",))
    mallory = ClientCrypto("victim", config, generate_identity(config), server.public_key)
    with pytest.raises(AuthenticationError):
        server.accept_handshake(0, mallory.start_handshake(0))


def test_bitflipped_signature_rejected(config):
    server, clients = make_parties(config, ("c0",))
    init = clients["c0"].start_handshake(0)
    for i in (0, len(init.signature) // 2, len(init.signature) - 1):
        with pytest.raises(AuthenticationError):
            server.accept_handshake(0, replace(init, signature=flip(init.signature, i)))
    with pytest.raises(AuthenticationError):
        server.accept_handshake(0, replace(init, signature=b""))


def test_substituted_kem_key_rejected(config):
    """MITM swaps in its own KEM public key but keeps the client's signature."""
    server, clients = make_parties(config, ("c0",))
    init = clients["c0"].start_handshake(0)
    attacker_pk = config.suite.kem.keygen().public
    with pytest.raises(AuthenticationError):
        server.accept_handshake(0, replace(init, kem_public=attacker_pk))


@pytest.mark.parametrize("name", REKEYING)
def test_signature_does_not_transfer_across_epochs(name):
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    init = clients["c0"].start_handshake(0)
    with pytest.raises(AuthenticationError):
        server.accept_handshake(N, replace(init, epoch=1))


def test_unknown_client_rejected(config):
    server, _ = make_parties(config, ())
    stranger = ClientCrypto("nobody", config, generate_identity(config), server.public_key)
    with pytest.raises(AuthenticationError):
        server.accept_handshake(0, stranger.start_handshake(0))


def test_forged_server_response_rejected(config):
    """MITM encapsulates to the client's KEM key itself and signs with its own key."""
    server, clients = make_parties(config, ("c0",))
    client = clients["c0"]
    init = client.start_handshake(0)
    genuine = server.accept_handshake(0, init)

    mitm_identity = generate_identity(config)
    ct, _ = config.suite.kem.encaps(init.kem_public)
    forged = replace(genuine, kem_ciphertext=ct)
    forged = replace(forged, signature=config.suite.sig.sign(
        mitm_identity.secret, _resp_payload(config.suite, init, forged)))
    with pytest.raises(AuthenticationError):
        client.finish_handshake(forged)
    with pytest.raises(AuthenticationError):
        client.finish_handshake(replace(genuine, kem_ciphertext=ct))  # genuine sig, swapped ct
    with pytest.raises(AuthenticationError):
        client.finish_handshake(replace(genuine, signature=flip(genuine.signature, 0)))

    client.finish_handshake(genuine)
    assert server.decrypt(0, client.encrypt(0, b"ok")) == b"ok"


# --- Config ----------------------------------------------------------------------

@pytest.mark.parametrize("mode,n", [
    (Mode.CLASSICAL, 0), (Mode.NAIVE_PQC, 3), (Mode.AMORTIZED, None), (Mode.AMORTIZED, 0),
])
def test_invalid_config_rejected(mode, n):
    with pytest.raises(ValueError):
        CryptoConfig(mode, rekey_interval=n)


def test_config_accepts_mode_strings():
    assert CryptoConfig("amortized", 4).mode is Mode.AMORTIZED
    assert CryptoConfig("naive_pqc").rekey_interval == 1


def test_classical_rekey_interval_accepted():
    assert CryptoConfig(Mode.CLASSICAL).rekey_interval is None
    assert CryptoConfig(Mode.CLASSICAL, 5).epoch_for_round(12) == 2


@pytest.mark.parametrize("name", list(CONFIGS))
def test_server_signs_response_in_every_mode(name):
    """All modes share one handshake shape: client signs the init, server signs the response."""
    config = CONFIGS[name]
    server, clients = make_parties(config, ("c0",))
    init = clients["c0"].start_handshake(0)
    resp = server.accept_handshake(0, init)
    sig = config.suite.sig
    assert sig.name == ("Ed25519" if config.mode is Mode.CLASSICAL else "ML-DSA-65")
    assert len(resp.signature) == (64 if config.mode is Mode.CLASSICAL else 3309)
    assert sig.verify(server.public_key, _resp_payload(config.suite, init, resp), resp.signature)


def test_wire_sizes():
    pqc = CONFIGS["amortized"]
    server, clients = make_parties(pqc, ("c0",))
    init = clients["c0"].start_handshake(0)
    resp = server.accept_handshake(0, init)
    assert len(init.kem_public) == 1184 and len(init.signature) == 3309
    assert len(resp.kem_ciphertext) == 1088 and len(resp.signature) == 3309
    clients["c0"].finish_handshake(resp)
    msg = clients["c0"].encrypt(0, bytes(100))
    assert len(msg.ciphertext) == 100 + 16  # GCM tag
    assert msg.nbytes == len(msg.to_bytes())
