import csv
from dataclasses import replace

import numpy as np
import pytest
import torch

from pqfl.bench.meter import Meter, metered_suite
from pqfl.crypto import ClientCrypto, CryptoConfig, ServerCrypto, generate_identity
from pqfl.fl.data import TensorData, split_dirichlet, split_iid
from pqfl.fl.fedavg import deserialize_update, fedavg, serialize_update
from pqfl.fl.harness import CSV_COLUMNS, RunConfig, run
from pqfl.fl.model import SmallCNN, get_flat, set_flat


def synthetic(n: int, seed: int = 0) -> TensorData:
    g = torch.Generator().manual_seed(seed)
    return TensorData(torch.randn(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g))


# --- Model / serialization / FedAvg ----------------------------------------------

def test_flat_params_roundtrip():
    a, b = SmallCNN(), SmallCNN()
    set_flat(b, get_flat(a))
    np.testing.assert_array_equal(get_flat(a), get_flat(b))
    with pytest.raises(ValueError):
        set_flat(b, get_flat(a)[:-1])


def test_update_serialization_roundtrip_and_validation():
    w = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    payload = serialize_update(w, 123)
    out, n = deserialize_update(payload, 1000)
    np.testing.assert_array_equal(out, w)
    assert n == 123
    with pytest.raises(ValueError):
        deserialize_update(payload, 999)          # wrong model size
    with pytest.raises(ValueError):
        deserialize_update(payload[:-4], 1000)    # truncated
    bad = w.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        deserialize_update(serialize_update(bad, 1), 1000)


def test_fedavg_is_sample_weighted():
    a, b = np.zeros(4, np.float32), np.ones(4, np.float32)
    np.testing.assert_allclose(fedavg([(a, 1), (b, 3)]), 0.75)
    np.testing.assert_allclose(fedavg([(a, 5), (b, 5)]), 0.5)


@pytest.mark.parametrize("splitter", ["iid", "dirichlet"])
def test_splits_partition_the_data(splitter):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 1000)
    shards = (split_iid(1000, 7, rng) if splitter == "iid"
              else split_dirichlet(labels, 7, 0.3, rng))
    allidx = np.concatenate(shards)
    assert len(shards) == 7
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000


# --- Metering --------------------------------------------------------------------

def test_metered_suite_times_ops_and_interoperates():
    cfg = CryptoConfig("amortized", 2)
    meter = Meter()
    suite = metered_suite(cfg.suite, meter)
    server = ServerCrypto(cfg, generate_identity(cfg))           # unmetered server
    ident = generate_identity(cfg)
    server.register_client("c0", ident.public)
    client = ClientCrypto("c0", cfg, ident, server.public_key, suite=suite)  # metered client
    client.finish_handshake(server.accept_handshake(0, client.start_handshake(0)))
    assert server.decrypt(0, client.encrypt(0, b"hi")) == b"hi"
    # Client side: 1 KEM keygen, 1 decaps, 1 sign, 1 verify; server ops are not metered.
    assert {k: meter.counts[k] for k in ("keygen", "decaps", "sign", "verify", "encaps")} == \
        {"keygen": 1, "decaps": 1, "sign": 1, "verify": 1, "encaps": 0}
    assert all(meter.seconds[k] > 0 for k in ("keygen", "decaps", "sign", "verify"))


# --- Harness end to end ----------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data():
    return synthetic(120, 0), synthetic(50, 1)


def tiny_cfg(mode, n=None, rounds=4, **kw):
    return RunConfig(mode=mode, clients=3, rounds=rounds, rekey_interval=n, batch_size=20, **kw)


@pytest.mark.parametrize("mode,n,expected_hs", [
    ("classical", None, [3, 0, 0, 0]),
    ("classical", 2, [3, 0, 3, 0]),
    ("naive_pqc", None, [3, 3, 3, 3]),
    ("amortized", 2, [3, 0, 3, 0]),
])
def test_harness_rekey_schedule_and_csv(tmp_path, tiny_data, mode, n, expected_hs):
    out = tmp_path / "run.csv"
    rows = run(tiny_cfg(mode, n, out=str(out)), train_data=tiny_data[0],
               test_data=tiny_data[1], log=lambda _: None)
    assert [r["handshakes"] for r in rows] == expected_hs
    assert all(r["accepted"] == 3 and r["rejected"] == 0 for r in rows)

    with open(out) as fh:
        csv_rows = list(csv.DictReader(fh))
    assert list(csv_rows[0].keys()) == CSV_COLUMNS and len(csv_rows) == 4

    pqc = mode != "classical"
    for r in rows:
        # Per-client bytes: payload = plaintext + 12-byte nonce + 16-byte tag.
        assert r["bytes_payload"] == r["plaintext_bytes"] + 12 + 16
        hs = r["handshakes"] > 0
        assert r["bytes_key"] == (1184 if pqc else 32) * hs
        assert r["bytes_kem_ct"] == (1088 if pqc else 32) * hs
        assert r["bytes_sig"] == (2 * 3309 if pqc else 2 * 64) * hs
        assert (r["keygen_s"] > 0) == hs and (r["sign_s"] > 0) == hs
        parts = sum(r[f"{k}_s"] for k in ("keygen", "encaps", "decaps", "sign", "verify",
                                          "sym_encrypt", "sym_decrypt"))
        assert r["crypto_total_s"] >= parts * 0.999
        assert r["sym_encrypt_s"] > 0 and r["sym_decrypt_s"] > 0


def test_harness_is_deterministic_and_mode_independent(tiny_data):
    accs = []
    for mode, n in (("classical", None), ("naive_pqc", None), ("amortized", 2)):
        rows = run(tiny_cfg(mode, n), train_data=tiny_data[0], test_data=tiny_data[1],
                   log=lambda _: None)
        accs.append([r["test_acc"] for r in rows])
    assert accs[0] == accs[1] == accs[2]


def test_harness_drops_tampered_and_replayed_updates(tiny_data):
    captured = {}

    def attacker(r, msg):
        if r == 0 and msg.client_id == "client-0":
            captured["m"] = msg
            return replace(msg, ciphertext=bytes([msg.ciphertext[0] ^ 1]) + msg.ciphertext[1:])
        if r == 1 and msg.client_id == "client-1":
            return captured["m"]  # client-0's untampered round-0 update, replayed
        return msg

    logs = []
    rows = run(tiny_cfg("amortized", 2), train_data=tiny_data[0], test_data=tiny_data[1],
               channel=attacker, log=logs.append)
    assert [(r["accepted"], r["rejected"]) for r in rows] == [(2, 1), (2, 1), (3, 0), (3, 0)]
    assert sum("rejected update" in line for line in logs) == 2


def test_harness_keeps_global_model_when_all_updates_rejected(tiny_data):
    def drop_all(r, msg):
        return replace(msg, nonce=bytes(12))

    rows = run(tiny_cfg("classical", rounds=2), train_data=tiny_data[0],
               test_data=tiny_data[1], channel=drop_all, log=lambda _: None)
    assert all(r["accepted"] == 0 for r in rows)
    assert rows[0]["test_acc"] == rows[1]["test_acc"]  # untouched initial model
