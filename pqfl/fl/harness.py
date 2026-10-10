"""FedAvg over MNIST with every client update sent through the crypto layer.

Per round, for each of the K simulated clients (sequentially, in one process):
  1. handshake if the client's key epoch is stale   (crypto, timed)
  2. local training from the global weights          (not crypto, timed separately)
  3. serialize + encrypt the update                  (serialize untimed, encrypt timed)
  4. server decrypts + authenticates the update      (crypto, timed); rejects are dropped
Then the server averages the accepted updates (FedAvg) and evaluates on the test set.

Crypto timings are summed over all clients and the server for the round. Byte columns are
per-client means for the round.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from pqfl.bench.meter import BYTE_KEYS, TIME_KEYS, Meter, metered_suite
from pqfl.crypto import (
    ClientCrypto,
    CryptoConfig,
    CryptoError,
    EncryptedUpdate,
    ServerCrypto,
    generate_identity,
)
from pqfl.fl.data import TensorData, load_mnist, split_dirichlet, split_iid
from pqfl.fl.fedavg import deserialize_update, evaluate, fedavg, local_train, serialize_update
from pqfl.fl.model import SmallCNN, get_flat, set_flat

CSV_COLUMNS = (
    ["mode", "K", "N", "seed", "round", "epoch", "handshakes", "accepted", "rejected"]
    + [f"{k}_s" for k in TIME_KEYS]
    + [f"bytes_{k}" for k in BYTE_KEYS]
    + ["plaintext_bytes", "train_s", "eval_s", "wall_s", "test_acc"]
)

Channel = Callable[[int, EncryptedUpdate], EncryptedUpdate]


@dataclass
class RunConfig:
    mode: str
    clients: int = 10
    rounds: int = 5
    rekey_interval: int | None = None
    seed: int = 0
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    split: str = "iid"           # "iid" | "dirichlet"
    alpha: float = 0.5           # Dirichlet concentration (split="dirichlet" only)
    samples_per_client: int | None = None  # cap each shard (random subset) for quick runs
    device: str = "cpu"
    data_dir: str = "data"
    out: str | None = None       # CSV path; None = don't write


def _warm_up(suite) -> None:
    """Run each primitive once untimed: the first call pays one-off library init costs
    (e.g. ~4 ms for OpenSSL's first X25519 keygen) that would otherwise land in round 0."""
    kem_kp, sig_kp = suite.kem.keygen(), suite.sig.keygen()
    ct, _ = suite.kem.encaps(kem_kp.public)
    suite.kem.decaps(kem_kp.secret, ct)
    suite.sig.verify(sig_kp.public, b"warm-up", suite.sig.sign(sig_kp.secret, b"warm-up"))
    from pqfl.crypto.primitives import aead_decrypt, aead_encrypt, hkdf_sha256
    key = hkdf_sha256(b"\0" * 32, b"warm-up")
    aead_decrypt(key, *aead_encrypt(key, b"warm-up", b""), b"")


def _client_generator(seed: int, round_num: int, client: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + round_num * 1_009 + client)


def run(cfg: RunConfig, *, train_data: TensorData | None = None,
        test_data: TensorData | None = None, channel: Channel | None = None,
        on_round: Callable[[dict], None] | None = None,
        log: Callable[[str], None] = print) -> list[dict]:
    """Run one FL session.

    `channel(round, update) -> update` simulates the network (tests and the demo use it to
    tamper with or replay updates in transit). `on_round(row)` is called after each round.
    """
    crypto_cfg = CryptoConfig(cfg.mode, cfg.rekey_interval)
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    if train_data is None or test_data is None:
        train_data, test_data = load_mnist(cfg.data_dir)
    if cfg.split == "iid":
        shards = split_iid(len(train_data), cfg.clients, rng)
    elif cfg.split == "dirichlet":
        shards = split_dirichlet(train_data.y.numpy(), cfg.clients, cfg.alpha, rng)
    else:
        raise ValueError(f"unknown split {cfg.split!r}")
    if cfg.samples_per_client is not None:
        shards = [np.sort(rng.choice(s, min(len(s), cfg.samples_per_client), replace=False))
                  for s in shards]
    client_data = [train_data.subset(s) for s in shards]

    # --- Setup: long-term identity keys (one-off, reported but not in per-round rows).
    _warm_up(crypto_cfg.suite)
    meter = Meter()
    suite = metered_suite(crypto_cfg.suite, meter)
    server = ServerCrypto(crypto_cfg, suite.sig.keygen(), suite=suite)
    clients = []
    for k in range(cfg.clients):
        cid = f"client-{k}"
        identity = suite.sig.keygen()
        server.register_client(cid, identity.public)
        clients.append(ClientCrypto(cid, crypto_cfg, identity, server.public_key, suite=suite))
    log(f"[setup] mode={crypto_cfg.mode.value} suite={suite.name} K={cfg.clients} "
        f"N={crypto_cfg.rekey_interval} shards={[len(d) for d in client_data]} "
        f"identity keygen={meter.seconds['sig_keygen'] * 1e3:.2f} ms")

    global_model, worker = SmallCNN(), SmallCNN()
    global_flat = get_flat(global_model)
    n_params = global_flat.size
    channel = channel or (lambda _r, msg: msg)

    writer = fh = None
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(cfg.out, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    rows = []
    try:
        for r in range(cfg.rounds):
            meter.reset()
            t_round = perf_counter()
            train_s = 0.0
            handshakes = rejected = plaintext_bytes = 0
            accepted: list[tuple[np.ndarray, int]] = []

            for k, client in enumerate(clients):
                if client.needs_handshake(r):
                    with meter.time("crypto_total"):
                        init = client.start_handshake(r)
                    with meter.time("crypto_total"):
                        resp = server.accept_handshake(r, init)
                    with meter.time("crypto_total"):
                        client.finish_handshake(resp)
                    handshakes += 1
                    meter.add_bytes("key", len(init.kem_public))
                    meter.add_bytes("kem_ct", len(resp.kem_ciphertext))
                    meter.add_bytes("sig", len(init.signature) + len(resp.signature))
                    meter.add_bytes("wire_total", init.nbytes + resp.nbytes)

                t0 = perf_counter()
                weights = local_train(worker, global_flat, client_data[k],
                                      epochs=cfg.local_epochs, batch_size=cfg.batch_size,
                                      lr=cfg.lr, generator=_client_generator(cfg.seed, r, k),
                                      device=device)
                train_s += perf_counter() - t0

                payload = serialize_update(weights, len(client_data[k]))
                plaintext_bytes += len(payload)
                with meter.time("sym_encrypt", "crypto_total"):
                    msg = client.encrypt(r, payload)
                meter.add_bytes("payload", len(msg.nonce) + len(msg.ciphertext))
                meter.add_bytes("wire_total", msg.nbytes)

                msg = channel(r, msg)
                try:
                    with meter.time("sym_decrypt", "crypto_total"):
                        opened = server.decrypt(r, msg)
                    accepted.append(deserialize_update(opened, n_params))
                except (CryptoError, ValueError) as e:
                    rejected += 1
                    log(f"[round {r}] rejected update from {msg.client_id}: {e}")

            if accepted:
                global_flat = fedavg(accepted)
            set_flat(global_model, global_flat)
            t0 = perf_counter()
            acc = evaluate(global_model, test_data, device)
            eval_s = perf_counter() - t0
            wall_s = perf_counter() - t_round

            K = cfg.clients
            row = {
                "mode": crypto_cfg.mode.value, "K": K, "N": crypto_cfg.rekey_interval,
                "seed": cfg.seed, "round": r, "epoch": crypto_cfg.epoch_for_round(r),
                "handshakes": handshakes, "accepted": len(accepted), "rejected": rejected,
                **{f"{key}_s": meter.seconds.get(key, 0.0) for key in TIME_KEYS},
                **{f"bytes_{key}": meter.bytes.get(key, 0) / K for key in BYTE_KEYS},
                "plaintext_bytes": plaintext_bytes / K,
                "train_s": train_s, "eval_s": eval_s, "wall_s": wall_s, "test_acc": acc,
            }
            rows.append(row)
            if writer:
                writer.writerow(row)
                fh.flush()
            if on_round:
                on_round(row)
            log(f"[round {r}] epoch={row['epoch']} hs={handshakes} acc={acc:.4f} "
                f"crypto={row['crypto_total_s'] * 1e3:.2f} ms train={train_s:.2f} s "
                f"wall={wall_s:.2f} s up+down/client={row['bytes_wire_total']:.0f} B")
    finally:
        if fh:
            fh.close()
    return rows
