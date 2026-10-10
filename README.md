# Amortized Post-Quantum Secure Aggregation for Federated Learning

## Layout

```
pqfl/crypto/   PQC + symmetric layer
pqfl/fl/       FedAvg harness, model, data split
pqfl/bench/    timing and byte logging
tests/
scripts/       demo and experiment runners
results/
```

## Setup

Tested on macOS 15 (Apple Silicon, arm64), Python 3.11.

### 1. Install the liboqs C library

```bash
brew install liboqs          # installed 0.16.0 → /opt/homebrew/opt/liboqs
```

`liboqs-python` loads `liboqs.dylib` from `/opt/homebrew/lib` automatically, so you
don't need `OQS_INSTALL_PATH` or a source build. **Keep the C library and
`liboqs-python` on the same major.minor version (0.16).** Otherwise the bindings
print a version-mismatch warning, and if they can't find a library at all they try
to download and build liboqs from source.

### 2. Create a virtualenv and install the Python dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The system default `python3` is 3.14. That is why the venv uses Homebrew's
`python3.11`: it is a conservative choice for wheel availability across torch and
the other dependencies.

### 3. Smoke test

```bash
python scripts/smoke_test.py
```

Expected output:

```
liboqs 0.16.0  |  liboqs-python 0.16.0.1
[ML-KEM-768] pk=1184 B  ct=1088 B  ss=32 B  secrets match: True
[ML-DSA-65] pk=1952 B  sig=3309 B  verify: True  tampered rejected: True
SMOKE TEST PASSED
```

The line `liboqs-python faulthandler is disabled` also appears. It is informational
and can be ignored.

## Crypto layer (`pqfl/crypto/`)

| Mode        | Handshake                  | Re-key                     | Per round   |
|-------------|----------------------------|----------------------------|-------------|
| `classical` | X25519 + Ed25519           | once, or every N rounds    | AES-256-GCM |
| `naive_pqc` | ML-KEM-768 + ML-DSA-65     | every round                | AES-256-GCM |
| `amortized` | ML-KEM-768 + ML-DSA-65     | every `rekey_interval` (N) | AES-256-GCM |

```python
from pqfl.crypto import ClientCrypto, CryptoConfig, ServerCrypto, generate_identity

cfg = CryptoConfig("amortized", rekey_interval=10)
server = ServerCrypto(cfg, generate_identity(cfg))
ident = generate_identity(cfg)
server.register_client("c0", ident.public)          # long-term key, pinned out of band
client = ClientCrypto("c0", cfg, ident, server.public_key)

for rnd in range(30):
    if client.needs_handshake(rnd):
        client.finish_handshake(server.accept_handshake(rnd, client.start_handshake(rnd)))
    update = server.decrypt(rnd, client.encrypt(rnd, b"serialized model update"))
```

- **Handshake.** The client signs `(suite, client_id, epoch, kem_pk)` with its identity
  key and the server verifies it. The server encapsulates and signs its response, which is
  bound to a hash of the client's init. The session key is HKDF-SHA256 over the shared
  secret, with the client id, epoch and transcript hash as info.
- **Per round.** `round_key = HKDF(session_key, info=round)`. Updates are encrypted with
  AES-256-GCM under a random 96-bit nonce. The AAD is `(mode, client_id, round, epoch)`.
- **Epochs** follow a fixed schedule, `epoch = round // N`. Clients that skip rounds
  therefore still agree with the server on which key is current.
- **Replay and reordering.** The server builds the AAD from its *own* current round and
  expected epoch, never from the message header. It also rejects any round at or before
  the last one it accepted from that client.
- **Byte counts.** Every wire message exposes `.nbytes` for benchmarking.

Design rationale, the threat analysis and the mutation-testing results are in
[DECISIONS.md](DECISIONS.md).

Run the tests with `python -m pytest tests/ -v`.

## FL harness (`pqfl/fl/`, `pqfl/bench/`)

This is a hand-rolled FedAvg setup. A small CNN (about 215k parameters, 842 KB per update)
trains on MNIST, split across K simulated clients. Every client update is serialized,
encrypted through the crypto layer, decrypted and authenticated by the server, and only
then averaged. Updates that fail authentication are dropped and counted. MNIST is
downloaded to `data/` on first use.

```bash
python scripts/run_fl.py --mode amortized -K 10 --rounds 20 -N 5 --seed 0
python scripts/run_fl.py --mode naive_pqc -K 10 --rounds 20 --seed 0
python scripts/run_fl.py --mode classical -K 10 --rounds 20 --seed 0         # key once
python scripts/run_fl.py --mode classical -K 10 --rounds 20 -N 5 --seed 0    # re-key every 5
# quick run: --samples-per-client 2000; non-IID split: --split dirichlet --alpha 0.5
```

Each run writes one row per round to `results/<mode>_K<K>_N<N>_seed<seed>.csv`:

| Column(s) | Meaning |
|---|---|
| `mode, K, N, seed, round, epoch` | Run configuration and the key epoch for this round |
| `handshakes, accepted, rejected` | Handshakes run this round; updates accepted or rejected by the server |
| `keygen_s, encaps_s, decaps_s, sign_s, verify_s` | Time in each primitive (KEM keygen only; identity keys are set up once, before round 0) |
| `sym_encrypt_s, sym_decrypt_s` | `ClientCrypto.encrypt` / `ServerCrypto.decrypt`: round-key HKDF + AES-256-GCM |
| `crypto_total_s` | All protocol calls, including handshake key derivation and overhead. Excludes training, serialization and evaluation |
| `bytes_key, bytes_kem_ct, bytes_sig, bytes_payload` | Per client: KEM public key, KEM ciphertext, both signatures, encrypted update (nonce + ciphertext + tag) |
| `bytes_wire_total` | Per client: all protocol messages including headers, both directions |
| `plaintext_bytes` | Per client: serialized update before encryption |
| `train_s, eval_s, wall_s, test_acc` | Local training (all clients), test-set evaluation, the whole round, accuracy after aggregation |

How to read the numbers:
- **Timing** uses `time.perf_counter`. Clients run one after another in a single
  process, so time columns are **summed over all K clients and the server**.
- **Warm-up.** Each primitive runs once, untimed, before round 0. The first call pays
  one-off library start-up costs, such as about 4 ms for OpenSSL's first X25519 keygen.
- **Byte columns are per-client means.**
- **The model broadcast from server to clients is not encrypted**, and it is not counted.
