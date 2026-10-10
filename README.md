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
