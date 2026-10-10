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

## Demo

```bash
python scripts/demo.py     # ~65 s on an M-series laptop
```

The demo runs K=3 clients for 10 rounds with N=5, under all three modes. Classical also
re-keys every 5 rounds. Each client trains on 2,000 samples. The demo prints:
- a per-round log, marking each round as a handshake round or symmetric-only, with
  crypto time, bytes sent and accuracy;
- a summary table comparing the three modes.

It then runs a live attack. The attack flips a bit in one update and replays two
updates from an old round, one of them with its round number rewritten. The server must
reject all three. Finally it saves `results/demo_crypto_time_per_round.png` and
`results/demo_final_accuracy.png`.

## Experiments

```bash
python scripts/run_experiments.py --dry-run   # list runs + rough time estimate
python scripts/run_experiments.py --quick     # pipeline smoke test, ~1 min -> results/quick/
python scripts/run_experiments.py             # all four experiments, ~4 h on an M-series laptop
python scripts/run_experiments.py --exp 1,4   # just the main comparison + accuracy validation
```

| # | Experiment | Configurations |
|---|---|---|
| 1 | Main comparison | 3 modes, K=5, 20 rounds, N=5, 5 seeds |
| 2 | Re-key cadence (amortized) | R=60, N in {1, 2, 3, 5, 10, 15, 20, 30, 60}; N=1 is checked against a naive_pqc run at the same R |
| 3 | Client scaling | K in {3, 5, 10, 20}, all modes |
| 4 | Accuracy validation | Final and per-round accuracy per mode and seed, from experiment 1 |

How the runner works:
- **Caching and resume.** Each distinct configuration runs once, and its per-round CSV is
  cached in `results/raw/`. Experiments share these runs. Re-running the script resumes
  from the cache; `--force` re-runs everything.
- **Sequential runs.** Runs execute one after another so they don't distort each other's
  timings.
- **Training subset.** By default each client trains on 2,000 samples
  (`--samples-per-client 0` uses full MNIST shards, which takes about 7× longer).
- **Outputs.** Summary tables (`mean ± std` over seeds, in `.md` and `.csv`), the per-run
  `all_runs.csv`, and a `manifest.json` with library versions and machine details go to
  `results/tables/`.

## Report assets

Figures, LaTeX tables, TikZ diagrams and an IEEEtran skeleton are built into `report_assets/`
(see [report_assets/README.md](report_assets/README.md)):

```bash
brew install tectonic                      # TeX engine for the diagrams and main.pdf
python scripts/make_report_assets.py       # after scripts/run_experiments.py
```
