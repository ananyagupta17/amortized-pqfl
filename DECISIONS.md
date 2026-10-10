# Design decisions: crypto layer (`pqfl/crypto/`)

This file records design choices in the protocol that the task specification did not
fix, or where we deliberately departed from it. It also records the evidence that the
test suite actually checks these properties.

## Protocol summary

| Mode        | Primitives                         | Handshake cadence            | Per round   |
|-------------|------------------------------------|------------------------------|-------------|
| `classical` | X25519 (as a KEM) + Ed25519        | once, or every N rounds      | AES-256-GCM |
| `naive_pqc` | ML-KEM-768 + ML-DSA-65             | every round (N = 1)          | AES-256-GCM |
| `amortized` | ML-KEM-768 + ML-DSA-65             | every N rounds               | AES-256-GCM |

All three modes run the same code path: `ClientCrypto` and `ServerCrypto` in
`protocol.py`. They differ only in the `Suite` (primitives) and `rekey_interval`.

```
Client                                                   Server
  (kem_pk, kem_sk) <- KEM.keygen()
  sig_C = Sign(sk_C, "init" | suite | id | epoch | kem_pk)
  ---------------- HandshakeInit(id, epoch, kem_pk, sig_C) --------------->
                                         Verify(pk_C, ...)  (pk_C pinned at registration)
                                         check epoch == round // N, epoch > last epoch
                                         (ct, ss) <- KEM.encaps(kem_pk)
                         sig_S = Sign(sk_S, "resp" | suite | H(init) | id | epoch | ct)
  <------------- HandshakeResponse(id, epoch, ct, sig_S) -----------------
  Verify(pk_S, ...)
  ss <- KEM.decaps(kem_sk, ct)
  Both: session_key = HKDF-SHA256(ss, info = "session" | suite | id | epoch | H(init | resp))

Each round r:
  round_key = HKDF-SHA256(session_key, info = "round" | r)
  ct = AES-256-GCM(round_key, random 96-bit nonce, update, aad = "aad" | mode | id | r | epoch)
```

## Decision 1: the server signs its handshake response, in every mode

**Spec:** only the client signs its KEM public key, and the server verifies it.

**Decision:** the server also signs its response. The signature covers a hash of the
client's init, so the response is bound to that specific handshake. The client verifies
it before decapsulating. In `classical` mode the server signs with Ed25519, and in the
PQC modes with ML-DSA-65, so the handshake has the same structure in all modes. This is
pinned by `test_server_signs_response_in_every_mode` and
`test_forged_server_response_rejected[*]`.

**Motivating attack (MITM on an unauthenticated response).** Suppose only the client
signs. An active network attacker can then do the following:

1. The client sends `HandshakeInit(id, epoch, kem_pk, sig_C)`. The attacker forwards it
   untouched, and the server accepts it because `sig_C` is valid.
2. The server replies with `ct_S`, which encapsulates `ss_S` to `kem_pk`.
3. The attacker drops `ct_S` and runs `(ct_M, ss_M) <- Encaps(kem_pk)` itself. Anyone can
   do this, because `kem_pk` is public. The attacker sends `ct_M` to the client.
4. The client decapsulates `ct_M`, gets `ss_M`, and derives a session key that the
   **attacker knows**.
5. Every model update the client encrypts in this epoch is readable by the attacker. The
   server can't decrypt these updates and rejects them. By then, though, confidentiality
   is already lost, and the update leaks information about the client's private training
   data.

The client signature does not help here, because it authenticates the client to the
server, not the other way around. With the server signature, step 3 fails. The attacker
cannot produce a valid `sig_S` over `ct_M` without the server's signing key. Binding
`H(init)` into the signed payload also stops the attacker from replaying a genuine
server response from a different handshake.

**Cost:** one extra sign (server) and one extra verify (client) per handshake. For
ML-DSA-65 that adds 3,309 bytes on the downlink. The benchmarks include this cost.

## Decision 2: long-term identity keys are never regenerated

Each client and the server has one long-term signing keypair, Ed25519 or ML-DSA-65
depending on the mode. Each client's public key is registered with the server out of
band (`ServerCrypto.register_client`), and the server's public key is given to every
client. Keys are pinned in this way rather than issued by a PKI.

A re-key, including the per-round re-key in `naive_pqc`, generates a **fresh KEM
keypair** but reuses the identity key to sign it. Rotating identity keys every round
would be meaningless: a newly generated key has nothing to vouch for it unless it is
signed by a key the server already trusts. As a result, "full ML-KEM + ML-DSA per round"
means one KEM keygen, one encapsulation, one decapsulation, two signs and two verifies
per round. It does not include signature key generation.

Because the KEM keypair is new in every epoch, compromising one epoch's KEM secret key
does not expose other epochs. Compromising a long-term signing key lets an attacker
impersonate that party in future handshakes, but past sessions stay secure.

## Decision 3: epochs follow a fixed schedule, `epoch = round // N`

The epoch is a function of the round number, not a count of handshakes. Real FL samples
a subset of clients each round, so a client may join late or skip rounds. With a fixed
schedule, the client and the server always agree on which key epoch a round belongs to,
without tracking each other's history. `test_partial_participation_late_joiner` covers
this.

- `naive_pqc`: N = 1, so epoch = round.
- `amortized`: N ≥ 1 is required.
- `classical`: N ≥ 1, or `None` to key once (epoch is always 0). See Decision 5.

The server accepts a handshake only if its epoch equals `round // N` and is strictly
greater than the epoch of the client's current session. The second condition stops a
replayed old handshake init from overwriting a live session; the replay could not
compromise the session key, but it would be a denial of service. The epoch is included
in the signed init payload, so a signature from one epoch cannot be reused in another.

## Decision 4: replay and reordering protection rests on server state, not the header

The `round` and `epoch` fields in an `EncryptedUpdate` header are **not trusted**.
`ServerCrypto.decrypt(round, update)` works as follows:

1. It derives the round key and builds the AAD from the server's **own** current round
   and the expected epoch, `round // N`. The client's session key is already bound to
   its epoch.
2. It requires the header's round and epoch to equal those expected values. This gives
   a clear error, and it means a lying header cannot get through even when the AEAD
   would still verify.
3. It remembers the last round it accepted from each client and rejects any round at or
   before it. This blocks duplicates within a round and rolls back to earlier rounds. It
   needs O(clients) state and never grows over time.
4. It records a round as used only after successful authentication. A forged or
   tampered message therefore cannot "burn" the round for the genuine update.

Results:

- An update replayed unchanged from an earlier round is rejected with `ReplayError`.
- The same update with its header rewritten to the current round is rejected with
  `AuthenticationError`, because both the round key and the AAD differ.
- An update sent early for a future round is rejected.
- An update encrypted under a stale epoch key is rejected with `EpochError` or
  `AuthenticationError`.

Nonces are 96 bits from `os.urandom`. Each round key is unique per
(client, epoch, round) and normally encrypts a single message, so a random nonce
collision has probability about 2^-96 per pair of messages.

## Decision 5: classical supports re-keying every N rounds

**Original spec:** classical runs X25519 + Ed25519 once.

**Decision:** `CryptoConfig("classical", rekey_interval=N)` re-keys on the same
`epoch = round // N` schedule as `amortized`. Classical and amortized therefore differ
only in their primitives, which gives a like-for-like comparison of classical and
post-quantum costs at the same re-key cadence. `rekey_interval=None` is still the
default for classical and means key once, which matches the original spec. All the
epoch and replay tests run against both `amortized` and `classical` with N = 3.

`naive_pqc` keeps N fixed at 1. `amortized` requires an explicit N.

## Decision 6: our wrapper length-checks inputs that liboqs-python does not

When we probed liboqs-python 0.16.0.1 against liboqs 0.16.0, we found that
`KeyEncapsulation.encap_secret()` **silently accepted an ML-KEM-768 public key one byte
short**: 1,183 bytes instead of 1,184. It returned a result instead of raising an error.
The binding passes the buffer to the C library without checking its length. The C
library then reads a fixed 1,184 bytes from a shorter buffer, which is out-of-bounds
read behaviour that depends on memory layout.

`MLKEM.encaps` and `MLKEM.decaps` in `primitives.py` now check the public-key and
ciphertext lengths against `details["length_public_key"]` and
`details["length_ciphertext"]`, and raise `ValueError` on a mismatch. The server turns a
malformed key in a handshake into an `AuthenticationError`.

The same probe found that `Signature.verify()` returns `False` for wrong-length
signatures and public keys instead of raising an error, so the signature wrapper needs
no extra checks. The X25519 wrapper relies on `cryptography`, which rejects low-order
points by raising `ValueError` when the shared secret is all zeros.

## Test evidence: mutation testing

To confirm the tests check the security properties rather than passing trivially, we
injected each fault below into `protocol.py`, ran the suite, and restored the file. The
unmodified suite has **85 passing tests**, covering the 4 configurations: `classical`,
`naive_pqc`, `amortized` (N = 3) and `classical_rekey` (N = 3).

| # | Injected fault                                                                         | Result              |
|---|----------------------------------------------------------------------------------------|---------------------|
| M1 | Round removed from the round key; round and epoch removed from the AAD; header and last-round checks removed | **12 failed** / 73 passed |
| M2 | Server skips verifying the client's handshake signature                               | **14 failed** / 71 passed |
| M3 | Client skips verifying the server's response signature (the Decision 1 attack)         | **4 failed** / 81 passed  |
| M4 | Server forgets the last accepted round (no duplicate or rollback check)               | **5 failed** / 80 passed  |
| M5 | Header epoch check removed                                                             | **4 failed** / 81 passed  |

M5 shows something worth knowing. The server builds the AAD from its own expected epoch,
so editing only the header's epoch field does not break the AEAD. The explicit header
check is the only thing that stops an update with a false epoch label from being
accepted. The forged label is cosmetic, since the key and AAD are still correct, but
accepting a message whose metadata is false would be a bug, so the check stays.

An earlier version of the suite (58 tests, 3 configurations, before Decision 5) gave
8 failures for M1 and 10 failures for M2.
