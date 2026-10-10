"""Smoke test for the PQC layer: ML-KEM-768 KEM round trip and ML-DSA-65 sign/verify."""

import sys

import oqs

KEM_ALG = "ML-KEM-768"
SIG_ALG = "ML-DSA-65"


def check_kem() -> None:
    assert KEM_ALG in oqs.get_enabled_kem_mechanisms(), f"{KEM_ALG} not enabled in liboqs"
    with oqs.KeyEncapsulation(KEM_ALG) as receiver, oqs.KeyEncapsulation(KEM_ALG) as sender:
        pk = receiver.generate_keypair()
        ct, ss_sender = sender.encap_secret(pk)
        ss_receiver = receiver.decap_secret(ct)
        assert ss_sender == ss_receiver, "shared secrets do not match"
        print(f"[{KEM_ALG}] pk={len(pk)} B  ct={len(ct)} B  ss={len(ss_sender)} B  "
              f"secrets match: {ss_sender == ss_receiver}")


def check_sig() -> None:
    assert SIG_ALG in oqs.get_enabled_sig_mechanisms(), f"{SIG_ALG} not enabled in liboqs"
    msg = b"pqfl smoke test message"
    with oqs.Signature(SIG_ALG) as signer, oqs.Signature(SIG_ALG) as verifier:
        pk = signer.generate_keypair()
        sig = signer.sign(msg)
        ok = verifier.verify(msg, sig, pk)
        assert ok, "signature failed to verify"
        tampered = verifier.verify(msg + b"!", sig, pk)
        assert not tampered, "tampered message verified"
        print(f"[{SIG_ALG}] pk={len(pk)} B  sig={len(sig)} B  "
              f"verify: {ok}  tampered rejected: {not tampered}")


def main() -> int:
    print(f"liboqs {oqs.oqs_version()}  |  liboqs-python {oqs.oqs_python_version()}")
    check_kem()
    check_sig()
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
