"""Controlled check: amortized N=1 vs naive_pqc, interleaved in one process.

Both configurations do identical work (one handshake per client per round). In the main
experiment run they executed in different processes at different times; this script runs
them back to back in alternating ABBA order across seeds so slow drift (thermal state,
background load, process start-up) cancels out of the paired difference.

    python scripts/check_n1_vs_naive.py            # K=5, R=60, 5 seeds (~30 min)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pqfl.fl.data import load_mnist  # noqa: E402
from pqfl.fl.harness import RunConfig, run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("-K", type=int, default=5)
    p.add_argument("--samples-per-client", type=int, default=2000)
    p.add_argument("--out", default="results/n1_vs_naive")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_mnist("data")
    configs = {"A": ("amortized", 1), "B": ("naive_pqc", None)}
    order = []
    for s in range(a.seeds):  # ABBA ABBA ...: A,B for even seeds, B,A for odd seeds
        order += [(k, s) for k in ("AB" if s % 2 == 0 else "BA")]

    rows = []
    for i, (key, seed) in enumerate(order):
        mode, n = configs[key]
        path = out / f"{mode}_K{a.K}_N1_seed{seed}_R{a.rounds}.csv"
        run(RunConfig(mode=mode, clients=a.K, rounds=a.rounds, rekey_interval=n, seed=seed,
                      samples_per_client=a.samples_per_client, out=str(path)),
            train_data=train, test_data=test, log=lambda _: None)
        df = pd.read_csv(path)
        sym = (df.sym_encrypt_s + df.sym_decrypt_s).sum() * 1e3
        rows.append({"order": i, "mode": mode, "seed": seed,
                     "crypto_ms": df.crypto_total_s.sum() * 1e3,
                     "handshake_ms": df.crypto_total_s.sum() * 1e3 - sym, "symmetric_ms": sym,
                     "handshakes": int(df.handshakes.sum()),
                     "handshake_bytes": float((df.bytes_key + df.bytes_kem_ct + df.bytes_sig).sum())})
        r = rows[-1]
        print(f"[{i + 1:>2}/{len(order)}] {mode:<9} seed={seed}  crypto={r['crypto_ms']:7.2f} ms  "
              f"handshake={r['handshake_ms']:7.2f}  symmetric={r['symmetric_ms']:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    am = df[df["mode"] == "amortized"].set_index("seed").sort_index()
    nv = df[df["mode"] == "naive_pqc"].set_index("seed").sort_index()
    summary = {"K": a.K, "rounds": a.rounds, "seeds": a.seeds, "order": "ABBA by seed",
               "handshakes_identical": bool((am.handshakes == nv.handshakes).all()),
               "handshake_bytes_identical": bool(np.allclose(am.handshake_bytes, nv.handshake_bytes))}
    for c in ("crypto_ms", "handshake_ms", "symmetric_ms"):
        d = am[c] - nv[c]
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        summary[c] = {"amortized_mean": float(am[c].mean()), "amortized_sd": float(am[c].std()),
                      "naive_mean": float(nv[c].mean()), "naive_sd": float(nv[c].std()),
                      "paired_diff_mean": float(d.mean()), "paired_diff_sd": float(d.std(ddof=1)),
                      "diff_pct": float(100 * d.mean() / nv[c].mean()), "paired_t": t}
        print(f"{c:<13} amortized {am[c].mean():7.2f} ± {am[c].std():5.2f}   naive {nv[c].mean():7.2f} "
              f"± {nv[c].std():5.2f}   diff {d.mean():+6.2f} ± {d.std(ddof=1):5.2f} ms "
              f"({100 * d.mean() / nv[c].mean():+.1f}%), paired t = {t:+.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"handshakes identical: {summary['handshakes_identical']}; "
          f"handshake bytes identical: {summary['handshake_bytes_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
