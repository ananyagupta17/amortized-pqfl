"""Controlled rerun of the main comparison with the three modes interleaved per seed.

The main experiment ran all classical seeds, then all naive_pqc seeds, then all amortized
seeds, so slow timing drift can leak into small between-mode gaps (DECISIONS.md, run-order
drift). This script runs the three modes back to back for each seed, rotating which mode goes
first, in one process. The AES-GCM path is identical code in every mode, so its per-mode
time is the drift control: it should agree across modes within noise.

    python scripts/check_main_interleaved.py      # K=5, R=20, N=5, 5 seeds (~15 min)
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

MODES = ("classical", "naive_pqc", "amortized")


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = a - b
    sd = float(d.std(ddof=1))
    return {"diff_mean": float(d.mean()), "diff_sd": sd, "diff_pct": float(100 * d.mean() / b.mean()),
            "paired_t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("-K", type=int, default=5)
    p.add_argument("-N", type=int, default=5)
    p.add_argument("--samples-per-client", type=int, default=2000)
    p.add_argument("--out", default="results/main_interleaved")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_mnist("data")
    order = [(MODES[(s + i) % 3], s) for s in range(a.seeds) for i in range(3)]  # rotate first mode

    rows = []
    for i, (mode, seed) in enumerate(order):
        n = None if mode == "naive_pqc" else a.N
        path = out / f"{mode}_K{a.K}_N{1 if n is None else n}_seed{seed}_R{a.rounds}.csv"
        run(RunConfig(mode=mode, clients=a.K, rounds=a.rounds, rekey_interval=n, seed=seed,
                      samples_per_client=a.samples_per_client, out=str(path)),
            train_data=train, test_data=test, log=lambda _: None)
        df = pd.read_csv(path)
        sym = (df.sym_encrypt_s + df.sym_decrypt_s).sum() * 1e3
        rows.append({"order": i, "mode": mode, "seed": seed,
                     "crypto_ms": df.crypto_total_s.sum() * 1e3,
                     "handshake_ms": df.crypto_total_s.sum() * 1e3 - sym, "symmetric_ms": sym,
                     "aes_us_per_client_round": 1e3 * sym / (a.K * a.rounds),
                     "final_acc": df.test_acc.iloc[-1] * 100})
        r = rows[-1]
        print(f"[{i + 1:>2}/{len(order)}] {mode:<9} seed={seed}  crypto={r['crypto_ms']:6.2f} ms  "
              f"handshake={r['handshake_ms']:6.2f}  AES-GCM={r['symmetric_ms']:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    by = {m: df[df["mode"] == m].set_index("seed").sort_index() for m in MODES}
    summary = {"K": a.K, "rounds": a.rounds, "N": a.N, "seeds": a.seeds,
               "order": "modes interleaved per seed, first mode rotated", "per_mode": {}, "gaps": {}}
    for m in MODES:
        summary["per_mode"][m] = {c: {"mean": float(by[m][c].mean()), "sd": float(by[m][c].std())}
                                  for c in ("crypto_ms", "handshake_ms", "symmetric_ms",
                                            "aes_us_per_client_round", "final_acc")}
    for x, y in (("amortized", "classical"), ("naive_pqc", "amortized"), ("naive_pqc", "classical")):
        summary["gaps"][f"{x}-{y}"] = {c: paired(by[x][c], by[y][c])
                                       for c in ("crypto_ms", "handshake_ms", "symmetric_ms")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nper mode (mean ± sd over seeds):")
    for m in MODES:
        s = summary["per_mode"][m]
        print(f"  {m:<9} crypto {s['crypto_ms']['mean']:6.2f} ± {s['crypto_ms']['sd']:4.2f}  "
              f"handshake {s['handshake_ms']['mean']:6.2f} ± {s['handshake_ms']['sd']:4.2f}  "
              f"AES-GCM {s['symmetric_ms']['mean']:6.2f} ± {s['symmetric_ms']['sd']:4.2f} ms "
              f"({s['aes_us_per_client_round']['mean']:.1f} us/client-round)  acc {s['final_acc']['mean']:.2f}")
    print("paired gaps:")
    for k, g in summary["gaps"].items():
        print("  " + k + ": " + "; ".join(
            f"{c} {v['diff_mean']:+.2f} ± {v['diff_sd']:.2f} ms ({v['diff_pct']:+.1f}%, t={v['paired_t']:+.2f})"
            for c, v in g.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
