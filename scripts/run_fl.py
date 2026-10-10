"""Run one federated-learning session with encrypted client updates and log per-round CSV.

Example:
    python scripts/run_fl.py --mode amortized -K 10 --rounds 20 -N 5 --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pqfl.fl.harness import RunConfig, run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["classical", "naive_pqc", "amortized"])
    p.add_argument("-K", "--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("-N", "--rekey-interval", type=int, default=None,
                   help="re-key every N rounds (required for amortized; optional for "
                        "classical, default key once; naive_pqc is always 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--split", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--samples-per-client", type=int, default=None,
                   help="train on a random subset of each client's shard (quick runs)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None,
                   help="CSV path (default results/<mode>_K<K>_N<N>_seed<seed>.csv)")
    a = p.parse_args()

    n = 1 if a.mode == "naive_pqc" else a.rekey_interval
    out = a.out or f"results/{a.mode}_K{a.clients}_N{n or 'once'}_seed{a.seed}.csv"
    cfg = RunConfig(mode=a.mode, clients=a.clients, rounds=a.rounds,
                    rekey_interval=a.rekey_interval, seed=a.seed, local_epochs=a.local_epochs,
                    batch_size=a.batch_size, lr=a.lr, split=a.split, alpha=a.alpha,
                    samples_per_client=a.samples_per_client,
                    device=a.device, data_dir=a.data_dir, out=out)
    run(cfg)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
