"""Run the paper experiments and write raw per-round CSVs plus summary tables.

    python scripts/run_experiments.py                 # everything (~2 h on an M-series laptop)
    python scripts/run_experiments.py --exp 1,4       # main comparison + accuracy validation
    python scripts/run_experiments.py --quick         # smoke test of the whole pipeline (~2 min)
    python scripts/run_experiments.py --dry-run       # list the runs and a rough time estimate

Experiments
  1. Main comparison: classical / naive_pqc / amortized, K=5, 20 rounds, N=5, 5 seeds.
  2. Re-key cadence sweep (amortized): N in {1, 2, 3, 5, 10, 15, 20, 30, 60} over R=60 rounds
     (longer than the other experiments so large N is not capped by R); N=1 is checked against a
     naive_pqc reference run at the same R.
  3. Client scaling: K in {3, 5, 10, 20}, all modes.
  4. Accuracy validation: final accuracy per mode and seed (from experiment 1's runs).

Each distinct (mode, K, N, seed, rounds, samples) configuration runs once. Results are cached in
results/raw/ and shared between experiments; re-running the script resumes where it stopped
(--force re-runs). Runs execute sequentially in one process so they don't compete for CPU and
distort each other's crypto timings.

Classical re-keys on the same N as amortized (DECISIONS.md, decision 5), so the two differ only in
primitives. Crypto times in the raw CSVs are summed over all clients and the server.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
RAW, TABLES = RESULTS / "raw", RESULTS / "tables"
MODES = ("classical", "naive_pqc", "amortized")


@dataclass(frozen=True)
class Spec:
    mode: str
    K: int
    N: int | None  # None only for naive_pqc (always 1)
    seed: int
    rounds: int
    samples: int | None

    @property
    def path(self) -> Path:
        n = 1 if self.mode == "naive_pqc" else self.N
        s = "full" if self.samples is None else self.samples
        return RAW / f"{self.mode}_K{self.K}_N{n}_seed{self.seed}_R{self.rounds}_S{s}.csv"


def plan(exps: set[int], seeds: list[int], rounds: int, sweep_rounds: int, samples: int | None,
         K_main: int, N_main: int, N_sweep: list[int], K_sweep: list[int]) -> dict[int, list[Spec]]:
    def spec(mode, K, N, seed, R=rounds):
        return Spec(mode, K, None if mode == "naive_pqc" else N, seed, R, samples)

    main = [spec(m, K_main, N_main, s) for m in MODES for s in seeds]
    out = {}
    if 1 in exps or 4 in exps:
        out[1] = main
    if 2 in exps:
        out[2] = ([spec("amortized", K_main, n, s, sweep_rounds) for n in N_sweep for s in seeds]
                  + [spec("naive_pqc", K_main, None, s, sweep_rounds) for s in seeds])
    if 3 in exps:
        out[3] = [spec(m, k, N_main, s) for k in K_sweep for m in MODES for s in seeds]
    return out


def estimate_seconds(spec: Spec) -> float:
    """Very rough: ~0.22 ms per training sample-step on CPU + ~0.7 s eval/round."""
    samples = spec.samples or 60_000 // spec.K
    return spec.rounds * (spec.K * samples * 0.22e-3 + 0.7)


def run_one(spec: Spec, train, test, force: bool) -> pd.DataFrame:
    from pqfl.fl.harness import RunConfig, run

    if spec.path.exists() and not force:
        df = pd.read_csv(spec.path)
        if len(df) == spec.rounds:
            return df
    part = spec.path.with_suffix(".csv.part")
    cfg = RunConfig(mode=spec.mode, clients=spec.K, rounds=spec.rounds, rekey_interval=spec.N,
                    seed=spec.seed, samples_per_client=spec.samples, out=str(part))
    run(cfg, train_data=train, test_data=test, log=lambda _: None)
    os.replace(part, spec.path)  # only complete runs ever appear under their final name
    return pd.read_csv(spec.path)


# --- Per-run metrics and aggregation ---------------------------------------------

def summarize_run(spec: Spec, df: pd.DataFrame) -> dict:
    hs_bytes = df.bytes_key + df.bytes_kem_ct + df.bytes_sig
    sym = df.sym_encrypt_s + df.sym_decrypt_s
    return {
        "mode": spec.mode, "K": spec.K, "N": 1 if spec.mode == "naive_pqc" else spec.N,
        "seed": spec.seed, "rounds": spec.rounds,
        "handshakes": int(df.handshakes.sum()),
        "crypto_ms": df.crypto_total_s.sum() * 1e3,
        "handshake_ms": (df.crypto_total_s - sym).sum() * 1e3,
        "symmetric_ms": sym.sum() * 1e3,
        "crypto_ms_per_round": df.crypto_total_s.mean() * 1e3,
        **{f"{op}_ms": df[f"{op}_s"].sum() * 1e3
           for op in ("keygen", "encaps", "decaps", "sign", "verify")},
        "handshake_kb_per_client": hs_bytes.sum() / 1024,
        "wire_mb_per_client": df.bytes_wire_total.sum() / 2**20,
        "crypto_pct_of_wall": 100 * df.crypto_total_s.sum() / df.wall_s.sum(),
        # Round time = wall clock incl. local training, excl. the test-set evaluation (which is
        # instrumentation, not part of the protocol).
        "crypto_pct_of_round": 100 * df.crypto_total_s.sum() / (df.wall_s - df.eval_s).sum(),
        "wire_kb_per_client_round": df.bytes_wire_total.mean() / 1024,
        "overhead_b_per_client_round": (df.bytes_wire_total - df.plaintext_bytes).mean(),
        "overhead_pct_of_payload": 100 * (df.bytes_wire_total - df.plaintext_bytes).sum()
                                   / df.plaintext_bytes.sum(),
        "train_s": df.train_s.sum(),
        "wall_s": df.wall_s.sum(),
        "final_acc": df.test_acc.iloc[-1] * 100,
        "rejected": int(df.rejected.sum()),
    }


def aggregate(runs: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    g = runs.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    g.insert(0, "n_seeds", runs.groupby(keys, sort=False).size())
    return g.reset_index()


FMT = {"handshakes": "{:.0f}", "crypto_ms": "{:.2f}", "handshake_ms": "{:.2f}",
       "symmetric_ms": "{:.2f}", "crypto_ms_per_round": "{:.3f}", "keygen_ms": "{:.3f}",
       "encaps_ms": "{:.3f}", "decaps_ms": "{:.3f}", "sign_ms": "{:.3f}", "verify_ms": "{:.3f}",
       "handshake_kb_per_client": "{:.1f}", "wire_mb_per_client": "{:.2f}",
       "crypto_pct_of_wall": "{:.3f}", "crypto_pct_of_round": "{:.3f}",
       "wire_kb_per_client_round": "{:.1f}", "overhead_b_per_client_round": "{:.0f}",
       "overhead_pct_of_payload": "{:.3f}", "train_s": "{:.1f}", "wall_s": "{:.1f}",
       "final_acc": "{:.2f}"}

LABELS = {"crypto_ms": "crypto total (ms)", "handshake_ms": "handshake crypto (ms)",
          "symmetric_ms": "AES-GCM path (ms)", "crypto_ms_per_round": "crypto / round (ms)",
          "handshake_kb_per_client": "handshake KB / client",
          "wire_mb_per_client": "wire MB / client", "crypto_pct_of_wall": "crypto % of wall",
          "crypto_pct_of_round": "crypto % of round time",
          "wire_kb_per_client_round": "wire KB / client / round",
          "overhead_b_per_client_round": "protocol overhead B / client / round",
          "overhead_pct_of_payload": "overhead % of payload",
          "final_acc": "final acc (%)", "handshakes": "handshakes",
          "keygen_ms": "keygen (ms)", "encaps_ms": "encaps (ms)", "decaps_ms": "decaps (ms)",
          "sign_ms": "sign (ms)", "verify_ms": "verify (ms)", "wall_s": "wall (s)"}


def to_markdown(agg: pd.DataFrame, keys: list[str], metrics: list[str]) -> str:
    """Markdown table with 'mean ± std' cells (std omitted when it is exactly zero or n=1)."""
    header = keys + [LABELS.get(m, m) for m in metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in agg.iterrows():
        cells = [str(row[k]) for k in keys]
        for m in metrics:
            f = FMT.get(m, "{:.3g}")
            mean, std = row[f"{m}_mean"], row[f"{m}_std"]
            cell = f.format(mean)
            if pd.notna(std) and std > 0:
                cell += f" ± {f.format(std)}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(name: str, agg: pd.DataFrame, keys: list[str], metrics: list[str],
               note: str = "") -> str:
    agg.to_csv(TABLES / f"{name}.csv", index=False)
    md = to_markdown(agg, keys, metrics)
    (TABLES / f"{name}.md").write_text((note + "\n\n" if note else "") + md)
    return md


# --- Experiments -----------------------------------------------------------------

MAIN_METRICS = ["handshakes", "crypto_ms", "handshake_ms", "symmetric_ms",
                "handshake_kb_per_client", "wire_mb_per_client", "crypto_pct_of_round",
                "final_acc"]
OVERHEAD_METRICS = ["crypto_ms_per_round", "crypto_pct_of_round", "crypto_pct_of_wall",
                    "wire_kb_per_client_round", "overhead_b_per_client_round",
                    "overhead_pct_of_payload"]
OP_METRICS = ["keygen_ms", "encaps_ms", "decaps_ms", "sign_ms", "verify_ms"]


def exp_main(runs: pd.DataFrame) -> str:
    agg = aggregate(runs, ["mode", "K", "N"], MAIN_METRICS + OP_METRICS)
    note = ("Experiment 1: main comparison. Mean ± std over seeds. Times are summed over all "
            "clients + server over the whole session.")
    md = save_table("main_comparison", agg, ["mode", "K", "N"], MAIN_METRICS, note)
    md += "\nPer-primitive time (whole session):\n\n" + save_table(
        "main_comparison_primitives", agg, ["mode", "K", "N"], OP_METRICS)
    agg = aggregate(runs, ["mode", "K", "N"], OVERHEAD_METRICS)
    md += "\nCompute vs bandwidth overhead (per round; round time includes local training, "
    md += "excludes test-set evaluation):\n\n" + save_table(
        "overhead", agg, ["mode", "K", "N"], OVERHEAD_METRICS)
    return md


def exp_rekey(runs: pd.DataFrame) -> str:
    amort = runs[runs["mode"] == "amortized"].sort_values(["N", "seed"])
    naive = runs[runs["mode"] == "naive_pqc"]
    both = pd.concat([amort, naive])
    metrics = ["handshakes", "crypto_ms", "crypto_ms_per_round", "handshake_ms",
               "handshake_kb_per_client", "crypto_pct_of_round"]
    agg = aggregate(both, ["mode", "N"], metrics)
    md = save_table("rekey_sweep", agg, ["mode", "N"], metrics,
                    "Experiment 2: re-key cadence sweep (amortized), naive_pqc as reference.")

    # N=1 must reproduce naive_pqc: identical handshake count and bytes; time within noise.
    a1 = amort[amort["N"] == 1].set_index("seed")
    nv = naive.set_index("seed")
    common = a1.index.intersection(nv.index)
    if len(common):
        same_hs = bool((a1.loc[common, "handshakes"] == nv.loc[common, "handshakes"]).all())
        same_bytes = bool(np.allclose(a1.loc[common, "handshake_kb_per_client"],
                                      nv.loc[common, "handshake_kb_per_client"]))
        diff = a1.loc[common, "crypto_ms"] - nv.loc[common, "crypto_ms"]
        ratio = a1.loc[common, "crypto_ms"].mean() / nv.loc[common, "crypto_ms"].mean()
        sd = diff.std() if len(diff) > 1 else float("nan")
        check = (f"N=1 vs naive_pqc over {len(common)} seed(s): handshakes identical: {same_hs}; "
                 f"handshake bytes identical: {same_bytes}; crypto time ratio {ratio:.3f} "
                 f"(paired diff {diff.mean():+.2f} ± {sd:.2f} ms).")
        (TABLES / "rekey_sweep.md").write_text((TABLES / "rekey_sweep.md").read_text()
                                               + "\n" + check + "\n")
        md += "\n" + check + "\n"
    return md


def exp_scaling(runs: pd.DataFrame) -> str:
    metrics = ["handshakes", "crypto_ms", "handshake_ms", "crypto_ms_per_round",
               "handshake_kb_per_client", "crypto_pct_of_round", "wall_s"]
    agg = aggregate(runs.sort_values(["K", "mode"]), ["K", "mode", "N"], metrics)
    return save_table("client_scaling", agg, ["K", "mode", "N"], metrics,
                      "Experiment 3: client scaling. Crypto time is summed over all K clients "
                      "+ server; byte columns are per client.")


def exp_accuracy(runs: pd.DataFrame, raw: dict[Spec, pd.DataFrame]) -> str:
    pivot = runs.pivot_table(index="seed", columns="mode", values="final_acc")[list(MODES)]
    pivot["max |Δ| across modes"] = pivot.max(axis=1) - pivot.min(axis=1)

    # Whole trajectories, not just the final round.
    traj: dict[int, list[np.ndarray]] = {}
    for spec, df in raw.items():
        traj.setdefault(spec.seed, []).append(df.test_acc.to_numpy())
    pivot["max |Δ| any round"] = [
        float(np.max(np.ptp(np.stack(traj[s]), axis=0))) * 100 for s in pivot.index]
    pivot.to_csv(TABLES / "accuracy_validation.csv")

    per_mode = runs.groupby("mode")["final_acc"].agg(["mean", "std"]).loc[list(MODES)]
    lines = ["Experiment 4: final test accuracy (%) per mode and seed.", "",
             "| seed | " + " | ".join(pivot.columns) + " |", "|" + "---|" * (len(pivot.columns) + 1)]
    for seed, row in pivot.iterrows():
        lines.append(f"| {seed} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    lines.append("| mean ± std | " + " | ".join(
        f"{per_mode.loc[m, 'mean']:.2f} ± {per_mode.loc[m, 'std']:.2f}" for m in MODES) + " | | |")
    identical = pivot["max |Δ| any round"].max() == 0
    lines += ["", ("Accuracy is bit-identical across modes for every seed and every round: the "
                   "crypto layer is lossless." if identical else
                   "WARNING: accuracy differs across modes for some seed; investigate.")]
    md = "\n".join(lines) + "\n"
    (TABLES / "accuracy_validation.md").write_text(md)
    return md


def manifest(args, effective: dict, specs: list[Spec], elapsed: float) -> None:
    import oqs
    import torch
    info = {
        "args": vars(args), "effective": effective, "runs": len(specs), "elapsed_s": round(elapsed, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "liboqs": oqs.oqs_version(), "liboqs_python": oqs.oqs_python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (TABLES / "manifest.json").write_text(json.dumps(info, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="1,2,3,4", help="comma-separated experiment numbers")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--samples-per-client", type=int, default=2000,
                   help="training subset per client; 0 = full MNIST shard (much slower)")
    p.add_argument("--K", type=int, default=5, help="clients for experiments 1, 2")
    p.add_argument("--N", type=int, default=5, help="re-key interval for experiments 1, 3")
    p.add_argument("--N-sweep", default="1,2,3,5,10,15,20,30,60")
    p.add_argument("--sweep-rounds", type=int, default=60, help="rounds for experiment 2 only")
    p.add_argument("--K-sweep", default="3,5,10,20")
    p.add_argument("--quick", action="store_true",
                   help="pipeline smoke test: 2 seeds, 4 rounds (8 for the N sweep), N=2, "
                        "small sweeps, 300 samples")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if raw CSVs exist")
    args = p.parse_args()

    exps = {int(x) for x in args.exp.split(",")}
    seeds, rounds, N, sweep_rounds = list(range(args.seeds)), args.rounds, args.N, args.sweep_rounds
    N_sweep = [int(x) for x in args.N_sweep.split(",")]
    K_sweep = [int(x) for x in args.K_sweep.split(",")]
    samples = args.samples_per_client or None
    global RAW, TABLES
    if args.quick:
        seeds, rounds, N, samples, sweep_rounds = [0, 1], 4, 2, 300, 8
        N_sweep, K_sweep = [1, 2, 4, 8], [3, 5]
        RAW, TABLES = RESULTS / "quick" / "raw", RESULTS / "quick" / "tables"

    effective = {"seeds": seeds, "rounds": rounds, "sweep_rounds": sweep_rounds, "K": args.K,
                 "N": N, "N_sweep": N_sweep,
                 "K_sweep": K_sweep, "samples_per_client": samples, "raw_dir": str(RAW)}
    by_exp = plan(exps, seeds, rounds, sweep_rounds, samples, args.K, N, N_sweep, K_sweep)
    unique = list(dict.fromkeys(s for specs in by_exp.values() for s in specs))
    todo = [s for s in unique if args.force or not s.path.exists()]
    est = sum(estimate_seconds(s) for s in todo)
    print(f"{len(unique)} distinct runs ({sum(map(len, by_exp.values()))} incl. shared), "
          f"{len(todo)} to execute, rough estimate {est / 60:.0f} min.")
    if args.dry_run:
        for s in todo:
            print(f"  {s.path.name}  ~{estimate_seconds(s):.0f}s")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    from pqfl.fl.data import load_mnist
    train, test = load_mnist("data")

    t0 = time.perf_counter()
    raw: dict[Spec, pd.DataFrame] = {}
    for i, spec in enumerate(unique, 1):
        cached = spec.path.exists() and not args.force
        t = time.perf_counter()
        raw[spec] = run_one(spec, train, test, args.force)
        r = summarize_run(spec, raw[spec])
        print(f"[{i:>3}/{len(unique)}] {spec.mode:<10} K={spec.K:<3} N={r['N']:<3} "
              f"seed={spec.seed}  {'cached' if cached else f'{time.perf_counter() - t:5.1f}s'}  "
              f"crypto={r['crypto_ms']:8.2f} ms  hs={r['handshakes']:<4} "
              f"acc={r['final_acc']:.2f}%", flush=True)

    all_runs = pd.DataFrame([summarize_run(s, df) for s, df in raw.items()])
    all_runs.to_csv(TABLES / "all_runs.csv", index=False)

    def runs_for(exp: int) -> pd.DataFrame:
        return pd.DataFrame([summarize_run(s, raw[s]) for s in by_exp[exp]])

    sections = []
    if 1 in exps:
        sections.append(exp_main(runs_for(1)))
    if 2 in exps:
        sections.append(exp_rekey(runs_for(2)))
    if 3 in exps:
        sections.append(exp_scaling(runs_for(3)))
    if 4 in exps:
        sections.append(exp_accuracy(runs_for(1), {s: raw[s] for s in by_exp[1]}))

    manifest(args, effective, unique, time.perf_counter() - t0)
    print("\n" + "\n\n".join(sections))
    print(f"Raw CSVs: {RAW}/   Tables: {TABLES}/   ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
