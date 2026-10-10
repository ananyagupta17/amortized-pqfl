"""Live demo: the same FL session under classical, naive PQC and amortized PQC encryption.

    python scripts/demo.py

1. Runs FedAvg on MNIST (K=3 clients, 10 rounds, re-key every N=5 rounds) once per mode
   and prints a per-round log: handshake vs symmetric-only round, crypto time, bytes,
   test accuracy.
2. Prints a summary table comparing the modes.
3. Runs a live security check: an attacker on the network tampers with one update and
   replays another from an old round; the server must reject both.
4. Saves two charts to results/.

Classical also re-keys every N rounds here, so classical vs amortized differ only in the
primitives (X25519/Ed25519 vs ML-KEM-768/ML-DSA-65). Training uses a 2,000-sample subset
per client so the whole demo finishes in about a minute on a laptop CPU.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pqfl.fl.data import load_mnist  # noqa: E402
from pqfl.fl.harness import RunConfig, run  # noqa: E402

K, ROUNDS, N, SEED = 3, 10, 5, 0
SAMPLES_PER_CLIENT = 2000
RESULTS = Path("results")

MODES = [  # (mode, rekey_interval, handshake label)
    ("classical", N, "classical handshake"),
    ("naive_pqc", None, "PQC handshake"),
    ("amortized", N, "PQC handshake"),
]

# First three slots of the validated categorical palette (light surface). Fixed per mode.
COLORS = {"classical": "#2a78d6", "naive_pqc": "#eb6834", "amortized": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

BOLD, DIM, RED, GREEN, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[0m"
if not sys.stdout.isatty():
    BOLD = DIM = RED = GREEN = RESET = ""


def kb(n: float) -> str:
    return f"{n / 1024:,.1f} KB"


def banner(text: str) -> None:
    print(f"\n{BOLD}{'=' * 78}\n {text}\n{'=' * 78}{RESET}")


def run_mode(mode: str, rekey: int | None, hs_label: str, train, test) -> list[dict]:
    banner(f"{mode}  (K={K}, rounds={ROUNDS}, re-key every "
           f"{1 if mode == 'naive_pqc' else rekey} round(s))")
    print(f"{DIM} round  key schedule                crypto     sent/client  "
          f"(handshake part)   test acc{RESET}")

    def on_round(row: dict) -> None:
        hs_bytes = row["bytes_key"] + row["bytes_kem_ct"] + row["bytes_sig"]
        if row["handshakes"]:
            kind = f"{BOLD}{hs_label} x{row['handshakes']}{RESET}"
            pad = 27 - len(f"{hs_label} x{row['handshakes']}")
        else:
            kind, pad = f"{DIM}symmetric only (AES-GCM){RESET}", 3
        print(f"  {row['round']:>3}  {kind}{' ' * pad} {row['crypto_total_s'] * 1e3:7.2f} ms  "
              f"{kb(row['bytes_wire_total']):>11}  ({kb(hs_bytes):>8})   "
              f"{row['test_acc'] * 100:6.2f}%")

    cfg = RunConfig(mode=mode, clients=K, rounds=ROUNDS, rekey_interval=rekey, seed=SEED,
                    samples_per_client=SAMPLES_PER_CLIENT,
                    out=str(RESULTS / f"demo_{mode}.csv"))
    return run(cfg, train_data=train, test_data=test, on_round=on_round, log=lambda _: None)


def summary(results: dict[str, list[dict]]) -> None:
    banner("Summary (whole session, all clients)")
    head = (f"{'mode':<11}{'handshakes':>11}{'crypto time':>14}{'  of which':>11}"
            f"{'total sent':>13}{'handshake bytes':>17}{'final acc':>11}")
    print(head)
    print("-" * len(head))
    for mode, rows in results.items():
        hs = sum(r["handshakes"] for r in rows)
        crypto = sum(r["crypto_total_s"] for r in rows) * 1e3
        sym = sum(r["sym_encrypt_s"] + r["sym_decrypt_s"] for r in rows) * 1e3
        total = sum(r["bytes_wire_total"] for r in rows) * K
        hs_bytes = sum(r["bytes_key"] + r["bytes_kem_ct"] + r["bytes_sig"] for r in rows) * K
        print(f"{mode:<11}{hs:>11}{crypto:>11.2f} ms{f'{crypto - sym:.2f} hs':>11}"
              f"{total / 2**20:>10.2f} MB{kb(hs_bytes):>17}{rows[-1]['test_acc'] * 100:>10.2f}%")
    print(f"{DIM}'of which hs' = crypto time outside per-round AES-GCM (handshakes + key "
          f"derivation), in ms.{RESET}")

    naive, amort = results["naive_pqc"], results["amortized"]
    t_naive = sum(r["crypto_total_s"] for r in naive)
    t_amort = sum(r["crypto_total_s"] for r in amort)
    b_naive = sum(r["bytes_key"] + r["bytes_kem_ct"] + r["bytes_sig"] for r in naive)
    b_amort = sum(r["bytes_key"] + r["bytes_kem_ct"] + r["bytes_sig"] for r in amort)
    print(f"\nAmortized vs naive PQC: {100 * (1 - t_amort / t_naive):.0f}% less crypto time, "
          f"{100 * (1 - b_amort / b_naive):.0f}% fewer handshake bytes.")
    accs = {round(rows[-1]["test_acc"], 6) for rows in results.values()}
    if len(accs) == 1:
        print("Final accuracy is identical across modes: encryption is lossless and the "
              "training run is seeded identically.")


def security_check(train, test) -> bool:
    banner("Live security check: attacker on the network (amortized mode)")
    captured = {}

    def attacker(r: int, msg):
        if r == 0 and msg.client_id == "client-1":
            print(f"  {RED}attack 1{RESET}: flip one bit in client-1's round-0 ciphertext")
            ct = bytearray(msg.ciphertext)
            ct[100] ^= 0x01
            return replace(msg, ciphertext=bytes(ct))
        if r == 0 and msg.client_id in ("client-0", "client-2"):
            captured[msg.client_id] = msg  # record, deliver untouched
        if r == 1 and msg.client_id == "client-2":
            print(f"  {RED}attack 3{RESET}: replay client-2's round-0 update in round 1, verbatim")
            return captured["client-2"]
        if r == 1 and msg.client_id == "client-0":
            print(f"  {RED}attack 2{RESET}: replay client-0's round-0 update in round 1, "
                  f"header rewritten to say round 1")
            return replace(captured["client-0"], round=1)
        return msg

    def log(line: str) -> None:
        if "rejected" in line:
            print(f"  {GREEN}server{RESET}: {line.split('] ', 1)[1]}")

    def on_round(row: dict) -> None:
        print(f"  round {row['round']}: {row['accepted']} accepted, {row['rejected']} rejected "
              f"-> FedAvg over {row['accepted']} update(s), acc {row['test_acc'] * 100:.2f}%\n")

    cfg = RunConfig(mode="amortized", clients=K, rounds=2, rekey_interval=N, seed=SEED,
                    samples_per_client=500)
    rows = run(cfg, train_data=train, test_data=test, channel=attacker,
               on_round=on_round, log=log)
    ok = rows[0]["rejected"] == 1 and rows[1]["rejected"] == 2
    verdict = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {verdict}: every tampered or replayed update was rejected before averaging; "
          f"genuine updates were accepted.")
    return ok


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(results: dict[str, list[dict]]) -> list[Path]:
    paths = []
    # 1. Crypto time per round, one line per mode.
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    for mode, rows in results.items():
        x = [r["round"] for r in rows]
        y = [r["crypto_total_s"] * 1e3 for r in rows]
        ax.plot(x, y, color=COLORS[mode], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=mode)
        ax.annotate(mode, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    for r in range(0, ROUNDS, N):
        ax.axvline(r, color=GRID, linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(range(ROUNDS))
    ax.set_xlim(-0.4, ROUNDS - 0.4 + 1.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("FL round (dashed: re-key rounds for N=5)", color=INK_2, fontsize=9)
    ax.set_ylabel("crypto time per round (ms)", color=INK_2, fontsize=9)
    ax.set_title(f"Crypto time per round, K={K} clients", color=INK, fontsize=11,
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    paths.append(RESULTS / "demo_crypto_time_per_round.png")
    fig.savefig(paths[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Final accuracy by mode.
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURFACE)
    _style(ax)
    modes = list(results)
    accs = [results[m][-1]["test_acc"] * 100 for m in modes]
    bars = ax.bar(modes, accs, width=0.5, color=[COLORS[m] for m in modes],
                  edgecolor=SURFACE, linewidth=2)
    for b, a in zip(bars, accs):
        ax.annotate(f"{a:.2f}%", (b.get_x() + b.get_width() / 2, a), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("test accuracy after final round (%)", color=INK_2, fontsize=9)
    ax.set_title("Final MNIST test accuracy by mode", color=INK, fontsize=11,
                 loc="left", fontweight="bold")
    fig.tight_layout()
    paths.append(RESULTS / "demo_final_accuracy.png")
    fig.savefig(paths[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return paths


def main() -> int:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    print("Loading MNIST ...")
    train, test = load_mnist("data")

    results = {mode: run_mode(mode, rekey, label, train, test) for mode, rekey, label in MODES}
    summary(results)
    ok = security_check(train, test)
    banner("Charts")
    for p in charts(results):
        print(f"  saved {p}")
    print(f"\nDemo finished in {time.perf_counter() - t0:.1f} s.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
