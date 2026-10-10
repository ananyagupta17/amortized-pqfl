"""Build report assets (figures, LaTeX tables, diagrams, skeleton) from experiment results.

    python scripts/make_report_assets.py                         # from results/ (full run)
    python scripts/make_report_assets.py --results results/quick # from the --quick profile

Reads <results>/tables/manifest.json and the per-round CSVs in its raw dir, and writes to
report_assets/:
  figures/*.pdf     vector figures sized for an IEEE column (3.5 in)
  tables/*.tex      booktabs tables (\\input from main.tex)
  diagrams/*.pdf    TikZ diagrams compiled with tectonic
  main.pdf          IEEEtran skeleton compiled as an integration check

Assets built from the --quick profile are stamped as placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"
MODES = ("classical", "naive_pqc", "amortized")
LABEL = {"classical": "Classical", "naive_pqc": "Naive PQC", "amortized": "Amortized PQC"}
# Validated categorical slots 1-3; each mode also gets its own marker + dash for print/grayscale.
COLOR = {"classical": "#2a78d6", "naive_pqc": "#eb6834", "amortized": "#1baf7a"}
MARKER = {"classical": "o", "naive_pqc": "s", "amortized": "^"}
DASH = {"classical": "-", "naive_pqc": "--", "amortized": "-."}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.5  # IEEE single-column width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

RAW_RE = re.compile(r"(?P<mode>\w+?)_K(?P<K>\d+)_N(?P<N>\d+)_seed(?P<seed>\d+)_R(?P<R>\d+)_S(?P<S>\w+)\.csv$")


# --- Data ------------------------------------------------------------------------

class Data:
    def __init__(self, results: Path) -> None:
        man = json.loads((results / "tables" / "manifest.json").read_text())
        self.eff = man["effective"]
        self.quick = bool(man["args"].get("quick"))
        self.manifest = man
        raw_dir = ROOT / self.eff["raw_dir"]
        s = self.eff["samples_per_client"]
        s = "full" if s is None else str(s)
        rows = []
        for p in sorted(raw_dir.glob("*.csv")):
            m = RAW_RE.match(p.name)
            if not m or int(m["R"]) != self.eff["rounds"] or m["S"] != s:
                continue
            df = pd.read_csv(p)
            if len(df) != self.eff["rounds"] or int(m["seed"]) not in self.eff["seeds"]:
                continue
            rows.append({"mode": m["mode"], "K": int(m["K"]), "N": int(m["N"]),
                         "seed": int(m["seed"]), "df": df})
        self.runs = pd.DataFrame(rows)
        if self.runs.empty:
            sys.exit(f"no raw runs matching the manifest in {raw_dir}")

    def select(self, mode=None, K=None, N=None) -> list[pd.DataFrame]:
        r = self.runs
        if mode is not None:
            r = r[r["mode"] == mode]
        if K is not None:
            r = r[r["K"] == K]
        if N is not None:
            r = r[r["N"] == N]
        return list(r.sort_values("seed")["df"])

    def main(self, mode: str, K: int | None = None) -> list[pd.DataFrame]:
        """Experiment-1 configuration (or the K-sweep point) for a mode."""
        return self.select(mode, K or self.eff["K"], 1 if mode == "naive_pqc" else self.eff["N"])


def summarize(df: pd.DataFrame) -> dict:
    sym = df.sym_encrypt_s + df.sym_decrypt_s
    hs_b = df.bytes_key + df.bytes_kem_ct + df.bytes_sig
    return {
        "handshakes": df.handshakes.sum(),
        "crypto_ms": df.crypto_total_s.sum() * 1e3,
        "handshake_ms": (df.crypto_total_s - sym).sum() * 1e3,
        "symmetric_ms": sym.sum() * 1e3,
        "per_client_round_ms": df.crypto_total_s.mean() * 1e3 / df.K.iloc[0],
        "hs_kb": hs_b.sum() / 1024,
        "wire_mb": df.bytes_wire_total.sum() / 2**20,
        "pct_wall": 100 * df.crypto_total_s.sum() / df.wall_s.sum(),
        "final_acc": df.test_acc.iloc[-1] * 100,
        **{f"{op}_ms": df[f"{op}_s"].sum() * 1e3
           for op in ("keygen", "encaps", "decaps", "sign", "verify")},
    }


def stats(dfs: list[pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    s = pd.DataFrame([summarize(d) for d in dfs])
    return s.mean(), s.std(ddof=1) if len(s) > 1 else s.mean() * 0


# --- Figures ---------------------------------------------------------------------

def _stamp(fig, data: Data) -> None:
    if data.quick:
        fig.text(0.5, 0.5, "PLACEHOLDER: quick-profile data", ha="center", va="center",
                 fontsize=11, color="#e34948", alpha=0.22, rotation=18, zorder=10)


def _save(fig, data: Data, name: str) -> Path:
    _stamp(fig, data)
    path = OUT / "figures" / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _legend_top(ax, ncol=3) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, handlelength=2.2,
              columnspacing=1.0, borderaxespad=0)


def _round_ticks(ax, R: int) -> None:
    ax.xaxis.set_major_locator(FixedLocator(
        list(range(1, R + 1)) if R <= 10 else [1] + list(range(5, R + 1, 5))))


def _rekey_lines(ax, R: int, N: int) -> None:
    for r in range(0, R, N):
        ax.axvline(r + 1, color=GRID, linewidth=0.8, linestyle=":", zorder=0)


def fig_per_round(d: Data) -> Path:
    R = d.eff["rounds"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(1, R + 1)
    for m in MODES:
        y = np.stack([df.crypto_total_s.to_numpy() * 1e3 for df in d.main(m)])
        mu, sd = y.mean(0), y.std(0, ddof=1) if len(y) > 1 else 0 * y[0]
        ax.plot(x, mu, DASH[m], color=COLOR[m], marker=MARKER[m], label=LABEL[m],
                markeredgecolor="white", markeredgewidth=0.5)
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.18, linewidth=0)
    _rekey_lines(ax, R, d.eff["N"])
    ax.set_xlim(0.5, R + 0.5)
    ax.set_ylim(bottom=0)
    _round_ticks(ax, R)
    ax.set_xlabel(f"FL round (dotted: re-key rounds, $N={d.eff['N']}$)")
    ax.set_ylabel("Crypto time per round (ms)")
    _legend_top(ax)
    return _save(fig, d, "crypto_per_round")


def fig_cumulative(d: Data) -> Path:
    R = d.eff["rounds"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(1, R + 1)
    for m in MODES:
        y = np.stack([np.cumsum(df.crypto_total_s.to_numpy()) * 1e3 for df in d.main(m)])
        mu, sd = y.mean(0), y.std(0, ddof=1) if len(y) > 1 else 0 * y[0]
        ax.plot(x, mu, DASH[m], color=COLOR[m], label=LABEL[m])
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.18, linewidth=0)
        ax.annotate(f"{mu[-1]:.1f}", (x[-1], mu[-1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_xlim(0.5, R + 2.5)
    ax.set_ylim(bottom=0)
    _round_ticks(ax, R)
    ax.set_xlabel("FL round")
    ax.set_ylabel("Cumulative crypto time (ms)")
    _legend_top(ax)
    return _save(fig, d, "crypto_cumulative")


def fig_bytes(d: Data) -> Path:
    comps = [("KEM public key", "bytes_key"), ("KEM ciphertext", "bytes_kem_ct"),
             ("Signatures", "bytes_sig"), ("AEAD nonce + tag", None), ("Message framing", None)]
    shades = ["#3d3d3a", "#73726c", "#a3a29b", "#cfcec8", "#ecebe7"]
    hatches = ["", "////", "", "....", ""]
    vals, payload = {}, {}
    for m in MODES:
        df = d.main(m)[0]  # byte counts are deterministic across seeds
        aead = (df.bytes_payload - df.plaintext_bytes).sum()
        framing = (df.bytes_wire_total - df.bytes_key - df.bytes_kem_ct - df.bytes_sig
                   - df.bytes_payload).sum()
        vals[m] = [df[c].sum() if c else v for (_, c), v in
                   zip(comps, [None, None, None, aead, framing])]
        payload[m] = df.plaintext_bytes.sum()
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    ys = np.arange(len(MODES))[::-1]
    for y, m in zip(ys, MODES):
        left = 0.0
        for i, v in enumerate(vals[m]):
            kb = v / 1024
            ax.barh(y, kb, left=left, height=0.55, color=shades[i], hatch=hatches[i],
                    edgecolor="white", linewidth=0.6, label=comps[i][0] if y == ys[0] else None)
            left += kb
        ax.annotate(f"{left:.1f} KB", (left, y), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_yticks(ys, [LABEL[m] for m in MODES])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(v) for v in vals.values()) / 1024 * 1.18)
    pl = payload[MODES[0]] / 2**20
    ax.set_xlabel(f"Protocol overhead per client over {d.eff['rounds']} rounds (KB)\n"
                  f"excludes the {pl:.1f} MB encrypted model payload, identical in all modes")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=6.5,
              handlelength=1.4, columnspacing=0.8, borderaxespad=0)
    return _save(fig, d, "bytes_components")


NSTAR_TOL = 0.10  # handshake overhead per round must be <= 10% of the symmetric-only floor


def compute_nstar(d: Data) -> dict:
    """N* = smallest swept N whose handshake overhead per round is within NSTAR_TOL of the
    asymptotic floor, i.e. the symmetric-only (large-N) per-round crypto cost.

    floor             = mean over all amortized sweep runs of (AES-GCM path time) / R
    overhead(N)       = (handshake crypto time over the session) / R
    criterion         = overhead(N) <= NSTAR_TOL * floor
    """
    Ns, R, K = d.eff["N_sweep"], d.eff["rounds"], d.eff["K"]
    per_n = {}
    sym_per_round = []
    for n in Ns:
        runs = pd.DataFrame([summarize(df) for df in d.select("amortized", K, n)])
        sym_per_round += list(runs["symmetric_ms"] / R)
        per_n[n] = runs
    floor = float(np.mean(sym_per_round))
    rows = []
    for n in Ns:
        runs = per_n[n]
        hs = runs["handshake_ms"] / R
        tot = runs["crypto_ms"] / R
        rows.append({
            "N": n, "handshakes": int(runs["handshakes"].mean()),
            "crypto_per_round_ms": float(tot.mean()), "crypto_per_round_sd": float(tot.std(ddof=1)) if len(tot) > 1 else 0.0,
            "hs_overhead_per_round_ms": float(hs.mean()),
            "hs_overhead_per_round_sd": float(hs.std(ddof=1)) if len(hs) > 1 else 0.0,
            "hs_overhead_pct_of_floor": float(100 * hs.mean() / floor),
            "within_tol": bool(hs.mean() <= NSTAR_TOL * floor),
        })
    nstar = next((r["N"] for r in rows if r["within_tol"]), None)
    return {"criterion": f"handshake overhead per round <= {NSTAR_TOL:.0%} of symmetric-only floor",
            "K": K, "rounds": R, "floor_per_round_ms": floor, "threshold_ms": NSTAR_TOL * floor,
            "N_star": nstar, "N_swept": Ns, "rows": rows}


def _nstar_label(ns: dict) -> str:
    if ns["N_star"] is None:
        return f"$N^* > {max(ns['N_swept'])}$ (not reached in sweep)"
    return f"$N^*={ns['N_star']}$"


def fig_rekey(d: Data) -> Path:
    ns = compute_nstar(d)
    Ns, R, K = d.eff["N_sweep"], d.eff["rounds"], d.eff["K"]
    rows = {r["N"]: r for r in ns["rows"]}
    floor, thr = ns["floor_per_round_ms"], ns["threshold_ms"]
    mu = {n: stats(d.select("amortized", K, n)) for n in Ns}
    nv_mu, _ = stats(d.select("naive_pqc", K, 1))
    # Cost model: handshake cost per handshake round, measured at the smallest N.
    per_hs_round = rows[Ns[0]]["hs_overhead_per_round_ms"] * R / np.ceil(R / Ns[0])
    xs = np.array(Ns)
    model_x = np.unique(np.geomspace(min(Ns), max(Ns), 200).astype(int))
    m = "amortized"

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.4), sharex=True)
    a.axhspan(floor, floor + thr, color=COLOR[m], alpha=0.10, linewidth=0,
              label=f"floor + {NSTAR_TOL:.0%} band")
    a.errorbar(xs, [rows[n]["crypto_per_round_ms"] for n in Ns],
               yerr=[rows[n]["crypto_per_round_sd"] for n in Ns], color=COLOR[m],
               marker=MARKER[m], capsize=2, linewidth=1.4, label=LABEL[m] + " (measured)")
    a.plot(model_x, floor + per_hs_round * np.ceil(R / model_x) / R, color=INK_2, linewidth=0.7,
           linestyle=(0, (4, 2)), label=r"model: floor $+\,\lceil R/N\rceil h/R$")
    a.axhline(nv_mu["crypto_ms"] / R, color=COLOR["naive_pqc"], linestyle="--", linewidth=1,
              label=LABEL["naive_pqc"])
    a.axhline(floor, color=INK_2, linestyle=":", linewidth=1, label="Symmetric-only floor")
    if ns["N_star"] is not None:
        a.axvline(ns["N_star"], color=GRID, linewidth=4, zorder=0)
        b.axvline(ns["N_star"], color=GRID, linewidth=4, zorder=0)
    a.text(0.02, floor / 2, _nstar_label(ns) + f": handshake overhead/round "
           f"$\\leq${NSTAR_TOL:.0%} of floor",
           transform=blended_transform_factory(a.transAxes, a.transData),
           ha="left", va="center", fontsize=6.5, color=INK)
    a.set_ylim(bottom=0)
    a.set_ylabel("Crypto time per round (ms)")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=6.5,
             handlelength=2.0, columnspacing=1.0, borderaxespad=0)

    b.plot(xs, [mu[n][0]["hs_kb"] for n in Ns], color=COLOR[m], marker=MARKER[m], label=LABEL[m])
    b.axhline(nv_mu["hs_kb"], color=COLOR["naive_pqc"], linestyle="--", linewidth=1,
              label=LABEL["naive_pqc"])
    b.set_ylim(bottom=0)
    b.set_ylabel(f"Handshake bytes / client,\n{R} rounds (KB)")
    b.set_xscale("log", base=2)
    b.xaxis.set_major_locator(FixedLocator(Ns))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xticklabels([str(n) for n in Ns])
    b.set_xlabel(f"Re-key interval $N$ (rounds), $K={K}$, $R={R}$")
    b.legend(loc="upper right", fontsize=6.5)
    fig.align_ylabels()
    return _save(fig, d, "rekey_sweep")


def fig_scaling(d: Data) -> Path:
    Ks = d.eff["K_sweep"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.1), sharex=True)
    for m in MODES:
        st = [stats(d.main(m, K)) for K in Ks]
        for ax, key in ((a, "crypto_ms"), (b, "per_client_round_ms")):
            ax.errorbar(Ks, [s[0][key] for s in st], yerr=[s[1][key] for s in st],
                        color=COLOR[m], marker=MARKER[m], linestyle=DASH[m], capsize=2,
                        label=LABEL[m])
    a.set_ylabel(f"Crypto time, {d.eff['rounds']} rounds,\nall clients + server (ms)")
    b.set_ylabel("Crypto time per client\nper round (ms)")
    for ax in (a, b):
        ax.set_ylim(bottom=0)
    b.xaxis.set_major_locator(FixedLocator(Ks))
    b.set_xlabel(f"Number of clients $K$ ($N={d.eff['N']}$)")
    _legend_top(a)
    fig.align_ylabels()
    return _save(fig, d, "client_scaling")


def fig_accuracy(d: Data) -> Path:
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    xs = np.arange(len(MODES))
    for x, m in zip(xs, MODES):
        accs = np.array([df.test_acc.iloc[-1] * 100 for df in d.main(m)])
        sd = accs.std(ddof=1) if len(accs) > 1 else 0
        ax.bar(x, accs.mean(), width=0.55, color=COLOR[m], edgecolor="white", linewidth=0.8)
        ax.errorbar(x, accs.mean(), yerr=sd, color=INK, capsize=3, linewidth=0.8)
        jitter = np.linspace(-0.12, 0.12, len(accs))
        ax.scatter(x + jitter, accs, s=7, color="white", edgecolor=INK, linewidth=0.5, zorder=3)
        ax.annotate(f"{accs.mean():.2f} ± {sd:.2f}", (x, accs.mean() + sd), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xticks(xs, [LABEL[m] for m in MODES])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Final test accuracy (%)")
    n = len(d.eff["seeds"])
    ax.set_title(f"MNIST, $K={d.eff['K']}$, {d.eff['rounds']} rounds; "
                 f"mean ± std over {n} seeds (dots: seeds)", fontsize=7, color=INK_2, loc="left")
    return _save(fig, d, "accuracy")


# --- LaTeX tables ----------------------------------------------------------------

def _pm(mu, sd, fmt="{:.2f}") -> str:
    s = fmt.format(mu)
    return s if not np.isfinite(sd) or sd == 0 else f"{s} $\\pm$ {fmt.format(sd)}"


def _write_table(name: str, d: Data, body: str, from_results: bool = True) -> Path:
    path = OUT / "tables" / f"{name}.tex"
    head = "% Generated by scripts/make_report_assets.py -- do not edit by hand.\n"
    if d.quick and from_results:
        head += "% PLACEHOLDER: built from --quick profile data, not the full experiment.\n"
        body = body.replace("\\caption{", "\\caption{[PLACEHOLDER: quick-profile data] ", 1)
    path.write_text(head + body)
    return path


def _table(env: str, caption: str, label: str, colspec: str, header: str, rows: list[str],
           note: str = "") -> str:
    out = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\setlength{\\tabcolsep}{4pt}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule",
           *[r + " \\\\" for r in rows], "\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


def table_main(d: Data) -> Path:
    rows = []
    for m in MODES:
        mu, sd = stats(d.main(m))
        n = 1 if m == "naive_pqc" else d.eff["N"]
        rows.append(" & ".join([LABEL[m], str(n), f"{mu['handshakes']:.0f}",
                                _pm(mu["crypto_ms"], sd["crypto_ms"]),
                                _pm(mu["handshake_ms"], sd["handshake_ms"]),
                                _pm(mu["symmetric_ms"], sd["symmetric_ms"]),
                                f"{mu['hs_kb']:.1f}", _pm(mu["pct_wall"], sd["pct_wall"], "{:.3f}"),
                                _pm(mu["final_acc"], sd["final_acc"])]))
    hdr = ("Mode & $N$ & HS & Crypto (ms) & Handshake (ms) & AES-GCM (ms) & "
           "HS KB/client & \\% wall & Acc. (\\%)")
    note = (f"$K={d.eff['K']}$, {d.eff['rounds']} rounds, {len(d.eff['seeds'])} seeds; "
            "times summed over all clients and the server; HS = handshakes.")
    return _write_table("main_comparison", d, _table(
        "table*", "Main comparison of the three modes", "tab:main", "lrrrrrrrr", hdr, rows, note))


def table_primitives(d: Data) -> Path:
    rows = []
    for m in MODES:
        mu, sd = stats(d.main(m))
        rows.append(" & ".join([LABEL[m]] + [_pm(mu[f"{op}_ms"], sd[f"{op}_ms"], "{:.3f}")
                                             for op in ("keygen", "encaps", "decaps", "sign",
                                                        "verify")]))
    hdr = "Mode & KeyGen & Encaps & Decaps & Sign & Verify"
    return _write_table("primitives", d, _table(
        "table", "Time per primitive over a session (ms)", "tab:primitives", "lrrrrr", hdr,
        rows, "KEM keygen only; long-term identity keys are generated once at setup."))


def table_rekey(d: Data) -> Path:
    ns = compute_nstar(d)
    (OUT / "tables" / "nstar.json").write_text(json.dumps(
        {**ns, "placeholder": d.quick}, indent=2))
    K, R = d.eff["K"], d.eff["rounds"]
    rows = []
    for r in ns["rows"]:
        mu, sd = stats(d.select("amortized", K, r["N"]))
        mark = "$\\checkmark$" if r["within_tol"] else ""
        n_cell = f"\\textbf{{{r['N']}}}" if r["N"] == ns["N_star"] else str(r["N"])
        rows.append(" & ".join([
            n_cell, f"{r['handshakes']}", _pm(mu["crypto_ms"], sd["crypto_ms"]),
            _pm(r["hs_overhead_per_round_ms"], r["hs_overhead_per_round_sd"], "{:.3f}"),
            f"{r['hs_overhead_pct_of_floor']:.1f} {mark}", f"{mu['hs_kb']:.1f}"]))
    mu, sd = stats(d.select("naive_pqc", K, 1))
    rows.append("\\midrule\nNaive PQC & " + " & ".join([
        f"{mu['handshakes']:.0f}", _pm(mu["crypto_ms"], sd["crypto_ms"]),
        f"{mu['handshake_ms'] / R:.3f}", f"{100 * mu['handshake_ms'] / R / ns['floor_per_round_ms']:.1f}",
        f"{mu['hs_kb']:.1f}"]))
    hdr = ("$N$ & HS & Crypto (ms) & HS overhead/round (ms) & \\% of floor & HS KB/client")
    star = ("not reached in the sweep" if ns["N_star"] is None else f"$N^*={ns['N_star']}$ (bold)")
    note = (f"$K={K}$, {R} rounds, {len(d.eff['seeds'])} seeds. Symmetric-only floor "
            f"{ns['floor_per_round_ms']:.3f}\\,ms/round; $N^*$ = smallest $N$ with handshake "
            f"overhead per round $\\leq${NSTAR_TOL * 100:.0f}\\% of the floor: {star}.")
    return _write_table("rekey_sweep", d, _table(
        "table", "Amortized PQC versus re-key interval $N$", "tab:rekey", "lrrrrr", hdr, rows,
        note))


def table_scaling(d: Data) -> Path:
    rows = []
    for K in d.eff["K_sweep"]:
        cells = [str(K)]
        for m in MODES:
            mu, sd = stats(d.main(m, K))
            cells += [_pm(mu["crypto_ms"], sd["crypto_ms"]), _pm(mu["pct_wall"], sd["pct_wall"], "{:.3f}")]
        rows.append(" & ".join(cells))
    hdr = ("$K$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{LABEL[m]}}}" for m in MODES)
           + " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & "
           + " & ".join(["Crypto (ms) & \\% wall"] * 3))
    return _write_table("client_scaling", d, _table(
        "table*", "Crypto cost versus number of clients $K$", "tab:scaling", "lrrrrrr", hdr, rows,
        f"$N={d.eff['N']}$ (Naive PQC: $N=1$), {d.eff['rounds']} rounds, "
        f"{len(d.eff['seeds'])} seeds; times summed over all clients and the server."))


def table_accuracy(d: Data) -> Path:
    per = {m: {s: df for s, df in zip(sorted(d.eff["seeds"]), d.main(m))} for m in MODES}
    rows, worst = [], 0.0
    for s in sorted(d.eff["seeds"]):
        accs = [per[m][s].test_acc.iloc[-1] * 100 for m in MODES]
        traj = np.stack([per[m][s].test_acc.to_numpy() for m in MODES])
        dmax = float(np.ptp(traj, axis=0).max()) * 100
        worst = max(worst, dmax)
        rows.append(" & ".join([str(s)] + [f"{a:.2f}" for a in accs] + [f"{dmax:.2f}"]))
    means = [stats(d.main(m)) for m in MODES]
    rows.append("\\midrule\nMean & " + " & ".join(
        _pm(mu["final_acc"], sd["final_acc"]) for mu, sd in means) + " & ")
    hdr = "Seed & " + " & ".join(LABEL[m] for m in MODES) + " & max $|\\Delta|$ any round"
    return _write_table("accuracy", d, _table(
        "table", "Final test accuracy (\\%) per mode and seed", "tab:accuracy", "lrrrr", hdr,
        rows))


# --- Tests table -----------------------------------------------------------------

TEST_PURPOSE = {
    # crypto layer
    "test_roundtrip_and_rekey_schedule": "Round trip over 7 rounds; handshake count follows the mode's re-key schedule",
    "test_nonces_unique_and_ciphertexts_differ": "50 encryptions of one plaintext use 50 distinct 96-bit nonces",
    "test_partial_participation_late_joiner": "Client joining late keys the epoch the schedule prescribes",
    "test_encrypt_without_session_raises": "Encryption is refused before a handshake",
    "test_tampered_ciphertext_rejected": "Bit flips in ciphertext, tag or nonce, truncation, client-ID swap fail the AEAD",
    "test_replayed_update_from_old_round_rejected": "Old-round update rejected verbatim, with rewritten header, and as a duplicate",
    "test_reordered_update_rejected": "Update for a future round is rejected",
    "test_rollback_to_earlier_round_rejected": "Fresh update for a round before the last accepted one is rejected",
    "test_old_epoch_key_rejected_after_rekey": "Update under the pre-re-key session key is rejected",
    "test_update_with_wrong_epoch_header_rejected": "Update whose header epoch disagrees with the schedule is rejected",
    "test_server_rejects_update_when_client_skipped_rekey": "Client that skipped a scheduled re-key is rejected",
    "test_handshake_for_wrong_epoch_rejected": "Handshake signed for a different epoch is rejected",
    "test_replayed_handshake_rejected": "Replayed old handshake cannot overwrite a live session",
    "test_impostor_signing_with_own_key_rejected": "Handshake signed with a key not registered for the claimed ID is rejected",
    "test_bitflipped_signature_rejected": "Corrupted or empty client signatures are rejected",
    "test_substituted_kem_key_rejected": "MITM substitution of the client's KEM public key is rejected",
    "test_signature_does_not_transfer_across_epochs": "A signature from one epoch cannot authorize another",
    "test_unknown_client_rejected": "Handshake from an unregistered client ID is rejected",
    "test_forged_server_response_rejected": "Client rejects a MITM re-encapsulated or re-signed server response",
    "test_server_signs_response_in_every_mode": "Server response is signed (Ed25519 / ML-DSA-65) in every mode",
    "test_invalid_config_rejected": "Invalid mode / re-key interval combinations are rejected",
    "test_config_accepts_mode_strings": "Modes can be given as strings; naive PQC fixes $N=1$",
    "test_classical_rekey_interval_accepted": "Classical mode accepts key-once or a re-key interval",
    "test_wire_sizes": "Key, ciphertext and signature sizes match FIPS 203/204; 16-byte GCM tag",
    # FL harness
    "test_flat_params_roundtrip": "Model parameters survive flatten / restore; size mismatch is rejected",
    "test_update_serialization_roundtrip_and_validation": "Update (de)serialization round trip; wrong size, truncation, NaN rejected",
    "test_fedavg_is_sample_weighted": "FedAvg computes the sample-weighted mean",
    "test_splits_partition_the_data": "IID and Dirichlet splits are disjoint and cover the dataset",
    "test_metered_suite_times_ops_and_interoperates": "Instrumented suite times each primitive and derives identical keys",
    "test_harness_rekey_schedule_and_csv": "Harness handshake schedule, CSV schema and per-component byte counts",
    "test_harness_is_deterministic_and_mode_independent": "Same seed gives identical accuracy in every mode",
    "test_harness_drops_tampered_and_replayed_updates": "Tampered and replayed updates are excluded from aggregation",
    "test_harness_keeps_global_model_when_all_updates_rejected": "If every update is rejected the global model is unchanged",
}


def _tex_escape(s: str) -> str:
    return s.replace("_", "\\_\\allowbreak{}")


def table_tests(d: Data) -> tuple[Path, int, int]:
    junit = OUT / "build" / "junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={junit}"], cwd=ROOT, capture_output=True, text=True)
    results: dict[tuple[str, str], list[str]] = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        module = tc.get("classname", "").split(".")[-1]
        func = tc.get("name").split("[")[0]
        status = ("fail" if tc.find("failure") is not None or tc.find("error") is not None
                  else "skip" if tc.find("skipped") is not None else "pass")
        results.setdefault((module, func), []).append(status)

    missing = sorted({f for _, f in results} - TEST_PURPOSE.keys())
    if missing:
        sys.exit(f"tests without a purpose entry in TEST_PURPOSE: {missing}")

    rows, total, passed = [], 0, 0
    current = None
    for (module, func), st in results.items():
        if module != current:
            title = {"test_crypto": "Crypto layer (\\texttt{tests/test\\_crypto.py})",
                     "test_fl": "FL harness (\\texttt{tests/test\\_fl.py})"}.get(module, module)
            rows.append(("\\midrule\n" if current else "") +
                        f"\\multicolumn{{3}}{{l}}{{\\textit{{{title}}}}}")
            current = module
        n_pass = st.count("pass")
        total, passed = total + len(st), passed + n_pass
        res = (f"pass ({n_pass}/{len(st)})" if n_pass == len(st)
               else f"\\textbf{{FAIL}} ({n_pass}/{len(st)})")
        rows.append(f"\\texttt{{\\scriptsize {_tex_escape(func.removeprefix('test_'))}}} & "
                    f"{TEST_PURPOSE[func]} & {res}")
    body = _table("table*", "Automated tests (pytest); parametrized cases are counted per test",
                  "tab:tests", ">{\\raggedright\\arraybackslash}p{0.31\\linewidth}p{0.54\\linewidth}r",
                  "Test & Purpose & Result", rows,
                  f"{passed}/{total} test cases passed.")
    return _write_table("tests", d, body, from_results=False), passed, total


# --- Diagrams & skeleton ---------------------------------------------------------

def tectonic(tex: Path) -> bool:
    exe = shutil.which("tectonic")
    if not exe:
        print(f"  ! tectonic not found; skipping {tex.name}")
        return False
    r = subprocess.run([exe, "-X", "compile", "--keep-logs", "--outdir", str(tex.parent),
                        str(tex)], cwd=tex.parent, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! {tex.name} failed to compile:\n{r.stderr[-2000:]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--skip-latex", action="store_true", help="don't compile diagrams / skeleton")
    args = ap.parse_args()

    d = Data(ROOT / args.results)
    for sub in ("figures", "tables", "diagrams"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(d.runs)} runs from {d.eff['raw_dir']}"
          + ("  [QUICK PROFILE: assets are placeholders]" if d.quick else ""))

    for f in (fig_per_round, fig_cumulative, fig_bytes, fig_rekey, fig_scaling, fig_accuracy):
        print(f"  figure  {f(d).relative_to(ROOT)}")
    for t in (table_main, table_primitives, table_rekey, table_scaling, table_accuracy):
        print(f"  table   {t(d).relative_to(ROOT)}")
    ns = compute_nstar(d)
    print(f"\n  N* ({ns['criterion']}): floor = {ns['floor_per_round_ms']:.4f} ms/round, "
          f"threshold = {ns['threshold_ms']:.4f} ms/round")
    print(f"  {'N':>4} {'HS':>4} {'crypto/round':>13} {'HS overhead/round':>18} {'% of floor':>11}  ok")
    for r in ns["rows"]:
        print(f"  {r['N']:>4} {r['handshakes']:>4} {r['crypto_per_round_ms']:>10.4f} ms "
              f"{r['hs_overhead_per_round_ms']:>15.4f} ms {r['hs_overhead_pct_of_floor']:>10.1f}%  "
              f"{'yes' if r['within_tol'] else 'no'}")
    print(f"  -> N* = {ns['N_star'] if ns['N_star'] is not None else 'not reached (> ' + str(max(ns['N_swept'])) + ')'}\n")
    path, passed, total = table_tests(d)
    print(f"  table   {path.relative_to(ROOT)}  ({passed}/{total} test cases passed)")

    if not args.skip_latex:
        for tex in sorted((OUT / "diagrams").glob("*.tex")) + [OUT / "main.tex"]:
            if tex.exists() and tectonic(tex):
                print(f"  latex   {tex.with_suffix('.pdf').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
