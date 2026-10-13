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
            Rs = {self.eff["rounds"], self.eff.get("sweep_rounds", self.eff["rounds"])}
            if not m or int(m["R"]) not in Rs or m["S"] != s:
                continue
            df = pd.read_csv(p)
            if len(df) != int(m["R"]) or int(m["seed"]) not in self.eff["seeds"]:
                continue
            rows.append({"mode": m["mode"], "K": int(m["K"]), "N": int(m["N"]),
                         "seed": int(m["seed"]), "R": int(m["R"]), "df": df})
        self.runs = pd.DataFrame(rows)
        if self.runs.empty:
            sys.exit(f"no raw runs matching the manifest in {raw_dir}")

    def select(self, mode=None, K=None, N=None, R=None) -> list[pd.DataFrame]:
        r = self.runs[self.runs["R"] == (R or self.eff["rounds"])]
        if mode is not None:
            r = r[r["mode"] == mode]
        if K is not None:
            r = r[r["K"] == K]
        if N is not None:
            r = r[r["N"] == N]
        return list(r.sort_values("seed")["df"])

    def main(self, mode: str, K: int | None = None, seeds=None) -> list[pd.DataFrame]:
        """Experiment-1 configuration (or the K-sweep point) for a mode, optionally per seed set."""
        r = self.runs[(self.runs["R"] == self.eff["rounds"]) & (self.runs["mode"] == mode)
                      & (self.runs["K"] == (K or self.eff["K"]))
                      & (self.runs["N"] == (1 if mode == "naive_pqc" else self.eff["N"]))]
        if seeds is not None:
            r = r[r["seed"].isin(seeds)]
        return list(r.sort_values("seed")["df"])

    @property
    def scaling_seeds(self) -> list[int]:
        return self.eff.get("scaling_seeds", self.eff["seeds"])


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
        # round time: wall clock incl. local training, excl. test-set evaluation
        "pct_round": 100 * df.crypto_total_s.sum() / (df.wall_s - df.eval_s).sum(),
        "crypto_ms_round": df.crypto_total_s.mean() * 1e3,
        "wire_kb_round": df.bytes_wire_total.mean() / 1024,
        "overhead_b_round": (df.bytes_wire_total - df.plaintext_bytes).mean(),
        "overhead_pct_payload": 100 * (df.bytes_wire_total - df.plaintext_bytes).sum()
                                / df.plaintext_bytes.sum(),
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


NSTAR_EPS = (0.05, 0.10, 0.25)  # tolerances: handshake overhead per round <= eps * floor
EPS_DASH = {0.05: (0, (1, 1.2)), 0.10: (0, (4, 1.5)), 0.25: (0, (6, 1.5, 1, 1.5))}


def compute_nstar(d: Data) -> dict:
    """Re-key cadence analysis on the amortized sweep (R = sweep_rounds).

    f  = symmetric-only floor: mean per-round AES-GCM path time (round-key HKDF + encrypt +
         decrypt, all clients + server), pooled over every round of every sweep run.
    h  = extra cost of one handshake round: mean over handshake rounds of
         (crypto time - AES-GCM path time), pooled over all sweep runs.
    overhead(N) = measured handshake crypto time over the session / R   (per round)
    Analytical  N*(eps) = ceil(h / (eps * f))   from the cost model overhead(N) ~ h / N.
    Measured    N*(eps) = smallest swept N with overhead(N) <= eps * f.
    """
    Ns, R, K = d.eff["N_sweep"], d.eff.get("sweep_rounds", d.eff["rounds"]), d.eff["K"]
    sym_rounds, hs_rounds, per_n = [], [], {}
    for n in Ns:
        dfs = d.select("amortized", K, n, R)
        per_n[n] = pd.DataFrame([summarize(df) for df in dfs])
        for df in dfs:
            sym = (df.sym_encrypt_s + df.sym_decrypt_s) * 1e3
            sym_rounds += list(sym)
            hs = df.handshakes > 0
            hs_rounds += list(df.crypto_total_s[hs] * 1e3 - sym[hs])
    f, h = float(np.mean(sym_rounds)), float(np.mean(hs_rounds))
    h_sd = float(np.std(hs_rounds, ddof=1)) if len(hs_rounds) > 1 else 0.0
    rows = []
    for n in Ns:
        runs = per_n[n]
        ov, tot = runs["handshake_ms"] / R, runs["crypto_ms"] / R
        sd = (lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0)
        rows.append({"N": n, "handshake_rounds": int(np.ceil(R / n)),
                     "crypto_per_round_ms": float(tot.mean()), "crypto_per_round_sd": sd(tot),
                     "hs_overhead_per_round_ms": float(ov.mean()),
                     "hs_overhead_per_round_sd": sd(ov),
                     "hs_overhead_pct_of_floor": float(100 * ov.mean() / f),
                     "model_overhead_per_round_ms": h / n})
    eps_rows = []
    for eps in NSTAR_EPS:
        pred = int(np.ceil(h / (eps * f)))
        meas = next((r["N"] for r in rows if r["hs_overhead_per_round_ms"] <= eps * f), None)
        eps_rows.append({"eps": eps, "threshold_ms": eps * f, "N_star_predicted": pred,
                         "N_star_predicted_on_grid": next((n for n in Ns if n >= pred), None),
                         "N_star_measured": meas})
    return {"definition": "N*(eps) = smallest N with handshake overhead per round <= eps * f; "
                          "analytical N*(eps) = ceil(h / (eps * f))",
            "K": K, "rounds": R, "N_swept": Ns, "f_floor_ms_per_round": f,
            "h_handshake_round_extra_ms": h, "h_sd_ms": h_sd,
            "n_handshake_rounds_pooled": len(hs_rounds), "eps": eps_rows, "rows": rows}


def _fmt_n(n) -> str:
    return "--" if n is None else str(n)


def fig_rekey(d: Data) -> Path:
    ns = compute_nstar(d)
    Ns, R, K = ns["N_swept"], ns["rounds"], ns["K"]
    rows = {r["N"]: r for r in ns["rows"]}
    f, h = ns["f_floor_ms_per_round"], ns["h_handshake_round_extra_ms"]
    nv_mu, _ = stats(d.select("naive_pqc", K, 1, R))
    xs = np.array(Ns)
    x_hi = max(Ns)
    model_x = np.geomspace(1, x_hi, 300)
    m = "amortized"

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.7), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1]})
    a.errorbar(xs, [rows[n]["crypto_per_round_ms"] for n in Ns],
               yerr=[rows[n]["crypto_per_round_sd"] for n in Ns], color=COLOR[m],
               marker=MARKER[m], capsize=2, linewidth=1.4, label=LABEL[m] + " (measured)", zorder=3)
    a.plot(model_x, f + h / model_x, color=INK, linewidth=0.8, label=r"model: $f + h/N$", zorder=2)
    a.axhline(nv_mu["crypto_ms"] / R, color=COLOR["naive_pqc"], linestyle="--", linewidth=1,
              label=LABEL["naive_pqc"])
    a.axhline(f, color=INK_2, linestyle=":", linewidth=1, label="symmetric-only floor $f$")
    lines, eps_handles = [], []
    for e in ns["eps"]:
        eps, pred, meas = e["eps"], e["N_star_predicted"], e["N_star_measured"]
        eps_handles.append(a.axhline(f * (1 + eps), color=INK_2, linewidth=0.6,
                                     linestyle=EPS_DASH[eps], zorder=1,
                                     label=f"$f(1+\\varepsilon)$, $\\varepsilon$={eps:.0%}"))
        if pred <= x_hi:  # predicted N*: open circle on the model curve
            a.plot(pred, f + h / pred, "o", ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=4)
        if meas is not None:  # measured N*: filled diamond on the measured curve
            a.plot(meas, rows[meas]["crypto_per_round_ms"], "D", ms=3.5, color=INK, zorder=5)
        lines.append(f"{eps:>4.0%}   {pred:>5}   {_fmt_n(meas):>5}")
    a.text(0.03, 0.04, " $\\varepsilon$   $N^*$pred  $N^*$meas\n" + "\n".join(lines),
           transform=a.transAxes, fontsize=6, family="monospace", color=INK, va="bottom",
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.5))
    a.plot([], [], "o", ms=4.5, mfc="white", mec=INK, mew=0.8, label="$N^*$ predicted")
    a.plot([], [], "D", ms=3.5, color=INK, label="$N^*$ measured")
    a.set_ylim(0, max(rows[Ns[0]]["crypto_per_round_ms"], nv_mu["crypto_ms"] / R) * 1.08)
    eps_key = a.legend(handles=eps_handles, loc="lower right", fontsize=6, handlelength=2.4,
                       frameon=True, framealpha=1, edgecolor=GRID, borderpad=0.3)
    eps_key.get_frame().set_linewidth(0.5)
    a.add_artist(eps_key)
    a.set_ylabel("Crypto time per round (ms)")
    main_h = [hd for hd in a.get_legend_handles_labels()[0] if hd not in eps_handles]
    a.legend(handles=main_h, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=6.5,
             handlelength=2.0, columnspacing=1.0, borderaxespad=0)

    b.plot(xs, [stats(d.select(m, K, n, R))[0]["hs_kb"] / R for n in Ns], color=COLOR[m],
           marker=MARKER[m], label=LABEL[m])
    b.axhline(nv_mu["hs_kb"] / R, color=COLOR["naive_pqc"], linestyle="--", linewidth=1,
              label=LABEL["naive_pqc"])
    b.set_ylim(bottom=0)
    b.set_ylabel("Handshake KB /\nclient / round")
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator(Ns))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xticklabels([str(n) for n in Ns])
    b.set_xlim(min(Ns) / 1.15, x_hi * 1.15)
    b.set_xlabel(f"Re-key interval $N$ (rounds), $K={K}$, $R={R}$")
    b.legend(loc="upper right", fontsize=6.5)
    fig.align_ylabels()
    return _save(fig, d, "rekey_sweep")


def fig_scaling(d: Data) -> Path:
    Ks = d.eff["K_sweep"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.1), sharex=True)
    for m in MODES:
        st = [stats(d.main(m, K, d.scaling_seeds)) for K in Ks]
        for ax, key in ((a, "crypto_ms"), (b, "per_client_round_ms")):
            ax.errorbar(Ks, [s[0][key] for s in st], yerr=[s[1][key] for s in st],
                        color=COLOR[m], marker=MARKER[m], linestyle=DASH[m], capsize=2,
                        label=LABEL[m])
    a.set_ylabel(f"Crypto time, {d.eff['rounds']} rounds,\nall clients + server (ms)")
    b.set_ylabel("Crypto time per client\nper round (ms)")
    for ax in (a, b):
        ax.set_ylim(bottom=0)
    b.xaxis.set_major_locator(FixedLocator(Ks))
    b.set_xlabel(f"Number of clients $K$ ($N={d.eff['N']}$, {len(d.scaling_seeds)} seeds)")
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
                                f"{mu['hs_kb']:.1f}", _pm(mu["pct_round"], sd["pct_round"], "{:.3f}"),
                                _pm(mu["final_acc"], sd["final_acc"])]))
    hdr = ("Mode & $N$ & HS & Crypto (ms) & Handshake (ms) & AES-GCM (ms) & "
           "HS KB/client & \\% round & Acc. (\\%)")
    note = (f"$K={d.eff['K']}$, {d.eff['rounds']} rounds, {len(d.eff['seeds'])} seeds; "
            "times summed over all clients and the server; HS = handshakes; \\% round = crypto "
            "share of wall-clock round time incl.\\ local training, excl.\\ test-set evaluation.")
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
    (OUT / "tables" / "nstar.json").write_text(json.dumps({**ns, "placeholder": d.quick}, indent=2))
    K, R = ns["K"], ns["rounds"]
    f, h = ns["f_floor_ms_per_round"], ns["h_handshake_round_extra_ms"]
    rows = []
    for r in ns["rows"]:
        rows.append(" & ".join([
            str(r["N"]), str(r["handshake_rounds"]),
            _pm(r["crypto_per_round_ms"], r["crypto_per_round_sd"], "{:.3f}"),
            _pm(r["hs_overhead_per_round_ms"], r["hs_overhead_per_round_sd"], "{:.3f}"),
            f"{r['model_overhead_per_round_ms']:.3f}", f"{r['hs_overhead_pct_of_floor']:.1f}"]))
    nv, _ = stats(d.select("naive_pqc", K, 1, R))
    rows.append(f"\\midrule\nNaive PQC & {R} & {nv['crypto_ms'] / R:.3f} & "
                f"{nv['handshake_ms'] / R:.3f} & -- & {100 * nv['handshake_ms'] / R / f:.1f}")
    hdr = ("$N$ & HS rounds & Crypto/round & HS overhead/round & Model $h/N$ & \\% of $f$")
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Amortized PQC versus re-key interval $N$ (times in ms)}", "\\label{tab:rekey}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrrr}", "\\toprule", hdr + " \\\\", "\\midrule",
        *[x + " \\\\" for x in rows], "\\bottomrule", "\\end{tabular}", "\\\\[6pt]",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "$\\varepsilon$ & $N^*$ predicted $\\lceil h/(\\varepsilon f)\\rceil$ & "
        "next swept $N$ & $N^*$ measured \\\\", "\\midrule",
        *[f"{e['eps'] * 100:.0f}\\% & {e['N_star_predicted']} & "
          f"{_fmt_n(e['N_star_predicted_on_grid'])} & {_fmt_n(e['N_star_measured'])} \\\\"
          for e in ns["eps"]],
        "\\bottomrule", "\\end{tabular}",
        f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize $K={K}$, $R={R}$, "
        f"{len(d.eff['seeds'])} seeds. $f={f:.3f}$\\,ms/round (symmetric-only floor), "
        f"$h={h:.3f}\\pm{ns['h_sd_ms']:.3f}$\\,ms (extra cost of one handshake round, "
        f"{ns['n_handshake_rounds_pooled']} rounds pooled). Measured $N^*$: smallest swept $N$ "
        f"with handshake overhead per round $\\leq\\varepsilon f$; -- = not reached.}}",
        "\\end{table}"]
    return _write_table("rekey_sweep", d, "\n".join(lines) + "\n")


def table_overhead(d: Data) -> Path:
    rows = []
    for m in MODES:
        mu, sd = stats(d.main(m))
        rows.append(" & ".join([
            LABEL[m], _pm(mu["crypto_ms_round"], sd["crypto_ms_round"], "{:.3f}"),
            _pm(mu["pct_round"], sd["pct_round"], "{:.3f}"),
            f"{mu['wire_kb_round']:.1f}", f"{mu['overhead_b_round']:,.0f}".replace(",", "{,}"),
            f"{mu['overhead_pct_payload']:.3f}"]))
    hdr = ("Mode & Crypto/round (ms) & Crypto \\% of round & Wire KB/client/round & "
           "Overhead B/client/round & Overhead \\% of payload")
    note = (f"$K={d.eff['K']}$, $N={d.eff['N']}$ (Naive PQC: $N=1$), {d.eff['rounds']} rounds, "
            f"{len(d.eff['seeds'])} seeds. Round time is wall-clock including local training, "
            "excluding test-set evaluation. Overhead bytes = all protocol bytes beyond the "
            "serialized update (KEM keys and ciphertexts, signatures, nonce, GCM tag, framing).")
    return _write_table("overhead", d, _table(
        "table*", "Compute versus bandwidth overhead per round", "tab:overhead", "lrrrrr",
        hdr, rows, note))


def table_scaling(d: Data) -> Path:
    rows = []
    for K in d.eff["K_sweep"]:
        cells = [str(K)]
        for m in MODES:
            mu, sd = stats(d.main(m, K, d.scaling_seeds))
            cells += [_pm(mu["crypto_ms"], sd["crypto_ms"]), _pm(mu["pct_round"], sd["pct_round"], "{:.3f}")]
        rows.append(" & ".join(cells))
    hdr = ("$K$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{LABEL[m]}}}" for m in MODES)
           + " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & "
           + " & ".join(["Crypto (ms) & \\% round"] * 3))
    return _write_table("client_scaling", d, _table(
        "table*", "Crypto cost versus number of clients $K$", "tab:scaling", "lrrrrrr", hdr, rows,
        f"$N={d.eff['N']}$ (Naive PQC: $N=1$), {d.eff['rounds']} rounds, "
        f"{len(d.scaling_seeds)} seeds; times summed over all clients and the server."))


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


# --- Model-size bandwidth analysis ------------------------------------------------

def bandwidth_constants(d: Data) -> dict:
    """Per-client byte constants measured in the main-comparison runs.

    o     = bytes added to every encrypted update beyond the serialized model update S
            (96-bit nonce, GCM tag, message framing), from rounds without a handshake.
    H_m   = bytes of one handshake per client (both messages incl. framing) for suite m,
            = overhead in handshake rounds - o.
    S_cnn = serialized size of our CNN update.
    """
    frames = {m: pd.concat(d.main(m)) for m in MODES}
    o_vals, H = [], {}
    for m, df in frames.items():
        ov = df.bytes_wire_total - df.plaintext_bytes
        hs = df.handshakes > 0
        o_vals += list(ov[~hs])
        H[m] = ov[hs]
    o = float(np.mean(o_vals))
    if np.ptp(o_vals) != 0:
        sys.exit(f"per-update overhead is not constant across rounds: {sorted(set(o_vals))}")
    H = {m: float(v.mean()) - o for m, v in H.items()}
    if H["naive_pqc"] != H["amortized"]:
        sys.exit("PQC handshake size differs between naive and amortized runs")
    return {"o_bytes": o, "H_classical_bytes": H["classical"], "H_pqc_bytes": H["amortized"],
            "S_cnn_bytes": float(frames["amortized"].plaintext_bytes.mean()),
            "measured_pct": {m: float(100 * (df.bytes_wire_total - df.plaintext_bytes).sum()
                                      / df.plaintext_bytes.sum()) for m, df in frames.items()}}


def _bw_configs(d: Data, c: dict) -> list[tuple[str, str, float, int, str]]:
    N = d.eff["N"]
    return [  # (label, color key, H, N, linestyle)
        (f"Classical, $N={N}$", "classical", c["H_classical_bytes"], N, DASH["classical"]),
        ("Naive PQC ($N=1$)", "naive_pqc", c["H_pqc_bytes"], 1, DASH["naive_pqc"]),
        (f"Amortized PQC, $N={N}$", "amortized", c["H_pqc_bytes"], N, DASH["amortized"]),
        ("Amortized PQC, $N=20$", "amortized", c["H_pqc_bytes"], 20, (0, (1, 1.2))),
    ]


def overhead_pct(S, o: float, H: float, N: int):
    """Per-round, per-client protocol overhead as % of the model update: 100 (o + H/N) / S."""
    return 100 * (o + H / N) / np.asarray(S, dtype=float)


BW_SIZES = [10 * 1024, 100 * 1024, 1024**2, 10 * 1024**2]  # bytes; plus our CNN


def _size_label(b: float) -> str:
    return f"{b / 1024**2:.0f} MB" if b >= 1024**2 else f"{b / 1024:.0f} KB"


def fig_bandwidth(d: Data) -> Path:
    c = bandwidth_constants(d)
    S = np.geomspace(10 * 1024, 10 * 1024**2, 300)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for label, ck, H, N, ls in _bw_configs(d, c):
        ax.plot(S / 1024, overhead_pct(S, c["o_bytes"], H, N), color=COLOR[ck], linestyle=ls,
                label=label)
        ax.plot(c["S_cnn_bytes"] / 1024, overhead_pct(c["S_cnn_bytes"], c["o_bytes"], H, N),
                marker=MARKER[ck], color=COLOR[ck], markeredgecolor="white", markeredgewidth=0.5,
                linestyle="none")
    ax.axvline(c["S_cnn_bytes"] / 1024, color=INK_2, linewidth=0.7, linestyle=":")
    ax.annotate(f"our CNN\n({c['S_cnn_bytes'] / 1024:.0f} KB)", (c["S_cnn_bytes"] / 1024, 1e-3),
                xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 10 * 1024)
    ax.set_ylim(1e-3, 200)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1024, 10 * 1024]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10 KB", "100 KB", "1 MB", "10 MB"])
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("Serialized model update size $S$ (per client)")
    ax.set_ylabel("Protocol overhead per round\n(% of update size)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=6.5, handlelength=2.2,
              columnspacing=1.0, borderaxespad=0)
    return _save(fig, d, "bandwidth_vs_model_size")


def table_bandwidth(d: Data) -> Path:
    c = bandwidth_constants(d)
    (OUT / "tables" / "bandwidth_constants.json").write_text(json.dumps(c, indent=2))
    cfgs = _bw_configs(d, c)
    sizes = sorted(BW_SIZES + [c["S_cnn_bytes"]])
    rows = []
    for S in sizes:
        ours = S == c["S_cnn_bytes"]
        name = f"{_size_label(S)} (our CNN)" if ours else _size_label(S)
        cells = [f"{overhead_pct(S, c['o_bytes'], H, N):.3g}" for _, _, H, N, _ in cfgs]
        rows.append(" & ".join([f"\\textbf{{{name}}}" if ours else name] + cells))
    hdr = "Update size $S$ & " + " & ".join(lbl for lbl, *_ in cfgs)
    o, Hc, Hp = c["o_bytes"], c["H_classical_bytes"], c["H_pqc_bytes"]
    note = (f"Overhead $= 100\\,(o + H/N)/S$\\,\\% per client per round, with measured "
            f"$o={o:.0f}$\\,B per update (nonce, GCM tag, framing), $H={Hc:.0f}$\\,B per "
            f"classical handshake and $H={Hp:,.0f}$\\,B per PQC handshake (KEM key and "
            f"ciphertext, two signatures, framing). Steady state ($R$ a multiple of $N$).")
    return _write_table("bandwidth", d, _table(
        "table*", "Per-round protocol overhead (\\% of model update) versus update size",
        "tab:bandwidth", "lrrrr", hdr.replace(",", "{,}"), rows, note.replace("8,954", "8{,}954")))


# --- Qualitative security comparison ----------------------------------------------

P, NP, B = "Protected", "Not protected", "Bounded by $N$"
SECURITY_ROWS = [
    # (threat, [(cell, [tests]) for classical, naive_pqc, amortized])
    ("Harvest-now-decrypt-later",
     [(NP + " (X25519 secrets recoverable by a future quantum adversary)", []),
      (P + " (ML-KEM-768)", ["test_wire_sizes"]),
      (P + " (ML-KEM-768)", ["test_wire_sizes"])]),
    ("Active MITM on the handshake",
     [(P + "$^{\\dagger}$ (Ed25519, both directions)", ["test_forged_server_response_rejected",
                                                      "test_substituted_kem_key_rejected"]),
      (P + " (ML-DSA-65, both directions)", ["test_forged_server_response_rejected",
                                             "test_substituted_kem_key_rejected"]),
      (P + " (ML-DSA-65, both directions)", ["test_forged_server_response_rejected",
                                             "test_substituted_kem_key_rejected"])]),
    ("Replayed update",
     [(P, ["test_replayed_update_from_old_round_rejected"]),
      (P, ["test_replayed_update_from_old_round_rejected"]),
      (P, ["test_replayed_update_from_old_round_rejected"])]),
    ("Reordered / duplicate update",
     [(P, ["test_reordered_update_rejected", "test_rollback_to_earlier_round_rejected"]),
      (P + " (future-round key does not exist yet)", ["test_replayed_update_from_old_round_rejected"]),
      (P, ["test_reordered_update_rejected"])]),
    ("Tampered ciphertext",
     [(P, ["test_tampered_ciphertext_rejected"]),
      (P, ["test_tampered_ciphertext_rejected"]),
      (P, ["test_tampered_ciphertext_rejected"])]),
    ("Session-key compromise exposure window",
     [(B + " rounds (whole session if keyed once)", ["test_old_epoch_key_rejected_after_rekey"]),
      (P + " beyond 1 round (fresh key every round)", ["test_roundtrip_and_rekey_schedule"]),
      (B + " rounds", ["test_old_epoch_key_rejected_after_rekey"])]),
]


def table_security(d: Data) -> Path:
    for _, cells in SECURITY_ROWS:
        for _, tests in cells:
            missing = [t for t in tests if t not in TEST_PURPOSE]
            if missing:
                sys.exit(f"security table cites tests not in the suite: {missing}")
    rows = []
    for threat, cells in SECURITY_ROWS:
        out = [threat]
        for text, tests in cells:
            t = "".join(f"\\newline{{\\scriptsize\\texttt{{{_tex_escape(x.removeprefix('test_'))}}}}}"
                        for x in tests) or "\\newline{\\scriptsize (no test: property of the primitive)}"
            out.append(text + t)
        rows.append(" & ".join(out))
    note = ("Test names refer to \\texttt{tests/test\\_crypto.py} (prefix \\texttt{test\\_} omitted). "
            "$^{\\dagger}$Against classical adversaries; a quantum-capable active attacker could "
            "forge Ed25519 in real time. Exposure window: updates readable after compromise of one "
            "session key $K_e$; long-term identity-key compromise exposes no past updates in any "
            "mode (fresh KEM keys per epoch) but allows impersonation in later handshakes.")
    colspec = (">{\\raggedright\\arraybackslash}p{0.17\\linewidth}"
               + ">{\\raggedright\\arraybackslash}p{0.255\\linewidth}" * 3)
    return _write_table("security", d, _table(
        "table*", "Qualitative security comparison", "tab:security", colspec,
        f"Threat & Classical ($N={d.eff['N']}$) & Naive PQC & Amortized PQC ($N={d.eff['N']}$)",
        rows, note), from_results=False)


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

    for f in (fig_per_round, fig_cumulative, fig_bytes, fig_rekey, fig_scaling, fig_accuracy,
              fig_bandwidth):
        print(f"  figure  {f(d).relative_to(ROOT)}")
    for t in (table_main, table_overhead, table_primitives, table_rekey, table_scaling,
              table_accuracy):
        print(f"  table   {t(d).relative_to(ROOT)}")
    ns = compute_nstar(d)
    f, h = ns["f_floor_ms_per_round"], ns["h_handshake_round_extra_ms"]
    print(f"\n  Re-key cadence (amortized, K={ns['K']}, R={ns['rounds']}): f = {f:.4f} ms/round, "
          f"h = {h:.4f} +/- {ns['h_sd_ms']:.4f} ms ({ns['n_handshake_rounds_pooled']} handshake rounds)")
    print(f"  {'N':>4} {'crypto/round':>13} {'HS overhead/round':>18} {'model h/N':>10} {'% of f':>8}")
    for r in ns["rows"]:
        print(f"  {r['N']:>4} {r['crypto_per_round_ms']:>10.4f} ms {r['hs_overhead_per_round_ms']:>15.4f} ms "
              f"{r['model_overhead_per_round_ms']:>10.4f} {r['hs_overhead_pct_of_floor']:>7.1f}%")
    for e in ns["eps"]:
        print(f"  eps={e['eps']:>4.0%}: N* predicted = ceil(h/(eps f)) = {e['N_star_predicted']} "
              f"(next swept N: {_fmt_n(e['N_star_predicted_on_grid'])}), measured = {_fmt_n(e['N_star_measured'])}")
    print()
    path, passed, total = table_tests(d)
    print(f"  table   {path.relative_to(ROOT)}  ({passed}/{total} test cases passed)")
    for t in (table_bandwidth, table_security):
        print(f"  table   {t(d).relative_to(ROOT)}")
    c = bandwidth_constants(d)
    print(f"\n  Bandwidth model: o = {c['o_bytes']:.0f} B/update, H = {c['H_classical_bytes']:.0f} B "
          f"(classical) / {c['H_pqc_bytes']:.0f} B (PQC) per handshake, S_cnn = {c['S_cnn_bytes']:.0f} B")
    for label, _, H, N, _ in _bw_configs(d, c):
        print(f"    {label:<24}" + "  ".join(
            f"{_size_label(S):>6}: {overhead_pct(S, c['o_bytes'], H, N):8.4f}%"
            for S in sorted(BW_SIZES + [c['S_cnn_bytes']])))
    print("    check vs measured at S_cnn: " + ", ".join(
        f"{m} {v:.4f}%" for m, v in c["measured_pct"].items()))

    if not args.skip_latex:
        for tex in sorted((OUT / "diagrams").glob("*.tex")) + [OUT / "main.tex"]:
            if tex.exists() and tectonic(tex):
                print(f"  latex   {tex.with_suffix('.pdf').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
