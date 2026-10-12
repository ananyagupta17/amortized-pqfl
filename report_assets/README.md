# Report assets

Regenerate everything (figures, tables, diagrams, `main.pdf`) from experiment results:

```bash
python scripts/run_experiments.py                    # full data -> results/
python scripts/make_report_assets.py                 # -> report_assets/
python scripts/make_report_assets.py --results results/quick   # placeholder build
```

| Path | Contents |
|---|---|
| `figures/crypto_per_round.pdf` | Per-round crypto time by mode (mean ± std) |
| `figures/crypto_cumulative.pdf` | Cumulative crypto time over rounds by mode |
| `figures/bytes_components.pdf` | Protocol bytes per client: KEM key, KEM ciphertext, signatures, AEAD nonce+tag, framing |
| `figures/rekey_sweep.pdf` | Crypto time per round versus N (R=60) against the cost model f + h/N, with predicted and measured N\*(ε) for ε ∈ {5, 10, 25}% |
| `tables/nstar.json` | The N\* analysis: f (symmetric-only floor), h (extra cost of a handshake round), per-N overhead, and predicted vs measured N\*(ε) |
| `figures/client_scaling.pdf` | Crypto time versus K (total, and per client per round) |
| `figures/accuracy.pdf` | Final accuracy per mode with error bars and per-seed points |
| `diagrams/architecture.{tex,pdf}` | System architecture (TikZ) |
| `diagrams/sequence.{tex,pdf}` | Re-key round followed by a symmetric-only round (TikZ) |
| `tables/*.tex` | booktabs tables: main comparison, compute vs bandwidth overhead, primitives, re-key sweep with N\*(ε), client scaling, accuracy, tests |
| `build_issues.{md,tex}` | Issues hit during the build and their fixes |
| `main.tex` / `main.pdf` | IEEEtran skeleton: headings only, all figures and tables placed |

Assets built from `results/quick` are marked as quick-profile placeholders (a watermark on figures, a caption prefix on tables).
Figures are vector PDFs with embedded TrueType fonts, sized for a 3.5 in IEEE column. The diagrams span both columns (`figure*`).
