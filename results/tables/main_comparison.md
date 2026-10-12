Experiment 1: main comparison. Mean ± std over seeds. Times are summed over all clients + server over the whole session.

| mode | K | N | handshakes | crypto total (ms) | handshake crypto (ms) | AES-GCM path (ms) | handshake KB / client | wire MB / client | crypto % of round time | final acc (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| classical | 5 | 5 | 20 | 38.00 ± 0.61 | 9.10 ± 0.08 | 28.89 ± 0.55 | 0.8 | 16.43 | 0.089 ± 0.001 | 98.21 ± 0.06 |
| naive_pqc | 5 | 1 | 100 | 79.87 ± 0.54 | 50.66 ± 0.84 | 29.21 ± 0.43 | 173.6 | 16.60 | 0.182 ± 0.003 | 98.21 ± 0.06 |
| amortized | 5 | 5 | 20 | 39.93 ± 0.60 | 10.25 ± 0.17 | 29.68 ± 0.46 | 34.7 | 16.47 | 0.091 ± 0.001 | 98.21 ± 0.06 |
