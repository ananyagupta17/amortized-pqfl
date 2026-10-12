Experiment 2: re-key cadence sweep (amortized), naive_pqc as reference.

| mode | N | handshakes | crypto total (ms) | crypto / round (ms) | handshake crypto (ms) | handshake KB / client | crypto % of round time |
|---|---|---|---|---|---|---|---|
| amortized | 1 | 300 | 242.46 ± 2.76 | 4.041 ± 0.046 | 153.81 ± 2.33 | 520.9 | 0.183 ± 0.003 |
| amortized | 2 | 150 | 173.26 ± 1.44 | 2.888 ± 0.024 | 81.49 ± 0.73 | 260.4 | 0.132 ± 0.001 |
| amortized | 3 | 100 | 144.96 ± 4.37 | 2.416 ± 0.073 | 53.53 ± 1.73 | 173.6 | 0.107 ± 0.005 |
| amortized | 5 | 60 | 120.07 ± 1.18 | 2.001 ± 0.020 | 31.74 ± 0.63 | 104.2 | 0.092 ± 0.001 |
| amortized | 10 | 30 | 108.71 ± 1.60 | 1.812 ± 0.027 | 16.22 ± 0.49 | 52.1 | 0.082 ± 0.001 |
| amortized | 15 | 20 | 103.44 ± 1.28 | 1.724 ± 0.021 | 10.88 ± 0.30 | 34.7 | 0.078 ± 0.001 |
| amortized | 20 | 15 | 98.48 ± 1.95 | 1.641 ± 0.032 | 7.99 ± 0.44 | 26.0 | 0.074 ± 0.001 |
| amortized | 30 | 10 | 96.78 ± 2.19 | 1.613 ± 0.037 | 5.59 ± 0.29 | 17.4 | 0.073 ± 0.002 |
| amortized | 60 | 5 | 94.47 ± 2.29 | 1.574 ± 0.038 | 2.88 ± 0.28 | 8.7 | 0.073 ± 0.001 |
| naive_pqc | 1 | 300 | 257.79 ± 3.83 | 4.296 ± 0.064 | 164.34 ± 2.39 | 520.9 | 0.196 ± 0.003 |

N=1 vs naive_pqc over 5 seed(s): handshakes identical: True; handshake bytes identical: True; crypto time ratio 0.941 (paired diff -15.33 ± 4.81 ms).
