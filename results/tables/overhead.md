| mode | K | N | crypto / round (ms) | crypto % of round time | crypto % of wall | wire KB / client / round | protocol overhead B / client / round | overhead % of payload |
|---|---|---|---|---|---|---|---|---|
| classical | 5 | 5 | 1.900 ± 0.031 | 0.089 ± 0.001 | 0.067 ± 0.001 | 841.4 | 123 | 0.014 |
| naive_pqc | 5 | 1 | 3.994 ± 0.027 | 0.182 ± 0.003 | 0.138 ± 0.002 | 850.1 | 9026 | 1.048 |
| amortized | 5 | 5 | 1.997 ± 0.030 | 0.091 ± 0.001 | 0.069 ± 0.001 | 843.1 | 1863 | 0.216 |
