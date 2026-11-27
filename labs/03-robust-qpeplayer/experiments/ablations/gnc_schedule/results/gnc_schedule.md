# GNC schedule ablation

N=100, c=0.05, 50 trials/cell, random outliers. Default: gamma=1.4, mu0 multiplier=1.0, 5 polish steps.

| config | p | success | rot med (deg) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| default | 0.3 | 100% | 0.124 | 1.74 | 2.26 |
| gamma=1.2 | 0.3 | 100% | 0.124 | 4.33 | 4.99 |
| gamma=1.7 | 0.3 | 100% | 0.124 | 2.03 | 2.22 |
| gamma=2.0 | 0.3 | 100% | 0.124 | 1.67 | 1.78 |
| mu0x=0.5 | 0.3 | 100% | 0.124 | 2.65 | 2.74 |
| mu0x=4.0 | 0.3 | 100% | 0.124 | 1.92 | 3.17 |
| polish=0 | 0.3 | 100% | 0.124 | 1.52 | 1.71 |
| polish=10 | 0.3 | 100% | 0.124 | 2.66 | 3.31 |
| default | 0.5 | 100% | 0.147 | 1.93 | 2.91 |
| gamma=1.2 | 0.5 | 100% | 0.147 | 2.83 | 2.98 |
| gamma=1.7 | 0.5 | 100% | 0.147 | 1.25 | 1.39 |
| gamma=2.0 | 0.5 | 100% | 0.147 | 1.08 | 1.45 |
| mu0x=0.5 | 0.5 | 100% | 0.147 | 2.34 | 2.45 |
| mu0x=4.0 | 0.5 | 100% | 0.147 | 2.74 | 2.89 |
| polish=0 | 0.5 | 100% | 0.147 | 1.54 | 1.72 |
| polish=10 | 0.5 | 100% | 0.147 | 1.98 | 2.51 |
| default | 0.7 | 100% | 0.206 | 1.73 | 2.39 |
| gamma=1.2 | 0.7 | 100% | 0.206 | 3.29 | 4.87 |
| gamma=1.7 | 0.7 | 100% | 0.206 | 1.26 | 1.31 |
| gamma=2.0 | 0.7 | 100% | 0.206 | 1.05 | 1.12 |
| mu0x=0.5 | 0.7 | 100% | 0.206 | 1.63 | 1.70 |
| mu0x=4.0 | 0.7 | 100% | 0.206 | 2.92 | 3.02 |
| polish=0 | 0.7 | 100% | 0.206 | 1.51 | 2.35 |
| polish=10 | 0.7 | 100% | 0.206 | 1.97 | 2.03 |

## Reading

* The default is only defensible if no schedule variant gives a material success/accuracy win at comparable or lower runtime.
* mu0 multiplier below 1 can start too narrow and miss the inlier basin; above 1 is safer but slower. polish=0 leaves the first-order residual larger.
