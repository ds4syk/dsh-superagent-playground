# Phase-5 anisotropic covariance grid

550/550 solver units accepted; N=100, sigma=0.01.

| regime | p | solver | n | success | rot med | trans med | p50 ms |
|---|---|---|---|---|---|---|---|
| identity | 0.0 | GNC-W | 11 | 100% | 0.091 | 0.0015 | 2.9 |
| identity | 0.0 | IRLS-W | 11 | 100% | 0.091 | 0.0015 | 6.9 |
| identity | 0.2 | GNC-W | 11 | 100% | 0.097 | 0.0020 | 21.5 |
| identity | 0.2 | IRLS-W | 11 | 100% | 0.097 | 0.0020 | 8.0 |
| identity | 0.4 | GNC-W | 11 | 100% | 0.196 | 0.0013 | 22.0 |
| identity | 0.4 | IRLS-W | 11 | 100% | 0.196 | 0.0013 | 6.0 |
| identity | 0.6 | GNC-W | 11 | 100% | 0.150 | 0.0026 | 27.9 |
| identity | 0.6 | IRLS-W | 11 | 100% | 0.150 | 0.0026 | 6.5 |
| identity | 0.7 | GNC-W | 11 | 91% | 0.250 | 0.0033 | 31.8 |
| identity | 0.7 | IRLS-W | 11 | 91% | 0.250 | 0.0033 | 7.2 |
| diag_matched | 0.0 | GNC-W | 11 | 100% | 0.095 | 0.0015 | 2.5 |
| diag_matched | 0.0 | IRLS-W | 11 | 100% | 0.095 | 0.0015 | 5.6 |
| diag_matched | 0.2 | GNC-W | 11 | 100% | 0.101 | 0.0011 | 15.9 |
| diag_matched | 0.2 | IRLS-W | 11 | 100% | 0.101 | 0.0011 | 5.7 |
| diag_matched | 0.4 | GNC-W | 11 | 100% | 0.111 | 0.0019 | 21.9 |
| diag_matched | 0.4 | IRLS-W | 11 | 100% | 0.111 | 0.0019 | 6.2 |
| diag_matched | 0.6 | GNC-W | 11 | 100% | 0.132 | 0.0023 | 27.8 |
| diag_matched | 0.6 | IRLS-W | 11 | 100% | 0.132 | 0.0023 | 6.4 |
| diag_matched | 0.7 | GNC-W | 11 | 100% | 0.210 | 0.0027 | 31.6 |
| diag_matched | 0.7 | IRLS-W | 11 | 100% | 0.210 | 0.0027 | 7.1 |
| full_matched | 0.0 | GNC-W | 11 | 100% | 0.062 | 0.0009 | 4.0 |
| full_matched | 0.0 | IRLS-W | 11 | 100% | 0.062 | 0.0009 | 8.9 |
| full_matched | 0.2 | GNC-W | 11 | 100% | 0.056 | 0.0010 | 24.6 |
| full_matched | 0.2 | IRLS-W | 11 | 100% | 0.056 | 0.0010 | 6.8 |
| full_matched | 0.4 | GNC-W | 11 | 100% | 0.082 | 0.0015 | 32.5 |
| full_matched | 0.4 | IRLS-W | 11 | 100% | 0.082 | 0.0015 | 8.2 |
| full_matched | 0.6 | GNC-W | 11 | 100% | 0.103 | 0.0013 | 53.0 |
| full_matched | 0.6 | IRLS-W | 11 | 100% | 0.103 | 0.0013 | 9.4 |
| full_matched | 0.7 | GNC-W | 11 | 100% | 0.184 | 0.0026 | 42.5 |
| full_matched | 0.7 | IRLS-W | 11 | 100% | 0.184 | 0.0026 | 7.5 |
| diag_mismatch | 0.0 | GNC-W | 11 | 100% | 0.109 | 0.0014 | 2.5 |
| diag_mismatch | 0.0 | IRLS-W | 11 | 100% | 0.109 | 0.0014 | 5.3 |
| diag_mismatch | 0.2 | GNC-W | 11 | 100% | 0.144 | 0.0019 | 17.6 |
| diag_mismatch | 0.2 | IRLS-W | 11 | 100% | 0.144 | 0.0019 | 6.3 |
| diag_mismatch | 0.4 | GNC-W | 11 | 100% | 0.147 | 0.0021 | 21.5 |
| diag_mismatch | 0.4 | IRLS-W | 11 | 100% | 0.147 | 0.0021 | 6.4 |
| diag_mismatch | 0.6 | GNC-W | 11 | 100% | 0.228 | 0.0034 | 30.0 |
| diag_mismatch | 0.6 | IRLS-W | 11 | 100% | 0.228 | 0.0034 | 6.7 |
| diag_mismatch | 0.7 | GNC-W | 11 | 100% | 0.159 | 0.0031 | 33.2 |
| diag_mismatch | 0.7 | IRLS-W | 11 | 100% | 0.159 | 0.0031 | 7.2 |
| full_mismatch | 0.0 | GNC-W | 11 | 100% | 0.183 | 0.0018 | 4.5 |
| full_mismatch | 0.0 | IRLS-W | 11 | 100% | 0.183 | 0.0018 | 7.5 |
| full_mismatch | 0.2 | GNC-W | 11 | 100% | 0.122 | 0.0028 | 21.6 |
| full_mismatch | 0.2 | IRLS-W | 11 | 100% | 0.122 | 0.0028 | 8.8 |
| full_mismatch | 0.4 | GNC-W | 11 | 100% | 0.241 | 0.0038 | 25.6 |
| full_mismatch | 0.4 | IRLS-W | 11 | 100% | 0.241 | 0.0038 | 9.4 |
| full_mismatch | 0.6 | GNC-W | 11 | 100% | 0.232 | 0.0035 | 32.5 |
| full_mismatch | 0.6 | IRLS-W | 11 | 100% | 0.232 | 0.0034 | 10.0 |
| full_mismatch | 0.7 | GNC-W | 11 | 91% | 0.383 | 0.0052 | 46.9 |
| full_mismatch | 0.7 | IRLS-W | 11 | 91% | 0.383 | 0.0050 | 12.2 |

## Reading

* Matched regimes use noise covariance W_i^{-1}; mismatched regimes use isotropic noise with the same W_i, so accuracy loss is misspecification, not solver error.
* Each row aggregates 11 seeds; each seed is one accepted ledger unit.
