# E2E CPU toy — final measured summary

Refiner: 2-layer MLP over 14 keypoints, 3 seeds x 2000 steps with the FB-reduced backward (CPU, ~144 s/seed). Four heads at evaluation on 64 validation scenes.

## Table 1. Four heads (n=64, medians / rates)

| head | ADD mm | rot deg | trans mm | pen mm | feasible | ms/pose |
|---|---|---|---|---|---|---|
| unc | 101.06 | 5.29 | 101.03 | 104.351 | 0% | 1.16 |
| alm | 12.34 | 3.78 | 11.06 | 0.000 | 100% | 10.19 |
| proj | 13.19 | 4.96 | 11.98 | 0.000 | 95% | 1.75 |
| soft | 12.34 | 3.78 | 11.06 | 0.012 | 0% | 5.76 |

## Table 2. FB backward vs central differences (10 seeds)

| mode | rel L2 median | rel L2 max | cos min |
|---|---|---|---|
| unc | 4.60e-10 | 3.49e-09 | 1.0000000000 |
| alm_eps_1e-02 | 4.86e-01 | 3.95e+00 | 0.2389895225 |
| alm_eps_1e-04 | 7.37e-04 | 4.46e-02 | 0.9990052186 |
| alm_eps_1e-06 | 5.73e-06 | 1.76e-03 | 0.9999987054 |
| alm_eps_1e-08 | 4.64e-06 | 1.76e-03 | 0.9999987053 |
| proj | 3.88e-08 | 3.99e-04 | 0.9999999205 |
| soft | 4.05e-07 | 5.07e-04 | 0.9999998793 |

Kink test (n=10): median rel_sym 5.73e-06, best-side 8.45e-05, asymmetry 1.65e-04.

Scope: CPU toy only; no VLA/detector, no GPU; the refiner is trained with the unconstrained FB backward and the constrained heads are applied at evaluation (ALM 10.2 ms/pose would make 2000-step ALM training ~8x costlier); the ALM FB backward is gradient-checked, not trained for 2000 steps. See `paper/draft.md` §5.8.
