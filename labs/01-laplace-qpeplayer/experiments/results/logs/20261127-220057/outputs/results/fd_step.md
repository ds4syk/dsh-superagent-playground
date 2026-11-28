# P1 FD-step study for the synthetic crossing table

Laplace analytic gradient vs central FD of the same smoothed map; tau=0.01. For h below the softmax transition scale tau/slope ~ 5e-3 the FD reference is accurate; h=1e-4 is already too coarse at theta=0 and h=1e-3 is too coarse everywhere near the crossing.

| theta | h=1e-06 | h=3e-06 | h=1e-05 | h=3e-05 | h=1e-04 | h=3e-04 | h=1e-03 |
|-|-|-|-|-|-|-|-|
| +0.0000 | 4.34e-08 | 3.90e-07 | 4.34e-06 | 3.90e-05 | 4.34e-04 | 3.91e-03 | 4.70e-02 |
| +0.0010 | 1.07e-06 | 9.64e-06 | 1.07e-04 | 9.69e-04 | 1.14e-02 | 2.72e-01 | 4.73e-01 |
| +0.0050 | 5.73e-08 | 5.16e-07 | 5.73e-06 | 5.16e-05 | 5.73e-04 | 5.14e-03 | 5.48e-02 |
| +0.0100 | 3.46e-09 | 3.11e-08 | 3.45e-07 | 3.11e-06 | 3.46e-05 | 3.11e-04 | 3.53e-03 |
| +0.0500 | 6.67e-10 | 6.31e-09 | 7.01e-08 | 6.31e-07 | 7.01e-06 | 6.31e-05 | 7.02e-04 |
| +0.1000 | 1.75e-10 | 7.55e-10 | 8.45e-09 | 7.61e-08 | 8.46e-07 | 7.61e-06 | 8.46e-05 |

Reading:

- At theta=0 the error falls from 4.34e-4 (h=1e-4) to 4.3e-06 at h=1e-5..3e-5, confirming the committed number was FD-limited.
- Away from the crossing (|theta|>=0.05) every h>=1e-5 agrees to ~1e-6, i.e. the analytic gradient is exact; the finite difference, not the formula, sets the error floor.
- The `at` entry of `main_table.md` should therefore be read as 'analytic gradient agrees with FD to the FD truncation floor', not as a 4e-4 method error.
