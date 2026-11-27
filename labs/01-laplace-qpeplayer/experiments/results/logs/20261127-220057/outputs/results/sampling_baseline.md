# P1 continuous sampling baseline (Laplace importance sampling)

tau=0.01, n_starts=256, 4000 SNIS samples, 12 seeds, proposal = per-mode Gaussian N(0, s*tau*H^-1) with s=1.5/2.0.

| theta | K | ESS(s=1.5) | |Laplace - SNIS(1.5)| | |SNIS(1.5) - SNIS(2.0)| | max |w_Lap - w_SNIS| | SNIS seed std | mode-level MC seed std |
|-|-|-|-|-|-|-|-|
| +0.000 | 4 | 6 | 4.12e-01 | 2.10e-01 | 4.204e-01 | 4.89e-01 | 8.31e-02 |
| +0.002 | 3 | 266 | 1.99e-01 | 8.69e-01 | 6.341e-02 | 1.32e-01 | 3.49e-02 |
| +0.010 | 3 | 501 | 3.58e-01 | 8.15e-02 | 9.956e-05 | 1.25e-01 | 4.61e-03 |
| -0.010 | 3 | 317 | 3.84e-01 | 3.21e-02 | 2.718e-04 | 1.24e-01 | 4.62e-03 |
| +0.050 | 2 | 3476 | 4.52e-02 | 1.04e-02 | 1.174e-16 | 4.54e-03 | 2.53e-16 |
| +0.100 | 2 | 3429 | 2.27e-02 | 1.51e-03 | 1.637e-31 | 3.21e-03 | 4.17e-16 |

Reading:

- `|SNIS(1.5) - SNIS(2.0)|` is a proposal-sensitivity check; it is the floor of the SNIS estimate. `|Laplace - SNIS|` above that floor is the Laplace-approximation bias.
- The mode-level MC seed std is ~0 because it samples whole modes with the Laplace weights; it measures the Monte-Carlo variance of the *same* Laplace posterior, not of the true Boltzmann target. Continuous SNIS has strictly larger seed std at the same sample budget, which is the fair variance tax of sampling the continuous posterior.

## STATUS: inconclusive at the crossing (2026-11-27, do not use these numbers as a bias estimate)

The self-normalised IS estimate collapses near the branch crossing: at theta=0 the
effective sample size is **6 / 4000** (s=1.5) and the s=1.5 vs s=2.0 estimates differ
by 0.21 in pose, i.e. larger than the Laplace/SNIS difference itself. The cause is the
near-degenerate mode: its Hessian is small, so the Laplace Gaussian proposal
N(0, tau H^-1) has very heavy tails in the soft tangent direction and the target's
anharmonic region is under-covered. Over-dispersion s=1..16 does not fix it (measured
ESS 4, 12, 16, 27, 38 at theta=0).

Consequence: this round is **negative/inconclusive**. It does not establish a
Laplace-vs-continuous-sampling bias on the 6D crossing. A valid version needs a
heavy-tailed (Student-t) or adaptive proposal, or a local numerical integral; that is
left as future work. Away from the crossing (|theta| >= 0.05) the ESS is >3400 and the
Laplace/SNIS pose difference is small (2.3e-2 at theta=0.1), but even there the
proposal-sensitivity floor (1.5e-3) is not negligible, so we do not report it in the
paper. The script is kept as the starting point for the proper version.
