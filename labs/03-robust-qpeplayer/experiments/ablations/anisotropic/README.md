# Anisotropic-weight ($W_i$) ablation — 3D-3D robust registration

Self-contained ablation answering: **can the GNC-GM / IRLS robust outer loop use
per-observation SPD weights $W_i$, and what does that cost?**

## The objective (and why scalar Kabsch is not enough)

The correct GM-robust objective is

$$\Phi(R,t)=\sum_i \rho(e_i),\qquad e_i=r_i^\top W_i r_i,\qquad
r_i=R x_i+t-y_i,\qquad \rho(e)=\frac{c^2 e}{c^2+e},\; c=0.05 .$$

At fixed GM weights $u_i=\rho'(e_i)$ the inner problem is
$\min_{R,t}\sum_i u_i\,r_i^\top W_i r_i$. Because $W_i$ sits **inside** the
residual, this is a nonlinear least-squares problem in the 6-DoF pose. It reduces
to scalar weighted Kabsch **only** when every $W_i$ is a scalar multiple of $I$;
`experiments/synthetic/common.py::kabsch` therefore cannot solve it in general.

This directory solves the inner problem by 6-DoF Gauss-Newton on the
Cholesky-whitened residuals $\sqrt{u_i}\,L_i^\top(R x_i+t-y_i)$ with
$W_i=L_iL_i^\top$, re-anchoring the rotation chart every iteration, and
cross-checks against `scipy.optimize.least_squares`. The GNC schedule, kernel
($c=0.05$, $\gamma=1.4$, $5$ polish steps at $\mu=1$) and instance generator
(`common.make_instance`, N=100, random outliers) are the same as the main
synthetic table; `common.py` is **not** modified.

Rows labelled **APPROX-\\*-sK** replace the inner solve with scalar weighted
Kabsch. They are an explicitly labelled approximation used only to quantify how
far the surrogate is from the correct weighted objective — they are *not* a
solution of $\min\sum_i r_i^\top W_i r_i$.

## Weight families

| name | construction | median cond |
|---|---|---|
| `identity` | $W_i=I_3$ | 1 |
| `diag` | $W_i=\mathrm{diag}(d_1,d_2,d_3)$, $d_k\sim U(0.5,2)$ | ~1.9 |
| `full` | $W_i=(A_iA_i^\top+0.05I)/\det(\cdot)^{1/3}$, $A_i\sim\mathcal N(0,I)$ (Wishart, $\det W_i=1$) | ~31–36 |

Weights are drawn from a seed stream independent of the data; the **same
instances are reused across the three families** so the effect of anisotropy can
be measured paired. 25 trials/cell, $p\in\{0,0.3,0.5,0.7\}$, $N=100$.

## Results (summary; numbers from `results/anisotropic.json`)

* Correctness: with $W_i=I$ the GN solver reproduces closed-form Kabsch to
  $5\times10^{-13}$ (Frobenius), and full GNC-W matches `common.solve_gnc` to
  $5\times10^{-12}$. `scipy.optimize.least_squares` agrees on the anisotropic
  weighted problems in the same chart ($1.7\times10^{-15}$ relative cost) and in
  an independent unit-quaternion + $t$ parameterisation ($1.9\times10^{-14}$
  relative cost, pose diff $<10^{-9}$).
* Scalar-Kabsch inner is exact for $W_i=I$ (objective ratio 1.000000) but not
  for anisotropic weights (median ratio 1.002–1.007 diag, 1.035–1.088 full;
  up to 3.8 on individual full-W trials). This is the direct evidence that
  scalar Kabsch does not solve the anisotropic weighted problem.
* Under this protocol the synthetic noise is **isotropic**, so anisotropic
  $W_i$ is a misspecified model: full-W GNC-W gives median rotation errors
  ~0.19–0.35 deg vs ~0.11–0.19 deg for $W=I$ (paired median change
  +0.055…+0.125 deg; anisotropic better on only 8–32% of paired trials).
  Diag-W is close to identity (+0.007…+0.020 deg). **Anisotropic weights do not
  improve accuracy here**; the correct solver still reaches a lower $\Phi$ than
  the scalar surrogate in 12/12 weight/p cells, which is what verifies the
  solver. Success stays 92–100% (5 deg / 0.1 threshold).
* At p=0.7, full-W GNC-W has 2/25 gross failures (159°, 175° rotation) on
  instances where W=I succeeds; their $\Phi$ (0.245) is worse than identity's
  (0.182) on the same data, i.e. the anisotropic weighting pushed GNC into a
  worse local minimum, not a threshold artefact.
* Runtime: the exact nonlinear inner solve costs 5–23× the scalar-Kabsch GNC
  call (p50 at p≥0.3: ~20–48 ms vs ~1.9–2.1 ms), because each inner problem is
  a real 6-DoF nonlinear solve.

## Files

| file | content |
|---|---|
| `run_anisotropic.py` | solver, correctness checks, main table, report/figures |
| `results/anisotropic.json` | config, aggregated rows, correctness checks, paired stats, inexactness ratios |
| `results/anisotropic_raw.jsonl` | one line per GNC-W trial (all solvers) |
| `results/anisotropic.md` | generated report (all tables) |
| `results/anisotropic.png` | success / rotation error / runtime |
| `results/logs/` | run log |

## Reproduce

```bash
cd labs/03-robust-qpeplayer/experiments/ablations/anisotropic
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
  /home/syk/miniconda3/envs/WM/bin/python run_anisotropic.py 2>&1 | \
  tee results/logs/anisotropic_$(date +%Y%m%d_%H%M%S).log
```

Fixed seed: `BASE_SEED=20261127`; correctness checks use `SeedSequence` seeds
`(20261127, 9, ...)` and abort the run on failure. Total wall time on the lab
CPU node: ~13 s for 300 instances × 4 solvers.
