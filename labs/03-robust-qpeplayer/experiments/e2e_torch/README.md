# e2e_torch — a real PyTorch `autograd.Function` for the 3D-3D robust pose layer

> **One-line result**: the exact IFT backward passes `torch.autograd.gradcheck`
> at N=6/8/10 with max absolute gradient error ~**1e-9** and reproduces the
> NumPy reference in `experiments/synthetic/common.py` to **1e-16**; the
> frozen-weight ("naive") backward fails gradcheck by **O(1)** relative error.
> In the small learned refinement toy the exact backward wins **4/5** paired
> seeds (median final pose loss **1.74e-5 vs 3.17e-5**, median rotation
> **0.068° vs 0.106°**), with one documented basin-switch failure.

## Scope (honest)

This is **not** In-The-Dark and not a realistic learned registration pipeline.
It is a small synthetic 3D-3D toy (N=10 points, 5 train / 3 held-out instances
per seed, boundary outliers `e/c² ~ U(1/3,3)`, c=0.05, 4 outliers/instance)
whose only purpose is to check that the layer's backward is *the* right
gradient and that it is usable in a learned end-to-end loop.

* GT-pose supervision, one tiny MLP (`10→32→32→3`), no descriptors, no real
  scans, no SE(3) network, no GPU, no large-scale training.
* The **refine** protocol deliberately allows memorisation (train = eval); it
  is an optimisation/gradient diagnostic.
* The **held-out** protocol is a *negative* scope check: a purely local
  per-point MLP cannot identify the random global pose correction, so both
  arms overfit and the held-out pose gets worse. Do **not** read this toy as
  evidence that the layer generalises.
* The forward GNC solve is non-convex: finite Adam steps can move boundary
  points across the GM band and switch the robust basin, which no gradient
  models (one such seed per protocol is reported, not hidden).

## Files

| file | what |
|---|---|
| `robust_layer.py` | float64 torch layer: weighted Kabsch, GNC-GM, Newton polish, exact/naive/stopgrad `autograd.Function` backward, `torch.autograd.gradcheck`-ready API |
| `gradcheck_layer.py` | gradcheck at N=6/8/10 + elementwise comparison with `common.py`; writes `results/gradcheck.json` |
| `learned_refine.py` | 5-seed learned refinement (MLP predicts `delta_y`), exact vs naive vs stopgrad, both protocols; writes `results/e2e_torch.*` |
| `results/e2e_torch.json` | all numbers (settings, gradcheck, protocols, per-seed, curves) |
| `results/e2e_torch.md` | report with tables and scope |
| `results/e2e_torch_curves.png` | median refine curves (pose loss / rotation error) |
| `results/gradcheck.json` | standalone gradcheck record |
| `results/run_full.log` | raw log of the run that produced the saved numbers |

The torch formulas are a direct port of `experiments/synthetic/common.py`
(`robust_hessian_exact`, `robust_hessian_naive`, `dphi_dy_exact`,
`dphi_dy_naive`, `newton_polish`); `common.py` is **not modified**.

## Layer

Forward: GNC-GM (GM kernel, weighted-Kabsch inner solve, `gamma=1.4`) then
Newton polish on `S³ × R³` until `|Phi| ≲ 1e-15`, so the returned pose is a
true stationary point of `Σ ρ(e_i)`.

Backward (`dL/dy` given `dL/dR, dL/dt`):

```
dz   = [B_q^T dL/dq ; dL/dt]
w    = H_rob^{-1} dz
dL/dy_i = 2 ρ'(e_i) (J_i w_q + w_t) + 2 ρ''(e_i) (g_i · w) r_i
H_rob = B^T [Σ ρ' H_i + ρ'' g_i g_i^T] B − λ_rob diag(I_3, 0_3)
```

`naive` sets `ρ'' = 0` in both `H` and `dL/dy` (IRLS-StopGrad); `stopgrad`
returns zeros. Only `y` receives gradient; `x` and the `A` tensors are
constants.

## Exact commands

```bash
cd /mnt/storage1/syk/.superagent-work/wt-p3-e2e/labs/03-robust-qpeplayer/experiments/e2e_torch

# 1) layer validation (gradcheck + port fidelity), ~30 s
/home/syk/miniconda3/envs/WM/bin/python gradcheck_layer.py

# 2) full learned experiment (re-runs gradcheck), ~2 min CPU
/home/syk/miniconda3/envs/WM/bin/python learned_refine.py --steps 50 --heldout-steps 60 --seeds 5

# re-render results/e2e_torch.md from the saved JSON without retraining
/home/syk/miniconda3/envs/WM/bin/python -c "import json,learned_refine as L; L.write_markdown(json.load(open('results/e2e_torch.json')),'results/e2e_torch.md')"
```

Environment: `/home/syk/miniconda3/envs/WM/bin/python`, torch 2.11.0 CPU
float64, numpy 2.2.6. The full run took ~2 min on CPU, so it was run directly
(no Slurm needed); anything longer should go through
`srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2`.

## Results (from `results/e2e_torch.json`)

Gradcheck, float64, eps=1e-6, atol=1e-7, rtol=1e-4:

| N | exact | max abs err | max rel err | naive gradcheck |
|---|---|---|---|---|
| 6 | PASS | 6.1e-10 | 8.1e-9 | FAIL (max rel 1.8) |
| 8 | PASS | 7.9e-10 | 1.1e-8 | FAIL (max rel 1.5) |
| 10 | PASS | 6.3e-10 | 8.2e-6* | FAIL (max rel 1.8) |

\* elementwise on one near-zero component; `max|Δg| / max|g| = 7e-10`.

Port fidelity vs `common.py`: pose 1.1e-16 / 5.6e-17, `dL/dy` 3.2e-16 (exact)
and 1.6e-16 (naive).

Learned (median over 5 seeds, refine protocol 50 Adam steps, lr=1e-2):

| arm | initial rot | final rot | initial loss | final loss |
|---|---|---|---|---|
| exact | 0.635° | **0.068°** | 3.48e-4 | **1.74e-5** |
| naive | 0.635° | 0.106° | 3.53e-4 | 3.17e-5 |
| stopgrad | 0.635° | 0.635° | 3.87e-4 | 3.87e-4 |

Paired: exact < naive on **4/5** seeds, median ratio 0.56; exact < stopgrad
4/5. Seed 4 is the documented basin-switch failure of the exact arm.

## Keep / revert

**Keep** as a small, self-contained validation of the layer backward. The
gradcheck + `common.py` agreement are the load-bearing evidence; the learned
toy is a supporting diagnostic with an honest negative held-out result.
