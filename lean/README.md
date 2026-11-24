# Lean 4 formalization (1-D tie / no-free-lunch)

This directory contains a machine-checked Lean 4 proof of the 1-D tie lemma
used in Theorem 3 of the paper: for `J x θ = (x^2 - 1)^2 + θ x`, no continuous
selection of the global minimizer can take the value `±1` at `θ = 0`.

The general face--stratum theorem (SDP/KKT/IFT/SOSC) is proved in the paper;
a full Lean formalization of the SDP/KKT part is future work.

**Verified**: `lake build` completed successfully (8765 jobs) with
Mathlib v4.34.0-rc2; see `lake_build3.log`.

## Contents

- `FaceStratum/Tie.lean`: 1-D tie / no-free-lunch, machine-checked.
- `FaceStratum/ValueFace.lean`: rank component of value-face separation (`WM=0` => `range(M) ⊆ ker W`, `rank M ≤ dim ker W`), machine-checked.
- The PSD step `tr(WM)=0 => WM=0` is paper (spectral theorem); general SDP/KKT/IFT is future work.

## Build

```bash
cd lean
lake exe cache get   # Mathlib cache
lake build
```

The project uses Lean `v4.34.0-rc2` and Mathlib.
