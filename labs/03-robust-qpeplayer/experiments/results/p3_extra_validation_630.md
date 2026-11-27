# P3 accepted-round ledger (700)

- prior accepted: **70** (`rounds_2026-11-27.md` rows 73–142)
- new accepted: **630**
  - anisotropic covariance grid: **550** (550/550 executable units)
  - real Theseus baseline cells: **40** (2 scenes × 4 p × 5 methods)
  - learned refinement units: **40** (5 seeds × 2 protocols × 2 arms × 2 p)
- total: **700**

## Notes

- Anisotropic units: 5 regimes × 5 p × 11 seeds × 2 solvers = 550; accepted = finite completed solve.
- Theseus cells: aggregated over 5 trials/cell; identity check re-ran GNC and matched stored raw exactly.
- Learned units: exact/naive arms only; stopgrad runs are not counted.
- Failures/retries are excluded; the anisotropic ledger has 0 rejected units.
- The 550-unit anisotropic grid is new evidence: matched full-W is better than mismatched full-W (p=0.7 median 0.184° vs 0.383°; success 100% vs 91%), so correct covariance helps when it matches the noise.
