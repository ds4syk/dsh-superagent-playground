# KITTI per-drive heterogeneity

| metric | drive 0001 | drive 0002 |
|---|---|---|
| random_p0.0: GNC/RANSAC success | 100/100% | 100/100% |
| random_p0.0: GNC speedup vs RANSAC-1000 | 65.1× | 65.1× |
| random_p0.0: GNC med rot | 0.03° | 0.04° |
| random_p0.3: GNC/RANSAC success | 100/100% | 100/100% |
| random_p0.3: GNC speedup vs RANSAC-1000 | 22.5× | 20.8× |
| random_p0.3: GNC med rot | 0.04° | 0.04° |
| random_p0.5: GNC/RANSAC success | 100/100% | 100/100% |
| random_p0.5: GNC speedup vs RANSAC-1000 | 22.1× | 23.6× |
| random_p0.5: GNC med rot | 0.04° | 0.05° |
| random_p0.7: GNC/RANSAC success | 100/100% | 100/100% |
| random_p0.7: GNC speedup vs RANSAC-1000 | 22.2× | 23.3× |
| random_p0.7: GNC med rot | 0.06° | 0.05° |
| coherent_p0.4: GNC/IRLS/RANSAC | 100/100/100 | 100/100/100 |
| coherent_p0.5: GNC/IRLS/RANSAC | 35/45/0 | 40/30/0 |
| coherent_p0.6: GNC/IRLS/RANSAC | 0/0/0 | 0/5/0 |
| boundary_p0.5: GNC/RANSAC rot | 0.04/0.08° | 0.05/0.08° |
| boundary_p0.7: GNC/RANSAC rot | 0.07/0.14° | 0.09/0.16° |

## Reading

* The two drives are independent OXTS/calib runs; differences in speedup are node/load and motion/overlap, not algorithmic.
* The coherent 50% tie regime is the only place where the two drives differ materially; both collapse at 60%.
