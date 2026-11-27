# 3DMatch: success vs base (clean) inlier fraction

| base-inlier bin | p | method | n | success | median rot |
|---|---|---|---|---|---|
| [0.90,0.95) | 0.7 | GNC-GM | 1 | 0% | 42.87° |
| [0.90,0.95) | 0.7 | RANSAC-1000 | 1 | 0% | 42.87° |
| [0.90,0.95) | 0.7 | IRLS-GM | 1 | 0% | 42.87° |
| [0.95,1.00) | 0.0 | GNC-GM | 240 | 100% | 0.23° |
| [0.95,1.00) | 0.0 | RANSAC-1000 | 240 | 100% | 0.27° |
| [0.95,1.00) | 0.0 | IRLS-GM | 240 | 100% | 0.23° |
| [0.95,1.00) | 0.3 | GNC-GM | 240 | 100% | 0.24° |
| [0.95,1.00) | 0.3 | RANSAC-1000 | 240 | 100% | 0.31° |
| [0.95,1.00) | 0.3 | IRLS-GM | 240 | 100% | 0.24° |
| [0.95,1.00) | 0.5 | GNC-GM | 240 | 81% | 0.30° |
| [0.95,1.00) | 0.5 | RANSAC-1000 | 240 | 67% | 0.61° |
| [0.95,1.00) | 0.5 | IRLS-GM | 240 | 85% | 0.28° |
| [0.95,1.00) | 0.7 | GNC-GM | 239 | 67% | 0.49° |
| [0.95,1.00) | 0.7 | RANSAC-1000 | 239 | 67% | 0.83° |
| [0.95,1.00) | 0.7 | IRLS-GM | 239 | 66% | 0.50° |

## Reading

The base-inlier fraction is the fraction of GT-NN pairs inside c before injected outliers. If success is flat across bins, the controlled outlier rate, not base quality, drives failures.
