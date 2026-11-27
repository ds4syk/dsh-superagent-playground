# P1 real-data attempt: LINEMOD symmetric objects (not used in the main table)

> **Status: negative/inconclusive attempt.** 我们尝试用 BOP LINEMOD 的真实对称物体（eggbox obj 10、glue obj 11）构造双峰代价：把同一组图像 bearing 分别与模型点 $P$ 和对称变换后的 $S P$ 配对，得到两个 bearing 代价矩阵，再按 $\theta$ 混合。目标是在某个 $\theta$ 让两个极小等值（分支交叉），用 Laplace/硬 argmin/MC 对比。

**结果**：在 9 帧候选里，`find_modes` 对混合代价始终只返回 1 个极小（或没有可用的等值交叉），因此没有产出可用的真实数据主表。**没有伪造数字，也没有把这个尝试写进 P1 主表**；P1 的主表仍是合成双峰实验（`experiments/README.md`）。

**可能原因**（待查）：
- 两个 bearing 代价矩阵的极小并不严格对应 $A$ 与 $A S$；凸混合后仍可能只有一个极小，除非两个代价在各自极小处的曲率/位置分离足够大。
- eggbox/glue 的对称变换与 LINEMOD 的物体坐标系/模型顶点并不完全对齐（模型本身有噪声/非理想对称）。
- 需要显式构造“半半点对应”的混合（一半点来自 A、一半来自 B），而不是混合两个完整代价矩阵。

**保留代码**：`run_symmetry.py`（可复现上述失败）；`results/symmetry.md` 为空表。真实数据加载本身已用投影 bbox 与 GT bbox 对齐校验通过（见 P2 的 `real/linemod/README.md`）。
