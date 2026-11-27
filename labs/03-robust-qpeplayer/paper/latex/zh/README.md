# 03-robust-qpeplayer — LaTeX draft (zh)

- **Target venue:** 中文双栏会议格式（IROS 式）
- **Document class:** `ctexart, twocolumn`
- **Compile command:**

  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
  /mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
  ```

- **Engine:** Tectonic 0.17.0 (`/mnt/storage1/syk/tools/tectonic/tectonic`).
- **Source of truth:** `../../draft.md` (Markdown; do not modify). All numbers in this
  tree must be traceable to that file and its `experiments/` result files.

## Build status

| item | value |
|---|---|
| last compile | see `build.log` |
| main-text pages | TBD |
| total pages | TBD |
| tables / figures | TBD |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## Known pending claims / gaps (must stay visible in the paper)

1. 正文须控制在 6 页以内；核函数对偶/消融明细表放入附录，并明确标注不计入页数限制。
2. 未实现 40 根 QPEP 求解器：内层是加权 Kabsch（3D-3D 的闭式特化），另有 4×4 Davenport/Horn 变体单独基准。
3. In-The-Dark 证据是两个白天 run、每个 run 15 对连续帧、手写 SIFT+StereoSGBM 前端且无训练；夜间、眩光、SDPR VGG/UNet 前端均未测试。
4. 3DMatch 覆盖 8 个官方场景（非全部 50 个）；KITTI odometry 00/05/07 每序列 33 个选定帧（非完整 00–10）。
5. BPnP/Pyatov/POLI 在 3D-3D 任务上不在范围内；Theseus-GM 与 SDPRLayers+SCS 仅是合成前向 baseline。
6. 4×4 内层求解器的 GPU/CUDA-graph 支路因节点占用未运行；只完成 CPU 验证。
7. 6D 符号翻转是数值观察，不是定理；严格结论是 1D 命题。
8. 没有全局最优性、崩溃点或对抗鲁棒性保证；相干多数外点可使鲁棒全局最优解本身错误。
9. KITTI 双目回退实验旋转为正、平移为负（0.1 m 成功率 0%），不能替代训练好的前端。

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
