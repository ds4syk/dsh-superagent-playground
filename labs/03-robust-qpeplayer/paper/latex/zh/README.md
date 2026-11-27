# 03-robust-qpeplayer — LaTeX 论文（zh）

- **目标形式：** 中文双栏会议格式（IROS 式）
- **文档类：** `ctexart, twocolumn`
- **编译命令：**

  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
  /mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
  ```

- **引擎：** Tectonic 0.17.0（`/mnt/storage1/syk/tools/tectonic/tectonic`）。
- **事实来源：** `../../draft.md` v0.4（Markdown，不修改）。本文树中的所有数字均可追溯到
  该文件及其 `experiments/` 结果文件。

## 构建状态

| 项目 | 值 |
|---|---|
| 最近编译 | 2026-11-28，退出码 0 |
| 正文页数 | 7（第 1–7 节；参考文献从第 7 页开始） |
| 总页数 | 13 |
| 表 / 图 | 10 / 3 |
| 警告 | 无未定义引用；最大 overfull hbox 2.59 pt |

## 文件

`main.tex`（文档类、标题、摘要、关键词、\input 行）、`preamble.tex`、
`sections/*.tex`、`figures/*.png`、`refs.bib`（与英文树一致）、`build.log`。

## 评测范围（v0.4）

1. 正文 7 页；附录不计入页数限制。
2. 内层求解器为加权 Kabsch，即 QPEP 的闭式 3D–3D 特化；4×4 Davenport/Horn 变体在附录中
   做了基准测试。通用 40 根 QPEP 内层求解器与 GPU 批处理留作未来工作。
3. In-The-Dark 证据为两次白天运行（各 15 对）与两个各 10 对的夜间窗口，前端为手写
   SIFT+StereoSGBM，且无训练。
4. 3DMatch 覆盖全部八个官方测试场景，并含严格的 0.20–0.50 低重叠带；KITTI odometry 覆盖
   00–10，每个序列 33 个选定帧。
5. Theseus-GM 与 SDPRLayers+SCS 是合成 3D–3D 上的外部基线。
6. 6D 符号翻转为数值结果；严格结论是一维命题。
7. 鲁棒代价为局部求解；相干多数外点可使全局最优本身出错。
8. KITTI 双目前端能求解旋转，但平移仍受前端限制。

## 编辑规则

- 表格：`booktabs`，宽表用 `table*` 与 `\scriptsize`；不使用竖线。
- 图：`figure` 内用 `\\includegraphics[width=\\columnwidth]{...}`；宽总览图用 `figure*`
  与 `width=\\textwidth`。
- 作者、单位与基金信息保持通用，不要杜撰。
