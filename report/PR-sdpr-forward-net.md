This is by an automated agent, not a human.

## 这是什么

回答 Feishu 问题：**SDPR 论文/代码里，可微分优化层的前向网络是怎么接的、是什么网络？数据集是从哪里来的？**

简短结论：

- **`SDPRLayer` 本身没有可学习参数，也不是一个网络。** 论文真实实验里，它位于可微管线末端。
- 前端是 **VGG16（ImageNet 预训练，截到 `conv5_3`）+ 两个 UNet 式解码器**（关键点 logits、得分），
  每帧输出 768 个关键点、960 维逐像素描述子和得分。
- 接法：双目图像 → 关键点/描述子/得分 → dense 描述子匹配 → 3D 点 + 双目协方差 → 标量×矩阵权重 →
  **13 维 QCQP** → `SDPRLayer`（Mosek）→ 相对位姿；位姿/关键点损失经隐函数微分回传前端。
- 真实数据实验用 **UTIAS In-The-Dark**（UTIAS Long-Term Localization and Mapping Dataset 的夜间子集：
  39 条 260 m 路线、约 30 小时每小时一条、Grizzly 机器人 + Bumblebee XB3 双目）。论文用
  10 万训练 + 2 万验证样本，测试 runs 2/11/16/17/23/28/35；仿真实验的数据由脚本现场生成。

## 交付物

- `report/sdpr_forward_network_dataset.md` — 主报告（中文），含逐段链路、张量形状、代码位置与数据集表。
- `report/figures/pipeline.png` — 前向管线图。
- `deliverables/SDPR_前向网络与数据集_2026-11.docx` — Word 版（默认交付，9 页）。
- `kb/sdpr-forward-net/00.md`、`kb/sdpr-forward-net/01.md` — Feishu 知识库源页。
- `src/` — 图与 DOCX 构建脚本。

## 验证

- 读论文（`arXiv:2405.19309v3`，T-RO 2025）和两个代码仓库。
- 代码核对：VGG16 截断位置、960 维描述子拼接、16×16/768 关键点、196,608 个目标的 dense matching、
  13×13 Q、24 个位姿约束、数据集划分，以及 `mat-weight-sdp-version` HEAD（`a9f0dc2`）缺失
  `pose_sdp_block.py`。
- DOCX 转 PDF 后逐页检查（9 页，中文、表头、图、参考文献均正常）；Feishu 00/01 页已 fetch 校验，
  owner 已授 `full_access`。
- 未做数值复现：没有下载 In-The-Dark，也没有跑训练/测试。

## 限制与注意

- Table VI/VII 的数字是论文声称，不是本 PR 复现结果。
- 公开的 `mat-weight-sdp-version` 分支在 HEAD 跑不起来：`pipeline.py` 导入
  `pose_sdp_block.PoseSDPBlock`，但该文件从未提交；最后可用提交是 `ba66e408`
  （`src/model/sdpr_block.py`）。
- 发布版 `SDPPoseEstimator` 默认 `diff_qcqp=False`（SDPR-SDP），而论文管线的 `SDPRBlock` 用
  `SDPRLayer` 默认 `diff_qcqp=True`（SDPR-IS）；比较梯度和耗时时先对齐这个开关。
