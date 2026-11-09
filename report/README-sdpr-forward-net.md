# SDPR 前向网络与数据集（2026-11）

回答 2026-11-09 的 Feishu 提问：**SDPR 论文的可微分优化层，前向网络是怎么接的、是个什么网络？数据集是哪里来的？**

| 材料 | 位置 |
|-|-|
| 主报告（Markdown） | [sdpr_forward_network_dataset.md](sdpr_forward_network_dataset.md) |
| 前向管线图 | [figures/pipeline.png](figures/pipeline.png) |
| Word 版（默认交付） | [../deliverables/SDPR_前向网络与数据集_2026-11.docx](../deliverables/SDPR_前向网络与数据集_2026-11.docx) |
| DOCX 构建脚本 | [../src/build_docx_sdpr_forward_net.py](../src/build_docx_sdpr_forward_net.py)、[../src/md2docx.py](../src/md2docx.py) |
| 管线图脚本 | [../src/make_pipeline_figure.py](../src/make_pipeline_figure.py) |

## 一句话结论

SDPR 层本身无可学习参数；论文真实实验里接在它前面的是 **VGG16（ImageNet 预训练，截到 `conv5_3`）+ 两个 UNet 式解码器** 的关键点前端，输出 768 个关键点、960 维描述子与得分，经 dense 匹配、双目三角化、标量/矩阵权重后，把 **13 维 QCQP** 的 Q 矩阵送进 `SDPRLayer`（Mosek），前向解 Shor SDP、反向做隐式微分。数据集是 **UTIAS In-The-Dark**（39 条 260 m 路线，约 30 小时每小时一条，Grizzly + Bumblebee XB3 双目）；论文用 10 万训练 + 2 万验证样本，测试 runs 2/11/16/17/23/28/35。

## 证据与边界

- **代码核对**：`utiasASRL/sdprlayer` 与 `utiasASRL/deep_learned_visual_features@ba66e408`；张量形状、960 维描述子、768 关键点、13 维 Q、24 个约束均来自源码。
- **未验证**：没有下载 In-The-Dark、没有 Mosek license、没有跑训练/测试；Table VI/VII 的数字是论文声称，不是本报告复现。
- **上游坑**：`mat-weight-sdp-version` HEAD 缺 `pose_sdp_block.py`（最后一次提交删了 `sdpr_block.py` 但没提交改名文件）；复现用 `ba66e408` 或改用 `sdprlayer` 包的 `SDPPoseEstimator`。

## 复现文档

```bash
# 图
python src/make_pipeline_figure.py

# DOCX
python src/build_docx_sdpr_forward_net.py
```

依赖：`matplotlib`、`python-docx`；中文字体用 Noto Sans/Serif CJK SC。
