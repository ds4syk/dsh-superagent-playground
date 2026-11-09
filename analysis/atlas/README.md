# SDPR 复现图表集

- `SDPR_复现图表集.docx`：7 张清洗后的图 + 读图说明 + 数据口径（默认交付）。
- `SDPR_复现图表集.pdf`：同上的 PDF 预览。
- `figures/`：单张 PNG（200 dpi），可直接插入 PPT/论文。
- `make_atlas.py`：从 `repro/results_clean/` 与 `repro/results_upstream/` 的原始数据重画。
- `build_docx.py`：组装 DOCX。

数据口径：只画我们复现的数据，不画论文值；轨迹类图用“中位数 + 50% 分位带”。
SDPR 复现的指标是 baseline error、outer loss、迭代数、紧性/证书、Jacobian；没有 episode/reward（那是 RL 训练指标）。
