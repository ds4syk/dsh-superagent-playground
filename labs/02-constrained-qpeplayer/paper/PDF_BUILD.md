# P2 paper PDF build (EN/ZH)

Toolchain (no LaTeX distro needed): conda-installed `tectonic` + the pandoc binary from the conda package cache.

```bash
export https_proxy=http://127.0.0.1:7899 http_proxy=http://127.0.0.1:7899
PANDOC=/home/syk/miniconda3/pkgs/pandoc-3.8-h06a4308_0/bin/pandoc
TECTONIC=/mnt/storage1/syk/miniconda3/envs/tex/bin/tectonic

# English (35 pages)
cd labs/02-constrained-qpeplayer/paper
$PANDOC draft.md -o draft_en.pdf --pdf-engine=$TECTONIC \
  --resource-path=.:../experiments/results -V geometry:margin=2cm -V fontsize=9pt

# Chinese (draft_zh.md; CJK main font)
$PANDOC draft_zh.md -o draft_zh.pdf --pdf-engine=$TECTONIC \
  --resource-path=.:../experiments/results -V geometry:margin=2cm -V fontsize=9pt \
  -V CJKmainfont="Noto Serif CJK SC" -V mainfont="Noto Serif CJK SC"
```

The PDFs are generated artifacts; the markdown sources (`draft.md`, `draft_zh.md`) are authoritative.
