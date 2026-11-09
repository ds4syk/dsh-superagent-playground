from pathlib import Path

from src.md2docx import render

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "report" / "sdpr_forward_network_dataset.md"
OUT = HERE / "deliverables" / "SDPR_前向网络与数据集_2026-11.docx"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render(
        SRC,
        OUT,
        title="SDPR 的可微优化层：前向网络、接法与数据集",
        subtitle="读论文 + 读代码：VGG16 + UNet 前端、13 维 QCQP / SDPRLayer、UTIAS In-The-Dark",
        meta="2026-11-09 · 依据 arXiv:2405.19309v3、utiasASRL/sdprlayer、deep_learned_visual_features（mat-weight-sdp-version）",
    )
    print(f"saved {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")
