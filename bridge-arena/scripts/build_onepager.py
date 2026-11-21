"""生成一页速览 PNG（docs/visuals 用）。"""
from __future__ import annotations

import base64
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report/figures")
OUT = os.path.join(ROOT, "..", "docs/visuals/2026-11-21-bridge-arena-models.png")
CHROME = os.path.expanduser("~/.cache/puppeteer/chrome/linux-131.0.6778.204/chrome-linux64/chrome")
CN = {"transformer": "Transformer", "diffusion": "Diffusion", "cnn": "CNN-VAE"}
SUB = {"transformer": "自回归 · 121 步逐杆生成", "diffusion": "DDIM 100 步 · 并行去噪",
       "cnn": "卷积 VAE · 一次前向"}
COL = {"transformer": "#1971c2", "diffusion": "#7048e8", "cnn": "#e8590c"}


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    met = {m["name"]: m for m in json.load(open(os.path.join(ROOT, "results/eval/metrics.json")))["metrics"]}
    pg = ["transformer", "diffusion", "cnn"]
    best_pass = max(pg, key=lambda m: met[m]["pass_rate"])
    best_mean = max(pg, key=lambda m: met[m]["score_mean"])
    best_single = max(pg, key=lambda m: met[m]["score_max"])
    best_div = max(pg, key=lambda m: met[m]["diversity"])
    cards = ""
    for m in pg:
        x = met[m]
        cards += f"""
        <div class="card">
          <div class="chead"><span class="dot" style="background:{COL[m]}"></span>
            <span class="cname">{CN[m]}</span><span class="csub">{SUB[m]}</span></div>
          <img src="data:image/png;base64,{b64(os.path.join(FIG, f'best_{m}_1.png'))}"/>
          <div class="stats">
            <div><b style="color:{COL[m]}">{x['pass_rate']*100:.0f}%</b><span>通关率</span></div>
            <div><b>{x['score_mean']:.1f}</b><span>平均分</span></div>
            <div><b>{x['score_max']:.1f}</b><span>最高分</span></div>
            <div><b>{x['cost_mean']:.0f}</b><span>平均造价</span></div>
          </div>
        </div>"""
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{width:1600px;height:1000px;overflow:hidden;background:#f7f9fc;
      font-family:"Noto Sans CJK SC",sans-serif;color:#1b1b1f;padding:34px 48px}}
    .top{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #1b1b1f;padding-bottom:16px}}
    .kick{{font-size:14px;letter-spacing:.22em;color:#1971c2;font-weight:700}}
    h1{{font-family:"Noto Serif CJK SC",serif;font-size:38px;line-height:1.25;margin-top:10px}}
    .sub{{font-size:17px;color:#495057;margin-top:8px}}
    .date{{font-size:14px;color:#868e96;text-align:right;line-height:1.7}}
    .row{{display:flex;gap:24px;margin-top:26px}}
    .card{{flex:1;background:#fff;border:1px solid #e3e6eb;border-radius:12px;padding:16px 18px 12px;box-shadow:0 2px 10px rgba(0,0,0,.04)}}
    .chead{{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}}
    .dot{{width:11px;height:11px;border-radius:50%;display:inline-block}}
    .cname{{font-size:21px;font-weight:700}}
    .csub{{font-size:12.5px;color:#868e96}}
    .card img{{width:100%;height:118px;object-fit:contain;background:#fff;border-radius:6px}}
    .stats{{display:flex;justify-content:space-between;margin-top:8px;border-top:1px solid #edf0f3;padding-top:10px}}
    .stats div{{text-align:center;flex:1}}
    .stats b{{display:block;font-size:20px;font-family:"Noto Serif CJK SC",serif}}
    .stats span{{font-size:12px;color:#868e96}}
    .bottom{{display:flex;gap:26px;margin-top:14px}}
    .panel{{background:#fff;border:1px solid #e3e6eb;border-radius:12px;padding:12px 18px}}
    .panel h2{{font-size:19px;margin-bottom:8px;font-family:"Noto Serif CJK SC",serif}}
    .panel li{{font-size:14px;line-height:1.55;margin-left:18px}}
    .wide{{flex:1.55}} .narrow{{flex:1}}
    .k{{font-weight:700}}
    .play{{margin-top:10px;background:#fff;border:1px solid #e3e6eb;border-radius:12px;padding:10px 18px}}
    .ptitle{{font-family:"Noto Serif CJK SC",serif;font-size:16px;margin-bottom:4px}}
    .pimgs{{display:flex;gap:14px}}
    .pimgs figure{{flex:1;margin:0}}
    .pimgs img{{width:100%;height:92px;object-fit:contain;border-radius:4px;background:#fff}}
    .pimgs figcaption{{font-size:11px;color:#5c677d;text-align:center}}
    .foot{{position:absolute;bottom:12px;left:52px;right:52px;display:flex;justify-content:space-between;
      font-size:12px;color:#adb5bd;border-top:1px solid #e3e6eb;padding-top:8px}}
    </style></head><body>
    <div class="top">
      <div>
        <div class="kick">DSH SUPERAGENT · 造桥模拟器实验</div>
        <h1>三个生成模型，在官方造桥引擎上造桥</h1>
        <div class="sub">arena.meowllm.top/ep9 同源引擎 · 121 维设计空间 · 30,000 个变异设计</div>
      </div>
      <div class="date">2026-11-21<br/>关卡 1「小水沟」<br/>每模型 512 座桥</div>
    </div>
    <div class="row">{cards}</div>
    <div class="bottom">
      <div class="panel wide">
        <h2>三个发现</h2>
        <ul>
          <li><span class="k" style="color:{COL[best_mean]}">{CN[best_mean]} 通关率和平均分最高（{met[best_mean]['pass_rate']*100:.0f}% / {met[best_mean]['score_mean']:.1f}）</span>，
              但 512 个样本里只有 {met[best_mean]['unique']*512:.0f} 个不同设计——VAE 后验塌缩，512 座桥完全相同。</li>
          <li><span class="k" style="color:{COL[best_div]}">{CN[best_div]} 多样性最高（{met[best_div]['diversity']:.2f}）</span>，
              唯一率 {met[best_div]['unique']*100:.0f}%，平均 {met[best_div]['score_mean']:.1f}，质量与多样性最均衡。</li>
          <li><span class="k" style="color:#7048e8">Diffusion 居中</span>：平均 {met['diffusion']['score_mean']:.1f}，
              唯一率 {met['diffusion']['unique']*100:.0f}%，采样只需 0.12 秒。</li>
        </ul>
        <div style="margin-top:8px;font-size:14px;color:#495057">
          手工试玩：关卡 1 从光板桥迭代到 <b>61.47 分</b>；关卡 2 的 30 米跨径直接放大桁架会失败，
          载入参考答案（100 分）和选手 Sol 的悬索桥（137.85 分）才通过。
        </div>
      </div>
      <div class="panel narrow">
        <h2>评测口径</h2>
        <ul>
          <li>官方 C++ 引擎，关卡 1 的 3 个公开场景全部跑完。</li>
          <li>通关率 + 平均分：先过关，再比造价。</li>
          <li>随机基线 {met['random']['pass_rate']*100:.0f}% / 专家数据 {met['expert']['pass_rate']*100:.0f}%。</li>
          <li>不做拒绝采样、不做模拟器引导；同一随机种子。</li>
          <li>模型平均分落在 9 位参赛选手的中下游（Kimi 57.87、DeepSeek 67.01 附近）。</li>
        </ul>
      </div>
    </div>
    <div class="play">
      <div class="ptitle">手工试玩：从光板桥到 61.47 分</div>
      <div class="pimgs">
        <figure><img src="data:image/png;base64,{b64(os.path.join(ROOT,'results/play/bridges/A.png'))}"/><figcaption>光板桥 · 0 分（落水）</figcaption></figure>
        <figure><img src="data:image/png;base64,{b64(os.path.join(ROOT,'results/play/bridges/B.png'))}"/><figcaption>上承式桁架 · 38.5 分</figcaption></figure>
        <figure><img src="data:image/png;base64,{b64(os.path.join(ROOT,'results/play/bridges/C.png'))}"/><figcaption>下承式桁架 · 44.3 分</figcaption></figure>
        <figure><img src="data:image/png;base64,{b64(os.path.join(ROOT,'results/play/bridges/D.png'))}"/><figcaption>逐根删减 · 61.5 分</figcaption></figure>
      </div>
    </div>
    <div class="foot"><span>代码、数据与完整 PDF：github.com/ds4syk/dsh-superagent-playground · deliverables/bridge-arena/</span>
      <span>Slurm · RTX 5090</span></div>
    </body></html>"""
    path = os.path.join(ROOT, "report/one_pager.html")
    open(path, "w", encoding="utf-8").write(html)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r = subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--window-size=1600,1000",
                        f"--screenshot={OUT}", "file://" + path], capture_output=True, text=True)
    print(r.stderr[-200:] if r.returncode else "ok", OUT)


if __name__ == "__main__":
    main()
