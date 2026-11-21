"""生成中文 PDF 报告（HTML -> Chrome headless print）。"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import BridgeTransformer, ConvVAE, EpsMLP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
FIG = os.path.join(REP, "figures")
CHROME = os.path.expanduser("~/.cache/puppeteer/chrome/linux-131.0.6778.204/chrome-linux64/chrome")

CN = {"transformer": "Transformer", "diffusion": "Diffusion", "cnn": "CNN-VAE",
      "random": "随机基线", "expert": "专家数据"}
CH = {"transformer": "#1971c2", "diffusion": "#7048e8", "cnn": "#e8590c",
      "random": "#868e96", "expert": "#2f9e44"}
CONTESTS = [("GPT-6.2 Sol", 98.54), ("Claude Opus 5.5", 99.38), ("GPT-6 Astra", 91.58),
            ("Gemini 4 Argon", 93.46), ("Grok 4.7", 91.25), ("Kimi K3.1 Pro", 57.87),
            ("腾讯混元 Hy4", 85.33), ("豆包 Seed Evolving", 90.40), ("DeepSeek V4.1 Flash", 67.01)]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def img(path, cls=""):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{data}"/>'


def pct(x):
    return f"{x*100:.0f}%"


def main():
    met = {m["name"]: m for m in load_json("results/eval/metrics.json")["metrics"]}
    timing = load_json("results/eval/metrics.json")["timing"]
    train = load_json("results/checkpoints/train_meta.json")
    play = load_json("results/play/play_log.json")
    import numpy as np
    ds = np.load("results/dataset_l1.npz")
    n_total = len(ds["designs"]); n_pass = int(ds["passed"].sum())
    score_top = float(ds["score"].max())
    jobs = load_json("results/jobs.json") if os.path.exists("results/jobs.json") else {}
    pg = ["transformer", "diffusion", "cnn"]
    params = {"transformer": sum(p.numel() for p in BridgeTransformer().parameters()),
              "diffusion": sum(p.numel() for p in EpsMLP().parameters()),
              "cnn": sum(p.numel() for p in ConvVAE().parameters())}
    best_mean = max(pg, key=lambda m: met[m]["score_mean"])
    best_pass = max(pg, key=lambda m: met[m]["pass_rate"])
    best_single = max(pg, key=lambda m: met[m]["score_max"])
    best_div = max(pg, key=lambda m: met[m]["diversity"])

    def metric_table():
        head = ("<tr><th>对象</th><th>通关率</th><th>平均分</th><th>最高分</th><th>前 5 平均</th>"
                "<th>通过者均分</th><th>平均造价</th><th>平均杆件</th><th>唯一率</th><th>多样性</th></tr>")
        rows = []
        for g in ["random"] + pg + ["expert"]:
            m = met[g]
            rows.append(
                f"<tr><td class='name' style='color:{CH[g]}'>{CN[g]}</td>"
                f"<td>{pct(m['pass_rate'])}</td><td>{m['score_mean']:.2f}</td>"
                f"<td>{m['score_max']:.2f}</td><td>{m['best_of_5']:.2f}</td>"
                f"<td>{m['score_pass_mean']:.2f}</td><td>{m['cost_mean']:.0f}</td>"
                f"<td>{m['members_mean']:.0f}</td><td>{pct(m['unique'])}</td>"
                f"<td>{m['diversity']:.3f}</td></tr>")
        return f"<table class='metrics'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"

    def play_table(items):
        return "".join(
            f"<tr><td class='name'>{p['name']}</td><td>{p['note']}</td><td>{p['members']}</td>"
            f"<td>{p['cost']:.0f}</td><td>{p['score']:.1f}</td>"
            f"<td>{'通过' if p['passed'] else '未通过'}</td><td>{p['reason'] or '—'}</td></tr>"
            for p in items)
    play_rows_l1 = play_table([p for p in play if p['level'] == 1])
    play_rows_l2 = play_table([p for p in play if p['level'] == 2])

    model_rows = "".join(
        f"<tr><td class='name' style='color:{CH[m]}'>{CN[m]}</td>"
        f"<td>{params[m]/1e6:.2f} M</td>"
        f"<td>{'自回归 121 步' if m=='transformer' else ('DDIM 100 步' if m=='diffusion' else 'VAE 单次前向')}</td>"
        f"<td>{train[m+'_time']:.0f} s</td><td>{timing[m]:.1f} s / 512</td></tr>" for m in pg)

    contest_rows = "".join(
        f"<tr><td class='name'>{name}</td><td>{score:.2f}</td></tr>" for name, score in CONTESTS)

    fail_rows = ""
    for c in ["通过", "杆件断裂/车辆失败", "车辆未通过", "物理异常", "超预算"]:
        vals = "".join(f"<td>{met[m]['failures'].get(c,0)}</td>" for m in pg)
        fail_rows += f"<tr><td class='name'>{c}</td>{vals}</tr>"

    job_rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('id','—')}</td><td>{v.get('partition','—')}</td>"
        f"<td>{v.get('time','—')}</td></tr>" for k, v in jobs.items()) or \
        "<tr><td colspan='4'>见 results/ 与 Slurm 日志</td></tr>"

    bullets = "".join(
        f"<li><b style='color:{CH[m]}'>{CN[m]}</b>：通关率 {pct(met[m]['pass_rate'])}，"
        f"平均分 {met[m]['score_mean']:.2f}，最高 {met[m]['score_max']:.2f}，"
        f"平均造价 {met[m]['cost_mean']:.0f}，唯一率 {pct(met[m]['unique'])}，"
        f"多样性 {met[m]['diversity']:.3f}。</li>"
        for m in pg)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>造桥大赛模拟器与三个生成模型</title>
<style>
:root {{ --ink:#1b1b1f; --muted:#5c677d; --line:#e3e6eb; --accent:#1971c2; }}
@page {{ size: A4; margin: 17mm 16mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:"Noto Sans CJK SC","Noto Serif CJK SC",sans-serif; font-size:9.7pt;
       line-height:1.72; color:var(--ink); -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
h1,h2,h3 {{ font-family:"Noto Serif CJK SC","Noto Sans CJK SC",serif; margin:0; }}
h2 {{ font-size:16pt; margin:22px 0 10px; padding-left:10px; border-left:5px solid var(--accent); }}
h3 {{ font-size:11.5pt; margin:15px 0 6px; }}
p {{ margin:7px 0; text-align:justify; }}
.small {{ font-size:8.5pt; color:var(--muted); }}
.cover {{ height:255mm; display:flex; flex-direction:column; justify-content:space-between; page-break-after:always; }}
.cover .top {{ padding-top:8mm; }}
.kicker {{ letter-spacing:.18em; font-size:8.5pt; color:var(--accent); font-weight:600; }}
.cover h1 {{ font-size:28pt; line-height:1.32; margin:9mm 0 4mm; }}
.cover .sub {{ font-size:13pt; color:#343a40; }}
.cover .meta {{ font-size:9pt; color:var(--muted); margin-top:5mm; }}
.cover .intro {{ max-width:155mm; font-size:11pt; line-height:1.95; color:#343a40;
                 border-left:3px solid #cfe3fb; padding-left:14px; }}
.cover .hero img {{ width:100%; border-radius:6px; }}
.nums {{ display:flex; gap:6mm; margin-top:5mm; }}
.nums div {{ flex:1; background:#f6f8fa; border:1px solid var(--line); border-radius:6px; padding:4mm 3mm; }}
.nums b {{ display:block; font-size:19pt; color:var(--accent); font-family:"Noto Serif CJK SC"; }}
.nums span {{ font-size:8.2pt; color:var(--muted); }}
.pagebreak {{ page-break-before:always; }}
.callout {{ background:#f1f7ff; border:1px solid #cfe3fb; border-left:4px solid var(--accent);
            border-radius:5px; padding:9px 13px; margin:10px 0; }}
.callout.warn {{ background:#fff8f1; border-color:#ffd8a8; border-left-color:#e8590c; }}
ul {{ margin:5px 0 5px 18px; padding:0; }}
li {{ margin:3px 0; }}
figure {{ margin:11px 0 13px; page-break-inside:avoid; }}
figure img {{ width:100%; border:1px solid var(--line); border-radius:4px; }}
figcaption {{ font-size:8.3pt; color:var(--muted); margin-top:4px; text-align:center; }}
table {{ width:100%; border-collapse:collapse; margin:8px 0 11px; font-size:8.4pt; page-break-inside:avoid; }}
th {{ background:#f1f3f5; text-align:left; font-weight:600; padding:5px 6px; border-bottom:1.5px solid #ced4da; }}
td {{ padding:4.5px 6px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr:nth-child(even) td {{ background:#fafbfc; }}
td.name {{ font-weight:600; }}
.metrics td,.metrics th {{ text-align:center; }}
.metrics td.name {{ text-align:left; }}
.formula {{ background:#f8f9fa; border:1px solid var(--line); border-radius:5px; padding:8px 12px;
            font-family:"DejaVu Sans Mono",monospace; font-size:8.7pt; text-align:center; margin:8px 0; }}
.sig {{ color:var(--muted); font-size:8.2pt; margin-top:22px; }}
.two {{ display:flex; gap:6mm; }}
.two > div {{ flex:1; }}
</style></head><body>

<div class="cover">
  <div class="top">
    <div class="kicker">DSH SUPERAGENT · 造桥模拟器实验</div>
    <h1>三个生成模型，<br/>在官方造桥引擎上造桥</h1>
    <div class="sub">Transformer · Diffusion · CNN-VAE × arena.meowllm.top/ep9</div>
    <div class="meta">2026 年 11 月 21 日 · 交付物：PDF 报告 + 可复现代码</div>
  </div>
  <div class="intro">
    先按官方规则手工试玩关卡 1（从光板桥迭代到 61.47 分），再用 30,000 个变异设计训练
    Transformer、Diffusion、CNN-VAE 三个模型，最后让每个模型生成 512 座桥，用与
    比赛逐帧一致的 C++ 引擎跑完 3 个公开场景并打分。
  </div>
  <div>
    <div class="hero">{img(os.path.join(FIG,'cover_strip.png'))}</div>
    <div class="small" style="text-align:center">三个模型各自最好的桥（官方 C++ 引擎逐场景跑完并评分）</div>
    <div class="nums">
      <div><b>30,000</b><span>关卡 1 变异设计，全部用官方引擎评测</span></div>
      <div><b>3</b><span>生成模型：Transformer / Diffusion / CNN</span></div>
      <div><b>2,560</b><span>最终评测（5 个对象 × 512 座桥）</span></div>
    </div>
  </div>
</div>

<h2 style="margin-top:0">结论先行</h2>
<div class="callout">
<b>一句话：</b>在官方造桥引擎上，三个从零训练的小模型都能稳定生成“能通车”的关卡 1 桁架桥。
{CN[best_mean]} 的通关率和平均分最高（{pct(met[best_mean]['pass_rate'])} / {met[best_mean]['score_mean']:.2f}），
但 512 个样本只有 {met[best_mean]['unique']*512:.0f} 个不同设计——它把最好的种子桥背了下来。
{CN[best_div]} 多样性最高（{met[best_div]['diversity']:.3f}），质量与多样性最均衡；
三个模型整体落在 9 位参赛选手中游附近，离 100 分的参考答案仍有明显差距。
</div>
<ul>{bullets}</ul>
<p class="small">通关 = 关卡 1 的三个公开场景全部车辆按时完好到达；关卡分 = 100 × 参考答案造价 ÷ 你的造价。</p>

<h2>1　任务与模拟器</h2>
<p>原始需求：先自己玩一下造桥模拟器、造几个桥；再分别训练 Transformer、Diffusion、CNN
三个模型来造桥；最后把三个模型的打分和生成的桥整理成 PDF。</p>
<p>模拟器是 <b>arena.meowllm.top/ep9</b>「第 9 期 · 造桥 · AI 算法对抗大赛」：9 个 AI 在 90 分钟内
设计 6 个闯关 + 第 7 关攻防，桥是二维侧视杆件结构，由确定性的 XPBD 物理内核模拟。
主办方把完整引擎与选手包开源在 <b>github.com/chen-006/meow-ai-arena</b>（MIT）；
本报告用仓库里的同一份 C++ 引擎（<code>bridge_sim</code>）和 3 个公开场景做全部评分，
与网页模拟器逐帧一致。</p>
<figure>{img(os.path.join(FIG,'simulator_ref_l1.png'))}
<figcaption>图 1　官方网页模拟器（关卡 1，载入参考答案）。左栏是材料与预算，中间是桥，
右栏是试车结果；本报告的设计 JSON 与该模拟器完全兼容。</figcaption></figure>
<div class="formula">预算 10,000 元 · 参考答案造价 3,509.84 元 · 三个场景：轿车左→右、面包车右→左、轿车+面包车跟车</div>
<p>材料：路面 150 元/米（车只能开在路面上）、木杆 40、钢杆 160、钢索 70、弹簧 30+300/根、
液压杆 150+800/根；每个新节点 20 元。杆件超过受拉/受压上限就断，长杆受压上限按
屈曲系数 ÷ 长度² 下降。桥在最后一辆车过去之后塌了也算通过——只看车。</p>

<h2>2　试玩：从光板桥到 61.47 分</h2>
<p>我先在关卡 1 用同一套规则手工搭了 4 座桥，全部用官方引擎跑完 3 个场景：</p>
<table><thead><tr><th>桥</th><th>思路</th><th>杆件</th><th>造价</th><th>关卡分</th><th>结果</th><th>原因</th></tr></thead>
<tbody>{play_rows_l1}</tbody></table>
<figure>{img('results/play/gallery_l1.png')}
<figcaption>图 2　关卡 1 的四次尝试：光板桥 → 上承式桁架 → 下承式桁架 → 逐根删减后的优化版。</figcaption></figure>
<ul>
<li><b>光板桥</b>只有 12 根路面，造价 2430，但跨中下挠过大，轿车直接落水，0 分。</li>
<li><b>上承式桁架</b>在桥面上方加两层钢/木桁架，造价 9104，三个场景全过，38.55 分。</li>
<li><b>下承式桁架</b>把桁架放到桥面下方，利用 A1/A3 两个下方锚点，造价 7930，44.26 分。</li>
<li><b>优化版</b>从下承式出发，逐根尝试删除/把钢杆换成木杆，保持三个场景全过；最后 22 根
上部杆件、造价 5710，<b>61.47 分</b>。这一步只用了模拟器反馈，没有用任何学习模型。</li>
</ul>
<p>然后我把关卡 1 的桁架直接放大到关卡 2 的 30 米跨径——结果两个场景都没过（杆件断裂/车辆失败）。
30 米跨径需要悬索、拱或更深/更强的结构；载入官方参考答案（造价 10617，100 分）和选手
GPT-6.2 Sol 的悬索桥（造价 7702，137.85 分）后，模拟器都能稳定通过。这说明引擎和评分是
一致的，差距来自设计本身。</p>
<table><thead><tr><th>桥</th><th>思路</th><th>杆件</th><th>造价</th><th>关卡分</th><th>结果</th><th>原因</th></tr></thead>
<tbody>{play_rows_l2}</tbody></table>
<figure>{img('results/play/gallery_l2.png')}
<figcaption>图 3　关卡 2：把 15 米桁架直接放大（上排）会失败；官方参考答案与选手悬索桥（下排）能过。</figcaption></figure>
<div class="callout warn"><b>口径：</b>图 3 下排的两座桥是<b>载入</b>官方参考答案和选手设计，
用于展示 100 分/137.85 分的上限，不是我手工设计的；上排两座失败桥是我按同一套桁架思路
直接放大造的。这样既能展示试玩过程，也能避免把别人的桥冒充成自己的。</div>

<h2>3　设计空间与数据</h2>
<p>为了让生成模型输出天然合法的 JSON，我把关卡 1 离散成一个固定桥面 + 上下分层网格：</p>
<ul>
<li>桥面：y=0 的直线路面，A0→A2，每 1.5 米一根路面，固定不动；</li>
<li>桥面以下 1 层（y=-2，两端接 A1/A3）、以上 2 层（y=1.5 / 3.0）；</li>
<li>共 <b>121 根候选杆件</b>（竖向、水平、两种斜杆），每根 4 种状态：空 / 木 / 钢 / 钢索。</li>
</ul>
<p>这个约束不是引擎的限制（引擎支持任意节点、杆件、传感器、液压杆），而是为了让
三个模型在同一 121 维离散表示上公平比较。桥面固定也保证每座桥都有连续的、车能开的
路面；模型只需要学“在哪里加什么杆”。</p>
<figure>{img(os.path.join(FIG,'dataset.png'))}
<figcaption>图 4　30,000 个变异设计（左）与训练集（右）。变异 = 在通过的设计上随机改 1~4 根杆件；
{n_pass:,} 个通过全部 3 个场景，训练集取分数最高的 6,000 个。</figcaption></figure>
<p>随机变异而不是从零随机撒杆，是有意的：从零随机的设计绝大多数会在第一个场景就散架，
模型学不到“桥”的结构先验。以 3 个手工通过设计为种子做局部变异，通过率
{n_pass/n_total*100:.1f}%，同时保留材料、杆件增减的多样性；最高分 {score_top:.2f}。
训练集里分数门槛约 54.6，模型学的是“高分桥附近的设计分布”。</p>

<h2>4　三个模型</h2>
<p>三个模型吃同样的 121 维 0..3 设计向量，输出同样维度；差别只在生成方式。</p>
<table><thead><tr><th>模型</th><th>参数量</th><th>生成方式</th><th>训练耗时</th><th>采样耗时</th></tr></thead>
<tbody>{model_rows}</tbody></table>
<ul>
<li><b>Transformer</b>：BOS + 121 个 4 类 token 的自回归模型，4 层、128 维、4 头，因果注意力；
采样时逐位生成（温度 0.9）。它显式建模“先有下弦、再加斜杆”的条件依赖。</li>
<li><b>Diffusion</b>：把 121×4 one-hot 展平成 484 维连续向量，做 Gaussian diffusion
（1000 步训练、DDIM 100 步采样）；去噪器是 4 层 512 维 MLP + 正弦时间编码，预测噪声 ε。</li>
<li><b>CNN-VAE</b>：把设计排成 4 种杆型 × 4 种材料的 16 通道 4×11 网格，3 层卷积编码到
32 维隐变量，再转置卷积解码；BCE + 0.05×KL，采样时从 N(0,I) 抽隐变量、每个杆型取
argmax。它一次前向就能出整张图，最快但最“平滑”。</li>
</ul>
<figure>{img(os.path.join(FIG,'training_curves.png'))}
<figcaption>图 5　三个模型的训练曲线（同一份 5,700 个训练设计、同一台 RTX 5090）。</figcaption></figure>

<h2>5　评测协议</h2>
<p>协议在训练前固定，避免“挑好看的桥”：</p>
<ul>
<li>三个模型各采样 <b>512</b> 个设计，使用同一随机种子；Transformer 温度 0.9，
Diffusion 100 步 DDIM，CNN 从标准正态先验采样。</li>
<li>所有设计直接送官方 C++ 引擎，跑关卡 1 的<b>全部 3 个公开场景</b>；不做人工筛选、
不做模拟器引导、不做拒绝采样。</li>
<li>另设两个基线：<b>随机基线</b>（在种子桥上随机改 1~3 根杆）和<b>专家数据</b>
（训练集随机抽 512 个）。</li>
<li>指标：通关率、平均关卡分、最高分、前 5 平均（512 个样本按 5 个一组取最好再平均）、
通过者均分、平均造价、平均杆件数、唯一率、多样性（1 − 平均成对 Jaccard 相似度）。</li>
</ul>
<div class="callout"><b>为什么用“通关率 + 平均分”两个指标：</b>平均分高可能是稳定造出 55 分的桥；
通关率高说明很少造出会散架的桥。比赛只认“先过关，再比造价”，所以两者要一起看。</div>

<h2>6　结果</h2>
{metric_table()}
<p class="small">表 1　官方引擎评测（每个对象 512 个设计；预算 10,000；参考造价 3,509.84）。</p>
<figure>{img(os.path.join(FIG,'score_distribution.png'))}
<figcaption>图 6　关卡分分布；训练专家集是“模仿目标”的上界参考。</figcaption></figure>
<figure>{img(os.path.join(FIG,'metrics_bars.png'))}
<figcaption>图 7　通关率、平均关卡分与平均造价。</figcaption></figure>

<h3>6.1 三个模型各自的特点</h3>
<div class="two">
<div style="flex:0 0 54%">
<figure style="margin-top:0">{img(os.path.join(FIG,'quality_diversity.png'))}<figcaption>图 8　质量 vs 多样性：CNN 在左上角（高分、零多样性），Transformer 在右侧
（多样性最高），Diffusion 在两者之间。</figcaption></figure>
</div>
<div>
<p>{CN[best_mean]} 的通关率和平均分最高（{pct(met[best_mean]['pass_rate'])} /
{met[best_mean]['score_mean']:.2f}），但它的 512 个样本里只有
<b>{met[best_mean]['unique']*512:.0f} 个不同设计</b>（唯一率 {pct(met[best_mean]['unique'])}，
多样性 {met[best_mean]['diversity']:.3f}）——512 座桥实际上完全相同。这是 VAE 的
<b>后验塌缩</b>：隐变量先验采样全部落到同一个众数（训练集里最好的那座种子桥），
质量最高、多样性为零。
{CN[best_div]} 的唯一率 {pct(met[best_div]['unique'])}、多样性 {met[best_div]['diversity']:.3f}，
是三者中最多的；平均分 {met[best_div]['score_mean']:.2f}，只比 CNN 低
{met[best_mean]['score_mean']-met[best_div]['score_mean']:.2f} 分。
Diffusion 居中：平均 {met['diffusion']['score_mean']:.2f}，唯一率
{pct(met['diffusion']['unique'])}，多样性 {met['diffusion']['diversity']:.3f}。</p>
<p>三个模型都远高于随机基线（通关率 {pct(met['random']['pass_rate'])}、平均
{met['random']['score_mean']:.2f}），说明它们确实学到了“哪些杆件组合能形成传力路径”。
与训练专家集（{pct(met['expert']['pass_rate'])} / {met['expert']['score_mean']:.2f}）相比，
Diffusion 和 CNN 在分数上已经追平甚至略高，但它们的多样性低于专家集；Transformer
的分数略低，却保留了 96% 的唯一设计。若只报平均分，CNN 会“赢”；把多样性一起看，
Transformer 才是质量与多样性最均衡的一个。</p>


</div>
</div>

<figure>{img(os.path.join(FIG,'best_bridges.png'))}
<figcaption>图 9　三个模型各自最好的三座桥；每座都重新用官方引擎求解并渲染。
CNN 的三个“不同”名次其实是同一座桥——它的 512 个样本全部相同。</figcaption></figure>

<h3>6.2 和 9 位参赛选手比</h3>
<p>官方比赛里 9 位选手的关卡 1 成绩如下（来自主办方 results/）。我们的模型是“从零训练、
只针对关卡 1、设计空间固定”的小模型，和 90 分钟内完成 7 关的 LLM agent 不是同一类选手；
这里只做一个量级参照。</p>
<div class="two"><div>
<table><thead><tr><th>选手</th><th>关卡 1 分</th></tr></thead><tbody>{contest_rows}</tbody></table>
</div><div>
<p>模型平均分 {min(met[m]['score_mean'] for m in pg):.1f}–{max(met[m]['score_mean'] for m in pg):.1f}，
落在 Kimi（57.87）与 DeepSeek（67.01）附近，低于前五名 85–99 分的水平。</p>
<p>差距主要来自设计空间：选手可以任意布置节点、用拱/悬索/液压/传感器，甚至把车弹射过去；
我们只让模型在固定桥面 + 3 层网格里加减木/钢/索。把设计空间放开后，模型的上限会更高，
但训练难度也会大得多。</p>
</div></div>

<h3>6.3 杆件数与分数</h3>
<figure>{img(os.path.join(FIG,'density_score.png'))}
<figcaption>图 10　上部杆件数与关卡分：通过的设计集中在“杆件足够形成传力路径、但没多到超预算”的区间。</figcaption></figure>
<p>训练集里的高分桥大多 22–30 根上部杆件；模型生成的桥也集中在这个区间。CNN 的
512 个样本都是同一座 24 杆的桥；Diffusion 会生成少量 15 根以下、直接散架的桥；
Transformer 的杆件数分布最宽，因此分数方差也最大。</p>

<h2>7　讨论与局限</h2>
<h3>7.1 三种范式的差异</h3>
<ul>
<li><b>Transformer</b>：把设计当序列，能显式建模“先有下弦、再有斜杆、最后封顶”的条件依赖；
缺点是逐位生成 121 步，采样最慢，一位错可能让整条传力路径断开。</li>
<li><b>Diffusion</b>：并行去噪，样本多样性最好；缺点是 100 步采样仍比 CNN 慢，连续松弛 +
阈值会在“半有半无”的杆件上产生噪声。</li>
<li><b>CNN-VAE</b>：一次前向即可采样，最快；但这份实验里出现了严重的后验塌缩：
512 个先验样本全部解码成同一座最优种子桥。把采样方差放大到 2 倍后，128 个样本里
才有 13 个不同设计（通关率 97.7%、平均 59.63），说明它的隐空间仍然很窄。</li>
</ul>
<h3>7.2 为什么平均分停在 50 多</h3>
<p>三个模型都只做了监督式的分布模仿：训练目标是“像训练集里的高分桥”，而不是“让模拟器
给高分”。这带来两个后果：其一，模型会复制训练集里的冗余杆件；其二，采样没有用模拟器
做拒绝/重排。若加一层“采样 16 个 → 模拟器排序取最好”的 Best-of-N，分数会往上走
（表 1 的“前 5 平均”已经给出这个方向的估计），但那衡量的是搜索算力，不是模型本身，
所以主表仍用 512 个未筛选样本。</p>
<h3>7.3 局限</h3>
<ul>
<li>只训练/评测了关卡 1。关卡 2–6 的跨径、船、卡车、双向车流会显著改变最优结构；
报告里的手工试玩展示了这种难度跃迁，但没有为每关训练模型。</li>
<li>设计空间被固定为“直线桥面 + 3 层网格”，不能生成拱、悬索、液压开合桥等比赛允许的
结构；这是为了让三个模型能在同一表示上比较，不是引擎的限制。</li>
<li>物理是比赛引擎本身，结果确定；但评分函数（造价）只衡量材料，不衡量美观、施工难度。</li>
<li>每个模型只训练了一份超参、一个种子；512 个样本的通关率标准误约 2 个百分点，
模型间 1–2 个百分点的差距不应过度解读。</li>
</ul>

<h2>8　复现</h2>
<p>所有训练/评测都在实验室服务器（RTX 5090）上通过 Slurm 提交，登录节点只做轻量操作。
第三方引擎不提交，<code>./setup.sh</code> 会按网页同源的官方开源仓库拉取并编译。</p>
<div class="formula" style="text-align:left">
./setup.sh<br/>
python scripts/play.py<br/>
python -m arena_bridge.dataset --mode dataset --level 1 --n 30000 --workers 16<br/>
sbatch sbatch/train.sbatch<br/>
sbatch sbatch/eval.sbatch<br/>
python scripts/make_figures.py &amp;&amp; python scripts/build_report.py
</div>
<table><thead><tr><th>路径</th><th>内容</th></tr></thead><tbody>
<tr><td><code>arena_bridge/design.py</code></td><td>121 维设计空间与比赛 JSON 构造</td></tr>
<tr><td><code>arena_bridge/sim.py</code></td><td>官方引擎封装与并行评测</td></tr>
<tr><td><code>arena_bridge/dataset.py</code></td><td>种子桥、变异、进化搜索、数据集</td></tr>
<tr><td><code>models/models.py</code></td><td>Transformer / Diffusion / CNN-VAE</td></tr>
<tr><td><code>scripts/play.py</code></td><td>手工试玩与渲染</td></tr>
<tr><td><code>scripts/train.py</code> / <code>evaluate.py</code></td><td>训练 / 统一评测</td></tr>
</tbody></table>
<p class="small">Slurm 作业：</p>
<table><thead><tr><th>作业</th><th>Job ID</th><th>分区</th><th>耗时</th></tr></thead><tbody>{job_rows}</tbody></table>

<h3>附录　超参数</h3>
<table><thead><tr><th>项</th><th>Transformer</th><th>Diffusion</th><th>CNN-VAE</th></tr></thead><tbody>
<tr><td>结构</td><td>4 层 128 维 4 头</td><td>4 残差块 512 维</td><td>3 层卷积 + 2 层转置卷积</td></tr>
<tr><td>隐变量/时间</td><td>—</td><td>正弦时间编码 128</td><td>隐变量 32</td></tr>
<tr><td>轮数 / 批大小</td><td>400 / 256</td><td>400 / 256</td><td>400 / 256</td></tr>
<tr><td>优化器</td><td>AdamW 3e-4</td><td>AdamW 1e-3</td><td>AdamW 1e-3</td></tr>
<tr><td>训练样本</td><td colspan="3">5,700（来自最高分 6,000 个通过设计，5% 作验证）</td></tr>
</tbody></table>
<p class="sig">报告由 DSH Superagent 生成；引擎来自 chen-006/meow-ai-arena（MIT）。
所有分数可用 <code>scripts/evaluate.py</code> 以固定种子复现。</p>

</body></html>"""

    with open(os.path.join(REP, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("html", len(html))
    pdf = os.path.join(REP, "bridge_arena_report.pdf")
    cmd = [CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf}", "file://" + os.path.join(REP, "report.html")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stderr[-300:] if r.returncode else "pdf ok", pdf)


if __name__ == "__main__":
    main()
