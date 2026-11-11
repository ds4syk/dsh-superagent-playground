import json, os
ROOT=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(ROOT,'../..'))
E=json.load(open(os.path.join(ROOT,'data/entries.json')))
M=["潜空间/结构化世界模型","联合嵌入预测（JEPA）","视频生成式世界模型","3D/占用世界模型","机器人视频-动作/VLA","评测/理论/综述"]
num={}; i=0
main=[]; app=[]
for m in M:
    for e in sorted([x for x in E if not x.get('appendix') and x['method']==m],key=lambda x:(-x['cites'],x['year'])):
        i+=1; e['num']=i; main.append(e)
for e in sorted([x for x in E if x.get('appendix')],key=lambda x:-x['cites']):
    app.append(e)
head='''# 08 参考文献

> **结论先行**：本页列出主表 46 篇、附录 4 篇与 9 篇种子论文。编号与 02 一致；arXiv 链接优先，正式出版信息附后。引用数为 OpenAlex 2026-11-11 抓取。

## 1. 种子论文（9 篇）

| 谱系 | 论文 | 年份 | OpenAlex 引用 | 链接 |
|-|-|-|-|-|
| Dreamer | Dream to Control: Learning Behaviors by Latent Imagination | 2020 | 1496 | [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) |
| Dreamer | Mastering Atari with Discrete World Models | 2021 | 822 | [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) |
| Dreamer | Mastering diverse control tasks through world models | 2025 | 1946 | [Nature](https://doi.org/10.1038/s41586-025-08744-2) · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| JEPA | A Path Towards Autonomous Machine Intelligence | 2022 | — | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) |
| JEPA | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | 2023 | 791 | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| JEPA | Revisiting Feature Prediction for Learning Visual Representations from Video | 2024 | 508 | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| JEPA | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | 2025 | 1112 | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| RWM | World Models | 2018 | 2203 | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) |
| RWM | Recurrent World Models Facilitate Policy Evolution | 2018 | 585 | [NeurIPS](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html) |

## 2. 主表论文（46 篇）
'''
lines=[head]
for m in M:
    lines.append(f"\n### {m}\n")
    for e in main:
        if e['method']!=m: continue
        ar=f"[arXiv:{e['arxix']}]" if False else (f"[arXiv:{e['arxiv']}](https://arxiv.org/abs/{e['arxiv']})" if e.get('arxiv') else "")
        lines.append(f"{e['num']}. **{e['zh']}**（{e['en']}）。{e['venue']}，{e['year']}。{ar} · OpenAlex 引用 {e['cites']}。")
lines.append("\n## 3. 附录 A：2023 年上半年奠基工作（4 篇）\n")
for j,e in enumerate(app,1):
    ar=f"[arXiv:{e['arxiv']}](https://arxiv.org/abs/{e['arxiv']})" if e.get('arxiv') else ""
    lines.append(f"A{j}. **{e['zh']}**（{e['en']}）。{e['venue']}，{e['year']}。{ar} · OpenAlex 引用 {e['cites']}。")
open(os.path.join(REPO,'kb/wm-citation/08.md'),'w').write("\n".join(lines))
print("ok", len(main), len(app))
