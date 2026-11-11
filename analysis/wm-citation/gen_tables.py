import json, csv, re
E=json.load(open('data/entries.json'))
M=["潜空间/结构化世界模型","联合嵌入预测（JEPA）","视频生成式世界模型","3D/占用世界模型","机器人视频-动作/VLA","评测/理论/综述"]

def lineage(s):
    s=s.replace('V-JEPA 2','JV2').replace('V-JEPA','JV').replace('I-JEPA','J')
    s=s.replace('DreamerV3','D3').replace('DreamerV2','D2').replace('Dreamer','D')
    s=s.replace('WorldModels','R').replace('RWM','R').replace('PlaNet','P').replace('JEPA','J')
    s=re.sub(r'(?<![A-Za-z0-9])V3(?![0-9])','D3',s); s=re.sub(r'(?<![A-Za-z0-9])V2(?![0-9])','D2',s)
    parts=[p.strip() for p in s.replace('，',',').split(',') if p.strip()]
    out=[]
    for p in parts:
        if p not in out: out.append(p)
    return '·'.join(out)

def cell(x,maxlen=110):
    return x.replace('|','｜').replace('\n',' ').strip()

def order(e):
    return (-e['cites'], e['year'])

main=[e for e in E if not e.get('appendix')]
app=[e for e in E if e.get('appendix')]
idx=0
lines=[]
for m in M:
    sel=sorted([e for e in main if e['method']==m],key=order)
    if not sel: continue
    lines.append(f"\n### {m}（{len(sel)} 篇）\n")
    lines.append("| # | 论文（年份 · OpenAlex 引用） | 谱系 | 动机 | 贡献 | 方法/技术路线 | 硬件 |")
    lines.append("|-|-|-|-|-|-|-|")
    for e in sel:
        idx+=1
        lines.append(f"| {idx} | **{e['zh']}**（{e['year']} · {e['cites']}） | {lineage(e['seeds'])} | {cell(e['motivation'])} | {cell(e['contribution'])} | {cell(e['approach'])} | {cell(e['hardware'])} |")
open('data/table.md','w').write('\n'.join(lines))

lines=["| 编号 | 论文（年份 · OpenAlex 引用） | 谱系 | 动机 | 贡献 | 方法/技术路线 | 硬件 |","|-|-|-|-|-|-|-|"]
for i,e in enumerate(sorted(app,key=order),1):
    lines.append(f"| A{i} | **{e['zh']}**（{e['year']} · {e['cites']}） | {lineage(e['seeds'])} | {cell(e['motivation'])} | {cell(e['contribution'])} | {cell(e['approach'])} | {cell(e['hardware'])} |")
open('data/table_appendix.md','w').write('\n'.join(lines))

with open('data/table.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f)
    w.writerow(["编号","中文标题","英文标题","年份","OpenAlex引用","发表场所","引用谱系","方法类型","创新点","动机","贡献","方法","硬件","arXiv","DOI"])
    for i,e in enumerate(sorted(E,key=lambda x:(x['method'],-x['cites'])),1):
        w.writerow([i,e['zh'],e['en'],e['year'],e['cites'],e['venue'],lineage(e['seeds']),e['method'],'、'.join(e['innov']),e['motivation'],e['contribution'],e['approach'],e['hardware'],e.get('arxiv',''),e.get('doi','')])
print("main rows",idx,"appendix",len(app))
print(open('data/table.md').read()[:800])
