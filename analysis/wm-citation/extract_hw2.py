import json, os, re
ROOT=os.path.dirname(os.path.abspath(__file__))
texts=json.load(open(os.path.join(ROOT,'raw/texts.json')))
sel=json.load(open(os.path.join(ROOT,'raw/selected.json')))
byid={w['id']:w for w in sel}
HW=r'(A100|A800|H100|H200|H800|V100|TPU[\s-]?v?\d|GPU|GPUs|RTX\s?\d{4}|\b4090\b|\b3090\b|L40S?|MI300|Ascend)'
TRAIN=r'(train|trained|training|compute|hour|day|week|step|batch|inference|FPS|throughput|node)'
out={}
for wid,path in texts.items():
    w=byid[wid]
    txt=open(path,encoding='utf-8',errors='ignore').read()
    txt=re.sub(r'\s+',' ',txt)
    sents=re.split(r'(?<=[.!?])\s+',txt)
    scored=[]
    for i,s in enumerate(sents):
        if len(s)>500: s=s[:500]
        hw=len(re.findall(HW,s,re.I)); tr=len(re.findall(TRAIN,s,re.I))
        if hw>=1 and tr>=1 and not re.search(r'(figure|table)\s*\d',s,re.I):
            scored.append((hw*2+min(tr,3),i,s))
    scored.sort(key=lambda x:(-x[0],x[1]))
    ctxs=[]; used=set()
    for sc,i,s in scored:
        if any(abs(i-j)<2 for j in used): continue
        used.add(i); ctxs.append(s)
        if len(ctxs)>=8: break
    out[wid]={'title':w['title'],'ctx':ctxs}
json.dump(out,open(os.path.join(ROOT,'raw/hardware_ctx2.json'),'w'))
for wid,d in out.items():
    print('\n###',d['title'][:85])
    for c in d['ctx'][:4]: print('  -',c[:260])
