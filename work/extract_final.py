import pymupdf as fitz, json, re, statistics, sys

PROSE_FONT = re.compile(r'^(Times|Helvetica|Courier|Arial|NimbusRom|NimbusSan)', re.I)
MATH_FONT  = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB|Math)', re.I)
LIG = {"\ufb01":"fi","\ufb02":"fl","\ufb00":"ff","\ufb03":"ffi","\ufb04":"ffl","\ufb05":"ft","\ufb06":"st"}

def span_cls(font, text):
    if MATH_FONT.match(font): return 'm'
    if PROSE_FONT.match(font): return 'p'
    if re.search(r'[A-Za-z]{2}', text): return 'p'
    return 'm'

def block_chars(b):
    """ordered char list with spaces inserted on gaps; returns list of dict(c,cls,bbox,size,font,bold,italic) or ('space',)"""
    chars=[]
    prev=None
    for l in b["lines"]:
        for s in l["spans"]:
            cls=span_cls(s["font"], "".join(c["c"] for c in s["chars"]))
            for c in s["chars"]:
                ch=c["c"]
                for k,v in LIG.items():
                    if k in ch: ch=ch.replace(k,v)
                if prev is not None:
                    cl=prev["cls"]; gap=c["bbox"][0]-prev["bbox"][2]
                    same_line = abs(c["origin"][1]-prev["origin"][1])<=2.0
                    if same_line and gap>max(2.0,0.34*s["size"]) and not prev["c"].isspace():
                        chars.append({"c":" ","cls":'p',"bbox":[prev["bbox"][2],c["bbox"][1],c["bbox"][0],c["bbox"][3]],
                                      "size":s["size"],"font":"space","bold":False,"italic":False,"origin":c["origin"]})
                for ch1 in ch:
                    chars.append({"c":ch1,"cls":cls,"bbox":list(c["bbox"]),"size":round(s["size"],2),
                                  "font":s["font"],"bold":bool(s["flags"]&16),"italic":bool(s["flags"]&2),
                                  "origin":c["origin"]})
                prev=chars[-1] if chars else None
                prev = {"cls":cls,"bbox":c["bbox"],"c":ch,"origin":c["origin"]}
    return chars

def build_blocks(page):
    out=[]
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"]==1:
            out.append({"kind":"image","bbox":list(b["bbox"])}); continue
        chars=block_chars(b)
        if not chars: continue
        pchars=sum(1 for c in chars if c["cls"]=="p")
        mchars=sum(1 for c in chars if c["cls"]=="m")
        # build text with placeholders for math runs
        parts=[]; math=[]; run=None; mi=0
        for c in chars:
            if c["cls"]=="m":
                if run is None: run={"cls":"m","chars":[],"x0":c["bbox"][0],"x1":c["bbox"][2],"y0":c["bbox"][1],"y1":c["bbox"][3]}
                run["chars"].append(c); run["x0"]=min(run["x0"],c["bbox"][0]); run["x1"]=max(run["x1"],c["bbox"][2])
                run["y0"]=min(run["y0"],c["bbox"][1]); run["y1"]=max(run["y1"],c["bbox"][3])
            else:
                if run is not None:
                    parts.append(f"⟦{mi}⟧"); math.append({"i":mi,"bbox":[run["x0"],run["y0"],run["x1"],run["y1"]],
                                                          "text":"".join(x["c"] for x in run["chars"])}); mi+=1; run=None
                parts.append(c["c"])
        if run is not None:
            parts.append(f"⟦{mi}⟧"); math.append({"i":mi,"bbox":[run["x0"],run["y0"],run["x1"],run["y1"]],
                                                  "text":"".join(x["c"] for x in run["chars"])})
        text=re.sub(r"[ \t]+"," ","".join(parts)).strip()
        text=text.replace(" ⟦","⟦").replace("⟧ ","⟧")
        sizes=[c["size"] for c in chars if c["cls"]=="p"] or [c["size"] for c in chars]
        xs=[c["bbox"][0] for c in chars if c["c"]!=" "]
        x1=[c["bbox"][2] for c in chars if c["c"]!=" "]
        out.append({"kind":"math" if (mchars>0 and pchars<=2) else "text",
                    "bbox":list(b["bbox"]),"x0":min(xs),"x1":max(x1),
                    "text":text,"math":math,"size":round(statistics.median(sizes),2),
                    "bold":sum(c["bold"] for c in chars)>len(chars)/2,
                    "italic":sum(c["italic"] for c in chars)>len(chars)/2,
                    "pchars":pchars,"mchars":mchars})
    return out

def page_units(page, pno):
    blocks=build_blocks(page)
    # column assigment
    for b in blocks:
        b["col"]=0 if (b["bbox"][2]<=300 or b.get("x0",b["bbox"][0])<296.5 and b["bbox"][2]<420) else 1
    blocks.sort(key=lambda b:(b["col"], b["bbox"][1], b["bbox"][0]))
    # merge
    units=[]
    cur=None
    for b in blocks:
        if cur is None or b["col"]!=cur["col"] or b["kind"]=="image" or cur["kind"]=="image":
            if cur: units.append(cur)
            cur={"page":pno,"col":b["col"],"kind":b["kind"],"items":[b],"bbox":list(b["bbox"]),
                 "text":b.get("text",""),"math":b.get("math",[]),"size":b.get("size",0),
                 "bold":b.get("bold",False),"italic":b.get("italic",False)}
            continue
        gap=b["bbox"][1]-cur["bbox"][3]
        if b["kind"]=="math":
            # display if clearly separated from surrounding text
            if gap>=3.0:
                units.append(cur); cur={"page":pno,"col":b["col"],"kind":"math","items":[b],"bbox":list(b["bbox"]),
                                        "text":b.get("text",""),"math":[],"size":b.get("size",0),"bold":False,"italic":True}
            else:
                # inline math: attach
                cur["items"].append(b)
                cur["text"]+=f"⟦{len(cur['math'])}⟧"
                m=[dict(m) for m in b["math"]] if b["math"] else [{"i":0,"bbox":b["bbox"],"text":b["text"]}]
                for k,x in enumerate(m):
                    cur["math"].append({"bbox":x["bbox"],"text":x["text"]})
                cur["bbox"][1]=min(cur["bbox"][1],b["bbox"][1]); cur["bbox"][3]=max(cur["bbox"][3],b["bbox"][3])
            continue
        if cur["kind"]=="math" and gap<3.0:
            # text following a display equation? treat as new unit
            units.append(cur); cur={"page":pno,"col":b["col"],"kind":"text","items":[b],"bbox":list(b["bbox"]),
                                    "text":b["text"],"math":b["math"],"size":b["size"],"bold":b["bold"],"italic":b["italic"]}
            continue
        if gap<=1.2 and cur["kind"]=="text" and b["kind"]=="text":
            cur["items"].append(b)
            if b["text"]:
                cur["text"]=(cur["text"]+" "+b["text"]).strip()
            for m in b["math"]:
                cur["math"].append({"bbox":m["bbox"],"text":m["text"]})
            cur["bbox"][1]=min(cur["bbox"][1],b["bbox"][1]); cur["bbox"][3]=max(cur["bbox"][3],b["bbox"][3])
            cur["bbox"][0]=min(cur["bbox"][0],b["bbox"][0]); cur["bbox"][2]=max(cur["bbox"][2],b["bbox"][2])
        else:
            units.append(cur); cur={"page":pno,"col":b["col"],"kind":b["kind"],"items":[b],"bbox":list(b["bbox"]),
                                    "text":b.get("text",""),"math":b.get("math",[]),"size":b.get("size",0),
                                    "bold":b.get("bold",False),"italic":b.get("italic",False)}
    if cur: units.append(cur)
    # dedupe math placeholder indexes in merged text: renumber sequentially
    for u in units:
        if u["kind"]!="text": continue
        # fix placeholders count vs math list
        idx=len(u["math"])
        u["text"]=re.sub(r"⟦\d+⟧", lambda m, cnt=[0]: f"⟦{cnt[0]:d}⟧" if not cnt.__setitem__(0,cnt[0]+1) else "", u["text"])
    for k,u in enumerate(units):
        u["id"]=f"p{pno}_{k}"
        u["bbox"]=[round(v,1) for v in u["bbox"]]
    return units

if __name__=="__main__":
    doc=fitz.open("source/original.pdf")
    pages=[]
    for pno in range(doc.page_count):
        units=page_units(doc[pno], pno+1)
        pages.append({"page":pno+1,"units":units})
    json.dump(pages, open("analysis/units.json","w"), ensure_ascii=False, indent=1)
    print("units per page:", [len(p["units"]) for p in pages])
    print("total chars:", sum(len(u.get("text","")) for p in pages for u in p["units"]))
