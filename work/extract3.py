import pymupdf as fitz, json, re, statistics, sys
from collections import defaultdict

PROSE_FONT = re.compile(r'^(Times|Helvetica|Courier|Arial|NimbusRom|NimbusSan)', re.I)
MATH_FONT  = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB|Math)', re.I)
LIG = {"\ufb01":"fi","\ufb02":"fl","\ufb00":"ff","\ufb03":"ffi","\ufb04":"ffl","\ufb05":"ft","\ufb06":"st"}

def span_cls(font, text):
    if MATH_FONT.match(font): return 'm'
    if PROSE_FONT.match(font): return 'p'
    if re.search(r'[A-Za-z]{2}', text): return 'p'
    return 'm'

def span_text_chars(span):
    """chars with their bbox; insert spaces on big gaps; normalize ligatures"""
    out=[]
    prev=None
    for c in span["chars"]:
        ch=c["c"]
        for k,v in LIG.items():
            if k in ch: ch=ch.replace(k,v)
        if prev is not None:
            gap=c["bbox"][0]-prev["bbox"][2]
            if gap>max(2.0,0.36*span["size"]) and not prev["c"].isspace():
                out.append((" ", None))
        for part in ch:
            out.append((part, c))
        prev=c
    return out

def build_lines(page):
    lines=[]
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"]!=0: continue
        for l in b["lines"]:
            segs=[]
            for s in l["spans"]:
                cls=span_cls(s["font"], "".join(c["c"] for c in s["chars"]))
                text="".join(c["c"] for c in s["chars" if False else "chars"])
                for item in span_text_chars(s):
                    ch, cbox = item
                    if cbox is None:
                        segs.append({"cls":'p',"text":" ","bbox":None,"font":"space","size":s["size"]})
                    else:
                        segs.append({"cls":cls,"text":ch,"bbox":list(cbox["bbox"]),"font":s["font"],
                                     "size":round(s["size"],2),"bold":bool(s["flags"]&16),"italic":bool(s["flags"]&2)})
            if not segs: continue
            for s_ in segs:
                if s_["bbox"] is None:
                    # give space a zero-width bbox at previous char end
                    pass
            # fill missing bboxes for spaces
            last_x=None
            for s_ in segs:
                if s_["bbox"] is None:
                    s_["bbox"]=[last_x,0,last_x,0] if last_x is not None else [0,0,0,0]
                else:
                    last_x=s_["bbox"][2]
            lines.append({"segs":segs,"bbox":list(l["bbox"])})
    return lines

def seg_runs(segs):
    """merge consecutive segs by class into runs (keep spaces inside prose runs)"""
    runs=[]
    for s_ in segs:
        if runs and runs[-1]["cls"]==s_["cls"] and s_["cls"]=="p" and s_["bbox"][0]-runs[-1]["x1"]<=1.2:
            r=runs[-1]; r["text"]+=s_["text"]; r["x1"]=max(r["x1"],s_["bbox"][2]); r["chars"].append(s_)
        elif runs and runs[-1]["cls"]==s_["cls"] and s_["cls"]=="m" and s_["bbox"][0]-runs[-1]["x1"]<=1.5:
            r=runs[-1]; r["text"]+=s_["text"]; r["x1"]=max(r["x1"],s_["bbox"][2]); r["y0"]=min(r["y0"],s_["bbox"][1]); r["y1"]=max(r["y1"],s_["bbox"][3]); r["chars"].append(s_)
        else:
            runs.append({"cls":s_["cls"],"text":s_["text"],"x0":s_["bbox"][0],"x1":s_["bbox"][2],
                         "y0":s_["bbox"][1],"y1":s_["bbox"][3],"chars":[s_]})
    for r in runs:
        if r["cls"]=="p":
            r["text"]=re.sub(r"\s+"," ",r["text"])
            r["size"]=statistics.median([c["size"] for c in r["chars"]])
            r["bold"]=sum(c["bold"] for c in r["chars"])>len(r["chars"])/2
            r["italic"]=sum(c["italic"] for c in r["chars"])>len(r["chars"])/2
        else:
            r["size"]=statistics.median([c["size"] for c in r["chars"]])
            r["bold"]=False; r["italic"]=True
    return runs

def line_info(line):
    runs=seg_runs(line["segs"])
    out=[]; mi=0; math=[]
    for r in runs:
        if r["cls"]=="p":
            if r["text"].strip(): out.append(r["text"])
        else:
            out.append(f"⟦{mi}⟧"); math.append({"i":mi,"bbox":[r["x0"],r["y0"],r["x1"],r["y1"]],"text":r["text"]})
            mi+=1
    text="".join(out)
    text=re.sub(r"\s+"," ",text).strip()
    x0=min(s_["bbox"][0] for s_ in line["segs"] if s_["bbox"][2]>0)
    x1=max(s_["bbox"][2] for s_ in line["segs"])
    sizes=[s_["size"] for s_ in line["segs"] if s_["cls"]=="p"] or [s_["size"] for s_ in line["segs"]]
    return {"text":text,"math":math,"runs":runs,"bbox":line["bbox"],"x0":x0,"x1":x1,
            "size":statistics.median(sizes),
            "bold":sum(s_["bold"] for s_ in line["segs"])>len(line["segs"])/2,
            "italic":sum(s_["italic"] for s_ in line["segs"])>len(line["segs"])/2,
            "pchars":sum(len(r["text"]) for r in runs if r["cls"]=="p"),
            "mchars":sum(len(r["text"]) for r in runs if r["cls"]=="m")}

if __name__=="__main__":
    doc=fitz.open("source/original.pdf")
    for pno in range(int(sys.argv[1]) if len(sys.argv)>1 else 1):
        page=doc[pno]
        print("="*30,"PAGE",pno+1)
        for i,line in enumerate(build_lines(page)):
            inf=line_info(line)
            print(f'{i} ({inf["x0"]:.0f},{inf["bbox"][1]:.0f}-{inf["bbox"][3]:.0f}) sz={inf["size"]:.1f}: {inf["text"][:130]!r}')
