import pymupdf as fitz, json, re, statistics
from collections import defaultdict

PROSE_FONT = re.compile(r'^(Times|Helvetica|Courier|Arial|NimbusRom|NimbusSan)', re.I)
MATH_FONT  = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB|Math)', re.I)
LIG = {"\ufb01":"fi","\ufb02":"fl","\ufb00":"ff","\ufb03":"ffi","\ufb04":"ffl","\ufb05":"ft","\ufb06":"st"}

def span_class(font, text):
    if MATH_FONT.match(font): return 'm'
    if PROSE_FONT.match(font): return 'p'
    if re.search(r'[A-Za-z]{2}', text): return 'p'
    return 'm'

doc = fitz.open("source/original.pdf")
pages=[]
for pno in range(doc.page_count):
    page=doc[pno]
    chars=[]
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"]!=0: continue
        for l in b["lines"]:
            for s in l["spans"]:
                cls=span_class(s["font"], "".join(c["c"] for c in s["chars"]))
                for c in s["chars"]:
                    ch=c["c"]
                    for k,v in LIG.items():
                        if k in ch: ch=ch.replace(k,v)
                    chars.append({"x0":c["bbox"][0],"y0":c["bbox"][1],"x1":c["bbox"][2],"y1":c["bbox"][3],
                                  "bx":c["origin"][0],"by":c["origin"][1],"c":ch,
                                  "font":s["font"],"size":round(s["size"],2),
                                  "bold":bool(s["flags"]&16),"italic":bool(s["flags"]&2),"cls":cls})
    pages.append({"page":pno+1,"chars":chars})
json.dump(pages, open("analysis/chars.json","w"), ensure_ascii=False)

GUTTER=296.5
def rows_from(chars):
    big=[c for c in chars if c["size"]>=7.15]
    small=[c for c in chars if c["size"]<7.15]
    big.sort(key=lambda c:(c["by"],c["x0"]))
    rows=[]
    for c in big:
        if rows and abs(c["by"]-rows[-1]["by"])<=1.6:
            rows[-1]["chars"].append(c)
        else:
            rows.append({"by":c["by"],"chars":[c]})
    # attach small chars
    for c in small:
        cy=(c["y0"]+c["y1"])/2; cx=(c["x0"]+c["x1"])/2
        best=None;bd=1e9
        for r in rows:
            y0=min(x["y0"] for x in r["chars"]); y1=max(x["y1"] for x in r["chars"])
            if y0-4<=cy<=y1+3:
                # horizontal proximity: distance to row x range
                x0=min(x["x0"] for x in r["chars"]); x1=max(x["x1"] for x in r["chars"])
                d=0 if x0<=cx<=x1 else min(abs(cx-x0),abs(cx-x1))
                if d<bd: bd=d; best=r
        if best is not None and bd<12:
            best["chars"].append(c)
        else:
            rows.append({"by":c["by"],"chars":[c]})
    # split rows into column segments by big gaps crossing gutter
    out=[]
    for r in rows:
        cs=sorted(r["chars"], key=lambda c:c["x0"])
        seg=[cs[0]]
        for c,nxt in zip(cs,cs[1:]):
            if nxt["x0"]-max(x["x1"] for x in seg)>8.0 and (max(x["x1"] for x in seg)<GUTTER+2 or c["x0"]>GUTTER-2) and nxt["x0"]>GUTTER-2 and max(x["x1"] for x in seg)<=GUTTER+2 or (nxt["x0"]-max(x["x1"] for x in seg)>8.0 and c["x0"]<GUTTER<nxt["x0"]):
                out.append(seg); seg=[nxt]
            else:
                seg.append(nxt)
        out.append(seg)
    res=[]
    for seg in out:
        seg.sort(key=lambda c:c["x0"])
        res.append({"chars":seg,"by":seg[0]["by"],"x0":min(c["x0"] for c in seg),
                    "x1":max(c["x1"] for c in seg),"y0":min(c["y0"] for c in seg),"y1":max(c["y1"] for c in seg)})
    res.sort(key=lambda r:(0 if r["x0"]<GUTTER else 1, r["by"], r["x0"]))
    return res

def runs_of(row):
    runs=[]
    for c in row["chars"]:
        if runs and runs[-1]["cls"]==c["cls"] and c["x0"]-runs[-1]["x1"]<=2.2 and abs(c["size"]-runs[-1]["chars"][-1]["size"])<1.6:
            r=runs[-1]; r["chars"].append(c); r["x1"]=max(r["x1"],c["x1"]); r["y1"]=max(r["y1"],c["y1"]); r["y0"]=min(r["y0"],c["y0"])
        else:
            runs.append({"cls":c["cls"],"chars":[c],"x0":c["x0"],"x1":c["x1"],"y0":c["y0"],"y1":c["y1"]})
    for r in runs:
        r["text"]="".join(c["c"] for c in r["chars"])
        r["size"]=statistics.median([c["size"] for c in r["chars"]])
        r["bold"]=sum(c["bold"] for c in r["chars"])>len(r["chars"])/2
        r["italic"]=sum(c["italic"] for c in r["chars"])>len(r["chars"])/2
        r["font"]=statistics.mode([c["font"] for c in r["chars"]])
    return runs

if __name__=="__main__":
    for pg in pages[:4]:
        print("="*30,"PAGE",pg["page"])
        rows=rows_from(pg["chars"])
        for row in rows:
            runs=runs_of(row)
            parts=[]
            for r in runs:
                if r["cls"]=="p": parts.append(r["text"])
                else: parts.append("⟨"+r["text"]+"⟩")
            print(f'({row["x0"]:.0f},{row["y0"]:.0f},w={row["x1"]-row["x0"]:.0f}) ', "".join(parts)[:180])
