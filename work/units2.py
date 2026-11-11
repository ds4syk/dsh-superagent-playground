import pymupdf as fitz, json, re, statistics, sys

PROSE_FONT = re.compile(r'^(Times|Helvetica|Courier|Arial|NimbusRom|NimbusSan)', re.I)
MATH_FONT  = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB|Math)', re.I)
LIG = {"\ufb01":"fi","\ufb02":"fl","\ufb00":"ff","\ufb03":"ffi","\ufb04":"ffl","\ufb05":"ft","\ufb06":"st"}
GUTTER=296.5

def span_cls(font, text):
    if MATH_FONT.match(font): return 'm'
    if PROSE_FONT.match(font): return 'p'
    if re.search(r'[A-Za-z]{2}', text): return 'p'
    return 'm'

DROPS=[]
def page_chars(page):
    chars=[]
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"]!=0: continue
        for l in b["lines"]:
            for s in l["spans"]:
                cls=span_cls(s["font"], "".join(c["c"] for c in s["chars"]))
                for c in s["chars"]:
                    ch=c["c"]
                    for k,v in LIG.items():
                        if k in ch: ch=ch.replace(k,v)
                    for ch1 in ch:
                        if s["size"]>20 and c["bbox"][1]>180:
                            DROPS.append({"c":ch1,"x0":c["bbox"][0],"y0":c["bbox"][1],"x1":c["bbox"][2],"y1":c["bbox"][3]})
                            continue
                        chars.append({"c":ch1,"cls":cls,"x0":c["bbox"][0],"y0":c["bbox"][1],
                                      "x1":c["bbox"][2],"y1":c["bbox"][3],"by":c["origin"][1],
                                      "size":round(s["size"],2),"font":s["font"],
                                      "bold":bool(s["flags"]&16),"italic":bool(s["flags"]&2)})
    return chars

def make_rows(chars):
    big=[c for c in chars if c["size"]>=7.15]
    small=[c for c in chars if c["size"]<7.15]
    big.sort(key=lambda c:(c["by"],c["x0"]))
    rows=[]
    for c in big:
        if rows and abs(c["by"]-rows[-1]["by"])<=1.6: rows[-1]["chars"].append(c)
        else: rows.append({"by":c["by"],"chars":[c]})
    for c in small:
        cy=(c["y0"]+c["y1"])/2; cx=(c["x0"]+c["x1"])/2; best=None; bd=1e9
        for r in rows:
            y0=min(x["y0"] for x in r["chars"]); y1=max(x["y1"] for x in r["chars"])
            if y0-4.5<=cy<=y1+3:
                x0=min(x["x0"] for x in r["chars"]); x1=max(x["x1"] for x in r["chars"])
                d=0 if x0<=cx<=x1 else min(abs(cx-x0),abs(cx-x1))
                if d<bd: bd=d; best=r
        if best is not None and bd<14: best["chars"].append(c)
        else: rows.append({"by":c["by"],"chars":[c]})
    # split rows crossing gutter
    out=[]
    for r in rows:
        cs=sorted(r["chars"],key=lambda c:c["x0"])
        seg=[cs[0]]
        for a,b in zip(cs,cs[1:]):
            if b["x0"]>GUTTER-2 and max(c["x1"] for c in seg)<=GUTTER+2 and b["x0"]-max(c["x1"] for c in seg)>4:
                out.append(seg); seg=[b]
            else: seg.append(b)
        out.append(seg)
    res=[]
    for seg in out:
        seg.sort(key=lambda c:c["x0"])
        res.append({"chars":seg,"by":seg[0]["by"],"x0":min(c["x0"] for c in seg),"x1":max(c["x1"] for c in seg),
                    "y0":min(c["y0"] for c in seg),"y1":max(c["y1"] for c in seg)})
    # merge rows that share a visual line (superscripts / subscripts)
    for r in res:
        r["col"]=0 if r["x1"]<=GUTTER+4 else (1 if r["x0"]>=GUTTER-4 else -1)
    res.sort(key=lambda r:(r["col"], r["x0"], r["by"]))
    lines=[]
    for r in res:
        placed=False
        for ln in lines:
            if r["col"]==ln["col"] and abs(r["by"]-ln["by"])<=4.2 and (r["x0"]<ln["x1"]+30 and ln["x0"]<r["x1"]+30):
                ln["chars"].extend(r["chars"]); ln["x0"]=min(ln["x0"],r["x0"]); ln["x1"]=max(ln["x1"],r["x1"])
                ln["y0"]=min(ln["y0"],r["y0"]); ln["y1"]=max(ln["y1"],r["y1"])
                # weighted baseline by char count
                n=len(ln["chars"]); m=len(r["chars"])
                ln["by"]=(ln["by"]*(n-m)+r["by"]*m)/n
                placed=True; break
        if not placed:
            lines.append(dict(r, chars=list(r["chars"])))
    for ln in lines:
        ln["chars"].sort(key=lambda c:c["x0"])
    lines=[l for l in lines if "".join(c["c"] for c in l["chars"]).strip()]
    for l in lines:
        pb=[c["by"] for c in l["chars"] if c["cls"]=="p"]
        l["baseline"]=statistics.median(pb) if pb else None
    for l in lines:
        if l["baseline"] is None:
            best=None
            for o in lines:
                if o["baseline"] is None or o["col"]!=l["col"]: continue
                d=abs(o["by"]-l["by"])
                if d<7.5 and (best is None or d<best[0]): best=(d,o["baseline"])
            l["baseline"]=best[1] if best else l["by"]
    return lines

def row_text(row):
    """returns (text with placeholders, math runs list) and per-char gap based spaces"""
    parts=[]; math=[]; run=None; mi=0; prev=None
    baseline=row.get("baseline")
    for c in row["chars"]:
        if prev is not None:
            gap=c["x0"]-prev["x1"]
            if gap>max(2.0,0.36*prev["size"]) and not prev["c"].isspace() and c["c"] not in ",.;:)]’":
                if run is None: parts.append(" ")
        if c["cls"]=="m":
            if run is None: run={"chars":[],"x0":c["x0"],"x1":c["x1"],"y0":c["y0"],"y1":c["y1"]}
            run["chars"].append(c); run["x0"]=min(run["x0"],c["x0"]); run["x1"]=max(run["x1"],c["x1"])
            run["y0"]=min(run["y0"],c["y0"]); run["y1"]=max(run["y1"],c["y1"])
        else:
            if run is not None:
                math.append({"i":mi,"bbox":[run["x0"],run["y0"],run["x1"],run["y1"]],
                             "text":"".join(x["c"] for x in run["chars"]),
                             "va": round((baseline-run["y1"]) if baseline else 0.0,2)})
                parts.append(f"⟦{mi}⟧"); mi+=1; run=None
            parts.append(c["c"])
        prev=c
    if run is not None:
        math.append({"i":mi,"bbox":[run["x0"],run["y0"],run["x1"],run["y1"]],
                     "text":"".join(x["c"] for x in run["chars"]),
                     "va": round((baseline-run["y1"]) if baseline else 0.0,2)})
        parts.append(f"⟦{mi}⟧")
    text="".join(parts)
    return text, math

HEAD_RE=re.compile(r'^(\d+\)|[A-Z]\.\s+[A-Z]|[IVX]+\.\s+[A-Z]|Abstract—|Index Terms—|Fig\.|Fig |TABLE)')

def build(pno, page):
    chars=page_chars(page)
    rows=make_rows(chars)
    # column assignment
    for r in rows:
        r["col"]=0 if (r["x0"]<GUTTER and not (r["x0"]<GUTTER<r["x1"])) else (1 if r["x0"]>=GUTTER else None)
        if r["x0"]<GUTTER-40 and r["x1"]>GUTTER+40: r["col"]=-1  # full width
    rows=[r for r in rows if r["col"] is not None]
    rows.sort(key=lambda r:(r["col"] if r["col"]>=0 else 0.5, r["y0"], r["x0"]))
    # info
    for r in rows:
        t,m=row_text(r)
        r["text"]=t; r["math"]=m
        pchars=sum(1 for c in r["chars"] if c["cls"]=="p"); mchars=sum(1 for c in r["chars"] if c["cls"]=="m")
        r["pchars"]=pchars; r["mchars"]=mchars
        sizes=[c["size"] for c in r["chars"] if c["cls"]=="p"] or [c["size"] for c in r["chars"]]
        r["size"]=statistics.median(sizes)
        r["bold"]=sum(c["bold"] for c in r["chars"])>len(r["chars"])/2
        r["italic"]=sum(c["italic"] for c in r["chars"])>len(r["chars"])/2
        r["display"]= (pchars<=2 and mchars>=2)
    col_left={0:None,1:None,-1:None}
    for r in rows:
        if r["col"] in col_left and (col_left[r["col"]] is None or r["x0"]<col_left[r["col"]]):
            col_left[r["col"]]=r["x0"]
    units=[]; cur=None
    for r in rows:
        new=True
        if cur is not None and not r["display"] and not cur["display"]:
            gap=r["y0"]-cur["y1"]
            same_col = r["col"]==cur["col"]
            if same_col and gap<=2.8:
                style_same = abs(r["size"]-cur["size"])<=0.6 and r["bold"]==cur["bold"]
                prev_full = cur["last_x1"] >= (289.0 if cur["col"]==0 else 552.0) - 20
                starts_head = bool(HEAD_RE.match(r["text"]))
                if style_same and prev_full and not starts_head:
                    new=False
        if new:
            if cur is not None: units.append(cur)
            cur={"page":pno,"col":r["col"],"rows":[r],"x0":r["x0"],"y0":r["y0"],"x1":r["x1"],"y1":r["y1"],
                 "size":r["size"],"bold":r["bold"],"italic":r["italic"],"display":r["display"],
                 "text":r["text"],"math":list(r["math"])}
        else:
            cur["rows"].append(r)
            cur["x0"]=min(cur["x0"],r["x0"]); cur["x1"]=max(cur["x1"],r["x1"])
            cur["y0"]=min(cur["y0"],r["y0"]); cur["y1"]=max(cur["y1"],r["y1"])
            base=len(cur["math"])
            for m in r["math"]:
                cur["math"].append({"i":base+m["i"],"bbox":m["bbox"],"text":m["text"],"va":m.get("va",0.0)})
            if re.search(r"⟦(\d+)⟧", r["text"]):
                cur["text"]+=" "+re.sub(r"⟦(\d+)⟧", lambda mm: f"⟦{base+int(mm.group(1))}⟧", r["text"])
            else:
                cur["text"]=(cur["text"]+" "+r["text"]).strip()
        cur["last_x1"]=r["x1"]
    if cur is not None: units.append(cur)
    # join text with hyphenation handling
    for u in units:
        t=u["text"]
        t=re.sub(r"-\s+([a-z])", r"\1", t)   # de-hyphenate
        t=re.sub(r"\s+"," ",t).strip()
        t=t.replace(" ⟦","⟦").replace("⟧ ","⟧")
        u["text"]=t
    # re-attach drop caps
    drops=[d for d in DROPS if pno==page.number+1]
    for d in drops:
        best=None
        for u in units:
            if u["y0"]-2<=d["y0"] and d["y1"]<=u["y1"]+2 and abs(u["x0"]-d["x0"])<8:
                if best is None or u["y0"]<best["y0"]: best=u
        if best is not None:
            best["text"]=d["c"]+best["text"]
            best["x0"]=min(best["x0"],d["x0"]); best["y0"]=min(best["y0"],d["y0"])
            best["bbox"]=[best["x0"],best["y0"],best["x1"],best["y1"]]
    return units

if __name__=="__main__":
    doc=fitz.open("source/original.pdf")
    pages=[]
    for pno in range(doc.page_count):
        units=build(pno+1, doc[pno])
        for k,u in enumerate(units):
            u["id"]=f"p{pno+1}_{k}"
            u["bbox"]=[round(v,1) for v in [u["x0"],u["y0"],u["x1"],u["y1"]]]
            u.pop("rows")
        pages.append({"page":pno+1,"units":units})
    json.dump(pages, open("analysis/units.json","w"), ensure_ascii=False, indent=1)
    print("units:", [len(p["units"]) for p in pages])
    print("chars:", sum(len(u["text"]) for p in pages for u in p["units"]))
