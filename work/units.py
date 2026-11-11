import json, re, statistics, sys

pages=json.load(open("analysis/chars.json"))
GUTTER=296.5

def rows_from(chars):
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
            if y0-4<=cy<=y1+3:
                x0=min(x["x0"] for x in r["chars"]); x1=max(x["x1"] for x in r["chars"])
                d=0 if x0<=cx<=x1 else min(abs(cx-x0),abs(cx-x1))
                if d<bd: bd=d; best=r
        if best is not None and bd<12: best["chars"].append(c)
        else: rows.append({"by":c["by"],"chars":[c]})
    out=[]
    for r in rows:
        cs=sorted(r["chars"], key=lambda c:c["x0"])
        seg=[cs[0]]
        for a,b in zip(cs,cs[1:]):
            gx=b["x0"]-max(x["x1"] for x in seg)
            if gx>8.0 and (max(x["x1"] for x in seg)<=GUTTER+2<b["x0"] or (b["x0"]-max(x["x1"] for x in seg)>8 and a["x0"]<GUTTER<b["x0"])):
                out.append(seg); seg=[b]
            else: seg.append(b)
        out.append(seg)
    res=[]
    for seg in out:
        seg.sort(key=lambda c:c["x0"])
        res.append({"chars":seg,"by":seg[0]["by"],"x0":min(c["x0"] for c in seg),
                    "x1":max(c["x1"] for c in seg),"y0":min(c["y0"] for c in seg),"y1":max(c["y1"] for c in seg)})
    res.sort(key=lambda r:(0 if r["x0"]<GUTTER else 1, r["by"], r["x0"]))
    return res

def text_of_chars(chars):
    """Join chars inserting spaces on wide gaps; return list of (piece, cls) preserving cls transitions."""
    out=[]; prev=None
    for c in chars:
        if prev is not None:
            gap=c["x0"]-prev["x1"]
            if gap>max(2.2,0.38*prev["size"]) and not prev["c"].isspace():
                if not (c["c"] in ",.;:)]" ): out.append((" ",'p'))
        out.append((c["c"], c["cls"]))
        prev=c
    return out

def runs_of(row):
    pieces=text_of_chars(row["chars"]) if False else None
    runs=[]
    prev=None
    for c in row["chars"]:
        if prev is not None:
            gap=c["x0"]-prev["x1"]
            if gap>max(2.2,0.38*prev["size"]):
                runs.append({"cls":'p',"text":" ","x0":prev["x1"],"x1":prev["x1"],"y0":c["y0"],"y1":c["y1"],
                             "size":c["size"],"bold":c["bold"],"italic":c["italic"],"font":"space"})
        if runs and runs[-1]["cls"]==c["cls"] and c["x0"]-runs[-1]["x1"]<=2.2 and abs(c["size"]-runs[-1].get("size",c["size"]))<1.6:
            r=runs[-1]; r["chars"].append(c); r["x1"]=max(r["x1"],c["x1"]); r["y1"]=max(r["y1"],c["y1"]); r["y0"]=min(r["y0"],c["y0"])
            r["text"]+=c["c"]
        else:
            runs.append({"cls":c["cls"],"chars":[c],"x0":c["x0"],"x1":c["x1"],"y0":c["y0"],"y1":c["y1"],
                         "text":c["c"],"size":c["size"],"bold":c["bold"],"italic":c["italic"],"font":c["font"]})
        prev=c
    for r in runs:
        if "chars" in r:
            r["size"]=statistics.median([c["size"] for c in r["chars"]])
            r["bold"]=sum(c["bold"] for c in r["chars"])>len(r["chars"])/2
            r["italic"]=sum(c["italic"] for c in r["chars"])>len(r["chars"])/2
            r["font"]=statistics.mode([c["font"] for c in r["chars"]])
    return runs

def row_info(row):
    runs=runs_of(row)
    out=[]; mi=0; math=[]
    for r in runs:
        if r["cls"]=="p": out.append(r["text"])
        else:
            out.append(f"⟦{mi}⟧"); math.append({"i":mi,"bbox":[r["x0"],r["y0"],r["x1"],r["y1"]],"text":r["text"]}); mi+=1
    return {"runs":runs,"text":"".join(out),"math":math,"row":row,
            "pchars":sum(len(r["text"]) for r in runs if r["cls"]=="p"),
            "mchars":sum(len(r["text"]) for r in runs if r["cls"]=="m")}

def is_display(info):
    return info["pchars"]<10 and info["mchars"]>=2 and info["row"]["x0"]>45 and info["row"]["x1"]<560

def build_units(pg):
    rows=rows_from(pg["chars"])
    infos=[row_info(r) for r in rows]
    col_left={0:37.9,1:300.9}
    units=[]; cur=None
    for info,row in zip(infos,rows):
        col=0 if row["x0"]<GUTTER else 1
        disp=is_display(info)
        sizes=[c["size"] for c in row["chars"]]
        rsize=statistics.median(sizes)
        rbold=sum(c["bold"] for c in row["chars"])>len(row["chars"])/2
        rital=sum(c["italic"] for c in row["chars"])>len(row["chars"])/2
        new=True
        if cur is not None and not disp and not cur.get("display"):
            gap=row["y0"]-cur["y1"]
            indent=row["x0"]-col_left[col]
            if col==cur["col"]:
                if gap<=4.2 and indent<=5.5 and abs(rsize-cur["size"])<=0.6 and rbold==cur["bold"] and (cur["last_x1"]>col_left[col]+150 or gap<2.2):
                    new=False
        if new:
            if cur: units.append(cur)
            cur={"page":pg["page"],"col":col,"rows":[row],"infos":[info],"x0":row["x0"],"y0":row["y0"],
                 "x1":row["x1"],"y1":row["y1"],"size":rsize,"bold":rbold,"italic":rital,"display":disp}
        else:
            cur["rows"].append(row); cur["infos"].append(info)
            cur["x0"]=min(cur["x0"],row["x0"]); cur["x1"]=max(cur["x1"],row["x1"])
            cur["y0"]=min(cur["y0"],row["y0"]); cur["y1"]=max(cur["y1"],row["y1"])
        cur["last_x1"]=row["x1"]
    if cur: units.append(cur)
    # text assembly + dehyphenation + dropcap merge
    for u in units:
        parts=[]
        for info in u["infos"]: parts.append(info["text"])
        txt=""
        for p in parts:
            p=p.strip()
            if not p: continue
            if txt.endswith("-") and re.match(r"[a-z]",p): txt=txt[:-1]+p
            elif txt.endswith("-"): txt=txt[:-1]+p
            else: txt=(txt+" "+p).strip() if txt else p
        u["text"]=re.sub(r"[ \t]+"," ",txt)
        u["math"]=[m for info in u["infos"] for m in info["math"]]
        u["n_lines"]=len(u["rows"])
    # dropcap merge
    merged=[]
    i=0
    while i<len(units):
        u=units[i]
        if (len(u["text"])==1 and u["size"]>20 and i+1<len(units)
                and re.match(r"^[A-Z]{2,}", units[i+1]["text"]) and units[i+1]["col"]==u["col"]):
            nxt=units[i+1]
            nxt["x0"]=min(nxt["x0"],u["x0"]); nxt["y0"]=min(nxt["y0"],u["y0"]); nxt["y1"]=max(nxt["y1"],u["y1"])
            nxt["text"]=u["text"]+nxt["text"]
            nxt["dropcap"]=[u["x0"],u["y0"],u["x1"],u["y1"]]
            i+=1; continue
        merged.append(u); i+=1
    for k,u in enumerate(merged):
        u["id"]=f'p{u["page"]}_{k}'
        # alignment
        l=u["x0"]; r=u["x1"]
        left_m = 37.9 if u["col"]==0 else 300.9
        right_m = 289.0 if u["col"]==0 else 552.0
        u["align"]="center" if (abs(l-left_m)>10 and abs(r-right_m)>10) else "justify"
        # remove leading spaces in text
        u["text"]=u["text"].lstrip()
    return merged

all_units=[]
for pg in pages:
    all_units.append({"page":pg["page"],"units":build_units(pg)})
json.dump(all_units, open("analysis/units.json","w"), ensure_ascii=False, indent=1)

if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 2
    for pg in all_units:
        if pg["page"]>n: break
        print("="*25,"PAGE",pg["page"])
        for u in pg["units"]:
            print(f'[{u["id"]}] box=({u["x0"]:.1f},{u["y0"]:.1f},{u["x1"]:.1f},{u["y1"]:.1f}) sz={u["size"]:.1f} b={int(u["bold"])} i={int(u["italic"])} {u["align"]} disp={int(u["display"])} nm={len(u["math"])}: {u["text"][:170]!r}')
