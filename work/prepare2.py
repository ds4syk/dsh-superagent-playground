import json, re, os, pymupdf as fitz

pages = json.load(open("analysis/units.json"))
doc = fitz.open("source/original.pdf")
os.makedirs("work/crops", exist_ok=True)

def only_eqnum(t):
    rest = re.sub(r"⟦\d+⟧", "", t).strip()
    return rest=="" or re.fullmatch(r"\(?\d+\)?\.?", rest) is not None

def pure_letters(t):
    return re.fullmatch(r"[A-Za-z0-9]{1,6}", t) is not None

# 1) tag kinds, merge display-math fragments
uni_all={}
for pg in pages:
    units=[dict(u) for u in pg["units"]]
    for u in units:
        u["kind"]="text"
        if u.get("display"): u["kind"]="math"
        if u["kind"]=="text" and only_eqnum(u["text"]): u["kind"]="math"
    merged=[]
    for u in units:
        if merged and merged[-1]["kind"]=="math" and u["kind"]=="math":
            p=merged[-1]
            gap=u["bbox"][1]-p["bbox"][3]
            if gap<5 and u["col"]==p["col"]:
                p["bbox"]=[min(p["bbox"][0],u["bbox"][0]), min(p["bbox"][1],u["bbox"][1]),
                           max(p["bbox"][2],u["bbox"][2]), max(p["bbox"][3],u["bbox"][3])]
                continue
        merged.append(u)
    # also absorb a lone eq-number text unit adjacent to a math unit
    out=[]
    for u in merged:
        if out and u["kind"]=="text" and only_eqnum(u["text"]):
            p=out[-1]
            if p["kind"]=="math" and u["bbox"][1]-p["bbox"][3]<6:
                p["bbox"]=[min(p["bbox"][0],u["bbox"][0]),min(p["bbox"][1],u["bbox"][1]),
                           max(p["bbox"][2],u["bbox"][2]),max(p["bbox"][3],u["bbox"][3])]
                continue
        out.append(u)
    for k,u in enumerate(out):
        u["id"]=f'p{pg["page"]}_{k}'
    uni_all[pg["page"]]=out

# 2) roles + keep + math render types
REF_START={20: ("REFERENCES", 522.0), 21: ("", 0.0), 22: ("", 0.0)}
def role_of(u, page):
    t=u["text"]; b=u["bbox"]; sz=u["size"]
    if b[1]>765: return "footer"
    if b[1]<50: return "header"
    if sz>=16: return "title"
    if page==1 and 140<b[1]<190: return "author"
    if re.match(r"^(Abstract—|Index Terms—)", t): return "abstract"
    if re.match(r"^[IVX]+\.\s+[A-Z]", t): return "section"
    if re.match(r"^[A-Z]\.\s+[A-Z]", t): return "subsection"
    if re.match(r"^(Fig\.|TABLE|TABLE [IVX]+)", t): return "caption"
    if re.match(r"^[A-Z ]{6,}$", t) and sz<9: return "section"
    return "body"

def merge_role(units, role):
    out=[]; buf=None
    for u in units:
        if u["role"]==role:
            if buf is None: buf=dict(u, text=u["text"], ids=[u["id"]])
            else:
                buf["text"]=(buf["text"]+" "+u["text"]).strip()
                buf["bbox"]=[min(buf["bbox"][0],u["bbox"][0]),min(buf["bbox"][1],u["bbox"][1]),
                             max(buf["bbox"][2],u["bbox"][2]),max(buf["bbox"][3],u["bbox"][3])]
                buf["ids"].append(u["id"])
        else:
            if buf: out.append(buf); buf=None
            out.append(u)
    if buf: out.append(buf)
    return out

prepared=[]
for pg in pages:
    page=pg["page"]
    units=[]
    for u in uni_all[page]:
        t=u["text"]
        role=role_of(u, page)
        keep=False
        if page==20 and re.match(r"^\[?\d+\]\s", t.strip()): keep=True
        if page>=21:
            keep=True
        if keep: role="keep"
        crops=[]
        if u["kind"]=="text" and not keep:
            for m in u["math"]:
                i=m["i"]; x0,y0,x1,y1=m["bbox"]
                rect=fitz.Rect(x0+0.30,y0+0.16,x1-0.10,y1-0.14) & doc[page-1].rect
                base_abs=(y1)+m.get("va",0.0)
                name=f'crops/{u["id"]}_{i}.png'
                if not os.path.exists(os.path.join("work",name)):
                    pix=doc[page-1].get_pixmap(dpi=600, clip=rect, alpha=False, annots=False)
                    pix.save(os.path.join("work",name))
                crops.append({"i":i,"file":name,"rect":[rect.x0,rect.y0,rect.x1,rect.y1],
                              "text":m["text"],"va":m.get("va",0.0),"base":base_abs,
                              "render":"text" if pure_letters(m["text"]) else "img"})
        units.append({"id":u["id"],"page":page,"col":u["col"],"kind":u["kind"],"role":role,
                      "bbox":[round(v,1) for v in u["bbox"]],"size":u["size"],
                      "bold":u["bold"],"italic":u["italic"],"text":t,"math":crops})
    units=merge_role(units,"title")
    units=merge_role(units,"author")
    prepared.append({"page":page,"units":units})
json.dump(prepared, open("analysis/prepared.json","w"), ensure_ascii=False, indent=1)

stats={}
for p in prepared:
    for u in p["units"]:
        stats[u["role"]]=stats.get(u["role"],0)+1
print(stats)
print("translatable chars:", sum(len(u["text"]) for p in prepared for u in p["units"] if u["role"]!="keep" and u["kind"]=="text"))
print("crops:", len(os.listdir("work/crops")))
