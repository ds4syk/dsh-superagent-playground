import json, re, os, pymupdf as fitz

pages = json.load(open("analysis/units.json"))
doc = fitz.open("source/original.pdf")
os.makedirs("work/crops", exist_ok=True)

def is_equation_number_only(t):
    rest = re.sub(r"⟦\d+⟧", "", t).strip()
    return rest=="" or re.fullmatch(r"\(?\d+\)?\.?", rest) is not None

def unit_math_crops(u):
    crops=[]
    for m in u["math"]:
        i=m["i"]; bb=m["bbox"]
        x0,y0,x1,y1=bb
        pad=0.7
        rect=fitz.Rect(x0-pad,y0-pad,x1+pad,y1+pad)
        rect &= doc[u["page"]-1].rect
        name=f'crops/{u["id"]}_{i}.png'
        path=os.path.join("work",name)
        if not os.path.exists(path):
            pix=doc[u["page"]-1].get_pixmap(dpi=600, clip=rect, alpha=False)
            pix.save(path)
        crops.append({"i":i,"file":path,"rect":[rect.x0,rect.y0,rect.x1,rect.y1],"text":m["text"]})
    return crops

out=[]
for pg in pages:
    units=[]
    page=doc[pg["page"]-1]
    for u in pg["units"]:
        kind = "text"
        if u.get("display"): kind="math"
        if kind=="text" and is_equation_number_only(u["text"]): kind="math"
        crops = unit_math_crops(u) if kind=="text" else []
        units.append({"id":u["id"],"page":u["page"],"col":u["col"],"kind":kind,
                      "bbox":[round(v,1) for v in u["bbox"]],"size":u["size"],
                      "bold":u["bold"],"italic":u["italic"],
                      "text":u["text"],"math":crops})
    out.append({"page":pg["page"],"units":units})
json.dump(out, open("analysis/prepared.json","w"), ensure_ascii=False, indent=1)
print("crops:", len(os.listdir("work/crops")))
print("text units:", sum(1 for p in out for u in p["units"] if u["kind"]=="text"))
