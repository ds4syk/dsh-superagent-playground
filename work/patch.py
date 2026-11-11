"""Targeted fixes on prepared.json: composite crops for split inline formulas."""
import json, os, pymupdf as fitz
prep=json.load(open("analysis/prepared.json"))
doc=fitz.open("source/original.pdf")
by_id={u["id"]:u for p in prep for u in p["units"]}

def set_crop(uid, idx, rect, base, text, page):
    u=by_id[uid]
    for m in u["math"]:
        if m["i"]==idx:
            name=f'crops/{uid}_{idx}.png'
            r=fitz.Rect(rect) & doc[page-1].rect
            pix=doc[page-1].get_pixmap(dpi=600, clip=r, alpha=False, annots=False)
            pix.save(os.path.join("work",name))
            m["rect"]=[r.x0,r.y0,r.x1,r.y1]; m["base"]=base
            m["text"]=text; m["file"]=name
            print("patched",uid,idx,[round(v,1) for v in m["rect"]])
            return
    print("missing",uid,idx)

# p2_22 item0: radical + tr(X^T X) together
set_crop("p2_22",0,[303.6,589.6,357.2,612.0],611.8,"√tr(X⊤X)",2)
json.dump(prep,open("analysis/prepared.json","w"),ensure_ascii=False,indent=1)
