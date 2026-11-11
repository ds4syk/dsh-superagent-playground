import json, re, os, sys, pymupdf as fitz
import typeset as T

PREP=json.load(open("analysis/prepared.json"))
ZH=json.load(open("analysis/zh.json"))

ROLE_SIZE={"body":None,"abstract":None,"caption":None,"section":None,"subsection":None,
           "title":22.0,"author":10.9,"header":None,"footer":None}
SCALE_UP={"body":1.12,"abstract":1.10,"caption":1.04,"section":1.0,"subsection":1.0,
          "title":1.0,"author":1.0,"header":1.0,"footer":1.0}
LH={"body":1.34,"abstract":1.36,"caption":1.38,"section":1.0,"subsection":1.0,
    "title":1.35,"author":1.0,"header":1.0,"footer":1.0}

def math_items(u):
    out={}
    for m in u["math"]:
        r=m["rect"]; w=(r[2]-r[0])-1.4; h=(r[3]-r[1])-1.4
        base=r[3]-0.7+m.get("va",0.0)
        item={"w":w,"h":h,"rect":r,"base":base,"file":m["file"],"text":m["text"],"render":m.get("render","img")}
        if item["render"]=="text":
            item["w"]=T.font().text_length(m["text"], fontsize=1.0)*0 or 0
        out[m["i"]]=item
    return out

def prepare_tokens(text, items, size):
    toks=T.tokenize(text, items)
    out=[]
    for t in toks:
        if t[0]=='m':
            it=dict(t[1])
            if it["render"]=="text":
                it["w"]=T.font().text_length(it["text"], fontsize=size)
                it["size"]=size*0.98
            out.append(('m',it))
        else:
            out.append(t)
    return out

def fit(tokens, W, H, base, role):
    lo, hi = base*0.62, base*SCALE_UP.get(role,1.0)
    lhf=LH.get(role,1.3)
    best=(lo,None)
    s=hi
    while s>=lo-1e-6:
        lead=lhf*s
        h,_=T.natural_height(tokens,W,s,lead,"bold" if role in ("section","subsection","title") else "regular")
        if h<=H+0.5:
            return s,lead
        s-=0.1
    return lo, lhf*lo

def unit_html_none(): pass

def build(out_pdf="source/translated.pdf", pages=None):
    doc=fitz.open("source/original.pdf")
    for pg in PREP:
        pno=pg["page"]
        if pages and pno not in pages: continue
        page=doc[pno-1]
        for u in pg["units"]:
            if u["kind"]!="text" or u["role"]=="keep" or u["id"] not in ZH: continue
            x0,y0,x1,y1=u["bbox"]
            page.add_redact_annot(fitz.Rect(x0-0.6,y0-0.6,x1+0.6,y1+0.6), fill=(1,1,1), cross_out=False)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    report=[]
    for pg in PREP:
        pno=pg["page"]
        if pages and pno not in pages: continue
        page=doc[pno-1]
        for u in pg["units"]:
            if u["kind"]!="text" or u["role"]=="keep": continue
            zh=ZH.get(u["id"])
            if not zh or not zh.strip(): continue
            role=u["role"]
            x0,y0,x1,y1=u["bbox"]
            W=(x1-x0); H=(y1-y0)
            if role=="header" or role=="footer":
                size=u["size"]; lead=size*1.0
            else:
                if role=="title": base=u["size"]*0.92
                elif role=="author": base=u["size"]*0.96
                elif role in ("section","subsection"): base=u["size"]*1.12
                else: base=u["size"]
                items=math_items(u)
                toks=prepare_tokens(zh, items, base)
                size,lead=fit(toks,W,H,base,role)
            items=math_items(u)
            toks=prepare_tokens(zh, items, size)
            kind="bold" if role in ("section","subsection","title") else "regular"
            align="center" if role in ("title","section") else "justify"
            used,nl=T.draw(page, toks, (x0,y0,x1,y1), size, lead, align=align, kind=kind)
            report.append((u["id"],role,round(size,2),round(lead,2),nl,round(used,1),round(H,1),round(used/max(H,1e-6),2)))
    doc.save(out_pdf, deflate=True, garbage=3)
    return report

if __name__=="__main__":
    pages=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 and sys.argv[1]!="all" else None
    rep=build(pages=pages)
    if pages:
        d=fitz.open("source/translated.pdf")
        for pno in pages:
            pix=d[pno-1].get_pixmap(dpi=140); pix.save(f"work/prev_p{pno:02d}.png")
    print("id role size lead lines used H fill")
    for r in rep: print(r)
