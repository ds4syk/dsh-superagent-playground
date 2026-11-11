import json, re, os, sys, pymupdf as fitz
import typeset as T

import glob
PREP=json.load(open("analysis/prepared.json"))
ZH={}
for f in ["analysis/zh_final.json"]:
    try: ZH.update(json.load(open(f)))
    except Exception as e: print("WARN",f,e)
print("translations:",len(ZH))

SIZE={"body":9.7,"abstract":9.2,"caption":7.9,"section":9.6,"subsection":9.5,
      "title":22.0,"author":10.5}
LMIN={"body":1.26,"abstract":1.32,"caption":1.32,"section":1.0,"subsection":1.0,
      "title":1.30,"author":1.0}
LMAX={"body":1.58,"abstract":1.60,"caption":1.55,"section":1.0,"subsection":1.0,
      "title":1.42,"author":1.0}

def math_items(u):
    out={}
    maxh=1e9
    for m in u["math"]:
        r=m["rect"]
        w=r[2]-r[0]; h=r[3]-r[1]; base=m.get("base", r[3])
        top=(base-r[1]); bot=(r[3]-base)
        sz=max(u["size"],6.0)
        k=1.0
        if top>1.3*sz: k=min(k,1.3*sz/top)
        if bot>0.45*sz: k=min(k,0.45*sz/bot)
        k=max(0.15,min(1.0,k))
        w=max(0.5,w*k); h=max(0.5,h*k); top=top*k; bot=bot*k
        item={"w":w,"h":h,"rect":r,"base":base,"file":m["file"],"text":m["text"],
              "top":top,"bot":bot,
              "render":m.get("render","img")}
        item["area"]=max(0.1,(r[2]-r[0]))*max(0.1,(r[3]-r[1]))
        out[m["i"]]=item
    return out

def tokens_at(text, items, size):
    toks=T.tokenize(text, items)
    out=[]
    for t in toks:
        if t[0]=='m':
            it=dict(t[1])
            if it["render"]=="text":
                it["w"]=T.font().text_length(it["text"], fontsize=size*0.98)
                it["size"]=size*0.98
            out.append(('m',it))
        else: out.append(t)
    return out

def fits(toks, W, H, s, lmin, kind):
    h,ls=T.natural_height(toks,W,s,lmin*s,kind)
    return h<=H+0.6, len(ls)

def choose(text, items, W, H, base, role, kind):
    lmin=LMIN.get(role,1.3); lmax=LMAX.get(role,1.5)
    s=base
    while s>=base*0.62:
        toks=tokens_at(text,items,s)
        ok,n=fits(toks,W,H,s,lmin,kind)
        if ok:
            lead=None
            if n>1:
                lead=(H-s)/(n-1)
                lead=max(lmin*s, min(lmax*s, lead))
            else:
                lead=s
            return s,lead,n
        s-=0.1
    toks=tokens_at(text,items,base*0.62)
    h,ls=T.natural_height(toks,W,base*0.62,lmin*base*0.62,kind)
    n=len(ls)
    if h>H+max(4.0,0.10*H): return None,None,None
    return base*0.62, max(lmin*base*0.62,(H-base*0.62)/(max(n-1,1))), n

def area(r):
    return max(0.0,r[2]-r[0])*max(0.0,r[3]-r[1])

def inter(a,b):
    return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]

def extra_rects(pg, u):
    out=[]
    ub=u["bbox"]
    for v in pg["units"]:
        if v["kind"]!="math": continue
        i=inter(ub, v["bbox"])
        if area(i) > 0.4*area(v["bbox"]) and area(i)>4:
            out.append(v["bbox"])
    return out

def plan(pg):
    """decide per-unit skip/size before any modification"""
    out={}
    for u in pg["units"]:
        if u["kind"]!="text" or u["role"]=="keep": continue
        zh=ZH.get(u["id"])
        if not zh or not zh.strip(): continue
        role=u["role"]; x0,y0,x1,y1=u["bbox"]; W=x1-x0; H=y1-y0
        items=math_items(u)
        kind="bold" if role in ("section","subsection","title") else "regular"
        if role in ("header","footer"):
            out[u["id"]]=(False,u["size"],u["size"]*1.05,1); continue
        base=SIZE.get(role,u["size"])
        size,lead,n=choose(zh,items,W,H,base,role,kind)
        if size is None:
            out[u["id"]]=(True,None,None,None); continue
        used=(n-1)*lead+size
        mx=max([m["rect"][2]-m["rect"][0] for m in u["math"]]+[0])
        mh=max([m["rect"][3]-m["rect"][1] for m in u["math"]]+[0])
        if mh>2.5*u["size"]:
            out[u["id"]]=(True,None,None,None); continue
        if len(u["math"])>=6 and mx>0.5*W and used>0.85*H:
            out[u["id"]]=(True,None,None,None); continue
        if used>H+max(6.0,0.25*H):
            out[u["id"]]=(True,None,None,None); continue
        out[u["id"]]=(False,size,lead,n)
    return out

def build(out_pdf="source/translated.pdf", pages=None, drop_annots=True):
    doc=fitz.open("source/original.pdf")
    if drop_annots:
        for p in doc:
            for a in list(p.annots() or []): p.delete_annot(a)
    PLANS={pg["page"]:plan(pg) for pg in PREP if (not pages or pg["page"] in pages)}
    for pg in PREP:
        pno=pg["page"]
        if pages and pno not in pages: continue
        page=doc[pno-1]
        plans=PLANS[pno]
        for u in pg["units"]:
            if u["kind"]!="text" or u["role"]=="keep" or u["id"] not in ZH: continue
            if plans.get(u["id"],(True,))[0]: continue
            x0,y0,x1,y1=u["bbox"]
            page.add_redact_annot(fitz.Rect(x0-0.6,y0-0.6,x1+0.6,y1+0.6), fill=(1,1,1), cross_out=False)
            for er in extra_rects(pg,u):
                page.add_redact_annot(fitz.Rect(er[0]-0.4,er[1]-0.4,er[2]+0.4,er[3]+0.4), fill=(1,1,1), cross_out=False)
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
            role=u["role"]; x0,y0,x1,y1=u["bbox"]; W=x1-x0; H=y1-y0
            items=math_items(u)
            kind="bold" if role in ("section","subsection","title") else "regular"
            skip,size,lead,n=PLANS[pno].get(u["id"],(True,None,None,None))
            if skip:
                report.append((u["id"],role,"SKIP",0,0,0,round(H,1),0)); continue
            toks=tokens_at(zh,items,size)
            used=(n-1)*lead+size
            align="center" if role in ("title","section") else ("left" if (role=="body" and size<8.6) else "justify")
            baseline0=None
            if role in ("title","section","author"):
                baseline0=y0+max(0,(H-used)/2)+size*0.86
            T.draw(page,toks,(x0,y0,x1,y1),size,lead,align=align,kind=kind,baseline0=baseline0)
            report.append((u["id"],role,round(size,2),round(lead,2),n,round(used,1),round(H,1),round(used/max(H,1e-6),2)))
    try:
        doc.subset_fonts(verbose=False)
    except Exception as e:
        print("WARN subset_fonts:", e)
    doc.save(out_pdf, deflate=True, garbage=4)
    return report

if __name__=="__main__":
    pages=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 and sys.argv[1]!="all" else None
    rep=build(pages=pages)
    if pages:
        d=fitz.open("source/translated.pdf")
        for pno in pages:
            pix=d[pno-1].get_pixmap(dpi=150); pix.save(f"work/prev_p{pno:02d}.png")
    print("id role size lead lines used H fill")
    for r in rep:
        if r[7]<0.80 or r[7]>1.02: print(r)
    print("n units", len(rep))
