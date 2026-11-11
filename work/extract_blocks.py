import pymupdf as fitz, json, re, collections

MATH_FONT = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB)', re.I)

def is_math_span(s):
    if MATH_FONT.match(s["font"]): return True
    t=s["text"].strip()
    if not t: return False
    letters=sum(c.isalpha() and ord(c)<128 for c in t)
    if letters==0: return True
    return False

doc=fitz.open("source/original.pdf")
out=[]
for pno in range(doc.page_count):
    page=doc[pno]
    d=page.get_text("dict")
    blocks=[]
    for bi,b in enumerate(d["blocks"]):
        if b["type"]!=0:
            blocks.append({"i":bi,"type":"image","bbox":list(b["bbox"])})
            continue
        lines=[]
        for l in b["lines"]:
            spans=[]
            for s in l["spans"]:
                spans.append({"t":s["text"],"f":s["font"],"sz":round(s["size"],2),
                              "bbox":[round(x,1) for x in s["bbox"]],"math":is_math_span(s)})
            lines.append({"bbox":[round(x,1) for x in l["bbox"]],"spans":spans})
        txt=" ".join("".join(s["t"] for s in l["spans"]) for l in lines)
        blocks.append({"i":bi,"type":"text","bbox":[round(x,1) for x in b["bbox"]],"lines":lines,"text":txt[:120]})
    out.append({"page":pno+1,"blocks":blocks})
json.dump(out,open("analysis/blocks.json","w"),ensure_ascii=False,indent=1)
for pg in out:
    if pg["page"] in (3,9):
        print("="*20,"page",pg["page"])
        for b in pg["blocks"]:
            print(b["i"],b["type"],b["bbox"],repr(b.get("text","")[:100]))
