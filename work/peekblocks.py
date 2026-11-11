import pymupdf as fitz, sys, re
doc=fitz.open("source/original.pdf")
pno=int(sys.argv[1])-1
page=doc[pno]
d=page.get_text("dict")
for i,b in enumerate(d["blocks"]):
    if b["type"]!=0:
        print(i,"IMAGE",[round(x) for x in b["bbox"]]); continue
    txt=""
    for l in b["lines"]:
        line="".join(s["text"] for s in l["spans"])
        txt += (" "+line)
    fonts={s["font"] for l in b["lines"] for s in l["spans"]}
    print(f'{i} bbox={[round(x,1) for x in b["bbox"]]} fonts={sorted(fonts)[:4]}')
    print("   ", re.sub(r"\s+"," ",txt)[:260])
