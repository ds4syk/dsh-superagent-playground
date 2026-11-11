import fitz, json, sys, collections
doc = fitz.open("source/original.pdf")
print("pages", doc.page_count, "size", doc[0].rect)
tot=collections.Counter()
for pno in range(doc.page_count):
    page=doc[pno]
    d=page.get_text("dict")
    fonts=collections.Counter()
    for b in d["blocks"]:
        for l in b.get("lines",[]):
            for s in l["spans"]:
                fonts[s["font"]]+=1
    print(pno+1, len(d["blocks"]), dict(list(fonts.items())[:6]))
