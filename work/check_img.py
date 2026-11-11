import pymupdf as fitz
doc=fitz.open("source/original.pdf")
for pno in range(doc.page_count):
    page=doc[pno]
    imgs=page.get_images(full=True)
    blocks=[b for b in page.get_text("dict")["blocks"] if b["type"]==1]
    print(pno+1, "images",len(imgs), "imgblocks",[ [round(v) for v in b["bbox"]] for b in blocks], "drawings", len(page.get_drawings()))
