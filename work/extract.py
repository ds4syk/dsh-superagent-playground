import pymupdf as fitz, json, re, sys, statistics, os
from collections import defaultdict

PROSE_FONT = re.compile(r'^(Times|Helvetica|Courier|Arial|NimbusRom|NimbusSan)', re.I)
MATH_FONT  = re.compile(r'^(CM|MT|EU|MSBM|MSAM|CSDR|RMT|MT2H?RB|Math)', re.I)

def span_class(font, text):
    if MATH_FONT.match(font): return 'm'
    if PROSE_FONT.match(font): return 'p'
    # unknown: decide by glyph content
    if re.search(r'[A-Za-z]{2}', text): return 'p'
    return 'm'

doc = fitz.open("source/original.pdf")
pages = []
for pno in range(doc.page_count):
    page = doc[pno]
    chars = []
    raw = page.get_text("rawdict")
    for b in raw["blocks"]:
        if b["type"] != 0: continue
        for l in b["lines"]:
            for s in l["spans"]:
                font = s["font"]; size = s["size"]; flags = s["flags"]
                cls = span_class(font, "".join(c["c"] for c in s["chars"]))
                for c in s["chars"]:
                    chars.append({
                        "x0": c["bbox"][0], "y0": c["bbox"][1],
                        "x1": c["bbox"][2], "y1": c["bbox"][3],
                        "bx": c["origin"][0], "by": c["origin"][1],
                        "c": c["c"], "font": font, "size": round(size,2),
                        "bold": bool(flags & 16), "italic": bool(flags & 2),
                        "cls": cls,
                    })
    pages.append({"page": pno+1, "chars": chars, "images": [list(b["bbox"]) for b in raw["blocks"] if b["type"]==1]})
json.dump(pages, open("analysis/chars.json","w"), ensure_ascii=False)
print("chars", sum(len(p["chars"]) for p in pages))
