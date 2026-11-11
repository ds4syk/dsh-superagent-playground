import sys, importlib.util
spec=importlib.util.spec_from_file_location("u2","work/units2.py"); u2=importlib.util.module_from_spec(spec); spec.loader.exec_module(u2)
import pymupdf as fitz, re
doc=fitz.open("source/original.pdf")
page=doc[int(sys.argv[1])-1]
chars=u2.page_chars(page); rows=u2.make_rows(chars)
for r in rows:
    t,m=u2.row_text(r)
    r["text"]=t
for r in rows:
    r["col"]=0 if (r["x0"]<u2.GUTTER and not (r["x0"]<u2.GUTTER<r["x1"])) else (1 if r["x0"]>=u2.GUTTER else None)
    if r["x0"]<u2.GUTTER-40 and r["x1"]>u2.GUTTER+40: r["col"]=-1
rows=[r for r in rows if r["col"] is not None]
rows.sort(key=lambda r:(r["col"] if r["col"]>=0 else 0.5, r["y0"], r["x0"]))
for i,r in enumerate(rows):
    print(f'{i} c{r["col"]} y0={r["y0"]:.1f} y1={r["y1"]:.1f} x0={r["x0"]:.1f} x1={r["x1"]:.1f} {r["text"][:70]!r}')
