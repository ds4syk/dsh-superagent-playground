import json,sys
pages=json.load(open("analysis/chars.json"))
pg=int(sys.argv[1]); side=sys.argv[2] if len(sys.argv)>2 else "r"
chars=pages[pg-1]["chars"]
sel=[c for c in chars if (c["x0"]>=296.5 if side=="r" else c["x0"]<296.5)]
rows={}
for c in sel:
    k=round(c["y0"]/1.0)
    rows.setdefault(k,[]).append(c)
for k in sorted(rows):
    cs=sorted(rows[k],key=lambda c:c["x0"])
    t="".join(c["c"] for c in cs)
    sz=sorted({c["size"] for c in cs})
    print(f'{k:5.0f} {cs[0]["x0"]:6.1f}-{cs[-1]["x1"]:6.1f} sz={sz} {t[:90]!r}')
