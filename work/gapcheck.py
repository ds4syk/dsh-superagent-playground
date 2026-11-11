import json
pages=json.load(open("analysis/chars.json"))
p1=pages[0]
for lo,hi in ((205,213),(590,600)):
    line=[c for c in p1["chars"] if lo<=c["y0"]<=hi and c["x0"]<297]
    line.sort(key=lambda c:c["x0"])
    print("== y0",lo,"chars",len(line), "".join(c["c"] for c in line)[:100])
    gaps=[(round(b["x0"]-a["x1"],2), a["c"], b["c"]) for a,b in zip(line,line[1:])]
    big=[g for g in gaps if g[0]>1.0]
    print("gaps>1:", big[:25])
