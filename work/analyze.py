import json, re, statistics, sys
from collections import defaultdict

pages = json.load(open("analysis/chars.json"))
GUTTER = 296.5
COL_L = (30, 296.5); COL_R = (296.5, 560)

def cluster_rows(chars):
    chars = sorted(chars, key=lambda c: (c["by"], c["x0"]))
    rows = []
    for c in chars:
        if rows and abs(c["by"] - rows[-1]["by"]) <= 1.4:
            rows[-1]["chars"].append(c)
        else:
            rows.append({"by": c["by"], "chars": [c]})
    out = []
    for r in rows:
        # split by column
        left = [c for c in r["chars"] if c["x0"] < GUTTER]
        right = [c for c in r["chars"] if c["x0"] >= GUTTER]
        for part in (left, right):
            if not part: continue
            part.sort(key=lambda c: c["x0"])
            out.append({"by": r["by"], "chars": part,
                        "x0": min(c["x0"] for c in part), "x1": max(c["x1"] for c in part),
                        "y0": min(c["y0"] for c in part), "y1": max(c["y1"] for c in part)})
    out.sort(key=lambda r: (0 if r["x0"] < GUTTER else 1, r["by"], r["x0"]))
    return out

def runs_of(row):
    runs = []
    for c in row["chars"]:
        cls = c["cls"]
        if runs and runs[-1]["cls"] == cls and c["x0"] - runs[-1]["x1"] <= 2.6 and abs(c["size"]-runs[-1]["chars"][-1]["size"])<1.5:
            r = runs[-1]; r["chars"].append(c); r["x1"] = max(r["x1"], c["x1"]); r["y1"]=max(r["y1"],c["y1"]); r["y0"]=min(r["y0"],c["y0"])
        else:
            runs.append({"cls": cls, "chars": [c], "x0": c["x0"], "x1": c["x1"], "y0": c["y0"], "y1": c["y1"]})
    for r in runs:
        r["text"] = "".join(c["c"] for c in r["chars"])
        sizes = [c["size"] for c in r["chars"]]
        r["size"] = statistics.median(sizes)
        r["bold"] = sum(c["bold"] for c in r["chars"]) > len(r["chars"])/2
        r["italic"] = sum(c["italic"] for c in r["chars"]) > len(r["chars"])/2
        r["font"] = statistics.mode([c["font"] for c in r["chars"]])
    return runs

def row_info(row):
    runs = runs_of(row)
    prose = [r for r in runs if r["cls"]=="p"]
    math = [r for r in runs if r["cls"]=="m"]
    ptext = "".join(r["text"] for r in prose)
    # reconstruct ordered text with placeholders
    out=[]; mi=0
    for r in runs:
        if r["cls"]=="p": out.append(r["text"])
        else:
            out.append(f"⟦{mi}⟧"); mi+=1
    joined="".join(out)
    pchars = sum(len(r["text"]) for r in prose)
    mchars = sum(len(r["text"]) for r in math)
    return {"runs": runs, "prose": prose, "math": math, "text": joined,
            "pchars": pchars, "mchars": mchars, "row": row}

def is_display(row, info):
    # rows dominated by math, centered short, no long prose run
    if info["pchars"] >= 12: return False
    if info["mchars"] >= 3: return True
    return False

report = {}
units_all = []
for pg in pages:
    rows = cluster_rows(pg["chars"])
    infos = [row_info(r) for r in rows]
    col_left = {0: 37.9, 1: 300.9}
    units = []
    cur = None
    for info, row in zip(infos, rows):
        col = 0 if row["x0"] < GUTTER else 1
        info["col"] = col
        disp = is_display(row, info)
        newpara = False
        if disp:
            newpara = True
        if cur is None:
            newpara = True
        else:
            gap = row["y0"] - cur["y1"]
            indent = row["x0"] - col_left[col]
            size_prev = cur["size"]; size_now = statistics.median([c["size"] for c in row["chars"]])
            bold_prev = cur["bold"]; bold_now = sum(c["bold"] for c in row["chars"]) > len(row["chars"])/2
            if col != cur["col"]: newpara = True
            elif gap > 4.2: newpara = True
            elif indent > 5.5: newpara = True
            elif abs(size_now - size_prev) > 0.6: newpara = True
            elif bold_now != bold_prev: newpara = True
            elif cur["last_x1"] < col_left[col] + 250 and gap > 2.2: newpara = True
        if newpara:
            if cur: units.append(cur)
            cur = {"page": pg["page"], "col": col, "rows": [row], "infos": [info],
                   "x0": row["x0"], "y0": row["y0"], "x1": row["x1"], "y1": row["y1"],
                   "size": statistics.median([c["size"] for c in row["chars"]]),
                   "bold": sum(c["bold"] for c in row["chars"]) > len(row["chars"])/2,
                   "italic": sum(c["italic"] for c in row["chars"]) > len(row["chars"])/2,
                   "display": disp}
        else:
            cur["rows"].append(row); cur["infos"].append(info)
            cur["x0"] = min(cur["x0"], row["x0"]); cur["x1"] = max(cur["x1"], row["x1"])
            cur["y0"] = min(cur["y0"], row["y0"]); cur["y1"] = max(cur["y1"], row["y1"])
        cur["last_x1"] = row["x1"]
    if cur: units.append(cur)
    # build text with dehyphenation
    for u in units:
        lines=[]
        for info in u["infos"]:
            lines.append(info["text"])
        txt = ""
        for i,l in enumerate(lines):
            l = l.strip()
            if txt.endswith("-") and re.match(r"[a-z]", l):
                txt = txt[:-1] + l
            elif txt.endswith("-\u200b"):
                txt = txt[:-1] + l
            else:
                txt = (txt + " " + l).strip() if txt else l
        u["text"] = re.sub(r"\s+", " ", txt)
        u["n_math"] = sum(len(i["math"]) for i in u["infos"])
        u["math_runs"] = [(m["x0"], m["y0"], m["x1"], m["y1"]) for i in u["infos"] for m in i["math"]]
    units_all.append({"page": pg["page"], "units": units})

if __name__ == "__main__":
    outp = int(sys.argv[1]) if len(sys.argv)>1 else 3
    for pg in units_all:
        if pg["page"] > outp: break
        print("="*30, "PAGE", pg["page"])
        for u in pg["units"]:
            print(f'[{u["page"]}-{u["col"]}] box=({u["x0"]:.0f},{u["y0"]:.0f},{u["x1"]:.0f},{u["y1"]:.0f}) size={u["size"]} b={int(u["bold"])} i={int(u["italic"])} nmath={u["n_math"]} : {u["text"][:150]!r}')
