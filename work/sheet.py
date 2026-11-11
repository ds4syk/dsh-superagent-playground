import json, sys, os
prep=json.load(open("analysis/prepared.json"))
os.makedirs("work/sheet", exist_ok=True)
pages=[int(x) for x in sys.argv[1].split(",")]
for p in prep:
    if p["page"] not in pages: continue
    lines=[f"# PAGE {p['page']}\n"]
    for u in p["units"]:
        if u["role"]=="keep": continue
        mathnote=", ".join(f'{m["i"]}={m["text"]!r}' for m in u["math"])
        lines.append(f'## {u["id"]} [{u["role"]}] sz={u["size"]} bbox={u["bbox"]}\n')
        if mathnote: lines.append(f"MATH: {mathnote}\n")
        lines.append(u["text"]+"\n")
    open(f"work/sheet/p{p['page']:02d}.md","w").write("\n".join(lines))
    print("wrote", f"work/sheet/p{p['page']:02d}.md", len("\n".join(lines)))
