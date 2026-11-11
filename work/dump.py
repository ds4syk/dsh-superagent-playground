import json,sys
pages=json.load(open("analysis/units.json"))
sel=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [1]
width=int(sys.argv[2]) if len(sys.argv)>2 else 400
for p in pages:
    if p["page"] not in sel: continue
    print("="*20,"PAGE",p["page"])
    for u in p["units"]:
        print(f'[{u["id"]}] {("math" if u.get("display") else "text")} c{u["col"]} bbox={u["bbox"]} sz={u["size"]} b={int(u["bold"])} i={int(u["italic"])} nm={len(u["math"])}')
        print("   ", u.get("text","")[:width])
