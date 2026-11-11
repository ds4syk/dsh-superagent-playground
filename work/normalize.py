import json, glob, re
prep=json.load(open("analysis/prepared.json"))
zh={}
for f in sorted(glob.glob("analysis/zh_pages/*.json")): zh.update(json.load(open(f)))

FOOTER="经授权许可，仅限香港科技大学使用。2022 年 3 月 31 日 10:40:21 UTC 从 IEEE Xplore 下载。适用使用限制。"
HEADER="本文已被接收并将发表于本刊，内容以最终排版为准（允许分页）。"
SUP={"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
SAFE={"ℝ":"R","ℂ":"C","ℕ":"N","ℤ":"Z","⊤":"T","⟨":"〈","⟩":"〉"}
for p in prep:
    for u in p["units"]:
        uid=u["id"]; role=u["role"]; b=u["bbox"]
        if role=="keep": zh.pop(uid,None); continue
        if role=="footer":
            zh[uid]=FOOTER; continue
        if role=="header" and b[1]<20:
            zh[uid]=HEADER; continue
        t=zh.get(uid)
        if t is None: continue
        for a,bb in SUP.items(): t=t.replace(a,bb)
        for a,bb in SAFE.items(): t=t.replace(a,bb)
        # placeholder sanity: drop ids not present in unit math
        valid={m["i"] for m in u["math"]}
        def fix(m):
            i=int(m.group(1))
            return m.group(0) if i in valid else ""
        t=re.sub(r"⟦(\d+)⟧", fix, t)
        t=re.sub(r"[ \t]+"," ",t).strip()
        zh[uid]=t
json.dump(zh,open("analysis/zh_final.json","w"),ensure_ascii=False,indent=1)
print("final entries",len(zh))
# report units with no translation
missing=[u["id"] for p in prep for u in p["units"] if u["kind"]=="text" and u["role"]!="keep" and u["id"] not in zh]
print("untranslated text units:",len(missing), missing[:12])
left=[u for p in prep for u in p["units"] if zh.get(u["id"]) and "⟦" in zh[u["id"]]]
print("units with leftover placeholders:",len(left))
