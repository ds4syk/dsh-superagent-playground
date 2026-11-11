import json, glob, os
from fontTools.ttLib import TTCollection
from fontTools import subset

zh=set()
for f in glob.glob("analysis/zh_pages/*.json"):
    for v in json.load(open(f)).values():
        zh.update(v)
base=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
punct=set(" .,;:!?()[]{}<>/\\|-–—_+=*&%$#@'\"“”‘’·…、。，；：？！（）《》【】〔〕—～≈≤≥≠±×÷°µαβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∈⊗⊕⊤⊥∥⟨⟩ℝℂℕℤ√∑∏∫∂∇′″→←↔⇒⇔⊂⊃⊆⊇∪∩∧∨¬∀∃∞")
chars=base|punct|zh
chars={c for c in chars if ord(c)>=32}
print("chars", len(chars))
os.makedirs("work/fonts", exist_ok=True)
def make(src, face_hint, out):
    coll=TTCollection(src)
    idx=None
    for i,f in enumerate(coll.fonts):
        name=" ".join(str(v) for v in f["name"].names if v.nameID in (1,4) for v in [v.toUnicode()])
        if "SC" in name or "Simplified" in name:
            idx=i; break
    if idx is None: idx=0
    font=coll.fonts[idx]
    # ensure all chars exist
    cmap=font.getBestCmap()
    missing=sorted(c for c in chars if ord(c) not in cmap)
    print(os.path.basename(out), "face", idx, "missing", len(missing), "".join(missing[:20]))
    opts=subset.Options()
    opts.layout_features=['*']
    opts.notdef_outline=True
    opts.recalc_bounds=True
    opts.drop_tables += ['DSIG']
    ss=subset.Subsetter(options=opts)
    ss.populate(unicodes=[ord(c) for c in chars])
    ss.subset(font)
    font.save(out)
    print("saved", out, round(os.path.getsize(out)/1e6,2),"MB")
make("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc","",  "work/fonts/NotoSerifSC-r.ttf")
make("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc","",     "work/fonts/NotoSerifSC-b.ttf")
