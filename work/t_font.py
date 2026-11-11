import pymupdf as fitz
for path,label in [("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc","NotoCJK"),
                   ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf","DejaVuSerif")]:
    f=fitz.Font(fontfile=path)
    missing=[ch for ch in "∈⊗∥⪰⊤°±×÷≤≥≠≈∑∏∫∂∇Ααβγπσω′″·—–ℝℂℕ…“”‘’" if not f.has_glyph(ord(ch))]
    print(label,"missing:",missing, "name:",f.name)
