import pymupdf as fitz
doc=fitz.open(); page=doc.new_page(width=400,height=200)
html='''<p>这是一个测试：二次位姿估计问题（QPEP）的全局最优解，<b>加粗</b><i>斜体</i>与 H<sup>2</sup>O。</p>'''
css='''@font-face{font-family:cjkserif;src:url(NotoSerifCJK-Regular.ttc);}
p{font-family:cjkserif;font-size:10pt;text-align:justify;}'''
try:
    r=page.insert_htmlbox(fitz.Rect(20,20,380,180), html, css=css, archive=fitz.Archive("/usr/share/fonts/opentype/noto/"))
    print("ttc ok", r)
except Exception as e:
    print("ttc fail", e)
try:
    r=page.insert_htmlbox(fitz.Rect(20,100,380,190), '<p>内置字体测试 二次位姿估计</p>', css='p{font-family:"cjk";font-size:10pt}', overlay=True)
    print("builtin", r)
except Exception as e:
    print("builtin fail", e)
pix=page.get_pixmap(dpi=150); pix.save("work/t_inbox.png")
print("ink", sum(1 for b in pix.samples if b<250))
