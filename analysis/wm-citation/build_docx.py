import glob, os
import pypandoc
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(ROOT,'../..'))
kb=os.path.join(REPO,'kb/wm-citation')
out=os.path.join(REPO,'deliverables/世界模型引用调研_2026-11.docx')
files=sorted(glob.glob(os.path.join(kb,'*.md')))
cover="""![](assets/2026-11-11-wm-citation-survey.png){width=21cm}

*图 0：一页速览。46 篇主表 + 4 篇附录，按方法类型与创新点双重归类；数据截至 2026-11-11。*
"""

text=cover+"\n\n"+"\n\n".join(open(p,encoding='utf-8').read() for p in files)
pypandoc.convert_text(text,'docx',format='markdown',outputfile=out,
    extra_args=['--reference-doc',os.path.join(ROOT,'reference.docx'),
                f'--resource-path={kb}','--metadata','title=世界模型三系引用调研（2026-11）'])

doc=Document(out)
sec=doc.sections[0]
usable=sec.page_width - sec.left_margin - sec.right_margin

MAIN_RATIO=[0.028,0.180,0.042,0.185,0.215,0.220,0.130]
def set_table_width(tbl, ratios):
    n=len(ratios)
    tbl.autofit=False
    tblPr=tbl._tbl.tblPr
    for tag in ('w:tblW','w:tblLayout'):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    tw=OxmlElement('w:tblW'); tw.set(qn('w:type'),'dxa'); tw.set(qn('w:w'),str(int(usable/635)))
    layout=OxmlElement('w:tblLayout'); layout.set(qn('w:type'),'fixed')
    # schema order: tblStyle, tblW, ..., tblLayout, ..., tblLook
    style=tblPr.find(qn('w:tblStyle'))
    anchor=style if style is not None else None
    if anchor is not None:
        anchor.addnext(tw)
    else:
        tblPr.insert(0,tw)
    look=tblPr.find(qn('w:tblLook'))
    if look is not None:
        look.addprevious(layout)
    else:
        tblPr.append(layout)
    grid=tbl._tbl.find(qn('w:tblGrid'))
    if grid is not None:
        for gc in list(grid):
            grid.remove(gc)
        for r in ratios:
            gc=OxmlElement('w:gridCol'); gc.set(qn('w:w'),str(int(usable*r/635))); grid.append(gc)
    for row in tbl.rows:
        for i,cell in enumerate(row.cells[:n]):
            cell.width=int(usable*ratios[i])

for tbl in doc.tables:
    ncols=len(tbl.columns)
    try:
        hdr=[c.text.strip() for c in tbl.rows[0].cells]
    except Exception:
        hdr=[]
    if ncols==7 and hdr and hdr[0]=='#':
        set_table_width(tbl, MAIN_RATIO)
    else:
        set_table_width(tbl, [1.0/ncols]*ncols)
doc.save(out)
print('wrote',out,os.path.getsize(out),'tables',len(doc.tables))
