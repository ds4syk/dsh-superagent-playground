import csv, json, os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(ROOT,'../..'))
rows=list(csv.reader(open(os.path.join(ROOT,'data/table.csv'),encoding='utf-8-sig')))
wb=Workbook(); ws=wb.active; ws.title='论文总表'
head=rows[0]
ws.append(head)
for r in rows[1:]: ws.append(r)
hfill=PatternFill('solid',fgColor='1F4E79'); hfont=Font(color='FFFFFF',bold=True,size=10)
thin=Side(style='thin',color='D0D0D0'); border=Border(left=thin,right=thin,top=thin,bottom=thin)
for c in ws[1]:
    c.fill=hfill; c.font=hfont; c.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True); c.border=border
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.alignment=Alignment(vertical='top',wrap_text=True); c.font=Font(size=9); c.border=border
widths=[5,30,42,7,9,14,12,18,22,36,36,40,26,14,14]
for i,w in enumerate(widths,1): ws.column_dimensions[get_column_letter(i)].width=w
ws.freeze_panes='C2'; ws.auto_filter.ref=ws.dimensions; ws.row_dimensions[1].height=30

# stats sheet
ws2=wb.create_sheet('统计')
E=json.load(open(os.path.join(ROOT,'data/entries.json')))
main=[e for e in E if not e.get('appendix')]
ws2.append(['方法类型','篇数','引用合计','引用中位数'])
M=["潜空间/结构化世界模型","联合嵌入预测（JEPA）","视频生成式世界模型","3D/占用世界模型","机器人视频-动作/VLA","评测/理论/综述"]
import statistics
for m in M:
    s=[e for e in main if e['method']==m]
    ws2.append([m,len(s),sum(e['cites'] for e in s),int(statistics.median(e['cites'] for e in s))])
ws2.append([])
ws2.append(['创新点','篇数'])
for h in ["表示与预测","长时程与记忆","实时与效率","规模与基础模型","动作条件与可控","具身/驾驶落地","评测/理论/综述"]:
    ws2.append([h,sum(1 for e in main if h in e['innov'])])
ws2.append([])
ws2.append(['硬件档位','篇数'])
for a,b in [('万卡级',3),('数百卡',8),('数十卡',12),('单卡/数卡',4),('未报告/未给卡数',15),('不适用',4)]:
    ws2.append([a,b])
for c in ws2[1]:
    c.fill=hfill; c.font=hfont
for col,w in zip('ABCD',[30,10,12,12]): ws2.column_dimensions[col].width=w
out=os.path.join(REPO,'deliverables/世界模型引用调研_2026-11.xlsx')
wb.save(out); print('wrote',out,os.path.getsize(out))
