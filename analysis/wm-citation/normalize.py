import json, sys
sys.path.insert(0,'.')
from data.entries import ENTRIES

M1="潜空间/结构化世界模型"; M2="联合嵌入预测（JEPA）"; M3="视频生成式世界模型"
M4="3D/占用世界模型"; M5="机器人视频-动作/VLA"; M6="评测/理论/综述"

H=["表示与预测","长时程与记忆","实时与效率","规模与基础模型","动作条件与可控","具身/驾驶落地","评测/理论/综述"]

MAP={
 "dreamer4":(M1,[H[1],H[2]]),
 "tdmpc2":(M1,[H[0],H[3]]),
 "swmhuman":(M1,[H[0],H[5]]),
 "lewm":(M2,[H[0],H[2]]),
 "vjepa21":(M2,[H[0]]),
 "vlajepa":(M2,[H[0],H[5]]),
 "dinowm":(M2,[H[0]]),
 "flare":(M5,[H[0],H[5]]),
 "iwm":(M2,[H[0]]),
 "lcm":(M2,[H[0],H[3]]),
 "platonic":(M6,[H[6]]),
 "cosmos":(M3,[H[3],H[4]]),
 "cosmos3":(M3,[H[3],H[0]]),
 "cosmosp25":(M3,[H[3],H[4]]),
 "cosmospolicy":(M5,[H[4],H[5]]),
 "gamengen":(M3,[H[2]]),
 "wm1m":(M3,[H[1],H[3]]),
 "lingbot":(M3,[H[2],H[1]]),
 "mg3":(M3,[H[2],H[1]]),
 "mineworld":(M3,[H[2]]),
 "worldmem":(M3,[H[1]]),
 "spatialmem":(M3,[H[1]]),
 "relic":(M3,[H[1],H[2]]),
 "pandora":(M3,[H[4]]),
 "nwm":(M3,[H[4],H[5]]),
 "yume":(M3,[H[2],H[4]]),
 "worlddreamer":(M3,[H[0]]),
 "gamegenx":(M3,[H[4]]),
 "worldsimbench":(M6,[H[6]]),
 "howfar":(M6,[H[6]]),
 "drivedreamer":(M3,[H[5],H[4]]),
 "drivedreamer2":(M3,[H[4],H[5]]),
 "vista":(M3,[H[5],H[4]]),
 "drivewm":(M3,[H[5],H[4]]),
 "occworld":(M4,[H[0],H[5]]),
 "dreamzero":(M5,[H[5],H[2],H[4]]),
 "worldvla":(M5,[H[0],H[5]]),
 "gr2":(M5,[H[3],H[5]]),
 "univla":(M5,[H[0],H[5]]),
 "dreamdojo":(M5,[H[3],H[5]]),
 "mimicvideo":(M5,[H[0],H[5]]),
 "ctrlworld":(M5,[H[5],H[4]]),
 "irasim":(M5,[H[5],H[4]]),
 "unifiedvla":(M5,[H[0],H[5]]),
 "gr1":(M5,[H[3],H[5]]),
 "survworld":(M6,[H[6]]),
 "survad":(M6,[H[6]]),
 "survmbrl":(M6,[H[6]]),
 "unisim":(M3,[H[3],H[5]]),
 "unipi":(M3,[H[4],H[5]]),
 "voltron":(M2,[H[0],H[5]]),
}
out=[]
for e in ENTRIES:
    if e['id'] not in MAP: continue
    m,inn=MAP[e['id']]
    e['method']=m; e['innov']=inn
    if e['id']=='voltron':  # 仅背景/邻近，不入主表
        continue
    out.append(e)
json.dump(out,open('data/entries.json','w'),ensure_ascii=False,indent=1)
print("entries",len(out),"main",sum(1 for e in out if not e.get('appendix')),"appendix",sum(1 for e in out if e.get('appendix')))
import collections
for m in [M1,M2,M3,M4,M5,M6]:
    sel=[e for e in out if e['method']==m]
    print(m,len(sel),sum(e['cites'] for e in sel))
for h in H:
    print(h,sum(1 for e in out if h in e['innov']))
