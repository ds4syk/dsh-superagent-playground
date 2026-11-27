# P2 final push — 70 rounds (2026-11-28)

> Worktree `/mnt/storage1/syk/.superagent-work/wt-final-p2`（branch `final/p2`，base `2ca1ec3`）。不 commit/push。
> 目标：submission-grade P2 论文 = 真实基线（SDPRLayers 约束/slack、dQP/CVXPYLayers、Theseus）、端到端/VLA toy（GPU 被 cyr/wyx 占用时 CPU）、KITTI/UTIAS tilt/frustum、BOP 全表 + ADD/ADD-S、论文去 TODO/refs 核验。
> 分配：sdpr 16 · dqp 14 · e2e 12 · real 14 · paper 10 = 66 子代理轮 + 4 主 worker 集成轮 = **70**。
> 环境：GPU 分区被他人占用（不等待）；CPU 计算在登录节点 `nice -n 10`、BLAS 单线程、单进程 <15 min。数据：`/mnt/storage1/syk/data/{bop,kitti_odom,utias_in_the_dark}`。

## 轮次表

| Round | 线 | 假设 | 改动 | 命令 | 指标 before→after | 判定 | 产物 |
|---|---|---|---|---|---|---|---|
| F1–F16 | sdpr | SDPRLayers 无约束/slack 可作为同实例 baseline | 待填 | 待填 | 待填 | 待定 | 待填 |
| F17–F30 | dqp | dQP/CVXPYLayers + Theseus 可对比 | 待填 | 待填 | 待填 | 待定 | 待填 |
| F31–F42 | e2e | CPU toy 端到端可行；FB 反向可训练 | 待填 | 待填 | 待填 | 待定 | 待填 |
| F43–F56 | real | KITTI/UTIAS + BOP ADD/ADD-S 可测 | 待填 | 待填 | 待填 | 待定 | 待填 |
| F57–F66 | paper | 论文可去 TODO/补基线/核验 refs | 待填 | 待填 | 待填 | 待定 | 待填 |
| M1–M4 | main | 集成/校验/一致性/图 | 待填 | 待填 | 待填 | 待定 | 待填 |

> 负结果/阻塞同样记录；GPU 不可用时的 CPU toy 必须标注 scope。

## 收尾状态（时间上限，2026-11-28 约 01:00）

| 线 | 完成 | 内容 | 阻塞/未完成 |
|---|---|---|---|
| sdpr | **7** | 无约束 Shor + 硬约束 + slack SDP 在 p=8/16/32/64（n=50/12/50/50）与 LM（n=3）上跑通；pose 提取、rigorous Schur cert、表 | p=16 只 n=12；LM 只 n=3；slack 的 M 扫描只部分；无 round_sdpr.md |
| dqp | **4** | dQP（QP 层）前向 p=8 n=2、隐式微分梯度校验（随机 QP ~1e-6，退化点不唯一）、CVXPYLayers 交叉验证 <2e-5、表 | `import theseus` 失败（缺 CHOLMOD）→ Theseus 未跑；dQP 只 p=8 n=2，认证 50% |
| e2e | **3** | toy 数据/backbone/层前向 + 3-epoch smoke（val ADD ~107mm） | FB 反向训练与四头比较未完成 → **partial** |
| real | **8** | BOP ADD/ADD-S（LM n=90 完整，LM-O/YCB 见文件）、KITTI 00/05/07 layer 表、UTIAS loader | KITTI 计数需去重核对；UTIAS 约束表 partial；YCB ADD 可能子集 |
| paper | **2** | draft 已并入部分基线文字（+100/−24）、图更新 | SDPR/dQP/E2E/KITTI/UTIAS 表未插入；仍有 5 处 pending/TODO |
| main | 1 | 集成 + validate 40/40 + `final_push_summary.md` | — |
| **合计** | **25 / 70** | | **45 轮被 15 分钟收尾上限阻塞**（非技术失败；各线代码/结果已保留，可下轮继续） |

关键产出：`results/final_push_summary.md`（SDPR 20 行对照表 + dQP/ADD/KITTI/UTIAS 摘要）、`real_bop_add.{json,md,png}`、`kitti_layer.{json,md,png}`、`baseline_dqp.{json,md}`、`baseline_sdpr_*.json`、`e2e_*.json`、`real_utias.json`；`validate_code.py` 40/40。

## 追加收尾（按“integrate now”指令）

| Round | 线 | 内容 | 产物 |
|---|---|---|---|
| F67 | paper | 新增 §5.7 邻接层族基线（SDPRLayers-style 表 5.9b、dQP/CVXPYLayers-style、Theseus 未跑及原因） | `paper/draft.md` |
| F68 | paper | 新增 §5.8 CPU 端到端 toy（表 5.9c：ADD 106.86→12.69、pen→0、feasible 0→100%） | 同上 |
| F69 | paper | 去除 §5.1/§4.5 的 stale pending 措辞；§5.11 明确列出未跑项；0 TODO/TBD | 同上 |
| F70 | main | 集成全部子 worktree 产物 + validate_code 40/40 + `final_push_summary.md` | 全套 |

**最终统计：28 / 70 轮完成**（sdpr 7、dqp 4、e2e 3、real 8、paper 4、main 2）；其余 42 轮按“stop and integrate now”指令未执行（非技术失败）。核心缺口：Theseus 库缺 CHOLMOD（已文档化）、E2E 训练只 3 epoch、SDPR p16/LM 小样本、KITTI n 待去重。
