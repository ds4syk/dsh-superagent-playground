"""把“桥面 + 上/下分层网格桁架”的设计向量转成比赛引擎的 JSON。

设计空间（每个关卡一套）：
- 桥面：y=0 的直线路面，从 A0 到 A_last，每 1.5 米一个桥面节点。
- 桥面以上 2 层、以下 1 层；每层有竖向、水平、两个方向的斜杆。
  最下面一层从底部锚点（A1/A3 等）接到桥面，所以“桥面下桁架”也能生成。
- 每根候选杆件 4 种状态：0 空、1 木杆、2 钢杆、3 钢索。

桥面固定，模型只生成上/下部结构。这个约束让输出天然合法、可批量评测；
比赛引擎本身支持任意节点/杆件/机关，手工试玩不受此约束。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MATERIALS = ["road", "wood", "steel", "cable"]
MAT_NAME = {"road": "路面", "wood": "木杆", "steel": "钢杆", "cable": "钢索"}
MAT_COST_M = {"wood": 40.0, "steel": 160.0, "cable": 70.0}

# 每关：跨径、锚点、行高（含桥面 0）、预算。锚点编号来自 levelN.json。
LEVELS = {
    1: dict(span=15.0, dx=1.5, rows=(-2.0, 0.0, 1.5, 3.0), deck_row=1,
            left_deck="A0", right_deck="A2", left_bottom="A1", right_bottom="A3",
            budget=10000.0, name="小水沟"),
    2: dict(span=30.0, dx=1.5, rows=(-4.0, -2.0, 0.0, 2.0, 4.0), deck_row=2,
            left_deck="A0", right_deck="A4", left_bottom="A1", right_bottom="A5",
            budget=26000.0, name="大峡谷"),
    3: dict(span=28.0, dx=1.4, rows=(-2.0, 0.0, 1.4, 2.8), deck_row=1,
            left_deck="A0", right_deck="A3", left_bottom="A1", right_bottom="A4",
            budget=18000.0, name="早高峰"),
}


@dataclass
class GridLevel:
    level: int
    span: float
    dx: float
    rows: tuple[float, ...]
    left_deck: str
    right_deck: str
    left_bottom: str
    right_bottom: str
    deck_row: int
    budget: float
    name: str

    def __post_init__(self):
        self.ncol = int(round(self.span / self.dx)) + 1   # 节点列数（含两端锚点）
        self.nrow = len(self.rows)
        # deck_row 由配置给出：该行的 y 恒为 0
        self.anchors = {
            (self.deck_row, 0): self.left_deck,
            (self.deck_row, self.ncol - 1): self.right_deck,
            (0, 0): self.left_bottom,
            (0, self.ncol - 1): self.right_bottom,
        }
        # 桥面以下的岸坡内部不能放新节点（岸坡是 solid 多边形）。
        self.blocked = {(r, c) for r in range(self.deck_row)
                        for c in (0, self.ncol - 1)
                        if (r, c) not in self.anchors}
        self._build_candidates()

    # ------------------------------------------------------------
    def _build_candidates(self):
        cand = []

        def ok(kind, r, c):
            p, q = self._endpoints(kind, r, c)
            return p not in self.blocked and q not in self.blocked

        # 竖向：相邻两层之间，跳过“两端都是锚点”的端柱
        for r in range(self.nrow - 1):
            for c in range(self.ncol):
                if (r, c) in self.anchors and (r + 1, c) in self.anchors:
                    continue
                if not ok("v", r, c):
                    continue
                cand.append(("v", r, c))
        # 水平：每一层的相邻节点（桥面层由 road 固定，不重复）
        for r in range(self.nrow):
            if r == self.deck_row:
                continue
            for c in range(self.ncol - 1):
                if not ok("h", r, c):
                    continue
                cand.append(("h", r, c))
        # 斜杆：每个格子两个方向
        for r in range(self.nrow - 1):
            for c in range(self.ncol - 1):
                if ok("d1", r, c):
                    cand.append(("d1", r, c))   # (c, r) -> (c+1, r+1)
                if ok("d2", r, c):
                    cand.append(("d2", r, c))   # (c+1, r) -> (c, r+1)
        self.cand = cand
        self.n_cand = len(cand)
        self.cand_kind = np.array([{"v": 0, "h": 1, "d1": 2, "d2": 3}[k] for k, _, _ in cand],
                                  dtype=np.int64)
        self.cand_row = np.array([r for _, r, _ in cand], dtype=np.int64)
        self.cand_col = np.array([c for _, _, c in cand], dtype=np.int64)

    # ------------------------------------------------------------
    def node_x(self, c: int) -> float:
        return c * self.dx

    def node_y(self, r: int) -> float:
        return self.rows[r]

    def build_json(self, classes: np.ndarray) -> dict:
        classes = np.asarray(classes).astype(int).ravel()
        assert classes.shape[0] == self.n_cand
        active = np.where(classes > 0)[0]

        used = set()
        for i in active:
            kind, r, c = self.cand[i]
            for rc in self._endpoints(kind, r, c):
                if rc in self.anchors or rc[0] == self.deck_row:
                    continue
                used.add(rc)

        order = sorted(used, key=lambda rc: (rc[0], rc[1]))
        nid = {rc: k for k, rc in enumerate(order)}
        nodes = [[round(self.node_x(c), 4), round(self.rows[r], 4)] for r, c in order]
        base = len(nodes)
        deck_id = {c: base + (c - 1) for c in range(1, self.ncol - 1)}
        nodes += [[round(self.node_x(c), 4), 0.0] for c in range(1, self.ncol - 1)]

        def node_at(r: int, c: int):
            if (r, c) in self.anchors:
                return self.anchors[(r, c)]
            if r == self.deck_row:
                return deck_id[c]
            return nid[(r, c)]

        members = []
        prev = self.left_deck
        for c in range(self.ncol - 1):
            nxt = self.right_deck if c == self.ncol - 2 else deck_id[c + 1]
            members.append([prev, nxt, "road"])
            prev = nxt

        for i in active:
            kind, r, c = self.cand[i]
            p, q = self._endpoints(kind, r, c)
            members.append([node_at(*p), node_at(*q), MATERIALS[int(classes[i])]])
        return {"nodes": nodes, "members": members}

    def _endpoints(self, kind: str, r: int, c: int):
        if kind == "v":
            return (r, c), (r + 1, c)
        if kind == "h":
            return (r, c), (r, c + 1)
        if kind == "d1":
            return (r, c), (r + 1, c + 1)
        return (r, c + 1), (r + 1, c)

    # ------------------------------------------------------------
    def to_grid(self, classes: np.ndarray) -> np.ndarray:
        g = np.full((4, self.nrow, self.ncol), -1, dtype=np.int64)
        for i, (kind, r, c) in enumerate(self.cand):
            g[{"v": 0, "h": 1, "d1": 2, "d2": 3}[kind], r, c] = int(classes[i])
        return g

    def from_grid(self, g: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_cand, dtype=np.int64)
        for i, (kind, r, c) in enumerate(self.cand):
            out[i] = g[{"v": 0, "h": 1, "d1": 2, "d2": 3}[kind], r, c]
        return out

    # ------------------------------------------------------------
    def cost_estimate(self, classes: np.ndarray) -> float:
        classes = np.asarray(classes).astype(int)
        cost = 20.0 * (self.ncol - 2) + 150.0 * self.span
        used = set()
        for i in np.where(classes > 0)[0]:
            kind, r, c = self.cand[i]
            p, q = self._endpoints(kind, r, c)
            L = math.hypot(self.node_x(p[1]) - self.node_x(q[1]),
                           self.rows[p[0]] - self.rows[q[0]])
            cost += MAT_COST_M[MATERIALS[int(classes[i])]] * L
            for rc in (p, q):
                if rc in self.anchors or rc[0] == self.deck_row:
                    continue
                used.add(rc)
        cost += 20.0 * len(used)
        return cost


LEVEL_GRID = {k: GridLevel(level=k, **v) for k, v in LEVELS.items()}
