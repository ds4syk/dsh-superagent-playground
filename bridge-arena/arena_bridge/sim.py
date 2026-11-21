"""调用比赛官方引擎（third_party/meow-ai-arena/bridge）评测设计。

官方前端 workspace/bridge.py 负责校验、算造价、调 bridge_sim.exe。
本模块只做“包装 + 并行”，保证模型和手工设计用的是同一个确定性引擎。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_DIR = os.path.join(HERE, "..", "third_party", "meow-ai-arena", "bridge", "workspace")
BRIDGE_DIR = os.path.abspath(BRIDGE_DIR)
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)
os.environ.setdefault("BRIDGE_NO_ARCHIVE", "1")

import bridge  # noqa: E402

REF_COST = {1: 3509.84, 2: 10616.73, 3: 7967.26, 4: 12342.52, 5: 8216.12, 6: 16709.27}


@dataclass
class EvalResult:
    legal: bool
    passed: bool
    cost: float
    score: float
    reason: str
    scenes: tuple

    def as_dict(self):
        return dict(legal=self.legal, passed=self.passed, cost=self.cost,
                    score=self.score, reason=self.reason, scenes=self.scenes)


class Evaluator:
    def __init__(self, level: int):
        self.level = level
        self.lv = bridge.load_level(level)
        self.samples = [bridge.load_json(p) for p in bridge.sample_paths(self.lv)]

    def check(self, design: dict) -> tuple[bool, float, str]:
        try:
            P = bridge.parse_design(self.lv, design)
        except Exception as e:  # DesignError
            return False, 0.0, str(e)[:160]
        return True, float(P["cost"]), ""

    def evaluate(self, design: dict) -> EvalResult:
        try:
            P = bridge.parse_design(self.lv, design)
        except Exception as e:
            return EvalResult(False, False, 0.0, 0.0, str(e)[:160], ())
        cost = float(P["cost"])
        scenes = []
        passed = True
        reason = ""
        for sc in self.samples:
            res, _ = bridge.simulate(self.lv, P, sc)
            ok = bool(res.get("ok"))
            scenes.append(dict(name=sc.get("name", ""), ok=ok,
                               broken=int(res.get("broken", 0)),
                               anomaly=int(res.get("anomaly", 0) or 0),
                               vehicles=[(int(v["status"]), int(v["reason"]))
                                         for v in res.get("vehicles", [])]))
            if not ok:
                passed = False
                if int(res.get("anomaly", 0) or 0):
                    reason = "物理异常"
                elif int(res.get("broken", 0)) > 0:
                    reason = "杆件断裂/车辆失败"
                else:
                    reason = "车辆未通过"
                break
        if cost > self.lv["budget"]:
            passed = False
            reason = "超预算"
        score = round(100.0 * REF_COST[self.level] / cost, 2) if (passed and cost > 0) else 0.0
        return EvalResult(True, passed, cost, score, reason, tuple(scenes))


# ---------------------------------------------------------------- 并行
_WORKER: dict[int, Evaluator] = {}


def _init_worker(level: int):
    _WORKER.clear()
    _WORKER[level] = Evaluator(level)


def _work(item):
    level, design = item
    return _WORKER[level].evaluate(design).as_dict()


def evaluate_many(level: int, designs: Iterable[dict], workers: int = 16) -> list[dict]:
    """多进程评测；每个进程持有自己的 Evaluator（引擎 exe 可被多个进程同时调用）。"""
    items = [(level, d) for d in designs]
    if workers <= 1:
        _init_worker(level)
        return [_work(it) for it in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(level,)) as pool:
        return pool.map(_work, items, chunksize=max(1, len(items) // (workers * 4)))
