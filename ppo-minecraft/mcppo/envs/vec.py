"""Subprocess vectorised Minecraft envs.

Each worker owns one Malmo/Minecraft instance, so the JVM work happens in
parallel processes.  Observations (64x64x3 uint8 + a 391-float inventory) are
small enough to pickle over a pipe; the parent only does tensor conversion.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np


def _worker(remote, parent_remote, cfg: Dict[str, Any], rank: int, seed: int) -> None:
    parent_remote.close()
    # Keep Java/MineRL chatter away from the training log.
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    try:
        from .minecraft import MinecraftDiamondEnv
        env = MinecraftDiamondEnv(cfg, seed=seed + rank)
        remote.send(("ready", {"n_actions": env.n_actions, "obs_dim": env.obs_dim,
                               "equipped_dim": env.equipped_dim}))
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(int(data))
                if done:
                    eng = env.reward
                    info["episode"] = {
                        "return": float(eng.episode_return),
                        "length": int(eng.t),
                        "score": int(eng.score()),
                        "milestones": sorted(eng.milestones),
                        "components": {k: float(v) for k, v in eng.components.items()},
                    }
                    obs = env.reset()
                remote.send(("ok", (obs, reward, done, info)))
            elif cmd == "reset":
                remote.send(("ok", env.reset()))
            elif cmd == "close":
                env.close()
                remote.send(("ok", None))
                break
            else:
                raise RuntimeError(f"unknown command {cmd}")
    except EOFError:
        pass
    except Exception:  # pragma: no cover - surfaced in the parent
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


class SubprocVecEnv:
    def __init__(self, cfg: Dict[str, Any], num_envs: int, seed: int = 0):
        self.cfg = cfg
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        self.parents: List[Any] = []
        self.workers: List[Any] = []
        for rank in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, parent, cfg, rank, seed),
                               daemon=True, name=f"mc-env-{rank}")
            proc.start()
            child.close()
            self.parents.append(parent)
            self.workers.append(proc)
        meta = [self._recv(i) for i in range(num_envs)]
        self.n_actions = meta[0]["n_actions"]
        self.obs_dim = meta[0]["obs_dim"]
        self.equipped_dim = meta[0]["equipped_dim"]
        assert all(m["n_actions"] == self.n_actions for m in meta)

    def _recv(self, i: int) -> Any:
        try:
            status, payload = self.parents[i].recv()
        except EOFError as exc:
            raise RuntimeError(f"env worker {i} died during startup") from exc
        if status == "error":
            raise RuntimeError(f"env worker {i} failed:\n{payload}")
        return payload

    def _send(self, cmd: str, data: Any = None) -> None:
        for parent in self.parents:
            parent.send((cmd, data))

    def reset(self) -> List[Dict[str, np.ndarray]]:
        self._send("reset")
        return [self._recv(i) for i in range(self.num_envs)]

    def step(self, actions: np.ndarray):
        for i, parent in enumerate(self.parents):
            parent.send(("step", int(actions[i])))
        results = [self._recv(i) for i in range(self.num_envs)]
        obs = [r[0] for r in results]
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        dones = np.asarray([r[2] for r in results], dtype=np.float32)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self) -> None:
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except Exception:
                pass
        for parent, proc in zip(self.parents, self.workers):
            try:
                parent.close()
            except Exception:
                pass
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()
