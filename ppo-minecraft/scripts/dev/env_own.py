import faulthandler, os, sys, time
faulthandler.dump_traceback_later(120, repeat=True)
from mcppo.config import load_config
cfg = load_config('configs/smoke.json')
from mcppo.envs.minecraft import MinecraftDiamondEnv
t = time.time()
env = MinecraftDiamondEnv(cfg, seed=0)
print('created', round(time.time() - t, 1), flush=True)
obs = env.reset()
print('reset', round(time.time() - t, 1), flush=True)
print('keys', sorted(obs.keys()), flush=True)
for i in range(5):
    obs, r, d, info = env.step(0)
    print('step', i, r, d, info, flush=True)
env.close()
print('done', flush=True)
