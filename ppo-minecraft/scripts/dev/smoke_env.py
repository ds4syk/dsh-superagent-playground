import os, sys, time, json
import numpy as np

MC = '/mnt/storage1/syk/dreamer/envs/mcppo'
os.environ['JAVA_HOME'] = MC
os.environ['TMPDIR'] = '/mnt/storage1/syk/dreamer/tmp'
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
sys.path.insert(0, '/home/syk/labs/Dreamer/repos/dreamerv3')

from embodied.envs.minecraft_flat import MineRLEnv

t0 = time.time()
spec = MineRLEnv(resolution=(64, 64), break_speed=100.0)
env = spec.make()
print('make_env_s', round(time.time() - t0, 2))
print('OBS SPACE:')
for k, v in env.observation_space.spaces.items():
    print(' ', k, v)
print('ACT SPACE:')
for k, v in env.action_space.spaces.items():
    print(' ', k, v)
obs = env.reset()
print('reset_s', round(time.time() - t0, 2))
print('obs keys:', sorted(obs.keys()))
for k in sorted(obs.keys()):
    a = obs[k]
    print(' ', k, type(a).__name__, getattr(a, 'shape', None), getattr(a, 'dtype', None), (a if np.size(a) < 6 else '...'))

# assemble action
act = {
    'camera': (0, 0), 'forward': 0, 'back': 0, 'left': 0, 'right': 0,
    'attack': 0, 'sprint': 0, 'jump': 0, 'sneak': 0,
    'craft': 'none', 'nearbyCraft': 'none', 'nearbySmelt': 'none',
    'place': 'none', 'equip': 'none',
}
t1 = time.time()
for i in range(5):
    obs, r, d, info = env.step(act)
print('step_s', round((time.time() - t1) / 5, 3), 'r', r, 'done', d, 'info', info)
print('is_first', obs['is_first'], 'inventory_log', obs.get('inventory/log'), 'life', obs.get('life_stats/life'))
env.close()
