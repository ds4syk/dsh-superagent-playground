"""MC-PPO: PPO on real Minecraft (MineRL / Malmo) with a hand-written dense reward.

The environment is the same simplified flat-world Minecraft task used by
DreamerV3 (``embodied/envs/minecraft_flat.py`` in the official repository):
Minecraft 1.11.2 through the patched MineRL 0.4.4 wheel, 64x64 RGB, and a
25-way abstract action space (movement / camera / craft / smelt).
"""

__version__ = "0.1.0"
