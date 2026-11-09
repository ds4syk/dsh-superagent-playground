import numpy as np
import torch

from mcppo.ppo.model import ActorCritic
from mcppo.ppo.ppo import Rollout, compute_gae


def test_model_forward_shapes():
    model = ActorCritic(n_actions=25, inv_dim=391, equipped_dim=50,
                        frame_channels=12, hidden_size=32, feature_size=64)
    t, b = 3, 2
    image = torch.zeros(t, b, 12, 64, 64)
    inv = torch.zeros(t, b, 391)
    scal = torch.zeros(t, b, 3)
    equ = torch.zeros(t, b, 50)
    hidden = model.initial_hidden(b, torch.device("cpu"))
    logits, values, (h, c) = model(image, inv, scal, equ, hidden)
    assert logits.shape == (t, b, 25)
    assert values.shape == (t, b)
    assert h.shape == (1, b, 32)


def test_gae_matches_manual_values():
    rollout = Rollout(steps=2, envs=1, frame_channels=12, size=64,
                      inv_dim=4, equipped_dim=2, chunk_len=2)
    rollout.rewards[0, 0] = 1.0
    rollout.rewards[1, 0] = 1.0
    rollout.values[0, 0] = 0.0
    rollout.values[1, 0] = 0.0
    rollout.dones[0, 0] = 0.0
    rollout.dones[1, 0] = 1.0
    adv, ret = compute_gae(rollout, torch.zeros(1), gamma=0.5, lam=0.5)
    # t=1: delta = 1 + 0 - 0 = 1, adv=1, ret=1
    # t=0: delta = 1 + 0.5*0 - 0 = 1, adv = 1 + 0.5*0.5*1 = 1.25
    assert abs(adv[1, 0].item() - 1.0) < 1e-6
    assert abs(adv[0, 0].item() - 1.25) < 1e-5
    assert abs(ret[0, 0].item() - 1.25) < 1e-5


def test_model_accepts_reset_mask():
    model = ActorCritic(n_actions=25, inv_dim=8, equipped_dim=4,
                        frame_channels=12, hidden_size=16, feature_size=32)
    t, b = 4, 2
    image = torch.zeros(t, b, 12, 64, 64)
    inv = torch.zeros(t, b, 8)
    scal = torch.zeros(t, b, 3)
    equ = torch.zeros(t, b, 4)
    hidden = model.initial_hidden(b, torch.device("cpu"))
    mask = torch.zeros(t, b)
    mask[1, 0] = 1.0
    logits, values, _ = model(image, inv, scal, equ, hidden, reset_mask=mask)
    assert logits.shape == (t, b, 25)
    assert values.shape == (t, b)


def test_ppo_update_runs_with_midchunk_done():
    import torch
    from mcppo.config import DEFAULTS
    from mcppo.ppo.ppo import Rollout, ppo_update
    model = ActorCritic(n_actions=25, inv_dim=8, equipped_dim=4,
                        frame_channels=12, hidden_size=16, feature_size=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    roll = Rollout(steps=4, envs=1, frame_channels=12, size=64,
                   inv_dim=8, equipped_dim=4, chunk_len=2, hidden_size=16)
    roll.dones[1, 0] = 1.0
    roll.chunk_h.zero_(); roll.chunk_c.zero_()
    adv = torch.zeros(4, 1); ret = torch.zeros(4, 1)
    cfg = dict(DEFAULTS["ppo"]); cfg["epochs"] = 1; cfg["minibatch_chunks"] = 1
    stats = ppo_update(model, opt, roll, adv, ret, cfg, torch.device("cpu"), 0.01)
    assert stats["n_updates"] > 0
