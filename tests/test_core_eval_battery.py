import torch

from core.base import VAGIConfig, VAGICore


def test_task_embedding_affects_logits() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
        use_task_embedding=True,
        task_vocab_size=2,
    )
    model = VAGICore(cfg)
    model.backbone.task_embed.weight.data[0].zero_()
    model.backbone.task_embed.weight.data[1].fill_(1.0)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 3), dtype=torch.long)
    state = model.init_state(2)
    task_ids0 = torch.zeros(2, dtype=torch.long)
    task_ids1 = torch.ones(2, dtype=torch.long)

    out0 = model.forward(input_ids=input_ids, task_ids=task_ids0, state=state, return_loss=False)
    out1 = model.forward(input_ids=input_ids, task_ids=task_ids1, state=state, return_loss=False)
    assert not torch.allclose(out0["action_logits"], out1["action_logits"])


def test_memory_decay_reduces_norm() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
        memory_decay=0.5,
    )
    model = VAGICore(cfg)
    mem_mod = model.backbone.memory
    mem_mod.write.weight.data.zero_()
    mem_mod.write.bias.data.zero_()
    mem_mod.erase.weight.data.zero_()
    mem_mod.erase.bias.data.fill_(-10.0)

    state = model.init_state(1)
    state.mem.fill_(1.0)
    obs = torch.zeros((1, cfg.obs_dim))
    token = torch.zeros((1, 1), dtype=torch.long)
    out = model.step(input_ids=token, obs=obs, state=state)
    assert out["state"].mem.norm().item() < state.mem.norm().item()


def test_memory_protection_blocks_overwrite() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
        memory_protect=True,
    )
    model = VAGICore(cfg)
    mem_mod = model.backbone.memory
    mem_mod.protect.weight.data.zero_()
    mem_mod.protect.bias.data.fill_(10.0)

    state = model.init_state(1)
    obs = torch.randn(1, cfg.obs_dim)
    token = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)
    out = model.step(input_ids=token, obs=obs, state=state)
    assert out["state"].mem.abs().sum().item() < 1e-3


def test_plan_fallback_policy_only() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=2,
        use_uncertainty=True,
    )
    model = VAGICore(cfg)
    model.world_logvar.proj.weight.data.zero_()
    model.world_logvar.proj.bias.data.fill_(5.0)

    obs = torch.randn(1, cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    greedy = torch.argmax(model.forward(input_ids=input_ids, obs=obs, state=state)["action_logits"], dim=-1)
    planned = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        uncertainty_fallback=1.0,
        horizon=2,
    )["action"]
    assert int(planned.item()) == int(greedy.item())


def test_tree_planning_action_in_topk() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=2,
        use_uncertainty=True,
    )
    model = VAGICore(cfg)
    obs = torch.randn(1, cfg.obs_dim)
    state = model.init_state(1)
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    logits = model.forward(input_ids=input_ids, obs=obs, state=state)["action_logits"].squeeze(0)
    topk = torch.topk(logits, k=2).indices.tolist()
    planned = model.plan_step(
        input_ids=input_ids,
        obs=obs,
        state=state,
        strategy="tree",
        tree_branching=2,
        horizon=2,
    )["action"].item()
    assert planned in topk
