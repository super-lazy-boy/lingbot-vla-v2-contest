#!/usr/bin/env python3
"""Blackwell 冒烟：FlashAttention2 / FlexAttention / fused MoE 的真实前向+反向。"""
import torch


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:
        print(f"FAIL {name}: {e!r}")
        raise


def _fwd_bwd(out, label):
    assert torch.isfinite(out).all(), f"{label} output not finite"
    out.sum().backward()


def flash_attn_fwd_bwd():
    from flash_attn import flash_attn_func
    torch.manual_seed(0)
    q = torch.randn(1, 2048, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    _fwd_bwd(out, "flash_attn")
    assert q.grad is not None and torch.isfinite(q.grad).all()


def flex_attention_fwd_bwd():
    from torch.nn.attention.flex_attention import flex_attention
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    def causal(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, -float("inf"))

    out = flex_attention(q, k, v, score_mod=causal)
    _fwd_bwd(out, "flex_attention")
    assert q.grad is not None and torch.isfinite(q.grad).all()


def fused_moe_fwd_bwd():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2TokenMoeBlock
    torch.manual_seed(0)
    # Same construction as modeling_lingbot_vla_v2.py:175-195 (robotwin settings).
    token_config = CONFIG_MAPPING["qwen2_moe"](
        num_experts=32,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        hidden_size=768,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=704,
        output_router_logits=False,
        torch_dtype="bfloat16",
    )
    token_config.bias_update_speed = 0
    token_config._moe_implementation = "fused"
    token_config.router_activation = "sigmoid"
    token_config.routed_scaling_factor = 4.0
    token_config.use_shared_expert_gate = False
    block = Qwen2TokenMoeBlock(token_config).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 128, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = block(x)
    if isinstance(out, tuple):  # (hidden_states, router_logits)
        out = out[0]
    _fwd_bwd(out, "fused_moe")
    assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    check("flash_attn fwd+bwd", flash_attn_fwd_bwd)
    check("flex_attention fwd+bwd", flex_attention_fwd_bwd)
    check("fused_moe fwd+bwd", fused_moe_fwd_bwd)
    print("gpu_smoke: ALL PASS")
