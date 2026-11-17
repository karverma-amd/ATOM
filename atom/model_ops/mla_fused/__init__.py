# SPDX-License-Identifier: MIT
"""Fused MLA decode kernels for gfx1250 (Gluon) and the checks that gate them.

* ATOM_MLA_FUSED_KUP_QPE: K-up GEMM + q_pe RoPE + KV-cache write in one kernel
  (fused_kup_qpe.kup_qpe_fused).
* ATOM_MLA_VUP_OPROJ_QUANT_FUSE: V-up GEMM with o_proj's MXFP4 activation quant
  in its epilogue (fused_vup_quant.vup_mxfp4).

Both are bit-exact with the paths they replace. The gates below only use
attributes and shapes, so they import without the GPU stack; the kernels are
imported lazily by the caller.
"""

import functools

import torch

_GFX1250 = ("gfx1250", "gfx1250-strict")


@functools.cache
def gfx_arch() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime()
    except Exception:  # noqa: BLE001 - no GPU stack means no gfx1250
        return ""


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def kup_qpe_fusion_ok(
    *,
    enabled: bool,
    arch: str,
    is_prefill: bool,
    use_qrep: bool,
    dcp_world_size: int,
    q_out: torch.Tensor | None,
    num_tokens: int,
    slot_mapping: torch.Tensor,
    w_k: torch.Tensor,
    cos_cache: torch.Tensor,
    rope_dim: int,
    kv_cache: torch.Tensor | None,
    k_scale,
) -> bool:
    """q_out is the prestored q_out of the BF16-absorb + shuffled-KV path (None
    when that path is off); kv_cache is the shuffled (blocks, 1, bs, d) view."""
    if not enabled or arch not in _GFX1250:
        return False
    # Pure decode only: no prefill (so no PCP), no DCP / query replication.
    if is_prefill or use_qrep or dcp_world_size > 1:
        return False
    if q_out is None or kv_cache is None:
        return False
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != num_tokens:
        return False
    if not (_is_fp8(q_out) and kv_cache.dtype == q_out.dtype):
        return False
    if w_k.dtype != torch.bfloat16 or w_k.dim() != 3:
        return False
    num_heads, kv_lora_rank, nope_dim = w_k.shape
    if kv_lora_rank % 256 or nope_dim % 32 or nope_dim > 128:
        return False
    if rope_dim not in (32, 64, 128, 256):
        return False
    cos = cos_cache
    while cos.dim() > 2 and cos.shape[1] == 1:
        cos = cos.squeeze(1)
    if cos.shape[-1] * 2 != rope_dim or cos.stride(-1) != 1:
        return False
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 1:
        return False
    if kv_cache.shape[2] % 16 or kv_cache.shape[3] != kv_lora_rank + rope_dim:
        return False
    if kv_cache.stride(3) != 1:
        return False
    if q_out.shape != (num_tokens, num_heads, kv_lora_rank + rope_dim):
        return False
    if q_out.stride(2) != 1 or num_tokens * q_out.stride(0) >= 2**31:
        return False
    if k_scale is not None and not (
        isinstance(k_scale, torch.Tensor) and k_scale.numel() == 1
    ):
        return False
    return num_tokens > 0


def vup_oproj_quant_fusion_ok(
    *,
    enabled: bool,
    arch: str,
    dcp_world_size: int,
    x: torch.Tensor,
    w_v: torch.Tensor,
    w_v_scale,
    fp4_bmm: bool,
    o_proj,
) -> bool:
    """x is the attention output (tokens, heads, kv_lora_rank)."""
    if not enabled or arch not in _GFX1250 or dcp_world_size > 1:
        return False
    # BF16 absorb path only.
    if w_v_scale is not None or fp4_bmm:
        return False
    if w_v.dtype != torch.bfloat16 or w_v.dim() != 3 or x.dtype != torch.bfloat16:
        return False
    num_heads, v_head_dim, kv_lora_rank = w_v.shape
    if v_head_dim != 128 or kv_lora_rank % 64:
        return False
    if x.numel() == 0 or x.numel() % (num_heads * kv_lora_rank):
        return False
    # o_proj: per-1x32 MXFP4 weight with the 32x4 (n32k4) weight scale, fed by
    # the m32k4 activation quant; no bias, no padded output, no native-a8 path.
    if not getattr(o_proj, "fp4_n32k4_scale", False):
        return False
    if getattr(getattr(o_proj, "quant_type", None), "name", None) != "per_1x32":
        return False
    if getattr(o_proj, "bias", None) is not None:
        return False
    if getattr(o_proj, "is_output_padded", False):
        return False
    return getattr(o_proj, "native_a8_group_rows", None) is None
