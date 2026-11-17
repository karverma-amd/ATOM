# SPDX-License-Identifier: MIT
# Gating of the fused MLA decode kernels (ATOM_MLA_FUSED_KUP_QPE,
# ATOM_MLA_VUP_OPROJ_QUANT_FUSE): both default off, and every unsupported case
# falls back to the existing path.

from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

from atom.model_ops.mla_fused import kup_qpe_fusion_ok, vup_oproj_quant_fusion_ok
from atom.utils import envs

FP8 = torch.float8_e4m3fn
H, L, P, R, BS, M = 128, 512, 128, 64, 64, 96


@pytest.mark.parametrize(
    "name", ["ATOM_MLA_FUSED_KUP_QPE", "ATOM_MLA_VUP_OPROJ_QUANT_FUSE"]
)
def test_flags_default_off(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False


def _kup_args(**over):
    args = {
        "enabled": True,
        "arch": "gfx1250",
        "is_prefill": False,
        "use_qrep": False,
        "dcp_world_size": 1,
        "q_out": torch.empty(M, H, L + R, dtype=FP8),
        "num_tokens": M,
        "slot_mapping": torch.zeros(M, dtype=torch.int64),
        "w_k": torch.empty(H, L, P, dtype=torch.bfloat16),
        "cos_cache": torch.empty(4096, 1, 1, R // 2, dtype=torch.bfloat16),
        "rope_dim": R,
        "kv_cache": torch.empty(16, 1, BS, L + R, dtype=FP8),
        "k_scale": torch.ones(1),
    }
    args.update(over)
    return args


def test_kup_qpe_eligible():
    assert kup_qpe_fusion_ok(**_kup_args())
    assert kup_qpe_fusion_ok(**_kup_args(k_scale=None))
    assert kup_qpe_fusion_ok(**_kup_args(arch="gfx1250-strict"))


@pytest.mark.parametrize(
    "over",
    [
        {"enabled": False},
        {"arch": "gfx950"},
        {"arch": ""},
        {"is_prefill": True},
        {"use_qrep": True},
        {"dcp_world_size": 2},
        {"q_out": None},
        {"kv_cache": None},
        {"slot_mapping": torch.zeros(M + 3, dtype=torch.int64)},
        {"q_out": torch.empty(M, H, L + R, dtype=torch.bfloat16)},
        {"kv_cache": torch.empty(16, 1, BS, L + R, dtype=torch.bfloat16)},
        {"kv_cache": torch.empty(16, 1, 8, L + R, dtype=FP8)},
        {"w_k": torch.empty(H, L, P, dtype=torch.float8_e4m3fn)},
        {"cos_cache": torch.empty(4096, R, dtype=torch.bfloat16)},
        {"k_scale": 1.0},
        {"k_scale": torch.ones(2)},
        {"num_tokens": 0, "q_out": torch.empty(0, H, L + R, dtype=FP8),
         "slot_mapping": torch.zeros(0, dtype=torch.int64)},
    ],
)  # fmt: skip
def test_kup_qpe_falls_back(over):
    assert not kup_qpe_fusion_ok(**_kup_args(**over))


def _o_proj(**over):
    attrs = {
        "fp4_n32k4_scale": True,
        "quant_type": SimpleNamespace(name="per_1x32"),
        "bias": None,
        "is_output_padded": False,
        "native_a8_group_rows": None,
    }
    attrs.update(over)
    return SimpleNamespace(**attrs)


def _vup_args(**over):
    args = {
        "enabled": True,
        "arch": "gfx1250",
        "dcp_world_size": 1,
        "x": torch.empty(M, H, L, dtype=torch.bfloat16),
        "w_v": torch.empty(H, 128, L, dtype=torch.bfloat16),
        "w_v_scale": None,
        "fp4_bmm": False,
        "o_proj": _o_proj(),
    }
    args.update(over)
    return args


def test_vup_oproj_eligible():
    assert vup_oproj_quant_fusion_ok(**_vup_args())


@pytest.mark.parametrize(
    "over",
    [
        {"enabled": False},
        {"arch": "gfx950"},
        {"dcp_world_size": 4},
        {"w_v_scale": torch.ones(1)},
        {"fp4_bmm": True},
        {"x": torch.empty(M, H, L, dtype=torch.float32)},
        {"w_v": torch.empty(H, 64, L, dtype=torch.bfloat16)},
        {"x": torch.empty(0, H, L, dtype=torch.bfloat16)},
        {"o_proj": _o_proj(fp4_n32k4_scale=False)},
        {"o_proj": _o_proj(quant_type=SimpleNamespace(name="per_1x128"))},
        {"o_proj": _o_proj(bias=torch.zeros(7168))},
        {"o_proj": _o_proj(is_output_padded=True)},
        {"o_proj": _o_proj(native_a8_group_rows=128)},
        {"o_proj": SimpleNamespace()},
    ],
)
def test_vup_oproj_falls_back(over):
    assert not vup_oproj_quant_fusion_ok(**_vup_args(**over))


def _attention_mla():
    return gpu_stack_import("atom.model_ops.attention_mla")


@pytest.mark.parametrize("flag, fused", [("0", False), ("1", True)])
def test_v_up_o_proj_dispatch(monkeypatch, flag, fused):
    mla = _attention_mla()
    monkeypatch.setenv("ATOM_MLA_VUP_OPROJ_QUANT_FUSE", flag)
    monkeypatch.delenv("ATOM_USE_TRITON_MXFP4_BMM", raising=False)
    monkeypatch.setattr(mla, "gfx_arch", lambda: "gfx1250")
    calls, o_proj_calls = [], []
    x = torch.empty(M, H, L, dtype=torch.bfloat16)
    v_out = torch.empty(M, H, 128, dtype=torch.bfloat16)

    class OProj(SimpleNamespace):
        def __call__(self, t, **kw):
            o_proj_calls.append((t.shape, kw))
            return "old_out"

    fake = SimpleNamespace(
        dcp_world_size=1,
        W_V=torch.empty(H, 128, L, dtype=torch.bfloat16),
        W_V_scale=None,
        num_heads=H,
        v_head_dim=128,
        o_proj=OProj(**vars(_o_proj())),
        _v_up_proj=lambda t: calls.append("v_up") or v_out,
        _v_up_proj_quant_and_o_proj=lambda t: calls.append("fused") or "fused_out",
    )
    out = mla.MLAAttention._v_up_proj_and_o_proj(fake, x)
    if fused:
        assert calls == ["fused"] and out == "fused_out" and not o_proj_calls
    else:
        assert calls == ["v_up"] and out == "old_out"
        assert o_proj_calls == [(torch.Size([M, H * 128]), {})]


def test_q_proj_heads_matches_old_split(monkeypatch):
    mla = _attention_mla()
    q = torch.randn(M, H * (P + R))
    fake = SimpleNamespace(
        qrep_enabled=False,
        num_heads=H,
        qrep_num_heads=H,
        qk_head_dim=P + R,
        q_proj=lambda x, s: x,
    )
    out = mla.MLAAttention._q_proj_heads(fake, q, None, group=False)
    assert out.shape == (M, H, P + R) and out.data_ptr() == q.data_ptr()


def test_o_proj_takes_4d_m32k4_scale_as_is(monkeypatch):
    # Fusion B hands o_proj its scale as a 4-D [m / 32, k / 128, 32, 4] view,
    # which gemm_a4w4_quant passes through without re-layout or requant.
    linear = gpu_stack_import("atom.model_ops.linear")

    seen = {}

    def fake_gemm(x, w, x_scale, w_scale, dtype=None):
        seen["x_dtype"], seen["x_scale"] = x.dtype, x_scale
        return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16)

    def no_convert(*a, **k):
        raise AssertionError("m32k4 scale must not be re-laid-out")

    monkeypatch.setattr(linear, "gemm_a4w4", fake_gemm)
    monkeypatch.setattr(linear, "_mxfp4_x_scale_to_m32k4", no_convert)
    k = H * 128
    x = torch.zeros(M, k // 2, dtype=torch.uint8)
    s = torch.arange((M + 31) // 32 * 32 * (k // 32), dtype=torch.int64).to(torch.uint8)
    s = s.view((M + 31) // 32 * 32, k // 32)
    w = torch.zeros(7168, k // 2, dtype=torch.uint8)
    fn = getattr(linear.gemm_a4w4_quant, "__wrapped__", linear.gemm_a4w4_quant)
    y = fn(
        x, s.view(-1, k // 128, 32, 4), w, torch.bfloat16, None, None, None, 7168, True
    )
    assert y.shape == (M, 7168)
    assert seen["x_dtype"] == torch.float4_e2m1fn_x2
    got = seen["x_scale"].view(torch.uint8)
    assert got.shape == s.shape and got.data_ptr() == s.data_ptr()
