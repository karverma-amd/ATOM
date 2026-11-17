# SPDX-License-Identifier: MIT
"""MLA V-up (BF16 absorb) with the o_proj MXFP4 activation quant in its epilogue,
Gluon, gfx1250.

Replaces, bit for bit,
  batched_gemm_bf16(x, W_V, YQ=out.transpose(0, 1))                 # bf16 (M, H, V)
  get_hip_quant(per_1x32)(out.view(M, H * V), fp4x2, scale_layout_m32k4=True)
and emits what gemm_a4w4 reads on gfx1250: packed fp4 (M, H*V/2) and the e8m0
scale in the "m32k4" layout, (pad32(M), H*V/32) bytes,
  scale[(m/32)*K + (k/128)*128 + (m%32)*4 + (k/32)%4],  K = H*V,
with 0x7F in the pad rows. Per 32-group: o = acc.to(bf16) (what the GEMM stores),
amax = max(1e-10, max|o|), byte = ceil_exp(bits(amax * fp32(1/6))) (RoundUp),
fp4 = hardware scaled convert of o. The K order is the production one, so the
fp32 accumulator (and so o) is unchanged.
"""

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon


@gluon.jit
def _vup_mxfp4_kernel(
    a_ptr,
    b_ptr,
    y_ptr,
    s_ptr,
    M,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ym,
    N: gl.constexpr,
    K: gl.constexpr,
    S_COLS: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    num_warps: gl.constexpr,
):
    gl.static_assert(BLOCK_N == N, "one CTA covers a whole head in N")
    gl.static_assert(N % 128 == 0, "m32k4 scale tile is 128 columns")
    if NUM_XCDS > 1:
        # Each XCD walks M with its run of adjacent heads innermost, so the 64B
        # fp4 row pieces of heads 2k / 2k+1 (one 128B line) meet in one L2.
        pid = gl.program_id(axis=0)
        HPX: gl.constexpr = NUM_HEADS // NUM_XCDS
        xcd = pid % NUM_XCDS
        j = pid // NUM_XCDS
        pid_m = j // HPX
        batch_id = xcd * HPX + j % HPX
    else:
        batch_id = gl.program_id(axis=0)
        pid_m = gl.program_id(axis=1)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + batch_id * stride_ab + pid_m * BLOCK_M * stride_am,
        shape=(M - pid_m * BLOCK_M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + batch_id * stride_bb,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
        layout=SHARED_LAYOUT_B,
    )
    load_idx = 0
    compute_idx = 0
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    num_k_tiles: gl.constexpr = K // BLOCK_K

    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        load_idx += 1

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1) - 1):
        # Every warp is done reading the slot this TDM load refills.
        gl.barrier()
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        load_idx += 1
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    k_last: gl.constexpr = (num_k_tiles - 1) * BLOCK_K
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, set_bounds=[M - pid_m * BLOCK_M, K - k_last]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, set_bounds=[N, K - k_last]
    )
    gl.barrier()
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    load_idx += 1
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
    )
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
    compute_idx += 1
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # One whole 32-group per lane: no cross-lane amax.
    QB: gl.constexpr = 32
    NQB: gl.constexpr = BLOCK_N // QB
    QL: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, QB],
        threads_per_warp=[32 // NQB, NQB],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    o = gl.convert_layout(accumulator.to(gl.bfloat16), QL)
    og = o.to(gl.float32).reshape(BLOCK_M, NQB, QB)
    amax = gl.max(gl.abs(og), axis=2)
    amax = gl.maximum(amax, 1e-10)
    u = (amax * 0.16666666666666666).to(gl.uint32, bitcast=True)
    e = gl.minimum(((u & 0x7FFFFFFF) + 0x7FFFFF) >> 23, 255)
    bs = e.to(gl.uint8)
    q = gl.amd.cdna5.scaled_downcast(o, bs, "e2m1", axis=1)

    SHARED_LAYOUT_Y: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // 2, 16]], [BLOCK_M, BLOCK_N // 2], [1, 0]
    )
    y_buf = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_M, BLOCK_N // 2], SHARED_LAYOUT_Y
    )
    y_buf.store(q.to(gl.uint8, bitcast=True))
    gl.barrier()
    y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=y_ptr,
        shape=(M, S_COLS // 2),
        strides=(stride_ym, 1),
        block_shape=(BLOCK_M, BLOCK_N // 2),
        layout=SHARED_LAYOUT_Y,
    )
    gl.amd.gfx1250.tdm.async_store(
        y_desc, [pid_m * BLOCK_M, batch_id * (BLOCK_N // 2)], y_buf
    )

    SL: gl.constexpr = bs.type.layout
    srow = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, SL))
    sg = gl.arange(0, NQB, layout=gl.SliceLayout(0, SL))
    m_pad = (M + 31) // 32 * 32
    kt = (batch_id * BLOCK_N) // 128 + sg // 4
    s_offs = (
        (srow // 32).to(gl.int64)[:, None] * S_COLS
        + kt[None, :] * 128
        + (srow % 32)[:, None] * 4
        + (sg % 4)[None, :]
    )
    bs_out = gl.where((srow < M)[:, None], bs, gl.full_like(bs, 0x7F))
    gl.store(s_ptr + s_offs, bs_out, mask=(srow < m_pad)[:, None])
    gl.amd.gfx1250.tdm.async_wait(0)


_CFG = {
    "BLOCK_M": 128,
    "BLOCK_K": 64,
    "NUM_BUFFERS": 4,
    "num_warps": 4,
    "waves_per_eu": 2,
}


def vup_mxfp4(
    x: torch.Tensor,
    w_v: torch.Tensor,
    y: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    num_xcds: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (H, M, L) bf16 (head-major view), w_v: (H, V, L) bf16.

    Returns y uint8 (M, H*V/2) packed fp4 and scale uint8 (pad32(M), H*V/32)
    in the m32k4 layout (pad rows 0x7F)."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    H, M, L = x.shape
    _, V, _ = w_v.shape
    assert x.dtype == torch.bfloat16 and w_v.dtype == torch.bfloat16
    assert w_v.shape == (H, V, L) and x.stride(2) == 1 and V % 128 == 0
    bm, bk, nw = _CFG["BLOCK_M"], _CFG["BLOCK_K"], _CFG["num_warps"]
    assert L % bk == 0
    k_tot = H * V
    if y is None:
        y = torch.empty((M, k_tot // 2), dtype=torch.uint8, device=x.device)
    if scale is None:
        scale = torch.empty(
            ((M + 31) // 32 * 32, k_tot // 32), dtype=torch.uint8, device=x.device
        )
    wmma, opa, opb = create_wmma_layouts(nw)
    sa, sb = create_shared_layouts(bm, V, bk, "TN", 16)
    wt = w_v.transpose(1, 2)
    if H % num_xcds:
        num_xcds = 1
    if num_xcds > 1:
        grid = (H * triton.cdiv(M, bm),)
    else:
        grid = (H, triton.cdiv(M, bm))
    _vup_mxfp4_kernel[grid](
        x, wt, y, scale, M,
        x.stride(0), x.stride(1), x.stride(2),
        wt.stride(0), wt.stride(1), wt.stride(2),
        y.stride(0),
        N=V, K=L, S_COLS=k_tot, NUM_HEADS=H,
        BLOCK_M=bm, BLOCK_N=V, BLOCK_K=bk,
        NUM_BUFFERS=min(_CFG["NUM_BUFFERS"], L // bk),
        SHARED_LAYOUT_A=sa, SHARED_LAYOUT_B=sb, WMMA_LAYOUT=wmma,
        OPERAND_LAYOUT_A=opa, OPERAND_LAYOUT_B=opb,
        NUM_XCDS=num_xcds, num_warps=nw, waves_per_eu=_CFG["waves_per_eu"],
    )  # fmt: skip
    return y, scale
