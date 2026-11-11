"""The pieces of the pre-#2582 `dspark_markov_sample` that the probabilistic
DSpark draft (ROCm/ATOM#2724 / #2877 / #3134 / #3139) is written against:
the stage-2 argmax kernel, `argmax_gather` and their tile sizes, copied
verbatim from pr-3139:atom/model_ops/dspark_markov_sample.py. Main renamed the module to
`dspark_markov_argmax` (no custom op is registered here, so the two coexist).
"""

import torch
import triton
import triton.language as tl

_BLOCK_V = 128
_BLOCK_K = 64
_MIN_BLOCK_ROW = 16
_MAX_BLOCK_ROW = 64
_ID_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _dspark_markov_argmax_stage2(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    num_rows,
    num_tiles,
    vocab_size,
    stride_out,
    packed_ptr,
    BLOCK_TILE: tl.constexpr,
    PACKED: tl.constexpr,
):
    """One row: pick the winning V tile, lowest id among tiles that tie.

    ``PACKED`` writes ``(max, id)`` as two fp32 to ``packed_ptr`` instead of
    the id to ``out_ptr``: the TP form, whose ranks then compare their bests.
    """
    row = tl.program_id(0)
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    vals = tl.load(
        part_val_ptr + offs_tile * num_rows + row,
        mask=tile_mask,
        other=float("-inf"),
    )
    idxs = tl.load(
        part_idx_ptr + offs_tile * num_rows + row, mask=tile_mask, other=vocab_size
    )
    best = tl.max(vals, axis=0)
    # Ids are global, so the sentinel is anything past every tile's id.
    cand = tl.where((vals == best) & tile_mask, idxs, 2147483647)
    winner = tl.min(cand, axis=0)
    if PACKED:
        # An all-NaN row keeps the sentinel; publish id 0 with the NaN, which
        # `torch.argmax` across ranks then treats as the max, as it would the
        # whole row's.
        winner = tl.where(winner == 2147483647, 0, winner)
        # ids < 2**24 are exact in fp32 (V4.1's vocab is 129280).
        tl.store(packed_ptr + row * 2, best)
        tl.store(packed_ptr + row * 2 + 1, winner.to(tl.float32))
        return
    # NaN never equals itself, so no lane passes its `== max` test and every
    # candidate keeps the out-of-range sentinel. Answer 0, as `torch.argmax`
    # does, so a caller gathering `W1[x]` cannot read past the table.
    winner = tl.where(winner == 2147483647, 0, winner)
    # Strided and in the destination's own dtype: the caller's `[B, T + 1]`
    # block is int32, and the copy this replaces was narrowing as well as
    # moving -- doing one and not the other would just relocate the cast.
    tl.store(out_ptr + row * stride_out, winner.to(out_ptr.dtype.element_ty))


@triton.jit
def _argmax_gather_kernel(
    pairs_ptr,
    out_ptr,
    num_rows,
    stride_out,
    NUM_J: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """``out[i] = ids[argmax_j vals[j, i], i]`` over ``pairs`` ``[J, N, 2]``
    fp32 ``(val, id)``: ``torch.argmax`` over ``j``, the gather of the ids at
    the winner and the copy into a strided, integer ``out`` in one launch.

    The same pick as ``torch.argmax``: the lowest ``j`` among equal values,
    and the first NaN if any (``torch.argmax`` takes NaN as the max).
    """
    rows = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask = rows < num_rows
    best = tl.load(pairs_ptr + rows * 2, mask=mask, other=0.0)
    best_id = tl.load(pairs_ptr + rows * 2 + 1, mask=mask, other=0.0)
    best_nan = best != best  # noqa: PLR0124 -- NaN test in Triton
    for j in tl.static_range(1, NUM_J):
        base = pairs_ptr + (j * num_rows + rows) * 2
        val = tl.load(base, mask=mask, other=0.0)
        idx = tl.load(base + 1, mask=mask, other=0.0)
        val_nan = val != val  # noqa: PLR0124
        # strict `>`: an equal value keeps the lower j
        take = (~best_nan) & (val_nan | (val > best))
        best = tl.where(take, val, best)
        best_id = tl.where(take, idx, best_id)
        best_nan = best_nan | val_nan
    tl.store(
        out_ptr + rows * stride_out,
        best_id.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def argmax_gather(pairs: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out[i] = pairs[argmax_j pairs[j, i, 0], i, 1]`` for fp32 ``pairs``
    ``[J, N, 2]`` of ``(value, id)``, written into integer ``out`` ``[N]``
    (strided is fine); see ``_argmax_gather_kernel``."""
    num_j, num_rows = pairs.shape[0], pairs.shape[1]
    if pairs.shape != (num_j, num_rows, 2) or pairs.dtype != torch.float32:
        raise ValueError(
            f"pairs {tuple(pairs.shape)} {pairs.dtype} is not [J, N, 2] fp32"
        )
    if out.shape != (num_rows,):
        raise ValueError(f"out {tuple(out.shape)} is not [{num_rows}]")
    if num_rows:
        pairs = pairs if pairs.is_contiguous() else pairs.contiguous()
        block_row = min(1024, triton.next_power_of_2(num_rows))
        _argmax_gather_kernel[(triton.cdiv(num_rows, block_row),)](
            pairs,
            out,
            num_rows,
            out.stride(0),
            NUM_J=num_j,
            BLOCK_ROW=block_row,
            num_warps=1 if block_row <= 64 else 4,
        )
    return out
