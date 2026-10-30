
import torch
import triton
import triton.language as tl


@triton.jit
def nf4_grad_x_v2(
    DY, PACKED, SCALES, CODE, DX,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    red = tl.arange(0, BN)

    acc = tl.full((BM, BK), 0, tl.float32)

    for start in range(tl.cdiv(N, BN)):
        n = start * BN + red

        dy = tl.load(
            DY + rows[:, None] * N + n[None, :],
            mask=(rows[:, None] < M) & (n[None, :] < N),
            other=0,
        )

        idx = n[:, None] * K + cols[None, :]
        valid = (n[:, None] < N) & (cols[None, :] < K)

        packed = tl.load(
            PACKED + idx // 2,
            mask=valid,
            other=0,
        )

        nibble = tl.where(
            idx % 2 == 0,
            packed >> 4,
            packed & 15,
        ).to(tl.int32)

        code = tl.load(CODE + nibble)

        scale = tl.load(
            SCALES + idx // GROUP,
            mask=valid,
            other=0,
        )

        weights = (
            code.to(tl.float32)
            * scale.to(tl.float32)
        ).to(tl.float16)

        weights = tl.where(valid, weights, 0.0)

        acc = tl.dot(dy, weights, acc)

    tl.store(
        DX + rows[:, None] * K + cols[None, :],
        acc.to(tl.float16),
        mask=(rows[:, None] < M) & (cols[None, :] < K),
    )
