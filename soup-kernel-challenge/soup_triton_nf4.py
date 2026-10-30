
import torch
import triton
import triton.language as tl
import bitsandbytes.functional as bnb


@triton.jit
def _nf4_kernel(
    PACKED, SCALES, CODE, OUT,
    TOTAL: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL

    packed = tl.load(
        PACKED + i // 2, mask=mask, other=0
    )

    index = tl.where(
        i % 2 == 0,
        packed >> 4,
        packed & 15,
    ).to(tl.int32)

    scale = tl.load(
        SCALES + i // GROUP,
        mask=mask,
        other=0,
    )

    value = tl.load(CODE + index)

    tl.store(
        OUT + i,
        value * scale,
        mask=mask,
    )


_NF4_CODE_CACHE = {}


def _get_nf4_code(device):
    key = str(device)

    if key not in _NF4_CODE_CACHE:
        _NF4_CODE_CACHE[key] = bnb.get_4bit_type(
            "nf4", device=device
        )

    return _NF4_CODE_CACHE[key]


def dequantize_4bit(packed, quant_state=None, **kwargs):
    original = bnb.dequantize_4bit

    if (
        quant_state is None
        or quant_state.quant_type != "nf4"
        or not packed.is_cuda
        or packed.dtype != torch.uint8
        or kwargs
    ):
        return original(
            packed,
            quant_state=quant_state,
            **kwargs,
        )

    if quant_state.nested:
        scales = bnb.dequantize_blockwise(
            quant_state.absmax,
            quant_state=quant_state.state2,
        )
        scales = (
            scales.float()
            + quant_state.offset.float()
        ).contiguous()
    else:
        scales = quant_state.absmax.contiguous()

    out = torch.empty(
        tuple(quant_state.shape),
        device=packed.device,
        dtype=quant_state.dtype,
    )

    code = quant_state.code
    if code is None:
        code = _get_nf4_code(packed.device)
    elif code.device != packed.device:
        code = code.to(packed.device)

    total = out.numel()

    _nf4_kernel[(triton.cdiv(total, 1024),)](
        packed,
        scales,
        code,
        out,
        total,
        quant_state.blocksize,
        1024,
        num_warps=8,
    )

    return out
