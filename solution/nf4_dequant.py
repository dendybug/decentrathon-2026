"""NF4 weight dequantization in one Triton launch, double quantization included.

bitsandbytes stores a 4-bit weight as packed bytes (two values per byte, high nibble first) plus a
per-64-values scale that is itself 8-bit: scale = code2[absmax8[block]] * absmax2[block // 256]
+ offset. Its default path dequantizes the scales in one kernel, adds the offset in another, and
dequantizes the weight in a third. This kernel does all of it per byte pair and writes the weight
in the compute dtype.

Where it is used:
  - the backward of every 4-bit linear (bitsandbytes.functional.dequantize_4bit),
  - the forward of every 4-bit linear when a gradient is needed (MatMul4Bit.forward).
Anything that is not exactly this case (FP4, no double quant, CPU, odd shapes) takes bnb's code.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax8_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_bytes, BLOCKSIZE: tl.constexpr, SUBSIZE: tl.constexpr,
                       BLOCK: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = pos < n_bytes
    byte = tl.load(packed_ptr + pos, mask=inside, other=0).to(tl.int32)
    block = (pos * 2) // BLOCKSIZE
    a8 = tl.load(absmax8_ptr + block, mask=inside, other=0).to(tl.int32)
    a2 = tl.load(absmax2_ptr + block // SUBSIZE, mask=inside, other=0.0)
    c2 = tl.load(code2_ptr + a8, mask=inside, other=0.0)
    scale = c2 * a2 + tl.load(offset_ptr)
    high = tl.load(nf4_ptr + (byte >> 4), mask=inside, other=0.0)
    low = tl.load(nf4_ptr + (byte & 15), mask=inside, other=0.0)
    tl.store(out_ptr + pos * 2, (high * scale).to(out_ptr.dtype.element_ty), mask=inside)
    tl.store(out_ptr + pos * 2 + 1, (low * scale).to(out_ptr.dtype.element_ty), mask=inside)


def _eligible(packed, quant_state):
    state2 = getattr(quant_state, "state2", None)
    return (
        quant_state is not None
        and getattr(quant_state, "quant_type", None) == "nf4"
        and getattr(quant_state, "nested", False)
        and state2 is not None
        and packed.is_cuda
        and packed.dtype == torch.uint8
        and len(quant_state.shape) == 2
        and quant_state.blocksize % 2 == 0
        and state2.blocksize > 0
        and (quant_state.shape[0] * quant_state.shape[1]) % quant_state.blocksize == 0
        and quant_state.absmax.dtype == torch.uint8
        and quant_state.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and getattr(quant_state, "code", None) is not None
    )


def nf4_dequantize(packed, quant_state):
    """The dequantized weight [out_features, in_features] in quant_state.dtype."""
    numel = quant_state.shape[0] * quant_state.shape[1]
    out = torch.empty(tuple(quant_state.shape), dtype=quant_state.dtype, device=packed.device)
    n_bytes = numel // 2
    state2 = quant_state.state2
    offset = quant_state.offset
    if not torch.is_tensor(offset):
        offset = torch.tensor([float(offset)], device=packed.device)
    nf4 = quant_state.code.to(device=packed.device, dtype=torch.float32)
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK),)](
        packed.reshape(-1), quant_state.absmax, state2.code, state2.absmax, offset, nf4, out,
        n_bytes, BLOCKSIZE=quant_state.blocksize, SUBSIZE=state2.blocksize, BLOCK=BLOCK,
        num_warps=4)
    return out


def patch_bnb_nf4():
    try:
        import bitsandbytes.functional as bnb_functional
        from bitsandbytes.autograd import _functions
    except Exception:  # noqa: BLE001 - without bitsandbytes there is nothing to patch
        return

    original_dequantize = bnb_functional.dequantize_4bit

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type=None):
        if absmax is None and out is None and _eligible(A, quant_state):
            return nf4_dequantize(A, quant_state)
        return original_dequantize(A, quant_state, absmax, out, blocksize, quant_type)

    bnb_functional.dequantize_4bit = dequantize_4bit

    original_forward = _functions.MatMul4Bit.forward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        needs_grad = any(ctx.needs_input_grad[:2]) or (bias is not None and ctx.needs_input_grad[3])
        if (quant_state is None or not needs_grad or A.numel() == 0 or not A.is_cuda
                or not _eligible(B, quant_state)
                or A.shape[-1] != quant_state.shape[1]):
            return original_forward(ctx, A, B, out, bias, quant_state)
        B = B.view(-1, 1)
        weight = nf4_dequantize(B, quant_state).to(A.dtype)
        output = torch.nn.functional.linear(A, weight, bias)
        if out is not None:
            out.copy_(output)
            output = out
        ctx.is_empty = False
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype
        ctx.dtype_bias = None if bias is None else bias.dtype
        ctx.tensors = (None, B) if any(ctx.needs_input_grad[:2]) else (None, None)
        return output

    _functions.MatMul4Bit.forward = staticmethod(forward)
