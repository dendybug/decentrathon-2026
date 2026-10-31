"""Fused RMSNorm (forward and backward) for Qwen2RMSNorm, in Triton.

Torch runs Qwen2RMSNorm as about eight element-wise kernels (cast, square, mean, add, rsqrt,
multiply, cast, multiply) and writes every intermediate to memory. One program per row does it all.
The norm weights are frozen in this task (LoRA is on q_proj / v_proj), so the backward gives only
the input gradient; a trainable weight, a CPU tensor, or a hidden size that does not fit one block
takes the original code.
"""

import torch
import triton
import triton.language as tl

MAX_BLOCK = 8192


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, width, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    x = tl.load(x_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / width
    rstd = 1.0 / tl.sqrt(var + eps)
    # torch rounds x * rstd to the input dtype before multiplying by the weight
    xhat = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * width + cols, (xhat * w).to(y_ptr.dtype.element_ty), mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, width, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    dy = tl.load(dy_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    dxhat = dy * w
    c = tl.sum(dxhat * xhat, axis=0) / width
    dx = rstd * (dxhat - xhat * c)
    tl.store(dx_ptr + row * width + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        width = shape[-1]
        x2 = x.contiguous().view(-1, width)
        rows = x2.shape[0]
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty((rows, width), dtype=out_dtype, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(width)
        warps = 4 if block <= 2048 else 8
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, width, eps, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block, ctx.warps = shape, block, warps
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        width = x2.shape[1]
        dy2 = dy.contiguous().view(-1, width)
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(x2.shape[0],)](dy2, x2, weight, rstd, dx, width, BLOCK=ctx.block,
                                           num_warps=ctx.warps)
        return dx.view(ctx.shape), None, None


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (not hidden_states.is_cuda or weight.requires_grad
                or triton.next_power_of_2(hidden_states.shape[-1]) > MAX_BLOCK):
            return original(self, hidden_states)
        return FusedRMSNorm.apply(hidden_states, weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
