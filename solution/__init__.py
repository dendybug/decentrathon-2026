"""Your solution.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

Own Triton kernels (one file each):
  head_loss.py   output head + cross-entropy + entropy + accuracy in chunks, no full logits
  rmsnorm.py     fused RMSNorm forward and backward
  nf4_dequant.py NF4 dequantization with double quantization, forward and backward of 4-bit linears
The SwiGLU example (swiglu.py) stays on next to them; it does not count as ours.
"""

from .head_loss import patch_head_loss
from .nf4_dequant import patch_bnb_nf4
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp


def apply():
    patch_qwen2_mlp()      # the example: works out of the box, does not count as ours
    patch_qwen2_rmsnorm()  # ours: fused RMSNorm
    patch_bnb_nf4()        # ours: NF4 dequantization in bitsandbytes' forward and backward
    patch_head_loss()      # ours: chunked head + loss + entropy + accuracy
