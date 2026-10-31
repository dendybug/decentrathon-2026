"""Our kernels against plain PyTorch / bitsandbytes. Run: python -m pytest solution -q (needs a GPU)."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
def test_head_loss_matches_cross_entropy():
    from solution.head_loss import chunked_head_loss

    torch.manual_seed(0)
    batch, seq, width, vocab = 3, 37, 64, 1000
    hidden0 = torch.randn(batch, seq, width, device="cuda")
    weight = (torch.randn(vocab, width, device="cuda") * 0.05).to(torch.float16)
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, :10] = -100
    labels[1, 20:] = -100

    hidden = hidden0.clone().requires_grad_(True)
    loss, step = chunked_head_loss(hidden, weight, labels, chunk=50)
    loss.backward()

    ref_hidden = hidden0.clone().requires_grad_(True)
    logits = (ref_hidden.half() @ weight.t()).float()
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    ref_loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab), shifted.reshape(-1),
                                                 ignore_index=-100)
    ref_loss.backward()

    probs = torch.softmax(logits, -1)
    ref_entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)
    labelled = shifted != -100
    ref_correct = ((logits.argmax(-1) == shifted) & labelled).sum()
    assert torch.allclose(loss, ref_loss, rtol=1e-3, atol=1e-4)
    assert torch.allclose(hidden.grad, ref_hidden.grad, rtol=2e-2, atol=1e-5)
    assert torch.allclose(step.token_entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    assert int(step.correct) == int(ref_correct)
    assert int(step.labelled) == int(labelled.sum())


@cuda
def test_rmsnorm_matches_torch():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.rmsnorm import FusedRMSNorm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(1536).cuda()
    norm.weight.data.uniform_(0.5, 1.5)
    norm.weight.requires_grad_(False)
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(2, 33, 1536, device="cuda", dtype=dtype).requires_grad_(True)
        y_ref = norm(x)
        y = FusedRMSNorm.apply(x, norm.weight, norm.variance_epsilon)
        assert y.dtype == y_ref.dtype
        assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
        dy = torch.randn_like(y_ref)
        (gx_ref,) = torch.autograd.grad(y_ref, x, dy)
        (gx,) = torch.autograd.grad(y, x, dy)
        assert torch.allclose(gx.float(), gx_ref.float(), rtol=2e-2, atol=2e-3)


@cuda
def test_nf4_dequant_matches_bitsandbytes():
    import bitsandbytes.functional as bnb_functional

    from solution.nf4_dequant import nf4_dequantize

    torch.manual_seed(0)
    weight = torch.randn(512, 768, device="cuda", dtype=torch.float16)
    packed, state = bnb_functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    assert state.nested
    expected = bnb_functional.dequantize_4bit(packed, state)
    got = nf4_dequantize(packed, state)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    assert (got.float() - expected.float()).abs().max().item() <= 1e-3
