"""Output head + causal-LM loss in token chunks, with our own Triton kernel.

Base Soup builds the full [batch, seq, vocab] logits (vocab = 151,936), upcasts them to fp32 for
the loss, and TRL then walks them again for entropy and accuracy. That is the largest block of
memory in the run and a lot of element-wise traffic.

Here, for every chunk of rows:
  1. logits = hidden_chunk @ W^T                       (fp16 matmul, one chunk only)
  2. head_loss_rows_kernel (Triton, ours) reads each row of the chunk and in one launch gives
       - the cross-entropy of the row's label,
       - the entropy of the row's distribution (what TRL logs as `entropy`),
       - whether argmax == label (what TRL logs as `mean_token_accuracy`),
       - and overwrites the row with d(loss)/d(logits), so no softmax tensor is ever built;
  3. grad_hidden_chunk = dlogits @ W                    (W is frozen: no weight gradient)
The chunk's logits are dropped before the next chunk, so memory no longer grows with the vocabulary
times the whole batch.

Design idea (not code): chunking the head and keeping TRL's metrics, as in Soup's own
`stream_loss_chunk` option (#1693), which is off here. The kernel, the autograd function and the
patches below are written for this task.

What is patched (see `patch_head_loss`):
  - Qwen2ForCausalLM.forward: only while our compute_loss has armed it, with labels, in training,
    when the head is frozen. It returns the loss and no logits.
  - trl.SFTTrainer.compute_loss: for such steps, logs entropy / num_tokens / mean_token_accuracy
    with TRL's own formulas and hands the per-token losses to Soup's rewind recorder.
Everything else (no labels, eval, other models, fake modules in Soup's tests) takes the original path.
"""

import torch
import triton
import triton.language as tl

IGNORE = -100
# dlogits are written in fp16: scaled up by this factor so tiny probabilities do not underflow,
# and scaled back in fp32 once the hidden-state gradient has been formed.
PRESCALE = 4096.0
BLOCK_V = 4096
CHUNK_ROWS = 512


@triton.jit
def head_loss_rows_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, correct_ptr, scale_ptr,
                          V, stride, BLOCK: tl.constexpr):
    """One program per row of logits [rows, V]; the row is overwritten by its gradient."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    label = tl.load(target_ptr + row)
    valid = label >= 0
    safe = tl.where(valid, label, 0)
    lane = tl.arange(0, BLOCK)

    # pass 1: maximum and the first index that reaches it
    mvec = tl.full([BLOCK], float("-inf"), tl.float32)
    ivec = tl.zeros([BLOCK], tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + lane
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        better = x > mvec
        mvec = tl.where(better, x, mvec)
        ivec = tl.where(better, offs, ivec)
    mx = tl.max(mvec, axis=0)
    cand = tl.where(mvec == mx, ivec, 2147483647)
    amax = tl.min(cand, axis=0)

    # pass 2: sum exp(x - max) and sum exp(x - max) * x  ->  logsumexp and entropy
    svec = tl.zeros([BLOCK], tl.float32)
    tvec = tl.zeros([BLOCK], tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + lane
        inside = offs < V
        x = tl.load(base + offs, mask=inside, other=0.0).to(tl.float32)
        e = tl.where(inside, tl.exp(x - mx), 0.0)
        svec += e
        tvec += e * x
    total = tl.sum(svec, axis=0)
    weighted = tl.sum(tvec, axis=0)
    lse = mx + tl.log(total)
    entropy = lse - weighted / total

    target_logit = tl.load(base + safe).to(tl.float32)
    loss = tl.where(valid, lse - target_logit, 0.0)
    hit = (valid & (amax == safe)).to(tl.int32)
    tl.store(loss_ptr + row, loss)
    tl.store(entropy_ptr + row, entropy)
    tl.store(correct_ptr + row, hit)

    # pass 3: gradient (softmax - one_hot) * scale, in place; rows without a label get zeros
    scale = tl.load(scale_ptr) * valid.to(tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + lane
        inside = offs < V
        x = tl.load(base + offs, mask=inside, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        g = (p - tl.where(offs == safe, 1.0, 0.0)) * scale
        tl.store(base + offs, g.to(logits_ptr.dtype.element_ty), mask=inside)


class _ChunkedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, denominator, chunk):
        batch, seq, width = hidden.shape
        vocab = weight.shape[0]
        n = batch * seq
        need_grad = ctx.needs_input_grad[0]
        device = hidden.device

        w16 = weight if weight.dtype == torch.float16 else weight.to(torch.float16)
        h16 = hidden.reshape(n, width).to(torch.float16)
        tgt = target.reshape(n).contiguous()
        loss_rows = torch.empty(n, dtype=torch.float32, device=device)
        entropy = torch.empty(n, dtype=torch.float32, device=device)
        hits = torch.empty(n, dtype=torch.int32, device=device)
        scale = (PRESCALE / denominator).reshape(1).to(torch.float32)
        grad16 = torch.empty(n, width, dtype=torch.float16, device=device) if need_grad else None

        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            logits = torch.mm(h16[i:j], w16.t())
            head_loss_rows_kernel[(j - i,)](
                logits, tgt[i:j], loss_rows[i:j], entropy[i:j], hits[i:j], scale,
                vocab, logits.stride(0), BLOCK=BLOCK_V, num_warps=8)
            if need_grad:
                torch.mm(logits, w16, out=grad16[i:j])
            del logits

        loss = loss_rows.sum() / denominator
        token_loss = loss_rows.view(batch, seq)
        token_entropy = entropy.view(batch, seq)
        correct = hits.sum().to(torch.long)
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)
        if need_grad:
            ctx.save_for_backward(grad16)
        ctx.need_grad = need_grad
        ctx.shape = (batch, seq, width)
        ctx.hidden_dtype = hidden.dtype
        return loss, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, *_):
        if not ctx.need_grad:
            return None, None, None, None, None
        (grad16,) = ctx.saved_tensors
        factor = grad_loss.to(torch.float32) / PRESCALE
        grad = (grad16.to(torch.float32) * factor).to(ctx.hidden_dtype).view(ctx.shape)
        return grad, None, None, None, None


class ChunkedStep:
    """What a chunked forward leaves for TRL's metrics and for Soup's rewind recorder."""

    def __init__(self, token_entropy, token_loss, correct, labelled):
        self.token_entropy = token_entropy  # [batch, seq] fp32
        self.token_loss = token_loss        # [batch, seq] fp32, 0 where there is no label
        self.correct = correct              # labelled positions where argmax == label
        self.labelled = labelled            # positions with a label


def chunked_head_loss(hidden, weight, labels, num_items_in_batch=None, chunk=CHUNK_ROWS):
    """Same loss as ForCausalLMLoss(lm_head(hidden), labels): the mean over labelled positions,
    or their sum divided by num_items_in_batch when the trainer gives one."""
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE)[..., 1:].contiguous()
    labelled = (target != IGNORE).sum()
    if num_items_in_batch is None:
        denominator = labelled.to(torch.float32).clamp(min=1.0)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device, torch.float32)
    else:
        denominator = torch.tensor(float(num_items_in_batch), device=hidden.device)
    loss, token_loss, token_entropy, correct = _ChunkedHeadLoss.apply(
        hidden, weight, target, denominator, chunk)
    return loss, ChunkedStep(token_entropy, token_loss, correct, labelled)


# --------------------------------------------------------------------------- patches

class _State:
    armed = False
    step = None


_STATE = _State()


def _causal_lm_of(model):
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _head_ok(model):
    """A real Qwen2 causal LM with a frozen output head on the GPU."""
    from transformers.models.qwen2 import modeling_qwen2

    base = _causal_lm_of(model)
    if not isinstance(base, modeling_qwen2.Qwen2ForCausalLM):
        return False
    weight = getattr(getattr(base, "lm_head", None), "weight", None)
    if weight is None or weight.dim() != 2 or not weight.is_cuda or weight.requires_grad:
        return False
    return getattr(base.config, "final_logit_softcapping", None) is None


def _log_metrics(trainer, inputs, step):
    """entropy, num_tokens and mean_token_accuracy, logged the way TRL's compute_loss logs them."""
    metrics = trainer._metrics["train"]
    gather = trainer.accelerator.gather_for_metrics
    mask = inputs.get("attention_mask")
    if mask is None:
        entropy = step.token_entropy.mean()
    else:
        entropy = (step.token_entropy * mask).sum() / mask.sum()
        trainer._total_train_tokens += gather(mask.sum()).sum().item()
    metrics["entropy"].append(gather(entropy).mean().item())
    metrics["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics["mean_token_accuracy"].append(accuracy)


def patch_head_loss():
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2
    from trl import SFTTrainer

    try:  # the attribute Soup's rewind recorder reads the per-token losses from
        from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    except Exception:  # noqa: BLE001
        STEP_ATTR = "_soup_chunked_step"

    original_forward = modeling_qwen2.Qwen2ForCausalLM.forward
    original_compute_loss = SFTTrainer.compute_loss

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if (not _STATE.armed or labels is None or not self.training
                or not torch.is_grad_enabled() or not isinstance(logits_to_keep, int)
                or logits_to_keep != 0 or not _head_ok(self)):
            return original_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, labels=labels,
                                    use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        loss, _STATE.step = chunked_head_loss(
            hidden, self.lm_head.weight, labels,
            num_items_in_batch=kwargs.get("num_items_in_batch"))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if ("labels" not in inputs or "shift_labels" in inputs
                or not getattr(model, "training", False) or not torch.is_grad_enabled()
                or getattr(self.args, "use_liger_kernel", False)
                or getattr(self, "compute_loss_func", None) is not None
                or getattr(self, "label_smoother", None) is not None or not _head_ok(model)):
            return original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _STATE.armed, _STATE.step = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            _STATE.armed = False
        step, _STATE.step = _STATE.step, None
        if step is None:
            raise RuntimeError("chunked head loss: the model did not take the chunked path")
        setattr(self, STEP_ATTR, step)
        _log_metrics(self, inputs, step)
        return (loss, outputs) if return_outputs else loss

    modeling_qwen2.Qwen2ForCausalLM.forward = forward
    SFTTrainer.compute_loss = compute_loss
