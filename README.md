# Soup Kernel Challenge: chunked head loss, fused RMSNorm and NF4 dequantization in Triton

Decentrathon 6.0, track "AI Engineering by Soup", task "Soup Kernel Challenge".
Public Kaggle notebook (saved version): https://www.kaggle.com/code/askhatlishev/notebook3fe5abc70d

**По-русски, коротко.** Мы ускорили и облегчили `soup train` (Qwen2.5-1.5B-Instruct, 4-bit NF4,
LoRA r=16) тремя собственными Triton-ядрами в папке `solution/`. Память на T4: 14 345 → 8 383 MiB
(в 1,71 раза меньше), время: ×1,138, качество не изменилось (eval loss 2,04966 → 2,04965).
Все 10 проверок `harness/check_rules.py` проходят. Как запустить: см. раздел «How to run».

## What we did

The fixed task (`task/task.yaml`) is not changed: same model, data, batches, steps, LoRA, optimizer
and Soup outputs. `solution.apply()` patches transformers, TRL and bitsandbytes at run time.

| Kernel (`@triton.jit`) | File | What it replaces |
|---|---|---|
| `head_loss_rows_kernel` | `solution/head_loss.py` | the full `[batch, seq, 151,936]` logits, the fp32 upcast for the loss, TRL's entropy and argmax-accuracy passes, and the softmax gradient |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` (about eight element-wise PyTorch kernels per call) |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | bitsandbytes' three-step NF4 dequantization (scales, offset, weights) in the forward and backward of every 4-bit linear |

`solution/swiglu.py` is the example from the starter kit; it runs next to our kernels and is not
counted as ours.

**Chunked head + loss (the memory win).** The head runs on 512 rows at a time. For each chunk the
kernel reads a row of logits in three passes and returns the row's cross-entropy, its entropy, whether
`argmax == label`, and overwrites the row with `d loss / d logits`. The gradient on the hidden states
is `dlogits @ W` (the head is frozen, so there is no weight gradient). The chunk's logits are freed
before the next chunk. The model returns `logits=None`; our `SFTTrainer.compute_loss` patch logs
`entropy`, `num_tokens` and `mean_token_accuracy` with TRL's formulas and hands the per-token losses
to Soup's rewind recorder, so Soup's outputs keep the same values. Gradients are written in fp16 with
a 4096x scale (undone in fp32 afterwards) so that small probabilities do not underflow.

**Fused RMSNorm.** One program per row; statistics in fp32, rounding to the input dtype exactly as
PyTorch does. Backward returns the input gradient only (norm weights are frozen); a trainable weight or
a CPU tensor takes the original code.

**NF4 dequantization.** One launch reads a packed byte (two weights, high nibble first), rebuilds the
scale from the 8-bit absmax, the second-level absmax and the offset (double quantization), and writes
both weights in the compute dtype. Anything that is not NF4 with double quantization on the GPU falls
back to bitsandbytes.

## How to run

On a Kaggle T4 (GPU T4 x2, Internet on): open the public notebook above, or import
`templates/kaggle_notebook.ipynb`, set `REPO_URL` to this repository and `COMMIT` to the submitted
commit, and press Run All (about 10 minutes). Locally on a Linux machine with a CUDA T4:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q           # our kernels against PyTorch / bitsandbytes
python harness/run_task.py             # base, then solution -> runs/result.json
python harness/check_rules.py          # the 10 checks -> runs/checks.json
```

## Results

Tesla T4, one run each, from `harness/run_task.py` and `harness/check_rules.py`
(Kaggle numbers: see the notebook above; they replace the Colab numbers below if they differ).

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 89.7 | 75.6 | speedup x1.186 |
| Peak GPU memory, MiB | 14,345 | 8,383 | base/ours x1.711 |
| Eval loss, public rows | 2.04963 | 2.04964 | +0.000% |



All 10 checks pass: our kernels take 9.1 % of the training time (check 4); first-step loss gap 0.01 %,
mean gap 0.01 % (check 6); Soup's 79 regression tests pass with the patches (check 10).

## Limits

- The speed gain is small (about 12 %): matrix multiplications and attention are untouched, and the
  chunked head trades a few extra launches for memory. Run-to-run noise of the base is about +-3 s.
- One run per side on the public rows; the jury uses hidden rows, another seed and two runs per side.
- The patches assume Qwen2 with a frozen, tied head; anything else (trainable head, soft-capped
  logits, Liger, `compute_loss_func`) takes the original code path.
- Next steps: fuse RoPE and the LoRA branches, recompute cheap activations, and keep the residual
  stream in fp16 to lower memory further.

## Sources and credits

- Design idea of a chunked head with TRL metrics: Soup's own `stream_loss_chunk` option (#1693,
  `soup_cli/utils/stream_loss_chunk.py`), which is off here. Our kernels and patches are written
  for this task and do not copy its code.
- NF4 layout (nibble order, double quantization) read from the bitsandbytes 0.50.2 source.
- `solution/swiglu.py`: the example from the starter kit,
  https://github.com/MakazhanAlpamys/soup-kernel-challenge.

