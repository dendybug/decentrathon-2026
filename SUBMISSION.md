# Submission: <TEAM NAME>

## Links
- Repository, final commit: https://github.com/dendybug/decentrathon-2026/commit/<FINAL_SHA>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README):
  https://www.kaggle.com/code/askhatlishev/notebooka4d91d51f1
- Demo video (up to 3 minutes): <VIDEO_LINK>
- Presentation: <PRESENTATION_LINK>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| <NAME> | <ROLE> | dendybug | <KAGGLE_USER> | <CONTACT> |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `head_loss_rows_kernel` | `solution/head_loss.py` | full logits, fp32 loss upcast, TRL entropy and accuracy passes, softmax gradient | <X> % |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` | <X> % |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | bitsandbytes NF4 dequantization, forward and backward | <X> % |
| all together | | | 9.1 % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- chunked head with TRL metrics: the idea of Soup's `stream_loss_chunk` option (#1693),
  https://github.com/MakazhanAlpamys/Soup
- NF4 layout: bitsandbytes 0.50.2, https://github.com/bitsandbytes-foundation/bitsandbytes

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.1 | 75.6 | speedup x1.138 |
| Peak GPU memory, MiB | 14,345 | 8,383 | base/ours x1.711 |
| Eval loss (public rows) | 2.04963 | 2.04965 | +0.001% |


(These numbers are from a Tesla T4 on Colab; replace them with the Kaggle run if they differ.)

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (head + loss over a 151,936-token vocabulary, RMSNorm, NF4
dequantization), how the kernels work, results against the base (speed, memory, quality), limits and
next steps.

## Limitations
Small speed gain (about 5 %); one run per side on public rows; patches assume Qwen2 with a frozen,
tied output head. Next: fuse RoPE and LoRA branches, recompute cheap activations, fp16 residual stream.
