# Submission: A.I.F.R-Company

## Links
- Repository, final commit: https://github.com/dendybug/decentrathon-2026/commit/74c039dd44350a3c3d2539e56bf8ea79bee5f63e
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README):
  https://www.kaggle.com/code/askhatlishev/notebook3fe5abc70d
- Demo video (up to 3 minutes): https://drive.google.com/file/d/1kvgYu2DpupRy11XhPS8k2W8rIG7QQ95R/view?usp=sharing
- Presentation: https://docs.google.com/presentation/d/1EBysEs42Wmu2EEo97aiWRJlthyPr-u3x/edit?usp=sharing&ouid=115017380854967258274&rtpof=true&sd=true

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Алишев Асхат Талгатович | Team Captain | dendybug | askhatlishev | |
| Абдуллин Радион | Team Member | A_I_F_R_Company | — | — |

## Our kernels


| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 89.7 | 75.6 | speedup x1.186 |
| Peak GPU memory, MiB | 14,345 | 8,383 | base/ours x1.711 |
| Eval loss (public rows) | 2.04963 | 2.04964 | +0.000% |




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



- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (head + loss over a 151,936-token vocabulary, RMSNorm, NF4
dequantization), how the kernels work, results against the base (speed, memory, quality), limits and
next steps.

## Limitations
Time reduction about 16 %; one run per side on public rows; patches assume Qwen2 with a frozen,
tied output head. Next: fuse RoPE and LoRA branches, recompute cheap activations, fp16 residual stream.
