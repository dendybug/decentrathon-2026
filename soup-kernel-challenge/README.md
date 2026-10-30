
# Soup Kernel Challenge — Decentrathon 6.0

GPU: NVIDIA Tesla T4

Model: Llama 3.1 8B

Optimization: Triton NF4 Dequantization and GRAD_X

## Baseline
Training runtime: 26.99 s

## Triton v1
Training runtime: 34.06 s
Triton calls: 4452
Train loss: 4.32
Eval loss: 3.886

## Current status
NF4 dequantization validated.
GRAD_X numerical correctness validated on synthetic inputs.
GRAD_X v2 optimization in progress.

## Goal
Reduce end-to-end training runtime while preserving correctness.
   
