---
layout: post
title: Optimizing GDN Chunk Prefill on Intel AMX CPU
permalink: /posts/cpu-gdn-chunk-kernel.html
description: Optimizing GDN chunk prefill on Intel AMX CPU for SGLang
eyebrow: Systems · CPU Kernels
date: 2026-06-22
meta: SGLang · chunk_gated_delta_rule_cpu
tagline: >-
  A fused AVX512 + brgemm implementation of Gated Delta Network (GDN) chunk prefill
  for hybrid models such as Qwen 3.5, targeting 4th-gen+ Intel Xeon with AMX.
---

## Background

**GDN (Gated Delta Network)** is a linear attention variant with O(n) prefill cost. Hybrid models interleave GDN layers with standard full attention. In SGLang, the GPU path uses FLA/Triton kernels; on CPU, prefill previously went through a Python/Triton reference stack. This work adds a native CPU kernel in `sgl-kernel/csrc/cpu/mamba/fla.cpp` exposed as `torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu`.

## Triton baseline (4+ kernels)

The FLA prefill pipeline launches several GPU kernels in sequence:

1. `chunk_local_cumsum` — cumulative gate per chunk
2. `chunk_gated_delta_rule_fwd_kkt_solve` — fused K@Kᵀ + triangular solve
3. `recompute_w_u_fwd` — derive `w` and `u` from solved `A`
4. `chunk_gated_delta_rule_fwd_h` — chunk-wise state recurrence + `v_new`
5. `chunk_fwd_o` — inter-chunk attention output

Intermediate tensors (`A`, per-chunk `h`, `v_new`) are materialized in HBM between stages.

## CPU design: two fused stages

The CPU path collapses the above into **two C++ kernels** plus lightweight prep:

- **Intra** (`chunk_gated_delta_rule_fwd_intra`): decay mask, K@Kᵀ, triangular solve, and `recompute_w_u` — without materializing the full `A` matrix to global memory.
- **Inter** (`chunk_gated_delta_rule_fwd_inter`): fuses `fwd_h` + `chunk_fwd_o` — chunk output and in-place FP32 state update without storing per-chunk `h` or `v_new`.

Public entry: `chunk_gated_delta_rule_cpu` (varlen-only, packed batch `B=1`).

## Key optimizations

### AVX512 micro-kernels + AMX brgemm

Hot paths use hand-written AVX512 (l2norm, cumsum, decay mask, triangular solve, elemwise updates). Heavy matmuls go through oneDNN **brgemm** on VNNI2-packed BF16. Recurrent state stays **FP32**; activations are BF16.

### Fusion

- `pack_vnni2` + state decay: packing state for GEMM also applies `state *= exp(g_last)`.
- Decay mask `d[B, NT, Hv, 64, 64]` precomputed once in intra and reused in inter.
- Q L2-norm fuses the `1/√D` scale into the query normalization kernel.

### Native GQA (no Q/K broadcast)

Triton/Python repeats Q/K from `HK` to `HV` heads. CPU indexes `h = hv / HG` directly: Q/K stay `[T, HK, D]`, value/state/gate stay `[T, HV, …]`. Intra parallelizes over `NT × HK` and shares one K@Kᵀ per K-head across `HG` value heads.

### Tail-chunk brgemm padding

brgemm `K` is aligned to `TILE_K=32`. In **inter**, `v' = u - w @ state` is zero-padded before VNNI pack. In **intra**, `k_beta`/`v_beta` are packed with `K=mb_size` while brgemm uses `K=padded_mb_size` — still correct because `attn2` is lower-triangular: for output rows `i < mb_size` and `k ≥ mb_size`, we always have `k > i`, so extra K columns are zero.

## CPU vs Triton

| Aspect | Triton / FLA | CPU (AMX) |
|--------|--------------|-----------|
| Pipeline | 4+ small kernels | 2 fused stages + prep |
| GQA | `repeat_interleave` on Q/K | Native head indexing |
| State | Pool `[V,K]`, often BF16 | Dense FP32, shape `[N, HV, V, K]` |
| Decay exp | `torch.exp` | `fexp_u20` (fast approximate exp) |
| Parallelism | GPU grid | Intra: `NT×H`; Inter: `seq×HV` |
| Head dim | General | Currently `D=128` (Qwen 3.5) |

## Correctness

Validated against a Triton-aligned PyTorch reference with varlen cases and tail chunk lengths (e.g. 17, 33, 50 tokens). `final_state` matches within ~1e-2. Output max diff is ~0.01–0.02 with L2-norm enabled — dominated by BF16 I/O and `fexp_u20` vs FP32 reference, not algorithmic error (mean diff ~1e-4).

## Current limitations

- Varlen-only (`B=1`, FlashAttention-style `cu_seqlens`).
- `D = Dv = 128` only; tuned for Qwen 3.5-style configs.
- State pool gather/scatter via `initial_state_indices` not yet wired on CPU prefill.
- Layout: API shape is `[V,K]`; internal GEMM indexing must stay consistent when `K≠V`.

## References

- Implementation: `sgl-kernel/csrc/cpu/mamba/fla.cpp`
- Triton reference: `python/sglang/srt/layers/attention/fla/chunk.py`
- CPU dispatch: `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`
- SGLang GDN backends: [attention_backend.md](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)
