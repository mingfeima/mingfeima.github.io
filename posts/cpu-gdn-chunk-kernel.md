---
layout: post
title: Optimizing GDN Chunk Prefill on Intel AMX CPU
permalink: /posts/cpu-gdn-chunk-kernel.html
description: Loop fusion, VNNI packing, and AMX brgemm for GDN chunk prefill in SGLang
eyebrow: Systems · CPU Kernels
date: 2026-06-22
meta: SGLang · FLA · AMX BF16
tagline: >-
  A tutorial-style note on optimizing Gated Delta Network chunk prefill on CPU:
  what can be fused, what should stay separate, and how to avoid materializing
  intermediate tensors from the Triton pipeline.
---

## Background

This note is about optimizing `chunk_gated_delta_rule` for CPU in SGLang. The implementation lives in `sgl-kernel/csrc/cpu/mamba/fla.cpp`.

Gated Delta Network (GDN) is a linear attention style layer. During prefill, it processes a long sequence by chunks. The math has both intra-chunk work and inter-chunk recurrent state update:

```text
within a chunk:
  A       = K @ K^T with causal / decay masking
  A_solve = solve_lower_triangular(I + A)
  w, u    = recompute from A_solve, K, V, beta, gate

across chunks:
  O_i     = local attention output + recurrent state output
  state   = state * decay + K_i^T @ V_i'
```

The original FLA/Triton implementation is a good GPU decomposition. On CPU, launching many small loops and materializing several intermediate tensors is not ideal. The CPU optimization tries to reduce memory traffic and keep hot temporary data in thread-local buffers while using AMX BF16 brgemm for the dense matrix pieces.

This article uses the same style as my PyTorch CPU optimization notes: first explain the basic rule, then walk through the concrete kernel choices.

## Contents

- Basic Knowledge: loop fusion is a parallel-domain problem
- Triton Pipeline: several loops and materialized tensors
- CPU Pipeline: two fused stages
- Optimization I: fuse `kkt_solve` and `recompute_w_u`
- Optimization II: fuse `fwd_h` and `chunk_fwd_o`
- Optimization III: VNNI packing plus elementwise work
- Optimization IV: AVX512 BF16 L2 norm
- Notes on correctness and limitations

## 1. Basic Knowledge: When Can Two Loops Be Fused?

Loop fusion is not just putting two adjacent loops into one loop body. The first thing to check is whether the two loops have the same parallel dimension.

A simple rule:

```text
Loop A can be fused with Loop B if:
  1. they iterate over the same independent work domain, or one domain is a local sub-loop of the other;
  2. the intermediate value produced by A is consumed only inside that same domain;
  3. fusion does not introduce cross-thread dependency or excessive duplicate compute.
```

For this kernel, the important domains are:

```text
intra stage:  [num_chunks, H]
inter stage:  [num_seqs, Hv]
```

They are not the same. So we do not merge the whole intra and inter pipeline into one giant loop. Instead we fuse loops inside each stage:

- Intra stage can fuse `K @ K^T`, triangular solve, and `recompute_w_u`, because they are local to one chunk and one K head, with a small inner loop over value heads in the same head group.
- Inter stage can fuse recurrent state output and state update, because both run per sequence and per value head.

This is the main idea: **fuse loops with matching parallel domains; keep different domains as separate stages.**

## 2. Triton Pipeline

The FLA/Triton implementation decomposes prefill into several kernels. Conceptually it looks like this:

```text
1. chunk_local_cumsum
   g -> cumulative gate per chunk

2. chunk_gated_delta_rule_fwd_kkt_solve
   K, beta, decay -> A_solve

3. recompute_w_u
   A_solve, K, V, beta, gate -> w, u

4. chunk_gated_delta_rule_fwd_h
   w, u, state -> recurrent state and intermediate chunk values

5. chunk_fwd_o
   Q, K, w/u, state -> output
```

This is natural for GPU because each Triton kernel has a clean grid. But it materializes intermediate tensors between kernels:

```text
A / A_solve        [B, NT, H or Hv, C, C]
w, u               [B, T, Hv, D]
h / v_new style    per-chunk recurrent intermediates
```

On CPU, these tensors are expensive for two reasons:

1. They are not large enough to amortize memory traffic like a big GEMM.
2. They often have a short lifetime and are consumed immediately by the next loop.

So the CPU version asks a different question: which intermediate tensors can be kept in the current thread's stack buffer and never written as a global tensor?

## 3. CPU Pipeline: Two Fused Stages

The optimized CPU path keeps the same mathematical boundary between intra-chunk and inter-chunk work, but compresses the loops:

```text
prepare:
  optional Q/K L2 norm
  chunk_local_cumsum(g)

stage 1: intra, parallel over [NT, H]
  decay mask
  K @ K^T
  apply beta and decay
  triangular solve
  recompute w and u

stage 2: inter, parallel over [num_seqs, Hv]
  local chunk attention
  recurrent state contribution
  output accumulation
  state update
```

The public wrapper reflects this two-stage structure:

```cpp
auto g_ = chunk_local_cumsum<CHUNK_SIZE>(g, cu_seqlens, chunk_indices);

auto [w, u, decay_mask] =
    chunk_gated_delta_rule_fwd_intra<CHUNK_SIZE>(
        key_, value, g_, beta, cu_seqlens, chunk_indices);

auto [output, final_state] =
    chunk_gated_delta_rule_fwd_inter<CHUNK_SIZE>(
        query_, key_, w, u, g_, decay_mask,
        initial_state, output_final_state, cu_seqlens, chunk_offsets);
```

It still materializes `w` and `u`, because they bridge the two different parallel domains. But it avoids materializing the larger or shorter-lived intermediates inside each stage.

## 4. Optimization I: Fuse `kkt_solve` and `recompute_w_u`

In the intra stage, the parallel domain is `[NT, H]` where `NT` is number of chunks and `H` is number of K/Q heads.

The CPU loop does:

```text
for each chunk nt and K head h:
  compute attn = K_h @ K_h^T once
  for each value head hv in this GQA group:
    attn2 = -attn * beta_hv * decay_hv
    attn2 = solve_tril(attn2)
    w_hv  = attn2 @ (K_h * beta_hv * exp(g_hv))
    u_hv  = attn2 @ (V_hv * beta_hv)
```

This removes two sources of waste.

First, `A_solve` is not stored to a global tensor. It is held in a thread-local `attn2[CHUNK_SIZE, CHUNK_SIZE]` buffer, consumed immediately by the two brgemm calls that generate `w` and `u`.

Second, with GQA, Q/K heads are fewer than value heads. The Triton path often works after repeating Q/K to value-head count. The CPU path keeps native head indexing:

```text
HG = Hv / H
hv in [h * HG, h * HG + HG)
```

So `K @ K^T` is computed once per K head and reused across value heads in the group.

The important fusion condition is satisfied here:

```text
producer: K @ K^T / solve_tril for one [chunk, h]
consumer: recompute w/u for hv inside the same [chunk, h] group
```

No other thread needs the temporary `attn2`, so it can stay local.

## 5. Optimization II: Fuse `fwd_h` and `chunk_fwd_o`

The inter stage uses a different parallel domain: `[num_seqs, Hv]`.

For each sequence and value head, the loop walks chunks in order. That order is sequential because of recurrent state:

```text
for each sequence bs and value head hv:
  state = initial_state[bs, hv]
  for each chunk in sequence order:
    local attention output
    recurrent output from state
    update output
    update state
```

This is why the whole kernel cannot be fully parallelized over chunks. The state recurrence is the dependency.

But inside one `[sequence, value_head]` lane, the intermediate values are short-lived. The CPU kernel fuses the state contribution and output update with the state update:

```text
v_prime     = u - w @ state
attn_inter  = (q * exp(g)) @ state
output      = local_attn @ v_prime + attn_inter
state       = state * exp(g_last) + (k * exp(g_last - g))^T @ v_prime
```

The fused loop avoids writing per-chunk `h` or `v_new` style tensors to global memory. The recurrent `state` stays as FP32 and is updated in place.

## 6. Optimization III: VNNI Packing Plus Elementwise Work

AMX BF16 brgemm wants the B matrix in a VNNI-friendly layout. A naive implementation would do:

```text
1. apply elementwise scale / beta / decay
2. convert FP32 to BF16
3. pack to VNNI layout
4. call brgemm
```

The CPU kernel fuses these whenever possible.

One example is `pack_vnni2`, used for state packing:

```cpp
// from [K/2, 2, N] FP32 to [K/2, N, 2] BF16
// also update src = src * exp(g_last)
pack_vnni2<scalar_t, D, D>(
    s_packed,
    state_ptr,
    g_last,
    D,
    D);
```

This does two jobs in one pass over memory:

- pack FP32 state into BF16 VNNI layout for brgemm;
- apply `state *= exp(g_last)` to the FP32 state buffer.

This is a typical CPU kernel optimization. If data must be read for layout conversion anyway, attach a cheap elementwise operation to the same pass.

Another example is applying beta and gate before packing `K` and `V`:

```text
k_beta = K * beta * exp(g)
v_beta = V * beta
pack k_beta / v_beta into VNNI layout
```

The goal is not to minimize the number of C++ functions. The goal is to minimize full memory passes over temporary tensors.

## 7. Optimization IV: AVX512 BF16 L2 Norm

The CPU path optionally performs Q/K L2 norm in the kernel. For BF16 input and `D = 64/128`, the kernel uses AVX512 BF16 dot product:

```text
load BF16 vector
vsum = dpbf16(vsum, x, x)
rscale = 1 / sqrt(sum + eps)
out = x * rscale
```
Generically, Norm would require read input twice, when reduction dim size is not big, this would usually hit L1. But for special case,
here for example `D = 128`, we can hold 128x BF16 with 8 avx512 regs and read just once. This trick applies to situation when reducetion
dim size is a constant small number, e.g. 128, 192, etc.

For query, it also fuses the attention scale:

```text
query = l2norm(query) * (1 / sqrt(D))
key   = l2norm(key)
```

This avoids producing extra norm-sum tensors and keeps the operation close to the data layout used by the following AMX path.

## 8. Optimization V: Triangular Work Should Stay Triangular

Several matrices in this algorithm are lower triangular by construction. The decay mask is causal, and the triangular solve only needs the lower part.

The CPU implementation uses compile-time chunk size and small fixed buffers, so upper-triangular work can be skipped explicitly in vectorized kernels. This matters because `CHUNK_SIZE = 64`; saving half of a `64 x 64` inner update is visible when repeated over many chunks and heads.

The idea is simple:

```text
for row i:
  only columns j <= i are meaningful
```

That rule applies both to decay mask generation and to the triangular solve/update kernels.

## 9. Correctness Notes

The tests compare the CPU kernel with a PyTorch reference. One subtle part is state layout:

```text
CPU / Triton-facing state: [N, Hv, V, K]
naive reference state:    [N, Hv, K, V]
```

The test code transposes only to bridge these two conventions. The kernel itself follows the same VK layout as the Triton-facing implementation.

The current CPU path focuses on the production shape used by Qwen 3.5-style models:

```text
B = 1 packed varlen batch
CHUNK_SIZE = 64
D = Dv = 64 or 128, multiples of 32
BF16 activations, FP32 recurrent state
```

## Summary

The main optimization is not a single instruction trick. It is choosing the right loop boundary.

Triton uses several clean kernels with global intermediates. The CPU version keeps only the stage boundary that changes the parallel domain:

```text
intra: [num_chunks, H]
inter: [num_seqs, Hv]
```

Inside each domain, it fuses aggressively to avoid materializing intermediate tensors. Around the fused loops, it uses AMX BF16 brgemm, VNNI packing, AVX512 BF16 reductions, and in-place FP32 state update.

This pattern is broadly useful for CPU kernel work:

1. identify the true parallel domain;
2. keep loop fusion inside that domain;
3. avoid global tensors whose lifetime is only one loop body;
4. combine layout conversion with cheap elementwise transforms;
5. keep recurrent state in FP32 and update it where it is consumed.

## References

- CPU implementation: `sgl-kernel/csrc/cpu/mamba/fla.cpp`
- Vector helpers: `sgl-kernel/csrc/cpu/vec.h`
- CPU tests: `test/registered/cpu/test_mamba.py`
- SGLang branch: `pr_flash_linear_attn_opt`
