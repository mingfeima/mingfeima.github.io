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

Gated Delta Network (GDN) is a linear attention style layer. During prefill, it processes a long sequence by chunks. The computation naturally splits into two phases:

| Phase | Formula sketch | Role |
|-------|----------------|------|
| Intra-chunk | `A = K @ K^T` with causal / decay masking<br>`A_solve = solve_lower_triangular(I + A)`<br>`w, u = recompute(A_solve, K, V, beta, gate)` | Build the local correction terms inside one chunk. |
| Inter-chunk | `O_i = local_attention_i + recurrent_state_i`<br>`state_i = state_{i-1} * decay + K_i^T @ V_i'` | Carry the recurrent state from previous chunks and produce the final output. |

This split is also the first optimization hint: intra-chunk work is local to a chunk, while inter-chunk work is ordered by the recurrent state.

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

The FLA/Triton implementation decomposes prefill into several kernels. Besides the math operation, the important thing to notice is the parallel dimension of each loop:

| Step | Kernel / loop | Parallel dimension | Materialized output |
|------|---------------|--------------------|---------------------|
| 1 | `chunk_local_cumsum` | `[B, NT, Hv]` | cumulative gate `g` per chunk |
| 2 | `chunk_gated_delta_rule_fwd_kkt_solve` | `[B, NT, Hv]` or `[B, NT, H]` depending on GQA layout | solved local attention matrix `A_solve` |
| 3 | `recompute_w_u` | `[B, NT, Hv]` | `w`, `u` |
| 4 | `chunk_gated_delta_rule_fwd_h` | `[B, Hv]` with a sequential scan over chunks | recurrent chunk state / `v_new`-like temporary |
| 5 | `chunk_fwd_o` | `[B, NT, Hv]` | final output chunk |

This table is a useful way to read a GPU kernel pipeline. If two rows have different parallel dimensions, they are usually not good candidates for direct loop fusion. If they have the same dimension and the output of the first row is consumed immediately by the second, fusion becomes interesting.

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

The optimized CPU path keeps the same mathematical boundary between intra-chunk and inter-chunk work, but compresses the loops. The easiest way to see it is side-by-side:

| Triton loop pipeline | CPU fused loop |
|----------------------|----------------|
| `chunk_local_cumsum`<br>parallel over `[B, NT, Hv]`<br>materializes cumulative `g` | Same preparation step. This one is still kept separate because the later stages reuse cumulative gate values. |
| `chunk_gated_delta_rule_fwd_kkt_solve`<br>parallel over chunk/head domain<br>materializes `A_solve` | **Intra stage**, parallel over `[NT, H]`:<br>`decay_mask` + `K @ K^T` + triangular solve + `recompute_w_u`.<br>**`A_solve` is not materialized globally**; it stays in thread-local `attn2`. |
| `recompute_w_u`<br>parallel over `[B, NT, Hv]`<br>materializes `w`, `u` | Folded into the same intra loop. `w` and `u` are still materialized because they cross from the intra domain to the inter domain. |
| `chunk_gated_delta_rule_fwd_h`<br>parallel over `[B, Hv]`, sequential over chunks<br>materializes recurrent chunk state / `v_new`-like temporary | **Inter stage**, parallel over `[num_seqs, Hv]` and sequential over chunks inside each lane. The recurrent state is kept in FP32 and updated in place. **`h` / `v_new` are not materialized globally**. |
| `chunk_fwd_o`<br>parallel over `[B, NT, Hv]`<br>materializes final output | Folded into the inter loop by changing loop order: compute local attention, recurrent contribution, output accumulation, and state update together. |

The key point is that CPU does not fuse everything into one loop. It only fuses across rows where the parallel dimension can be made compatible without introducing cross-thread dependency. There is also a throughput trade-off: for GQA, looping over `[num_seqs, H]` can reuse `q @ k^T` across value heads and save one GEMM, but for Qwen 3.5 `H = 16` is too small to fill a 32-core CPU when `num_seqs` is small. The inter stage therefore uses `[num_seqs, Hv]` to expose more parallel work, accepting some duplicated local attention compute.

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

There is an alternative design: parallel over `[num_seqs, H]`, then loop over the `HG = Hv / H` value heads inside each K head group. This would allow `q @ k^T` to be computed once and reused, similar to the intra stage. The reason not to use it here is practical parallelism. On Qwen 3.5, `H = 16`; with a single sequence, `[num_seqs, H]` exposes only 16 independent lanes, which is not enough to occupy a 32-core CPU. Using `[num_seqs, Hv]` increases the parallel work, at the cost of duplicating the small local `q @ k^T` computation across value heads.

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

In a generic norm kernel, input is often read twice: one pass for reduction and one pass for scaling. When the reduction dimension is small, the second pass usually hits L1, so this is acceptable. But for fixed small dimensions such as `D = 128`, the whole row can be kept in AVX512 registers: 128 BF16 values fit in 8 vector registers. In this special case, the kernel can load once, reduce, compute the scale, and then write the normalized result.

This trick applies when the reduction dimension is a compile-time small constant, for example `128` or `192`.

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
