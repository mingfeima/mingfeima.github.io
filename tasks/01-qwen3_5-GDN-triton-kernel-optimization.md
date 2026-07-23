# Qwen3.5 GDN Triton Kernel Optimization

Optimize the Qwen3.5 Gated Delta Net (GDN) Triton kernels on Intel XPU, using
the B60 performance gap against NVIDIA L20 as the main investigation target.

## Background

The tracking issue is
[intel-xpu-backend-for-triton#6910](https://github.com/intel/intel-xpu-backend-for-triton/issues/6910).
The issue reports that two GDN kernels used by SGLang Qwen3.5 are much slower on
Intel B60 than comparable CUDA results on NVIDIA L20:

- `chunk_gated_delta_rule_fwd_h`
- `chunk_gated_delta_rule_fwd_intra`

The B60 kernels have already gone through several rounds of tuning, including
tensor descriptor experiments, tile-size tuning, `num_warps` / `num_stages`
tuning, explicit K-loop changes, and reduced-buffer variants to avoid register
spill. The gap is smaller than the original baseline, but still large enough to
matter for Qwen3.5 prefill performance in SGLang.

Recent discussion suggests that `fwd_h` is mostly latency-bound because of the
state recurrence, while `intra` still has more optimization headroom. For
`intra`, the current suspects include `recompute_w_u`, the serial
forward-substitution in `kkt_solve`, small-shape `tl.dot` behavior, and XPU
backend lowering of scalar reduction loops.

SGLang now also has a SYCL-TLA GDN attention implementation under
[`sgl-kernel-xpu/src/sycl/gdn_attn`](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/src/sycl/gdn_attn).
It is currently about 2x faster than the Triton implementation for prefill on
B60, while decode performance is similar. That makes the SYCL-TLA path the most
important local reference for understanding what the Triton version still needs
to close.

## Starting Points

Use the issue links and current SGLang / XPU kernel code as the source of truth:

| Area | Link |
|------|------|
| Tracking issue | [intel-xpu-backend-for-triton#6910](https://github.com/intel/intel-xpu-backend-for-triton/issues/6910) |
| `fwd_h` CUDA kernel | [SGLang `chunk_delta_h.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_delta_h.py) |
| `fwd_h` XPU kernel | [SGLang XPU `chunk_delta_h.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py) |
| `intra` CUDA kernel | [SGLang `chunk_fwd.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_fwd.py) |
| `intra` XPU kernel | [SGLang XPU `chunk_fwd.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_fwd.py) |
| SYCL-TLA comparison implementation | [sgl-kernel-xpu `src/sycl/gdn_attn`](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/src/sycl/gdn_attn) |

The issue also references standalone benchmark scripts for both kernels. Use
those scripts first to reproduce the isolated-kernel numbers before moving to
end-to-end SGLang profiling.

## Project Goal

Optimize the B60 Triton implementation against two performance targets:

1. Shrink the remaining performance gap between B60 Triton and NVIDIA L20 CUDA.
2. Make B60 Triton match or outperform the SYCL-TLA GDN implementation, especially
  for prefill where SYCL-TLA is currently about 2x faster.

If either target cannot be reached, produce a clear explanation for the remaining
gap and identify whether the blocker is kernel structure, Triton XPU lowering,
compiler behavior, or hardware limits.

The work should answer three questions:

1. Which part of the gap is caused by Triton kernel structure and tunable launch
   parameters?
2. Which part is caused by Triton XPU lowering, register allocation, memory
   behavior, or small-shape dot efficiency?
3. Which parts should stay in Triton, and which parts are better handled by SYCL
   kernels for near-term SGLang performance?

## Scope

Focus on the GDN kernels that dominate Qwen3.5 prefill on Intel XPU. Avoid broad
SGLang serving changes unless they are needed to measure the kernels correctly.

Expected work:

- Reproduce isolated benchmark results for `chunk_gated_delta_rule_fwd_h` and
  `chunk_gated_delta_rule_fwd_intra` on B60.
- Reproduce the relevant Qwen3.5 SGLang prefill profiling command from the issue
  discussion.
- Compare the current Triton kernels against the SYCL-TLA implementation path.
- Profile `fwd_h`, `recompute_w_u`, and `kkt_solve` with `unitrace` or another
  XPU-capable profiler.
- Inspect generated Triton / MLIR / LLVM / device code for the suspected hot
  loops when profiling points to lowering overhead.
- Extend the investigation beyond the GDN Triton kernel implementation itself
  when needed, including fixes or improvements in
  [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton),
  where transpose-matrix mapping is currently problematic.
- Try targeted kernel changes rather than broad rewrites.

## Suggested Investigation Plan

### 1. Reproduce the Baseline

- Install an SGLang XPU environment and the Triton XPU wheel version used by the
  current issue discussion.
- Run the isolated `fwd_h` benchmark script on B60.
- Run the isolated `intra` benchmark script on B60.
- Record device name, driver, oneAPI, PyTorch, SGLang, and Triton XPU versions.
- Compare reproduced numbers with the issue tables before making changes.

### 2. Profile the Current Kernels

- Profile `chunk_gated_delta_rule_fwd_h` and separate recurrence cost from memory
  traffic.
- Profile `chunk_gated_delta_rule_fwd_intra` and break down time across
  `recompute_w_u`, `kkt_solve`, and dot-heavy regions.
- Measure bandwidth utilization, occupancy, stalls, L1 / cache reuse when
  counters are available.
- Confirm whether `kkt_solve` is latency-bound rather than bandwidth-bound.
- Confirm whether `recompute_w_u` reaches the bandwidth roofline after tuning.

### 3. Tune and Test Triton Variants

- Sweep `num_warps` and `num_stages` for `fwd_h` across the issue shape set.
- Re-test tensor descriptor vs block pointer variants for `fwd_h` on short and
  long sequence shapes.
- Re-enable or extend autotune configs for `recompute_w_u`, including `BV=64` and
  `BV=128` where applicable.
- Test `_MERGE_DOT_PRECISION` choices and document correctness or compiler
  failures separately from performance.
- Isolate small-shape `tl.dot` patterns with microbenchmarks before changing the
  production kernel.

### 4. Compare Against SYCL-TLA

- Build or install the SYCL-TLA GDN kernel implementation from `sgl-kernel-xpu`.
- Run the same isolated kernel shapes against the SYCL path.
- Run the same SGLang Qwen3.5 prefill profiling command against Triton and SYCL
  paths.
- Confirm the expected pattern: SYCL-TLA is about 2x faster for prefill, while
  decode performance is close to Triton.
- Identify which SYCL choices explain the performance difference: memory layout,
  explicit scheduling, subgroup operations, SLM usage, or lower-level compiler
  control.

### 5. Decide the Landing Strategy

- Keep Triton-only changes when they are simple, correct, and improve both
  isolated and end-to-end results.
- Use SYCL-TLA kernels for near-term serving performance when Triton lowering or
  latency limitations block progress.
- File focused Triton XPU backend issues for compiler or lowering problems that
  cannot be fixed cleanly in SGLang kernel code.

## Deliverables

- Reproduction notes with exact environment and commands.
- Baseline tables for isolated kernels and SGLang prefill.
- Profiling notes for `fwd_h`, `recompute_w_u`, and `kkt_solve`.
- A small set of Triton kernel patches or tuning patches, if they improve the
  measured results.
- A Triton-vs-SYCL-TLA comparison explaining which path should be used in SGLang.
- Follow-up issues for Triton XPU backend problems that need compiler-side work.

## References

- [intel-xpu-backend-for-triton#6910](https://github.com/intel/intel-xpu-backend-for-triton/issues/6910)
- [SGLang XPU installation guide](https://docs.sglang.io/docs/hardware-platforms/xpu)
- [SGLang](https://github.com/sgl-project/sglang)
- [sgl-kernel-xpu `src/sycl/gdn_attn`](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/src/sycl/gdn_attn)