# SFC GEMM Optimization with SYCL-TLA

Implement and evaluate space-filling-curve (SFC) tile scheduling for GEMM on
Intel XPU using SYCL-TLA, with performance modeling as a first-class part of the
work.

## Background

Modern GEMM performance depends not only on the inner compute micro-kernel, but
also on how output tiles are scheduled across the device. Tile traversal order
can affect cache locality, memory traffic, load balance, synchronization, and
the amount of useful overlap between compute and data movement.

Space-filling curves such as Morton/Z-order and Hilbert order provide a way to
map a 1D tile execution sequence onto a 2D GEMM tile grid while preserving more
spatial locality than plain row-major or column-major traversal. This can be
useful when adjacent output tiles reuse nearby A or B matrix panels and when the
cache hierarchy can retain those panels long enough for the schedule to matter.

The goal of this task is not to reproduce a tutorial GEMM. The goal is to build a
performance-model-driven SFC scheduler in SYCL-TLA and determine whether it can
beat oneDNN on selected Intel XPU GEMM shapes. If it cannot, the final result
should explain why from the performance model rather than only reporting that it
was slower.

## Traversal Orders

![GEMM tile traversal orders](../interns/assets/scheduler-traversal-orders.svg)

The figure shows the scheduling idea being evaluated: keep the GEMM mainloop and
tile shape comparable, but change how a 1D work order maps onto the 2D output
tile grid.

- Raster traversal is the control path. It walks linearly through the tile grid
  along one major dimension and then advances to the next row or column.
- Morton/Z-order improves 2D locality with relatively cheap bit-interleaving
  arithmetic.
- Hilbert order usually gives stronger neighbor locality, but has higher mapping
  overhead and may only pay off when cache reuse is the actual limiter.

## oneDNN Baseline Traversal

Treat oneDNN as the production baseline, but do not assume that its internal GEMM
tile traversal is a stable public interface. For this task, the working baseline
is that oneDNN GEMM on Intel XPU uses implementation-specific blocked GEMM
kernels with a conventional linear/raster-style work decomposition, not an
explicit Morton or Hilbert SFC traversal exposed to users.

Part of the work is to verify the exact oneDNN behavior for the target dtype,
layout, and shape family by reading the selected oneDNN implementation path,
profiling the executed kernel, and matching observed cache behavior against the
performance model. If oneDNN uses additional blocking, swizzling, batching, or
kernel selection that changes the effective traversal, document that and compare
SFC against the measured oneDNN behavior rather than against a simplified
assumption.

## Starting Points

Use SYCL-TLA as the implementation framework and oneDNN as the main production
baseline:

| Area | Link |
|------|------|
| SYCL-TLA | [intel/sycl-tla](https://github.com/intel/sycl-tla) |
| oneDNN | [uxlfoundation/oneDNN](https://github.com/uxlfoundation/oneDNN) |

The implementation should modify or extend the SYCL-TLA GEMM scheduling path
rather than building a separate standalone GEMM stack. Existing raster scheduling
and Stream-K-style paths should be treated as reference points, not as the final
deliverable.

## Project Goal

Build an SFC-based GEMM scheduler in SYCL-TLA and evaluate it against oneDNN with
an explicit performance model.

The work has five targets:

1. Implement one or more SFC tile traversal policies, such as Morton/Z-order and
   Hilbert order, in the SYCL-TLA GEMM scheduler path.
2. Achieve additional performance over oneDNN on selected Intel XPU GEMM shapes.
3. Try ahead-of-time weight prepacking and determine whether it provides
  additional steady-state GEMM performance.
4. Try K-split scheduling and map the CA algorithm from
  [arXiv:2601.16294](https://arxiv.org/pdf/2601.16294) into the SYCL-TLA GEMM
  scheduling path.
5. If SFC scheduling, weight prepacking, or K-split / CA-style scheduling does
  not beat oneDNN, provide a concrete performance-model
   explanation for the miss, including whether the limit is cache reuse,
  scheduler overhead, prepack overhead, reduction overhead, memory bandwidth,
  occupancy, DPAS utilization, launch overhead, or shape-specific behavior.

## Scope

Focus on GEMM scheduling, modeling, and measurement. Do not spend project time on
introductory tutorial work, broad framework rewrites, or unrelated epilogue
features unless they are needed for a fair oneDNN comparison.

Expected work:

- Identify the SYCL-TLA point where a linear tile id is mapped to GEMM tile
  coordinates.
- Implement SFC mapping policies suitable for Intel XPU execution.
- Keep the existing raster-style scheduler available as an internal reference.
- Build a benchmark suite from the GEMM hotspot shapes observed in Qwen3.5-9B
  prefill with input lengths 4k and 16k, for both `bfloat16` and `int8_w8a8`.
- Compare the SYCL-TLA SFC scheduler against oneDNN on the same shapes, dtypes,
  layouts, warmup policy, and measurement protocol.
- Build a performance model that predicts when SFC scheduling should help and
  when it should not.
- Try prepacking weights ahead of execution and measure whether it provides
  additional performance beyond SFC scheduling.
- Try K-split scheduling and map the CA algorithm from
  [arXiv:2601.16294](https://arxiv.org/pdf/2601.16294) into the same SYCL-TLA
  evaluation framework.
- Validate the model with profiling counters and measured timing.
- Explain wins and misses from the model rather than relying only on benchmark
  tables.

## Performance Modeling Requirements

The performance model should be developed before final tuning decisions and then
updated as measurements come in.

At minimum, model these components:

- arithmetic intensity and expected compute roofline for each GEMM shape
- A/B/C memory traffic under the baseline scheduler and the SFC scheduler
- expected cache-panel reuse from tile traversal order
- scheduler-indexing overhead for Morton/Z-order and Hilbert mappings
- occupancy and work distribution across compute units
- DPAS utilization limits for the selected tile shapes and dtypes
- sensitivity to the Qwen3.5-9B prefill hotspot shapes at input lengths 4k and
  16k
- conditions under which oneDNN is expected to remain faster

The model should produce a clear prediction before benchmark results are used to
judge success. For example, it should identify shapes where SFC should reduce
memory traffic or cache misses enough to offset mapping overhead, and shapes
where oneDNN should remain ahead because the kernel is already compute-bound or
the relevant panels do not fit in cache.

## Implementation Plan

### 1. Scheduler Design

- Define the scheduler interface changes needed to plug SFC mappings into
  SYCL-TLA GEMM.
- Implement Morton/Z-order as the lower-overhead SFC baseline.
- Implement Hilbert order as the stronger-locality but higher-overhead SFC
  variant.
- Support non-power-of-two tile grids without changing GEMM correctness.
- Keep mapping cost measurable and separable from GEMM compute time.

### 2. Performance Model

- Build an initial model for the Qwen3.5-9B prefill GEMM hotspot shapes at input
  lengths 4k and 16k on the selected BMG/B60 target.
- Estimate expected reuse distance for raster, Morton, and Hilbert traversal.
- Estimate cache capacity pressure for A and B panels.
- Estimate whether the SFC mapping arithmetic can be hidden or whether it is on
  the critical path.
- Use the model to choose the first shape set and the first SFC variant to tune.

### 3. SYCL-TLA Implementation

- Add the SFC scheduler policies in the SYCL-TLA GEMM scheduler path.
- Integrate the policies without changing the mainloop or epilogue unless the
  performance model shows that scheduler-only changes are insufficient.
- Add an experiment for ahead-of-time weight prepacking, keeping prepack cost
  separate from steady-state GEMM timing.
- Add an experiment for K-split scheduling and CA-style mapping from
  [arXiv:2601.16294](https://arxiv.org/pdf/2601.16294), using the same benchmark
  and modeling framework as the SFC schedulers.
- Add correctness validation against a CPU, framework, or oneDNN reference.
- Keep scheduler selection configurable for benchmark runs.

### 4. Profiling and Validation

- Measure kernel time, achieved TFLOPS, and bandwidth for every shape.
- Use `unitrace`, VTune, or another Intel XPU profiler to collect cache, memory,
  and EU utilization counters when available.
- Compare measured cache behavior against the model's predicted reuse benefits.
- Separate scheduler overhead from memory-locality benefits where possible.
- Record cases where profiling disagrees with the model and update the model.

### 5. oneDNN Comparison

- Compare against oneDNN GEMM as the primary external baseline.
- Use matching dtype, layout, batch shape, warmup, iteration count, and problem
  sizes.
- Run two required dtype tracks: `bfloat16` and `int8_w8a8`.
- Use GEMM shapes extracted from Qwen3.5-9B prefill profiling at input lengths
  4k and 16k, prioritizing the GEMMs that dominate total prefill time.
- For `int8_w8a8`, verify with oneDNN verbose output and profiling whether the
  target shapes dispatch to an optimized Intel GPU microkernel / DPAS path or to
  a reference GPU matmul implementation.
- Report both best-case and representative-case results.
- For any shape where SYCL-TLA SFC is faster than oneDNN, explain the source of
  the win using the performance model.
- For any target shape where SYCL-TLA SFC is not faster than oneDNN, explain the
  miss using the performance model and identify the next bottleneck.

## Evaluation Plan

Compare these modes:

1. oneDNN GEMM on Intel XPU.
2. SYCL-TLA GEMM with the existing raster-style scheduler.
3. SYCL-TLA GEMM with Morton/Z-order scheduling.
4. SYCL-TLA GEMM with Hilbert scheduling.

Measure:

- dtype track: `bfloat16` or `int8_w8a8`
- correctness against a trusted reference
- kernel time and achieved TFLOPS
- effective memory bandwidth
- cache hit rate, miss count, or related memory hierarchy counters when available
- EU active / stall metrics when available
- scheduler overhead
- performance-model predicted speedup vs measured speedup
- speedup or slowdown against oneDNN
- incremental speedup from weight prepacking, with prepack cost reported
  separately
- incremental speedup from K-split / CA-style scheduling

For `int8_w8a8`, also report:

- quantization scheme, scales, zero points, and accumulation type
- whether oneDNN dispatches to an optimized GPU kernel or a reference path
- INT8 TOPS or equivalent normalized throughput
- dequantization, requantization, and post-op overhead when present

The final report should include both benchmark tables and a modeling section. A
result that does not beat oneDNN is acceptable only if the model gives a specific
technical reason for the gap and identifies whether SFC scheduling is the wrong
lever for that shape or whether another bottleneck should be attacked next.
Likewise, if weight prepacking or K-split / CA-style scheduling does not provide
additional speedup, explain the miss with concrete evidence, such as reuse being
too low, prepack cost exceeding steady-state benefit, K-split increasing memory
traffic or reduction overhead, CA scheduling not improving locality on the target
shape, or the kernel already being limited by DPAS utilization rather than data
movement.

## Deliverables

- SYCL-TLA SFC scheduler implementation.
- Weight prepack experiment with separate prepack and steady-state timing.
- K-split / CA-style scheduling experiment based on
  [arXiv:2601.16294](https://arxiv.org/pdf/2601.16294).
- Correctness validation for all supported scheduler policies.
- oneDNN comparison benchmark suite for `bfloat16` and `int8_w8a8`, with
  reproducible run commands.
- Performance model for the selected GEMM shapes.
- Profiling report connecting measured behavior to the model.
- Final recommendation on whether SFC scheduling should be kept for Intel XPU
  GEMM and for which shape families.

## References

- [SYCL-TLA](https://github.com/intel/sycl-tla)
- [oneDNN](https://github.com/uxlfoundation/oneDNN)