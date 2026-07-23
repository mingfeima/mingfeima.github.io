# DuoServe-MoE on SGLang XPU

Implement a DuoServe-MoE-style expert offload, cache, and prefetch path for
SGLang XPU, with separate strategies for prefill and decode on PCIe-only Intel
GPU systems such as 4x Arc Pro B60.

## Background

[DuoServe-MoE](https://arxiv.org/abs/2509.07379) argues that prefill and decode
should not share one expert cache policy. The core contribution is the phase
separation: prefill and decode have different expert activation distributions,
so they need different expert residency, prefetch, and eviction strategies.

Traditional MoE offloading keeps expert weights in CPU memory and moves selected
experts to GPU memory over PCIe when needed. A single cache policy across all
phases is inefficient because the bottleneck changes across the request:

- In prefill, thousands of prompt tokens make the union of routed experts nearly
  dense. Expert cache hit rate is low because most experts in a layer may be
  touched anyway. The main bottleneck becomes PCIe copy and peak GPU residency.
- In decode, `q_len=1` makes expert activation sparse and often stable across
  steps or adjacent layers. A hot expert cache and predictive prefetch can hide
  PCIe latency and reduce tail latency.

This distinction matters even more on PCIe-only Intel GPU systems than on
NVLink/NVSwitch systems. On B60, the expert GEMM may not be the only limiting
factor; CPU-to-GPU and GPU-to-GPU expert weight movement can dominate latency if
it sits on the critical path.
Therefore, the theoretical benefit of DuoServe-style phase-separated expert
management should be higher on PCIe-only cards than on systems with high-bandwidth
GPU interconnects, because there is more exposed transfer latency to hide or
avoid.

## Starting Points

Use the DuoServe-MoE paper and SGLang's XPU serving path as the starting point:

| Area | Link |
|------|------|
| DuoServe-MoE paper | [arXiv:2509.07379](https://arxiv.org/abs/2509.07379) |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| SGLang XPU docs | [SGLang XPU installation guide](https://docs.sglang.io/docs/hardware-platforms/xpu) |

The goal is not to copy DuoServe-MoE's CUDA implementation details directly.
Instead, map its phase-separated expert management idea to SGLang XPU, using
Intel-friendly async copy, stream/queue overlap, and expert cache management.

## Project Goal

Build a SGLang XPU expert offload runtime that treats prefill and decode as
different workloads.

The expected target workload is DeepSeek V4 on 4x B60. The primary comparison is
SGLang's current EPLB-based behavior versus DuoServe-style phase-separated expert
management.
Because 4x B60 is PCIe-only, the expected upside should come primarily from
reducing exposed PCIe wait time and improving decode expert cache hit rate, not
from faster expert GEMM alone.

The work has four targets:

1. Implement a prefill path that overlaps expert weight transfer with non-MoE
   compute and minimizes expert residency time on GPU.
2. Implement a decode path with a hot expert cache and optional lightweight expert
   predictor for async prefetch.
3. Evaluate whether phase-separated expert management improves QoS for DeepSeek
  V4 on 4x B60 compared with SGLang EPLB.
4. If the approach does not improve latency or memory usage, explain whether the
   limit is PCIe bandwidth, copy/compute overlap failure, cache hit rate,
   predictor accuracy, expert GEMM cost, or SGLang runtime overhead.

## Scope

Focus on MoE expert weight offload, prefetch, cache residency, and latency QoS in
SGLang XPU. This task should not replace expert GEMM kernels or MoE routing
logic unless instrumentation shows they are blocking overlap.

Expected work:

- Identify the SGLang XPU MoE execution path where expert weights are loaded,
  cached, evicted, and used for expert GEMM.
- Add instrumentation for expert activation, expert residency, transfer time,
  wait time, cache hit/miss rate, and GPU memory footprint.
- Implement a prefill-specific expert scheduling path that avoids long-lived
  expert cache assumptions.
- Implement a decode-specific hot expert cache policy.
- Add optional decode expert prediction based on activation traces, popularity,
  and inter-layer affinity.
- Keep prefill and decode policies independently configurable so they can be
  evaluated separately.
- Prioritize DeepSeek V4 on PCIe-only 4x B60 and report whether the same design
  would generalize to other XPU systems.

## Architecture Plan

### 1. Baseline and Instrumentation

- Establish the current SGLang XPU EPLB baseline for DeepSeek V4 on 4x B60.
- Measure TTFT, end-to-end latency, P50/P95/P99 latency, tokens/s, and peak GPU
  memory.
- Record per-layer expert activation density for prefill and decode separately.
- Record expert transfer time, exposed PCIe wait time, and overlap with compute.
- Record cache hit rate and wrong-prefetch recovery cost when prediction is
  enabled.

### 2. Prefill Pipeline

- Treat prefill as dense expert activation across many tokens.
- Avoid relying on long-lived expert cache hits in prefill.
- Use an async copy stream or SYCL queue to prefetch expert weights while
  attention, layernorm, or other non-MoE work is running.
- Keep only the needed expert working set resident and release or recycle expert
  buffers as soon as their computation finishes.
- Measure whether PCIe copy latency is hidden by non-MoE compute and whether
  peak GPU memory falls compared with full-layer prefetch.

### 3. Decode Hot Expert Cache

- Treat decode as sparse and locality-friendly expert activation.
- Maintain a hot expert cache across decode steps.
- Start with simple policies such as LRU, LFU, and per-layer hot expert tracking.
- Add request-aware cache accounting so long-running decode requests can benefit
  from stable expert activation.
- Measure hit rate, miss penalty, and tail-latency impact.

### 4. Decode Predictor

- Collect offline activation traces for the target model and workload.
- Build popularity and inter-layer affinity statistics per layer.
- Start with a lightweight predictor before adding a larger MLP.
- Predict next-layer likely experts from previous activations and layer-level
  statistics.
- Prefetch predicted experts asynchronously before router output reaches the
  next layer.
- On prediction miss, fall back to fetching the router-selected experts and
  record the recovery cost.

### 5. Prefill/Decode Disaggregation

- Evaluate whether SGLang should use separate prefill and decode workers for
  expert cache management.
- Let prefill workers focus on throughput, short expert residency, and transfer /
  compute overlap.
- Let decode workers maintain a larger hot expert cache when memory allows.
- Measure whether PD separation improves QoS compared with one mixed worker using
  one shared expert cache policy.

## Evaluation Plan

Compare these primary modes:

1. DeepSeek V4 on 4x B60 with SGLang's current EPLB-based behavior.
2. DeepSeek V4 on 4x B60 with DuoServe-style prefill pipeline only.
3. DeepSeek V4 on 4x B60 with DuoServe-style decode hot cache only.
4. DeepSeek V4 on 4x B60 with DuoServe-style prefill pipeline plus decode hot
  cache / predictor.
5. Optional DeepSeek V4 on 4x B60 with PD-separated prefill and decode workers.

Measure:

- TTFT
- end-to-end latency
- P50, P95, and P99 latency
- tokens per second
- peak GPU memory and expert residency time
- CPU-to-GPU expert transfer volume and time
- exposed PCIe wait time on the critical path
- overlap efficiency between expert copy and compute
- decode expert cache hit rate
- predictor top-k hit rate and wrong-prefetch rate
- fallback recovery cost after prediction misses

The target platform should include PCIe-only B60. The report should make clear
whether the optimization helps because it improves cache hit rate, hides PCIe
copy, reduces peak expert residency, or simply shifts the bottleneck elsewhere.
The final headline result should be the DeepSeek V4 4x B60 comparison between
SGLang EPLB and the best DuoServe-style configuration.

## Deliverables

- SGLang XPU instrumentation for expert activation, transfer, residency, and
  cache behavior.
- Prefill two-stream / two-queue expert prefetch pipeline.
- Decode hot expert cache implementation.
- Optional lightweight decode expert predictor based on activation traces,
  popularity, and inter-layer affinity.
- Benchmark report comparing DeepSeek V4 on 4x B60 with SGLang EPLB versus
  DuoServe-style phase-separated scheduling.
- Analysis of whether the design is worthwhile on PCIe-only B60 systems and what
  bottleneck remains.

## References

- [DuoServe-MoE](https://arxiv.org/abs/2509.07379)
- [SGLang](https://github.com/sgl-project/sglang)
- [SGLang XPU installation guide](https://docs.sglang.io/docs/hardware-platforms/xpu)