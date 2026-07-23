# UltraEP on SGLang XPU

Implement an UltraEP-style real-time expert load balancing path for SGLang XPU,
starting with serving prefill and using the current SGLang MoE hot path as the
integration target.

## Background

UltraEP is a precise, real-time load balancing system for large expert-parallel
MoE workloads on rack-scale nodes. The paper is
[arXiv:2606.04101](https://arxiv.org/abs/2606.04101), from Peking University,
Xiaohongshu, Shanghai AI Lab, and collaborators.

Large expert parallelism amplifies expert imbalance into several system-level
problems:

- compute stragglers across ranks
- token all-to-all bottlenecks
- activation memory spikes

Existing approaches such as DeepSeek EPLB use historical expert loads to
periodically place redundant experts. That can work for slow-changing serving
patterns, but training and prefill can change expert hotness quickly across
microbatches, layers, and data domains. Stale historical placement can miss the
current hot experts or even make imbalance worse.

UltraEP moves the balancing decision into the hot path after gating and before
dispatch. For each microbatch and each layer, it observes the current true expert
load and uses a quota-driven planner to jointly decide:

- which experts to replicate
- where to place physical replicas
- how many tokens each physical replica should receive

The planner directly optimizes the post-reroute load upper bound with threshold
binary search. Only replicas that receive useful load are materialized, which is
different from first creating redundant experts and then heuristically splitting
traffic.

The paper connects UltraEP to Megatron-LM for training and SGLang for serving
prefill. Token dispatch and combine still use DeepEP. UltraEP is not a MoE
compute kernel replacement; it is a real-time expert placement and token reroute
layer above the token all-to-all backend.

## Starting Points

Use the paper and current SGLang XPU communication stack as the source of truth:

| Area | Link |
|------|------|
| UltraEP paper | [arXiv:2606.04101](https://arxiv.org/abs/2606.04101) |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| SGLang XPU docs | [SGLang XPU installation guide](https://docs.sglang.io/docs/hardware-platforms/xpu) |
| DeepEP | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) |

At the time of writing, there is no ready-to-use Intel XPU implementation of
UltraEP. SGLang's XPU path has basic tensor-parallel communication, but the MoE
A2A backends such as DeepEP are CUDA-oriented and are not wired to XPU. That
makes XPU MoE expert parallel dispatch the first prerequisite before a full
UltraEP implementation can pay off.

## Project Goal

Build a practical UltraEP path for SGLang XPU prefill in layers:

1. Add a usable XPU MoE all-to-all dispatcher for expert-parallel prefill.
2. Port the UltraEP quota planner and connect it after gating and before token
   dispatch.
3. Add an XPU data plane for expert replica materialization on 4x B60 over PCIe,
  using peer-access USM, Intel SHMEM, Level Zero, or oneCCL depending on what is
  available and performant.
4. Compare UltraEP against SGLang's current EPLB algorithm on Intel XPU.
5. Establish a projected performance target for DeepSeekV4-Flash on 4x B60 and
  measure what fraction of that projection UltraEP can reach.

The first success target is serving prefill. Decode can initially keep UltraEP
disabled because decode is more memory-bound and less sensitive to compute-side
expert imbalance.

## Scope

Focus on SGLang XPU expert-parallel MoE serving. The work should include both
control-plane planning and data-plane communication, but it should be staged so
each layer can be validated independently.

Expected work:

- Add or prototype an XPU MoE A2A dispatcher for prefill, using oneCCL,
  `torch.distributed.all_to_all_single`, or a custom backend as the first
  implementation.
- Use SGLang's current EPLB algorithm as the baseline for UltraEP comparison on
  XPU.
- Port the UltraEP quota-driven planner so it consumes post-gating load and
  emits replica placement plus token reroute decisions per layer and microbatch.
- Integrate the planner into the SGLang hot path:
  `gating -> gather load -> UltraEP.plan() -> materialize replicas -> reroute -> dispatch -> expert GEMM -> combine`.
- Implement expert replica buffers that are not part of checkpoint or optimizer
  state and can be reused across layers.
- Build an XPU expert-copy data path for PCIe-connected B60 cards using Intel
  SHMEM, peer USM, Level Zero, or oneCCL, depending on what the target platform
  supports.
- Compare against SYCL / Triton / oneDNN / ESIMD expert GEMM paths as needed, but
  do not treat UltraEP as a replacement for the expert compute kernel.
- Keep decode support optional until prefill shows clear balancing gains.

## Architecture Plan

### L0. XPU MoE Expert Parallel Baseline

- Use DeepSeekV4-Flash on 4x B60 as the initial bring-up and benchmarking
  target.
- Build a projected performance model for DeepSeekV4-Flash on 4x B60 before
  using UltraEP results as the optimization target.
- Extend SGLang's MoE A2A backend selection with an XPU-capable dispatcher.
- Start with a correctness-first all-to-all path, even if the first version uses
  higher-overhead oneCCL or PyTorch distributed collectives.
- Support irregular token counts per expert and rank for prefill.
- Verify expert dispatch, expert GEMM, and combine correctness before enabling
  live balancing.
- Enable SGLang's current EPLB algorithm on XPU as the primary baseline.

### L1. UltraEP Control Plane

- Implement the hardware-independent quota planner.
- Consume the current microbatch and layer load matrix after gating.
- Use threshold binary search to find the rerouted load upper bound.
- Emit physical replica placement and token assignment decisions.
- Reuse SGLang EPLB metadata concepts where helpful, but make the decision
  real-time and per-layer rather than periodic and history-based.

### L2. XPU Expert Replica Data Plane

- Reserve redundant expert slots per rank and reuse their buffers across layers.
- Build a device-visible peer address table for replica destinations.
- Prototype expert copy with host-driven collectives first if needed.
- Move toward device-side peer copies with Intel SHMEM symmetric heap, SYCL USM
  peer access, or Level Zero peer memory when available.
- Implement tile streaming for large expert weights so copy latency can overlap
  with useful work.
- Add relay-based fan-out for hot experts when direct single-source fan-out
  becomes the bottleneck.

### L3. SGLang Runtime Integration

- Add an `--enable-ultraep` or equivalent serving flag for prefill.
- Keep replica weights out of checkpoint loading and optimizer state.
- Lazily register main expert weight pointers for replica materialization.
- Keep UltraEP independent from the token A2A backend: UltraEP decides physical
  replicas and token landing points; A2A only moves tokens.
- Add fallback behavior so serving can disable UltraEP when PCIe peer copy or the
  selected communication backend is not fast enough.

## Evaluation Plan

Compare these primary modes:

1. SGLang XPU MoE with the current EPLB algorithm.
2. SGLang XPU MoE with UltraEP planning but simple host-driven replica copy.
3. SGLang XPU MoE with UltraEP planning and optimized XPU peer-copy data plane.

Measure:

- prefill throughput
- time-to-first-token for prefill-heavy requests
- rank-level expert load imbalance
- token all-to-all time
- expert GEMM time and straggler time
- replica materialization time and bandwidth
- activation memory spikes
- planner overhead per layer and microbatch
- end-to-end speedup against the SGLang EPLB baseline
- achieved percentage of the DeepSeekV4-Flash 4x B60 projected performance

The target workload should prioritize large-batch and long-prefill cases where
expert imbalance matters most. Decode should be measured, but it is not the first
optimization target.

## Hardware Constraints

UltraEP is only attractive when expert replica materialization is cheaper than
the imbalance it removes. The initial target is 4x B60, where the GPUs are
PCIe-connected.

Because B60 uses PCIe, the real-time weight-copy path may be too expensive unless
the copied expert volume is tightly bounded and the copy can overlap with useful
prefill work. EPLB or slower periodic placement remains the baseline to beat.

Before committing to the full data plane, confirm:

- the available PCIe topology and peer-to-peer bandwidth across the 4x B60 setup
- whether peer memory access is available and stable
- whether device-side put/get or equivalent peer copy can be used
- whether token A2A is fast enough that expert balancing remains the bottleneck

## Deliverables

- XPU MoE A2A dispatcher prototype for SGLang prefill.
- EPLB baseline results on XPU expert parallelism.
- DeepSeekV4-Flash 4x B60 projected performance model and assumptions.
- UltraEP quota planner implementation and integration point in SGLang.
- Correctness tests for replica placement, token reroute, dispatch, and combine.
- Prototype and optimized versions of XPU expert replica materialization.
- Benchmark report comparing SGLang EPLB and UltraEP on prefill.
- Reported percentage of projected DeepSeekV4-Flash performance reached by
  UltraEP.
- Clear recommendation on whether UltraEP is practical on the target Intel XPU
  platform and what communication backend is required.

## References

- [UltraEP paper](https://arxiv.org/abs/2606.04101)
- [SGLang](https://github.com/sgl-project/sglang)
- [SGLang XPU installation guide](https://docs.sglang.io/docs/hardware-platforms/xpu)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)