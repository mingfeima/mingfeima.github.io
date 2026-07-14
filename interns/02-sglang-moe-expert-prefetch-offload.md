# SGLang MoE Expert Prefetching with Weight Offload

Use SGLang's weight offload path to serve larger MoE models on small-memory Intel
XPU cards, then use expert-route prediction to prefetch likely-needed experts so
expert transfer can overlap with GPU computation.

## Background

### Paper: Forecasting MoE Expert Movement

Large MoE models are attractive because only a subset of experts is activated per
token, but the expert routing pattern also creates heavy and irregular data
movement. The paper [Patterns behind Chaos: Forecasting Data Movement for
Efficient Large-Scale MoE LLM Inference](https://arxiv.org/abs/2510.05497)
studies this behavior with data-movement-centric profiling across recent
large-scale MoE models. Its core message is that expert selection looks chaotic,
but it has temporal and spatial structure that a serving system can exploit.

Main ideas from the paper:

- Expert routing has temporal correlation across adjacent layers, adjacent
   tokens, and prefill/decode stages.
- Prefill expert-selection traces can help predict early decode-stage expert
   demand.
- Different time scales map to different memory tiers: short-distance layer
   correlation can guide faster caches, while token-level correlation can guide
   slower/larger memory tiers.
- Expert activation is skewed: some experts are much hotter than average and may
   deserve replication or more aggressive caching.
- Expert-pair co-activation is also skewed, so frequently co-activated experts
   should be placed or moved with parallelism and bandwidth in mind.
- Task type and language can shift hot-expert distributions, suggesting
   workload-aware expert placement or migration.

For this intern project, the most relevant pieces are prefill-driven prediction,
recent-token expert reuse, and cross-hierarchy memory management. These can be
adapted from multi-GPU / wafer-scale settings to CPU-to-XPU expert prefetching.

### SGLang Weight Offload

SGLang already has weight offload support that can help when a model is slightly
larger than available GPU memory. For example, SGLang supports CPU offload and
V2 layer-group offload options such as:

The server-argument documentation is in the SGLang repo at
`docs/advanced_features/server_arguments.md` and
`docs_new/docs/advanced_features/server_arguments.mdx`; search for
`--cpu-offload-gb`, `--offload-group-size`, `--offload-num-in-group`, and
`--offload-prefetch-step`. The implementation entry point is
`python/sglang/srt/utils/offloader.py`.

```bash
--cpu-offload-gb N
--offload-group-size 1 \
--offload-num-in-group 1 \
--offload-prefetch-step 1 \
--offload-mode cpu
```

This is useful for serving workloads because SGLang integrates offload with API
serving, tensor parallelism, quantization, and MoE execution. However,
`--cpu-offload-gb` is often misunderstood as full-model offload. In practice,
it is better viewed as a tool for models that are slightly over GPU memory, while
very large models may need more specialized systems such as KTransformers plus
SGLang.

## Project Idea

Combine SGLang's existing weight loading and offload machinery with MoE expert
route prediction:

1. Use SGLang offload so a small-memory card can serve a larger MoE model than
   would normally fit entirely in GPU memory.
2. Predict which experts are likely to be needed soon, using ideas inspired by
   the MoE expert-selection and data-movement patterns in the paper.
3. Prefetch those experts from CPU memory to GPU memory early enough to overlap
   transfer with current GPU computation.
4. Measure whether prediction-guided expert prefetch reduces stalls and improves
   end-to-end serving throughput or latency.

This is especially relevant for consumer or low-memory Intel XPU cards such as
BMG, where GPU memory capacity is often the limiting factor.

## SGLang Offload Starting Point

Start from SGLang's existing offload options:

| Option | Role |
|--------|------|
| `--cpu-offload-gb N` | Reserve `N` GB of CPU RAM for offload. Useful for KV / partial weight pressure, but not full-model offload by itself. |
| `--offload-group-size` | Size of each layer group in V2 offload. |
| `--offload-num-in-group` | Number of layer groups to offload. |
| `--offload-prefetch-step` | How many steps ahead SGLang prefetches offloaded weights. |
| `--offload-mode cpu` | Use CPU as the offload target. |

A minimal V2 offload experiment can start with:

```bash
--offload-group-size 1 \
--offload-num-in-group 1 \
--offload-prefetch-step 1 \
--offload-mode cpu
```

Start by understanding where SGLang decides which weights are kept on GPU, which
weights are offloaded, and how prefetch is scheduled today.

## Expert Prediction Direction

The core research question is whether expert routing has enough predictability to
hide expert movement cost.

Possible predictors:

- last-token or recent-window expert reuse
- per-layer expert popularity statistics
- request-level or prompt-level expert locality
- prefill-aware expert placement or prefetch policies
- lightweight online predictors that update during serving

The predictor does not need to be perfect. A useful system only needs to prefetch
experts early enough that the transfer is mostly hidden, while avoiding too many
wrong-prefetches that evict useful experts or waste bandwidth.

## Evaluation Plan

Compare at least three modes:

1. SGLang baseline without expert prediction.
2. SGLang offload with the existing static prefetch behavior.
3. SGLang offload with route-prediction-guided expert prefetch.

Measure:

- peak model size that can be served on the target XPU
- end-to-end throughput
- time-to-first-token and inter-token latency
- expert transfer volume and transfer latency
- prefetch hit rate and wrong-prefetch rate
- GPU idle time caused by waiting for expert weights
- CPU-to-GPU bandwidth usage

The main success criterion is not only fitting a larger model, but also making
offload practical by overlapping expert movement with useful GPU compute.

## Suggested Milestones

1. Study SGLang's current weight offload implementation and reproduce a V2 CPU
   offload run with a small MoE model.
2. Add instrumentation for expert routing, expert residency, prefetch timing,
   transfer latency, and GPU wait time.
3. Implement a simple baseline predictor, such as recent-window expert reuse or
   per-layer expert popularity.
4. Connect the predictor to SGLang's offload / prefetch path so predicted experts
   can be moved before they are needed.
5. Benchmark baseline offload vs prediction-guided prefetch on BMG or another
   small-memory Intel XPU.
6. Analyze where prediction helps, where it fails, and whether wrong-prefetches
   or bandwidth contention dominate.

## References

- [Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference](https://arxiv.org/abs/2510.05497)
- [SGLang](https://github.com/sgl-project/sglang)
