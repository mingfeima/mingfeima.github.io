# Enable Qwen3-TTS on SGLang Omni for Intel XPU

Enable the SGLang Omni path for Qwen3-TTS on Intel XPU, then profile and optimize
GPU utilization and kernel hotspots. The goal is to make TTS serving usable on
BMG-class consumer Intel GPUs, not only on high-end data-center GPUs.

## Background

[vLLM's Omni TTS engineering blog](https://vllm.ai/blog/2026-06-23-vllm-omni-tts)
uses Qwen3-TTS as the main example of a full TTS optimization path. The key
lesson is that TTS serving is not just text LLM serving with a different output
format. It is a multi-stage pipeline with different bottlenecks in each stage.

Qwen3-TTS has a standard two-stage TTS structure:

1. **Talker**: an autoregressive model that predicts codec tokens. This stage is
   latency-sensitive and often runs with `q_len=1` decode steps.
2. **Connector**: moves generated codec tokens from Talker to Code2Wav. In
   streaming mode, connector chunk size controls Time To First Audio Packet
   (TTFP) and interacts with audio quality.
3. **Code2Wav**: reconstructs waveform audio from codec tokens. This stage is
   more parallel and throughput-oriented than the Talker.

The vLLM blog reports several practical Qwen3-TTS optimization themes that are
useful references for SGLang Omni:

- decouple connector streaming chunks from the Code2Wav decode window
- batch Talker decode preprocessing at high concurrency
- remove Python hot-path overhead in per-token decode loops
- avoid repeated CPU work and small CPU-to-GPU transfers
- use graph capture or stable-shape buckets when decode shapes are predictable
- validate with TTFP, end-to-end latency, and audio throughput

## SGLang Starting Point

SGLang already has multimodal and Omni-related surfaces. Start by inspecting
these areas in the SGLang repo:

- `python/sglang/srt/configs/qwen3_omni.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/mm_utils.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/detokenizer_manager.py`
- `python/sglang/srt/managers/scheduler_components/output_streamer.py`

The first task is to identify which parts of the existing Qwen3-Omni / audio
path can be reused for Qwen3-TTS, and which parts need a new model adapter,
connector, tokenizer/codec handling, or output streaming path.

## Project Goal

Enable Qwen3-TTS on SGLang Omni for Intel XPU and make it performance-debuggable:

1. Load and run Qwen3-TTS through the SGLang Omni serving stack on Intel XPU.
2. Split the runtime into Talker, connector, and Code2Wav stages with clear
   timing and queueing metrics.
3. Analyze GPU utilization for each stage and for the full pipeline.
4. Identify GPU kernel hotspots and CPU-side serving overhead.
5. Improve the bottlenecked path with XPU-friendly batching, staging, graphing,
   or kernel changes.

## Concrete Plan

### 1. Bring up Qwen3-TTS correctness on XPU

- Add or adapt the model config / loader path needed for Qwen3-TTS.
- Confirm tokenizer, codec-token format, speaker / reference-audio inputs, and
  waveform output format.
- Run a minimal non-streaming request on Intel XPU and compare output shape,
  token counts, and basic audio sanity against a reference run.
- Keep the first target narrow: one Qwen3-TTS variant, one dtype, one device, and
  one simple request format.

### 2. Expose TTS pipeline stages in SGLang Omni

- Make Talker, connector, and Code2Wav visible as separate timed regions.
- Track queue time and execution time per stage.
- Track chunk sizes: codec chunk frames, initial codec chunk frames, decode chunk
  frames, and left-context frames if supported.
- Emit metrics for TTFP, end-to-end latency, generated audio seconds per second,
  and request throughput.

### 3. Profile GPU utilization

- Run concurrency sweeps such as `c=1, 4, 8, 16, 32, 64`.
- Record per-stage GPU utilization, wall time, queue time, and audio throughput.
- Look for the common TTS failure mode: the GPU is underutilized because the
  Talker decode loop, Python preprocessing, connector staging, or small kernel
  launches keep the device waiting.
- Separate Talker utilization from Code2Wav utilization; averaging the full
  process can hide that one stage is starved while another is overloaded.

### 4. Profile GPU kernel hotspots

Use Intel XPU profiling tools to identify where time goes:

- `unitrace` for SYCL / Level Zero timeline and device timing
- Intel VTune GPU Hotspots for EU active / stall analysis when available
- PyTorch profiler or SGLang internal timers for framework-level operator and
  Python overhead attribution

The hotspot report should classify time into:

- Talker attention / MLP / logits kernels
- Talker preprocessing and embedding construction
- connector transfer or staging overhead
- Code2Wav decoder kernels
- CPU-side scheduling, Python loops, synchronization, and small tensor allocation
- host-to-device or device-to-host copies

### 5. Optimize the first confirmed bottleneck

Pick the first bottleneck from profiling rather than guessing. Likely candidates:

- batch Talker decode preprocessing across requests
- move repeated mel/STFT or speaker-embedding preparation to cached GPU buffers
- remove per-step tensor slicing / concatenation in sliding-window state
- decouple connector chunk size from Code2Wav decode window
- bucket stable Code2Wav shapes for graph capture or compiled execution
- fuse or specialize small q_len=1 decode kernels on XPU when generic kernels
  dominate
- reduce CPU/GPU synchronization and tiny H2D/D2H copies

Each optimization should include before/after measurements for utilization,
latency, throughput, and kernel timeline.

## Suggested Milestones

1. Enable Qwen3-TTS on Intel XPU through SGLang Omni. Find the existing
  SGLang Omni/audio path, add the minimal Qwen3-TTS model adapter, and run one
  correctness-checked request on XPU.
2. Map the relevant vLLM Omni Qwen3-TTS optimizations to SGLang. Start with the
  optimizations that fit this pipeline best, such as connector chunk decoupling,
  batched Talker decode preprocessing, hot-path cleanup, GPU-side cached buffers,
  and stable-shape Code2Wav graph/bucket handling.
3. Profile and optimize based on evidence. Break down GPU utilization by Talker,
  connector, and Code2Wav; use `unitrace`, VTune, PyTorch profiler, or SGLang
  timers to identify kernel hotspots; then implement the highest-impact
  optimization and re-measure TTFP, end-to-end latency, audio throughput,
  request throughput, GPU utilization, and kernel timeline.

## References

- [Engineering TTS Inference in vLLM-Omni](https://vllm.ai/blog/2026-06-23-vllm-omni-tts)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [SGLang](https://github.com/sgl-project/sglang)
