# CPU Kernel Performance Benchmark Skill

Use this workflow when making a CPU kernel performance optimization and you need a reproducible before/after benchmark against `main`.

The example that motivated this skill was replacing `at::vec::Vectorized<float>::exp()` with `exp_u20()` in `sgl-kernel/csrc/cpu/activation.cpp` for `silu_and_mul_cpu`, then benchmarking Qwen3.5-like CPU activation shapes.

## Goals

- Benchmark the exact kernel path affected by the optimization.
- Use shapes observed from a real model workload when possible.
- Build and install `sgl-kernel` from `main` as the before version.
- Build and install `sgl-kernel` from the current branch as the after version.
- Run the same benchmark script under the same CPU pinning/runtime environment.
- Report latency, normalized per-element cost, and speedup.
- Run correctness tests after the after build.

## Prerequisites

- Activate the Python/build environment with the shell alias used by the machine:

```bash
source ~/.bashrc && sa
```

- The CPU benchmark harness is under:

```bash
/home/mingfeima/sgl-cpu-tests
```

- The CPU benchmark runner is:

```bash
/home/mingfeima/sgl-cpu-tests/run_bench_cpu.sh
```

It pins CPU cores and configures OpenMP/runtime settings. Prefer using it instead of invoking Python directly for CPU microbenchmarks.

- `sgl-kernel` can be rebuilt from the SGLang repo with:

```bash
cd /home/mingfeima/sglang-mingfei/sgl-kernel
./install.sh
```

If `install.sh` does not exist in a fresh worktree, copy the local helper from the working tree or use the equivalent install command:

```bash
uv pip install -v --force-reinstall -Cbuild.verbose=true .
```

## Step 1: Identify the Hot Shape

Prefer shapes from real profiler output, model logs, or a workload-specific benchmark.

For the `silu_and_mul_cpu` optimization, Qwen3.5 CPU profiling showed activation calls like:

```text
sgl_kernel::silu_and_mul_cpu [[1, 18432]]
```

Useful benchmark shapes were:

```python
SHAPES = [
    (1, 18432),     # decode MLP activation
    (17, 18432),    # small non-ideal token count
    (1000, 18432),  # prefill-like token count
]
```

Use both `torch.bfloat16` and `torch.float16` if the kernel supports both.

## Step 2: Write a Focused Benchmark

Place the benchmark script under `/home/mingfeima/sgl-cpu-tests`, for example:

```bash
/home/mingfeima/sgl-cpu-tests/bench_silu_and_mul.py
```

Keep the benchmark simple:

- Import `sgl_kernel` to load `torch.ops.sgl_kernel`.
- Generate fixed random inputs with `torch.manual_seed(...)`.
- Check correctness once before timing.
- Warm up before timing.
- Use enough iterations to reduce noise.
- Print one stable line per shape/dtype that is easy to parse.

Example structure:

```python
import os
from time import time

import torch
import torch.nn.functional as F

import sgl_kernel  # noqa: F401  # load torch.ops.sgl_kernel


torch.manual_seed(1111)

DTYPES = [torch.bfloat16, torch.float16]
SHAPES = [(1, 18432), (17, 18432), (1000, 18432)]


def ref_silu_and_mul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def bench_one(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    ref = ref_silu_and_mul(x)
    out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
    tol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)

    num_tokens = shape[0]
    niters = int(os.environ.get(
        "NITERS", 20000 if num_tokens == 1 else 2000 if num_tokens < 128 else 100
    ))
    warmup = int(os.environ.get("WARMUP", min(20, niters // 10)))

    for _ in range(warmup):
        torch.ops.sgl_kernel.silu_and_mul_cpu(x)

    t0 = time()
    for _ in range(niters):
        torch.ops.sgl_kernel.silu_and_mul_cpu(x)
    t1 = time()

    us = (t1 - t0) * 1_000_000 / niters
    ns_per_elem = us * 1000 / x.numel()
    print(
        f"### silu_and_mul_cpu: shape={shape}, dtype={dtype}, "
        f"niters={niters}, latency={us:.3f} us, ns/elem={ns_per_elem:.4f}"
    )


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())))
    print(f"torch_num_threads={torch.get_num_threads()}")
    for dtype in DTYPES:
        for shape in SHAPES:
            bench_one(shape, dtype)


if __name__ == "__main__":
    main()
```

Run a quick smoke test first:

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sgl-cpu-tests
NITERS=5 WARMUP=1 ./run_bench_cpu.sh bench_silu_and_mul.py
```

## Step 3: Prepare the `main` Worktree

Do not overwrite or stash the current branch unless necessary. Prefer a separate worktree:

```bash
cd /home/mingfeima/sglang-mingfei
rm -rf /tmp/sglang-main-bench
git worktree add /tmp/sglang-main-bench main
```

If the branch is dirty, this keeps current work intact.

## Step 4: Build and Benchmark `main` as Before

Build and install `sgl-kernel` from the `main` worktree:

```bash
source ~/.bashrc && sa
cd /tmp/sglang-main-bench/sgl-kernel
./install.sh
```

If the helper script is not present in that worktree, copy it from the working tree:

```bash
cp /home/mingfeima/sglang-mingfei/sgl-kernel/install.sh ./install.sh
chmod +x ./install.sh
./install.sh
```

Run the benchmark and save output:

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sgl-cpu-tests
./run_bench_cpu.sh bench_silu_and_mul.py | tee /tmp/silu_before_main.log
```

## Step 5: Build and Benchmark the Current Branch as After

Build and install `sgl-kernel` from the current branch:

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sglang-mingfei/sgl-kernel
./install.sh
```

Run the same benchmark and save output:

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sgl-cpu-tests
./run_bench_cpu.sh bench_silu_and_mul.py | tee /tmp/silu_after_branch.log
```

## Step 6: Summarize the Results

Parse both logs and compute speedups:

```bash
python - <<'PY'
import re
from pathlib import Path

pat = re.compile(
    r"shape=\(([^)]*)\), dtype=([^,]+), niters=(\d+), "
    r"latency=([0-9.]+) us, ns/elem=([0-9.]+)"
)


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        match = pat.search(line)
        if match:
            shape, dtype, niters, latency, ns = match.groups()
            out[(shape, dtype)] = (float(latency), float(ns), int(niters))
    return out

before = load('/tmp/silu_before_main.log')
after = load('/tmp/silu_after_branch.log')

print('shape,dtype,before_us,after_us,speedup,before_ns_per_elem,after_ns_per_elem')
for key in before:
    b_us, b_ns, _ = before[key]
    a_us, a_ns, _ = after[key]
    print(f'{key[0]},{key[1]},{b_us:.3f},{a_us:.3f},{b_us / a_us:.3f}x,{b_ns:.4f},{a_ns:.4f}')
PY
```

Example output from the `exp()` to `exp_u20()` change:

| Shape | Dtype | Before | After | Speedup |
|---|---:|---:|---:|---:|
| `(1, 18432)` | bf16 | 10.311 us | 9.103 us | 1.133x |
| `(17, 18432)` | bf16 | 29.097 us | 22.425 us | 1.298x |
| `(1000, 18432)` | bf16 | 462.275 us | 392.141 us | 1.179x |
| `(1, 18432)` | fp16 | 12.866 us | 11.206 us | 1.148x |
| `(17, 18432)` | fp16 | 27.284 us | 25.828 us | 1.056x |
| `(1000, 18432)` | fp16 | 518.754 us | 401.485 us | 1.292x |

## Step 7: Run Correctness Tests

After the current-branch build is installed, run the focused correctness tests:

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sglang-mingfei
python test/registered/cpu/test_activation.py
```

Example result:

```text
Ran 1 test in 3.423s

OK
```

If the test file is pytest-based, the output may look like:

```text
collected 48 items
48 passed, 2 warnings
```

## PR Description Template

Use this template in the PR description:

```markdown
## Summary

This PR speeds up the CPU `<kernel_name>` vector path by replacing `<old_op>` with `<new_op>`.

## Change

```cpp
<old code>
```

to:

```cpp
<new code>
```

## Benchmark

Benchmark script:

```bash
cd /home/mingfeima/sgl-cpu-tests
./run_bench_cpu.sh <bench_file>.py
```

Before: `main`  
After: this branch

| Shape | Dtype | Before | After | Speedup |
|---|---:|---:|---:|---:|
| ... | ... | ... | ... | ... |

## Correctness

```bash
source ~/.bashrc && sa
cd /home/mingfeima/sglang-mingfei/sgl-kernel
./install.sh

cd /home/mingfeima/sglang-mingfei
python <test_file>.py
```

Result:

```text
...
```
```

## Notes and Pitfalls

- Always rebuild and reinstall `sgl-kernel` before each before/after run. Python will otherwise keep using the previously installed wheel.
- Keep before and after on the same machine, same CPU binding, same environment, and same `run_bench_cpu.sh` runner.
- Prefer a separate `main` worktree over stashing or checking out over a dirty branch.
- Save raw logs under `/tmp` so the result table can be audited.
- For tiny decode shapes, use many iterations; for large prefill shapes, use fewer iterations.
- Include a correctness check in the benchmark itself when practical, but still run the registered test suite afterward.
- Check `git status --short --branch` before and after the benchmark so untracked helper files or worktrees do not surprise you.
