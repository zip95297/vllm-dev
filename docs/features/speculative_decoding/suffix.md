# Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding ([technical report](https://arxiv.org/abs/2411.04975)).

Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration to get better acceptance rates.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

!!! tip "Install Arctic Inference"
    Suffix Decoding requires [Arctic Inference](https://github.com/snowflakedb/ArcticInference). You can install it with `pip install arctic-inference`.

!!! tip "Suffix Decoding Speculative Tokens"
    Suffix Decoding will speculate a dynamic number of tokens for each request at each decoding step, so the `num_speculative_tokens` configuration specifies the *maximum* number of speculative tokens. It is suggested to use a high number such as `16` or `32` (default).

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## SuffixGPU phase-2 development branches

The GPU-native Suffix proposer performance work is split across two paired
development branches. Check out both branches when reproducing or extending
the implementation:

- vLLM adapter: [`zip95297/vllm-dev`, branch
  `perf/phase2-low-concurrency`](https://github.com/zip95297/vllm-dev/tree/perf/phase2-low-concurrency)
- SuffixGPU kernels: [`zip95297/SuffixGPU`, branch
  `perf/phase2-low-concurrency`](https://github.com/zip95297/SuffixGPU/tree/perf/phase2-low-concurrency)

The vLLM branch provides the framework-side policy and integration:

- the captured Suffix draft graph contains proposal work only; one fused
  SuffixGPU kernel performs dynamic token-state update and graph-input staging
  before replay;
- graph keys are `(batch_bucket, scan_bucket)`, so short active histories do
  not scan a full `max_model_len` buffer;
- the scan upper bound is computed from existing CPU sequence-length metadata
  plus the complete sampled-token width, avoiding a GPU-to-host read or CUDA
  synchronization;
- all Suffix bucket graphs use one independent graph memory pool and are
  captured largest-workspace first; they do not share storage with the model
  graph pool;
- the adapter preserves the existing eager fallback if graph capture fails.

The paired SuffixGPU branch implements fused graph staging, match-back/support
counting, support-based candidate selection, and direct occurrence-
continuation gathering. It also retains the legacy path above batch size 64,
where profiling found it faster, and includes exact fused-versus-legacy output
tests.

Qwen3-8B Spec-Bench validation used `k=5` across TP1, TP2, and TP4 at
concurrency 1, 4, 8, 32, 64, 128, and 256. C1 throughput remained effectively
neutral; TP1 c64/c128 improved by about 4.1%/3.5%; and the optimized TP4 c256
configuration completed three consecutive runs where the private-pool
baseline ran out of memory. Acceptance-rate changes were mixed run-to-run and
showed no directional regression. The fused local matcher reduced its
isolated GPU operation count from 70--90 to 36 and improved its Nsight
projected time by about 1.8--2.0x.
