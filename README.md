# qwen-spec

Tree speculative decoding for the hybrid (Gated DeltaNet + attention) Qwen3.5 / 3.6 / 3.8 models on Apple
Silicon with MLX. The 4-bit Qwen3.8-27B goes from 8 tok/s to 23–52 tok/s on a base M5 MacBook Pro, greedy or
sampled, with the same output distribution as plain decoding.

```
uv tool install git+https://github.com/JonasLoos/qwen-spec      # or: uv sync inside a clone
qwen-spec                                                       # interactive chat
qwen-spec "Why is the sky blue?"                                # one answer (thinking dimmed, then the reply)
qwen-spec --no-think -n 300 "..."                               # no thinking, short answer
qwen-spec --effort low "..."                                    # the chat template's reasoning-effort instruction
qwen-spec-server                                                # OpenAI-compatible API on http://127.0.0.1:8080
```

The first run downloads the target (15 GB) and the drafter (3.8 GB) from the Hugging Face hub; an existing LM Studio
download of the same repo is used instead. Defaults follow the model's own recommendations: thinking on, temperature
1.0 / top-p 0.95 / top-k 20 (0.7 / 0.8 / 20 without thinking), up to 32768 output tokens. `--help` lists everything,
including the chat commands (`/think off`, `/effort low`, `/temp`, `/system`, `/file`, `/reset`). The server takes the
same flags as request defaults, returns thinking as `reasoning_content`, and reports reused prompt tokens in `usage`.

## Requirements

- An M5-class Apple GPU on macOS 26. The matmul and tree-attention kernels use the M5 tensor units (Metal 4 tensor
  ops), which is what makes checking 16 draft tokens cost the same as 1. On other Apple GPUs or older macOS the package
  falls back to the stock MLX kernels (untested; expect a much smaller speedup).
- 32 GB of memory for the 27B model in 4 bit plus the drafter.
- Python 3.13, MLX 0.32.1 and mlx-lm 0.31.3 (pinned: the tree verifier relies on mlx-lm's `qwen3_5` internals).

## Results

`scripts/bench.py` on a base M5 (10 GPU cores, 153 GB/s, 32 GB), Qwen3.8-27B 4-bit, DFlash2 drafter, 16-node trees;
answers of 150 tokens (300 with thinking, mostly reasoning); tok/s (accepted tokens per step), 2026-09-04:

| | chat | code | math |
|---|---|---|---|
| plain decoding (mlx-lm) | 8.0 | 8.1 | 8.2 |
| greedy | 23.0 (3.5) | 35.1 (5.4) | 51.7 (7.9) |
| sampled, T=0.7 / top-p 0.8 / top-k 20 (no thinking) | 28.9 (4.4) | 42.8 (6.5) | 49.0 (7.5) |
| thinking, T=1.0 / top-p 0.95 / top-k 20, exact sampling | 24.4 (3.7) | 23.1 (3.6) | 54.5 (8.5) |
| thinking, same, default acceptance rule | 26.5 (4.1) | 27.1 (4.2) | 52.9 (8.3) |

Sampled rows are single runs (seed 0) and vary by about 10% between runs; over more prompts the default rule gives +24%
tokens per step in thinking mode (below). Long prompts cost little per step: the chat prompt decodes at 152 ms per step
with 24 tokens of context, 156 ms at 3k and 165 ms at 8k (prefill runs at about 210 tok/s).

The default acceptance rule for sampled decoding (`ratio:theta=0.3`) accepts a drafted branch when the target's random
draw misses it but the probability mass moved per gained token is at most 0.3; that forces about 4% of the tokens, gives
+24% tokens per step in thinking mode, and left GSM8K accuracy unchanged. `--accept lossless` restores exact sampling,
`--accept cov:eps=1,tau=1` forces only the target's own argmax, `--accept cov:eps=1` always takes the best drafted child
(fastest). Greedy decoding is exact under every rule. `scripts/bench.py` reproduces the table.

## How it works

1. A 4-bit matmul kernel that reads every weight tile once and multiplies all rows on the tensor units, so one
   forward pass over 16 tree nodes costs the same as over 1 token.
2. Tree verification on the hybrid model: the DeltaNet layers are evaluated in their parallel form (prefix state read
   once per layer, per-node path walk in a small kernel), the attention layers with an ancestor mask in a flash-decoding
   style tree-attention kernel that reads the KV cache in place (2x faster than the stock attention at long contexts);
   the accepted path is replayed into the recurrent state and the KV cache is compacted.
3. Draft trees from the DFlash2 block drafter's candidate lattice (parent-conditioned branches at no extra draft cost),
   with path values calibrated per depth and the node budget chosen from the measured cost curve.
4. Exact sampling: draw one token per node from the target, follow the tree while the draw matches a child (fused
   top-k / top-p / min-p sampling kernel), with optional relaxed acceptance rules on top.
5. The KV cache, DeltaNet states and drafter context persist across turns and requests; a prompt that extends an
   earlier one prefills only the new tokens.

The verification cost curve that sets the tree budget is measured by the engine itself on its first start on a machine
(the real verification pass at 1–32 nodes, about 4 s) and cached under `~/.cache/qwen-spec/` per GPU, kernel mode, MLX
version and Low Power Mode; `--recalibrate` repeats it. `scripts/check_lossless.py` compares the engine with sequential
decoding; `scripts/calibrate.py` re-records the depth calibration in `qwen_spec/data/` for another drafter or model.

## Notes

- Only 4-bit, group-64 affine weights use the custom kernel; other layers fall back to MLX.
- Benchmark on AC power with Low Power Mode off; sustained runs throttle the GPU by 5–10%, Low Power Mode halves the speed.
- Token ties in bf16 can resolve differently than in stock mlx-lm; that is floating-point noise, not a loss of exactness.

How it works, with interactive figures: [jonasloos.github.io/qwen-spec](https://jonasloos.github.io/qwen-spec/) (source in
`docs/`). License: MIT; see `THIRD_PARTY.md` for the vendored drafter code and the weights.
