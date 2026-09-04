"""Throughput benchmark (the README table): tok/s and accepted tokens per step for three prompts, under greedy decoding, the
sampled defaults without and with thinking, and plain sequential decoding (mlx-lm generate) as the baseline. Answers are
150 tokens (300 with thinking, mostly reasoning); the sampled rows use seed 0 and vary by about 10% between runs. Run on
AC power with nothing else on the GPU.

  uv run scripts/bench.py [--model ID|DIR] [--drafter dflash|mtp|ngram] [--drafter-model ID|DIR] [-n 150] [--no-baseline]
"""
import argparse, gc

import mlx.core as mx

from qwen_spec.engine import add_engine_args, engine_from_args, resolve

PROMPTS = {
    "chat": "Explain in two short paragraphs why the sky is blue.",
    "code": "Write a Python class implementing an LRU cache with get and put methods, with docstrings.",
    "math": "Solve step by step: a train travels 120 km in 1.5 hours, then 200 km in 2.5 hours. What is its average speed over the whole trip?",
}
MODES = [
    ("greedy", dict(temp=0.0, thinking=False)),
    ("sampled (T=0.7, no thinking)", dict(thinking=False)),
    ("thinking (T=1.0), lossless", dict(thinking=True, accept="lossless")),
    ("thinking (T=1.0), default rule", dict(thinking=True)),
]


def baseline(model_id, n):
    """Plain sequential greedy decoding of the unpatched model with mlx-lm, tok/s per prompt."""
    from mlx_lm import load, stream_generate
    model, tok = load(resolve(model_id))
    out = {}
    for name, text in PROMPTS.items():
        p = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        for r in stream_generate(model, tok, p, max_tokens=n):
            pass
        out[name] = r.generation_tps
        print(f"  baseline {name}: {r.generation_tps:.1f} tok/s", flush=True)
    del model, tok
    gc.collect(); mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--max-tokens", type=int, default=150)
    ap.add_argument("--thinking-tokens", type=int, default=300)
    ap.add_argument("--no-baseline", action="store_true")
    add_engine_args(ap)
    a = ap.parse_args()
    rows = []
    if not a.no_baseline:
        rows.append(("plain decoding (mlx-lm)", {k: f"{v:.1f}" for k, v in baseline(a.model, a.max_tokens).items()}))
    eng = engine_from_args(a)
    eng.generate([{"role": "user", "content": PROMPTS["chat"]}], max_tokens=16, thinking=False)     # warm-up
    for label, kw in MODES:
        row = {}
        for name, text in PROMPTS.items():
            n = a.thinking_tokens if kw["thinking"] else a.max_tokens
            eng.session.reset()                                                    # no prefix reuse: every run prefills its prompt
            _, _, st = eng.generate([{"role": "user", "content": text}], max_tokens=n, seed=0, **kw)
            row[name] = f"{st['tok_s']:.1f} ({st['tokens_per_step']:.1f})"
            print(f"  {label}, {name}: {st['tok_s']:.1f} tok/s, {st['tokens_per_step']:.2f} tok/step, {st['tokens']} tokens" + (f", {st['forced']} forced" if st["forced"] else ""), flush=True)
        rows.append((label, row))
    print(f"\n{eng.name}, {a.drafter} drafter, {mx.device_info()['device_name']}: tok/s (accepted tokens per step)\n")
    print("| | " + " | ".join(PROMPTS) + " |\n|---|" + "---|" * len(PROMPTS))
    for label, row in rows:
        print(f"| {label} | " + " | ".join(row[k] for k in PROMPTS) + " |")
    mx.synchronize()


if __name__ == "__main__":
    main()
