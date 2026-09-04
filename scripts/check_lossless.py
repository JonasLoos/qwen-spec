"""Losslessness check: engine output (greedy) against plain sequential greedy decoding of the same patched model.

Prints the first divergence per prompt with the reference logit margin (top-1 minus top-2) at that position: a tiny
margin is a bf16 tie resolved differently (noise), a large margin is a bug.

  uv run scripts/check_lossless.py [--model ID|DIR] [--drafter-model ID|DIR] [-n 150]
"""
import argparse

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from qwen_spec.engine import add_engine_args, engine_from_args
from qwen_spec.spec_decode import generate, make_sampler

PROMPTS = {
    "chat": "Explain in two short paragraphs why the sky is blue.",
    "code": "Write a Python class implementing an LRU cache with get and put methods, with docstrings.",
    "math": "Solve step by step: a train travels 120 km in 1.5 hours, then 200 km in 2.5 hours. What is its average speed over the whole trip?",
}


def reference(model, prompt, n, eos):
    """Sequential greedy decode with the standard forward; returns tokens and per-step (top1 - top2) margins."""
    c = cache_mod.make_prompt_cache(model)
    logits = model(mx.array(prompt, mx.uint32)[None], cache=c)[0, -1]
    out, margins = [], []
    for _ in range(n):
        lf = logits.astype(mx.float32)
        top2 = mx.sort(lf)[-2:]
        t = int(lf.argmax().item()); margins.append(float((top2[1] - top2[0]).item()))
        out.append(t)
        if t in eos:
            break
        logits = model(mx.array([[t]], mx.uint32), cache=c)[0, -1]
    return out, margins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--max-tokens", type=int, default=150)
    add_engine_args(ap)
    a = ap.parse_args()
    eng = engine_from_args(a)
    model, tok = eng.model, eng.tok
    eos = set(tok.eos_token_ids)
    for name, text in PROMPTS.items():
        p = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        ref, margins = reference(model, p, a.max_tokens, eos)
        out, st = generate(model, tok, p, eng.drafter, max_tokens=a.max_tokens, sampler=make_sampler())
        n = min(len(out), len(ref))
        div = next((i for i in range(n) if out[i] != ref[i]), None)
        if div is None:
            print(f"{name:5s}: identical to sequential decoding for {n} tokens ({st['tokens_per_step']:.2f} tokens/step)", flush=True)
        else:
            print(f"{name:5s}: diverges at token {div}/{n}; reference margin there {margins[div]:.4f}; "
                  f"reference {tok.decode(ref[div:div+1])!r} vs engine {tok.decode(out[div:div+1])!r}; context ...{tok.decode(ref[max(0, div-8):div])!r}", flush=True)
    mx.synchronize()


if __name__ == "__main__":
    main()
