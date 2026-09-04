"""Re-record the depth calibration the DFlash2 drafter ships with (qwen_spec/data/calib.json): the observed acceptance
probability of the drafter's tree values, per value bin and depth. Run it for another drafter or another model of the family.
(The verification cost curve that sets the tree budget is measured by the engine itself on its first start, see cost_curve.py.)

  uv run scripts/calibrate.py [--model ID|DIR] [--drafter-model ID|DIR] [--out DIR] [--max-tokens 200]
"""
import argparse, json, os

import mlx.core as mx
import numpy as np

from qwen_spec.engine import CACHE, DATA, MODELS, Engine, cache_key, resolve
from qwen_spec.spec_decode import generate, make_sampler

PROMPTS = [
    "Explain in two short paragraphs why the sky is blue.",
    "Write a Python class implementing an LRU cache with get and put methods, with docstrings.",
    "Solve step by step: a train travels 120 km in 1.5 hours, then 200 km in 2.5 hours. What is its average speed over the whole trip?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Reason step by step and put the final answer in \\boxed{}.",
    "A bakery makes 12 dozen cookies a day and sells them in boxes of 8. Boxes cost $6 each. How much money does the bakery make per day if everything sells? Reason step by step.",
    "Complete the following Python function. Reply with the complete function in a single ```python code block, no tests.\n\n```python\ndef longest_common_prefix(strs: list[str]) -> str:\n    \"\"\"Return the longest common prefix of all strings in strs (empty string if none).\"\"\"\n```",
    "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between TCP and UDP? Answer with a short table and one paragraph.",
]
EDGES = np.array([-4, -2.5, -2.0, -1.6, -1.3, -1.0, -0.8, -0.6, -0.45, -0.3, -0.2, -0.1, -0.03, 0.0001])
GROUPS = {"d1": [1], "d2": [2], "d3-4": [3, 4], "d5-8": [5, 6, 7, 8], "d9+": list(range(9, 40))}


def record_nodes(model, tok, drafter, max_tokens):
    """(drafter value, depth, accepted) for every drafted node over the prompts, greedy and sampled (thinking)."""
    vals, deps, accs = [], [], []
    for mode, sampler, thinking in (("greedy", make_sampler(), False), ("sampled", make_sampler(1.0, 20, 0.95), True)):
        for text in PROMPTS:
            p = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=thinking)
            log = []
            mx.random.seed(0)
            out, st = generate(model, tok, p, drafter, max_tokens=max_tokens, sampler=sampler, node_log=log)
            vals += [x[0] for x in log]; deps += [x[1] for x in log]; accs += [x[2] for x in log]
            print(f"  {mode:7s} {text[:40]!r:44s} {st['tokens_per_step']:.2f} tok/step, {len(log)} nodes", flush=True)
    return np.array(vals, np.float64), np.array(deps, int), np.array(accs, float)


def fit(v, dep, acc):
    """Piecewise-linear map log10(value) -> observed acceptance per depth group, monotone, as calib.json."""
    lv = np.log10(np.clip(v, 1e-4, 1.0))
    print(f"  {len(v)} nodes; mean value {v.mean():.3f} vs acceptance {acc.mean():.3f}")
    calib = {}
    for g, depths in GROUPS.items():
        gs = np.isin(dep, depths)
        xs, ys = [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = gs & (lv >= lo) & (lv < hi)
            if s.sum() < 15:
                continue
            xs.append(float(np.log10(v[s].mean()))); ys.append(float(acc[s].mean()))
        if xs:
            ys = np.maximum.accumulate(np.clip(ys, 1e-4, 1.0))
            calib[g] = {"depths": depths, "log10_v": xs, "p": [float(y) for y in ys]}
    return calib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DATA, help="output directory (default: the package data directory, i.e. replace the shipped files)")
    ap.add_argument("--max-tokens", type=int, default=200, help="tokens per prompt for the calibration recording")
    ap.add_argument("--model", default=MODELS["target"], metavar="ID|DIR", help="target model (default: %(default)s)")
    ap.add_argument("--drafter-model", metavar="ID|DIR", help=f"DFlash2 drafter weights (default: {MODELS['dflash']})")
    a = ap.parse_args()
    eng = Engine(a.model, drafter="ngram")                 # loads only the target; the recording drafter is built below
    model, tok = eng.model, eng.tok
    os.makedirs(a.out, exist_ok=True)
    print("calibration recording (generous tree budget so that low-value nodes are observed too):")
    from qwen_spec.dflash_drafter import DFlashDrafter
    name = a.drafter_model or MODELS["dflash"]
    flat = {1: 1.0, 128: 1.0}                              # flat cost curve: trimming keeps every node up to max_nodes
    drafter = DFlashDrafter(model, resolve(name), flat, quant_bits=4, draft_vocab=np.load(os.path.join(DATA, "draft_vocab.npy")), branch=4,
                            max_nodes=32, block_size=16, min_value=0.003, frontier=8, cache_dir=os.path.join(CACHE, cache_key(name)))
    v, dep, acc = record_nodes(model, tok, drafter, a.max_tokens)
    json.dump(fit(v, dep, acc), open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    print(f"wrote calib.json to {a.out}")
    mx.synchronize()


if __name__ == "__main__":
    main()
