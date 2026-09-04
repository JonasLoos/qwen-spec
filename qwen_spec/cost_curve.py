"""Verification cost curve, measured on this machine at the first start and cached.

The drafter picks its per-step node budget from c(n), the cost of verifying n tree nodes relative to one, so the curve must
match the machine: on a base M5 with the tensor-unit kernels it is flat to 16 nodes and +13-17% at 20-32 (Low Power Mode:
+33%); another compute/bandwidth ratio (M5 Pro/Max) or the stock MLX kernels (16 tokens cost 2.7x one) give another shape.
`load_cost_curve` times the real tree-verification pass at 1-32 nodes (about 4 s, once) and caches the result in the cache
directory, keyed by GPU architecture, kernel mode, MLX version, Low Power Mode and model name. Battery power is not in the
key: it scales the whole curve (~5%) and the budget rule depends on the shape only. The drafter's own cost stays the fixed
ratio draft_cost=0.08 of a one-node pass (both passes are bandwidth-bound, so the ratio transfers).

Env: QWEN_SPEC_CURVE=<json with "cost_curve_ms"> uses that file instead; QWEN_SPEC_RECALIBRATE=1 measures again (= --recalibrate)."""
import json, os, subprocess, sys, time
import numpy as np
import mlx.core as mx

SIZES = (1, 2, 4, 8, 12, 16, 24, 32)


def low_power_mode():
    """macOS Low Power Mode (pmset); False if unknown."""
    try:
        cfg = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return any(line.split()[:2] == ["lowpowermode", "1"] for line in cfg.splitlines())


def curve_key(model_name):
    from . import patch_model, tree_attn
    mode = ("nax" if patch_model.kernel_available() else "stock") + ("" if tree_attn.available() else "-sdpa")
    lpm = "_lpm" if low_power_mode() else ""
    return f"cost_curve_{mx.device_info()['architecture']}_{mode}_mlx{mx.__version__}{lpm}_{model_name}"


def calib_tree(n, rng):
    """n-node tree shaped like the drafter's: a chain of up to 16 nodes; further nodes hang off the chain (depth <= 16)."""
    from .tree_verify import Tree
    parent = [-1] + [i - 1 if i <= 16 else i - 16 for i in range(1, n)]
    return Tree([int(t) for t in rng.integers(1000, 150000, n)], parent)


def measure(model, sizes=SIZES, reps=3, prefix_len=64, seed=0, retry=True):
    """Wall time (ms) of the tree-verification pass per node count (best of `reps`, round-robin), after a short random prefix."""
    from mlx_lm.models import cache as cache_mod
    from .tree_verify import TreeVerifier
    rng = np.random.default_rng(seed)
    tv = TreeVerifier(model, cache_mod.make_prompt_cache(model))
    mx.eval(tv.prefill([int(t) for t in rng.integers(1000, 150000, prefix_len)]))

    def timed(n):
        mx.synchronize(); t0 = time.perf_counter()
        logits, h = tv.forward(calib_tree(n, rng))
        mx.eval(logits, h)
        return time.perf_counter() - t0

    last = timed(max(sizes))                              # warm up until the pass time is stable: kernel compilation, cache growth,
    for _ in range(8):                                    # and the GPU clock ramping up after idle (several passes in Low Power Mode)
        t = timed(max(sizes))
        if abs(t - last) < 0.03 * last:
            break
        last = t
    times = {n: [] for n in sizes}
    for _ in range(reps):                                 # round-robin over the sizes, so clock drift hits all of them alike
        for n in sizes:
            times[n].append(timed(n))
    sizes = sorted(sizes)
    best = [1000 * min(times[n]) for n in sizes]          # min: GPU timings only have slow outliers
    if retry and abs(1000 * timed(sizes[0]) - best[0]) > 0.1 * best[0]:   # the clock moved during the run: measure once more
        return measure(model, sizes, reps, prefix_len, seed, retry=False)
    return dict(zip(sizes, _isotonic(best)))              # the cost cannot fall with more rows: pool the residual noise


def _isotonic(ys):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = []
    for y in ys:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, c1), (m2, c2) = blocks[-2], blocks.pop()
            blocks[-1] = [(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2]
    return [m for m, c in blocks for _ in range(c)]


def _normalize(cc):
    cc = {int(k): v for k, v in cc.items()}
    return {k: v / cc[1] for k, v in cc.items()}


def load_cost_curve(model, model_name, cache_dir, quiet=False, force=False):
    """{n: verify cost relative to n=1} for this machine, measured on the first start (~4 s) and cached in cache_dir."""
    if os.environ.get("QWEN_SPEC_CURVE"):
        return _normalize(json.load(open(os.environ["QWEN_SPEC_CURVE"]))["cost_curve_ms"])
    path = os.path.join(cache_dir, curve_key(model_name) + ".json")
    if not (force or os.environ.get("QWEN_SPEC_RECALIBRATE")) and os.path.exists(path):
        return _normalize(json.load(open(path))["cost_curve_ms"])
    t0 = time.perf_counter()
    cc = measure(model)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(dict(device=mx.device_info()["device_name"], measured=time.strftime("%Y-%m-%d %H:%M"), low_power_mode=low_power_mode(),
                   cost_curve_ms={str(k): v for k, v in cc.items()}), open(path, "w"), indent=1)
    if not quiet:
        rel = " ".join(f"{k}:{v / cc[1]:.2f}" for k, v in cc.items())
        print(f"[measured the verify cost curve in {time.perf_counter() - t0:.1f}s: {cc[1]:.0f} ms per pass, relative {rel}; cached at {path}]", file=sys.stderr)
    return _normalize(cc)
