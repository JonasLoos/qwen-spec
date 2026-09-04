"""Swap eligible 4-bit QuantizedLinear layers of an mlx-lm model to the flat small-M NAX kernel (qmm_small_m.py).

The kernels use the M5 tensor units (MLX's steel/gemm/nax.h); `kernel_available()` probes them once
and `patch_quantized_linears` leaves the model untouched (stock MLX kernels) where they do not run."""
import sys
import mlx.core as mx
import mlx.nn as nn
from .qmm_small_m import qmm_small_m, tile_weights, untile_weights

CHUNK = 64
_available = None


def kernel_available():
    """True if the tensor-unit kernel compiles and matches the stock kernel on this GPU (probed once per process)."""
    global _available
    if _available is None:
        try:
            w = mx.random.normal((64, 512)).astype(mx.bfloat16); q = mx.quantize(w, group_size=64, bits=4)
            x = mx.random.normal((3, 512)).astype(mx.bfloat16)
            y, ref = qmm_small_m(x, *tile_weights(*q)), mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4)
            _available = bool(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max() < 0.05 * mx.abs(ref).max() + 0.05)
        except Exception:
            _available = False
        if not _available:
            print("[qwen-spec] the tensor-unit matmul kernel does not run on this GPU (needs an M5-class GPU): using the stock MLX kernels", file=sys.stderr)
    return _available


# --------------------------------------------------------------------------- projection fusion
# in_proj_b / in_proj_a stay on the exact stock kernel: the DeltaNet recurrence amplifies bf16-dequant noise in its gates.
# Fused q/k/v change k/v prefill results by 1 bf16 ulp (MLX's GEMM tiles a 14336-wide matrix differently than 1024-wide ones).
FUSE_GROUPS = [("in_proj_qkv", "in_proj_z"), ("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]


class _FusedSlice:
    """Stands in for one member of a fused projection group: runs the fused matmul once per distinct
    input array (memoized on identity in `group`, shared by the members, so sibling calls with the same
    input reuse it) and returns its column slice. The memo is dropped once every member has consumed it,
    so prefill-sized outputs are not retained across layers."""

    def __init__(self, fused, group, idx, n_members, lo, hi):
        self.fused, self.group, self.idx, self.n, self.lo, self.hi = fused, group, idx, n_members, lo, hi

    def __call__(self, x):
        memo = self.group.memo
        if memo is None or memo[0] is not x or self.idx in memo[2]:
            self.group.memo = memo = (x, self.fused(x), set())
        memo[2].add(self.idx)
        y = memo[1][..., self.lo:self.hi]
        if len(memo[2]) == self.n:
            self.group.memo = None
        # prefill-sized inputs: hand consumers a contiguous array
        return mx.contiguous(y) if x.size // x.shape[-1] > CHUNK else y


class _Group:
    memo = None


def fuse_projections(model):
    """Concatenate sibling QuantizedLinear projections (same input) into one QuantizedLinear per group:
    one kernel instead of 2-4 per layer, and the tiny b/a projections (N=48) ride along in the big
    matmul instead of running on the stock kernel. Call before patch_quantized_linears."""
    n_groups = 0
    for name, mod in list(model.named_modules()):
        for names in FUSE_GROUPS:
            mems = [getattr(mod, n, None) for n in names]
            if not all(isinstance(m, nn.QuantizedLinear) for m in mems):
                continue
            if any("bias" in m for m in mems) or len({(m["weight"].shape[1], m.bits, m.group_size) for m in mems}) != 1:
                continue
            K8, bits, gs = mems[0]["weight"].shape[1], mems[0].bits, mems[0].group_size
            w = mx.concatenate([m["weight"] for m in mems], axis=0)
            sc = mx.concatenate([m["scales"] for m in mems], axis=0)
            bi = mx.concatenate([m["biases"] for m in mems], axis=0)
            fused = nn.QuantizedLinear(K8 * 32 // bits, w.shape[0], bias=False, group_size=gs, bits=bits)
            fused.weight, fused.scales, fused.biases = w, sc, bi
            mx.eval(w, sc, bi)
            lo, group = 0, _Group()
            for i, (n, m) in enumerate(zip(names, mems)):
                hi = lo + m["weight"].shape[0]
                setattr(mod, n, _FusedSlice(fused, group, i, len(mems), lo, hi))
                lo = hi
            setattr(mod, "fused_" + names[0], fused)
            n_groups += 1
    mx.clear_cache()
    return n_groups


def _tiled_call(self, x):
    t = getattr(self, "_tiled", None)
    if t is None:
        return _orig_call(self, x)
    wt, sct, bit = t
    *lead, K = x.shape
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if x2.dtype != mx.bfloat16:
        x2 = x2.astype(mx.bfloat16)
    if M <= CHUNK:
        y = qmm_small_m(x2, wt, sct, bit)
    else:   # prefill-sized inputs: stock kernel on un-tiled weights (one-off copy, cheap vs. prefill)
        w, sc, bi = untile_weights(wt, sct, bit)
        y = mx.quantized_matmul(x2, w, sc, bi, transpose=True, group_size=self.group_size, bits=self.bits)
    if "bias" in self:
        y = y + self["bias"]
    return y.reshape(*lead, -1)


_orig_call = nn.QuantizedLinear.__call__


def patch_quantized_linears(model, fuse=True):
    """Move every eligible QuantizedLinear (4-bit, group 64, K % 512 == 0, N % 32 == 0) of `model` to the tiled kernels,
    optionally fusing sibling projections first. Leaves the model untouched when the kernel is unavailable."""
    if not kernel_available():
        return
    nn.QuantizedLinear.__call__ = _tiled_call
    if fuse:
        fuse_projections(model)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.QuantizedLinear):
            continue
        N, K8 = mod["weight"].shape
        K = K8 * 32 // mod.bits
        if mod.bits == 4 and mod.group_size == 64 and K % 512 == 0 and N % 32 == 0:
            wt, sct, bit = tile_weights(mod["weight"], mod["scales"], mod["biases"])
            mx.eval(wt, sct, bit)
            mod._tiled = (wt, sct, bit)
            del mod["weight"], mod["scales"], mod["biases"]   # free the untiled copies
    mx.clear_cache()
