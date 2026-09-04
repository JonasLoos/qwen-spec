"""Small-M 4-bit (affine, group 64) quantized matmul for speculative verification on Apple GPUs.

out[M, N] = x[M, K] @ dequant(w)[N, K]^T

Design: one threadgroup = 32 output columns x all M rows. 4 simdgroups split K (split-K); each
simdgroup dequantizes a 32k x 32n weight tile into threadgroup memory once per step and reuses it
for every row block via simdgroup_matrix MMA. Partial sums are reduced across simdgroups in 3
rounds through threadgroup memory. Weights are read exactly once -> cost ~flat in M.
Requires K % 512 == 0, N % 32 == 0, M <= 128 (padded to a multiple of 16).
"""
import os, re
import mlx.core as mx

_INC = os.path.join(os.path.dirname(mx.__file__), "include")


def _inline(path, seen):
    """Inline MLX kernel headers recursively (the JIT has no include path for them)."""
    if path in seen:
        return ""
    seen.add(path)
    out = []
    for line in open(os.path.join(_INC, path)):
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "".join(out)


_HEADER = _inline("mlx/backend/metal/kernels/steel/gemm/nax.h", set()) + """
using namespace metal;
using namespace mlx::steel;
"""

_SRC = r"""
    constexpr int KPS = K / 4;                 // K per simdgroup (split-K over 4 simdgroups)
    constexpr int TILE = MP * 32;              // floats per partial-sum tile

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid >> 5, lane = tid & 31;
    const int n0 = (int)threadgroup_position_in_grid.x * 32;
    const int n = n0 + (int)lane;

    threadgroup bfloat16_t stage[4 * 1024];    // 4 x (32k x 32n) bf16 = 8 KB
    threadgroup bfloat16_t* bt = stage + sg * 1024;
    NAXTile<float, MP / 16, 2> Dtile;
    Dtile.clear();

    const int tile = (int)threadgroup_position_in_grid.x;
    const device uint4* wt = (const device uint4*)w + (size_t)tile * (K / 32) * 32 + lane;
    const device bfloat16_t* st = sc + (size_t)tile * (K / 64) * 32 + lane;
    const device bfloat16_t* bt_ = bi + (size_t)tile * (K / 64) * 32 + lane;
    const int kbeg = (int)sg * KPS;

    // software pipeline: the next step's weights + scales are loaded before this step's MMA
    uint4 pv = wt[(size_t)(kbeg >> 5) * 32];
    float s = (float)st[(kbeg >> 6) * 32];
    float bb = (float)bt_[(kbeg >> 6) * 32];
    for (int kk = 0; kk < KPS; kk += 32) {
        const int ka = kbeg + kk;
        uint p[4] = {pv.x, pv.y, pv.z, pv.w};
        const float s_cur = s, bb_cur = bb;
        if (kk + 32 < KPS) {
            const int kn = ka + 32;
            pv = wt[(size_t)(kn >> 5) * 32];
            s = (float)st[(kn >> 6) * 32];
            bb = (float)bt_[(kn >> 6) * 32];
        }
        for (int u = 0; u < 4; ++u)
            for (int t = 0; t < 8; ++t)
                bt[(u * 8 + t) * 32 + lane] = (bfloat16_t)((float)((p[u] >> (4 * t)) & 15u) * s_cur + bb_cur);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            NAXTile<bfloat16_t, MP / 16, 2> Atile;
            NAXTile<bfloat16_t, 2, 2> Btile;
            Atile.load(x + ka, K);
            Btile.template load<bfloat16_t, 32, 1>(bt);
            tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<false>{});
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce the 4 split-K partials: each simdgroup hands its raw fragments to simdgroup 0.
    constexpr int NE = MP;                      // floats per lane in Dtile (MP/16*2 frags * 8)
    threadgroup float* red = (threadgroup float*)stage;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    thread float* de = Dtile.elems();
    for (int src = 1; src < 4; ++src) {
        if ((int)sg == src)
            for (int e = 0; e < NE; ++e) red[e * 32 + lane] = de[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0)
            for (int e = 0; e < NE; ++e) de[e] += red[e * 32 + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0) Dtile.store(out + n0, N);
"""

_kernel = mx.fast.metal_kernel(
    name="qmm_small_m4_s4",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    header=_HEADER,
    source=_SRC,
)


def tile_weights(w, sc, bi):
    """Repack MLX 4-bit gs64 weights into the tiled layout the kernel reads coalesced.
    w [N, K/8] uint32 -> [N/32, K/32, 32, 4]; sc/bi [N, K/64] -> [N/32, K/64, 32]."""
    N, K8 = w.shape
    K = K8 * 8
    wt = w.reshape(N // 32, 32, K // 32, 4).transpose(0, 2, 1, 3)
    sct = sc.reshape(N // 32, 32, K // 64).transpose(0, 2, 1)
    bit = bi.reshape(N // 32, 32, K // 64).transpose(0, 2, 1)
    return mx.contiguous(wt), mx.contiguous(sct), mx.contiguous(bit)


def untile_weights(wt, sct, bit):
    """Inverse of tile_weights (returns standard MLX [N, K/8], [N, K/64] layouts)."""
    NT, K32, _, _ = wt.shape
    w = wt.transpose(0, 2, 1, 3).reshape(NT * 32, K32 * 4)
    sc = sct.transpose(0, 2, 1).reshape(NT * 32, -1)
    bi = bit.transpose(0, 2, 1).reshape(NT * 32, -1)
    return w, sc, bi


def qmm_small_m(x, wt, sct, bit, group_size=64, bits=4):
    """x: [..., M, K] bf16; (wt, sct, bit) from tile_weights. Returns [..., M, N] bf16."""
    assert bits == 4 and group_size == 64
    *lead, M, K = x.shape
    N = wt.shape[0] * 32
    assert K % 512 == 0 and N % 32 == 0 and M <= 128, (M, K, N)
    MP = ((M + 15) // 16) * 16
    x2 = x.reshape(-1, K)
    if x2.shape[0] != MP:
        x2 = mx.concatenate([x2, mx.zeros((MP - x2.shape[0], K), dtype=x2.dtype)], axis=0)
    if x2.dtype != mx.bfloat16:
        x2 = x2.astype(mx.bfloat16)
    (out,) = _kernel(
        inputs=[x2, wt, sct, bit],
        template=[("MP", MP), ("K", K), ("N", N)],
        grid=(N // 32 * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(MP, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out[:M].reshape(*lead, M, N)
