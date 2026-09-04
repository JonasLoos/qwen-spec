"""Flash-decoding-style tree attention for the verification pass (Apple M5 tensor units via the Metal tensor ops
that MLX's steel NAX tiles wrap; `available()` probes them once, tree_verify.py falls back to SDPA otherwise).

out[n, h, :] = softmax_j(scale * q[h, n] . k[g(h), j]) v[g(h), j]  over the visible keys j:
  j < offset (cached prefix) is always visible, tree column j = offset + t is visible iff anc[n] bit t is set.

Kernel 1 (`tree_attn_partial`): grid over (row groups of 16 (head, node) rows, kv head, chunk of 32*NSUB keys).
A threadgroup holds RG row groups x NS = D/DS d-splits simdgroups; each simdgroup owns 16 query rows x DS output
columns (fp32 accumulators in registers) and streams the chunk in 32-key sub-tiles:
  S = Q K^T with one mpp::tensor_ops::matmul2d (16x32xDS, the op streams Q and K from device memory itself,
  which is ~2x faster than per-element fragment loads), the d-split partials are summed through threadgroup
  memory, mask, online softmax (exp2 domain, row statistics in fp32), O += P V with 16x32x16 NAX MMAs on
  vectorized 8-byte fragment loads of V.
With one chunk it writes the normalized bf16 result directly; with several chunks it writes fp16 normalized
partials + (m, l) per row and kernel 2 (`tree_attn_merge`) combines them into bf16 [N, H*D] (token-major, ready
for o_proj). K/V are read in place from the full cache buffers (no per-call slice copy). Chunk size: ~5 chunks
(`chunk_subtiles`), RG = 3 row groups per threadgroup (192 threads); both measured on the M5 base.
"""
import mlx.core as mx

from .qmm_small_m import _HEADER as _NAX_HEADER

_HEADER = _NAX_HEADER + r"""
struct MaxOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); } };
struct SumOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; } };
struct MulOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; } };
struct ExpSubOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); } };
// 16x16 bf16 NAX fragment of a row-major matrix: the lane holds rows (r, r+8) x 4 consecutive columns; p is already offset by (r, c)
METAL_FUNC void vload(thread metal::vec<bfloat16_t, 8>& dst, const device bfloat16_t* p, int ld) {
    const ushort4 a = *(const device ushort4*)p;
    const ushort4 b = *(const device ushort4*)(p + 8 * ld);
    dst = metal::vec<bfloat16_t, 8>(as_type<metal::vec<bfloat16_t, 4>>(a), as_type<metal::vec<bfloat16_t, 4>>(b));
}
METAL_FUNC void vload_rows(thread metal::vec<bfloat16_t, 8>& dst, const device bfloat16_t* p, int ld, int lim) {   // rows >= lim (relative to the lane's row) read as 0
    const ushort4 a = lim > 0 ? *(const device ushort4*)p : ushort4(0);
    const ushort4 b = lim > 8 ? *(const device ushort4*)(p + 8 * ld) : ushort4(0);
    dst = metal::vec<bfloat16_t, 8>(as_type<metal::vec<bfloat16_t, 4>>(a), as_type<metal::vec<bfloat16_t, 4>>(b));
}
"""

# template: D head dim, DS d-columns per simdgroup, RG row groups per threadgroup, NSUB 32-key sub-tiles per chunk,
# N tree nodes, GQA q heads per kv head.  inputs: q [H*N, D] bf16, k/v [HK, cap, D] bf16 (full cache buffers),
# prm int32 [S, offset], scl f32 [scale * log2 e], anc u32 [N].  outputs: po [HK, NC, R, D] f32, pm/pl [HK, NC, R] f32.
_PARTIAL_SRC = r"""
    constexpr int NS = D / DS;                 // d-splits (simdgroups per row group)
    constexpr int R = GQA * N;                 // query rows per kv head
    constexpr int C = 32 * NSUB;               // keys per chunk
    constexpr int TD = DS / 16;                // 16-wide d frags per simdgroup
    constexpr int NBUF = (RG * NS * 4096 <= 32768) ? 2 : 1;   // double-buffered exchange (one barrier per sub-tile) if it fits in 32 KB
    constexpr float NEG = -3.0e38f;
    using namespace mpp::tensor_ops;
    using tens_t = metal::tensor<device bfloat16_t, metal::dextents<int32_t, 2>, metal::tensor_inline>;
    constexpr auto qk_desc = matmul2d_descriptor(16, 32, DS, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, metal::execution_simdgroup> qk_op;

    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int rgl = sg / NS, ds = sg % NS;
    const int rg = (int)threadgroup_position_in_grid.x * RG + rgl;
    const int g = (int)threadgroup_position_in_grid.y;
    const int c = (int)threadgroup_position_in_grid.z;
    const int S = prm[0], offset = prm[1], base = prm[2];   // key j of this call = cache row base + j
    const int cap = (int)k_shape[1];
    const int NC = (S + C - 1) / C;
    const int r0 = rg * 16;
    const int nrows = clamp(R - r0, 0, 16);    // 0: padding row group (still joins the barriers)
    const float scale = scl[0];

    threadgroup float4 xbuf[NBUF * RG * NS * 128];   // S partial exchange: 16x32 f32 per simdgroup

    const short2 sc = BaseNAXFrag::get_coord();
    const int sm = sc.y, sn = sc.x;
    const device bfloat16_t* qp = q + (size_t)(g * R + r0) * D + ds * DS;
    const device bfloat16_t* kb = k + ((size_t)g * cap + base) * D + ds * DS;
    const device bfloat16_t* vb = v + ((size_t)g * cap + base) * D + ds * DS + sm * D + sn;   // fragment-lane offset applied
    tens_t tQ((device bfloat16_t*)qp, metal::dextents<int32_t, 2>(DS, nrows), metal::array<int32_t, 2>{1, D});
    auto cS = qk_op.template get_destination_cooperative_tensor<tens_t, tens_t, float>();

    uint abits[2];
    for (int ii = 0; ii < 2; ++ii) {
        const int row = r0 + sm + ii * 8;
        abits[ii] = row < R ? anc[row % N] : 0u;
    }

    NAXTile<float, 1, TD> Ot;
    Ot.clear();
    metal::vec<float, 2> mrow = {NEG, NEG}, lrow = {0.f, 0.f};

    for (int s = 0; s < NSUB; ++s) {
        const int key0 = c * C + s * 32;
        if (key0 >= S) break;
        const int nvalid = min(32, S - key0);
        // S = Q K^T over this simdgroup's d-split (the tensor op streams Q and K itself)
        tens_t tK((device bfloat16_t*)(kb + (size_t)key0 * D), metal::dextents<int32_t, 2>(DS, nvalid), metal::array<int32_t, 2>{1, D});
        qk_op.run(tQ, tK, cS);
        NAXTile<float, 1, 2> St;
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < 16; ++e) St.elems()[e] = cS[e];
        // sum the d-split partials of S across the NS simdgroups of this row group
        if (NS > 1) {
            threadgroup float4* mine = xbuf + ((s % NBUF) * RG * NS + sg) * 128 + lane * 2;
            const thread float4* sf = (const thread float4*)St.elems();
            mine[0] = sf[0]; mine[1] = sf[1]; mine[64] = sf[2]; mine[65] = sf[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            thread float4* sd = (thread float4*)St.elems();
            for (int o = 1; o < NS; ++o) {
                const threadgroup float4* other = xbuf + ((s % NBUF) * RG * NS + rgl * NS + ((ds + o) % NS)) * 128 + lane * 2;
                sd[0] += other[0]; sd[1] += other[1]; sd[2] += other[64]; sd[3] += other[65];
            }
            if (NBUF == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < 16; ++e) St.elems()[e] *= scale;
        if (key0 + 32 > offset || nvalid < 32) {      // tree columns / tail: apply the mask
            STEEL_PRAGMA_UNROLL
            for (int ik = 0; ik < 2; ++ik) {
                thread auto& fg = St.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (int ii = 0; ii < 2; ++ii) {
                    STEEL_PRAGMA_UNROLL
                    for (int jj = 0; jj < 4; ++jj) {
                        const int col = key0 + ik * 16 + sn + jj;
                        const bool vis = col < S && (col < offset || ((abits[ii] >> (col - offset)) & 1u));
                        fg[ii * 4 + jj] = vis ? fg[ii * 4 + jj] : NEG;
                    }
                }
            }
        }
        // online softmax (exp2 domain)
        metal::vec<float, 2> mnew = mrow;
        St.template row_reduce<MaxOp>(mnew);
        St.template row_bin_op<ExpSubOp>(mnew);
        metal::vec<float, 2> factor;
        for (int i = 0; i < 2; ++i) { factor[i] = fast::exp2(mrow[i] - mnew[i]); mrow[i] = mnew[i]; lrow[i] *= factor[i]; }
        St.template row_reduce<SumOp>(lrow);
        Ot.template row_bin_op<MulOp>(factor);
        // O += P V  (V frags: rows sm, sm+8 of each 16-key block, 4 consecutive d)
        const device bfloat16_t* vp = vb + (size_t)key0 * D;
        if (nvalid < 32) {
            STEEL_PRAGMA_UNROLL
            for (int id = 0; id < TD; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (int ik = 0; ik < 2; ++ik) {
                    metal::vec<bfloat16_t, 8> v0, v1;
                    vload_rows(v0, vp + ik * 16 * D + id * 16, D, nvalid - ik * 16 - sm);
                    vload_rows(v1, vp + ik * 16 * D + id * 16 + 16, D, nvalid - ik * 16 - sm);
                    BaseNAXFrag::mma(Ot.frag_at(0, id), Ot.frag_at(0, id + 1), St.frag_at(0, ik), metal::false_type{}, v0, v1, metal::false_type{});
                }
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (int id = 0; id < TD; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (int ik = 0; ik < 2; ++ik) {
                    metal::vec<bfloat16_t, 8> v0, v1;
                    vload(v0, vp + ik * 16 * D + id * 16, D);
                    vload(v1, vp + ik * 16 * D + id * 16 + 16, D);
                    BaseNAXFrag::mma(Ot.frag_at(0, id), Ot.frag_at(0, id + 1), St.frag_at(0, ik), metal::false_type{}, v0, v1, metal::false_type{});
                }
            }
        }
    }
    if (nrows <= 0) return;
    // normalized rows (the chunk-local softmax output); FINAL writes bf16 [N, H*D], else fp16 partials + (m, l)
    const metal::vec<float, 2> inv = {1.f / lrow[0], 1.f / lrow[1]};
    if (FINAL) {
        STEEL_PRAGMA_UNROLL
        for (int ii = 0; ii < 2; ++ii) {
            const int row = sm + ii * 8;
            if (row >= nrows) continue;
            const int r = r0 + row;
            device bfloat16_t* op = out + (size_t)(r % N) * (HK * GQA * D) + (size_t)(g * GQA + r / N) * D + ds * DS + sn;
            STEEL_PRAGMA_UNROLL
            for (int id = 0; id < TD; ++id) {
                thread auto& fg = Ot.frag_at(0, id);
                const metal::vec<bfloat16_t, 4> o4 = {(bfloat16_t)(fg[ii * 4] * inv[ii]), (bfloat16_t)(fg[ii * 4 + 1] * inv[ii]), (bfloat16_t)(fg[ii * 4 + 2] * inv[ii]), (bfloat16_t)(fg[ii * 4 + 3] * inv[ii])};
                *(device ushort4*)(op + id * 16) = as_type<ushort4>(o4);
            }
        }
    } else {
        const size_t pbase = ((size_t)g * NC + c) * R + r0;
        STEEL_PRAGMA_UNROLL
        for (int ii = 0; ii < 2; ++ii) {
            const int row = sm + ii * 8;
            if (row >= nrows) continue;
            device half* op = po + (pbase + row) * D + ds * DS + sn;
            STEEL_PRAGMA_UNROLL
            for (int id = 0; id < TD; ++id) {
                thread auto& fg = Ot.frag_at(0, id);
                *(device half4*)(op + id * 16) = half4(fg[ii * 4] * inv[ii], fg[ii * 4 + 1] * inv[ii], fg[ii * 4 + 2] * inv[ii], fg[ii * 4 + 3] * inv[ii]);
            }
            if (ds == 0 && sn == 0) { pm[pbase + row] = mrow[ii]; pl[pbase + row] = lrow[ii]; }
        }
    }
"""

# one thread per (kv head, row, 8 d-columns): out[n, h*D + d] = sum_c w_c l_c po_c / sum_c w_c l_c, w_c = exp2(m_c - max m)
_MERGE_SRC = r"""
    constexpr int R = GQA * N;
    constexpr int D8 = D / 8;
    const int idx = (int)thread_position_in_grid.x;
    if (idx >= HK * R * D8) return;
    const int d8 = idx % D8, r = (idx / D8) % R, g = idx / (D8 * R);
    const int NC = (int)pm_shape[1];
    const device float* pmr = pm + (size_t)g * NC * R + r;
    const device float* plr = pl + (size_t)g * NC * R + r;
    float M = -3.0e38f;
    for (int c = 0; c < NC; ++c) M = max(M, pmr[(size_t)c * R]);
    float L = 0.f;
    float4 o0 = 0.f, o1 = 0.f;
    for (int c = 0; c < NC; ++c) {
        const float w = fast::exp2(pmr[(size_t)c * R] - M) * plr[(size_t)c * R];
        L += w;
        const device half4* p = (const device half4*)(po + (((size_t)g * NC + c) * R + r) * D + d8 * 8);
        o0 += w * float4(p[0]); o1 += w * float4(p[1]);
    }
    const float inv = 1.f / L;
    o0 *= inv; o1 *= inv;
    const metal::vec<bfloat16_t, 4> b0 = {(bfloat16_t)o0.x, (bfloat16_t)o0.y, (bfloat16_t)o0.z, (bfloat16_t)o0.w};
    const metal::vec<bfloat16_t, 4> b1 = {(bfloat16_t)o1.x, (bfloat16_t)o1.y, (bfloat16_t)o1.z, (bfloat16_t)o1.w};
    device ushort4* op = (device ushort4*)(out + (size_t)(r % N) * (HK * GQA * D) + (size_t)(g * GQA + r / N) * D + d8 * 8);
    op[0] = as_type<ushort4>(b0); op[1] = as_type<ushort4>(b1);
"""

_partial_kernel = mx.fast.metal_kernel(name="tree_attn_partial", input_names=["q", "k", "v", "prm", "scl", "anc"],
                                       output_names=["po", "pm", "pl", "out"], header=_HEADER, source=_PARTIAL_SRC)
_merge_kernel = mx.fast.metal_kernel(name="tree_attn_merge", input_names=["po", "pm", "pl"],
                                     output_names=["out"], source=_MERGE_SRC)

LOG2E = 1.4426950408889634
RG = 3        # row groups per threadgroup
DS = 128      # d-columns per simdgroup


def chunk_subtiles(S):
    """32-key sub-tiles per chunk for a valid length S: about 5 chunks (measured optimum on the M5 base for RG=3:
    S=1.5k -> 512 keys/chunk, 3k -> 1024, 5k -> 1024, 8k -> 2048, 16k -> 4096; S <= 256 -> one chunk, no merge)."""
    C = 256
    while C < min(4096, (S + 4) // 5):
        C *= 2
    return C // 32


def attn_params(S, offset, scale):
    """The small parameter arrays of tree_attention (build once per forward, shared by all layers)."""
    return mx.array([S, offset, 0], mx.int32), mx.array([scale * LOG2E], mx.float32)


def tree_attention(q, k, v, S, offset, anc, scale, prm=None, scl=None, nsub=None):
    """q: [1, H, N, D] bf16; k, v: [1, HK, cap, D] bf16 cache buffers read in place (key j = cache row j, j < S = offset + N);
    keys j < offset (the cached prefix) are visible to every query, tree key offset + t is visible to node i iff bit t of
    anc[i] (uint32 [N]) is set. (prm, scl) = attn_params(S, offset, scale) may be passed in to share them across layers.
    Returns bf16 [N, H*D] (token-major, ready for o_proj)."""
    _, H, N, D = q.shape
    _, HK, cap, _ = k.shape
    GQA = H // HK
    R = GQA * N
    nsub = nsub or chunk_subtiles(S)
    C = 32 * nsub
    NC = (S + C - 1) // C
    tgx = ((R + 15) // 16 + RG - 1) // RG
    tpg = RG * (D // DS) * 32
    if prm is None:
        prm, scl = attn_params(S, offset, scale)
    final = NC == 1
    po, pm, pl, out = _partial_kernel(
        inputs=[q.reshape(H * N, D), k.reshape(HK, cap, D), v.reshape(HK, cap, D), prm, scl, anc],
        template=[("D", D), ("DS", DS), ("RG", RG), ("NSUB", nsub), ("N", N), ("GQA", GQA), ("HK", HK), ("FINAL", int(final))],
        grid=(tgx * tpg, HK, NC), threadgroup=(tpg, 1, 1),
        output_shapes=[(1,) if final else (HK, NC, R, D), (1,) if final else (HK, NC, R), (1,) if final else (HK, NC, R), (N, H * D) if final else (1,)],
        output_dtypes=[mx.float16, mx.float32, mx.float32, mx.bfloat16])
    if final:
        return out
    nthr = HK * R * (D // 8)
    (out,) = _merge_kernel(
        inputs=[po, pm, pl],
        template=[("D", D), ("N", N), ("GQA", GQA), ("HK", HK)],
        grid=(((nthr + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(N, H * D)], output_dtypes=[mx.bfloat16])
    return out


_available = None


def available():
    """True if the kernels compile and run on this machine (M5 tensor units); checked once on a tiny problem."""
    global _available
    if _available is None:
        try:
            q = mx.random.normal((1, 2, 3, 128)).astype(mx.bfloat16)
            k = mx.random.normal((1, 1, 256, 128)).astype(mx.bfloat16)
            v = mx.random.normal((1, 1, 256, 128)).astype(mx.bfloat16)
            ab = anc_bits([-1, 0, 0])
            ref = mx.fast.scaled_dot_product_attention(q, k[..., :43, :], v[..., :43, :], scale=0.1, mask=_tree_mask([-1, 0, 0], 40)).transpose(0, 2, 1, 3).reshape(3, 256)
            out = tree_attention(q, k, v, 43, 40, ab, 0.1, nsub=1)   # 2 chunks: partial + merge path
            out2 = tree_attention(q, k, v, 43, 40, ab, 0.1)          # single chunk: fused path
            err = max((out.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item(), (out2.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item())
            _available = err < 0.05
        except Exception:
            _available = False
    return _available


def anc_bits(parent):
    """uint32 [N]: bit j set iff node j is node i or an ancestor of i (parent[0] == -1)."""
    assert len(parent) <= 32, f"tree-attention kernel: at most 32 tree nodes ({len(parent)} given); the ancestor mask is 32 bits wide"
    bits = []
    for i in range(len(parent)):
        b, j = 0, i
        while j >= 0:
            b |= 1 << j
            j = parent[j]
        bits.append(b)
    return mx.array(bits, mx.uint32)


def _tree_mask(parent, offset):
    N = len(parent)
    m = [[False] * N for _ in range(N)]
    for i in range(N):
        j = i
        while j >= 0:
            m[i][j] = True
            j = parent[j]
    return mx.concatenate([mx.ones((N, offset), dtype=mx.bool_), mx.array(m)], axis=1)
