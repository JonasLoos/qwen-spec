"""Fused exact top-k / top-p / min-p / temperature sampling for bf16 logit rows.

One threadgroup (256 threads) per row. Exact top-K selection by radix select on the bf16 bit
patterns (two 256-bin histogram passes in threadgroup memory, no private arrays), then
logsumexp over the full row (so top-p is exact), nucleus/min-p filtering on the sorted top-K,
and one inverse-CDF draw.
"""
import mlx.core as mx

_SRC = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;    // 256 threads per row
    if (row >= N) return;
    const device bfloat16_t* x = logits + (size_t)row * V;
    const device ushort* xb = (const device ushort*)x;
    threadgroup atomic_uint hist[256];
    threadgroup uint sel_i[K];
    threadgroup float sel_v[K];
    threadgroup atomic_uint counters[2];     // [0] slots used, [1] ties taken
    threadgroup float tg_s[8];
    threadgroup uint thr_key;
    threadgroup uint need_ties;

    // ---- pass 1: histogram of the high byte of the order-preserving key ----
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    if (tid < 2) atomic_store_explicit(&counters[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        atomic_fetch_add_explicit(&hist[key >> 8], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0; uint hi = 0;
        for (int b = 255; b >= 0; --b) {
            uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (acc + c >= (uint)K) { hi = (uint)b; break; }
            acc += c;
        }
        thr_key = hi;
        need_ties = acc;          // elements strictly above the selected high-byte bin
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hi_sel = thr_key;
    const uint above_hi = need_ties;
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- pass 2: histogram of the low byte within the selected high-byte bin ----
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        if ((key >> 8) == hi_sel) atomic_fetch_add_explicit(&hist[key & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = above_hi; uint lo = 0;
        for (int b = 255; b >= 0; --b) {
            uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (acc + c >= (uint)K) { lo = (uint)b; break; }
            acc += c;
        }
        thr_key = (hi_sel << 8) | lo;
        need_ties = (uint)K - acc;   // how many elements equal to the threshold we take
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint T = thr_key; const uint ties = need_ties;
    // ---- pass 3: collect the top-K (unsorted) ----
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        if (key > T) {
            uint s = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
            if (s < (uint)K) { sel_i[s] = (uint)j; sel_v[s] = (float)x[j]; }
        } else if (key == T) {
            uint t = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
            if (t < ties) {
                uint s = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
                if (s < (uint)K) { sel_i[s] = (uint)j; sel_v[s] = (float)x[j]; }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- sort the K selected (descending), thread 0 ----
    if (tid == 0) {
        for (int a = 1; a < K; ++a) {
            float v = sel_v[a]; uint ii = sel_i[a]; int p = a - 1;
            while (p >= 0 && sel_v[p] < v) { sel_v[p + 1] = sel_v[p]; sel_i[p + 1] = sel_i[p]; --p; }
            sel_v[p + 1] = v; sel_i[p + 1] = ii;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- pass 4: logsumexp over the whole row at temperature ----
    const float inv_t = 1.0f / temp;
    const float gmax = sel_v[0] * inv_t;
    float s = 0.0f;
    for (int j = (int)tid; j < V; j += 256) s += metal::exp((float)x[j] * inv_t - gmax);
    s = simd_sum(s);
    if ((tid & 31) == 0) tg_s[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (int q = 0; q < 8; ++q) tot += tg_s[q];
        const float lse = gmax + metal::log(tot);
        const int kk = top_k > 0 ? min(top_k, K) : K;
        float cum = 0.0f; int keep = 0; float p0 = metal::exp(sel_v[0] * inv_t - lse);
        for (int i = 0; i < kk; ++i) {
            float pi = metal::exp(sel_v[i] * inv_t - lse);
            if (top_p < 1.0f && i > 0 && cum >= top_p) break;
            if (min_p > 0.0f && pi < min_p * p0) break;
            cum += pi; keep = i + 1;
        }
        float u = uniforms[row] * cum;
        float acc = 0.0f; int pick = keep - 1;
        for (int i = 0; i < keep; ++i) { acc += metal::exp(sel_v[i] * inv_t - lse); if (u < acc) { pick = i; break; } }
        out[row] = sel_i[pick];
        if (STATS) {          // processed distribution over the top-K (renormalized nucleus; zero outside it)
            for (int i = 0; i < K; ++i) {
                top_ids[row * K + i] = sel_i[i];
                top_probs[row * K + i] = i < keep ? metal::exp(sel_v[i] * inv_t - lse) / cum : 0.0f;
            }
        }
    }
"""

_kernels = {}


def _kernel(K):
    if K not in _kernels:
        _kernels[K] = mx.fast.metal_kernel(name=f"sample_radix_{K}", input_names=["logits", "uniforms", "temp", "top_k", "top_p", "min_p"],
                                           output_names=["out", "top_ids", "top_probs"], source=_SRC)
    return _kernels[K]


def sample_rows(logits, temp=1.0, top_k=20, top_p=1.0, min_p=0.0, K=None, stats=False):
    """logits [N, V] (bf16) -> tokens [N] uint32. Exact for top_k <= K (K = max(top_k, 32) by default; with top_k == 0 the
    nucleus is restricted to the top-K tokens). With `stats` also returns the processed distribution the draw was taken
    from: top_ids [N, K] uint32 sorted by logit, top_probs [N, K] float32 (temperature / top-k / top-p / min-p applied and
    renormalized, zero outside the kept set)."""
    if logits.dtype != mx.bfloat16:
        logits = logits.astype(mx.bfloat16)
    N, V = logits.shape
    K = K or max(top_k, 32)
    assert top_k <= K
    u = mx.random.uniform(shape=(N,))
    out, ids, probs = _kernel(K)(
        inputs=[logits, u, mx.array(float(temp)), mx.array(int(top_k)), mx.array(float(top_p)), mx.array(float(min_p))],
        template=[("N", N), ("V", V), ("K", K), ("STATS", int(stats))],
        grid=(N * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N,), (N, K), (N, K)] if stats else [(N,), (1,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32],
    )
    return (out, ids, probs) if stats else out
