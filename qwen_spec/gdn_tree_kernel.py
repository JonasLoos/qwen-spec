"""Metal kernel for the Gated-DeltaNet tree path walk.

For every (node n, value-head h): start from p = q[n,h]; walk the ancestors j = anc[n, s] (s = 0 is
the node itself), accumulating  c = beta_j * <k_j, p>;  y += c * v_j;  p = alpha_j * (p - c * k_j).
Returns p (to be multiplied with the prefix state) and y (the intra-tree contribution).
One simdgroup per (n, h); lane l owns elements [4l, 4l+4) of the 128-dim vectors.
"""
import mlx.core as mx

_SRC = r"""
    const uint sg_global = thread_position_in_grid.x / 32;      // (n, h) index
    const uint lane = thread_position_in_grid.x % 32;
    const int n = (int)(sg_global / HV), h = (int)(sg_global % HV);
    if (n >= N) return;
    const int base = (n * HV + h) * 128 + (int)lane * 4;
    float4 p = *(const device float4*)(q + base);
    float4 y = float4(0.0f);
    for (int s = 0; s < S; ++s) {
        const int j = anc[n * S + s];
        if (j < 0) break;
        const int jb = (j * HV + h) * 128 + (int)lane * 4;
        const float4 kj = *(const device float4*)(k + jb);
        const float4 vj = *(const device float4*)(v + jb);
        const float bj = beta[j * HV + h];
        const float aj = alpha[j * HV + h];
        float c = simd_sum(kj.x * p.x + kj.y * p.y + kj.z * p.z + kj.w * p.w) * bj;
        y += c * vj;
        p = aj * (p - c * kj);
    }
    *(device float4*)(p_out + base) = p;
    *(device float4*)(y_out + base) = y;
"""

_kernel = mx.fast.metal_kernel(
    name="gdn_tree_walk",
    input_names=["q", "k", "v", "alpha", "beta", "anc"],
    output_names=["p_out", "y_out"],
    source=_SRC,
)


def gdn_tree_walk(q, k, v, alpha, beta, anc):
    """q,k: [N, Hv, 128] f32; v: [N, Hv, 128] f32; alpha, beta: [N, Hv] f32; anc: [N, S] int32."""
    N, HV, Dk = q.shape
    S = anc.shape[1]
    assert Dk == 128 and v.shape[-1] == 128
    n_sg = N * HV
    p, y = _kernel(
        inputs=[q, k, v, alpha, beta, anc],
        template=[("N", N), ("HV", HV), ("S", S)],
        grid=(n_sg * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, v.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return p, y
