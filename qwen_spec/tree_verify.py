"""Tree verification forward for Qwen3.5/3.6/3.8 hybrid models (mlx-lm `qwen3_5`).

One forward pass computes next-token logits for every node of a token tree that hangs off the
current cache state. Nodes must be in BFS/topological order (parents before children, node 0 =
root = the current token that is not yet in the cache).

* Gated DeltaNet layers use the parallel form: y_i = S_prefix p_i + sum_{j on path} beta_j c_j v_j,
  where p_i is the node's query propagated backwards through its ancestors' rank-1 transforms.
  The prefix state is read once per layer (one batched matmul); no per-node state is materialized.
* Full-attention layers use an ancestor mask with per-node positions (prefix + depth); the attention runs in the
  flash-decoding tree kernel of tree_attn.py (ancestor bits, reads the cache buffers in place; SDPA where it is
  unavailable).
* `commit(path)` advances the caches along the accepted path (replay of the recurrence for the
  accepted tokens, compaction of the KV cache).
"""
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g, gated_delta_update
from .gdn_tree_kernel import gdn_tree_walk
from . import tree_attn

ASYNC_EVERY = 8   # dispatch the graph every k layers so the GPU runs while Python builds the rest


@dataclass
class Tree:
    tokens: list[int]
    parent: list[int]   # parent[0] == -1

    def __post_init__(self):
        n = len(self.tokens)
        assert len(self.parent) == n and self.parent[0] == -1
        self.depth = [0] * n
        for i in range(1, n):
            assert 0 <= self.parent[i] < i, "nodes must be in topological order"
            self.depth[i] = self.depth[self.parent[i]] + 1
        self.max_depth = max(self.depth)
        # anc[i][s] = s-th ancestor of i (anc[i][0] = i), -1 beyond the root
        self.anc = [[-1] * (self.max_depth + 1) for _ in range(n)]
        for i in range(n):
            j = i
            for s in range(self.max_depth + 1):
                self.anc[i][s] = j
                j = self.parent[j] if j >= 0 else -1

    @property
    def n(self):
        return len(self.tokens)


def _rope(x, pos, dims, base):
    """RoPE (non-traditional, rotate halves) on the first `dims` dims with per-token positions.
    x: [B, H, N, D]; pos: [N] float."""
    half = dims // 2
    inv = base ** (-mx.arange(0, half, dtype=mx.float32) * 2 / dims)
    ang = pos[:, None] * inv[None, :]                       # [N, half]
    cos, sin = mx.cos(ang), mx.sin(ang)
    x1, x2, rest = x[..., :half].astype(mx.float32), x[..., half:dims].astype(mx.float32), x[..., dims:]
    rot = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return mx.concatenate([rot, rest], axis=-1)


class TreeVerifier:
    def __init__(self, model, cache, tap=None):
        self.model = model
        self.tap = set(tap) if tap else set()   # layer indices whose outputs are returned fused
        lm = model.language_model
        self.tm = lm.model
        self.lm_head = lm.lm_head
        self.cache = cache
        self.args = lm.args
        # per-layer records of the last tree forward (needed by commit)
        self._lin = {}
        self._attn_offset = None
        self._attn = None          # (anc_bits, prm, scl) for the tree-attention kernel, None -> SDPA with self._mask

    # ------------------------------------------------------------------ linear attention
    def _linear_tree(self, layer_idx, mod, x, hist_idx, conv_w):
        """x: [N, D] normalized input. Returns [N, D]."""
        c = self.cache[layer_idx]
        N = x.shape[0]
        qkv = mod.in_proj_qkv(x)                                   # [N, conv_dim]
        z = mod.in_proj_z(x).reshape(N, mod.num_v_heads, mod.head_v_dim)
        b = mod.in_proj_b(x)
        a = mod.in_proj_a(x)
        conv_state = c[0][0] if c[0] is not None else mx.zeros((mod.conv_kernel_size - 1, qkv.shape[-1]), dtype=x.dtype)
        hist = mx.concatenate([conv_state, qkv], axis=0)           # [3 + N, conv_dim]
        # depthwise causal conv over each node's own path (taps 0..3 = 3rd ancestor .. self)
        conv_out = (hist[hist_idx].astype(mx.float32) * conv_w.astype(mx.float32)).sum(axis=1)
        conv_out = nn.silu(conv_out).astype(x.dtype)                # [N, conv_dim]
        q, k, v = [t.reshape(N, h, d) for t, h, d in zip(
            mx.split(conv_out, [mod.key_dim, 2 * mod.key_dim], -1),
            [mod.num_k_heads, mod.num_k_heads, mod.num_v_heads],
            [mod.head_k_dim, mod.head_k_dim, mod.head_v_dim])]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        rep = mod.num_v_heads // mod.num_k_heads
        qf = mx.repeat(q, rep, axis=1).astype(mx.float32)          # [N, Hv, Dk]
        kf = mx.repeat(k, rep, axis=1).astype(mx.float32)
        vf = v.astype(mx.float32)                                  # [N, Hv, Dv]
        beta = mx.sigmoid(b).astype(mx.float32)                    # [N, Hv]
        alpha = compute_g(mod.A_log, a, mod.dt_bias).astype(mx.float32)  # [N, Hv]

        # walk each node's path from itself up to the root (kernel, vectorized over nodes):
        # p = query propagated through the ancestors' rank-1 transforms, y = the ancestors' contributions
        p, y = gdn_tree_walk(qf, kf, vf, alpha, beta, self._anc)
        # prefix-state term: y += S0 p   (S0: [Hv, Dv, Dk], read once)
        S0 = c[1][0] if c[1] is not None else mx.zeros((mod.num_v_heads, mod.head_v_dim, mod.head_k_dim), dtype=mx.float32)
        y = y + mx.matmul(p.transpose(1, 0, 2), S0.transpose(0, 2, 1)).transpose(1, 0, 2)
        y = y.astype(x.dtype)
        out = mod.norm(y, z)
        out = mod.out_proj(out.reshape(N, -1))
        self._lin[layer_idx] = (hist, q, k, v, a, b)
        return out

    # ------------------------------------------------------------------ full attention
    def _attn_tree(self, layer_idx, mod, x, mask, pos):
        c = self.cache[layer_idx]
        N = x.shape[0]
        x = x[None]
        q_out = mod.q_proj(x)
        queries, gate = mx.split(q_out.reshape(1, N, mod.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(1, N, -1)
        keys, values = mod.k_proj(x), mod.v_proj(x)
        queries = mod.q_norm(queries).transpose(0, 2, 1, 3)
        keys = mod.k_norm(keys.reshape(1, N, mod.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(1, N, mod.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        dims = int(mod.head_dim * self.args.partial_rotary_factor)
        queries = _rope(queries, pos, dims, self.args.rope_theta)
        keys = _rope(keys, pos, dims, self.args.rope_theta)
        keys, values = c.update_and_fetch(keys, values)          # writes the tree rows into the cache buffers
        if self._attn is not None:
            anc, prm, scl = self._attn                           # kernel reads the full cache buffers, returns [N, H*D] token-major
            out = tree_attn.tree_attention(queries, c.keys, c.values, c.offset, self._attn_offset, anc, mod.scale, prm, scl)[None]
        else:
            out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=mod.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(1, N, -1)
        return mod.o_proj(out * mx.sigmoid(gate))[0]

    # ------------------------------------------------------------------ forward
    def prefill(self, tokens, chunk=2048):    # bigger chunks: the per-call weight un-tiling of the prefill path is paid once per chunk
        """Process `tokens` into the cache with the standard (sequential) forward.
        Returns the pre-final-norm hidden states [T, D] (input of the MTP drafter)."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        tm = self.tm
        hs = []
        fused_parts = []
        for i in range(0, len(tokens), chunk):
            x = mx.array(tokens[i:i + chunk], mx.uint32)[None]
            h = tm.embed_tokens(x)
            fa_mask = create_attention_mask(h, self.cache[tm.fa_idx])
            ssm_mask = create_ssm_mask(h, self.cache[tm.ssm_idx])
            taps = []
            for li, (layer, c) in enumerate(zip(tm.layers, self.cache)):
                h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
                if li in self.tap:
                    taps.append(h[0])
            mx.eval(h, *taps)
            hs.append(h[0])
            if taps:
                fused_parts.append(mx.concatenate(taps, axis=-1))
        self.last_fused = mx.concatenate(fused_parts, axis=0) if fused_parts else None
        return mx.concatenate(hs, axis=0) if len(hs) > 1 else hs[0]

    def forward(self, tree: Tree):
        """Returns (logits [N, V], pre-norm hidden [N, D]) for all nodes."""
        N = tree.n
        tm = self.tm
        tokens = mx.array(tree.tokens, mx.uint32)
        h = tm.embed_tokens(tokens)                                # [N, D]
        self._anc = mx.array(tree.anc, mx.int32)                  # [N, Dmax+1]
        # conv history index per node and tap: taps 0..3 -> 3rd ancestor .. self; missing ancestors
        # fall back to the prefix conv state rows (row 2 = last prefix token, then 1, then 0)
        hist_idx = []
        for i in range(N):
            d = tree.depth[i]
            row = []
            for s in (3, 2, 1, 0):                                  # tap s <-> s-th ancestor
                if s <= d:
                    row.append(3 + tree.anc[i][s])
                else:
                    row.append(2 - (s - d - 1))                     # prefix rows 2,1,0
            hist_idx.append(row)
        hist_idx = mx.array(hist_idx, mx.int32)                   # [N, 4]
        # attention: ancestor mask over [prefix | tree]
        attn_cache = self.cache[tm.fa_idx]
        offset = attn_cache.offset
        self._attn_offset = offset
        mask = self._attn = None
        if tree_attn.available():
            self._attn = (tree_attn.anc_bits(tree.parent), *tree_attn.attn_params(offset + N, offset, tm.layers[tm.fa_idx].self_attn.scale))
        else:
            anc_mat = [[False] * N for _ in range(N)]
            for i in range(N):
                for j in tree.anc[i]:
                    if j >= 0:
                        anc_mat[i][j] = True
            mask = mx.concatenate([mx.ones((N, offset), dtype=mx.bool_), mx.array(anc_mat)], axis=1)
        pos = mx.array([offset + d for d in tree.depth], mx.float32)

        taps = []
        for li, layer in enumerate(tm.layers):
            xn = layer.input_layernorm(h)
            if layer.is_linear:
                mod = layer.linear_attn
                conv_w = mod.conv1d.weight[:, :, 0].T                  # [4, conv_dim]
                r = self._linear_tree(li, mod, xn, hist_idx, conv_w)
            else:
                r = self._attn_tree(li, layer.self_attn, xn, mask, pos)
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if li in self.tap:
                taps.append(h)
            if ASYNC_EVERY and li % ASYNC_EVERY == ASYNC_EVERY - 1:
                mx.async_eval(h)
        self.last_fused = mx.concatenate(taps, axis=-1) if taps else None
        return self.lm_head(tm.norm(h)), h

    # ------------------------------------------------------------------ commit
    def commit(self, path: list[int]):
        """Advance caches along the accepted root-to-node path (list of node indices, starting at 0)."""
        a = len(path)
        idx = mx.array(path, mx.int32)
        tm = self.tm
        off = self._attn_offset
        # KV compaction: gather the accepted rows of every attention layer first and evaluate them, so that the buffers
        # are unreferenced when they are written and the slice updates run in place (a pending gather from the same
        # buffer would force a copy of the whole cache per layer, ~3 ms at 3k context, ~8 ms at 8k)
        rows = {li: (c.keys[..., off + idx, :], c.values[..., off + idx, :]) for li, c in enumerate(self.cache) if not tm.layers[li].is_linear}
        mx.eval(*[x for kv in rows.values() for x in kv])
        for li, layer in enumerate(tm.layers):
            c = self.cache[li]
            if layer.is_linear:
                mod = layer.linear_attn
                hist, q, k, v, aa, b = self._lin[li]
                c[0] = mx.concatenate([hist[:3], hist[3 + idx]], axis=0)[-3:][None]
                _, c[1] = gated_delta_update(q[idx][None], k[idx][None], v[idx][None], aa[idx][None], b[idx][None],
                                             mod.A_log, mod.dt_bias, c[1], None)
            else:
                c.keys[..., off:off + a, :], c.values[..., off:off + a, :] = rows[li]
                c.offset = off + a
        self._lin = {}
