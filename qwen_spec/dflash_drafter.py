"""DFlash2 block-diffusion drafter wrapped for the tree-speculation loop.

The drafter consumes the target's residual stream after layers `target_layer_ids` (fused) as
KV-injected context, and drafts a block of `block_size - 1` tokens in ONE pass. DFlash2's
candidate selector scores K x K transitions between adjacent slots; we use those transition
distributions to build a draft TREE (parent-conditioned branching) at no extra drafter cost.
"""
import glob
import json
import os

import mlx.core as mx
import mlx.nn as nn

from .dflash_model import DFlashConfig, DFlashDraftModel
from .mtp_drafter import SubLMHead
from .spec_decode import trim_by_cost


def load_dflash(path, quant_bits=4, group_size=64, cache_dir=None):
    """Load the drafter; the quantized weights are cached in `cache_dir` (default: next to the weights) on first use."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    dfc = cfg.get("dflash_config", {})
    rope = cfg.get("rope_parameters") or {}
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=cfg.get("rope_theta", rope.get("rope_theta", 1e6)),
        max_position_embeddings=cfg["max_position_embeddings"], block_size=int(dfc.get("block_size", cfg.get("block_size"))),
        target_layer_ids=tuple(dfc.get("target_layer_ids") or cfg["target_layer_ids"]),
        mask_token_id=dfc.get("mask_token_id", cfg.get("mask_token_id", 0)),
        rope_scaling=cfg.get("rope_scaling"), layer_types=tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]),
        sliding_window=cfg.get("sliding_window"), final_logit_softcapping=dfc.get("final_logit_softcapping", cfg.get("final_logit_softcapping")),
        selector_rank=int(dfc.get("selector_rank") or 0), selector_top_k=int(dfc.get("selector_top_k") or 0),
        conv_kernel_size=int(dfc.get("conv_kernel_size") or 0), conv_group_size=int(dfc.get("conv_group_size") or 16),
        output_multiplier=float(dfc.get("output_multiplier") or 1.0))
    drafter = DFlashDraftModel(config)
    pred = lambda p, m: isinstance(m, nn.Linear) and "_conv" not in p and "candidate_selector" not in p
    cached = os.path.join(cache_dir or path, f"quantized_{quant_bits}bit_g{group_size}.safetensors") if quant_bits else None
    if cached and os.path.exists(cached):
        nn.quantize(drafter, group_size=group_size, bits=quant_bits, class_predicate=pred)
        drafter.load_weights(list(mx.load(cached).items()))
    else:
        weights = {}
        for st in glob.glob(os.path.join(path, "*.safetensors")):
            if "quantized_" not in os.path.basename(st):
                weights.update(mx.load(st))
        drafter.load_weights(list(weights.items()))
        if quant_bits:
            nn.quantize(drafter, group_size=group_size, bits=quant_bits, class_predicate=pred)
            mx.eval(drafter.parameters())
            from mlx.utils import tree_flatten
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            mx.save_safetensors(cached, dict(tree_flatten(drafter.parameters())))
    mx.eval(drafter.parameters())
    return drafter, config


def load_calib(path):
    """calib.json (scripts/calibrate.py) -> callable(values, depths) returning calibrated acceptance probabilities."""
    import numpy as np
    spec = json.load(open(path))
    by_depth = {}
    for g in spec.values():
        for dd in g["depths"]:
            by_depth[dd] = (np.array(g["log10_v"]), np.array(g["p"]))
    dmax = max(by_depth)

    def calib(values, depths):
        v = np.asarray(values, np.float64)
        lv = np.log10(np.clip(v, 1e-6, 1.0))
        dep = np.minimum(np.asarray(depths), dmax)
        out = v.copy()
        for dd in np.unique(dep):
            if int(dd) == 0:
                continue
            xs, ps = by_depth[int(dd)]
            m = dep == dd
            out[m] = np.interp(lv[m], xs, ps)
        out[0] = 1.0
        return out.tolist()
    return calib


class CtxCache:
    """Preallocated windowed K/V cache for the drafter's context rows. `offset` counts all rows ever
    appended (absolute RoPE positions); the last `window` rows are retained at buffer rows [start, end).
    Rows [end, end + scratch) are scratch for the current block's own K/V, so attention reads one
    contiguous slice [start, end + L) and nothing is concatenated per step."""

    STEP = 512

    def __init__(self, window, scratch=32):
        self.window, self.scratch = window, scratch
        self.keys = self.values = None
        self.offset = 0
        self.start = self.end = 0

    def _compact(self, n, like):
        keep = 0 if self.keys is None else min(self.window, self.end - self.start)
        cap = self.window + n + self.scratch + self.STEP
        B, H, _, D = like.shape
        k = mx.zeros((B, H, cap, D), dtype=like.dtype)
        v = mx.zeros((B, H, cap, D), dtype=like.dtype)
        if keep:
            k[..., :keep, :] = self.keys[..., self.end - keep:self.end, :]
            v[..., :keep, :] = self.values[..., self.end - keep:self.end, :]
        self.keys, self.values, self.start, self.end = k, v, 0, keep

    def append(self, keys, values):
        n = keys.shape[2]
        if self.keys is None or self.end + n + self.scratch > self.keys.shape[2]:
            self._compact(n, keys)
        self.keys[..., self.end:self.end + n, :] = keys
        self.values[..., self.end:self.end + n, :] = values
        self.end += n
        self.offset += n
        self.start = max(self.start, self.end - self.window)

    @property
    def n_rows(self):
        return self.end - self.start

    def state(self):
        """Copy of the retained rows (for spec_decode.Session snapshots; the buffers are mutated in place)."""
        if self.keys is None:
            return None
        k = mx.contiguous(self.keys[..., self.start:self.end, :]); v = mx.contiguous(self.values[..., self.start:self.end, :])
        mx.eval(k, v)
        return self.offset, k, v

    def restore(self, s):
        self.keys = self.values = None
        self.offset = self.start = self.end = 0
        if s is not None:
            _, k, v = s
            self.append(k, v)
            self.offset = s[0]

    def with_block(self, k, v):
        """Place the block's K/V in the scratch rows and return (keys, values) over [start, end + L)."""
        L = k.shape[2]
        self.keys[..., self.end:self.end + L, :] = k
        self.values[..., self.end:self.end + L, :] = v
        return self.keys[..., self.start:self.end + L, :], self.values[..., self.start:self.end + L, :]


class DFlashDrafter:
    def __init__(self, model, path, cost_curve=None, quant_bits=4, draft_vocab=None, branch=6, max_nodes=16,
                 min_value=0.02, draft_cost=0.08, frontier=None, block_size=None, calib=None, cache_dir=None):
        self.drafter, self.cfg = load_dflash(path, quant_bits, cache_dir=cache_dir)
        if block_size:                      # the drafter was trained with 8-slot blocks; its slots stay useful up to 15
            self.cfg.block_size = block_size
        if quant_bits == 4:
            from .patch_model import patch_quantized_linears
            patch_quantized_linears(self.drafter, fuse=False)     # context appends use only k/v: fused q/k/v would waste the q part
        lm = model.language_model
        self.embed = lm.model.embed_tokens
        self.sub_head = SubLMHead(lm.lm_head, draft_vocab) if draft_vocab is not None else None
        self.lm_head = self.sub_head or lm.lm_head
        self.tap = list(self.cfg.target_layer_ids)
        self.window = (self.cfg.sliding_window or 10**9) - self.cfg.block_size   # ctx + block never exceeds the window -> plain causal mask
        self.caches = [CtxCache(self.window, scratch=self.cfg.block_size) for _ in self.drafter.layers]
        self.cap = self.cfg.block_size - 1                                       # drafted slots per block
        self.branch, self.max_nodes, self.min_value, self.draft_cost = branch, max_nodes, min_value, draft_cost
        self.frontier = frontier or branch * 2
        self.cost = cost_curve or {1: 1.0}
        self.calib = calib        # optional: (values, depths) -> calibrated acceptance probabilities (lists)

    # ------------------------------------------------------------------ context
    def _append_ctx(self, fused):
        """fused: [S, n_tap*H] target hidden of newly committed tokens -> drafter ctx KV."""
        h_ctx = self.drafter.project_ctx(fused[None])
        B, S, _ = h_ctx.shape
        rope = self.drafter.rope
        for layer, c in zip(self.drafter.layers, self.caches):
            attn = layer.self_attn
            x = h_ctx
            if S > self.window:                                  # only the last window rows matter
                c.offset += S - self.window
                x = h_ctx[:, -self.window:]
            k = attn.k_norm(attn.k_proj(x).reshape(B, x.shape[1], attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(B, x.shape[1], attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            c.append(rope(k, offset=c.offset), v)

    def prefill(self, tokens, fused):
        self.caches = [CtxCache(self.window, scratch=self.cfg.block_size) for _ in self.drafter.layers]
        self._append_ctx(fused)

    def accept(self, tokens, fused):
        self._append_ctx(fused)

    def state(self):
        return [c.state() for c in self.caches]

    def restore(self, s):
        for c, x in zip(self.caches, s):
            c.restore(x)

    # ------------------------------------------------------------------ block forward (no ctx append)
    def _attn(self, attn, x, c):
        B, L, _ = x.shape
        rope = self.drafter.rope
        ctx_len = c.n_rows
        q = attn.q_norm(attn.q_proj(x).reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = rope(q, offset=c.offset); k = rope(k, offset=c.offset)
        if c.keys is not None:
            keys, values = c.with_block(k, v)
        else:
            keys, values = k, v
        assert not attn.is_sliding or ctx_len + L <= attn.sliding_window
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask="causal")
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def _block_hidden(self, block):
        d = self.drafter
        h = self.embed(block)
        for layer, c in zip(d.layers, self.caches):
            a = layer.input_layernorm(h)
            if layer.attention_conv is not None:
                a, oc = layer.attention_conv.prepare(a)
            att = self._attn(layer.self_attn, a, c)
            if layer.attention_conv is not None:
                att = layer.attention_conv.finish(att, oc)
            h = h + att
            m = layer.post_attention_layernorm(h)
            if layer.mlp_conv is not None:
                m, oc = layer.mlp_conv.prepare(m)
            mm = layer.mlp(m)
            if layer.mlp_conv is not None:
                mm = layer.mlp_conv.finish(mm, oc)
            h = h + mm
        return d.norm(h[:, 1:])[0]            # mask slots only: [block-1, H]

    # ------------------------------------------------------------------ drafting
    def draft(self, root_token, root_hidden=None):
        d, sel = self.drafter, self.drafter.candidate_selector
        bs = self.cfg.block_size
        block = mx.array([[root_token] + [self.cfg.mask_token_id] * (bs - 1)], mx.uint32)
        hidden = self._block_hidden(block)[:self.cap]                          # [cap, H]
        logits = self.lm_head(hidden)
        k = sel.top_k
        cand_idx = mx.argpartition(logits, kth=-k, axis=-1)[:, -k:]
        unary = d._transform_unary(mx.take_along_axis(logits, cand_idx, axis=-1))
        cand_ids = (self.sub_head.to_token(cand_idx) if self.sub_head is not None else cand_idx).astype(mx.int32)
        scores = sel.lattice(cand_ids, unary, hidden, root_token)               # [cap, K, K] (slot, pred, cand)
        # tree: parent-conditioned branching through the selector lattice
        probs = mx.softmax(scores, axis=-1)                                       # P(cand | slot, pred)
        mx.eval(probs, cand_ids)
        P, C = probs.tolist(), cand_ids.tolist()
        toks, parents, values = [root_token], [-1], [1.0]
        frontier = [(0, 0, 1.0)]     # (node, pred index into slot-1 candidates (0 = anchor row), value)
        for s in range(self.cap):
            nxt = []
            for node, pred, val in frontier:
                row = P[s][pred]
                order = sorted(range(k), key=lambda i: -row[i])[:self.branch]
                seen = set()
                for i in order:
                    v = val * row[i]
                    t = C[s][i]
                    if v < self.min_value or t in seen:
                        continue
                    seen.add(t)
                    toks.append(t); parents.append(node); values.append(v)
                    nxt.append((len(toks) - 1, i, v))
            nxt.sort(key=lambda x: -x[2])
            frontier = nxt[:self.frontier]
            if not frontier:
                break
        return self._trim(toks, parents, values)

    def _trim(self, toks, parents, values):
        raw = values
        if self.calib is not None:
            depth = [0] * len(toks)
            for i in range(1, len(toks)):
                depth[i] = depth[parents[i]] + 1
            values = self.calib(values, depth)
        return trim_by_cost(toks, parents, values, self.cost, self.draft_cost, self.max_nodes, raw_values=raw)
