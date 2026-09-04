"""Native MTP-head drafter for Qwen3.5/3.6/3.8 (beam-expanded draft tree).

MTP forward for one pair (token t, hidden h):  x = fc(concat(norm_e(embed(t)), norm_h(h)));
x -> one full-attention decoder layer -> norm -> h' ; logits = lm_head(h').
The pair (x_{i+1}, hidden_i) predicts x_{i+2}. Drafting feeds h' back as the hidden.

Draft tree: the root pair (root token, true hidden of the last committed token) is depth 0; its
top-k children are expanded with (child token, parent's h'), attending to the committed prefix plus
their own ancestors (ancestor mask, positions = prefix + depth).
"""
import glob
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from .spec_decode import trim_by_cost
from .tree_verify import _rope


class SubLMHead:
    """lm_head restricted to a token subset (rows gathered from the target's quantized head)."""

    def __init__(self, lm_head, ids):
        from .patch_model import kernel_available
        from .qmm_small_m import qmm_small_m, tile_weights, untile_weights
        ids = np.asarray(ids, dtype=np.int32)
        pad = (-len(ids)) % 32
        if pad:                                      # rows must be a multiple of 32: repeat the last id
            ids = np.concatenate([ids, np.full(pad, ids[-1], dtype=np.int32)])
        self.ids = mx.array(ids)
        self.n_real = len(ids) - pad
        t = getattr(lm_head, "_tiled", None)
        w, sc, bi = untile_weights(*t) if t is not None else (lm_head["weight"], lm_head["scales"], lm_head["biases"])
        w, sc, bi = w[self.ids], sc[self.ids], bi[self.ids]
        if kernel_available():
            self.w, self._qmm = tile_weights(w, sc, bi), qmm_small_m
        else:
            self.w = (w, sc, bi)
            self._qmm = lambda h, w, sc, bi: mx.quantized_matmul(h, w, sc, bi, transpose=True, group_size=lm_head.group_size, bits=lm_head.bits)
        mx.eval(*self.w)

    def __call__(self, h):
        return self._qmm(h, *self.w)[..., :self.n_real]

    def to_token(self, idx):
        return self.ids[idx]


class MTPHead(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        D = args.hidden_size
        self.fc = nn.Linear(2 * D, D, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]  # full attention
        self.norm = nn.RMSNorm(D, eps=args.rms_norm_eps)


def load_mtp_head(path, args, quant_bits=None, group_size=64):
    path = os.path.expanduser(path)
    cfg = json.load(open(os.path.join(path, "config.json")))
    head = MTPHead(args)
    weights = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(f))
    norm_suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
                     ".k_norm.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
    clean = {}
    for k, v in weights.items():
        hf = k.startswith("mtp.")
        if hf:
            k = k[4:]
            if any(k.endswith(s) for s in norm_suffixes) and v.ndim == 1:
                v = v + 1.0
        clean[k] = v
    q = cfg.get("quantization")
    if q:
        nn.quantize(head, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in clean)
    head.load_weights(list(clean.items()), strict=True)
    if quant_bits and not q:
        nn.quantize(head, group_size=group_size, bits=quant_bits)
    mx.eval(head.parameters())
    return head


class MTPDrafter:
    def __init__(self, model, head_path, top_k=6, quant_bits=None, draft_vocab=None):
        self.model = model
        lm = model.language_model
        self.args = lm.args
        self.head = load_mtp_head(head_path, self.args, quant_bits)
        self.embed = lm.model.embed_tokens
        self.sub_head = SubLMHead(lm.lm_head, draft_vocab) if draft_vocab is not None else None
        self.lm_head = self.sub_head or lm.lm_head
        self.attn = self.head.layers[0].self_attn
        self.layer = self.head.layers[0]
        self.top_k = top_k
        self.cache = KVCache()          # MTP KV over committed pairs
        self.n_committed_pairs = 0      # entries in self.cache
        self.dims = int(self.attn.head_dim * self.args.partial_rotary_factor)

    # ---------------------------------------------------------------- core forward
    def _forward(self, tokens, hidden, pos, mask, logits=True):
        """tokens: [N] int; hidden: [N, D]; pos: [N] float positions; mask: [N, offset+N] bool or None.
        Appends N entries to the KV cache. Returns (logits [N, V], h' [N, D])."""
        N = tokens.shape[0]
        x = mx.concatenate([self.head.pre_fc_norm_embedding(self.embed(tokens)),
                            self.head.pre_fc_norm_hidden(hidden)], axis=-1)
        x = self.head.fc(x)
        layer, attn = self.layer, self.attn
        xn = layer.input_layernorm(x)[None]
        q_out = attn.q_proj(xn)
        queries, gate = mx.split(q_out.reshape(1, N, attn.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(1, N, -1)
        keys, values = attn.k_proj(xn), attn.v_proj(xn)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(1, N, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(1, N, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = _rope(queries, pos, self.dims, self.args.rope_theta)
        keys = _rope(keys, pos, self.dims, self.args.rope_theta)
        keys, values = self.cache.update_and_fetch(keys, values)
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(1, N, -1)
        r = attn.o_proj(out * mx.sigmoid(gate))[0]
        h = x + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.head.norm(h)
        return (self.lm_head(h) if logits else None), h

    def _trim_to(self, n):
        self.cache.offset = n

    # ---------------------------------------------------------------- drafter interface
    def prefill(self, tokens, hidden):
        """tokens: full prompt (len T+1); hidden: target pre-norm hidden for tokens[:T] ([T, D])."""
        T = hidden.shape[0]
        self.cache = KVCache()
        self.n_committed_pairs = 0
        pairs_tok = mx.array(tokens[1:T], mx.uint32)           # pairs i = 0..T-2: (x_{i+1}, hidden_i)
        if T - 1 > 0:
            pos = mx.arange(T - 1, dtype=mx.float32)
            self._forward(pairs_tok, hidden[:T - 1], pos, "causal" if T - 1 > 1 else None, logits=False)
            self.n_committed_pairs = T - 1
        self._last_hidden = hidden[T - 1]                         # true hidden of token x_{T-1}

    def accept(self, tokens, hidden):
        """tokens: accepted path tokens (root first, a of them); hidden: their true target hiddens [a, D].
        After this, the committed pairs are 0..L+a-2 and the new root pair will use hidden[-1]."""
        self._trim_to(self.n_committed_pairs)
        a = len(tokens)
        pair_tok = mx.array(tokens, mx.uint32)                                      # x_L .. x_{L+a-1}
        pair_hid = mx.concatenate([self._last_hidden[None], hidden[:a - 1]], axis=0)  # hidden_{L-1} .. hidden_{L+a-2}
        pos = mx.arange(self.n_committed_pairs, self.n_committed_pairs + a, dtype=mx.float32)
        self._forward(pair_tok, pair_hid, pos, "causal" if a > 1 else None, logits=False)
        self.n_committed_pairs += a
        self._last_hidden = hidden[a - 1]

    def state(self):
        """Snapshot for spec_decode.Session: KV rows below the offset are never rewritten, so the offset suffices."""
        return self.cache, self.n_committed_pairs, self._last_hidden

    def restore(self, s):
        self.cache, self.n_committed_pairs, self._last_hidden = s
        self._trim_to(self.n_committed_pairs)


class MTPBeamDrafter(MTPDrafter):
    """Beam expansion of the draft tree on the GPU with static shapes and a single device sync per
    round; tree selection (value-sorted prefix under the cost curve) stays in Python on ~D*B*k numbers."""

    def __init__(self, model, head_path, cost_curve, top_k=4, beam=4, depth=5, max_nodes=32,
                 min_value=0.02, draft_cost_per_level=0.06, quant_bits=8, draft_vocab=None):
        super().__init__(model, head_path, top_k=top_k, quant_bits=quant_bits, draft_vocab=draft_vocab)
        self.cost, self.draft_cost = cost_curve, draft_cost_per_level
        self.beam, self.depth, self.max_nodes, self.min_value = beam, depth, max_nodes, min_value

    def _level_forward(self, tokens, hidden, depth, anc_draft, n_prev):
        """tokens [B]; hidden [B, D]; anc_draft [B, depth] draft indices of ancestors (levels 0..depth-1).
        Returns (log-probs [B, Vs], h' [B, D])."""
        B = tokens.shape[0]
        base = self.n_committed_pairs
        own = mx.arange(n_prev, n_prev + B)[:, None]
        idx = mx.concatenate([anc_draft, own], axis=1) if depth > 0 else own    # [B, depth+1]
        cols = mx.arange(n_prev + B)[None, None, :]
        draft_mask = mx.any(idx[:, :, None] == cols, axis=1)                     # [B, n_prev+B]
        mask = mx.concatenate([mx.ones((B, base), dtype=mx.bool_), draft_mask], axis=1)
        pos = mx.full((B,), float(base + depth), dtype=mx.float32)
        logits, h = self._forward(tokens, hidden, pos, mask)
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return lp, h

    def draft(self, root_token, root_hidden=None):
        self._trim_to(self.n_committed_pairs)
        k, B, D = self.top_k, self.beam, self.depth
        tokens = mx.array([root_token], mx.uint32)
        hidden = self._last_hidden[None]
        cum = mx.zeros((1,), dtype=mx.float32)
        anc_draft = mx.zeros((1, 0), dtype=mx.int32)
        n_prev = 0
        levels = []   # per level: (cand_tok [Bl, k], cand_cum [Bl, k], sel [B_next] flat candidate idx)
        sel_hist = []
        for d in range(D):
            Bl = tokens.shape[0]
            lp, h = self._level_forward(tokens, hidden, d, anc_draft, n_prev)
            top_idx = mx.argpartition(-lp, k - 1, axis=-1)[:, :k]
            top_lp = mx.take_along_axis(lp, top_idx, axis=-1)
            order = mx.argsort(-top_lp, axis=-1)
            top_idx = mx.take_along_axis(top_idx, order, axis=-1)
            top_lp = mx.take_along_axis(top_lp, order, axis=-1)
            cand_tok = self.sub_head.to_token(top_idx) if self.sub_head is not None else top_idx   # [Bl, k]
            cand_cum = cum[:, None] + top_lp                                                     # [Bl, k]
            levels.append((cand_tok, cand_cum))
            if d == D - 1:
                break
            flat = cand_cum.reshape(-1)
            Bn = min(B, Bl * k)
            sel = mx.argsort(-flat)[:Bn]                                                         # flat candidate idx
            sel_hist.append(sel)
            row = sel // k
            tokens = cand_tok.reshape(-1)[sel].astype(mx.uint32)
            hidden = h[row]
            cum = flat[sel]
            own = mx.arange(n_prev, n_prev + Bl)
            anc_draft = mx.concatenate([anc_draft[row], own[row][:, None]], axis=1)
            n_prev += Bl
        mx.eval(*[t for lv in levels for t in lv], *sel_hist)
        # ---- build candidate list (python, small) ----
        toks, parents, values = [root_token], [-1], [1.0]
        offsets = []
        for d, (cand_tok, cand_cum) in enumerate(levels):
            ct, cc = cand_tok.tolist(), cand_cum.tolist()
            Bl = len(ct)
            sel = sel_hist[d - 1].tolist() if d > 0 else None
            offsets.append(len(toks))
            for b in range(Bl):
                parent = 0 if d == 0 else offsets[d - 1] + sel[b]
                for i in range(k):
                    toks.append(ct[b][i]); parents.append(parent); values.append(math.exp(cc[b][i]))
        # siblings with identical tokens (keep the first) and low-value nodes drop out before the cost-curve trim
        seen = set()
        for i in range(1, len(toks)):
            key = (parents[i], toks[i])
            if key in seen or values[i] < self.min_value:
                values[i] = 0.0
            seen.add(key)
        return trim_by_cost(toks, parents, values, self.cost, D * self.draft_cost, self.max_nodes)
