"""DFlash block-diffusion drafter model (MLX), vendored from z-lab/dflash (MIT) via mlx-dspark and trimmed to the
classes this package uses: the drafter's config, attention / MLP / grouped-convolution layers, the DFlash 2 candidate
selector's transition lattice, the context projection and the unary-logit transform. The block forward, context
cache and tree construction live in dflash_drafter.py.

Upstream: https://github.com/z-lab/dflash  (dflash/model_mlx.py)
Paper:    DFlash: Block Diffusion for Flash Speculative Decoding, Chen et al., arXiv:2602.06036
DFlash 2 components (grouped dynamic convolution, candidate selector) follow the SGLang reference implementation
(sgl-project/sglang, srt/models/dflash.py, Apache-2.0).

MIT License. Copyright (c) 2026 Z Lab.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    # DFlash 2 (0 = absent -> DFlash 1 behavior, byte-identical)
    selector_rank: int = 0
    selector_top_k: int = 0
    conv_kernel_size: int = 0        # taps; 2 for every published DFlash 2 head
    conv_group_size: int = 16        # channels sharing one dynamic coefficient correction
    output_multiplier: float = 1.0   # scales the selector's unary logits (the bilinear term is trained against the transformed scale)


def _build_rope(head_dim, rope_theta, max_position_embeddings, rope_scaling):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)


class DFlashGroupedConv(nn.Module):
    """DFlash 2: grouped dynamic depthwise K-tap convolution across one draft block.

    Wraps one sublayer: ``prepare`` convolves its input and returns the coefficient set for
    ``finish`` to convolve its output — BOTH sets come from one projection of the input.
    Each coefficient is a learned per-channel base plus a content-adaptive per-group
    correction (``group_size`` channels share one correction). Applied only to the block
    rows (the ctx path never sees it), so the zero-padded tap shift is exactly the
    reference's block-boundary mask: position 0 (the anchor) has no in-block predecessor,
    position 1 reads the anchor's representation.
    """

    def __init__(self, hidden_size: int, taps: int, group_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}")
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        # [side (in/out), tap, channel] — identity at init (tap 0 = 1), the layout training exports
        self.base_kernel = mx.concatenate(
            [mx.ones((2, 1, hidden_size)), mx.zeros((2, taps - 1, hidden_size))], axis=1)
        self.kernel_projection = nn.Linear(hidden_size, 2 * taps * self.num_groups, bias=False)

    def _convolve(self, x, delta, side: int):
        # x [B, L, H]; delta [B, L, taps, num_groups]
        B, L, H = x.shape
        xg = x.reshape(B, L, self.num_groups, self.group_size)
        coeff = (self.base_kernel[side].reshape(1, 1, self.taps, self.num_groups, self.group_size)
                 + delta[..., None])
        out = coeff[:, :, 0] * xg
        for t in range(1, self.taps):
            shifted = mx.pad(xg[:, :-t], ((0, 0), (t, 0), (0, 0), (0, 0)))
            out = out + coeff[:, :, t] * shifted
        return out.reshape(B, L, H)

    def prepare(self, x):
        coeff = self.kernel_projection(x).reshape(*x.shape[:-1], 2, self.taps, self.num_groups)
        return self._convolve(x, coeff[..., 0, :, :], 0), coeff[..., 1, :, :]

    def finish(self, y, delta):
        return self._convolve(y, delta, 1)


class CandidateSelector(nn.Module):
    """DFlash 2: scores the K x K transitions between adjacent mask slots.

    ``lattice`` builds ``scores[slot, p, c] = unary[slot, c] + <A[pred] * proj(h_slot), B[c]>``
    where the predecessors of slot 0 are all the verified anchor token and of slot s are
    slot s-1's candidates. Scores are fp32 like the reference (bf16 bilinear + fp32 unary).
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = mx.zeros((vocab_size, rank))
        self.successor_codebook = mx.zeros((vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def lattice(self, candidate_ids, unary_logits, hidden, anchor_id: int):
        # candidate_ids [g, K] int; unary_logits [g, K] fp32; hidden [g, H] (post-final-norm)
        k = candidate_ids.shape[1]
        h = self.hidden_projection(hidden)                                    # [g, r]
        anchor = mx.full((1, k), anchor_id, dtype=candidate_ids.dtype)
        pred_ids = mx.concatenate([anchor, candidate_ids[:-1]], axis=0)       # [g, K]
        pre = self.predecessor_codebook[pred_ids] * h[:, None, :]             # [g, K, r]
        suc = self.successor_codebook[candidate_ids]                          # [g, K, r]
        bilinear = pre @ suc.transpose(0, 2, 1)                               # [g, Kpred, Kcand]
        return unary_logits[:, None, :].astype(mx.float32) + bilinear.astype(mx.float32)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.conv_kernel_size:
            self.attention_conv = DFlashGroupedConv(
                config.hidden_size, config.conv_kernel_size, config.conv_group_size)
            self.mlp_conv = DFlashGroupedConv(
                config.hidden_size, config.conv_kernel_size, config.conv_group_size)
        else:
            self.attention_conv = None
            self.mlp_conv = None


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ("full_attention",) * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim, config.rope_theta, config.max_position_embeddings, config.rope_scaling
        )
        self.candidate_selector = (
            CandidateSelector(config.hidden_size, config.vocab_size,
                              config.selector_rank, config.selector_top_k)
            if config.selector_rank else None
        )

    def project_ctx(self, target_hidden):
        """Fused target hidden [B, S, taps*H] -> the drafter's context rows [B, S, H]
        (``hidden_norm(fc(.))``). Per-position, so computing it over any slice matches the
        full-width result up to qmm width ulps (fp-tie class)."""
        return self.hidden_norm(self.fc(target_hidden))

    def _transform_unary(self, logits):
        """DFlash 2 unary-logit transform (reference ``_transform_unary_logits``): fp32, then the
        target's output multiplier + softcap; the bilinear selector term is trained against this scale."""
        logits = logits.astype(mx.float32)
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits
