"""Tree speculative decoding loop.

Drafter interface:
    prefill(tokens: list[int], hidden: mx.array [T, D])        # target pre-norm hidden for tokens[:T]
    draft(root_token: int, root_hidden: mx.array [D] | None) -> Tree   (node 0 = root)
    accept(tokens: list[int], hidden: mx.array [len, D])       # committed path (root..last accepted)
    state() -> opaque / restore(state)                         # snapshot of the drafter context (Session prefix reuse)
"""
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from .accept_rules import RULES, walk
from .sample_kernel import sample_rows
from .tree_verify import Tree, TreeVerifier


class NgramDrafter:
    """Prompt-lookup drafter: proposes continuations seen after the same n-gram in the history.
    Builds a small tree (up to `branch` alternatives per depth, `depth` levels)."""

    def __init__(self, n=3, depth=8, branch=2, max_nodes=16):
        self.n, self.depth, self.branch, self.max_nodes = n, depth, branch, max_nodes
        self.hist = []

    def prefill(self, tokens, hidden=None):
        self.hist = list(tokens[:-1])          # the last prompt token is the root of the first draft and arrives with the first accept()

    def accept(self, tokens, hidden=None):
        self.hist.extend(tokens)

    def state(self):
        return len(self.hist)

    def restore(self, n):
        del self.hist[n:]

    def _continuations(self, ctx):
        """tokens that followed `ctx` (tuple) in the history, most recent first, deduplicated."""
        out, seen = [], set()
        h = self.hist
        L = len(ctx)
        for i in range(len(h) - L - 1, -1, -1):
            if tuple(h[i:i + L]) == ctx:
                t = h[i + L]
                if t not in seen:
                    seen.add(t); out.append(t)
                if len(out) >= self.branch:
                    break
        return out

    def draft(self, root_token, root_hidden=None):
        tokens, parent = [root_token], [-1]
        frontier = [(0, tuple(self.hist[-(self.n - 1):]) + (root_token,))]
        for _ in range(self.depth):
            nxt = []
            for node, ctx in frontier:
                for t in self._continuations(ctx):
                    if len(tokens) >= self.max_nodes:
                        break
                    tokens.append(t); parent.append(node)
                    nxt.append((len(tokens) - 1, (ctx + (t,))[-self.n:]))
            frontier = nxt
            if not frontier:
                break
        return Tree(tokens, parent)


def make_sampler(temp=0.0, top_k=0, top_p=1.0, min_p=0.0):
    """Batched sampler over [N, V] logits -> [N] tokens, matching mlx-lm's processing order
    (temperature, then top-k, top-p, min-p filtering). temp == 0 -> argmax."""
    cfg = dict(temp=temp, top_k=top_k, top_p=top_p, min_p=min_p)
    if temp == 0:
        f = lambda logits: logits.argmax(axis=-1)
        f.cfg = cfg
        return f
    if top_k > 64:
        raise ValueError("top_k must be <= 64 (the fused sampling kernel's candidate limit)")
    K = 64 if top_k == 0 else max(top_k, 32)   # top_k == 0: nucleus limited to the top-64
    f = lambda logits: sample_rows(logits, temp, top_k, top_p, min_p, K=K)
    f.cfg = dict(cfg, K=K)
    return f


def verify_tree(tree, sampled):
    """Deterministic-children tree verification (lossless and optimal for top-k-built trees):
    `sampled[n]` is a draw from the target distribution at node n (argmax when greedy).
    Follow the tree while the draw equals a child's token. Returns (path node indices, bonus token)."""
    am = sampled.tolist() if not isinstance(sampled, list) else sampled
    node, path = 0, [0]
    children = defaultdict(list)
    for j in range(1, tree.n):
        children[tree.parent[j]].append(j)
    while True:
        nxt = [j for j in children[node] if tree.tokens[j] == am[node]]
        if not nxt:
            break
        node = nxt[0]; path.append(node)
    return path, am[node]


class Acceptor:
    """Acceptance rule: `__call__(tree, logits, sampler) -> (path, bonus, forced)`.
    `forced` = number of accepted path tokens that differ from the target's own draw (0 when lossless).
    Rules (accept_rules.py) only act when sampling; at temperature 0 every rule is plain lossless verification."""

    def __init__(self, rule="lossless", **cfg):
        self.cfg = {**RULES[rule], **cfg, "rule": rule}

    def __call__(self, tree, logits, sampler):
        if sampler.cfg["temp"] == 0:
            path, bonus = verify_tree(tree, sampler(logits))
            return path, bonus, 0
        c = sampler.cfg
        y, ids, probs = sample_rows(logits, c["temp"], c["top_k"], c["top_p"], c["min_p"], K=c["K"], stats=True)   # draw + processed top-K distribution per node
        mx.eval(y, ids, probs)
        path, forced, bonus = walk(tree.tokens, tree.parent, np.array(y), np.array(ids), np.array(probs), self.cfg)
        return path, bonus, forced


class Session:
    """Decode state that persists across generate() calls: the target's caches plus the drafter's context, with
    snapshots at the end of every prompt and every generation. A later prompt that shares a prefix with the
    tokens in the cache is restored to the longest snapshot inside that prefix and only the rest is prefilled
    (multi-turn chat, regeneration, server clients that resend the conversation).

    Snapshots are cheap to take: the DeltaNet states are replaced (never mutated) by the model and the tree
    verifier, so a snapshot only holds references; the attention KV rows below an offset are never rewritten,
    so the offset suffices; the drafter copies its own (small) context window. Each snapshot pins ~150 MB of
    DeltaNet states for the 27B model, hence `keep`."""

    def __init__(self, model, drafter, keep=2):
        self.model, self.drafter, self.keep = model, drafter, keep
        self.reset()

    def reset(self):
        self.cache = None
        self.tokens = []           # tokens whose state the caches hold (prompt prefix + every committed path)
        self.snaps = []            # snapshots, increasing in n = len(tokens) at the time

    def _snap(self, last_hidden):
        lin = {li: (c[0], c[1]) for li, c in enumerate(self.cache) if isinstance(c, ArraysCache)}
        return dict(n=len(self.tokens), lin=lin, drafter=self.drafter.state(), last_hidden=last_hidden)

    def snapshot(self, last_hidden):
        """Record the current state (`last_hidden`: target pre-norm hidden of the last cached token)."""
        if not self.keep:
            return
        self.snaps = [s for s in self.snaps if s["n"] < len(self.tokens)] + [self._snap(last_hidden)]
        del self.snaps[:-self.keep]

    def _restore(self, s):
        for li, (conv, st) in s["lin"].items():
            self.cache[li][0] = conv; self.cache[li][1] = st
        for c in self.cache:
            if isinstance(c, KVCache):
                c.offset = s["n"]
        self.drafter.restore(s["drafter"])
        del self.tokens[s["n"]:]
        self.snaps = [x for x in self.snaps if x["n"] <= s["n"]]

    def prepare(self, prompt_tokens):
        """Bring the state to prompt_tokens[:-1]. Returns (verifier, last_hidden, number of reused tokens)."""
        prefix = prompt_tokens[:-1]
        tap = getattr(self.drafter, "tap", None)
        best = None
        if self.cache is not None:
            L = 0
            for a, b in zip(prefix, self.tokens):
                if a != b:
                    break
                L += 1
            best = max((s for s in self.snaps if s["n"] <= L), key=lambda s: s["n"], default=None)
        if best is None:
            self.reset()
            self.cache = make_prompt_cache(self.model)
            tv = TreeVerifier(self.model, self.cache, tap=tap)
            hidden = tv.prefill(prefix)
            self.drafter.prefill(prompt_tokens, tv.last_fused if tap else hidden)
            mx.eval(hidden)
            self.tokens = list(prefix)
            last_hidden, reused = hidden[-1], 0
        else:
            self._restore(best)
            tv = TreeVerifier(self.model, self.cache, tap=tap)
            delta = prefix[best["n"]:]
            last_hidden, reused = best["last_hidden"], best["n"]
            if delta:
                hidden = tv.prefill(delta)
                self.drafter.accept(delta, tv.last_fused if tap else hidden)
                mx.eval(hidden)
                self.tokens.extend(delta)
                last_hidden = hidden[-1]
        self.snapshot(last_hidden)
        return tv, last_hidden, reused


def generate(model, tok, prompt_tokens, drafter, max_tokens=200, verbose=False, sampler=None, on_tokens=None, acceptor=None, node_log=None, session=None, stop=None):
    """One generation. `session`: a Session to reuse cached prefixes across calls (default: fresh caches, no snapshots).
    `stop`: callable polled once per step; True ends the generation (stats["finish"] = "stop" | "length" | "interrupted")."""
    sampler = sampler or make_sampler()
    acceptor = acceptor or Acceptor()
    session = session or Session(model, drafter, keep=0)
    tap = getattr(drafter, "tap", None)
    root = prompt_tokens[-1]
    t0 = time.perf_counter()
    tv, root_hidden, reused = session.prepare(prompt_tokens)
    t_prefill = time.perf_counter() - t0
    out, stats = [], dict(steps=0, forced=0, finish="length", prompt_tokens=len(prompt_tokens), reused=reused)
    eos = set(tok.eos_token_ids)
    t_start = time.perf_counter()
    while len(out) < max_tokens:
        if stop is not None and stop():
            stats["finish"] = "interrupted"; break
        tree = drafter.draft(root, root_hidden)
        logits, h = tv.forward(tree)
        path, bonus, forced = acceptor(tree, logits, sampler)
        tv.commit(path)
        mx.eval(h)
        new = ([tree.tokens[i] for i in path[1:]] + [bonus])[:max_tokens - len(out)]     # the caches keep the whole accepted path
        if node_log is not None and hasattr(tree, "raw_values"):     # per drafted node: (raw value, depth, on accepted path)
            on = set(path)
            node_log.extend((tree.raw_values[i], tree.depth[i], i in on) for i in range(1, tree.n))
        pidx = mx.array(path)
        drafter.accept([tree.tokens[i] for i in path], tv.last_fused[pidx] if tap else h[pidx])
        session.tokens.extend(tree.tokens[i] for i in path)
        root_hidden = h[path[-1]]
        stats["steps"] += 1; stats["forced"] += forced
        if verbose:
            print(f"step {stats['steps']}: tree {tree.n} nodes depth {tree.max_depth} -> accepted {len(path)-1} (+1 bonus)  {tok.decode(new)!r}")
        emitted = []
        done = False
        for t in new:
            out.append(t); emitted.append(t)
            if t in eos:
                done = True; break
        if on_tokens is not None:
            on_tokens(emitted)
        if done:
            stats["finish"] = "stop"; break
        root = bonus
    session.snapshot(root_hidden)
    stats["t_total"] = time.perf_counter() - t_start
    stats["t_prefill"] = t_prefill
    stats["tokens"] = len(out)
    stats["tok_s"] = len(out) / max(1e-9, stats["t_total"])
    stats["tokens_per_step"] = len(out) / max(1, stats["steps"])
    return out, stats


def trim_by_cost(tokens, parents, values, cost, draft_cost, max_nodes, raw_values=None):
    """Hardware-aware tree budget: with c(n) the measured relative verify cost of n nodes (dict n -> cost, 1.0 at n=1)
    and value(node) the estimated acceptance probability (drafter path probability), keep the value-sorted prefix of
    the candidate nodes that maximizes (1 + sum value) / (c(n) + draft_cost), restricted to nodes whose parent is kept.
    Node 0 is the root. Returns the Tree (with `raw_values` per node when given, for calibration)."""
    ks = sorted(cost)

    def c(n):
        for k in ks:
            if k >= n:
                return cost[k]
        return cost[ks[-1]] * n / ks[-1]
    order = sorted(range(1, len(tokens)), key=lambda i: (-values[i], i))
    best_n, best, cum = 1, 1.0 / (c(1) + draft_cost), 0.0
    for j, i in enumerate(order[:max_nodes - 1]):
        cum += values[i]
        score = (1.0 + cum) / (c(j + 2) + draft_cost)
        if score > best:
            best_n, best = j + 2, score
    chosen = set(order[:best_n - 1])
    changed = True
    while changed:                                     # drop nodes whose parent was not chosen
        changed = False
        for i in list(chosen):
            if parents[i] > 0 and parents[i] not in chosen:
                chosen.discard(i); changed = True
    remap, t2, p2 = {0: 0}, [tokens[0]], [-1]
    for i in sorted(chosen):
        remap[i] = len(t2); t2.append(tokens[i]); p2.append(remap[parents[i]])
    tree = Tree(t2, p2)
    if raw_values is not None:
        tree.raw_values = [1.0] + [raw_values[i] for i in sorted(chosen)]
    return tree
