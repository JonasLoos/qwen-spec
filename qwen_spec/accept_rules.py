"""Acceptance rules for sampled decoding (temperature > 0), applied while walking the verified draft tree.

At a node the verifier holds the target's processed distribution P (temperature / top-k / top-p / min-p applied), a draw
y ~ P, and the drafted children C. Lossless: accept the child equal to y, else y is the bonus token. A relaxed rule may
accept a child although y is not drafted ("forced"); every such rule moves the non-drafted mass 1 - P(C) onto the
children, so its per-token distortion is TV(P, Q) = 1 - P(C) wherever it is eligible. All rules are lossless at
temperature 0.

Rule strings ("rule:key=value,..."; defaults in RULES):
  lossless                     exact sampling
  ratio:theta=0.3              gain per deviation: when y misses, force the child with the longest lossless continuation
                               below it (known within the step) iff (1 - P(C)) / (1 + continuation) <= theta;
                               tau: only children with probability >= tau * max P
  cov:eps=1,tau=1              force the most probable child iff 1 - P(C) <= eps and its probability is >= tau * max P
                               (tau=1: only the target's own argmax; tau=0: always the best drafted child)
"""
from collections import defaultdict

RULES = dict(lossless={}, ratio=dict(theta=0.3, tau=0.0), cov=dict(eps=1.0, tau=0.0))


def parse_cfg(s):
    """'ratio:theta=0.3,tau=0.1' -> dict(rule=..., theta=..., tau=...)"""
    rule, _, kv = s.partition(":")
    if rule not in RULES:
        raise ValueError(f"unknown acceptance rule {rule!r} (choose from {', '.join(RULES)})")
    cfg = dict(RULES[rule], rule=rule)
    for x in kv.split(","):
        if x:
            k, _, v = x.partition("=")
            if k not in RULES[rule]:
                raise ValueError(f"rule {rule!r} has no option {k!r} (options: {', '.join(RULES[rule]) or 'none'})")
            cfg[k] = float(v)
    return cfg


def walk(tokens, parent, sampled, top_ids, top_probs, cfg):
    """Accepted path for one step. tokens/parent: tree (node 0 = root); sampled[n]: draw at node n;
    top_ids/top_probs [N, K]: processed distribution at node n (sorted by probability, zeros outside the nucleus).
    Returns (path, number of forced tokens, bonus token)."""
    rule = cfg["rule"]
    children = defaultdict(list)
    for j in range(1, len(tokens)):
        children[int(parent[j])].append(j)

    def cont(c):                                   # length of the lossless continuation below child c
        k = 0
        while True:
            yy = int(sampled[c]); nx = [j for j in children[c] if int(tokens[j]) == yy]
            if not nx:
                return k
            c = nx[0]; k += 1
    node, path, forced = 0, [0], 0
    while True:
        ch = children[node]
        y = int(sampled[node])
        hit = [j for j in ch if int(tokens[j]) == y]
        if hit:
            node = hit[0]; path.append(node); continue
        if not ch or rule == "lossless":
            return path, forced, y
        ids, pr = top_ids[node], top_probs[node]
        lut = {int(t): float(p) for t, p in zip(ids, pr) if p > 0}
        pc = [lut.get(int(tokens[j]), 0.0) for j in ch]
        cov = sum(pc)
        cands = [i for i in range(len(ch)) if pc[i] > 0 and pc[i] >= cfg["tau"] * float(pr[0])]
        if not cands:
            return path, forced, y
        if rule == "ratio":                        # cost of the override: 1 - P(C); gain: tokens the branch unlocks
            gains = [1 + cont(ch[i]) for i in cands]
            j = max(range(len(cands)), key=lambda i: (gains[i], pc[cands[i]]))
            ok, best = (1.0 - cov) / gains[j] <= cfg["theta"], ch[cands[j]]
        else:                                      # cov
            ok, best = (1.0 - cov) <= cfg["eps"], ch[max(cands, key=lambda i: pc[i])]
        if not ok:
            return path, forced, y
        forced += 1; node = best; path.append(node)
