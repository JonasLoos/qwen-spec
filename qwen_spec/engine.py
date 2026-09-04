"""Engine: loaded target model + drafter + persistent decode session, one `generate(messages)` call per chat completion."""
import os, sys, time
import numpy as np
import mlx.core as mx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE = os.path.expanduser("~/.cache/qwen-spec")
MODELS = dict(
    target="lmstudio-community/Qwen3.8-27B-MLX-4bit",
    dflash="z-lab/Qwen3.8-27B-DFlash2",
    mtp="mlx-community/Qwen3.8-27B-MTP-bf16",
)
DEFAULT_ACCEPT = "ratio:theta=0.3"
DEFAULT_MAX_TOKENS = 32768      # Qwen's recommended output budget for thinking mode
SAMPLING = {True: dict(temp=1.0, top_p=0.95, top_k=20), False: dict(temp=0.7, top_p=0.8, top_k=20)}   # Qwen's recommended settings: thinking / no thinking
EFFORTS = ("low", "medium", "xhigh")
ACCEPT_HELP = ('acceptance rule for sampled decoding (temperature > 0; every rule is lossless at 0): "ratio:theta=0.3" (default: when the '
               'target\'s draw misses the tree, accept the drafted branch with the longest verified continuation if the moved probability '
               'mass per extra token is <= 0.3; +24%% tokens/step in thinking mode, task accuracy unchanged) | "ratio:theta=0.2" (conservative) | '
               '"cov:eps=1,tau=1" (accept only the target\'s own argmax) | "cov:eps=1" (always the best drafted child, fastest) | '
               'lossless (exact sampling). See accept_rules.py.')


def cache_key(name):
    """Directory name under CACHE for a model given as a path or a repo id."""
    p = os.path.expanduser(name)
    return os.path.basename(os.path.normpath(p)) if os.path.isdir(p) else name.strip("/").replace("/", "--")


def resolve(name):
    """Model directory for a local path, an LM Studio download of the same repo, or a Hugging Face repo id (downloaded on first use)."""
    p = os.path.expanduser(name)
    if os.path.isdir(p):
        return p
    lm = os.path.expanduser(f"~/.lmstudio/models/{name}")
    if os.path.isdir(lm):
        return lm
    from huggingface_hub import snapshot_download
    return snapshot_download(name)


class Engine:
    """All decoding runs on one worker thread (MLX ties lazily built state to the thread that built it, and the session
    keeps such state between calls), so `generate` may be called from any thread; calls are served one at a time."""

    def __init__(self, model=MODELS["target"], drafter="dflash", drafter_model=None, max_nodes=16, quiet=False, reuse_cache=True, recalibrate=False):
        from concurrent.futures import ThreadPoolExecutor
        from mlx_lm import load
        from .patch_model import patch_quantized_linears
        from .spec_decode import Session
        t0 = time.perf_counter()
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.model, self.tok = load(resolve(model))
        patch_quantized_linears(self.model)
        from .cost_curve import load_cost_curve
        curve = load_cost_curve(self.model, cache_key(model), CACHE, quiet=quiet, force=recalibrate)   # measured once per machine (~4 s), then cached
        dv = np.load(os.path.join(DATA, "draft_vocab.npy"))
        if drafter == "dflash":
            from .dflash_drafter import DFlashDrafter, load_calib
            name = drafter_model or MODELS["dflash"]
            self.drafter = DFlashDrafter(self.model, resolve(name), curve, quant_bits=4, draft_vocab=dv, branch=6, frontier=12, max_nodes=max_nodes,
                                         block_size=16, calib=load_calib(os.path.join(DATA, "calib.json")), cache_dir=os.path.join(CACHE, cache_key(name)))
        elif drafter == "mtp":
            from .mtp_drafter import MTPBeamDrafter
            self.drafter = MTPBeamDrafter(self.model, resolve(drafter_model or MODELS["mtp"]), curve, quant_bits=4, draft_vocab=dv, depth=6, beam=6, top_k=6,
                                          max_nodes=max_nodes, min_value=0.02, draft_cost_per_level=0.04)
            patch_quantized_linears(self.drafter.head)
        elif drafter == "ngram":
            from .spec_decode import NgramDrafter
            self.drafter = NgramDrafter(max_nodes=max_nodes)
        else:
            raise ValueError(drafter)
        self.session = Session(self.model, self.drafter, keep=2 if reuse_cache else 0)
        self._worker = ThreadPoolExecutor(max_workers=1)
        self.eos = set(self.tok.eos_token_ids)
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.name = os.path.basename(model.rstrip("/"))
        if not quiet:
            print(f"[loaded {self.name} + {drafter} drafter in {time.perf_counter()-t0:.1f}s, {mx.get_active_memory()/1e9:.1f} GB]", file=sys.stderr)

    def generate(self, messages, max_tokens=DEFAULT_MAX_TOKENS, temp=None, top_p=None, top_k=None, min_p=0.0, thinking=True, effort="xhigh",
                 seed=None, accept=DEFAULT_ACCEPT, on_text=None, stop=None, verbose=False):
        """Chat completion for OpenAI-style `messages` -> (content, reasoning, stats).
        temp / top_p / top_k default to the model's recommended values for the mode (SAMPLING); temp 0 = greedy.
        effort: chat-template reasoning effort ("low" = brief thinking, "medium" = no instruction, "xhigh" = careful; thinking only).
        on_text(segment, is_thinking) streams text as it is decoded (called on the worker thread); stop() is polled every step
        (True ends the generation). stats["finish"]: "stop" (end of turn) | "length" (max_tokens) | "interrupted"."""
        cancel = [False]                        # an exception in the waiting thread (e.g. KeyboardInterrupt) stops the worker at the next step
        fut = self._worker.submit(self._generate, messages, max_tokens, temp, top_p, top_k, min_p, thinking, effort, seed, accept, on_text,
                                  lambda: cancel[0] or (stop is not None and stop()), verbose)
        try:
            return fut.result()
        except BaseException:
            cancel[0] = True
            raise

    def _generate(self, messages, max_tokens, temp, top_p, top_k, min_p, thinking, effort, seed, accept, on_text, stop, verbose):
        from .accept_rules import parse_cfg
        from .spec_decode import Acceptor, generate, make_sampler
        s = SAMPLING[bool(thinking)]
        temp = s["temp"] if temp is None else temp
        top_p = s["top_p"] if top_p is None else top_p
        top_k = s["top_k"] if top_k is None else top_k
        if seed is not None:
            mx.random.seed(seed)
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, reasoning_effort=effort)
        sampler = make_sampler(temp, top_k, top_p, min_p)
        acceptor = Acceptor(**parse_cfg(accept))
        detok = self.tok.detokenizer
        detok.reset()
        parts = {True: [], False: []}
        st = dict(think=bool(thinking), started=False, think_tokens=0)    # the thinking prompt ends with "<think>\n": reasoning comes first

        def emit(seg, thinking_):
            if seg:
                parts[thinking_].append(seg)
                if on_text is not None:
                    on_text(seg, thinking_)

        def on_tokens(toks):
            for t in toks:
                if t in self.eos:
                    continue
                if st["think"] and t == self.think_end:
                    detok.finalize(); emit(detok.last_segment, True); detok.reset()
                    st["think"] = False
                    continue
                if st["think"]:
                    st["think_tokens"] += 1
                detok.add_token(t)
                seg = detok.last_segment
                if not st["think"] and not st["started"]:                # drop the "\n\n" the template puts after </think>
                    seg = seg.lstrip(); st["started"] = bool(seg)
                emit(seg, st["think"])
        out, stats = generate(self.model, self.tok, prompt, self.drafter, max_tokens=max_tokens if max_tokens > 0 else 10**9, sampler=sampler,
                              on_tokens=on_tokens, acceptor=acceptor, session=self.session, stop=stop, verbose=verbose)
        detok.finalize(); emit(detok.last_segment, st["think"])
        stats["think_tokens"] = st["think_tokens"]
        stats["sampling"] = dict(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
        return "".join(parts[False]).strip(), "".join(parts[True]).strip(), stats


def add_engine_args(ap):
    g = ap.add_argument_group("engine")
    g.add_argument("--model", default=MODELS["target"], metavar="ID|DIR", help="target model: Hugging Face repo id (downloaded on first use, LM Studio downloads are picked up) or a local directory (default: %(default)s)")
    g.add_argument("--drafter", default="dflash", choices=["dflash", "mtp", "ngram"], help="dflash = DFlash2 block drafter (default, fastest), mtp = the model's own MTP head, ngram = prompt lookup (no extra weights)")
    g.add_argument("--drafter-model", metavar="ID|DIR", help=f"drafter weights (default: {MODELS['dflash']} / {MODELS['mtp']})")
    g.add_argument("--max-nodes", type=int, default=16, metavar="N", help="draft-tree budget per step (default: %(default)s, at most 32; verifying up to 16 tokens costs the same as 1)")
    g.add_argument("--no-cache", action="store_true", help="do not keep the KV/state cache between turns and requests (always prefill the whole prompt)")
    g.add_argument("--recalibrate", action="store_true", help="measure the verification cost curve again (it sets the per-step tree budget; measured on the first start per machine, ~4 s, cached under ~/.cache/qwen-spec/)")
    return g


def engine_from_args(a, quiet=False):
    return Engine(a.model, a.drafter, a.drafter_model, a.max_nodes, quiet=quiet, reuse_cache=not a.no_cache, recalibrate=a.recalibrate)


def add_sampling_args(ap, cli=True):
    import argparse
    g = ap.add_argument_group("generation" if cli else "request defaults (a request's own fields override them)")
    g.add_argument("--think", action=argparse.BooleanOptionalAction, default=True, help="thinking mode (default: on, as in the model's chat template)")
    g.add_argument("--effort", choices=EFFORTS, default="xhigh", help="reasoning-effort instruction of the chat template: low = brief thinking, medium = no instruction, xhigh = careful (default, the template's own default)")
    g.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, metavar="N", help="output token limit (default: %(default)s = Qwen's recommended thinking budget; 0 = unlimited)")
    g.add_argument("-t", "--temp", type=float, metavar="T", help="temperature (default: 1.0 with thinking, 0.7 without = the model's recommendation; 0 = greedy)")
    g.add_argument("--top-p", type=float, metavar="P", help="nucleus sampling (default: 0.95 with thinking, 0.8 without)")
    g.add_argument("--top-k", type=int, metavar="K", help="top-k sampling (default: 20)")
    g.add_argument("--min-p", type=float, default=0.0, metavar="P", help="min-p sampling (default: off)")
    g.add_argument("--greedy", action="store_true", help="greedy decoding (same as --temp 0; Qwen advises against it in thinking mode)")
    g.add_argument("--seed", type=int, help="random seed for sampling")
    g.add_argument("--accept", default=DEFAULT_ACCEPT, metavar="RULE", help=ACCEPT_HELP)
    return g
