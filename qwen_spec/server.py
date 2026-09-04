"""OpenAI-compatible chat server for the tree-speculative Qwen3.8-27B engine.

  qwen-spec-server                          # http://127.0.0.1:8080/v1/chat/completions
  curl -N localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

Request fields: messages, stream, max_tokens (or max_completion_tokens), temperature, top_p, top_k, min_p, seed,
thinking (also enable_thinking, chat_template_kwargs.enable_thinking), reasoning_effort (low | medium | xhigh), accept.
Missing fields take the server defaults (see --help). Thinking is returned as `reasoning_content` (DeepSeek / vLLM
convention) and the answer as `content`; usage reports the reused (cached) prompt tokens, tok/s and tokens per step.
Requests are decoded one at a time (in arrival order); the cache is kept between them, so a resent conversation only
prefills its new tokens.
"""
import argparse, json, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .accept_rules import parse_cfg
from .engine import EFFORTS, Engine, add_engine_args, add_sampling_args, engine_from_args

ENGINE: Engine = None
ARGS = None
COUNT = [0]


def parse_request(req):
    """OpenAI-style request body -> Engine.generate keyword arguments (ValueError on bad input)."""
    if not isinstance(req, dict):
        raise ValueError("the request body must be a JSON object")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
        raise ValueError('"messages" must be a non-empty list of {"role": ..., "content": ...}')
    tk = req.get("chat_template_kwargs") or {}
    thinking = req.get("thinking", req.get("enable_thinking", tk.get("enable_thinking", ARGS.think)))
    effort = req.get("reasoning_effort", tk.get("reasoning_effort", ARGS.effort))
    if effort not in EFFORTS:
        raise ValueError(f'"reasoning_effort" must be one of {", ".join(EFFORTS)}')
    max_tokens = req.get("max_completion_tokens", req.get("max_tokens", ARGS.max_tokens))
    try:
        kw = dict(max_tokens=int(max_tokens) if max_tokens is not None else ARGS.max_tokens,
                  temp=None if req.get("temperature", ARGS.temp) is None else float(req.get("temperature", ARGS.temp)),
                  top_p=None if req.get("top_p", ARGS.top_p) is None else float(req.get("top_p", ARGS.top_p)),
                  top_k=None if req.get("top_k", ARGS.top_k) is None else int(req.get("top_k", ARGS.top_k)),
                  min_p=float(req.get("min_p", ARGS.min_p)), seed=None if req.get("seed", ARGS.seed) is None else int(req.get("seed", ARGS.seed)),
                  thinking=bool(thinking), effort=effort, accept=str(req.get("accept", ARGS.accept)))
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid sampling field: {e}")
    if kw["top_k"] is not None and kw["top_k"] > 64:
        raise ValueError('"top_k" must be <= 64')
    parse_cfg(kw["accept"])
    return messages, kw


def usage(st):
    return {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["tokens"], "total_tokens": st["prompt_tokens"] + st["tokens"],
            "prompt_tokens_details": {"cached_tokens": st["reused"]}, "completion_tokens_details": {"reasoning_tokens": st["think_tokens"]},
            "tok_s": round(st["tok_s"], 2), "tokens_per_step": round(st["tokens_per_step"], 2)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _error(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error" if code == 400 else "server_error"}})

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": ENGINE.name, "object": "model", "owned_by": "local"}]})
        if self.path.startswith("/health"):
            return self._json(200, {"status": "ok", "model": ENGINE.name, "cached_tokens": len(ENGINE.session.tokens)})
        self._error(404, "not found")

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._error(404, "not found")
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages, kw = parse_request(req)
        except ValueError as e:                       # includes JSONDecodeError
            return self._error(400, str(e))
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": ENGINE.name}
        COUNT[0] += 1; rid = COUNT[0]
        t0 = time.perf_counter()
        try:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close"); self.end_headers()
                dead = [False]

                def send(obj):
                    if dead[0]:
                        return
                    try:
                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()); self.wfile.flush()
                    except OSError:                       # client went away: stop generating at the next step
                        dead[0] = True
                send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

                def on_text(seg, thinking):
                    send({**base, "choices": [{"index": 0, "delta": {"reasoning_content" if thinking else "content": seg}, "finish_reason": None}]})
                try:
                    content, reasoning, st = ENGINE.generate(messages, on_text=on_text, stop=lambda: dead[0], **kw)
                    send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if st["finish"] == "length" else "stop"}], "usage": usage(st)})
                except Exception as e:                # the headers are out: report in-stream
                    send({"error": {"message": str(e), "type": "server_error"}})
                    raise
                finally:
                    if not dead[0]:
                        try:
                            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                        except OSError:
                            pass
                    self.close_connection = True
            else:
                content, reasoning, st = ENGINE.generate(messages, **kw)
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": ENGINE.name,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "length" if st["finish"] == "length" else "stop"}], "usage": usage(st)})
        except Exception as e:
            print(f"[req {rid}] error: {e!r}", file=sys.stderr)
            if not req.get("stream"):
                try:
                    self._error(500, str(e))
                except OSError:
                    pass
            return
        mode = f"think/{kw['effort']}" if kw["thinking"] else "no-think"
        print(f"[req {rid}] {len(messages)} msgs, {mode}, T={st['sampling']['temp']}: prompt {st['prompt_tokens']} tok ({st['reused']} cached, prefill {st['t_prefill']:.2f}s), "
              f"{st['tokens']} tok ({st['think_tokens']} thinking) in {time.perf_counter()-t0:.1f}s = {st['tok_s']:.1f} tok/s, {st['tokens_per_step']:.2f} tok/step, {st['finish']}", file=sys.stderr)


def main():
    global ENGINE, ARGS
    ap = argparse.ArgumentParser(prog="qwen-spec-server", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s; 0.0.0.0 for the LAN)")
    ap.add_argument("--port", type=int, default=8080, help="port (default: %(default)s)")
    add_sampling_args(ap, cli=False)
    add_engine_args(ap)
    ARGS = ap.parse_args()
    if ARGS.greedy:
        ARGS.temp = 0.0
    ENGINE = engine_from_args(ARGS)
    print(f"serving {ENGINE.name} on http://{ARGS.host}:{ARGS.port}/v1/chat/completions  (thinking {'on' if ARGS.think else 'off'}, "
          f"effort {ARGS.effort}, max_tokens {ARGS.max_tokens}, accept {ARGS.accept})", file=sys.stderr)
    try:
        ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n[stopped]", file=sys.stderr)


if __name__ == "__main__":
    main()
