"""Command line: one answer, or an interactive chat."""
import argparse, os, signal, sys, time
import mlx.core as mx

from .engine import EFFORTS, add_engine_args, add_sampling_args, engine_from_args

DESCRIPTION = """Tree-speculative decoding for Qwen3.8-27B on Apple Silicon (MLX): 3-6x faster than plain decoding, same output
distribution (greedy: identical; sampled: exact with --accept lossless, a mild relaxed acceptance rule by default).

Defaults follow the model's own recommendations: thinking mode on, temperature 1.0 / top-p 0.95 / top-k 20
(0.7 / 0.8 / 20 without thinking), up to 32k output tokens. The KV/state cache is kept between turns, so a
follow-up only prefills the new tokens."""
EXAMPLES = """examples:
  qwen-spec                                    interactive chat
  qwen-spec "Why is the sky blue?"             one answer (thinking shown dimmed, then the reply)
  qwen-spec --no-think -n 200 "Say hi"         quick reply without thinking
  qwen-spec --effort low "Prove 2+2=4"         shorter thinking
  qwen-spec -f prompt.md --greedy              prompt from a file, greedy decoding
  cat notes.txt | qwen-spec "Summarize:"       stdin is appended to the prompt
  qwen-spec --drafter mtp -i                   chat with the MTP-head drafter

chat commands:  /think on|off  /effort low|medium|xhigh  /temp T  /system TEXT  /file PATH  /reset  /help  /quit
                ctrl-c stops the current reply, ctrl-d quits; end a line with \\ to continue it on the next line"""
DIM, RESET = "\x1b[2m", "\x1b[0m"


def read_stdin(timeout=0.2):
    """Text piped or redirected into stdin, "" otherwise. Does not block on an idle pipe (a prompt word "-" forces a blocking read)."""
    if sys.stdin.isatty():
        return ""
    import select, stat
    try:
        if stat.S_ISREG(os.fstat(0).st_mode) or select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read()
    except (OSError, ValueError):
        pass
    return ""


def main():
    ap = argparse.ArgumentParser(prog="qwen-spec", description=DESCRIPTION, epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="*", help="the prompt (words are joined; piped stdin is appended, \"-\" reads it explicitly). Without a prompt an interactive chat starts")
    g = ap.add_argument_group("input")
    g.add_argument("-f", "--file", metavar="FILE", help="read the prompt from FILE (appended to any prompt words)")
    g.add_argument("-i", "--chat", action="store_true", help="stay in interactive chat after the first prompt")
    g.add_argument("-s", "--system", metavar="TEXT", help="system prompt")
    add_sampling_args(ap)
    add_engine_args(ap)
    g = ap.add_argument_group("output")
    g.add_argument("-v", "--verbose", action="store_true", help="one line per decoding step (tree size, accepted tokens)")
    g.add_argument("-q", "--quiet", action="store_true", help="no load message and no statistics line")
    a = ap.parse_args()
    if a.greedy:
        a.temp = 0.0
    if a.top_k is not None and a.top_k > 64:
        ap.error("--top-k must be <= 64")

    prompt = " ".join(w for w in a.prompt if w != "-")
    if a.file:
        prompt = (prompt + "\n\n" if prompt else "") + open(a.file).read()
    piped = sys.stdin.read() if "-" in a.prompt else read_stdin()
    if piped.strip():
        prompt = (prompt + "\n\n" if prompt else "") + piped
    interactive = a.chat or not prompt
    if interactive and not sys.stdin.isatty():
        ap.error("no prompt given and stdin is not a terminal")

    eng = engine_from_args(a, quiet=a.quiet)
    tty = sys.stdout.isatty()
    history = [{"role": "system", "content": a.system}] if a.system else []

    def ask(user):
        history.append({"role": "user", "content": user})
        mode = [None, "\n"]                                     # [what is on screen: None | True (thinking) | False (content), last char written]

        def on_text(seg, thinking):
            if thinking != mode[0]:
                nl = "" if mode[1] == "\n" else "\n"
                if thinking:
                    sys.stdout.write(DIM if tty else "<think>\n")
                elif mode[0]:
                    sys.stdout.write(RESET + nl + "\n" if tty else nl + "</think>\n\n")
                mode[0] = thinking
            sys.stdout.write(seg); sys.stdout.flush()
            mode[1] = seg[-1]
        flag = [False]
        prev = signal.signal(signal.SIGINT, lambda *_: flag.__setitem__(0, True))
        t0 = time.perf_counter()
        try:
            content, reasoning, st = eng.generate(history, a.max_tokens, a.temp, a.top_p, a.top_k, a.min_p, a.think, a.effort, a.seed, a.accept,
                                                  on_text=on_text, stop=lambda: flag[0], verbose=a.verbose)
        finally:
            signal.signal(signal.SIGINT, prev)
        if mode[0]:
            sys.stdout.write(RESET if tty else ("" if mode[1] == "\n" else "\n") + "</think>")
        sys.stdout.write("\n"); sys.stdout.flush()
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning          # the template keeps earlier turns' reasoning -> the cache continues verbatim
        history.append(msg)
        if not a.quiet:
            note = {"length": " · stopped at the token limit", "interrupted": " · interrupted"}.get(st["finish"], "")
            think = f" ({st['think_tokens']} thinking)" if st["think_tokens"] else ""
            forced = f" · {st['forced']} forced" if st["forced"] else ""
            cached = f" ({st['reused']} cached)" if st["reused"] else ""
            print(f"{DIM if tty else ''}[{st['tokens']} tokens{think} · {st['tok_s']:.1f} tok/s · {st['tokens_per_step']:.2f} tok/step{forced} · "
                  f"prefill {st['t_prefill']:.2f}s{cached} · {time.perf_counter()-t0:.1f}s{note}]{RESET if tty else ''}", file=sys.stderr)

    def command(line):
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "help":
            print(EXAMPLES.split("chat commands:")[1].strip(), file=sys.stderr)
        elif cmd == "reset":
            del history[1 if history and history[0]["role"] == "system" else 0:]
            print("[history cleared]", file=sys.stderr)
        elif cmd == "think":
            a.think = arg.lower() not in ("off", "0", "false", "no") if arg else not a.think
            print(f"[thinking {'on' if a.think else 'off'}]", file=sys.stderr)
        elif cmd == "effort":
            if arg in EFFORTS:
                a.effort = arg; print(f"[reasoning effort {arg}]", file=sys.stderr)
            else:
                print(f"[usage: /effort {'|'.join(EFFORTS)}]", file=sys.stderr)
        elif cmd == "temp":
            try:
                a.temp = float(arg); print(f"[temperature {a.temp}]", file=sys.stderr)
            except ValueError:
                print("[usage: /temp T]", file=sys.stderr)
        elif cmd == "system":
            if history and history[0]["role"] == "system":
                del history[0]
            if arg:
                history.insert(0, {"role": "system", "content": arg})
            print(f"[system prompt {'set' if arg else 'removed'}]", file=sys.stderr)
        elif cmd == "file":
            try:
                ask(open(os.path.expanduser(arg)).read())
            except OSError as e:
                print(f"[{e}]", file=sys.stderr)
        else:
            print(f"[unknown command: {line}; /help lists the commands]", file=sys.stderr)
        return True

    if prompt:
        ask(prompt)
    if interactive:
        try:
            import readline  # noqa: F401  (line editing + history for input())
        except ImportError:
            pass
        print(f"{eng.name}, {a.drafter} drafter. /help for commands, ctrl-d to quit.", file=sys.stderr)
        while True:
            try:
                line = input("\n> ")
                while line.endswith("\\"):
                    line = line[:-1] + "\n" + input("  ")
            except EOFError:
                print(); break
            except KeyboardInterrupt:                     # ctrl-c at the prompt (often meant for a reply that just ended): keep the chat
                print("  (ctrl-d or /quit exits)", file=sys.stderr); continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not command(line):
                    break
            else:
                ask(line)
    mx.synchronize()


if __name__ == "__main__":
    main()
