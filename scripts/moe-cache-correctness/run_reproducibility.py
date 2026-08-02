#!/usr/bin/env python3
"""Does this configuration reproduce its own output?

The question every A/B in this tree depends on and none of the other
scripts here asked. `run_seq.sh` compares condition A against condition B;
if neither reproduces itself, that comparison means nothing. For a month it
did not, and nobody noticed because greedy argmax absorbs large logit
perturbations -- token sequences agreed for dozens of steps while the
numbers underneath differed by nats. So this compares LOGPROBS, not only
token ids.

Launches a fresh server per run, sends one fixed greedy request with
`cache_prompt=false`, and reports, across runs: token identity, logprob
identity, and the largest absolute logprob delta seen. Within one step,
differences of log-softmax values are exactly differences of logits.

Usage:
  run_reproducibility.py --bin-dir DIR --model GGUF [options] -- [server args...]

Options:
  --runs N        number of fresh servers to launch (default 3)
  --predict N     tokens to generate per run (default 8)
  --port N        scratch port (default 18157)
  --env K=V       extra environment for the server (repeatable)

Exit status is 0 only if every run agreed with the first on both tokens and
logprobs.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

PROMPT = ("Explain, in plain terms, how a mixture-of-experts transformer "
          "decides which experts to activate for a given token.")


def port_free(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_port_free(port, seconds=180):
    for _ in range(seconds):
        if port_free(port):
            return True
        time.sleep(1)
    return False


def request(port, n_predict, timeout):
    body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "seed": 42, "cache_prompt": False, "id_slot": 0,
            "return_tokens": True, "n_probs": 16, "post_sampling_probs": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def steps_of(resp):
    out = []
    for p in resp.get("completion_probabilities") or []:
        out.append({t["id"]: t["logprob"] for t in (p.get("top_logprobs") or [])})
    return out


def one_run(args, server_args, env):
    if not wait_port_free(args.port):
        raise SystemExit(f"port {args.port} never became free")
    argv = [os.path.join(args.bin_dir, "llama-server"), "-m", args.model,
            "--port", str(args.port), "--no-webui", "--metrics",
            "--parallel", "1"] + server_args
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    try:
        for _ in range(1200):
            if proc.poll() is not None:
                raise SystemExit(f"server exited rc={proc.returncode}")
            try:
                url = f"http://127.0.0.1:{args.port}/health"
                if json.loads(urllib.request.urlopen(url, timeout=2).read()
                              ).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        r = request(args.port, args.predict, 1800)
        return r.get("tokens"), steps_of(r)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=180)
        except Exception:
            proc.kill()
            proc.wait(timeout=60)
        wait_port_free(args.port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--predict", type=int, default=8)
    ap.add_argument("--port", type=int, default=18157)
    ap.add_argument("--env", action="append", default=[])
    args, server_args = ap.parse_known_args()
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = args.bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    results = [one_run(args, server_args, env) for _ in range(args.runs)]
    (bt, bs) = results[0]
    tok_ok = lp_ok = 1
    worst = 0.0
    for t, s in results[1:]:
        if t == bt:
            tok_ok += 1
        w = 0.0
        for i in range(min(len(bs), len(s))):
            for tid, lp in s[i].items():
                if tid in bs[i]:
                    w = max(w, abs(bs[i][tid] - lp))
        if w == 0.0:
            lp_ok += 1
        worst = max(worst, w)

    n = len(results)
    print(f"runs={n}  token-identical={tok_ok}/{n}  logprob-identical={lp_ok}/{n}  "
          f"max|dlogprob|={worst:.8f}")
    for i, (t, _) in enumerate(results):
        print(f"  run {i}: {t}")
    ok = (tok_ok == n and lp_ok == n)
    print("PASS" if ok else "FAIL -- this configuration does not reproduce itself")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
