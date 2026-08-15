"""E26: run a second judge panel over the E25 sample.

Eight judges drawn from six provider families label the same 300 items on the
same three-level scale, so the panel has the family structure the theory needs
and the collection is independent of the one used everywhere else in the paper.

The prompt is fixed across judges and states the TREC-COVID label definitions
verbatim, so differences between judges are differences between judges and not
between instructions.  Temperature is zero and the response is constrained to a
single digit; anything else is recorded as a parse failure rather than silently
coerced.

Usage
    python e26_run_second_panel.py --pilot        # 10 items x 2 judges
    python e26_run_second_panel.py                # full run

Costs are accumulated from the token counts the providers return and written
into the output, so the spend is auditable rather than estimated.

No GPU.  API calls are made (approved for this experiment).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ssl
import urllib.request

import certifi

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "second_panel_treccovid" / "sample.json"
OUT_DIR = ROOT / "data" / "second_panel_treccovid"
KEYS_FILE = os.environ.get("API_KEYS_ENV", "")  # optional dotenv path

# judge id -> (family, provider, model)
JUDGES = {
    "gpt-4o-mini":        ("GPT",     "openai",     "gpt-4o-mini"),
    "gpt-4.1-mini":       ("GPT",     "openai",     "gpt-4.1-mini"),
    "claude-haiku":       ("Claude",  "anthropic",  "claude-haiku-4-5-20251001"),
    "claude-opus":        ("Claude",  "anthropic",  "claude-opus-4-5-20251101"),
    "gemini-flash":       ("Gemini",  "openrouter", "google/gemini-3.6-flash"),
    "llama-70b":          ("Llama",   "openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "qwen-72b":           ("Qwen",    "openrouter", "qwen/qwen-2.5-72b-instruct"),
    "mistral-large":      ("Mistral", "openrouter", "mistralai/mistral-large"),
    "deepseek-v4-flash":  ("DeepSeek", "openrouter", "deepseek/deepseek-v4-flash"),
}

PROMPT = """You are assessing the relevance of a scientific abstract to a search query about COVID-19.

Use exactly these levels, which are the TREC-COVID definitions:
0 = not relevant: the abstract does not answer the query and is off topic.
1 = partially relevant: the abstract is on topic but does not fully answer the query.
2 = fully relevant: the abstract answers the query.

Query: {query}

Abstract:
{passage}

Reply with the single digit 0, 1 or 2 and nothing else."""


def load_keys() -> dict:
    """Keys come from the environment; a dotenv file may be given instead."""
    keys = {k: os.environ[k] for k in
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
            if k in os.environ}
    if KEYS_FILE:
        for line in open(KEYS_FILE):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                           "OPENROUTER_API_KEY") if k not in keys]
    if missing:
        raise SystemExit("missing credentials: " + ", ".join(missing))
    return keys


def post(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return json.loads(r.read().decode())


def call(provider: str, model: str, prompt: str, keys: dict):
    """Return (raw_text, prompt_tokens, completion_tokens)."""
    if provider == "anthropic":
        d = post("https://api.anthropic.com/v1/messages",
                 {"model": model, "max_tokens": 64, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": keys["ANTHROPIC_API_KEY"],
                  "anthropic-version": "2023-06-01"})
        u = d.get("usage", {})
        txt = " ".join(b.get("text", "") for b in d.get("content", [])
                       if isinstance(b, dict))
        return txt, u.get("input_tokens", 0), u.get("output_tokens", 0)

    if provider == "openai":
        d = post("https://api.openai.com/v1/chat/completions",
                 {"model": model, "max_completion_tokens": 64, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"Authorization": "Bearer " + keys["OPENAI_API_KEY"]})
    else:
        d = post("https://openrouter.ai/api/v1/chat/completions",
                 {"model": model, "max_tokens": 64, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"Authorization": "Bearer " + keys["OPENROUTER_API_KEY"]})
    u = d.get("usage", {})
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
    txt = msg.get("content") or msg.get("reasoning") or ""
    if isinstance(txt, list):
        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
    return txt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def parse(raw: str):
    """Last standalone 0/1/2 in the reply.

    Some judges emit reasoning before the answer, so taking the first digit
    anywhere would read a digit out of the reasoning.  We take the last
    standalone digit, and if the reply is a bare digit this is the same thing.
    """
    if not raw:
        return None
    hits = re.findall(r"(?<![0-9.])([012])(?![0-9.])", raw)
    return int(hits[-1]) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--judges", default="")
    args = ap.parse_args()

    keys = load_keys()
    sample = json.load(open(SAMPLE))
    items = sample["items"]
    judges = dict(JUDGES)
    if args.pilot:
        items = items[:10]
        judges = {k: JUDGES[k] for k in ("gpt-4o-mini", "claude-haiku")}
    if args.judges:
        want = set(args.judges.split(","))
        judges = {k: v for k, v in JUDGES.items() if k in want}

    out = {
        "env": {
            "script": "e26_run_second_panel.py",
            "pilot": args.pilot,
            "n_items": len(items),
            "judges": {k: {"family": v[0], "provider": v[1], "model": v[2]}
                       for k, v in judges.items()},
            "temperature": 0,
            "gpu_used": False,
        },
        "labels": {}, "failures": [], "usage": {},
    }

    t0 = time.time()

    def run_judge(entry):
        jid, (fam, provider, model) = entry
        labs, pt, ct, fails, fail_log = {}, 0, 0, 0, []
        for n, it in enumerate(items, 1):
            prompt = PROMPT.format(query=it["query"], passage=it["passage"])
            raw = None
            for attempt in range(4):
                try:
                    raw, a, b = call(provider, model, prompt, keys)
                    pt += a
                    ct += b
                    break
                except Exception as e:
                    if attempt == 3:
                        fail_log.append({"judge": jid, "item": it["item_id"],
                                         "error": str(e)[:180]})
                        fails += 1
                    else:
                        time.sleep(2 * (attempt + 1))
            v = parse(raw) if raw is not None else None
            if v is None:
                fails += 1
                fail_log.append({"judge": jid, "item": it["item_id"],
                                 "raw": (raw or "")[:60]})
            else:
                labs[it["item_id"]] = v
            if n % 100 == 0:
                print(f"  {jid}: {n}/{len(items)}", flush=True)
        print(f"[{jid:18s}] labelled={len(labs)}/{len(items)} failed={fails} "
              f"tokens={pt}+{ct}", flush=True)
        return jid, labs, {"prompt_tokens": pt, "completion_tokens": ct,
                           "n_labelled": len(labs), "n_failed": fails}, fail_log

    with ThreadPoolExecutor(max_workers=len(judges)) as ex:
        for jid, labs, usage, flog in ex.map(run_judge, list(judges.items())):
            out["labels"][jid] = labs
            out["usage"][jid] = usage
            out["failures"].extend(flog)

    out["env"]["elapsed_s"] = round(time.time() - t0, 1)
    out["env"]["api_calls"] = sum(u["n_labelled"] + u["n_failed"]
                                  for u in out["usage"].values())
    name = "pilot_labels.json" if args.pilot else "panel_labels.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_DIR / name, "w"), indent=1)
    print("saved", OUT_DIR / name)


if __name__ == "__main__":
    main()
