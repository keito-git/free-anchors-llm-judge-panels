"""E27: recover the labels E26 lost to the output-token cap.

Two of the nine judges emit reasoning before answering.  E26 capped the reply
at 64 tokens, which truncated that reasoning before the digit appeared, so the
label was recorded as a parse failure rather than being read wrongly.  The
failures are therefore missing data, not noise, and they are concentrated in
two judges, which would bias the panel if left as is.

This pass retries only the items a judge failed to label, with a cap large
enough for the reasoning to finish, and merges the recovered labels into the
panel file.  Items still unlabelled after the retry are reported and left out.

No GPU.  API calls are made (approved for this experiment).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e26_run_second_panel as e26

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "second_panel_treccovid" / "panel_labels.json"
SAMPLE = ROOT / "data" / "second_panel_treccovid" / "sample.json"
MAX_TOKENS = 700


def call_long(provider: str, model: str, prompt: str, keys: dict):
    """Same as e26.call but with room for the reasoning to finish."""
    if provider == "anthropic":
        d = e26.post("https://api.anthropic.com/v1/messages",
                     {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"x-api-key": keys["ANTHROPIC_API_KEY"],
                      "anthropic-version": "2023-06-01"})
        u = d.get("usage", {})
        txt = " ".join(b.get("text", "") for b in d.get("content", [])
                       if isinstance(b, dict))
        return txt, u.get("input_tokens", 0), u.get("output_tokens", 0)
    if provider == "openai":
        d = e26.post("https://api.openai.com/v1/chat/completions",
                     {"model": model, "max_completion_tokens": MAX_TOKENS,
                      "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"Authorization": "Bearer " + keys["OPENAI_API_KEY"]})
    else:
        d = e26.post("https://openrouter.ai/api/v1/chat/completions",
                     {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"Authorization": "Bearer " + keys["OPENROUTER_API_KEY"]})
    u = d.get("usage", {})
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
    txt = msg.get("content") or msg.get("reasoning") or ""
    if isinstance(txt, list):
        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
    return txt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def main() -> None:
    keys = e26.load_keys()
    panel = json.load(open(PANEL))
    items = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}

    todo = []
    for jid, labs in panel["labels"].items():
        missing = [k for k in items if k not in labs]
        if missing:
            todo.append((jid, missing))
    print("retry queue:", {j: len(m) for j, m in todo}, flush=True)

    def run(entry):
        jid, missing = entry
        fam, provider, model = e26.JUDGES[jid]
        got, pt, ct, still = {}, 0, 0, []
        for k in missing:
            it = items[k]
            prompt = e26.PROMPT.format(query=it["query"], passage=it["passage"])
            raw = None
            for attempt in range(3):
                try:
                    raw, a, b = call_long(provider, model, prompt, keys)
                    pt += a
                    ct += b
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
            v = e26.parse(raw) if raw else None
            if v is None:
                still.append(k)
            else:
                got[k] = v
        print(f"[{jid:18s}] recovered {len(got)}/{len(missing)}  "
              f"tokens={pt}+{ct}", flush=True)
        return jid, got, still, pt, ct

    recovered = {}
    with ThreadPoolExecutor(max_workers=max(1, len(todo))) as ex:
        for jid, got, still, pt, ct in ex.map(run, todo):
            panel["labels"][jid].update(got)
            u = panel["usage"][jid]
            u["prompt_tokens"] += pt
            u["completion_tokens"] += ct
            u["n_labelled"] = len(panel["labels"][jid])
            u["n_failed"] = len(still)
            recovered[jid] = {"recovered": len(got), "still_missing": len(still)}

    panel["retry"] = {"script": "e27_retry_missing_labels.py",
                      "max_tokens": MAX_TOKENS, "per_judge": recovered}
    sets = [set(v) for v in panel["labels"].values()]
    panel["complete_items"] = len(set.intersection(*sets)) if sets else 0
    json.dump(panel, open(PANEL, "w"), indent=1)
    print("\ncoverage after retry:")
    for j, u in panel["usage"].items():
        print(f"  {j:18s} {u['n_labelled']:3d}/300")
    print("items labelled by all judges:", panel["complete_items"])
    print("saved", PANEL)


if __name__ == "__main__":
    main()
