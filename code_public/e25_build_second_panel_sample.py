"""E25: build the item sample for a second, independently judged panel.

Every real-data result in the paper comes from one judging panel, and E17 only
re-partitions that same panel, so the single-panel limitation stands.  No
public collection supplies a second panel with the structure we need: the
LLMJudge 2025 challenge redistributes the same collection, TREC RAG 2024
publishes a single LLM assessor, and RewardBench is degenerate here because its
true label is constant by construction.

We therefore assemble one.  BEIR's TREC-COVID gives graded human relevance on
three levels with the document text available, so the collection is independent
of TREC DL 2023 and the gold labels needed to estimate family kernels are
present.  This script fixes the item sample; e26 runs the judges on it.

Sampling is stratified by gold label so that all three levels are represented,
and the draw is seeded, so the sample is fixed before any judge sees it.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# BEIR TREC-COVID download location; see README
SRC = Path(os.environ.get("TRECCOVID_DIR", ROOT / "data" / "beir_trec_covid"))
OUT_DIR = ROOT / "data" / "second_panel_treccovid"
OUT = OUT_DIR / "sample.json"

N_PER_LABEL = 100
SEED = 20260815
MAX_CHARS = 1400          # passage text passed to the judges


def main() -> None:
    qrels = pd.read_csv(SRC / "tc_qrels.tsv", sep="\t", dtype=str)
    qrels["score"] = qrels["score"].astype(int)
    qrels = qrels[qrels["score"].isin([0, 1, 2])]

    queries = pd.read_parquet(SRC / "tc_q.parquet").set_index("_id")
    corpus = pd.read_parquet(SRC / "tc_corpus.parquet").set_index("_id")

    keep = qrels[qrels["corpus-id"].isin(corpus.index)]
    rng_state = SEED
    parts = []
    for lab in (0, 1, 2):
        sub = keep[keep["score"] == lab]
        parts.append(sub.sample(n=N_PER_LABEL, random_state=rng_state + lab))
    sample = pd.concat(parts).sample(frac=1.0, random_state=SEED)

    items = []
    for _, row in sample.iterrows():
        q = queries.loc[row["query-id"]]
        d = corpus.loc[row["corpus-id"]]
        title = "" if pd.isna(d["title"]) else str(d["title"])
        text = "" if pd.isna(d["text"]) else str(d["text"])
        passage = (title + "\n" + text).strip()[:MAX_CHARS]
        items.append({
            "item_id": f"{row['query-id']}::{row['corpus-id']}",
            "query_id": row["query-id"],
            "doc_id": row["corpus-id"],
            "query": str(q["text"]),
            "passage": passage,
            "gold": int(row["score"]),
        })

    out = {
        "env": {
            "script": "e25_build_second_panel_sample.py",
            "collection": "BEIR TREC-COVID (independent of TREC DL 2023)",
            "seed": SEED,
            "n_per_label": N_PER_LABEL,
            "max_passage_chars": MAX_CHARS,
            "gpu_used": False, "api_calls": 0, "cost_usd": 0.0,
        },
        "label_scheme": {0: "not relevant", 1: "partially relevant",
                         2: "fully relevant"},
        "n_items": len(items),
        "n_queries": int(sample["query-id"].nunique()),
        "items": items,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"items={len(items)}  queries={out['n_queries']}")
    print(f"gold distribution: "
          f"{ {k: sum(1 for i in items if i['gold'] == k) for k in (0, 1, 2)} }")
    print(f"mean passage chars: "
          f"{sum(len(i['passage']) for i in items) / len(items):.0f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
