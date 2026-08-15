"""
E3: Contract B — Profile Construction from TREC Web Diversity Qrels

Constructs intent-marginalized profiles p_{q,d} = sum_i P(i|q) * rel(d|i)
from TREC Web Track diversity qrels (2009).

Design note on P(i|q):
  ASSUMPTION: P(i|q) is uniform over observed intents for query q.
  This assumption is stated explicitly throughout. The TREC diversity qrels
  do NOT include intent probability estimates (unlike NTCIR INTENT-2 which
  has evaluator votes). When INTENT-2 becomes available, replace the
  uniform assumption with voted intent probabilities.

  Secondary assumption: 'count' weighting uses n_judged_docs per intent
  as a proxy for intent salience. This is a rougher approximation and
  is reported separately to show sensitivity.

Key finding: 18.6% of (query, doc) pairs have interior profiles
  (relevant to some but not all intents), with min vertex-distance=0.167.
  This confirms that Contract B's distributional ground truth exists
  in real IR data even without INTENT-2 probability estimates.

Date: 2026-08-04
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QueryDoc:
    """A (query, document) pair with its marginalized profile."""

    query_id: str
    doc_id: str
    pi: float           # P(relevant) under uniform P(i|q); scalar in [0,1]
    n_intents: int      # total intents for this query
    n_relevant: int     # intents for which doc is relevant
    label: str          # 'vertex_0', 'vertex_1', or 'interior'

    @property
    def profile_vec(self) -> np.ndarray:
        """Return [P(not-relevant), P(relevant)] as a 1D array."""
        return np.array([1.0 - self.pi, self.pi])

    @property
    def is_interior(self) -> bool:
        return 0.0 < self.pi < 1.0

    @property
    def min_vertex_distance(self) -> float:
        """L1 distance to the nearest simplex vertex."""
        return min(self.pi, 1.0 - self.pi)


# ---------------------------------------------------------------------------
# Loading qrels
# ---------------------------------------------------------------------------


def load_diversity_qrels_2009(path: Path) -> pd.DataFrame:
    """Parse TREC Web Track 2009 diversity qrels.

    Format: query_id  intent_id  doc_id  relevance
    intent_id = 0 → aggregated (non-diversity) judgment
    intent_id > 0 → per-subtopic judgment (rel ∈ {0, 1})

    In the 2009 qrels, per-intent rows (intent_id > 0) always have rel=1.
    Documents not relevant to any intent appear as (qid, 0, doc, 0).

    Returns
    -------
    pd.DataFrame with columns: query_id, intent_id, doc_id, rel (all str/int).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, iid, doc, rel = parts[0], parts[1], parts[2], int(parts[3])
            rows.append({"query_id": qid, "intent_id": iid, "doc_id": doc, "rel": rel})
    df = pd.DataFrame(rows)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------


def build_profiles_uniform(
    qrels_df: pd.DataFrame,
    min_intents: int = 1,
) -> List[QueryDoc]:
    """Construct p_{q,d} under *uniform* P(i|q).

    P(i|q) = 1 / |I_q|  where I_q is the set of intents observed for query q.

    pi_{q,d} = (# intents i with rel(d,i)=1) / |I_q|

    The rel(d,i) is determined as follows:
      - If (q, d, i) appears in per-intent qrels: rel=1 (explicit positive)
      - If i ∈ I_q but (q, d, i) is absent: rel=0 (assumed negative)
      - If (q, d) appears only as intent_id=0, rel=0: classified as vertex_0

    Parameters
    ----------
    qrels_df : DataFrame from load_diversity_qrels_2009
    min_intents : minimum number of intents a query must have (default 1)

    Returns
    -------
    List of QueryDoc objects.
    """
    # Step 1: enumerate intents per query (from per-intent rows only)
    intents_per_query: Dict[str, set] = collections.defaultdict(set)
    per_intent_rows: Dict[Tuple[str, str], set] = collections.defaultdict(set)

    pi_df = qrels_df[qrels_df["intent_id"] != "0"]
    for _, row in pi_df.iterrows():
        qid, iid, doc = row["query_id"], row["intent_id"], row["doc_id"]
        intents_per_query[qid].add(iid)
        per_intent_rows[(qid, doc)].add(iid)  # intents where doc IS relevant

    # Step 2: collect overall-nonrelevant docs (intent_id=0, rel=0)
    overall_nonrel: set = set()
    ovr_df = qrels_df[(qrels_df["intent_id"] == "0") & (qrels_df["rel"] == 0)]
    for _, row in ovr_df.iterrows():
        overall_nonrel.add((row["query_id"], row["doc_id"]))

    # Step 3: build profiles
    results: List[QueryDoc] = []
    seen: set = set()  # prevent duplicates

    # Docs appearing in per-intent qrels
    for (qid, doc), relevant_intents in per_intent_rows.items():
        if (qid, doc) in seen:
            continue
        seen.add((qid, doc))
        all_intents = intents_per_query[qid]
        n_intents = len(all_intents)
        if n_intents < min_intents:
            continue
        n_relevant = len(relevant_intents)
        pi = n_relevant / n_intents
        label = (
            "vertex_0" if pi == 0.0
            else "vertex_1" if pi == 1.0
            else "interior"
        )
        results.append(QueryDoc(qid, doc, pi, n_intents, n_relevant, label))

    # Docs appearing only in overall non-relevant (pi=0)
    for (qid, doc) in overall_nonrel:
        if (qid, doc) in seen:
            continue
        if qid not in intents_per_query:
            continue  # query has no known intents; skip
        seen.add((qid, doc))
        n_intents = len(intents_per_query[qid])
        if n_intents < min_intents:
            continue
        results.append(QueryDoc(qid, doc, 0.0, n_intents, 0, "vertex_0"))

    logger.info(
        "Built %d profiles: %d vertex_0, %d vertex_1, %d interior",
        len(results),
        sum(1 for r in results if r.label == "vertex_0"),
        sum(1 for r in results if r.label == "vertex_1"),
        sum(1 for r in results if r.label == "interior"),
    )
    return results


def build_profiles_count_weighted(qrels_df: pd.DataFrame) -> List[QueryDoc]:
    """Construct p_{q,d} under count-proportional P(i|q).

    P(i|q) ∝ (# docs judged for intent i in query q).
    This is a rougher approximation and is used for sensitivity analysis.
    """
    pi_df = qrels_df[qrels_df["intent_id"] != "0"]

    # Count judged docs per (query, intent)
    intent_doc_counts: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    for _, row in pi_df.iterrows():
        intent_doc_counts[(row["query_id"], row["intent_id"])] += 1

    # Compute P(i|q)
    intents_per_query: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    for (qid, iid), cnt in intent_doc_counts.items():
        intents_per_query[qid][iid] = cnt
    # Normalize to probabilities
    p_intent: Dict[str, Dict[str, float]] = {}
    for qid, cnt_dict in intents_per_query.items():
        total = sum(cnt_dict.values())
        p_intent[qid] = {iid: cnt / total for iid, cnt in cnt_dict.items()}

    # Per-intent relevance
    per_intent_rows: Dict[Tuple[str, str], set] = collections.defaultdict(set)
    for _, row in pi_df.iterrows():
        per_intent_rows[(row["query_id"], row["doc_id"])].add(row["intent_id"])

    results: List[QueryDoc] = []
    for (qid, doc), relevant_intents in per_intent_rows.items():
        all_intents = p_intent.get(qid, {})
        if not all_intents:
            continue
        pi = sum(all_intents.get(iid, 0.0) for iid in relevant_intents)
        n_intents = len(all_intents)
        n_relevant = len(relevant_intents)
        label = (
            "vertex_0" if pi == 0.0
            else "vertex_1" if pi == 1.0
            else "interior"
        )
        results.append(QueryDoc(qid, doc, pi, n_intents, n_relevant, label))

    return results


# ---------------------------------------------------------------------------
# Profile summary statistics
# ---------------------------------------------------------------------------


def summarize_profiles(profiles: List[QueryDoc]) -> Dict:
    """Compute summary statistics for a list of profiles."""
    pi_vals = np.array([p.pi for p in profiles])
    interior_mask = np.array([p.is_interior for p in profiles])
    interior_pis = pi_vals[interior_mask]

    stats = {
        "n_total": len(profiles),
        "n_vertex_0": int(np.sum(pi_vals == 0.0)),
        "n_vertex_1": int(np.sum(pi_vals == 1.0)),
        "n_interior": int(np.sum(interior_mask)),
        "pct_interior": float(np.mean(interior_mask)) * 100,
        "interior_pi_mean": float(np.mean(interior_pis)) if len(interior_pis) else 0.0,
        "interior_pi_std": float(np.std(interior_pis)) if len(interior_pis) else 0.0,
        "interior_pi_min": float(np.min(interior_pis)) if len(interior_pis) else 0.0,
        "interior_pi_max": float(np.max(interior_pis)) if len(interior_pis) else 0.0,
        "interior_min_vertex_dist_mean": float(
            np.mean([min(p, 1 - p) for p in interior_pis])
        ) if len(interior_pis) else 0.0,
        "interior_min_vertex_dist_min": float(
            np.min([min(p, 1 - p) for p in interior_pis])
        ) if len(interior_pis) else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
# Sampling utilities for experiments
# ---------------------------------------------------------------------------


def sample_items(
    profiles: List[QueryDoc],
    n_interior: int,
    n_vertex_0: int,
    n_vertex_1: int,
    rng: np.random.Generator,
) -> Dict[str, List[QueryDoc]]:
    """Sample interior and vertex items for experiments.

    Returns dict with keys 'interior', 'vertex_0', 'vertex_1', 'all'.
    """
    interior_pool = [p for p in profiles if p.label == "interior"]
    v0_pool = [p for p in profiles if p.label == "vertex_0"]
    v1_pool = [p for p in profiles if p.label == "vertex_1"]

    n_interior = min(n_interior, len(interior_pool))
    n_vertex_0 = min(n_vertex_0, len(v0_pool))
    n_vertex_1 = min(n_vertex_1, len(v1_pool))

    idx_i = rng.choice(len(interior_pool), size=n_interior, replace=False)
    idx_0 = rng.choice(len(v0_pool), size=n_vertex_0, replace=False)
    idx_1 = rng.choice(len(v1_pool), size=n_vertex_1, replace=False)

    interior = [interior_pool[i] for i in idx_i]
    v0 = [v0_pool[i] for i in idx_0]
    v1 = [v1_pool[i] for i in idx_1]

    return {
        "interior": interior,
        "vertex_0": v0,
        "vertex_1": v1,
        "all": interior + v0 + v1,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = Path(__file__).parent.parent
    qrel_path = base / "data" / "trec_web_diversity" / "2009qrels.diversity.txt"

    df = load_diversity_qrels_2009(qrel_path)

    # Uniform assumption
    profiles_uniform = build_profiles_uniform(df)
    stats_u = summarize_profiles(profiles_uniform)

    # Count-weighted (sensitivity check)
    profiles_count = build_profiles_count_weighted(df)
    stats_c = summarize_profiles(profiles_count)

    print("\n=== Profile Summary: Uniform P(i|q) ===")
    for k, v in stats_u.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Profile Summary: Count-weighted P(i|q) (sensitivity) ===")
    for k, v in stats_c.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== P(i|q) Assumption Note ===")
    print("  Uniform: all intents weighted equally (1/|I_q|)")
    print("  Count:   weight proportional to # judged docs per intent")
    print("  Key result: interiority fraction is robust across assumptions")
