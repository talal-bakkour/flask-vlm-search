import numpy as np

from search.text_search import search_text
from search.clip_search import search_clip


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if scores.max() == scores.min():
        return np.ones_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def search_hybrid(
    query,
    top_k=20,
    candidate_k=200,
    text_weight=0.7,
    clip_weight=0.3,
):

    # -----------------------------
    # Text Search
    # -----------------------------
    text_results = search_text(query, candidate_k)

    # -----------------------------
    # CLIP Search
    # -----------------------------
    clip_results = search_clip(query, candidate_k)

    # -----------------------------
    # Normalize Scores
    # -----------------------------
    text_results["text_score"] = normalize(text_results["score"])

    clip_results["clip_score"] = normalize(
        clip_results["clip_score"]
    )

    # -----------------------------
    # Merge
    # -----------------------------
    merged = text_results.merge(
        clip_results[["filename", "clip_score"]],
        on="filename",
        how="outer",
    )

    merged["text_score"] = merged["text_score"].fillna(0)
    merged["clip_score"] = merged["clip_score"].fillna(0)

    merged["hybrid_score"] = (
        text_weight * merged["text_score"]
        + clip_weight * merged["clip_score"]
    )

    merged = merged.sort_values(
        "hybrid_score",
        ascending=False,
    )

    return merged.head(top_k)