# signals/bm25.py

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from config import TOP_K


def simple_tokenize(text):
    """
    Basic tokenizer:
    - lowercase
    - whitespace split
    """
    if not isinstance(text, str):
        return []
    return text.lower().split()


def compute_bm25_scores(df):
    """
    Compute BM25 scores within each query candidate set.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - query_id
            - query_text
            - item_id
            - item_text

    Returns
    -------
    pd.DataFrame
        Columns:
            - query_id
            - item_id
            - bm25_score (normalized per query, in [0,1])
    """

    required_columns = {"query_id", "query_text", "item_id", "item_text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    results = []

    # Group by query
    grouped = df.groupby("query_id")

    for query_id, group in grouped:

        # Extract query text
        query_text = group["query_text"].iloc[0]
        query_tokens = simple_tokenize(query_text)

        # Tokenize candidate item texts
        corpus = group["item_text"].apply(simple_tokenize).tolist()

        if len(corpus) == 0:
            continue

        # Build BM25 index for this query's candidate set
        bm25 = BM25Okapi(corpus)

        # Compute raw BM25 scores
        raw_scores = np.array(bm25.get_scores(query_tokens))

        # Query-level min-max normalization
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score - min_score > 1e-8:
            norm_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            # If all scores are identical, set to zero
            norm_scores = np.zeros_like(raw_scores)

        # Store results
        for idx, (_, row) in enumerate(group.iterrows()):
            results.append({
                "query_id": query_id,
                "item_id": row["item_id"],
                "bm25_score": float(norm_scores[idx])
            })

    scores_df = pd.DataFrame(results)

    if scores_df.empty:
        return scores_df

    # Truncate to Top-K BM25 candidates per query for downstream reranking.
    scores_df = (
        scores_df
        .sort_values(by=["query_id", "bm25_score"], ascending=[True, False])
        .groupby("query_id", group_keys=False)
        .head(TOP_K)
    )

    return scores_df
