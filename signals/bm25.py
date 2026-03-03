# signals/bm25.py

import numpy as np
import pandas as pd
import pickle
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

def build_global_bm25_index(df_products):
    """Builds a BM25 index over the entire product catalog."""
    print("Building global BM25 index (this takes a moment)...")
    
    # Ensure text is combined
    if 'item_text' not in df_products.columns:
        df_products["item_text"] = (
            df_products["product_title"].fillna("") + " " +
            df_products["product_description"].fillna("") + " " +
            df_products["product_bullet_point"].fillna("")
        )
    
    item_ids = df_products['product_id'].tolist()
    corpus = df_products['item_text'].apply(simple_tokenize).tolist()
    bm25_index = BM25Okapi(corpus)
    
    return bm25_index, item_ids

def search_bm25_global(bm25_index, item_ids, query_text, k=TOP_K):
    """Searches the global index for a single text query."""
    query_tokens = simple_tokenize(query_text)
    raw_scores = bm25_index.get_scores(query_tokens)
    
    # Get top K indices using numpy's fast sorting
    top_k_indices = np.argsort(raw_scores)[::-1][:k]
    top_k_scores = raw_scores[top_k_indices]
    
    # Min-Max normalize the top K scores so they are between 0 and 1
    min_score, max_score = top_k_scores.min(), top_k_scores.max()
    if max_score - min_score > 1e-8:
        norm_scores = (top_k_scores - min_score) / (max_score - min_score)
    else:
        norm_scores = np.zeros_like(top_k_scores)
        
    results = [
        {"product_id": str(item_ids[idx]), "bm25_score": float(norm_scores[i])}
        for i, idx in enumerate(top_k_indices)
    ]
    return pd.DataFrame(results)

def save_bm25_index(bm25_index, item_ids, index_path="output/bm25_index.pkl", ids_path="output/bm25_ids.pkl"):
    """Saves the BM25 index and IDs to disk."""
    with open(index_path, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(ids_path, "wb") as f:
        pickle.dump(item_ids, f)

def load_bm25_index(index_path="output/bm25_index.pkl", ids_path="output/bm25_ids.pkl"):
    """Loads the BM25 index and IDs from disk."""
    with open(index_path, "rb") as f:
        bm25_index = pickle.load(f)
    with open(ids_path, "rb") as f:
        item_ids = pickle.load(f)
    return bm25_index, item_ids