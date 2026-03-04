# signals/bm25.py
import bm25s
import numpy as np
import pandas as pd
import json
from config import TOP_K, ROOT_DIR


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

    # bm25s uses its own C-optimized tokenizer
    corpus_tokens = bm25s.tokenize(corpus)

    # Create and index the sparse matrix
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens)
    
    return bm25_index, item_ids

def search_bm25_global(bm25_index, item_ids, query_text, k=TOP_K):
    """Searches the global index for a single text query."""
    # Tokenize the single query
    query_tokens = simple_tokenize(query_text)

    # Retrieve top K results directly via sparse matrix slicing
    # retrieve() returns the integer indices and the scores
    doc_indices, raw_scores = bm25_index.retrieve(query_tokens, k=k)
    
    # Extract the first (and only) query's results
    top_k_indices = doc_indices[0]
    top_k_scores = raw_scores[0]
    
    # Min-Max normalize
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

def save_bm25_index(bm25_index, item_ids, index_dir=f'{ROOT_DIR}/output/bm25s_index', ids_path=f'{ROOT_DIR}/output/bm25_ids.json'):
    # bm25s natively saves the matrix to a directory
    bm25_index.save(index_dir)
    with open(ids_path, "w") as f:
        json.dump(item_ids, f)

def load_bm25_index(index_dir=f"{ROOT_DIR}/output/bm25s_index", ids_path=f"{ROOT_DIR}/output/bm25_ids.json"):
    # Load the matrix without loading the original raw text corpus into memory
    bm25_index = bm25s.BM25.load(index_dir, load_corpus=False)
    with open(ids_path, "r") as f:
        item_ids = json.load(f)
    return bm25_index, item_ids