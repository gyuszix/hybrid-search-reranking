import bm25s
import numpy as np
import pandas as pd
import json
import os
import sys
from joblib import Parallel, delayed

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K, ROOT_DIR

def _score_single_group(query_id, group):
    """
    Helper function to score a single query's candidate set using C-Native bm25s.
    Designed to run in parallel across multiple CPU cores.
    """
    query_text = group["query_text"].iloc[0]
    item_texts = group["item_text"].tolist()
    item_ids = group["item_id"].tolist()
    
    # C-Native Tokenization
    corpus_tokens = bm25s.tokenize(item_texts)
    query_tokens = bm25s.tokenize([query_text])
    
    # Build a tiny, fast index for just this candidate set
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Retrieve
    doc_indices, raw_scores = retriever.retrieve(query_tokens, k=len(item_texts))
    
    top_k_indices = doc_indices[0]
    top_k_scores = raw_scores[0]
    
    # Min-Max Normalization
    min_score, max_score = top_k_scores.min(), top_k_scores.max()
    if max_score - min_score > 1e-8:
        norm_scores = (top_k_scores - min_score) / (max_score - min_score)
    else:
        norm_scores = np.zeros_like(top_k_scores)
        
    return [
        {"query_id": query_id, "item_id": item_ids[idx], "bm25_score": float(norm_scores[i])}
        for i, idx in enumerate(top_k_indices)
    ]

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
    """

    required_columns = {"query_id", "query_text", "item_id", "item_text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Group by query
    grouped = df.groupby("query_id")

    # Run the C-native scoring across all available CPU cores
    results_nested = Parallel(n_jobs=-1, batch_size=100)(
        delayed(_score_single_group)(query_id, group) for query_id, group in grouped
    )

    # Flatten the results
    results_flat = [item for sublist in results_nested for item in sublist]
    scores_df = pd.DataFrame(results_flat)

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
    # Fill any remaining NaNs with empty strings to prevent errors.
    corpus = df_products['item_text'].fillna("").tolist()

    # bm25s uses its own C-optimized tokenizer
    corpus_tokens = bm25s.tokenize(corpus)

    # Create and index the sparse matrix
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens)
    
    return bm25_index, item_ids

def search_bm25_global(bm25_index, item_ids, query_text, k=TOP_K):
    """Searches the global index for a single text query."""
    # Use bm25s tokenizer for the query, wrapped in a list
    query_tokens = bm25s.tokenize([query_text])

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