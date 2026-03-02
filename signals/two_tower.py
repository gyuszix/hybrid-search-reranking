import torch
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from config import TOP_K

MODEL_NAME = "models/two_tower_finetuned"


def encode_texts(model, texts, batch_size=256):
    """
    Encode a list of texts into L2-normalized embeddings.
    Larger default batch_size to better utilize GPU.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,   # helpful for large encodes
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=None # SentenceTransformer auto-detects GPU
    )
    return embeddings.astype(np.float32)


def build_faiss_index(product_embeddings, use_gpu=False):
    """
    Build a FAISS index. Optionally move to GPU for faster search.
    """
    embedding_dim = product_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(product_embeddings)

    if use_gpu and faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    return index


def compute_two_tower_scores(df):
    """
    Optimized two-tower scoring:
    1. Encode all unique products ONCE (biggest speedup)
    2. Encode all unique queries ONCE
    3. Build a single FAISS index over all products
    4. Batch-search all queries at once
    5. Map results back to per-query item_ids
    """
    required_columns = {"query_id", "query_text", "item_id", "item_text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # --- Hardware Detection ---
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"Hardware detected: {device_name} (CUDA)")
    else:
        device = "cpu"
        print("Hardware detected: CPU (CUDA is unavailable)")

    print(f"Loading encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # --- Step 1: Encode all unique products ONCE ---
    unique_items = df[["item_id", "item_text"]].drop_duplicates(subset=["item_id", "item_text"])
    # If same item_id has multiple texts (e.g. different locales), keep first
    unique_items = unique_items.drop_duplicates(subset="item_id", keep="first")
    item_id_list = unique_items["item_id"].tolist()
    item_text_list = unique_items["item_text"].tolist()

    # Map item_id -> index in the embedding matrix
    item_id_to_idx = {iid: i for i, iid in enumerate(item_id_list)}

    # Replace empty/whitespace-only texts with a placeholder to avoid degenerate embeddings
    item_text_list = [t if t.strip() else "unknown product" for t in item_text_list]

    print(f"Encoding {len(item_text_list)} unique products...")
    product_embeddings = encode_texts(model, item_text_list, batch_size=256)
    # Replace any NaN/inf embeddings with zeros to avoid matmul warnings
    product_embeddings = np.nan_to_num(product_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 2: Encode all unique queries ONCE ---
    unique_queries = df[["query_id", "query_text"]].drop_duplicates(subset="query_id")
    query_id_list = unique_queries["query_id"].tolist()
    query_text_list = unique_queries["query_text"].tolist()

    print(f"Encoding {len(query_text_list)} unique queries...")
    query_embeddings = encode_texts(model, query_text_list, batch_size=256)
    query_embeddings = np.nan_to_num(query_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Map query_id -> query embedding
    query_id_to_emb = {
        qid: query_embeddings[i] for i, qid in enumerate(query_id_list)
    }

    # --- Step 3: Score per query using its candidate set ---
    # We still need per-query scoring because each query has its own candidate pool
    # But now we just LOOK UP precomputed embeddings instead of re-encoding

    results = []
    grouped = df.groupby("query_id")

    for query_id, group in grouped:
        q_emb = query_id_to_emb[query_id].reshape(1, -1)  # (1, dim)

        # Look up precomputed product embeddings for this query's candidates
        candidate_item_ids = group["item_id"].tolist()
        candidate_indices = [item_id_to_idx[iid] for iid in candidate_item_ids]
        candidate_embeddings = product_embeddings[candidate_indices]  # (n_candidates, dim)

        # Compute cosine similarities via dot product (embeddings are L2-normed)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            cosine_scores = (candidate_embeddings @ q_emb.T).flatten()  # (n_candidates,)
        cosine_scores = np.nan_to_num(cosine_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. NORMALIZE FIRST (across all candidates for this query)
        min_score = cosine_scores.min()
        max_score = cosine_scores.max()
        if max_score - min_score > 1e-8:
            norm_scores = (cosine_scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.zeros_like(cosine_scores)

        # 2. THEN GET TOP-K INDICES
        k = min(TOP_K, len(candidate_item_ids))
        if k < len(norm_scores):
            top_k_indices = np.argpartition(norm_scores, -k)[-k:]
            # Sort the top k partition in descending order
            top_k_indices = top_k_indices[np.argsort(norm_scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(norm_scores)[::-1]

        # 3. APPEND RESULTS
        for idx in top_k_indices:
            results.append({
                "query_id": query_id,
                "item_id": candidate_item_ids[idx],
                "two_tower_score": float(norm_scores[idx])
            })

    scores_df = pd.DataFrame(results)

    if scores_df.empty:
        return scores_df

    scores_df = (
        scores_df
        .sort_values(by=["query_id", "two_tower_score"], ascending=[True, False])
        .groupby("query_id", group_keys=False)
        .head(TOP_K)
    )

    return scores_df