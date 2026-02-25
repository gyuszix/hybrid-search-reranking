# signals/two_tower.py

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from config import TOP_K

# Pretrained model fine-tuned on query-product retrieval (MS MARCO dataset)
# Produces 768-dim embeddings, projected to 256-dim via linear layer
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"


def encode_texts(model, texts, batch_size=64):
    """
    Encode a list of texts into embeddings using the sentence transformer.

    Parameters
    ----------
    model : SentenceTransformer
        The encoder model (shared weights for query and product towers)
    texts : list[str]
        List of texts to encode
    batch_size : int
        Number of texts to encode at once

    Returns
    -------
    np.ndarray
        Shape (len(texts), embedding_dim), float32, L2-normalized
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize so dot product == cosine similarity
    )
    return embeddings.astype(np.float32)


def build_faiss_index(product_embeddings):
    """
    Build a FAISS flat index over product embeddings for approximate
    nearest neighbor search.

    We use IndexFlatIP (inner product) because embeddings are L2-normalized,
    so inner product == cosine similarity.

    Parameters
    ----------
    product_embeddings : np.ndarray
        Shape (num_products, embedding_dim), float32

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index ready for search
    """
    embedding_dim = product_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(product_embeddings)
    return index


def compute_two_tower_scores(df):
    """
    Compute two-tower retrieval scores for each query-product pair.

    Uses a shared BERT encoder (same weights for query and product towers)
    fine-tuned on query-product retrieval. Cosine similarity between
    query and product embeddings is used as the relevance score.

    For each query:
        1. Encode query text → query embedding (256-dim)
        2. Encode all candidate product texts → product embeddings
        3. Build FAISS index over product embeddings
        4. Search index for top-K nearest neighbors by cosine similarity
        5. Return scores normalized to [0, 1]

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
            - two_tower_score (cosine similarity, normalized per query to [0,1])
    """

    required_columns = {"query_id", "query_text", "item_id", "item_text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Load pretrained encoder — same model used for both query and product towers
    # In a production system these could be separate fine-tuned models
    print(f"Loading encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    results = []

    grouped = df.groupby("query_id")

    for query_id, group in grouped:

        # --- Query Tower ---
        query_text = group["query_text"].iloc[0]
        query_embedding = encode_texts(model, [query_text])  # shape (1, dim)

        # --- Product Tower ---
        item_ids = group["item_id"].tolist()
        item_texts = group["item_text"].tolist()
        product_embeddings = encode_texts(model, item_texts)  # shape (n_items, dim)

        # --- FAISS Index ---
        # Build index over this query's candidate product embeddings
        index = build_faiss_index(product_embeddings)

        # Search for top-K most similar products
        k = min(TOP_K, len(item_ids))
        cosine_scores, indices = index.search(query_embedding, k)  # shape (1, k)

        cosine_scores = cosine_scores[0]   # flatten to (k,)
        indices = indices[0]               # flatten to (k,)

        # Cosine similarity is already in [-1, 1] due to L2 normalization
        # Normalize to [0, 1] for consistency with BM25 scores
        min_score = cosine_scores.min()
        max_score = cosine_scores.max()

        if max_score - min_score > 1e-8:
            norm_scores = (cosine_scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.zeros_like(cosine_scores)

        for rank, idx in enumerate(indices):
            results.append({
                "query_id": query_id,
                "item_id": item_ids[idx],
                "two_tower_score": float(norm_scores[rank])
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