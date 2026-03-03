import numpy as np

def dcg(scores, k=10):
    """Calculates Discounted Cumulative Gain."""
    scores = np.asarray(scores)[:k]
    if scores.size:
        # Standard DCG formula
        return np.sum(
            (2**scores - 1) / np.log2(np.arange(2, scores.size + 2))
        )
    return 0.0

def ndcg_at_k(df, score_col, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain.
    
    Parameters:
    - df: Pandas DataFrame containing query_id, relevance, and the predicted score.
    - score_col: The string name of the column containing the model's predictions.
    - k: The cutoff rank.
    """
    ndcgs = []

    # Ensure the necessary columns exist
    if 'relevance' not in df.columns or score_col not in df.columns:
        raise ValueError(f"DataFrame must contain 'relevance' and '{score_col}' columns.")

    for _, group in df.groupby("query_id"):
        # 1. Sort the group by the model's predicted score (descending)
        sorted_group = group.sort_values(by=score_col, ascending=False)
        
        # 2. Extract the actual relevance values in that ranked order
        rel = sorted_group["relevance"].values
        
        # 3. Calculate the ideal sort (perfect ranking)
        ideal_rel = sorted(rel, reverse=True)

        # 4. Compute DCG and IDCG
        dcg_val = dcg(rel, k)
        idcg_val = dcg(ideal_rel, k)

        if idcg_val > 0:
            ndcgs.append(dcg_val / idcg_val)

    # Return 0.0 if no valid queries were found to prevent returning NaN
    return np.mean(ndcgs) if ndcgs else 0.0

def recall_at_k(df_preds, df_truth, score_col, k=150, rel_threshold=0.0):
    """
    Calculates Recall@K.
    
    Parameters:
    - df_preds: DataFrame with retrieved items and predictions.
    - df_truth: DataFrame with the ground truth labels for ALL items.
    - score_col: The string name of the column containing the model's predictions.
    - k: The cutoff rank (Usually higher for retrieval evaluation, e.g., 100 or 150).
    - rel_threshold: Items strictly greater than this score are considered "Relevant".
    """
    if 'relevance' not in df_preds.columns or score_col not in df_preds.columns:
        raise ValueError(f"DataFrame must contain 'relevance' and '{score_col}' columns.")

    # 1. Calculate the TOTAL number of relevant items per query in the ground truth
    # With a threshold of 0.0, 'E' (1.0), 'S' (0.1), and 'C' (0.01) are considered relevant.
    truth_relevant = df_truth[df_truth['relevance'] > rel_threshold]
    total_rel_per_query = truth_relevant.groupby('query_id').size().to_dict()

    recalls = []
    
    for query_id, group in df_preds.groupby("query_id"):
        total_possible_hits = total_rel_per_query.get(query_id, 0)
        
        # If there are no relevant items to find for this query, skip it
        if total_possible_hits == 0:
            continue

        # 2. Sort the retrieved group by the model's score and isolate Top K
        top_k_retrieved = group.sort_values(by=score_col, ascending=False).head(k)
        
        # 3. Count how many of those Top K items were actually relevant
        hits = (top_k_retrieved['relevance'] > rel_threshold).sum()
        
        recalls.append(hits / total_possible_hits)

    return np.mean(recalls) if recalls else 0.0