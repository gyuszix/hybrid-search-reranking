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