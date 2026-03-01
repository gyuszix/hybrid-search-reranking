import numpy as np


def dcg(scores, k=10):
    scores = np.asarray(scores)[:k]
    if scores.size:
        return np.sum(
            (2**scores - 1) / np.log2(np.arange(2, scores.size + 2))
        )
    return 0.0


def ndcg_at_k(df, k=10, score_col="bm25_score"):
    ndcgs = []

    for _, group in df.groupby("query_id"):
        group_sorted = group.sort_values(by=score_col, ascending=False)

        rel = group_sorted["relevance"].values
        ideal_rel = np.sort(rel)[::-1]

        dcg_val = dcg(rel, k)
        idcg_val = dcg(ideal_rel, k)

        if idcg_val > 0:
            ndcgs.append(dcg_val / idcg_val)

    return float(np.mean(ndcgs))
