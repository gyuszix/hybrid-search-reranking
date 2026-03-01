import json
import pandas as pd
import os
from config import EXAMPLES_PATH, PRODUCTS_PATH, USE_SMALL_VERSION, USE_SPLIT
from signals.bm25 import compute_bm25_scores
from signals.two_tower import compute_two_tower_scores
from evaluation.metrics import ndcg_at_k


def evaluate_signal(df, score_col, k=10):
    score = ndcg_at_k(df, k=k, score_col=score_col)
    print(f"{score_col} NDCG@{k}: {score:.4f}")
    return score


def main():

    # ----------------------
    # 1. Load data
    # ----------------------
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)

    # ----------------------
    # 2. Filter
    # ----------------------
    if USE_SMALL_VERSION:
        df_examples = df_examples[df_examples["small_version"] == 1]

    # # comment these two out when running the actual testing or training
    # sample_queries = df_examples["query_id"].unique()[:5]
    # df_examples = df_examples[df_examples["query_id"].isin(sample_queries)]

    df_examples = df_examples[df_examples["split"] == USE_SPLIT]

    # ----------------------
    # 3. Merge
    # ----------------------
    df = df_examples.merge(
        df_products,
        on=["product_id", "product_locale"],
        how="left"
    )

    # ----------------------
    # 4. Create item_text
    # ----------------------
    df["item_text"] = (
        df["product_title"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["product_bullet_point"].fillna("")
    )

    df = df.rename(columns={
        "query": "query_text",
        "product_id": "item_id"
    })

    print("Filtered shape:", df.shape)
    print("Query distribution:")
    print(df.groupby("query_id").size().describe())

    # ----------------------
    # 5. Compute signals
    # ----------------------
    bm25_df = compute_bm25_scores(df)
    df = df.merge(bm25_df, on=["query_id", "item_id"], how="left")
    df["bm25_score"] = df["bm25_score"].fillna(0.0)

    two_tower_df = compute_two_tower_scores(df)
    df = df.merge(two_tower_df, on=["query_id", "item_id"], how="left")
    df["two_tower_score"] = df["two_tower_score"].fillna(0.0)

    # ----------------------
    # 6. Hybrid score
    # ----------------------
    BM25_WEIGHT = 0.4
    TWO_TOWER_WEIGHT = 0.6
    df["hybrid_score"] = (
        BM25_WEIGHT * df["bm25_score"] +
       TWO_TOWER_WEIGHT * df["two_tower_score"]
    )

    # ----------------------
    # 7. Convert ESCI label to numeric relevance
    # ----------------------
    label_map = {"E": 3, "S": 2, "C": 1, "I": 0}
    df["relevance"] = df["esci_label"].map(label_map)

    # ----------------------
    # 8. Evaluate
    # ----------------------

    results = {
        "bm25_score": evaluate_signal(df, "bm25_score"),
        "two_tower_score": evaluate_signal(df, "two_tower_score"),
        "hybrid_score": evaluate_signal(df, "hybrid_score"),
    }

    # evaluate_signal(df, "bm25_score")
    # evaluate_signal(df, "hybrid_score")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()