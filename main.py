import pandas as pd
from config import EXAMPLES_PATH, PRODUCTS_PATH, USE_SMALL_VERSION, USE_SPLIT
from signals.bm25 import compute_bm25_scores
from evaluation.metrics import ndcg_at_k


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
        df_examples = df_examples[df_examples["small_version"] == True]

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

    # Rename columns for BM25 module
    df = df.rename(columns={
        "query": "query_text",
        "product_id": "item_id"
    })

    print("Filtered shape:", df.shape)
    print("Query distribution:")
    print(df.groupby("query_id").size().describe())

    # ----------------------
    # 5. Compute BM25
    # ----------------------
    bm25_df = compute_bm25_scores(df)

    df = df.merge(
        bm25_df,
        on=["query_id", "item_id"]
    )

    print("BM25 score distribution:")
    print(df["bm25_score"].describe())

    # ----------------------
    # 6. Convert ESCI label to numeric relevance
    # ----------------------
    label_map = {
        "E": 3,
        "S": 2,
        "C": 1,
        "I": 0
    }

    df["relevance"] = df["esci_label"].map(label_map)

    # ----------------------
    # 7. Sort by BM25 score
    # ----------------------
    df_sorted = df.sort_values(
        by=["query_id", "bm25_score"],
        ascending=[True, False]
    )

    # ----------------------
    # 8. Evaluate NDCG@10
    # ----------------------
    score = ndcg_at_k(df_sorted, k=10)

    print("BM25 NDCG@10:", score)


if __name__ == "__main__":
    main()
