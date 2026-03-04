import pandas as pd
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import ndcg_at_k, recall_at_k
from config import EXAMPLES_PATH, ROOT_DIR

def evaluate_predictions(scores_csv_path, df_truth, score_col, k_ndcg=10, k_recall=150):
    try:
        df_scores = pd.read_csv(scores_csv_path)
    except FileNotFoundError:
        print(f"File not found: {scores_csv_path}. Run your scoring script first.")
        return 0.0, 0.0

    # Ensure IDs are strings to prevent merge errors
    df_scores['query_id'] = df_scores['query_id'].astype(str)
    df_scores['item_id'] = df_scores['item_id'].astype(str)

    # LEFT JOIN: We keep everything the model retrieved. 
    # If the model retrieved an item Amazon didn't label, it gets a NaN.
    df_merged = pd.merge(df_scores, df_truth, on=['query_id', 'item_id'], how='left')

    # Fill unjudged retrieved items with 0.0 relevance (Hard Negatives)
    df_merged['relevance'] = df_merged['relevance'].fillna(0.0)

    # metrics.py now handles the sorting and grouping internally based on the score_col
    final_ndcg = ndcg_at_k(df_merged, score_col=score_col, k=k_ndcg)

    # We pass df_truth into recall so it knows exactly how many items were missed
    final_recall = recall_at_k(df_merged, df_truth, score_col=score_col, k=k_recall)

    return final_ndcg, final_recall


def main():
    print("Loading Ground Truth labels for the TEST split...")
    df_ex = pd.read_parquet(f'{ROOT_DIR}/{EXAMPLES_PATH}')
    
    # Isolate ONLY the test data to prevent leakage
    df_test_truth = df_ex[df_ex['split'] == 'test'].copy()
    
    # Rename columns to match your baseline outputs
    df_test_truth['query_id'] = df_test_truth['query_id'].astype(str)
    df_test_truth['item_id'] = df_test_truth['product_id'].astype(str)
    
    # Map the ESCI letters to standard KDD Cup NDCG weights
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df_test_truth['relevance'] = df_test_truth['esci_label'].map(label_map).fillna(0.0)
    
    # We only need the IDs and the mapped relevance score
    df_test_truth = df_test_truth[['query_id', 'item_id', 'relevance']].drop_duplicates()

    print("\n" + "="*50)
    print("BASELINE EVALUATION (NDCG@10 & Recall@150)")
    print("="*50)

    bm_file_path = os.path.join(ROOT_DIR, "output", "bm25_scores_test.csv")
    tt_file_path = os.path.join(ROOT_DIR, "output", "two_tower_scores_test.csv")

    # 1. Evaluate BM25
    bm25_ndcg, bm25_recall = evaluate_predictions(
        bm_file_path, 
        df_test_truth, 
        "bm25_score"
    )
    print(f"BM25       | NDCG@10: {bm25_ndcg:.4f} | Recall@150: {bm25_recall:.4f}")

    # 2. Evaluate Two-Tower
    tt_ndcg, tt_recall = evaluate_predictions(
        tt_file_path, 
        df_test_truth, 
        "two_tower_score"
    )
    print(f"Two-Tower  | NDCG@10: {tt_ndcg:.4f} | Recall@150: {tt_recall:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()