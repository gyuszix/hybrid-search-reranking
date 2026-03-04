import os
import sys
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCTS_PATH, ROOT_DIR
from retrieval.bm25 import build_global_bm25_index, save_bm25_index
from retrieval.two_tower import build_global_tt_index, save_tt_index

def main():
    os.makedirs(f'{ROOT_DIR}/output', exist_ok=True)
    print("Loading product catalog...")
    df_pr = pd.read_parquet(PRODUCTS_PATH)

    # Use this for testing instead
    # df_pr = pd.read_parquet(PRODUCTS_PATH).head(50000)

    df_pr['product_id'] = df_pr['product_id'].astype(str)

    print("\n--- Building BM25 Index ---")
    bm25_index, bm25_ids = build_global_bm25_index(df_pr)
    save_bm25_index(bm25_index, bm25_ids)
    print("BM25 Index saved!")

    print("\n--- Building Two-Tower Index ---")
    _, tt_index, tt_ids = build_global_tt_index(df_pr)
    save_tt_index(tt_index, tt_ids)
    print("Two-Tower Index saved!")

    print("\nAll indices successfully precomputed and saved to 'output/' folder.")

if __name__ == "__main__":
    main()