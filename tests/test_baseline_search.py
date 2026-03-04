import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCTS_PATH, TOP_K, ROOT_DIR
from retrieval.bm25 import load_bm25_index, search_bm25_global
from retrieval.two_tower import load_tt_index, search_tt_global
from reranking.model import DeepESCIReranker

# ==========================================
# Interactive Pipeline System
# ==========================================
class SearchPipeline:
    def __init__(self, products_path, weights_path, stats_path):
        print("1. Loading Product Catalog...")
        self.df_pr = pd.read_parquet(products_path)
        self.df_pr['product_id'] = self.df_pr['product_id'].astype(str)
        self.df_pr.set_index('product_id', inplace=True)
        
        # Optionally load ESCI-S enriched data here if you implemented it!
        # df_enriched = pd.read_parquet("esci-data/esci_s_products.parquet")
        # self.df_pr = pd.merge(self.df_pr, df_enriched, on='product_id', how='left')

        print("2. Loading Precomputed Retrievers (BM25 & Two-Tower)...")
        try:
            self.bm25_index, self.bm25_ids = load_bm25_index()
            self.tt_model, self.tt_index, self.tt_ids = load_tt_index()
        except FileNotFoundError:
            print("Indices not found! Please run `python scripts/build_indices.py` first.")
            exit(1)
        
        print("3. Loading Reranker Weights and Stats...")
        with open(stats_path, "r") as f:
            stats = json.load(f)
            self.mean = np.array(stats["mean"])
            self.std = np.array(stats["std"])
            
        # Initialize model with 9 base features
        self.model = DeepESCIReranker(input_dim=len(self.mean))
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.model.eval()
        
        self.stemmer = PorterStemmer()
        print("\nSearch Engine Ready!\n")

    def _mock_retrieve(self, query_text, k=TOP_K):
        # Pass the pre-built indexes into the search functions
        df_bm25 = search_bm25_global(self.bm25_index, self.bm25_ids, query_text, k=k)
        df_sem = search_tt_global(self.tt_model, self.tt_index, self.tt_ids, query_text, k=k)
        return df_bm25, df_sem
    
    def search(self, query_text, top_k_retrieve=TOP_K, final_k=25):
        # 1. Retrieval Stage
        df_bm25, df_sem = self._mock_retrieve(query_text, k=top_k_retrieve)
        
        if df_bm25.empty and df_sem.empty:
            return "No candidates retrieved."

        # Outer join to get all unique candidates
        candidates = pd.merge(df_bm25, df_sem, on='product_id', how='outer')
        candidates['bm25_score'] = candidates['bm25_score'].fillna(0.0)
        candidates['semantic_score'] = candidates['semantic_score'].fillna(-1.0)
        
        # 2. Add Product Metadata
        df = candidates.join(self.df_pr, on='product_id', how='inner').reset_index()
        df['query'] = query_text
        
        # 3. Feature Extraction (Exact same logic as training)
        q_words = set(self.stemmer.stem(str(w)) for w in query_text.lower().split())
        
        def calc_overlap(title):
            t_words = set(self.stemmer.stem(str(w)) for w in str(title).lower().split())
            if len(q_words) == 0: return 0.0
            return len(q_words.intersection(t_words)) / len(q_words)
            
        df['word_overlap'] = df['product_title'].apply(calc_overlap)
        df['query_length'] = len(query_text.split())
        df['title_length'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
        df['has_brand'] = df['product_brand'].notna().astype(float)
        df['bullet_count'] = df['product_bullet_point'].fillna("").astype(str).apply(lambda x: len(x.split('\n')) if x.strip() and x.strip() != 'None' else 0)
        
        # Frequency features: For a single interactive query, these are baseline 1
        df['log_product_freq'] = np.log1p(1)
        df['log_brand_freq'] = np.log1p(1)
        
        # Ensure column order matches training exactly
        feature_cols = [
            'bm25_score', 'semantic_score', 'word_overlap', 
            'query_length', 'title_length', 'has_brand', 
            'bullet_count', 'log_product_freq', 'log_brand_freq'
        ]
        
        # 4. Normalize and Predict
        features_raw = df[feature_cols].values
        features_normalized = (features_raw - self.mean) / self.std
        
        with torch.no_grad():
            x_tensor = torch.tensor(features_normalized, dtype=torch.float32)
            predictions = self.model(x_tensor).squeeze().numpy()
            
        df['rerank_score'] = predictions
        
        # 5. Sort and Return Top K
        df_sorted = df.sort_values(by='rerank_score', ascending=False).head(final_k)
        
        # Return just the clean, readable columns
        return df_sorted[['rerank_score', 'bm25_score', 'semantic_score', 'product_brand', 'product_title']]

# ==========================================
# 3. The Interactive Loop
# ==========================================
if __name__ == "__main__":
    # Initialize the engine once
    engine = SearchPipeline(
        products_path=f'{ROOT_DIR}/{PRODUCTS_PATH}',
        weights_path=f"{ROOT_DIR}/output/best_esci_reranker.pth",
        stats_path=f"{ROOT_DIR}/output/normalization_stats.json"
    )
    
    print("==================================================")
    print("  Welcome to the Hybrid Search Engine! ")
    print("  Type 'quit' or 'exit' to stop.")
    print("==================================================\n")
    
    while True:
        user_query = input("\nEnter search query: ").strip()
        if user_query.lower() in ['quit', 'exit']:
            print("Shutting down engine...")
            break
        if not user_query:
            continue
            
        results = engine.search(user_query)
        print(f"\n--- Top Results for '{user_query}' ---")
        
        if isinstance(results, str):
            print(results)
        else:
            # Print cleanly formatted Pandas dataframe
            pd.set_option('display.max_colwidth', 80)
            print(results.to_string(index=False))