import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import re
from collections import Counter
from nltk.stem import PorterStemmer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from evaluation.metrics import ndcg_at_k, recall_at_k, apply_business_ndcg_labels
from reranking.advanced_model import AdvancedDeepReranker

def extract_test_advanced_features(idf_map):
    """Extracts the exact same 17 features, strictly for the TEST queries."""
    print("Loading Test Data...")
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_pr['product_id'] = df_pr['product_id'].astype(str)

    # Load ESCI-S attributes
    esci_s_path = f'{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet'
    try:
        df_esci_s = pd.read_parquet(esci_s_path)
        
        # Standardize the ID column name
        if 'asin' in df_esci_s.columns:
            df_esci_s = df_esci_s.rename(columns={'asin': 'product_id'})
        elif 'item_id' in df_esci_s.columns:
            df_esci_s = df_esci_s.rename(columns={'item_id': 'product_id'})
        elif 'id' in df_esci_s.columns:
            df_esci_s = df_esci_s.rename(columns={'id': 'product_id'})
            
        df_esci_s['product_id'] = df_esci_s['product_id'].astype(str)
        df_pr = pd.merge(df_pr, df_esci_s[['product_id', 'price', 'stars', 'ratings', 'category']], on='product_id', how='left')
    except FileNotFoundError:
        print("ESCI-S data not found. Creating dummy columns.")
        for col in ['price', 'stars', 'ratings', 'category']:
            df_pr[col] = np.nan

    print("Isolating TEST split...")
    df_test_queries = df_ex[df_ex['split'] == 'test'][['query_id', 'query']].drop_duplicates()

    # Load TEST retrievals
    df_bm25 = pd.read_csv(f'{ROOT_DIR}/output/bm25_scores_test.csv')
    df_bm25.columns = ['query_id', 'product_id', 'bm25_score']
    df_sem = pd.read_csv(f'{ROOT_DIR}/output/two_tower_scores_test.csv')
    df_sem.columns = ['query_id', 'product_id', 'semantic_score']

    for df in [df_bm25, df_sem]:
        df['query_id'], df['product_id'] = df['query_id'].astype(str), df['product_id'].astype(str)

    candidates = pd.merge(df_bm25, df_sem, on=['query_id', 'product_id'], how='outer')
    df = pd.merge(candidates, df_test_queries, on='query_id', how='inner')
    df = pd.merge(df, df_pr, on='product_id', how='inner')

    # Load labels
    df_labels = df_ex[['query_id', 'product_id', 'esci_label']].drop_duplicates()
    df = pd.merge(df, df_labels, on=['query_id', 'product_id'], how='left')

    df['bm25_score'] = df['bm25_score'].fillna(0.0)
    df['semantic_score'] = df['semantic_score'].fillna(-1.0)
    
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df['target_score'] = df['esci_label'].map(label_map).fillna(0.0)

    print("Extracting Test Features...")
    
    # --- A. Query Intent Features ---
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    
    def parse_budget(q):
        match = re.search(r'under\s*\$?(\d+)', str(q).lower())
        return float(match.group(1)) if match else -1.0
    df['user_budget'] = df['query'].apply(parse_budget)
    df['cheap_intent'] = df['query'].str.lower().str.contains('cheap|affordable|budget').astype(float)

    # Use the TRAINING idf_map to maintain exact parity
    def get_idf_stats(q):
        words = str(q).lower().split()
        if not words: return 0.0, 0.0
        idfs = [idf_map.get(w, 10.0) for w in words]
        return np.mean(idfs), np.max(idfs)
        
    idf_stats = df['query'].apply(get_idf_stats)
    df['query_mean_idf'] = [x[0] for x in idf_stats]
    df['query_max_idf'] = [x[1] for x in idf_stats]

    # --- B. Item Authority & Intelligent Imputation ---
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    
    # Force categories to string to prevent unhashable array errors
    df['category'] = df['category'].fillna("Unknown").astype(str)

    # Price Imputation
    df['is_price_missing'] = df['price'].isna().astype(float)
    cat_median_price = df.groupby('category')['price'].transform('median')
    global_median_price = df['price'].median() if not df['price'].isna().all() else 25.0
    imputed_price = df['price'].fillna(cat_median_price).fillna(global_median_price).fillna(0.0)
    df['log_price'] = np.log1p(imputed_price)

    # Stars Imputation
    df['stars_clean'] = df['stars'].astype(str).str.extract(r'([\d\.]+)').astype(float)
    df['is_rating_missing'] = df['stars_clean'].isna().astype(float)
    cat_median_stars = df.groupby('category')['stars_clean'].transform('median')
    global_median_stars = df['stars_clean'].median() if not df['stars_clean'].isna().all() else 4.0
    df['stars_clean'] = df['stars_clean'].fillna(cat_median_stars).fillna(global_median_stars).fillna(0.0)

    # Reviews Imputation
    cat_median_ratings = df.groupby('category')['ratings'].transform('median')
    global_median_ratings = df['ratings'].median() if not df['ratings'].isna().all() else 0.0
    imputed_ratings = df['ratings'].fillna(cat_median_ratings).fillna(global_median_ratings).fillna(0.0)
    df['log_review_count'] = np.log1p(imputed_ratings)

    # --- C. Interaction & Match Features ---
    stemmer = PorterStemmer()
    unique_queries = df['query'].astype(str).unique()
    unique_titles = df['product_title'].astype(str).unique()
    query_stem_map = {q: set(stemmer.stem(w) for w in q.lower().split()) for q in unique_queries}
    title_stem_map = {t: set(stemmer.stem(w) for w in t.lower().split()) for t in unique_titles}

    def fast_overlap(q, t):
        q_set = query_stem_map.get(q, set())
        t_set = title_stem_map.get(t, set())
        if not q_set: return 0.0
        return len(q_set.intersection(t_set)) / len(q_set)
        
    df['word_overlap'] = [fast_overlap(str(q), str(t)) for q, t in zip(df['query'], df['product_title'])]
    df['is_over_budget'] = ((df['user_budget'] > 0) & (imputed_price > df['user_budget'])).astype(float)
    
    def check_brand(q, b):
        if pd.isna(b): return 0.0
        return 1.0 if str(b).lower() in str(q).lower() else 0.0
    df['brand_match'] = [check_brand(q, b) for q, b in zip(df['query'], df['product_brand'])]

    COLORS = {'red', 'black', 'blue', 'white', 'green', 'yellow', 'silver', 'gold', 'grey', 'gray', 'pink', 'purple', 'brown'}
    def check_color(q, t):
        q_colors = set(str(q).lower().split()).intersection(COLORS)
        if not q_colors: return 0.0
        t_words = set(str(t).lower().split())
        return 1.0 if q_colors.intersection(t_words) else 0.0
    df['color_match'] = [check_color(q, t) for q, t in zip(df['query'], df['product_title'])]

    # Dominant Category calculation (Strictly from Top 20 BM25 results)
    top_20_bm25 = df.sort_values(['query_id', 'bm25_score'], ascending=[True, False]).groupby('query_id').head(20)
    dominant_cats = top_20_bm25.groupby('query_id')['category'].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    df['query_dominant_category'] = df['query_id'].map(dominant_cats)
    df['is_dominant_category'] = (df['category'] == df['query_dominant_category']).astype(float)

    return df


def main():
    print("\n--- Phase 1: Loading Normalization Stats & IDF Map ---")
    try:
        with open(f'{ROOT_DIR}/output/advanced_normalization_stats.json', "r") as f:
            stats = json.load(f)
        train_mean = np.array(stats["mean"])
        train_std = np.array(stats["std"])
        feature_cols = stats["features"]
        idf_map = stats.get("idf_map", {})
    except FileNotFoundError:
        print("ERROR: advanced_normalization_stats.json not found. Run training first.")
        return

    print("\n--- Phase 2: Feature Extraction ---")
    df_test = extract_test_advanced_features(idf_map)

    print("\n--- Phase 3: Generating Ground Truths ---")
    # Apply Business Rules to the Test Ground Truth
    df_truth = apply_business_ndcg_labels(df_test.copy(), budget_col='user_budget', price_col='price', stars_col='stars_clean')
    
    df_truth_standard = df_truth.copy()
    df_truth_business = df_truth.copy()
    df_truth_business['relevance'] = df_truth_business['business_relevance'] 

    # Filter out missing features and normalize
    df_test = df_test.dropna(subset=feature_cols)
    features_raw = df_test[feature_cols].values
    features_normalized = (features_raw - train_mean) / train_std

    print("\n--- Phase 4: Running Inference ---")
    weights_path = f'{ROOT_DIR}/output/best_advanced_reranker.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AdvancedDeepReranker(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(features_normalized, dtype=torch.float32).to(device)
        predictions = model(x_tensor).cpu().squeeze().numpy()
        
    df_test['predicted_score'] = predictions

    print("\n--- Phase 5: Calculating Metrics ---")

    df_test['relevance'] = df_truth_standard['relevance']
    # 1. Standard Metrics
    standard_ndcg = ndcg_at_k(df_test.copy(), score_col='predicted_score', k=10)
    standard_recall = recall_at_k(df_test.copy(), df_truth_standard, score_col='predicted_score', k=10)

    # 2. Business Metrics (Evaluated with Budget Penalties and Quality Tie-Breakers)
    # We swap out the standard relevance for the business relevance here
    df_test_business = df_test.copy()
    df_test_business['relevance'] = df_truth_business['business_relevance'] 
    business_ndcg = ndcg_at_k(df_test_business, score_col='predicted_score', k=10)

    print("\n" + "="*60)
    print(" ADVANCED NEURAL RERANKER EVALUATION")
    print("="*60)
    print("1. Standard Textual Relevance (Compared to ESCI Baseline)")
    print(f"   Standard NDCG@10:  {standard_ndcg:.4f}")
    print(f"   Recall@10:         {standard_recall:.4f}")
    print("-" * 60)
    print("2. Business Relevance (Intent, Budget, Quality)")
    print(f"   Business NDCG@10:  {business_ndcg:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()