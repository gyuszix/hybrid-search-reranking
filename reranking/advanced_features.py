import pandas as pd
import numpy as np
import torch
import re
from collections import Counter
from torch.utils.data import Dataset
from nltk.stem import PorterStemmer

# ==========================================
# 1. Advanced Feature Extraction Pipeline
# ==========================================
def extract_advanced_features(examples_path, products_path, bm25_csv_path, semantic_csv_path, esci_s_path):
    print("Loading raw Data...")
    df_ex = pd.read_parquet(examples_path)
    df_pr = pd.read_parquet(products_path)
    
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_pr['product_id'] = df_pr['product_id'].astype(str)

    # Merge ESCI-S data with Imputation Handlers
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
        print("ESCI-S data not found. Creating dummy columns for missing data handles.")
        for col in ['price', 'stars', 'ratings', 'category']:
            df_pr[col] = np.nan

    print("Isolating training split and labels...")
    df_train_queries = df_ex[df_ex['split'] == 'train'][['query_id', 'query']].drop_duplicates()

    df_bm25 = pd.read_csv(bm25_csv_path)
    df_bm25.columns = ['query_id', 'product_id', 'bm25_score']
    df_sem = pd.read_csv(semantic_csv_path)
    df_sem.columns = ['query_id', 'product_id', 'semantic_score']

    for df in [df_bm25, df_sem]:
        df['query_id'], df['product_id'] = df['query_id'].astype(str), df['product_id'].astype(str)

    candidates = pd.merge(df_bm25, df_sem, on=['query_id', 'product_id'], how='outer')
    df = pd.merge(candidates, df_train_queries, on='query_id', how='inner')
    df = pd.merge(df, df_pr, on='product_id', how='inner')

    df_labels = df_ex[['query_id', 'product_id', 'esci_label']].drop_duplicates()
    df = pd.merge(df, df_labels, on=['query_id', 'product_id'], how='left')

    df['bm25_score'] = df['bm25_score'].fillna(0.0)
    df['semantic_score'] = df['semantic_score'].fillna(-1.0)
    
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df['target_score'] = df['esci_label'].map(label_map).fillna(0.0)

    print("Extracting Advanced Features...")
    
    # --- A. Query Intent Features ---
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    
    def parse_budget(q):
        match = re.search(r'under\s*\$?(\d+)', str(q).lower())
        return float(match.group(1)) if match else -1.0
    df['user_budget'] = df['query'].apply(parse_budget)
    df['cheap_intent'] = df['query'].str.lower().str.contains('cheap|affordable|budget').astype(float)

    all_words = " ".join(df_train_queries['query'].str.lower().tolist()).split()
    N = len(df_train_queries)
    word_counts = Counter(all_words)
    idf_map = {word: np.log((N + 1) / (count + 1)) + 1 for word, count in word_counts.items()}
    
    def get_idf_stats(q):
        words = str(q).lower().split()
        if not words: return 0.0, 0.0
        idfs = [idf_map.get(w, 10.0) for w in words]
        return np.mean(idfs), np.max(idfs)
        
    idf_stats = df['query'].apply(get_idf_stats)
    df['query_mean_idf'] = [x[0] for x in idf_stats]
    df['query_max_idf'] = [x[1] for x in idf_stats]

    # Force categories to string to prevent unhashable array errors during groupby
    df['category'] = df['category'].fillna("Unknown").astype(str)

    # --- B. Item Authority & Intelligent Imputation ---
    # 1. Force strict numeric types to prevent median() TypeError crashes
    # This strips out any stray characters (like '$') and forces floats
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')

    # Price Imputation: Category Median -> Global Median -> Neutral 0.0
    df['is_price_missing'] = df['price'].isna().astype(float)
    cat_median_price = df.groupby('category')['price'].transform('median')
    global_median_price = df['price'].median() if not df['price'].isna().all() else 25.0
    imputed_price = df['price'].fillna(cat_median_price).fillna(global_median_price).fillna(0.0)
    df['log_price'] = np.log1p(imputed_price)

    # Stars Imputation: Category Median -> Global Median -> Neutral 0.0
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

    # Hard Constraint: Color
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

    feature_cols = [
        'query_length', 'query_mean_idf', 'query_max_idf', 'user_budget', 'cheap_intent',
        'log_price', 'is_price_missing', 'stars_clean', 'log_review_count', 'is_rating_missing',
        'bm25_score', 'semantic_score', 'word_overlap', 'is_dominant_category', 'brand_match', 
        'color_match', 'is_over_budget'
    ]
    df = df.dropna(subset=feature_cols)
    
    # Return the idf_map so the training script can save it
    return df, feature_cols, idf_map

# ==========================================
# 2. Vectorized Dataset Class
# ==========================================
class AdvancedPairwiseDataset(Dataset):
    def __init__(self, df, feature_cols, mean=None, std=None):
        features = df[feature_cols].values

        if mean is None or std is None:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.features = (features - self.mean) / self.std

        df_scaled = df[['query_id', 'target_score']].copy()
        df_scaled['feature_idx'] = np.arange(len(df))

        print(f"Vectorizing pairwise combinations for {len(df_scaled)} rows...")

        pos_indices_list = []
        neg_indices_list = []

        for _, group in df_scaled.groupby('query_id'):
            targets = group['target_score'].values
            indices = group['feature_idx'].values

            if len(targets) < 2 or len(np.unique(targets)) == 1:
                continue

            mask = targets[:, None] > targets[None, :]
            pos_i, neg_i = np.where(mask)

            neg_targets = targets[neg_i]
            unjudged_mask = (neg_targets == 0.0)

            keep_mask = np.ones(len(pos_i), dtype=bool)
            keep_mask[unjudged_mask] = np.random.rand(np.sum(unjudged_mask)) <= 0.10

            pos_i = pos_i[keep_mask]
            neg_i = neg_i[keep_mask]

            pos_indices_list.extend(indices[pos_i])
            neg_indices_list.extend(indices[neg_i])

        self.pos_indices = np.array(pos_indices_list, dtype=np.int32)
        self.neg_indices = np.array(neg_indices_list, dtype=np.int32)

        print(f"Total training pairs generated: {len(self.pos_indices)}")

    def __len__(self):
        return len(self.pos_indices)

    def __getitem__(self, idx):
        x_pos = torch.tensor(self.features[self.pos_indices[idx]], dtype=torch.float32)
        x_neg = torch.tensor(self.features[self.neg_indices[idx]], dtype=torch.float32)
        y = torch.tensor(1.0, dtype=torch.float32)
        return x_pos, x_neg, y