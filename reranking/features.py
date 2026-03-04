import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.stem import PorterStemmer

# ==========================================
# 1. Feature Extraction Pipeline
# ==========================================
def extract_esci_features(examples_path, products_path, bm25_csv_path, semantic_csv_path):
    print("Loading raw ESCI Parquet files...")
    df_ex = pd.read_parquet(examples_path)
    df_pr = pd.read_parquet(products_path)

    # Force IDs to string
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_pr['product_id'] = df_pr['product_id'].astype(str)

    # --- CRITICAL: FILTER TO TRAIN SPLIT ONLY ---
    print("Isolating training split and labels...")
    df_train_queries = df_ex[df_ex['split'] == 'train'][['query_id', 'query']].drop_duplicates()

    # Load retrieved candidates (ensure these files exist in your output/ folder)
    df_bm25 = pd.read_csv(bm25_csv_path)
    df_bm25.columns = ['query_id', 'product_id', 'bm25_score']
    df_bm25['query_id'], df_bm25['product_id'] = df_bm25['query_id'].astype(str), df_bm25['product_id'].astype(str)

    df_sem = pd.read_csv(semantic_csv_path)
    df_sem.columns = ['query_id', 'product_id', 'semantic_score']
    df_sem['query_id'], df_sem['product_id'] = df_sem['query_id'].astype(str), df_sem['product_id'].astype(str)

    # 1. Start with the candidate pool (Outer join to keep all BM25 and Semantic candidates)
    candidates = pd.merge(df_bm25, df_sem, on=['query_id', 'product_id'], how='outer')

    # Filter candidates to only include those for our training queries
    df = pd.merge(candidates, df_train_queries, on='query_id', how='inner')

    # 2. Merge with Product Data to get features
    df = pd.merge(df, df_pr, on='product_id', how='inner')

    # 3. Left join the Explicit ESCI Labels
    df_labels = df_ex[['query_id', 'product_id', 'esci_label']].drop_duplicates()
    df = pd.merge(df, df_labels, on=['query_id', 'product_id'], how='left')

    # Fill missing retrieval scores
    df['bm25_score'] = df['bm25_score'].fillna(0.0)
    df['semantic_score'] = df['semantic_score'].fillna(-1.0)

    # --- Target Labels (HARD NEGATIVES APPLIED HERE) ---
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    # Any candidate that lacks a ground truth label is assumed irrelevant (0.0)
    df['target_score'] = df['esci_label'].map(label_map).fillna(0.0)

    # Initialize the stemmer
    stemmer = PorterStemmer()

    print("Calculating stemming-based word overlap features...")

    # 1. Extract unique strings to avoid redundant processing
    unique_queries = df['query'].astype(str).unique()
    unique_titles = df['product_title'].astype(str).unique()

    print(f"  -> Stemming {len(unique_queries)} unique queries...")
    query_stem_map = {
        q: set(stemmer.stem(w) for w in q.lower().split())
        for q in unique_queries
    }

    print(f"  -> Stemming {len(unique_titles)} unique titles...")
    title_stem_map = {
        t: set(stemmer.stem(w) for w in t.lower().split())
        for t in unique_titles
    }

    # 2. Fast lookup function
    def fast_overlap(q, t):
        q_set = query_stem_map.get(q, set())
        t_set = title_stem_map.get(t, set())
        if not q_set:
            return 0.0
        return len(q_set.intersection(t_set)) / len(q_set)

    # 3. Apply using zip (which is vastly faster than pandas .apply for multiple columns)
    print("  -> Mapping intersections to dataframe...")
    df['word_overlap'] = [
        fast_overlap(str(q), str(t))
        for q, t in zip(df['query'], df['product_title'])
    ]
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    df['title_length'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
    df['has_brand'] = df['product_brand'].notna().astype(float)
    df['bullet_count'] = df['product_bullet_point'].fillna("").astype(str).apply(lambda x: len(x.split('\n')) if x.strip() and x.strip() != 'None' else 0)

    prod_counts = df.groupby('product_id')['query_id'].transform('count')
    df['log_product_freq'] = np.log1p(prod_counts)
    brand_counts = df.groupby('product_brand')['product_id'].transform('count')
    df['log_brand_freq'] = np.log1p(brand_counts.fillna(0))

    feature_cols = [
        'bm25_score', 'semantic_score', 'word_overlap',
        'query_length', 'title_length', 'has_brand',
        'bullet_count', 'log_product_freq', 'log_brand_freq'
    ]
    df = df.dropna(subset=feature_cols)
    return df, feature_cols

# ==========================================
# 2. Pairwise Dataset Class
# ==========================================
class PairwiseESCIDataset(Dataset):
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

        # Group by query
        for _, group in df_scaled.groupby('query_id'):
            targets = group['target_score'].values
            indices = group['feature_idx'].values

            # If all items have the same score, skip
            if len(targets) < 2 or len(np.unique(targets)) == 1:
                continue

            # 1. BROADCASTING: Create a 2D boolean mask where matrix[i, j] is True if target[i] > target[j]
            # This replaces the nested "for i... for j..." loop entirely
            mask = targets[:, None] > targets[None, :]

            # 2. Get the row (pos) and col (neg) indices where the mask is True
            pos_i, neg_i = np.where(mask)

            # 3. DOWNSAMPLING: Identify which negative items are unjudged (0.0)
            neg_targets = targets[neg_i]
            unjudged_mask = (neg_targets == 0.0)

            # Create a keep mask: Keep 100% of labeled pairs, and 10% of unjudged pairs
            keep_mask = np.ones(len(pos_i), dtype=bool)
            keep_mask[unjudged_mask] = np.random.rand(np.sum(unjudged_mask)) <= 0.10

            # 4. Filter the indices using the keep mask
            pos_i = pos_i[keep_mask]
            neg_i = neg_i[keep_mask]

            # 5. Map the local group indices back to the global dataframe indices
            pos_indices_list.extend(indices[pos_i])
            neg_indices_list.extend(indices[neg_i])

        # Store as fast NumPy arrays instead of a massive list of dictionaries
        self.pos_indices = np.array(pos_indices_list, dtype=np.int32)
        self.neg_indices = np.array(neg_indices_list, dtype=np.int32)

        print(f"Total training pairs generated: {len(self.pos_indices)}")

    def __len__(self):
        return len(self.pos_indices)

    def __getitem__(self, idx):
        # Direct array indexing is significantly faster during the PyTorch DataLoader loop
        x_pos = torch.tensor(self.features[self.pos_indices[idx]], dtype=torch.float32)
        x_neg = torch.tensor(self.features[self.neg_indices[idx]], dtype=torch.float32)
        y = torch.tensor(1.0, dtype=torch.float32)
        return x_pos, x_neg, y
