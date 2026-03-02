import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from config import EXAMPLES_PATH, PRODUCTS_PATH

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

    # --- Feature Engineering ---
    def calc_overlap(row):
        q_words = set(stemmer.stem(str(w)) for w in str(row['query']).lower().split())
        t_words = set(stemmer.stem(str(w)) for w in str(row['product_title']).lower().split())
        if len(q_words) == 0: return 0.0
        return len(q_words.intersection(t_words)) / len(q_words)
    
    print("Calculating stemming-based word overlap features...")
    df['word_overlap'] = df.apply(calc_overlap, axis=1)
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    df['title_length'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
    df['has_brand'] = df['product_brand'].notna().astype(float)
    df['bullet_count'] = df['product_bullet_point'].astype(str).apply(lambda x: len(x.split('\n')) if x != 'None' else 0)
    
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
            
        normalized_features = (features - self.mean) / self.std
        df_scaled = df[['query_id', 'target_score']].copy()
        df_scaled['feature_idx'] = np.arange(len(df))
        
        print(f"Generating pairwise combinations for {len(df_scaled)} rows...")
        self.pairs = []
        
        # Group by query to generate valid pairs where item A > item B
        for _, group in df_scaled.groupby('query_id'):
            group = group.sort_values('target_score', ascending=False)
            if group['target_score'].nunique() <= 1:
                continue
                
            items = group.to_dict('records')
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i]['target_score'] > items[j]['target_score']:
                        self.pairs.append({
                            'pos_idx': items[i]['feature_idx'],
                            'neg_idx': items[j]['feature_idx']
                        })
        
        self.features = normalized_features
        print(f"Total training pairs generated: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        x_pos = torch.tensor(self.features[pair['pos_idx']], dtype=torch.float32)
        x_neg = torch.tensor(self.features[pair['neg_idx']], dtype=torch.float32)
        y = torch.tensor(1.0, dtype=torch.float32)
        return x_pos, x_neg, y

# ==========================================
# 3. Model & Training Loop
# ==========================================
class DeepESCIReranker(nn.Module):
    def __init__(self, input_dim):
        super(DeepESCIReranker, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

def train_model():
    bm25_csv_path = "output/bm25_scores_train.csv" 
    semantic_csv_path = "output/two_tower_scores_train.csv"
    
    df_all, feature_columns = extract_esci_features(EXAMPLES_PATH, PRODUCTS_PATH, bm25_csv_path, semantic_csv_path)
    
    # Split into Train and Validation
    print("\nSplitting queries into Train (85%) and Validation (15%)...")
    unique_queries = df_all['query_id'].unique()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)
    
    df_train = df_all[df_all['query_id'].isin(train_queries)].copy()
    df_val = df_all[df_all['query_id'].isin(val_queries)].copy()
    
    train_dataset = PairwiseESCIDataset(df_train, feature_columns)
    val_dataset = PairwiseESCIDataset(df_val, feature_columns, mean=train_dataset.mean, std=train_dataset.std)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # Save normalization stats
    os.makedirs("output", exist_ok=True)
    with open("output/normalization_stats.json", "w") as f:
        json.dump({"mean": train_dataset.mean.tolist(), "std": train_dataset.std.tolist()}, f)
        print("Saved training stats for later evaluation.")
    
    model = DeepESCIReranker(input_dim=len(feature_columns))
    criterion = nn.MarginRankingLoss(margin=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    epochs, patience, patience_counter, best_val_loss = 50, 6, 0, float('inf')
    
    print(f"\nStarting Pairwise Training Loop...")
    for epoch in range(epochs):
        model.train() 
        total_train_loss = 0.0
        for batch_x_pos, batch_x_neg, batch_y in train_loader:
            # FIX: Added zero_grad()
            optimizer.zero_grad()
            
            pos_scores = model(batch_x_pos).squeeze()
            neg_scores = model(batch_x_neg).squeeze()
            loss = criterion(pos_scores, neg_scores, batch_y)
            
            # FIX: Added backward and step
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x_pos, batch_x_neg, batch_y in val_loader:
                pos_scores = model(batch_x_pos).squeeze()
                neg_scores = model(batch_x_neg).squeeze()
                loss = criterion(pos_scores, neg_scores, batch_y)
                total_val_loss += loss.item()
                
        avg_train_loss, avg_val_loss = total_train_loss/len(train_loader), total_val_loss/len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_esci_reranker.pth")
            print("  --> Best weights updated.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n[!] Early stopping triggered.")
                break
    return model

if __name__ == "__main__":
    train_model()