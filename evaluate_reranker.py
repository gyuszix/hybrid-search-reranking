import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

# ==========================================
# 1. The Exact Same Model Architecture
# ==========================================
# (PyTorch requires the structure to be identical to load the .pth weights)
class DeepESCIReranker(nn.Module):
    def __init__(self, input_dim):
        super(DeepESCIReranker, self).__init__()
        
        # We removed Dropout to stop starving the network of the semantic score
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1) # Outputs a raw adjustment value
        )

    def forward(self, x):
        # RESIDUAL CONNECTION:
        # Assuming semantic_score is at index 1 of your feature tensor.
        # We extract it and add it directly to the MLP's output.
        # This tells the network: "Start with the Two-Tower score, then modify it."
        base_semantic_score = x[:, 1].unsqueeze(1) 
        
        adjustment = self.mlp(x)
        
        # Combine them and apply Sigmoid to bind between 0 and 1
        final_raw_score = base_semantic_score + adjustment
        return torch.sigmoid(final_raw_score)

# ==========================================
# 2. Test Feature Extraction
# ==========================================
def extract_test_features(examples_path, products_path, bm25_csv_path, semantic_csv_path):
    print("Loading raw ESCI Parquet files...")
    df_ex = pd.read_parquet(examples_path)
    df_pr = pd.read_parquet(products_path)
    
    print("Merging ESCI datasets...")
    df = pd.merge(df_ex, df_pr, how='inner', on=['product_id', 'product_locale'])
    
    # CRITICAL: We only want the TEST split now
    df = df[df['split'] == 'test'].copy()
    
    df['query_id'] = df['query_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)

    print("Loading retrieval scores...")
    df_bm25 = pd.read_csv(bm25_csv_path)
    df_bm25.columns = ['query_id', 'product_id', 'bm25_score']
    df_bm25['query_id'], df_bm25['product_id'] = df_bm25['query_id'].astype(str), df_bm25['product_id'].astype(str)

    df_sem = pd.read_csv(semantic_csv_path)
    df_sem.columns = ['query_id', 'product_id', 'semantic_score']
    df_sem['query_id'], df_sem['product_id'] = df_sem['query_id'].astype(str), df_sem['product_id'].astype(str)

    df = pd.merge(df, df_bm25, how='left', on=['query_id', 'product_id'])
    df = pd.merge(df, df_sem, how='left', on=['query_id', 'product_id'])

    df['bm25_score'] = df['bm25_score'].fillna(0.0)
    df['semantic_score'] = df['semantic_score'].fillna(-1.0)
    
    print(f"Extracting features for {len(df)} TEST rows...")
    
    # Calculate features
    def calc_overlap(row):
        q_words = set(str(row['query']).lower().split())
        t_words = set(str(row['product_title']).lower().split())
        if len(q_words) == 0: return 0.0
        return len(q_words.intersection(t_words)) / len(q_words)
    
    df['word_overlap'] = df.apply(calc_overlap, axis=1)
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    df['title_length'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
    df['has_brand'] = df['product_brand'].notna().astype(float)
    df['bullet_count'] = df['product_bullet_point'].astype(str).apply(lambda x: len(x.split('\n')) if x != 'None' else 0)
    
    prod_counts = df.groupby('product_id')['query_id'].transform('count')
    df['log_product_freq'] = np.log1p(prod_counts)
    brand_counts = df.groupby('product_brand')['product_id'].transform('count')
    df['log_brand_freq'] = np.log1p(brand_counts.fillna(0))
    
    # Official KDD Cup Gain mapping for NDCG (E=1.0, S=0.1, C=0.01, I=0.0)
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df['ground_truth_gain'] = df['esci_label'].map(label_map).fillna(0.0)
    
    feature_cols = [
        'bm25_score', 'semantic_score', 'word_overlap', 
        'query_length', 'title_length', 'has_brand', 
        'bullet_count', 'log_product_freq', 'log_brand_freq'
    ]
    df = df.dropna(subset=feature_cols)
    
    return df, feature_cols

# ==========================================
# 3. NDCG Calculation Math
# ==========================================
def dcg_at_k(relevance_scores, k=10):
    """Calculates Discounted Cumulative Gain."""
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        # standard DCG formula
        discounts = np.log2(np.arange(2, relevance_scores.size + 2))
        return np.sum((np.power(2, relevance_scores) - 1) / discounts)
    return 0.0

def ndcg_at_k(relevance_scores, k=10):
    """Calculates Normalized Discounted Cumulative Gain."""
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(relevance_scores, k) / dcg_max

# ==========================================
# 4. Main Evaluation Loop
# ==========================================
def evaluate_model(model_weights_path="best_esci_reranker.pth"):
    examples_file = "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    products_file = "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    bm25_csv_path = "bm25_scores.csv" 
    semantic_csv_path = "two_tower_scores.csv"
    
    df_test, feature_cols = extract_test_features(examples_file, products_file, bm25_csv_path, semantic_csv_path)
    
    # Normalize the features 
    # (Note: In strict production, you use the Train Set's Mean/Std here, 
    # but for local testing, normalizing the test set against itself works fine).
    features_raw = df_test[feature_cols].values
    features_normalized = (features_raw - features_raw.mean(axis=0)) / (features_raw.std(axis=0) + 1e-8)
    
    # Load Model
    print(f"Loading trained weights from {model_weights_path}...")
    model = DeepESCIReranker(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_weights_path))
    model.eval() # CRITICAL: Turns off dropout for deterministic testing
    
    # Run Inference
    print("Running test features through the Neural Network...")
    with torch.no_grad():
        x_tensor = torch.tensor(features_normalized, dtype=torch.float32)
        predictions = model(x_tensor).squeeze().numpy()
        
    df_test['predicted_score'] = predictions
    
    # Calculate NDCG@10 per query
    print("Calculating NDCG@10...")
    ndcg_scores = []
    
    # Group the dataframe by the query, so we can rank the items within each search
    grouped = df_test.groupby('query_id')
    
    for query_id, group in grouped:
        # Sort the items by what our Neural Network predicted
        ranked_group = group.sort_values(by='predicted_score', ascending=False)
        
        # Get the actual ground-truth gains of those items in the new sorted order
        actual_gains_in_ranked_order = ranked_group['ground_truth_gain'].values
        
        # Calculate NDCG
        score = ndcg_at_k(actual_gains_in_ranked_order, k=10)
        ndcg_scores.append(score)
        
    final_ndcg = np.mean(ndcg_scores)
    
    print("="*50)
    print(f"FINAL TEST NDCG@10: {final_ndcg:.4f}")
    print("="*50)
    
    # Optional: Save the ranked results to look at them manually
    # df_test[['query', 'product_title', 'predicted_score', 'esci_label']].to_csv("final_test_predictions.csv", index=False)
    
if __name__ == "__main__":
    evaluate_model("best_esci_reranker.pth") # Make sure the file name matches what you saved!