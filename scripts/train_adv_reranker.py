import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from reranking.advanced_features import extract_advanced_features, AdvancedPairwiseDataset
from reranking.advanced_model import AdvancedDeepReranker

def train_advanced_model():
    bm25_csv_path = f'{ROOT_DIR}/output/bm25_scores_train.csv'
    semantic_csv_path = f'{ROOT_DIR}/output/two_tower_scores_train.csv'
    esci_s_path = f'{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet'

    # Catch the idf_map returned from the updated feature extractor
    df_all, feature_columns, idf_map = extract_advanced_features(
        EXAMPLES_PATH, PRODUCTS_PATH, bm25_csv_path, semantic_csv_path, esci_s_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    print("\nSplitting queries into Train (85%) and Validation (15%)...")
    unique_queries = df_all['query_id'].unique().tolist()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)

    df_train = df_all[df_all['query_id'].isin(train_queries)].copy()
    df_val = df_all[df_all['query_id'].isin(val_queries)].copy()

    train_dataset = AdvancedPairwiseDataset(df_train, feature_columns)
    val_dataset = AdvancedPairwiseDataset(df_val, feature_columns, mean=train_dataset.mean, std=train_dataset.std)

    num_workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1024, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Save advanced normalization stats AND the IDF map
    os.makedirs(f'{ROOT_DIR}/output', exist_ok=True)
    with open(f'{ROOT_DIR}/output/advanced_normalization_stats.json', "w") as f:
        json.dump({
            "mean": train_dataset.mean.tolist(), 
            "std": train_dataset.std.tolist(),
            "features": feature_columns,
            "idf_map": idf_map  # Injected here for Training-Inference Parity
        }, f)
        print("Saved advanced training stats and IDF map for later evaluation.")

    model = AdvancedDeepReranker(input_dim=len(feature_columns)).to(device)
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs, patience, patience_counter, best_val_loss = 50, 6, 0, float('inf')

    print(f"\nStarting Advanced Pairwise Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x_pos, batch_x_neg, batch_y in train_loader:
            optimizer.zero_grad()

            batch_x_pos = batch_x_pos.to(device)
            batch_x_neg = batch_x_neg.to(device)
            batch_y = batch_y.to(device)

            pos_scores = model(batch_x_pos).squeeze()
            neg_scores = model(batch_x_neg).squeeze()
            loss = criterion(pos_scores, neg_scores, batch_y)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x_pos, batch_x_neg, batch_y in val_loader:
                batch_x_pos = batch_x_pos.to(device)
                batch_x_neg = batch_x_neg.to(device)
                batch_y = batch_y.to(device)

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
            torch.save(model.state_dict(), f'{ROOT_DIR}/output/best_advanced_reranker.pth')
            print("  --> Best weights updated.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                break
    return model

if __name__ == "__main__":
    train_advanced_model()