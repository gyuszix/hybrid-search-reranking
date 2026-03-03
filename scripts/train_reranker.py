import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from reranking.features import extract_esci_features, PairwiseESCIDataset
from reranking.model import DeepESCIReranker

def train_model():
    bm25_csv_path = f'{ROOT_DIR}/output/bm25_scores_train.csv'
    semantic_csv_path = f'{ROOT_DIR}/output/two_tower_scores_train.csv'

    df_all, feature_columns = extract_esci_features(EXAMPLES_PATH, PRODUCTS_PATH, bm25_csv_path, semantic_csv_path)

    # Detect Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Training on: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    # Split into Train and Validation
    print("\nSplitting queries into Train (85%) and Validation (15%)...")
    unique_queries = df_all['query_id'].unique().tolist()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)

    df_train = df_all[df_all['query_id'].isin(train_queries)].copy()
    df_val = df_all[df_all['query_id'].isin(val_queries)].copy()

    train_dataset = PairwiseESCIDataset(df_train, feature_columns)
    val_dataset = PairwiseESCIDataset(df_val, feature_columns, mean=train_dataset.mean, std=train_dataset.std)

    # Use 4-8 workers depending on the CPU (4 is a safe start)
    # pin_memory=True speeds up the transfer to the GPU
    num_workers = 4 if os.name != 'nt' else 0 # Use 0 on Windows if you get freezing issues, otherwise try 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,        # <-- MULTI-CORE
        pin_memory=True,      # <-- FAST GPU TRANSFER
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Save normalization stats
    with open(f'{ROOT_DIR}/output/normalization_stats.json', "w") as f:
        json.dump({"mean": train_dataset.mean.tolist(), "std": train_dataset.std.tolist()}, f)
        print("Saved training stats for later evaluation.")

    # Move Model to GPU
    model = DeepESCIReranker(input_dim=len(feature_columns)).to(device)
    # increased margin from 0.1 to 1.0 to force strong separation
    criterion = nn.MarginRankingLoss(margin=1.0)
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

            # --- MOVE BATCH TO GPU ---
            batch_x_pos = batch_x_pos.to(device)
            batch_x_neg = batch_x_neg.to(device)
            batch_y = batch_y.to(device)

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
                # --- MOVE BATCH TO GPU ---
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
            torch.save(model.state_dict(), f'{ROOT_DIR}/output/best_esci_reranker.pth')
            print("  --> Best weights updated.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                break
    return model

if __name__ == "__main__":
    train_model()
