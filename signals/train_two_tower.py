# signals/train_two_tower.py
#
# Fine-tunes the two-tower SentenceTransformer on the ESCI train split.
# Teaches the model to rank relevant products higher than irrelevant ones
# by using ESCI labels as similarity scores.
#
# Output: models/two_tower_finetuned/
# Usage:  python signals/train_two_tower.py

import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXAMPLES_PATH, PRODUCTS_PATH

# ----------------------
# Settings
# ----------------------
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"
MODEL_SAVE_PATH = "models/two_tower_finetuned"
LOCALE = "us"
BATCH_SIZE = 64
NUM_EPOCHS = 1


def main():

    # ----------------------
    # 1. Load data
    # ----------------------
    print("Loading data...")
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)

    df = pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")
    df = df[df["small_version"] == 1]
    df = df[df["split"] == "train"]
    df = df[df["product_locale"] == LOCALE]

    # Filter to ONLY positive examples (Exact or Substitute)
    # MNRL expects positive pairs. It creates negatives from other pairs in the batch.
    df = df[df["esci_label"].isin(["E", "S"])]
    df["item_text"] = df["product_title"].fillna("")

    print(f"Training samples (Positives only): {len(df)}")

    # ----------------------
    # 2. Build training examples
    # ----------------------
    train_samples = []
    for _, row in df.iterrows():
        train_samples.append(InputExample(
            texts=[row["query"], row["item_text"]]
        ))

    # Shuffle is CRITICAL for MNRL to ensure diverse in-batch negatives
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    # ----------------------
    # 3. Load model and loss
    # ----------------------
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # ----------------------
    # 4. Train
    # ----------------------
    print("Training...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        output_path=MODEL_SAVE_PATH,
        show_progress_bar=True,
    )

    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()