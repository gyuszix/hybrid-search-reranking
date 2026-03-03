import os
import pandas as pd
from config import EXAMPLES_PATH, PRODUCTS_PATH, USE_SMALL_VERSION
from signals.two_tower import compute_two_tower_scores

def main():
    print("Loading raw ESCI data for Two-Tower scoring...")
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)

    if USE_SMALL_VERSION:
        print("Using small version of dataset...")
        df_examples = df_examples[df_examples["small_version"] == 1]

    print("Merging datasets and preparing text features...")
    # Merge on product_id and locale
    df = df_examples.merge(
        df_products,
        on=["product_id", "product_locale"],
        how="left"
    )

    # Combine text fields for the Two-Tower encoder
    df["item_text"] = (
        df["product_title"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["product_bullet_point"].fillna("")
    )

    # Rename columns to match what compute_two_tower_scores expects
    df = df.rename(columns={
        "query": "query_text",
        "product_id": "item_id"
    })

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Loop through both splits to generate separate CSVs
    for split in ['train', 'test']:
        print(f"\n--- Processing '{split}' split ---")
        
        # Filter the dataframe for the current split
        df_split = df[df["split"] == split].copy()
        
        if df_split.empty:
            print(f"No data found for {split} split. Skipping.")
            continue

        print(f"Computing Two-Tower scores for {len(df_split)} rows in the {split} set...")
        print("This may take a while depending on hardware...")
        
        # Call the existing logic from two_tower.py
        two_tower_df = compute_two_tower_scores(df_split)
        
        if two_tower_df.empty:
            print(f"Warning: No scores generated for {split} split.")
            continue
            
        # Save to a distinct CSV file
        output_file = f"output/two_tower_scores_{split}.csv"
        two_tower_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(two_tower_df)} Two-Tower scores to {output_file}")

if __name__ == "__main__":
    main()