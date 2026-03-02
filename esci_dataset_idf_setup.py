import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_esci_data(data_dir="./esci-data/shopping_queries_dataset"):
    print("Loading ESCI Parquet files...")
    # 1. Load the raw files
    df_examples = pd.read_parquet(f'{data_dir}/shopping_queries_dataset_examples.parquet')
    df_products = pd.read_parquet(f'{data_dir}/shopping_queries_dataset_products.parquet')
    
    # 2. Merge queries with product details (Title, Brand, Bullets)
    print("Merging queries with product metadata...")
    df_merged = pd.merge(
        df_examples, 
        df_products, 
        how='left', 
        on=['product_locale', 'product_id']
    )
    
    # Optional: Filter for English (US) only to speed up initial testing
    df_merged = df_merged[df_merged['product_locale'] == 'us']
    
    # 3. Filter for Task 1 (The "Small" Version for prototyping)
    df_task1 = df_merged[df_merged["small_version"] == 1]
    
    # 4. Isolate the Official Test Set (DO NOT train on this)
    test_df = df_task1[df_task1["split"] == "test"].copy()
    
    # 5. Create Train and Validation splits from the Official Train set
    official_train_df = df_task1[df_task1["split"] == "train"].copy()
    
    # We group by 'query_id' before splitting to ensure all results for a single 
    # query end up in the SAME split. (Preventing data leakage)
    unique_queries = official_train_df['query_id'].unique()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)
    
    train_df = official_train_df[official_train_df['query_id'].isin(train_queries)].copy()
    val_df = official_train_df[official_train_df['query_id'].isin(val_queries)].copy()
    
    print("\n--- Data Setup Complete ---")
    print(f"Training Rows:   {len(train_df)} (Used to update weights)")
    print(f"Validation Rows: {len(val_df)} (Used for early stopping)")
    print(f"Test Rows:       {len(test_df)} (Locked away for final report)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_data, val_data, test_data = load_and_split_esci_data()
    
    # Save these clean splits to CSV so you don't have to run this merge every time
    train_data.to_csv("esci_train_clean.csv", index=False)
    val_data.to_csv("esci_val_clean.csv", index=False)
    test_data.to_csv("esci_test_clean.csv", index=False)
    print("Saved clean splits to CSV!")