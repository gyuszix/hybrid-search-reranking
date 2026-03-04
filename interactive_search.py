import os
import sys

# --- THE C++ THREADING CRASH FIX ---
# 1. Stops HuggingFace from using Rust parallel threads inside a GUI
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. Prevents Anaconda's OpenMP libraries from colliding with FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 3. Forces math operations to use a single thread during inference
os.environ["OMP_NUM_THREADS"] = "1"
# -----------------------------------

import json
import torch
import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter
from nltk.stem import PorterStemmer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR, PRODUCTS_PATH
from reranking.advanced_model import AdvancedDeepReranker

# Real Retrievers Imported
from retrieval.bm25 import load_bm25_index, search_bm25_global
from retrieval.two_tower import load_tt_index, search_tt_global

def apply_mmr(df_results, score_col='predicted_score', brand_col='product_brand', lambda_param=0.6, top_n=20):
    if df_results.empty: return df_results
    df_sorted = df_results.sort_values(by=score_col, ascending=False).copy()
    selected_indices, seen_brands = [], set()
    indices = df_sorted.index.tolist()
    scores = df_sorted[score_col].values
    brands = df_sorted[brand_col].fillna("Unknown").values
    
    while len(selected_indices) < min(top_n, len(df_sorted)) and indices:
        best_mmr_score, best_idx_pos = -float('inf'), -1
        for i, idx in enumerate(indices):
            rel_score, brand = scores[i], brands[i]
            brand_penalty = 1.0 if brand in seen_brands and brand != "Unknown" else 0.0
            mmr_score = (lambda_param * rel_score) - ((1.0 - lambda_param) * brand_penalty)
            
            if mmr_score > best_mmr_score:
                best_mmr_score, best_idx_pos = mmr_score, i
                
        winning_idx = indices.pop(best_idx_pos)
        seen_brands.add(brands[best_idx_pos])
        scores = np.delete(scores, best_idx_pos)
        brands = np.delete(brands, best_idx_pos)
        selected_indices.append(winning_idx)

    return df_results.loc[selected_indices]

# ==========================================
# The Search Engine Backend
# ==========================================
class SearchEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Boot] Initializing Search Engine on {self.device}...")
        
        # 1. Load Vector Indices
        print("[Boot] Loading Vector Indices...")
        self.bm25_index, self.bm25_ids = load_bm25_index()
        self.tt_model, self.tt_index, self.tt_ids = load_tt_index()
        
        # 2. Load Catalog & ESCI-S Data
        print("[Boot] Loading Catalog Metadata...")
        self.df_pr = pd.read_parquet(PRODUCTS_PATH)
        self.df_pr['product_id'] = self.df_pr['product_id'].astype(str)

        esci_s_path = f'{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet'
        try:
            df_esci_s = pd.read_parquet(esci_s_path)
            if 'asin' in df_esci_s.columns: df_esci_s = df_esci_s.rename(columns={'asin': 'product_id'})
            elif 'item_id' in df_esci_s.columns: df_esci_s = df_esci_s.rename(columns={'item_id': 'product_id'})
            elif 'id' in df_esci_s.columns: df_esci_s = df_esci_s.rename(columns={'id': 'product_id'})
            
            df_esci_s['product_id'] = df_esci_s['product_id'].astype(str)
            self.df_pr = pd.merge(self.df_pr, df_esci_s[['product_id', 'price', 'stars', 'ratings', 'category']], on='product_id', how='left')
            
            # Pre-compute global medians for live inference fallbacks
            self.df_pr['price'] = pd.to_numeric(self.df_pr['price'].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
            self.df_pr['stars'] = self.df_pr['stars'].astype(str).str.extract(r'([\d\.]+)').astype(float)
            self.df_pr['ratings'] = pd.to_numeric(self.df_pr['ratings'], errors='coerce')
            
            self.global_price = self.df_pr['price'].median()
            self.global_stars = self.df_pr['stars'].median()
            self.global_ratings = self.df_pr['ratings'].median()
        except FileNotFoundError:
            print("ESCI-S data not found. Live advanced reranking will fail.")

        # 3. Load Advanced Reranker (Baseline Completely Removed)
        print("[Boot] Loading Advanced Reranker (17-Feature)...")
        with open(f'{ROOT_DIR}/output/advanced_normalization_stats.json', "r") as f:
            self.adv_stats = json.load(f)
        self.adv_model = AdvancedDeepReranker(input_dim=17).to(self.device)
        self.adv_model.load_state_dict(torch.load(f'{ROOT_DIR}/output/best_advanced_reranker.pth', map_location=self.device))
        self.adv_model.eval()
        print("Engine Ready.")

    def extract_advanced_features_live(self, query_text, df):
        query_lower = str(query_text).lower()
        
        df['query_length'] = len(query_lower.split())
        match = re.search(r'under\s*\$?(\d+)', query_lower)
        user_budget = float(match.group(1)) if match else -1.0
        df['user_budget'] = user_budget
        df['cheap_intent'] = 1.0 if any(w in query_lower for w in ['cheap', 'affordable', 'budget']) else 0.0

        idf_map = self.adv_stats.get("idf_map", {})
        words = query_lower.split()
        idfs = [idf_map.get(w, 10.0) for w in words] if words else [0.0]
        df['query_mean_idf'] = np.mean(idfs)
        df['query_max_idf'] = np.max(idfs)

        df['category'] = df['category'].fillna("Unknown").astype(str)

        # Imputation
        df['is_price_missing'] = df['price'].isna().astype(float)
        cat_median_price = df.groupby('category')['price'].transform('median')
        imputed_price = df['price'].fillna(cat_median_price).fillna(self.global_price).fillna(0.0)
        df['log_price'] = np.log1p(imputed_price)

        df['is_rating_missing'] = df['stars'].isna().astype(float)
        cat_median_stars = df.groupby('category')['stars'].transform('median')
        df['stars_clean'] = df['stars'].fillna(cat_median_stars).fillna(self.global_stars).fillna(0.0)

        cat_median_ratings = df.groupby('category')['ratings'].transform('median')
        imputed_ratings = df['ratings'].fillna(cat_median_ratings).fillna(self.global_ratings).fillna(0.0)
        df['log_review_count'] = np.log1p(imputed_ratings)

        # Interactions
        stemmer = PorterStemmer()
        q_stemmed = set(stemmer.stem(w) for w in query_lower.split())
        def get_overlap(t):
            if not q_stemmed: return 0.0
            t_stemmed = set(stemmer.stem(w) for w in str(t).lower().split())
            return len(q_stemmed.intersection(t_stemmed)) / len(q_stemmed)
            
        df['word_overlap'] = df['product_title'].apply(get_overlap)
        df['is_over_budget'] = ((user_budget > 0) & (imputed_price > user_budget)).astype(float)
        df['brand_match'] = df['product_brand'].apply(lambda b: 1.0 if pd.notna(b) and str(b).lower() in query_lower else 0.0)

        COLORS = {'red', 'black', 'blue', 'white', 'green', 'yellow', 'silver', 'gold', 'grey', 'gray', 'pink', 'purple', 'brown'}
        q_colors = set(query_lower.split()).intersection(COLORS)
        df['color_match'] = df['product_title'].apply(lambda t: 1.0 if q_colors and q_colors.intersection(set(str(t).lower().split())) else 0.0)

        top_20 = df.sort_values('bm25_score', ascending=False).head(20)
        dominant_cat = top_20['category'].mode()[0] if not top_20['category'].mode().empty else "Unknown"
        df['is_dominant_category'] = (df['category'] == dominant_cat).astype(float)

        feature_cols = self.adv_stats['features'] 
        return df[feature_cols].values

    def run_search(self, query_text):
        # 1. Real Retrieval via C-Optimized Indices
        df_bm25 = search_bm25_global(self.bm25_index, self.bm25_ids, query_text)
        df_tt = search_tt_global(self.tt_model, self.tt_index, self.tt_ids, query_text)
        
        # Merge Top candidates and attach catalog text/data
        candidates = pd.merge(df_bm25, df_tt, on='product_id', how='outer').fillna(0.0)
        df_candidates = pd.merge(candidates, self.df_pr, on='product_id', how='inner')

        if df_candidates.empty:
            return pd.DataFrame()

        # 2. Dynamic Feature Extraction & PyTorch Inference
        features_raw = self.extract_advanced_features_live(query_text, df_candidates)
        features_norm = (features_raw - np.array(self.adv_stats['mean'])) / np.array(self.adv_stats['std'])
        
        with torch.no_grad():
            tensor_x = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
            scores = self.adv_model(tensor_x).cpu().squeeze().numpy()
        
        df_candidates['predicted_score'] = scores
        df_final = apply_mmr(df_candidates, lambda_param=0.6, top_n=20)

        return df_final


# ==========================================
# The GUI Application
# ==========================================
class SearchApp:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.root.title("ESCI Neural Search Engine (Advanced)")
        self.root.geometry("1100x600")

        # --- Top Frame: Search Bar & Controls ---
        top_frame = tk.Frame(root, pady=10, padx=10)
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text="Search Query:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(top_frame, textvariable=self.search_var, font=("Arial", 12), width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<Return>', lambda event: self.execute_search())

        self.search_btn = tk.Button(top_frame, text="Search", command=self.execute_search, bg="#4CAF50", fg="black", font=("Arial", 10, "bold"))
        self.search_btn.pack(side=tk.LEFT, padx=5)

        # Output label showing engine readiness
        tk.Label(top_frame, text="Neural Network Active", font=("Arial", 10, "italic"), fg="gray").pack(side=tk.RIGHT, padx=5)

        # --- Bottom Frame: Results Table ---
        bottom_frame = tk.Frame(root, pady=10, padx=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("Rank", "Score", "Brand", "Price", "Stars", "Reviews", "Category", "Title")
        self.tree = ttk.Treeview(bottom_frame, columns=columns, show="headings", height=20)

        # Define Column Headings and Widths
        col_widths = {"Rank": 40, "Score": 60, "Brand": 120, "Price": 60, "Stars": 50, "Reviews": 70, "Category": 120, "Title": 400}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths[col], anchor=tk.W if col in ["Brand", "Title", "Category"] else tk.CENTER)

        # Add Scrollbar
        scrollbar = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def execute_search(self):
        query = self.search_var.get().strip()
        if not query:
            return

        self.tree.delete(*self.tree.get_children()) # Clear old results

        try:
            results_df = self.engine.run_search(query)
            
            if results_df.empty:
                messagebox.showinfo("No Results", "No candidates found for this query.")
                return

            # Populate Treeview
            for rank, (_, row) in enumerate(results_df.iterrows(), 1):
                score = f"{row['predicted_score']:.3f}"
                brand = str(row['product_brand']) if pd.notna(row['product_brand']) else "-"
                price = f"${row['price']:.2f}" if pd.notna(row['price']) else "N/A"
                stars = f"{row['stars']:.1f}" if pd.notna(row['stars']) else "-"
                reviews = int(row['ratings']) if pd.notna(row['ratings']) else 0
                category = str(row['category']).split('>')[-1].strip()[:20] if pd.notna(row['category']) else "Unknown"
                title = str(row['product_title'])
                
                self.tree.insert("", tk.END, values=(rank, score, brand, price, stars, reviews, category, title))

        except Exception as e:
            messagebox.showerror("Search Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    engine = SearchEngine()
    root = tk.Tk()
    app = SearchApp(root, engine)
    root.mainloop()