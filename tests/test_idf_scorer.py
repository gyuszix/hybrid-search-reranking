import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

class QuerySpecificityScorer:
    def __init__(self):
        # We drop English stop words because they add noise to specificity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.is_fitted = False
        
    def fit_corpus(self, list_of_all_queries):
        print(f"Learning vocabulary and IDF scores from {len(list_of_all_queries)} training queries...")
        self.vectorizer.fit(list_of_all_queries)
        
        # Map words to their IDF scores
        self.idf_dict = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))
        self.max_idf = max(self.vectorizer.idf_)
        self.is_fitted = True

    def calculate_specificity(self, query):
        if not self.is_fitted:
            raise ValueError("You must call fit_corpus() before calculating scores.")
            
        analyzer = self.vectorizer.build_analyzer()
        words = analyzer(str(query))
        
        # Handle empty queries or queries made entirely of stop words (e.g., "to the")
        if not words:
            return 0.0 
            
        # Get the rarity score for each word
        word_scores = [self.idf_dict.get(word, self.max_idf) for word in words]
        
        # Use Top-2 Rarest Words to prevent dilution from common words
        word_scores.sort(reverse=True)
        top_k_scores = word_scores[:2] 
        
        return np.mean(top_k_scores)


def run_idf_evaluation(train_csv_path, test_csv_path):
    print("Loading datasets...")
    # Load just the queries to save memory
    train_df = pd.read_csv(train_csv_path, usecols=['query'])
    test_df = pd.read_csv(test_csv_path, usecols=['query'])
    
    # Get unique queries (no need to score the same query 40 times)
    train_queries = train_df['query'].dropna().unique().tolist()
    test_queries = test_df['query'].dropna().unique().tolist()
    
    # 1. Initialize and Fit the Scorer on TRAINING data
    scorer = QuerySpecificityScorer()
    scorer.fit_corpus(train_queries)
    
    # 2. Score the TEST data
    print(f"\nScoring {len(test_queries)} unseen test queries...")
    results = []
    for q in test_queries:
        score = scorer.calculate_specificity(q)
        results.append({
            'query': q, 
            'specificity_score': score,
            'word_count': len(str(q).split())
        })
        
    # 3. Analyze the Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='specificity_score', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 5 MOST SPECIFIC QUERIES (High Score)")
    print("Notice how these contain obscure brands or exact part numbers.")
    print("="*50)
    print(results_df.head(5).to_string(index=False))
    
    print("\n" + "="*50)
    print("TOP 5 MOST VAGUE QUERIES (Low Score)")
    print("Notice how these are common, broad, generic categories.")
    print("="*50)
    print(results_df.tail(5).to_string(index=False))
    
    # 4. Save the dictionary map for the Reranker
    # This creates a lookup table so your PyTorch dataset can grab scores instantly
    results_df.to_csv(f'{ROOT_DIR}/output/IDF/test_queries_idf_scored.csv', index=False)
    print("\nSaved scores to 'test_queries_scored.csv'")
    
    return results_df

if __name__ == "__main__":
    train_csv_path = f'{ROOT_DIR}/output/IDF/esci_train_clean.csv'
    test_csv_path = f'{ROOT_DIR}/output/IDF/esci_test_clean.csv'

    try:
        scored_test_data = run_idf_evaluation(train_csv_path, test_csv_path)
    except FileNotFoundError:
        print("CSV files not found. Make sure you ran 'idf_setup.py' first!")