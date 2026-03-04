import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import warnings

# Suppress huggingface warnings for cleaner output
warnings.filterwarnings("ignore")

class ConceptEntropyExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initializes the frozen Transformer model specifically for extracting attention.
        DistilBERT is perfect here because it is fast and lightweight.
        """
        print(f"Loading {model_name} for Entropy Extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # CRITICAL: We must request output_attentions=True
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval() # Freeze the model - we are not training this!

    def calculate_entropy(self, query):
        """
        Calculates the normalized attention entropy of a given query.
        Returns a float between 0.0 (Hyper-specific) and 1.0 (Completely vague).
        """
        # 1. Tokenize the query
        inputs = self.tokenizer(query, return_tensors="pt")
        
        # 2. Forward pass (No gradients needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. Extract Attention from the LAST layer
        # outputs.attentions is a tuple of all layers. We want the last one [-1].
        # Shape: [Batch_Size, Num_Heads, Seq_Length, Seq_Length]
        last_layer_attention = outputs.attentions[-1]
        
        # 4. Average the attention across all attention heads
        # Shape becomes: [Seq_Length, Seq_Length]
        avg_attention = torch.mean(last_layer_attention, dim=1).squeeze(0)
        
        # 5. Isolate the [CLS] token's attention (Index 0)
        # This represents how the "overall sentence meaning" attends to individual words
        cls_attention = avg_attention[0, :]
        
        # 6. Strip out the artificial [CLS] and [SEP] tokens for pure word-entropy
        # cls_attention[0] is [CLS], cls_attention[-1] is [SEP]
        word_probs = cls_attention[1:-1]
        
        # 7. Edge Case Handling: 1-word queries
        # If there is only 1 word, it gets 100% attention, so entropy is mathematically 0.
        # We can default to 0.0 (Specific) or handle it as a special case.
        if len(word_probs) <= 1:
            return 0.0 
            
        # 8. Re-normalize probabilities so they sum to 1.0 after removing CLS/SEP
        word_probs = word_probs / word_probs.sum()
        
        # 9. Calculate Shannon Entropy
        # Add a tiny epsilon (1e-9) to prevent log(0) errors
        entropy = -torch.sum(word_probs * torch.log(word_probs + 1e-9)).item()
        
        # 10. Normalize by the maximum possible entropy (log of sequence length)
        # This ensures our feature is always scaled between 0.0 and 1.0
        max_entropy = np.log(len(word_probs))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy

# ==========================================
# Testing the Extractor
# ==========================================
if __name__ == "__main__":
    extractor = ConceptEntropyExtractor()

    # Test Case 1: Highly Specific (Expect Low Entropy)
    specific_query = "Nike Air Zoom Pegasus running shoes mens size 10 red"
    ent_specific = extractor.calculate_entropy(specific_query)
    
    # Test Case 2: Vague/Broad (Expect High Entropy)
    vague_query = "cool comfortable things to wear outside"
    ent_vague = extractor.calculate_entropy(vague_query)
    
    print("\n--- Results ---")
    print(f"Specific Query: '{specific_query}'")
    print(f"Entropy Score : {ent_specific:.4f}\n")
    
    print(f"Vague Query   : '{vague_query}'")
    print(f"Entropy Score : {ent_vague:.4f}\n")
    
    print(f"Difference: The vague query is {ent_vague / ent_specific:.1f}x more 'confusing' to the model.")