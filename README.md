# Hybrid Search Reranking

Hybrid product search system combining keyword matching (BM25) and semantic understanding (Two-Tower BERT) with a learned reranker.

## What it does

User searches "running shoes for seniors" → returns the 48 most relevant products from 2.6M items.

## How it works

1. **BM25** - finds products containing the search words
2. **Two-Tower** - finds products with similar meaning (even without exact word matches)
3. **Merge** - combines both result sets
4. **Rerank** - neural network picks the best 48

## Why both?

| BM25 wins | Neural wins |
|-----------|-------------|
| "iPhone 15 Pro Max" | "gift for outdoorsy dad" |
| "SKU-12345" | "comfy WFH chair" |

## Tech stack

- Python, PyTorch
- BERT (embeddings)
- FAISS (fast vector search)
- rank_bm25 (keyword search)

## Team

Gyula Planky, Jian Gao, Hyuk Jin Chung


## Preliminary Architecture
<img width="8192" height="6999" alt="Untitled Diagram-2026-02-18-013721" src="https://github.com/user-attachments/assets/2c2ea3c6-ed8a-48fa-a84b-04c393847488" />


# Hybrid Search Reranking

Hybrid product search system combining keyword matching (BM25) and semantic understanding (Two-Tower BERT) with a learned reranker.

## What it does

User searches "running shoes for seniors" → returns the 48 most relevant products from 2.6M items.

## How it works

1. **BM25** - finds products containing the search words
2. **Two-Tower** - finds products with similar meaning (even without exact word matches)
3. **Merge** - combines both result sets
4. **Rerank** - neural network picks the best 48

## Why both?

| BM25 wins | Neural wins |
|-----------|-------------|
| "iPhone 15 Pro Max" | "gift for outdoorsy dad" |
| "SKU-12345" | "comfy WFH chair" |

## Tech stack

- Python, PyTorch
- BERT (embeddings)
- FAISS (fast vector search)
- rank_bm25 (keyword search)

## Team

Gyula Planky, Jian Gao, Hyuk Jin Chung


## Preliminary Architecture
<img width="8192" height="6999" alt="Untitled Diagram-2026-02-18-013721" src="https://github.com/user-attachments/assets/2c2ea3c6-ed8a-48fa-a84b-04c393847488" />

## Tests

Run from the project root:

```bash
python3 -m tests.test_two_tower
```

The test suite validates the two-tower retrieval logic against a small synthetic dataset without requiring real data or GPU. It checks:

- **Output shape** — result contains exactly the three expected columns (`query_id`, `item_id`, `two_tower_score`)
- **No empty results** — the function returns at least one row
- **Score normalization** — all scores are in [0, 1]
- **Query coverage** — every query in the input appears in the output
- **Item coverage** — every item in the input appears in the output
- **Semantic relevance** — the model ranks semantically relevant items higher than irrelevant ones (e.g. "running shoes" scores higher than "coffee maker" for a running shoes query)
- **No duplicates** — each item appears only once per query
- **Input validation** — passing a dataframe with missing required columns raises a `ValueError`