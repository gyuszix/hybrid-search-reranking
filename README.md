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

## 2nd Iteration of our Architecture

<img width="716" height="663" alt="Screenshot 2026-03-03 at 19 06 37" src="https://github.com/user-attachments/assets/7d3e1622-e48d-49af-8927-7a18a5aff9ae" />

<img width="716" height="608" alt="Screenshot 2026-03-03 at 19 06 28" src="https://github.com/user-attachments/assets/e45cca65-5c2f-4667-a54c-c5229e151de5" />

<img width="716" height="661" alt="Screenshot 2026-03-03 at 19 06 23" src="https://github.com/user-attachments/assets/ec700fcc-558f-408d-9073-03e7286dfac1" />

<img width="716" height="661" alt="Screenshot 2026-03-03 at 19 06 02" src="https://github.com/user-attachments/assets/2215ddbf-5c7f-4351-a4a0-19f1d672f0f6" />



## Project Structure

```
.
├── retrieval/          # First-stage retrieval (BM25 + Two-Tower)
│   ├── bm25.py         # BM25 lexical scoring and index building
│   └── two_tower.py    # Two-tower semantic scoring and index building
├── reranking/          # Second-stage neural reranking
│   ├── model.py        # DeepESCIReranker architecture
│   └── features.py     # Feature extraction + PairwiseESCIDataset
├── evaluation/         # Metrics and evaluation
│   ├── metrics.py      # NDCG@K, DCG, Recall@K
│   ├── evaluate_retrieval.py
│   └── evaluate_reranker.py
├── analysis/           # Query complexity analysis
│   ├── concept_entropy.py
│   └── idf_setup.py
├── scripts/            # All runnable entry points
│   ├── run_pipeline.py             # Main pipeline (load, score, evaluate)
│   ├── build_indices.py            # Build BM25 + Two-Tower indices
│   ├── generate_bm25_scores.py     # Generate BM25 scores for train/test
│   ├── generate_two_tower_scores.py # Generate Two-Tower scores for train/test
│   ├── train_two_tower.py          # Fine-tune the Two-Tower model
│   └── train_reranker.py           # Train the neural reranker
├── tests/
├── models/             # Saved model weights (gitignored)
├── output/             # Generated scores, predictions, indices (gitignored)
├── esci-data/          # Amazon ESCI dataset (gitignored)
└── config.py           # Global paths and settings
```

## How to run

Everything is run from the project root. Follow these steps in order:

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Fine-tune the Two-Tower model** (only needed once)
```bash
python scripts/train_two_tower.py
```
This trains the semantic encoder on ESCI data and saves weights to `models/two_tower_finetuned/`.

**Step 3 — Generate retrieval scores** (only needed once, or when data changes)
```bash
python scripts/generate_bm25_scores.py
python scripts/generate_two_tower_scores.py
```
These produce `output/bm25_scores_{split}.csv` and `output/two_tower_scores_{split}.csv` for train and test.

**Step 4 — Build search indices** (only needed once)
```bash
python scripts/build_indices.py
```
Pre-computes BM25 and FAISS indices for fast retrieval at search time.

**Step 5 — Train the reranker**
```bash
python scripts/train_reranker.py
```
Trains the neural reranker using the scores from Step 3. Saves the best model to `output/best_esci_reranker.pth`.

**Step 6 — Evaluate**
```bash
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_reranker.py
```
Prints NDCG@10 and Recall@K for the retrieval baselines and the reranker.

**Step 7 — Run the full pipeline** (optional)
```bash
python scripts/run_pipeline.py
```

**Try an interactive search** (requires Steps 2-5 completed)
```bash
python tests/test_custom_search.py
```
Type a query and see ranked results from the full pipeline.

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

## Recent restructuring

The project folder structure was reorganized to make navigation easier. No logic was changed — only file locations and imports.

### What moved where

| Old location | New location | Why |
|---|---|---|
| `signals/bm25.py`, `signals/two_tower.py` | `retrieval/bm25.py`, `retrieval/two_tower.py` | "Signals" was vague — these are retrieval modules |
| `signals/generate_bm25_scores.py` | `scripts/generate_bm25_scores.py` | Runnable scripts belong together |
| `signals/generate_two_tower_scores.py` | `scripts/generate_two_tower_scores.py` | Same |
| `signals/train_two_tower.py` | `scripts/train_two_tower.py` | Same |
| `main.py` | `scripts/run_pipeline.py` | Keep root clean, scripts in one place |
| `build_search_engine_indices.py` | `scripts/build_indices.py` | Same |
| `query_complexity_metric/` | `analysis/` | Shorter name, added `__init__.py` |

### `baseline_reranker.py` split into `reranking/`

The 317-line monolith contained the model class, feature extraction, dataset class, and training loop all in one file. It's now three files:

- **`reranking/model.py`** — `DeepESCIReranker` class (single source of truth)
- **`reranking/features.py`** — `extract_esci_features()` + `PairwiseESCIDataset`
- **`scripts/train_reranker.py`** — the training loop

`DeepESCIReranker` was previously copy-pasted in `baseline_reranker.py`, `evaluation/evaluate_reranker.py`, and `tests/test_custom_search.py`. Now they all import from `reranking.model`.

### Import cleanup

- All `from signals.` imports updated to `from retrieval.`
- Old relative imports (e.g. `from bm25 import`, `from metrics import`) changed to absolute (e.g. `from retrieval.bm25 import`, `from evaluation.metrics import`)
- `sys.path` hacks standardized and placed before project imports in every file that needs them
