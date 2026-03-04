import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.two_tower import compute_two_tower_scores


# ----------------------
# Shared test data
# ----------------------

SAMPLE_DF = pd.DataFrame({
    "query_id":   [1, 1, 1, 2, 2],
    "query_text": [
        "running shoes",
        "running shoes",
        "running shoes",
        "laptop stand",
        "laptop stand"
    ],
    "item_id": [101, 102, 103, 201, 202],
    "item_text": [
        "Nike Air Zoom lightweight running shoe for seniors",
        "Coffee maker with grinder",
        "Trail running shoes waterproof",
        "Adjustable aluminum laptop stand",
        "Wooden cutting board",
    ]
})


def get_results():
    return compute_two_tower_scores(SAMPLE_DF)


# ----------------------
# Tests
# ----------------------

def test_output_columns():
    """Output must have exactly these three columns"""
    result = get_results()
    assert set(result.columns) == {"query_id", "item_id", "two_tower_score"}, \
        f"Unexpected columns: {result.columns.tolist()}"


def test_output_not_empty():
    """Result should not be empty"""
    result = get_results()
    assert not result.empty, "Result is empty"


def test_scores_normalized():
    """All scores must be in [0, 1]"""
    result = get_results()
    assert result["two_tower_score"].between(0, 1).all(), \
        f"Scores out of [0,1] range:\n{result}"


def test_all_queries_present():
    """Both query_ids should appear in the output"""
    result = get_results()
    assert set(result["query_id"]) == {1, 2}, \
        f"Missing queries in output: {set(result['query_id'])}"


def test_all_items_present():
    """All item_ids should appear in the output"""
    result = get_results()
    assert set(result["item_id"]) == {101, 102, 103, 201, 202}, \
        f"Missing items in output: {set(result['item_id'])}"


def test_semantic_relevance():
    """
    Relevant items should score higher than irrelevant ones.
    Query 1: 'running shoes' — item 101 and 103 should outscore item 102 (coffee maker)
    Query 2: 'laptop stand'  — item 201 should outscore item 202 (cutting board)
    """
    result = get_results()

    q1 = result[result["query_id"] == 1].set_index("item_id")["two_tower_score"]
    assert q1[101] > q1[102], "Running shoe should score higher than coffee maker"
    assert q1[103] > q1[102], "Trail running shoe should score higher than coffee maker"

    q2 = result[result["query_id"] == 2].set_index("item_id")["two_tower_score"]
    assert q2[201] > q2[202], "Laptop stand should score higher than cutting board"


def test_no_duplicate_items_per_query():
    """Each item_id should appear only once per query"""
    result = get_results()
    dupes = result.groupby(["query_id", "item_id"]).size()
    assert (dupes == 1).all(), f"Duplicate items found:\n{dupes[dupes > 1]}"


def test_missing_columns_raises():
    """Should raise ValueError if required columns are missing"""
    bad_df = SAMPLE_DF.drop(columns=["item_text"])
    try:
        compute_two_tower_scores(bad_df)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_output_columns,
        test_output_not_empty,
        test_scores_normalized,
        test_all_queries_present,
        test_all_items_present,
        test_semantic_relevance,
        test_no_duplicate_items_per_query,
        test_missing_columns_raises,
    ]

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")