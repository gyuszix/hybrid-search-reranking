# config.py

EXAMPLES_PATH = "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
PRODUCTS_PATH = "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"

# True is 336K vs False is 1.9Mn Rows
USE_SMALL_VERSION = False
# Use Test for testing and Train for when we have the entire architecture
USE_SPLIT = "test"

# Number of top candidates to keep per query
TOP_K = 150
