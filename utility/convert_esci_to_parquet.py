import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR
file_path = f"{ROOT_DIR}/esci-data/esci-s_dataset/esci.json.zst"
out_path = f"{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet"
chunksize = 100_000 # Only load 100k rows into RAM at a time

print("Loading data in chunks to prevent memory crash...")

# Create an iterator that yields chunks of the dataframe
reader = pd.read_json(file_path, lines=True, compression="zstd", chunksize=chunksize)

parquet_writer = None

for i, chunk in enumerate(reader):
    print(f"Processing chunk {i+1}...")
    
    # Convert the pandas chunk to a PyArrow Table
    table = pa.Table.from_pandas(chunk)
    
    # Initialize the writer on the very first chunk so it learns the schema (columns/types)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(out_path, table.schema)
        
    # Append the chunk to the file
    parquet_writer.write_table(table)

# Close the file connection
if parquet_writer:
    parquet_writer.close()

print(f"Successfully saved full dataset to {out_path}!")