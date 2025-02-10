import ray
import time
import random
import pandas as pd

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Load dataset correctly
file_path = "D:\\February_task\\Elastic_search\\customers.csv"
documents_df = pd.read_csv(file_path)

# Add a unique ID column if missing
documents_df["id"] = documents_df.index.astype(str)  # Use row index as ID

# Convert DataFrame to dictionary
documents = {row["id"]: row.to_dict() for _, row in documents_df.iterrows()}
print(f"Loaded {len(documents)} documents")

@ray.remote
def retrieve_from_dict(doc_id):
    """Retrieve document from in-memory dictionary"""
    start = time.time()
    result = documents.get(doc_id, "Not Found")
    return time.time() - start  # Return retrieval time

# Select random IDs to retrieve (handle small datasets)
num_queries = min(10, len(documents))  # Avoid error if dataset is small
query_ids = random.sample(list(documents.keys()), num_queries)

# Run parallel retrieval
start_time = time.time()
future_results = [retrieve_from_dict.remote(doc_id) for doc_id in query_ids]
ray_results = ray.get(future_results)
end_time = time.time()

print(f"Ray (in-memory) Retrieval Time: {end_time - start_time:.6f} sec for {num_queries} queries")
