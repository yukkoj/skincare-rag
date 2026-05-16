import os
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "chunk_embeddings.npy")
METADATA_FILE = os.path.join(DATA_DIR, "chunk_metadata.json")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_FILE = os.path.join(OUTPUTS_DIR, "search_results.json")
