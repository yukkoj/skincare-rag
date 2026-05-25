"""
Embeddings Logic (Semantic Profiles)
Converts unified product semantic profiles into vector embeddings incrementally, 
skipping ones that are already done, and builds a FAISS search index.
"""
import json
import pickle
import numpy as np
from rank_bm25 import BM25Okapi # Add this import
import faiss
from sentence_transformers import SentenceTransformer
import config

# Setup
config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_PATH = config.GENERATED_DIR / "semantic_profiles.json"
INDEX_PATH = config.EMBED_DIR / "faiss.index"
CHUNKS_PATH = config.EMBED_DIR / "chunks.pkl"
BM25_PATH = config.EMBED_DIR / "bm25.pkl"

bm25 = None
model = SentenceTransformer('all-MiniLM-L6-v2')

def save_bm25_index(bm25_model):
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25_model, f)

def load_bm25_index():
    if BM25_PATH.exists():
        with open(BM25_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def build_keyword_index(chunks):
    """Tokenize and build a BM25 index from your chunks."""
    tokenized_corpus = [c['text'].lower().split() for c in chunks]
    return BM25Okapi(tokenized_corpus)

def get_embeddings_batch(texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

def get_sentence_embeddings(query: str):
    """Convert a single string into an embedding vector."""
    # We pass [query] as a list because model.encode expects a list/iterable
    return model.encode([query], convert_to_numpy=True)

def ensure_chunk_embeddings():
    """Reads semantic profiles, creates embeddings (skipping existing), and saves."""
    existing_chunks = []
    existing_embeddings = None
    processed_ids = set()

    # 1. Load existing data
    if CHUNKS_PATH.exists():
        try:
            with open(CHUNKS_PATH, 'rb') as f:
                data = pickle.load(f)
                existing_chunks = data.get("chunks", [])
                existing_embeddings = data.get("embeddings")
                processed_ids = {c['product_id'] for c in existing_chunks}
            print(f"  ⏭️ Found {len(processed_ids)} already embedded products.")
        except Exception as e:
            print(f"  ⚠ Could not read existing chunks: {e}")

    # 2. Load master profiles
    if not PROFILES_PATH.exists():
        print(f"  ⚠ Master profiles file not found at {PROFILES_PATH}.")
        return

    with open(PROFILES_PATH, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    new_chunks = []
    for profile in profiles:
        pid = profile.get("Product_ID")
        if pid in processed_ids:
            continue
            
        semantic_data = profile.get("Semantic_Profile", {})
        # Create a rich text representation for the embedding
        full_text = (f"Product: {semantic_data.get('Product_Name', 'Unknown')}. "
                     f"Specs: {json.dumps(semantic_data)}. "
                     f"Consensus: {semantic_data.get('Customer_Consensus', '')}")
        
        new_chunks.append({"product_id": pid, "type": "semantic_profile", "text": full_text})

    # 3. Process new data only
    if not new_chunks:
        print("  ✓ No new data to process.")
        return

    print(f"  Vectorizing {len(new_chunks)} new products...")
    new_vectors = get_embeddings_batch([c['text'] for c in new_chunks])
    
    # 4. Merge and Save
    final_chunks = existing_chunks + new_chunks
    final_embeddings = np.vstack((existing_embeddings, new_vectors)) if existing_embeddings is not None else new_vectors
    
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump({"chunks": final_chunks, "embeddings": final_embeddings}, f)
    print(f"  ✓ Saved {len(final_chunks)} total profiles to {CHUNKS_PATH.name}")

def build_faiss_index():
    """Takes the embedded profiles and builds a searchable FAISS database."""
    if not CHUNKS_PATH.exists():
        print("  ⚠ Chunks file not found. Run ensure_chunk_embeddings first.")
        return
        
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
        
    embeddings = np.array(data["embeddings"]).astype('float32')
    
    if len(embeddings) == 0:
        print("  ⚠ No embeddings to index.")
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    faiss.write_index(index, str(INDEX_PATH))
    print(f"  ✓ Built FAISS index with {index.ntotal} vectors at {INDEX_PATH.name}")
# ==========================================
# SEARCH FUNCTIONS
# ==========================================

def search_products(query: str, k: int = 20):
    global bm25
    
    # 1. Lazy load indices
    if bm25 is None:
        bm25 = load_bm25_index() # Load pre-saved BM25
        
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    
    # 2. Parallel-style lookup
    query_vector = get_sentence_embeddings(query).astype('float32')
    index = faiss.read_index(str(INDEX_PATH))
    
    # 3. Vector Search
    v_distances, v_indices = index.search(query_vector, k)
    
    # 4. Keyword Search
    tokenized_query = query.lower().split()
    k_scores = bm25.get_scores(tokenized_query)
    
    # 5. Hybrid Fusion (RRF)
    combined_scores = {}
    
    # Vector: lower distance is better, but we need to normalize to rank
    # FAISS returns distances, we convert to rank
    for i, idx in enumerate(v_indices[0]):
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (i + 1))
        
    # Keyword: Higher BM25 score is better
    k_indices = np.argsort(k_scores)[::-1][:k]
    for i, idx in enumerate(k_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (i + 1))
        
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [ {**chunks[idx], 'distance': score} for idx, score in sorted_indices[:k] ]

def aggregate_chunk_hits(results):
    """Aggregates raw FAISS results into unique product IDs."""
    aggregated = {}
    for res in results:
        pid = res['product_id']
        score = res['distance'] # Note: this is a score
        # Keep the LARGEST score (higher is better in RRF)
        if pid not in aggregated or score > aggregated[pid]:
            aggregated[pid] = score
            
    # Sort by score descending (highest first)
    return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

# ==========================================
# TEST EXECUTION
# ==========================================

if __name__ == "__main__":
    # To build the database for the first time, uncomment these lines:
    # ensure_chunk_embeddings()
    # build_faiss_index()
    
    # Testing search functionality:
    print("\n--- Testing Search ---")
    results = search_products("I need a cheap, fragrance-free lotion that won't clog my pores.")
    for rank, prod in enumerate(results, 1):
        print(f"#{rank}: {prod['product_id']} (Distance: {prod['distance']:.2f})")