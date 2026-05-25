"""
Embeddings Logic (Semantic Profiles)
Converts unified product semantic profiles into vector embeddings incrementally, 
skipping ones that are already done, and builds a FAISS search index.
"""
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config

# Setup Directories from config
config.EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Define paths
PROFILES_PATH = config.GENERATED_DIR / "semantic_profiles.json"
INDEX_PATH = config.EMBED_DIR / "faiss.index"
CHUNKS_PATH = config.EMBED_DIR / "chunks.pkl"

# Load the local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_embeddings(texts):
    """Convert a single string or list of strings into embeddings"""
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, convert_to_numpy=True)

def get_embeddings_batch(texts, batch_size=32):
    """Process large amounts of text in batches"""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

def ensure_chunk_embeddings():
    """Reads semantic profiles, creates embeddings (skipping existing), and saves."""
    existing_chunks = []
    existing_embeddings = None
    processed_product_ids = set()

    # --- Check what we already processed ---
    if CHUNKS_PATH.exists():
        try:
            with open(CHUNKS_PATH, 'rb') as f:
                data = pickle.load(f)
                existing_chunks = data.get("chunks", [])
                existing_embeddings = data.get("embeddings")
                
                for chunk in existing_chunks:
                    processed_product_ids.add(chunk['product_id'])
                    
            print(f"  ⏭️ Found {len(processed_product_ids)} already embedded products. Skipping those...")
        except Exception as e:
            print(f"  ⚠ Could not read existing chunks, starting fresh. ({e})")
            existing_chunks = []
            existing_embeddings = None
            processed_product_ids = set()

    new_chunks = []
    
    # --- Load the single unified profiles file ---
    if not PROFILES_PATH.exists():
        print(f"  ⚠ Master profiles file not found at {PROFILES_PATH}. Run the profile generator first.")
        return

    with open(PROFILES_PATH, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # --- Iterate through the unified profiles ---
    for profile in profiles:
        product_id = profile.get("Product_ID")
        semantic_text = profile.get("Semantic_Text", "")
        
        # Skip if we already have it
        if product_id in processed_product_ids:
            continue
            
        # Treat the entire semantic profile as one single chunk
        new_chunks.append({
            "product_id": product_id,
            "type": "semantic_profile",
            "text": semantic_text
        })

    # --- Only vectorize the new ones and merge them ---
    if not new_chunks:
        print("  ✓ All products are already embedded! No new data to process.")
        return

    print(f"  Generated {len(new_chunks)} NEW semantic profile chunks. Vectorizing now...")
    
    texts = [chunk['text'] for chunk in new_chunks]
    new_embedded_vectors = get_embeddings_batch(texts)
    
    # Combine old chunks with new chunks
    final_chunks = existing_chunks + new_chunks
    
    # Combine old vectors with new vectors
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        final_embeddings = np.vstack((existing_embeddings, new_embedded_vectors))
    else:
        final_embeddings = new_embedded_vectors
    
    # Save everything back to the master pickle file
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump({"chunks": final_chunks, "embeddings": final_embeddings}, f)
        
    print(f"  ✓ Saved {len(final_chunks)} total embedded profiles to {CHUNKS_PATH.name}")

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

def search_products(query: str, k: int = 5):
    """
    Takes a user's text query, embeds it, searches the FAISS index, 
    and returns the top K matching product profiles.
    """
    # 1. Ensure the index exists
    if not INDEX_PATH.exists():
        print("⚠ FAISS index not found. Run build_faiss_index() first.")
        return []

    # 2. Embed the user's search query
    query_vector = get_sentence_embeddings(query)
    
    # 3. Search the FAISS index
    index = faiss.read_index(str(INDEX_PATH))
    query_vector = query_vector.astype('float32')
    distances, indices = index.search(query_vector, k)
    
    # 4. Map the mathematical results back to the text profiles
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    
    results = []
    # indices[0] contains the top k matches
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            # Distance: Lower is better (closer match)
            result['distance'] = float(distances[0][i]) 
            results.append(result)
            
    return results

# --- Example of how to use it ---
if __name__ == "__main__":
    # Uncomment these to build the database for the first time:
    ensure_chunk_embeddings()
    build_faiss_index()
    
    # Try a search!
    # print("\n--- Testing Search ---")
    # top_products = search_products("I need a cheap, fragrance-free lotion that won't clog my pores.")
    # for rank, prod in enumerate(top_products, 1):
    #     print(f"#{rank}: {prod['product_id']} (Distance: {prod['distance']:.2f})")