"""
Core Embeddings Logic
Converts product summaries and reviews into vector embeddings incrementally, 
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
    """Reads summaries and reviews, chunks them, and creates embeddings (skipping existing)."""
    existing_chunks = []
    existing_embeddings = None
    processed_product_ids = set()

    # --- NEW: Check what we already processed ---
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
    
    # Look for ANY json file in the summaries folder
    summary_files = list(config.SUMMARIES_DIR.glob("*.json"))
    if not summary_files:
        print("  ⚠ No summary files found to embed. Run the summarizer first.")
        return

    for summary_file in summary_files:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            
        product_id = summary.get('product_id', 'unknown')
        
        # --- NEW: Skip if we already have it ---
        if product_id in processed_product_ids:
            continue
            
        summary_text = (
            f"Product: {product_id}. "
            f"Overall Sentiment: {summary.get('llm_summary', '')}. "
            f"Key themes: {', '.join([p['phrase'] for p in summary.get('top_phrases', [])])}"
        )
        new_chunks.append({
            "product_id": product_id,
            "type": "summary",
            "text": summary_text
        })
        
        review_file = config.REVIEWS_DIR / f"{product_id}.json"
        if review_file.exists():
            with open(review_file, 'r', encoding='utf-8') as rf:
                reviews = json.load(rf)
                
                top_reviews = sorted(reviews, key=lambda x: x.get('score', 0), reverse=True)[:10]
                
                for rev in top_reviews:
                    rev_text = (
                        f"Review for {product_id}: "
                        f"{rev.get('title', '')} - {rev.get('text', '')}"
                    )
                    new_chunks.append({
                        "product_id": product_id,
                        "type": "review",
                        "text": rev_text,
                        "url": rev.get('url', '')
                    })

    # --- NEW: Only vectorize the new ones and merge them ---
    if not new_chunks:
        print("  ✓ All products are already embedded! No new data to process.")
        return

    print(f"  Generated {len(new_chunks)} NEW text chunks. Vectorizing now...")
    
    texts = [chunk['text'] for chunk in new_chunks]
    new_embedded_vectors = get_embeddings_batch(texts)
    
    # Combine old chunks with new chunks
    final_chunks = existing_chunks + new_chunks
    
    # Combine old vectors with new vectors
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        final_embeddings = np.vstack((existing_embeddings, new_embedded_vectors))
    else:
        final_embeddings = new_embedded_vectors
    
    # Save everything back to the master file
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump({"chunks": final_chunks, "embeddings": final_embeddings}, f)
        
    print(f"  ✓ Saved {len(final_chunks)} total embedded chunks to {CHUNKS_PATH.name}")

def build_faiss_index():
    """Takes the embedded chunks and builds a searchable FAISS database."""
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

def search_faiss_index(query_embedding, index_path=INDEX_PATH, k=5):
    index = faiss.read_index(str(index_path))
    query_embedding = query_embedding.astype('float32')
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def find_similar_chunks(query, k=5):
    query_vector = get_sentence_embeddings(query)
    distances, indices = search_faiss_index(query_vector, k=k)
    
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            result['distance'] = float(distances[0][i])
            results.append(result)
            
    return results

def aggregate_chunk_hits(results):
    product_scores = {}
    for r in results:
        pid = r['product_id']
        score = 1.0 / (r['distance'] + 0.1) 
        if pid not in product_scores:
            product_scores[pid] = 0
        product_scores[pid] += score
        
    return sorted(product_scores.items(), key=lambda x: x[1], reverse=True)