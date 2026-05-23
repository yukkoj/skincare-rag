"""
Core Embeddings Logic
Converts product summaries and reviews into vector embeddings and builds a FAISS search index.
"""
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import config

# Setup Directories from config
config.EMBED_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = config.EMBED_DIR / "faiss.index"
CHUNKS_PATH = config.EMBED_DIR / "chunks.pkl"

# Load the local embedding model (downloads automatically the first time, very fast)
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_sentence_embeddings(texts):
    """Convert a single string or list of strings into embeddings"""
    if isinstance(texts, str):
        texts = [texts]
    # return as numpy array for FAISS
    return model.encode(texts, convert_to_numpy=True)


def get_embeddings_batch(texts, batch_size=32):
    """Process large amounts of text in batches"""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def ensure_chunk_embeddings():
    """Reads summaries and reviews, chunks them, and creates embeddings."""
    chunks = []
    
    # Check if we have summaries to process
    summary_files = list(config.SUMMARIES_DIR.glob("*_summary.json"))
    print(f"  DEBUG: Found {len(summary_files)} files in {config.SUMMARIES_DIR}")

    if not summary_files:
        print("  ⚠ No summary files found to embed. Run the summarizer first.")
        return

    for summary_file in summary_files:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            
        product_id = summary.get('product_id', 'unknown')
        
        # --- CHUNK 1: The Product Summary ---
        # This makes the product searchable by its overall vibe and themes
        summary_text = (
            f"Product: {product_id}. "
            f"Overall Sentiment: {summary.get('llm_summary', '')}. "
            f"Key themes: {', '.join([p['phrase'] for p in summary.get('top_phrases', [])])}"
        )
        chunks.append({
            "product_id": product_id,
            "type": "summary",
            "text": summary_text
        })
        
        # --- CHUNKS 2+: The Raw Reviews ---
        # We also embed the raw reviews so users can search for highly specific complaints/benefits
        review_file = config.REVIEWS_DIR / f"{product_id}.json"
        if review_file.exists():
            with open(review_file, 'r', encoding='utf-8') as rf:
                reviews = json.load(rf)
                
                # Only take the top 10 most helpful reviews so we don't bloat the database
                top_reviews = sorted(reviews, key=lambda x: x.get('score', 0), reverse=True)[:10]
                
                for rev in top_reviews:
                    rev_text = (
                        f"Review for {product_id}: "
                        f"{rev.get('title', '')} - {rev.get('text', '')}"
                    )
                    chunks.append({
                        "product_id": product_id,
                        "type": "review",
                        "text": rev_text,
                        "url": rev.get('url', '')
                    })

    print(f"  Generated {len(chunks)} text chunks. Vectorizing now...")
    
    # Extract just the text strings to feed into the model
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = get_embeddings_batch(texts)
    
    # Save the chunks (metadata) and their embeddings together
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
        
    print(f"  ✓ Saved {len(chunks)} embedded chunks to {CHUNKS_PATH.name}")


def build_faiss_index():
    """Takes the embedded chunks and builds a searchable FAISS database."""
    if not CHUNKS_PATH.exists():
        print("  ⚠ Chunks file not found. Run ensure_chunk_embeddings first.")
        return
    
    # load the data properly
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
        
    embeddings = np.array(data["embeddings"]).astype('float32')
    
    if len(embeddings) == 0:
        print("  ⚠ No embeddings to index.")
        return

    # Determine the dimensionality of our vectors (MiniLM uses 384 dimensions)
    dim = embeddings.shape[1]
    
    # Initialize the FAISS index (L2 distance is standard for similarity)
    index = faiss.IndexFlatL2(dim)
    
    # Add the vectors to the index
    index.add(embeddings)
    
    # Save the index to disk
    faiss.write_index(index, str(INDEX_PATH))
    
    print(f"  ✓ Built FAISS index with {index.ntotal} vectors at {INDEX_PATH.name}")


# ==========================================
# SEARCH FUNCTIONS (For your future Search script)
# ==========================================

def search_faiss_index(query_embedding, index_path=INDEX_PATH, k=5):
    """Low-level function to search the index"""
    index = faiss.read_index(str(index_path))
    query_embedding = query_embedding.astype('float32')
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def find_similar_chunks(query, k=5):
    """Finds the most relevant chunks for a text query"""
    # 1. Embed the user's search query
    query_vector = get_sentence_embeddings(query)
    
    # 2. Search FAISS
    distances, indices = search_faiss_index(query_vector, k=k)
    
    # 3. Look up the metadata for the winning vectors
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
    """Groups multiple chunk hits by product to figure out the best overall product"""
    product_scores = {}
    
    for r in results:
        pid = r['product_id']
        # Lower distance means closer/better match.
        # We invert the distance to create a positive score.
        score = 1.0 / (r['distance'] + 0.1) 
        
        if pid not in product_scores:
            product_scores[pid] = 0
        product_scores[pid] += score
        
    # Return products sorted by highest aggregate score
    return sorted(product_scores.items(), key=lambda x: x[1], reverse=True)