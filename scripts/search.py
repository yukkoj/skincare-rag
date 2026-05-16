import os
import json
from typing import List
import config
from scripts import embeddings
from scripts.data_manager import load_products, save_json

def format_results(results: List[dict]) -> List[dict]:
    return results

def run_query(query: str, top_k: int = 5, price_filter: float = None):
    # Ensure embeddings exist (delegates to existing implementation)
    embeddings.ensure_chunk_embeddings(config.PRODUCTS_FILE, config.EMBEDDINGS_FILE, config.METADATA_FILE)
    # build index and search
    index = embeddings.build_faiss_index(config.EMBEDDINGS_FILE)
    hits = embeddings.search_faiss_index(index, query, top_k=top_k)
    # hits are (idx, score) pairs or similar; fall back to cluster function if present
    try:
        chunks = embeddings.find_similar_chunks(query, top_k=top_k, embeddings_file=config.EMBEDDINGS_FILE)
    except Exception:
        chunks = hits
    results = embeddings.aggregate_chunk_hits(chunks, config.PRODUCTS_FILE)
    if price_filter is not None:
        results = [r for r in results if r.get("Price_USD") is not None and r.get("Price_USD") <= price_filter]
    # save outputs
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    save_json(config.RESULTS_FILE, results)
    return results

def run_cli():
    print("Simple search CLI (type 'exit' to quit)")
    while True:
        q = input("Query> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        results = run_query(q)
        for i, r in enumerate(results[:10], 1):
            print(f"{i}. {r.get('Product_Name')} — ${r.get('Price_USD')} ({r.get('Brand')})")
