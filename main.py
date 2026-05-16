import os
import re
import json
import argparse
from typing import Dict

import embeddings


def parse_price_filter(query: str):
    if not query:
        return None

    patterns = [
        r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"(?:up to|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"<\s*\$?\s*(\d+(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, query, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def determine_query_with_ai(user_input: str) -> Dict:
    """Prefer Google Gemini via google.generativeai when available; otherwise fall back to local parsing."""
    default = {
        'query': user_input,
        'max_price': parse_price_filter(user_input),
        'top_k': 3
    }

    g_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    system = (
        "You are a JSON-only extractor that converts a user's free-form skincare product search "
        "into a concise search query and optional constraints. Respond with valid JSON only."
    )

    user_prompt = (
        "Extract a concise `query` string to use for semantic search, an optional `max_price` "
        "(number or null), and `top_k` (integer, default 3). Return only JSON.\n\n"
        f"User input: {user_input}"
    )

    if g_key:
        try:
            import google.generativeai as genai
            if os.getenv('GOOGLE_API_KEY'):
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

            resp = genai.chat.create(model="gemini-1.5-mini", messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], temperature=0)
            resp_text = None
            try:
                resp_text = getattr(resp, 'last', None) or getattr(resp, 'content', None) or str(resp)
            except Exception:
                resp_text = str(resp)

            m = re.search(r"\{.*\}", resp_text, flags=re.S)
            if m:
                parsed = json.loads(m.group(0))
                q = parsed.get('query', user_input)
                mp = parsed.get('max_price', None)
                if isinstance(mp, str):
                    try:
                        mp = float(re.sub('[^0-9.]', '', mp))
                    except Exception:
                        mp = None
                tk = parsed.get('top_k', 3)
                try:
                    tk = int(tk)
                except Exception:
                    tk = 3
                return {'query': q, 'max_price': mp if mp is not None else parse_price_filter(user_input), 'top_k': tk}
        except Exception:
            pass

    return default


def main():
    print("Loading products...")
    products = embeddings.load_products('products.json')
    print(f"✓ Loaded {len(products)} products")
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Force full re-embedding of all products (ignore cache)')
    parser.add_argument('--top-k', type=int, default=3, help='Default top_k for results')
    args = parser.parse_args()

    embeddings_file = 'chunk_embeddings.npy'
    metadata_file = 'chunk_metadata.json'
    model_name = 'all-MiniLM-L6-v2'
    chunk_size = 60
    chunk_overlap = 15

    print("\nChecking embeddings...")
    chunk_embeddings, chunk_metadata = embeddings.ensure_chunk_embeddings(
        products,
        model_name=model_name,
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=bool(getattr(args, 'force', False)),
    )
    if embeddings.FAISS_AVAILABLE:
        faiss_index = embeddings.build_faiss_index(chunk_embeddings)
    else:
        faiss_index = None

    print("\n" + "="*70)
    print("SEMANTIC SEARCH")
    print("="*70)
    print("Enter your skincare query (or 'quit' to exit):")

    while True:
        query = input('> ').strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue

        ai_result = determine_query_with_ai(query)
        canonical_query = ai_result.get('query', query)
        max_price = ai_result.get('max_price')
        top_k = ai_result.get('top_k', 3)

        query_embedding = embeddings.get_sentence_embeddings([canonical_query], model_name=model_name)[0]

        if max_price is not None:
            filtered_indices = [i for i, md in enumerate(chunk_metadata)
                                if float(products[md['product_index']].get('Price_USD', float('inf'))) <= max_price]
            if not filtered_indices:
                print(f"⚠️ No products found under ${max_price:.2f}; showing closest matches instead.")
                if faiss_index is not None:
                    chunk_hits = embeddings.search_faiss_index(faiss_index, query_embedding, chunk_metadata, top_k=top_k * 3)
                else:
                    chunk_hits = embeddings.find_similar_chunks(query_embedding, chunk_embeddings, chunk_metadata, top_k=top_k * 3)
            else:
                filtered_embeddings = chunk_embeddings[filtered_indices]
                filtered_metadata = [chunk_metadata[i] for i in filtered_indices]
                if embeddings.FAISS_AVAILABLE:
                    temp_index = embeddings.build_faiss_index(filtered_embeddings)
                    chunk_hits = embeddings.search_faiss_index(temp_index, query_embedding, filtered_metadata, top_k=top_k * 3)
                else:
                    chunk_hits = embeddings.find_similar_chunks(query_embedding, filtered_embeddings, filtered_metadata, top_k=top_k * 3)
        else:
            if faiss_index is not None:
                chunk_hits = embeddings.search_faiss_index(faiss_index, query_embedding, chunk_metadata, top_k=top_k * 3)
            else:
                chunk_hits = embeddings.find_similar_chunks(query_embedding, chunk_embeddings, chunk_metadata, top_k=top_k * 3)
        results = embeddings.aggregate_chunk_hits(chunk_hits, products, top_k=top_k)

        for i, (product, score, metadata) in enumerate(results, 1):
            print(f"\n  {i}. {product['Product_Name']} ({product['Brand']})")
            print(f"     Similarity: {score:.4f}")
            print(f"     Price: ${product['Price_USD']}")
            print(f"     Category: {product['Category']}")
            print(f"     Benefits: {product['Primary_Benefit']}")
        print("\n" + "-"*50)


if __name__ == '__main__':
    main()
