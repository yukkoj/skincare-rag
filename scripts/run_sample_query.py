import os
import sys
# Ensure project root is on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from scripts import core_embeddings as ce
from scripts.data_manager import load_products

products = load_products(config.PRODUCTS_FILE)
embs, meta = ce.load_chunk_index(config.EMBEDDINGS_FILE, config.METADATA_FILE)
if embs is None:
    embs, meta = ce.ensure_chunk_embeddings(products, model_name='all-MiniLM-L6-v2', embeddings_file=config.EMBEDDINGS_FILE, metadata_file=config.METADATA_FILE)

print('embs shape:', None if embs is None else embs.shape)

query = "hydrating moisturizer under 20 dollars"
q_emb = ce.get_sentence_embeddings([query])[0]

if getattr(ce, 'FAISS_AVAILABLE', False):
    idx = ce.build_faiss_index(embs)
    hits = ce.search_faiss_index(idx, q_emb, meta, top_k=5)
else:
    hits = ce.find_similar_chunks(q_emb, embs, meta, top_k=5)

res = ce.aggregate_chunk_hits(hits, products, top_k=5)
print('Results:')
for i, (p, score, m) in enumerate(res, 1):
    print(f"{i}. {p.get('Product_Name')} (${p.get('Price_USD')}) score={score}")
