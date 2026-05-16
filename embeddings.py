import importlib
import hashlib
import json
import numpy as np
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer

try:
    faiss = importlib.import_module('faiss')
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


def load_products(json_file: str) -> List[Dict]:
    """Load products from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_product_sections(product: Dict) -> List[str]:
    """Return structured product sections keyed by headings."""
    sections = []
    fields = [
        ('Product Name', product.get('Product_Name')),
        ('Brand', product.get('Brand')),
        ('Category', product.get('Category')),
        ('Key ingredients', product.get('Key_Ingredients')),
        ('Benefits', product.get('Primary_Benefit')),
        (f"Suitable for {product.get('Skin_Type')} skin" if product.get('Skin_Type') else None, None),
    ]
    for heading, value in fields:
        if value:
            sections.append(f"{heading}: {value}")
        elif heading and value is None and heading.startswith('Suitable for'):
            sections.append(heading)

    extra_keys = ['Description', 'How_To_Use', 'Directions', 'Details']
    for key in extra_keys:
        if product.get(key):
            sections.append(f"{key.replace('_', ' ').title()}: {product[key]}")

    return sections


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> List[str]:
    """Split a text string into smaller overlapping chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += max(1, chunk_size - overlap)
    return chunks


def create_product_chunks(product: Dict, chunk_size: int = 60, chunk_overlap: int = 15) -> List[str]:
    """Create smaller text chunks for a product before embedding."""
    chunks = []
    for section in create_product_sections(product):
        section = section.strip()
        if not section:
            continue
        section_heading = section.split(':', 1)[0].strip() if ':' in section else 'section'
        for text in chunk_text(section, chunk_size=chunk_size, overlap=chunk_overlap):
            chunks.append({
                'product_index': None,
                'product_name': product.get('Product_Name', ''),
                'brand': product.get('Brand', ''),
                'section': section_heading,
                'text': text,
            })
    return chunks


def average_embeddings_by_group(embeddings: np.ndarray, group_counts: List[int]) -> np.ndarray:
    """Average chunk embeddings back into one embedding per product."""
    if embeddings.size == 0 or not group_counts:
        return np.empty((0, 0), dtype=float)

    averaged = [
        np.mean(embeddings[offset:offset + count], axis=0)
        for offset, count in zip(
            np.cumsum([0] + group_counts[:-1]).tolist(),
            group_counts,
        )
        if count > 0
    ]
    return np.vstack(averaged)


def get_sentence_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings using a local SentenceTransformer model."""
    if not texts:
        return np.empty((0, 0), dtype=float)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    print(f"✓ Generated {embeddings.shape[0]} embeddings using {model_name}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    return embeddings


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_faiss_index(embeddings: np.ndarray):
    if not FAISS_AVAILABLE:
        raise RuntimeError('Faiss is not installed; cannot build Faiss index.')
    embeddings = embeddings.astype('float32')
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_faiss_index(index, query_embedding: np.ndarray, entries: List[Dict], top_k: int = 5) -> List[tuple]:
    query = query_embedding.astype('float32')
    query = normalize_embeddings(query.reshape(1, -1))
    distances, indices = index.search(query, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(entries):
            continue
        results.append((entries[idx], float(score)))
    return results


def find_similar_products(query_embedding: np.ndarray,
                         product_embeddings: np.ndarray,
                         products: List[Dict],
                         top_k: int = 5) -> List[tuple]:
    similarities = []
    for i, prod_embedding in enumerate(product_embeddings):
        similarity = cosine_similarity(query_embedding, prod_embedding)
        similarities.append((products[i], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_embeddings_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 100) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=float)

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Encoding {len(texts)} product texts in {total_batches} batch(es)...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        batch_embeddings = get_sentence_embeddings(batch, model_name)
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)



def save_chunk_index(chunk_embeddings: np.ndarray, chunk_metadata: List[Dict], embeddings_file: str, metadata_file: str):
    np.save(embeddings_file, chunk_embeddings)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2)


def load_chunk_index(embeddings_file: str, metadata_file: str):
    if not os.path.exists(embeddings_file) or not os.path.exists(metadata_file):
        return None, None

    embeddings = np.load(embeddings_file)
    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings file: {embeddings_file}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError(f"Invalid metadata file: {metadata_file}")

    return embeddings, metadata


def aggregate_chunk_hits(chunk_hits: List[tuple], products: List[Dict], top_k: int = 5) -> List[tuple]:
    product_best = {}
    for metadata, score in chunk_hits:
        pid = metadata['product_index']
        if pid not in product_best or score > product_best[pid]['score']:
            product_best[pid] = {'score': score, 'metadata': metadata}

    sorted_products = sorted(product_best.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
    return [(products[pid], info['score'], info['metadata']) for pid, info in sorted_products]


def find_similar_chunks(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunk_metadata: List[Dict], top_k: int = 5) -> List[tuple]:
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((chunk_metadata[i], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def ensure_chunk_embeddings(products: List[Dict], model_name: str, embeddings_file: str, metadata_file: str, chunk_size: int = 60, chunk_overlap: int = 15, force_rebuild: bool = False):
    def product_hash(product: Dict) -> str:
        sections = create_product_sections(product)
        joined = "\n".join(sections)
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()

    # Try to load an existing chunk index unless forced to rebuild
    if force_rebuild:
        loaded_embeddings, loaded_metadata = None, None
        print("--force specified: skipping cached chunk load and rebuilding all product chunks")
    else:
        loaded_embeddings, loaded_metadata = load_chunk_index(embeddings_file, metadata_file)
    if loaded_embeddings is None or loaded_metadata is None:
        # No cache — build from scratch but attach product hashes to metadata
        chunk_metadata = []
        chunk_texts = []
        for product_index, product in enumerate(products):
            p_hash = product_hash(product)
            for section in create_product_sections(product):
                section = section.strip()
                if not section:
                    continue
                section_heading = section.split(':', 1)[0].strip() if ':' in section else 'section'
                for text in chunk_text(section, chunk_size=chunk_size, overlap=chunk_overlap):
                    chunk_metadata.append({
                        'product_index': product_index,
                        'product_name': product.get('Product_Name', ''),
                        'brand': product.get('Brand', ''),
                        'section': section_heading,
                        'text': text,
                        'product_hash': p_hash,
                    })
                    chunk_texts.append(text)

        if not chunk_texts:
            return np.empty((0, 0), dtype=float), chunk_metadata

        print(f"Embedding {len(chunk_texts)} chunks across {len(products)} products...")
        chunk_embeddings = get_embeddings_batch(chunk_texts, model_name)
        save_chunk_index(chunk_embeddings, chunk_metadata, embeddings_file, metadata_file)
        return chunk_embeddings, chunk_metadata

    # We have a cached index — determine which products are already represented.
    print(f"✓ Loaded {len(loaded_embeddings)} cached chunk embeddings")
    # Compute current product hashes and chunk counts
    product_hashes = [product_hash(p) for p in products]
    current_chunk_counts = []
    for product in products:
        cnt = 0
        for section in create_product_sections(product):
            for _ in chunk_text(section, chunk_size=chunk_size, overlap=chunk_overlap):
                cnt += 1
        current_chunk_counts.append(cnt)

    # Inspect loaded metadata for product_hash if available
    metadata_has_hash = all('product_hash' in m for m in loaded_metadata)

    # Build mapping from product_index to list of indices in the loaded arrays
    from collections import defaultdict
    existing_by_product = defaultdict(list)
    for idx, m in enumerate(loaded_metadata):
        existing_by_product[m.get('product_index')].append(idx)

    keep_indices = []
    for pid, p_hash in enumerate(product_hashes):
        if metadata_has_hash:
            # If any loaded metadata for this product has the same hash, keep all its chunks
            found = any(m.get('product_hash') == p_hash for m in loaded_metadata)
            if found:
                keep_indices.extend(existing_by_product.get(pid, []))
        else:
            # Fallback: if loaded metadata contains the same product_index and chunk counts match, reuse
            if pid in existing_by_product and len(existing_by_product[pid]) == current_chunk_counts[pid]:
                keep_indices.extend(existing_by_product[pid])

    keep_indices = sorted(set(keep_indices))

    # Prepare lists for final metadata and embeddings
    final_metadata = []
    final_embeddings_list = []

    # Reuse kept chunks
    if keep_indices:
        for i in keep_indices:
            final_metadata.append(loaded_metadata[i])
            final_embeddings_list.append(loaded_embeddings[i])

    # Determine which products still need embedding
    products_to_embed = []
    for pid in range(len(products)):
        # If any chunk for this product is kept, assume product covered
        if any(m.get('product_index') == pid for m in final_metadata):
            continue
        products_to_embed.append(pid)

    # Create and embed chunks for remaining products
    new_texts = []
    new_metadata = []
    for pid in products_to_embed:
        product = products[pid]
        p_hash = product_hashes[pid]
        for section in create_product_sections(product):
            section = section.strip()
            if not section:
                continue
            section_heading = section.split(':', 1)[0].strip() if ':' in section else 'section'
            for text in chunk_text(section, chunk_size=chunk_size, overlap=chunk_overlap):
                new_metadata.append({
                    'product_index': pid,
                    'product_name': product.get('Product_Name', ''),
                    'brand': product.get('Brand', ''),
                    'section': section_heading,
                    'text': text,
                    'product_hash': p_hash,
                })
                new_texts.append(text)

    if new_texts:
        print(f"Embedding {len(new_texts)} new chunks for {len(products_to_embed)} changed/new products...")
        new_embeddings = get_embeddings_batch(new_texts, model_name)
        # append new embeddings/metadata
        final_metadata.extend(new_metadata)
        final_embeddings_list.extend(list(new_embeddings))

    if not final_embeddings_list:
        return np.empty((0, 0), dtype=float), final_metadata

    final_embeddings = np.vstack(final_embeddings_list)
    save_chunk_index(final_embeddings, final_metadata, embeddings_file, metadata_file)
    print(f"✓ Saved {final_embeddings.shape[0]} chunk embeddings to cache")
    return final_embeddings, final_metadata
