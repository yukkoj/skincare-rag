"""Wrapper to expose the project's core embeddings implementation from `scripts.core_embeddings`.
This keeps external imports using `from scripts import embeddings` working while
the heavy implementation lives in `scripts/core_embeddings.py`.
"""
from . import core_embeddings as core

ensure_chunk_embeddings = core.ensure_chunk_embeddings
get_sentence_embeddings = core.get_sentence_embeddings
get_embeddings_batch = core.get_embeddings_batch
build_faiss_index = core.build_faiss_index
search_faiss_index = core.search_faiss_index
find_similar_chunks = core.find_similar_chunks
aggregate_chunk_hits = core.aggregate_chunk_hits
