# Lightweight wrapper that re-exports the project's top-level embeddings module.
# Keeps the original embeddings.py implementation but provides a module under scripts.
import embeddings as core_embeddings

ensure_chunk_embeddings = core_embeddings.ensure_chunk_embeddings
get_sentence_embeddings = core_embeddings.get_sentence_embeddings
get_embeddings_batch = core_embeddings.get_embeddings_batch
build_faiss_index = core_embeddings.build_faiss_index
search_faiss_index = core_embeddings.search_faiss_index
find_similar_chunks = core_embeddings.find_similar_chunks
aggregate_chunk_hits = core_embeddings.aggregate_chunk_hits
