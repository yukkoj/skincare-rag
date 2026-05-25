
# Skincare Semantic Search Engine

An intelligent, AI-powered search engine designed for skincare enthusiasts. This tool moves beyond standard keyword-based searching by using **Semantic Search** and **Hybrid Ranking** to match user queries with product ingredients, consensus, and specifications.

## Key Features

* **Hybrid Search Engine:** Combines Vector Search (FAISS) for intent-based matching with BM25 Keyword Search for technical ingredient precision.
* **Intelligent Aggregation:** Deduplicates retrieved data chunks to provide a clean list of unique product recommendations.
* **LLM Synthesis:** Integrates Gemini API to generate personalized recommendations based on the retrieved product profiles.
* **Persistent Logging:** Automatically saves user queries and AI responses in a structured, readable history log.
* **Incremental Processing:** Efficiently processes new products without re-indexing the entire database.

## System Architecture

The pipeline follows a sophisticated data-flow process to ensure accuracy and relevance:

## Setup & Installation

Install the required dependencies:

```bash
pip install sentence-transformers faiss-cpu rank-bm25 numpy
```

## Usage

To perform a search and generate an expert-level recommendation:

from search import search_products, generate_ai_recommendation

1. Query the hybrid engine
top_products = search_products("I need a cheap, fragrance-free lotion for sensitive skin.")

2. Generate expert advice

response = generate_ai_recommendation(query, top_products, full_db)
print(response)

## Project Structure

main.py: Generates the hybrid embeddings as a pre-processing step.

search.py: The core search logic, hybrid fusion, and history logging.

config.py: Centralized configuration for file paths and project constants.

generated/: Stores the semantic_profiles.json, search_history.json, and embedding indices.
