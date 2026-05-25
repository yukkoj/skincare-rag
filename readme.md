
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

This project is executed in two main steps:

### 1. Initialize the Database

Before searching, you must build the vector and keyword indices. Run this once (or whenever you add new products):

```bash
python main.py
```

### 2. Run the Search Engine

Once the indices are built, you can run the interactive search script:

```bash
python search.py
```

## Project Structure

* main.py: Generates the hybrid embeddings as a pre-processing step.
* search.py: The core search logic, hybrid fusion, and history logging.
* config.py: Centralized configuration for file paths and project constants.
* generated/: Stores the semantic_profiles.json, search_history.json
