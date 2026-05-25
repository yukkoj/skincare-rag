
# Skincare Semantic Search Engine

An intelligent, AI-powered search engine designed for skincare enthusiasts. This tool moves beyond standard keyword-based searching by using **Semantic Search** and **Hybrid Ranking** to match user queries with product ingredients, consensus, and specifications.

## Key Features

* **Hybrid Search Engine:** Combines Vector Search (FAISS) for intent-based matching with BM25 Keyword Search for technical ingredient precision.
* **Intelligent Aggregation:** Deduplicates retrieved data chunks to provide a clean list of unique product recommendations.
* **Sentiment-Aware Summarization:** Uses positive and negative keyword analysis to categorize feedback ensuring balanced and honest product insights.
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

Before searching, you must build the vector and keyword indices. Run this if you modify products.json otherwise it has been done.

```bash
python main.py
```

### 2. Run the Search Engine

Once the indices are built, you can run the interactive search script:

```bash
python search.py
```

## Project Structure

* `main.py`: Pipeline entry point (data ingestion, embedding, and indexing).
* `embeddings.py`: Handles vectorization, FAISS indexing, and BM25 keyword setup.
* `semantic_profile.py`: Defines the logic for creating unique "semantic profiles" for each product.
* `search.py`: Core search logic, hybrid fusion, and history logging.
* `scraper.py`: Extracts Reddit reviews and performs on-the-fly sentiment scoring.
* `summarizer.py`: Orchestrates the sentiment-categorized feedback and generates LLM summaries.
* `config.py`: Centralized configuration (paths, subreddits, positive/negative word lists).
* `data/raw/`: Source data files.
* `data/generated/`: Computed outputs (indices, profiles, and sentiment-tagged reviews).
* `data/prompts/`: LLM prompt templates
