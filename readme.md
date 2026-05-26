
# Skincare Semantic Search Engine

An intelligent, AI-powered search engine for moisturizers.This tool replaces standard keyword searching with Pure Vector Search, utilizing Gemini cloud embeddings to intuitively match the intent of user queries directly to product ingredients, consensus, and specifications.

# Key Features

* AI-Driven Vector Search: Powered by ChromaDB and Gemini Embeddings to understand the complex meaning and intent behind skincare queries.
* Fast Execution: Fully offloads heavy ML processing to the cloud, allowing the app to run instantly on any hardware without local lag.
* Smart Database Syncing: Automatically detects and indexes only new products added to the master file without rebuilding the existing database.
* Grounded AI Recommendations: Uses Retrieval-Augmented Generation (RAG) to feed exact product specs to Gemini, generating personalized, factual advice based on the data.
* Sentiment & Consensus Analysis: Automatically categorizes positive and negative product feedback to provide unbiased recommendations.

## System Architecture

The pipeline follows a data-flow process to ensure accuracy and relevance.

## Setup & Installation

Install the required dependencies:

```bash
pip install chromadb google-genai google-generativeai
```

## Usage

This project is executed in two main steps:

### 1. Initialize the Database

Before searching, you must build ChromaDB. Run this if you modify products.json otherwise it has been done.

```bash
python main.py
```

### 2. Run the Search Engine

Once the database is built, you can run the interactive search script:

```bash
python search.py
```

## Customize

Append in or replace products.json to broaden choices. You must delete reviews and summaries files if you replace the products.json.

## Project Structure

* `main.py`: Pipeline entry point (data ingestion, embedding, and indexing).
* `embeddings.py`: Handles vectorization, FAISS indexing, and BM25 keyword setup.
* `semantic_profile.py`: Defines the logic for creating unique "semantic profiles" for each product.
* `search.py`: Core search logic, hybrid fusion, and history logging.
* `scraper.py`: Extracts Reddit reviews and performs on-the-fly sentiment scoring.
* `summarizer.py`: Orchestrates the sentiment-categorized feedback and generates LLM summaries.
* `config.py`: Centralized configuration (paths, subreddits, positive/negative word lists).
* `data/raw/`: Source data files.
* `data/generated/`: Computed outputs (summaries, profiles, and database).
* `data/prompts/`: LLM prompt templates
