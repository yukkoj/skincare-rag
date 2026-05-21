import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PRODUCTS_DIR = os.path.join(DATA_DIR, "products")
PRODUCTS_FILE = os.path.join(PRODUCTS_DIR, "products.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "chunk_embeddings.npy")
METADATA_FILE = os.path.join(DATA_DIR, "chunk_metadata.json")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_FILE = os.path.join(OUTPUTS_DIR, "search_results.json")

# Scraping settings
SUBREDDITS = ['SkincareAddiction', '30PlusSkinCare', 'AsianBeauty']
MAX_REVIEWS_PER_PRODUCT = 5
SLEEP_BETWEEN_PRODUCTS = 3

# Sentiment words
POSITIVE_WORDS = ['love', 'amazing', 'hg', 'holy grail', 'great', 'best', 
                  'perfect', 'favorite', 'recommend', 'excellent']
NEGATIVE_WORDS = ['hate', 'broke out', 'breakout', 'irritated', 'worst', 
                  'disappointed', 'terrible', 'awful', 'bad', 'avoid']