import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRODUCTS_DIR = DATA_DIR / "products"
PRODUCTS_FILE = PRODUCTS_DIR / "products.json"
REVIEWS_DIR = DATA_DIR / "reviews"
SUMMARIES_DIR = DATA_DIR / "summaries"
EMBEDDINGS_FILE = DATA_DIR / "chunk_embeddings.npy"
METADATA_FILE = DATA_DIR / "chunk_metadata.json"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_FILE = OUTPUTS_DIR / "search_results.json"

SUBREDDITS = ["SkincareAddiction", "30PlusSkinCare", "AsianBeauty"]
MAX_REVIEWS_PER_PRODUCT = 5
SLEEP_BETWEEN_PRODUCTS = 3

POSITIVE_WORDS = [
    "love", "amazing", "hg", "holy grail", "great", "best",
    "perfect", "favorite", "recommend", "excellent"
]
NEGATIVE_WORDS = [
    "hate", "broke out", "breakout", "irritated", "worst",
    "disappointed", "terrible", "awful", "bad", "avoid"
]