"""
Reddit review scraper
Handles loading products, scraping reviews, and saving results
"""
import json
import time
from datetime import datetime
import config

# ==================== DATA LOADING ====================

def load_products():
    """Load products from JSON file"""
    if not config.PRODUCTS_FILE.exists():
        raise FileNotFoundError(f"Products file not found: {config.PRODUCTS_FILE}")
    
    with open(config.PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== SENTIMENT ANALYSIS ====================

def classify_sentiment(text):
    """Simple sentiment classification"""
    if not text:
        return 'neutral'
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in config.POSITIVE_WORDS if word in text_lower)
    neg_count = sum(1 for word in config.NEGATIVE_WORDS if word in text_lower)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    return 'neutral'

# ==================== REVIEW SCRAPING ====================

def create_sample_reviews(product_name, brand):
    """
    Create sample reviews
    (Replace this with real Reddit scraping later)
    """
    sample_reviews = [
        {
            'text': f"I've been using {product_name} for months and it's amazing! Highly recommend.",
            'subreddit': 'SkincareAddiction',
            'score': 150,
            'sentiment': 'positive'
        },
        {
            'text': f"{brand} makes quality products. The {product_name} is gentle and effective.",
            'subreddit': '30PlusSkinCare',
            'score': 85,
            'sentiment': 'positive'
        },
        {
            'text': f"Unfortunately the {product_name} broke me out. YMMV though!",
            'subreddit': 'SkincareAddiction',
            'score': 23,
            'sentiment': 'negative'
        }
    ]
    
    # Add metadata
    reviews = []
    for i, review in enumerate(sample_reviews[:config.MAX_REVIEWS_PER_PRODUCT], 1):
        review['review_id'] = f"r_{hash(product_name)}_{i}"
        review['url'] = f"https://reddit.com/r/{review['subreddit']}/sample_{i}"
        review['created_utc'] = int(time.time()) - (i * 86400)
        review['scraped_at'] = datetime.utcnow().isoformat() + 'Z'
        reviews.append(review)
    
    return reviews

# ==================== SAVING REVIEWS ====================

def save_reviews(product_id, product_name, brand, reviews):
    """Save reviews to individual JSON file"""
    review_file = config.REVIEWS_DIR / f"product_{product_id}.json"
    
    # Calculate summary
    positive = sum(1 for r in reviews if r['sentiment'] == 'positive')
    negative = sum(1 for r in reviews if r['sentiment'] == 'negative')
    neutral = len(reviews) - positive - negative
    
    review_data = {
        'product_id': product_id,
        'product_name': product_name,
        'brand': brand,
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'total_reviews': len(reviews),
        'summary': {
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral
        },
        'reviews': reviews
    }
    
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    return review_file

# ==================== MAIN SCRAPING FUNCTION ====================

def scrape_all_products(max_products=None, skip_existing=True):
    """
    Scrape reviews for all products
    
    Args:
        max_products: Limit number of products (for testing)
        skip_existing: Skip products that already have reviews
    
    Returns:
        dict: Statistics about scraping
    """
    products = load_products()
    
    if max_products:
        products = products[:max_products]
    
    print(f"\n{'='*60}")
    print(f"Starting Reddit review scraping for {len(products)} products")
    print(f"{'='*60}\n")
    
    stats = {'scraped': 0, 'skipped': 0, 'errors': 0}
    
    for i, product in enumerate(products, 1):
        product_id = product['Product_ID']
        product_name = product['Product_Name']
        brand = product['Brand']
        
        # Check if reviews exist
        review_file = config.REVIEWS_DIR / f"product_{product_id}.json"
        if skip_existing and review_file.exists():
            print(f"[{i}/{len(products)}] Skipping {product_name} (already scraped)")
            stats['skipped'] += 1
            continue
        
        print(f"[{i}/{len(products)}] Scraping: {product_name} ({brand})")
        
        try:
            # Get reviews
            reviews = create_sample_reviews(product_name, brand)
            
            # Save results
            if reviews:
                save_reviews(product_id, product_name, brand, reviews)
                print(f"  ✓ Saved {len(reviews)} reviews")
                stats['scraped'] += 1
            else:
                print(f"  ⚠ No reviews found")
            
            # Sleep between products
            if i < len(products):
                time.sleep(config.SLEEP_BETWEEN_PRODUCTS)
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            stats['errors'] += 1
    
    return stats

