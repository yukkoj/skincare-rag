import json
import time
import requests
from datetime import datetime
import config

def calculate_sentiment(text):
    """Simple net sentiment calculation."""
    # Use your defined words from config.py
    pos_score = sum(text.count(word) for word in config.POSITIVE_WORDS)
    neg_score = sum(text.count(word) for word in config.NEGATIVE_WORDS)
    return pos_score - neg_score

def scrape_reddit_without_api(product_name, target_subreddits, max_retries=3):
    """
    Searches Reddit using the public .json endpoint.
    """
    reviews = []
    # Use the '+' joined string from your config
    url = f"https://www.reddit.com/r/{target_subreddits}/search.json"
    
    headers = {
        "User-Agent": "python:skincare_search_pipeline:v1.0 (by /u/YourRedditUsername)"
    }
    
    params = {
        "q": f'"{product_name}"',
        "restrict_sr": "1",
        "sort": "relevance",
        "limit": 25 
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 429:
                print(f"  ⚠ Rate limited (Attempt {attempt + 1}/{max_retries}). Sleeping...")
                time.sleep(10)
                continue
                
            if response.status_code != 200:
                print(f"  ⚠ HTTP Error: {response.status_code}")
                return []

            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            for post in posts:
                post_data = post['data']
                title = post_data.get('title', '').lower()
                body = post_data.get('selftext', '').lower()
                combined_text = f"{title} {body}"
        
                if product_name.lower() in combined_text:
                # Calculate sentiment on the fly
                    sentiment = calculate_sentiment(combined_text)
            
                reviews.append({
                    'review_id': f"reddit_json_{post_data['id']}",
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'sentiment_score': sentiment, # Now saved in your JSON
                    'score': post_data.get('score', 0),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'scraped_at': datetime.utcnow().isoformat() + 'Z'
                    })
            return reviews

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            time.sleep(5)

    print("  ❌ Max retries reached.")
    return []


def scrape_all_products(max_products=None, skip_existing=True, max_retries=3):
    stats = {"scraped": 0, "skipped": 0, "errors": 0}
    
    # Use your config variable for subreddits
    target_subs_string = "+".join(config.SUBREDDITS)
    
    # Ensure directory exists using config.REVIEWS_DIR
    config.REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load products using config.PRODUCTS_FILE
    if not config.PRODUCTS_FILE.exists():
        print(f"❌ Error: {config.PRODUCTS_FILE} not found.")
        return stats
        
    with open(config.PRODUCTS_FILE, 'r', encoding='utf-16') as f:
        products = json.load(f)
        
    if max_products:
        products = products[:max_products]
        
    for i, product in enumerate(products, start=1):
        try:
            # Handle list of strings or list of dicts
            current_product_name = product if isinstance(product, str) else product.get('Product_Name')
            if not current_product_name:
                continue

            # Set the filename path using config.REVIEWS_DIR
            filepath = config.REVIEWS_DIR / f"product{i}.json"

            if skip_existing and filepath.exists():
                stats['skipped'] += 1
                continue

            print(f"Scraping: {current_product_name}...")

            reviews = scrape_reddit_without_api(
                product_name=current_product_name, 
                target_subreddits=target_subs_string,
                max_retries=max_retries
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(reviews if reviews else [], f, indent=4)
            
            print(f"  ✓ Saved {len(reviews)} reviews.")
            stats['scraped'] += 1
            
            print(f"  Sleeping for {config.SLEEP_BETWEEN_PRODUCTS} seconds...")
            time.sleep(config.SLEEP_BETWEEN_PRODUCTS)
            
        except Exception as e:
            print(f"  ❌ Error processing product {i}: {e}")
            stats['errors'] += 1

    return stats