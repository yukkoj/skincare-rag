import os
import json
import requests
import time
from datetime import datetime
import re
import config

def scrape_reddit_without_api(product_name, target_subreddits="SUBREDDITS", max_retries=3):
    """
    Searches Reddit using the public .json endpoint.
    """
    reviews = []
    url = f"https://www.reddit.com/r/{target_subreddits}/search.json"
    
    # Use a descriptive User-Agent as per Reddit's API rules
    headers = {
        "User-Agent": "python:my_product_scraper:v1.0 (by /u/YourRedditUsername)"
    }
    
    params = {
        "q": f'"{product_name}"',
        "restrict_sr": "1" if target_subreddits != "all" else "0",
        "sort": "relevance",
        "limit": 25  # Increased limit 
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 429:
                print(f"  ⚠ Rate limited (Attempt {attempt + 1}/{max_retries}). Sleeping...")
                time.sleep(10)
                continue  # Try the request again
                
            if response.status_code != 200:
                print(f"  ⚠ HTTP Error: {response.status_code}")
                return []

            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            for post in posts:
                post_data = post['data']
                title = post_data.get('title', '').lower()
                body = post_data.get('selftext', '').lower()
                
                # Check BOTH title and body
                if product_name.lower() in title or product_name.lower() in body:
                    reviews.append({
                        'review_id': f"reddit_json_{post_data['id']}",
                        'title': post_data.get('title', ''),
                        'text': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'scraped_at': datetime.utcnow().isoformat() + 'Z'
                    })

            return reviews # Success, return the data

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            time.sleep(5) # Brief pause before retry on network error

    print("  ❌ Max retries reached.")
    return []


def scrape_all_products(max_products=None, skip_existing=True, target_subreddits="SUBREDDITS", max_retries=3):
    stats = {"scraped": 0, "skipped": 0, "errors": 0}
    
    target_subs_string = "+".join(config.SUBREDDITS)
    
    save_dir = os.path.join("data", "reviews")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with open(os.path.join("data", "products", "products.json"), 'r', encoding='utf-16') as f:
            products = json.load(f)
    except FileNotFoundError:
        print("❌ Error: data/products/products.json not found.")
        return stats
        
    if max_products:
        products = products[:max_products]
        
    for i, product in enumerate(products, start=1):
        try:
            if isinstance(product, str):
                current_product_name = product
            elif isinstance(product, dict):
                current_product_name = product.get('Product_Name') or product.get('name') or product.get('product_name')
            else:
                continue
                
            if not current_product_name:
                continue

            # --- NEW: Set the filename using the loop counter ---
            filepath = os.path.join(save_dir, f"product{i}.json")

            if skip_existing and os.path.exists(filepath):
                print(f"⏭️ Skipping: {current_product_name} (product{i}.json already exists)")
                stats['skipped'] += 1
                continue

            print(f"Scraping: {current_product_name} (Saving as product{i}.json)...")

            reviews = scrape_reddit_without_api(
                product_name=current_product_name,  
                target_subreddits=target_subs_string,
                max_retries=max_retries
            )
            
            if reviews:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(reviews, f, indent=4)
                print(f"  ✓ Saved {len(reviews)} reviews to {filepath}")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                print(f"  ⚠ No reviews found, saved empty file to prevent re-scraping.")

            stats['scraped'] += 1
            
            print(f"  Sleeping for {config.SLEEP_BETWEEN_PRODUCTS} seconds...")
            time.sleep(config.SLEEP_BETWEEN_PRODUCTS)
            
        except Exception as e:
            print(f"  ❌ Error processing {product}: {e}")
            stats['errors'] += 1

    return stats