"""
Review summarizer
Reads reviews and creates summaries with rate-limit handling
"""
import json
import time
from pathlib import Path
from collections import Counter
import google.generativeai as genai
import config

genai.configure(api_key=config.GOOGLE_API_KEY)
# Using your project's high-RPM choice: 15 RPM limit
model = genai.GenerativeModel("gemini-3.1-flash-lite")

def load_product_reviews(filepath):
    """Load reviews for a specific product file"""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_key_phrases(reviews, top_n=10):
    """Extract most common phrases from reviews"""
    words = []
    for review in reviews:
        # Use both title and text since our scraper grabs both
        text = (review.get('title', '') + " " + review.get('text', '')).lower()
        # Remove common words
        words.extend([w for w in text.split() 
                     if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'have', 'just', 'like']])
    
    return Counter(words).most_common(top_n)

def generate_llm_summary(product_name, reviews, top_phrases):
    """Use Gemini to generate natural language summary from raw reviews"""
    
    # Grab the top 7 most upvoted reviews to feed to the LLM
    top_reviews = sorted(reviews, key=lambda x: x.get('score', 0), reverse=True)[:7]
    review_texts = "\n".join([f"- {r.get('title', '')}: {r.get('text', '')}" for r in top_reviews])

    prompt_template = Path(config.PROMPTS_DIR / "summarize.txt").read_text(encoding='utf-8')

    full_prompt = (
        f"Product: {product_name}\n"
        f"Common Keywords: {', '.join([p[0] for p in top_phrases])}\n\n"
        f"{prompt_template}\n\n"
        f"Reviews:\n{review_texts}"
    )

    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠ LLM generation failed: {e}")
        return None

def summarize_product_reviews(filepath, product_name):
    """Create summary object"""
    reviews = load_product_reviews(filepath)
    if not reviews:
        return None
        
    product_id = filepath.stem
    avg_score = sum(r.get('score', 0) for r in reviews) / len(reviews)
    top_phrases = extract_key_phrases(reviews)[:5]
    
    # Generate LLM summary
    llm_summary = generate_llm_summary(product_name, reviews, top_phrases)
    
    return {
        'product_id': product_id,
        'total_reviews': len(reviews),
        'avg_reddit_score': round(avg_score, 1),
        'top_phrases': [{'phrase': phrase, 'count': count} for phrase, count in top_phrases],
        'llm_summary': llm_summary,
        'generated_at': reviews[0].get('scraped_at', '') if reviews else ''
    }

def save_summary(product_id, summary):
    """Save summary to file using config-defined directories."""
    
    # Ensure the directory exists (config.SUMMARIES_DIR is already a Path object)
    config.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define the file path
    summary_file = config.SUMMARIES_DIR / f"{product_id}_summary.json"
    
    # Write the JSON file
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_file

def summarize_all_products(skip_existing=True):
    with open(config.PRODUCTS_FILE, 'r', encoding='utf-16') as f:
        products = json.load(f)
    
    product_map = {f"product{i+1}": p.get('Product_Name') for i, p in enumerate(products)}
    review_files = list(config.REVIEWS_DIR.glob("product*.json"))
    total_files = len(review_files) # Define this!

    for idx, review_file in enumerate(review_files):
        product_id = review_file.stem
        product_name = product_map.get(product_id, "Unknown Product")
        expected_summary_file = config.SUMMARIES_DIR / f"{product_id}_summary.json"
        
        # Check existing
        if skip_existing and expected_summary_file.exists():
            print(f"[{idx + 1}/{total_files}] ⏭️ Skipping {review_file.name}")
            continue
            
        print(f"[{idx + 1}/{total_files}] Processing {product_name}...")
        
        # Call the function ONCE
        summary = summarize_product_reviews(review_file, product_name)
        
        if summary and summary.get('llm_summary'):
            save_summary(product_id, summary)

            print(f"✓ Summarized: {summary['product_id']}")
            print(f"  📝 {summary['llm_summary']}\n")
        else:
            print(f"  ⚠ Skipped or failed to generate summary.")
        
        # --- Rate-limiting logic ---
        if idx < total_files - 1:
            print("⏳ Pacing requests to respect 15 RPM free tier limit (sleeping 4s)...")
            time.sleep(4)
    
    print(f"\n✓ All summaries generated!\n")

if __name__ == "__main__":
    summarize_all_products()