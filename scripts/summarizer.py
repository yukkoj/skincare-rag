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
    
    # Grab the top 5 most upvoted reviews to feed to the LLM
    top_reviews = sorted(reviews, key=lambda x: x.get('score', 0), reverse=True)[:5]
    review_texts = "\n".join([f"- {r.get('title', '')}: {r.get('text', '')}" for r in top_reviews])

    prompt = f"""You are a skincare product review analyst. 
Based on the following Reddit reviews, write a concise 
2-3 sentence summary that captures the overall sentiment and key themes.

Product: {product_name}
Total Reviews Analyzed: {len(reviews)}
Top Mentioned Words: {', '.join([phrase for phrase, count in top_phrases])}

Sample Reviews:
{review_texts}

Write a summary that:
1. Starts with overall sentiment (positive, negative, or mixed)
2. Mentions key benefits users love
3. Notes any common complaints or skin reactions
4. Is concise and specific to this product

Summary:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠ LLM generation failed: {e}")
        return None

def summarize_product_reviews(filepath):
    """Create summary with analytics and LLM-generated text"""
    reviews = load_product_reviews(filepath)
    
    if not reviews: # If it's an empty list or file doesn't exist
        return None
        
    # Get product name from the filename (e.g. "product1")
    product_id = filepath.stem
    
    # Calculate basic stats from our scraper output
    avg_score = sum(r.get('score', 0) for r in reviews) / len(reviews)
    top_phrases = extract_key_phrases(reviews)[:5]
    
    # Generate LLM summary
    llm_summary = generate_llm_summary(product_id, reviews, top_phrases)
    
    # Create the final summary object
    full_summary = {
        'product_id': product_id,
        'total_reviews': len(reviews),
        'avg_reddit_score': round(avg_score, 1),
        'top_phrases': [{'phrase': phrase, 'count': count} for phrase, count in top_phrases],
        'llm_summary': llm_summary,
        'generated_at': reviews[0].get('scraped_at', '') if reviews else ''
    }
    
    return full_summary

def save_summary(product_id, summary):
    """Save summary to file"""
    # Ensure config.REVIEWS_DIR is a Path object
    reviews_dir = Path(config.REVIEWS_DIR)
    summaries_dir = reviews_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    summary_file = summaries_dir / f"{product_id}_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary_file

def summarize_all_products(skip_existing=True):
    """Generate summaries for all products with reviews"""
    reviews_dir = Path(config.REVIEWS_DIR)
    summaries_dir = reviews_dir / "summaries"
    
    # Look for files matching the scraper's format (product1.json, etc.)
    review_files = list(reviews_dir.glob("product*.json"))
    total_files = len(review_files)
    
    if total_files == 0:
        print("No review files found to summarize!")
        return
        
    print(f"\nGenerating LLM summaries for {total_files} products...\n")
    
    for idx, review_file in enumerate(review_files):
        # Determine what the summary filename would be
        product_id = review_file.stem
        expected_summary_file = summaries_dir / f"{product_id}_summary.json"
        
        # --- NEW: Skip existing check ---
        if skip_existing and expected_summary_file.exists():
            print(f"[{idx + 1}/{total_files}] ⏭️ Skipping {review_file.name} (Summary already exists)")
            continue
            
        print(f"[{idx + 1}/{total_files}] Processing {review_file.name}...")
        
        summary = summarize_product_reviews(review_file)
        
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