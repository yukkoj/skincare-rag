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
model = genai.GenerativeModel("gemini-3.1-flash-lite")

def load_product_reviews(filepath):
    if not filepath.exists(): return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_key_phrases(reviews, top_n=10):
    words = []
    for review in reviews:
        text = (review.get('title', '') + " " + review.get('text', '')).lower()
        words.extend([w for w in text.split() 
                      if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'have', 'just', 'like']])
    return Counter(words).most_common(top_n)

def generate_llm_summary(product_name, reviews, top_phrases):
    # 1. Categorize reviews by sentiment score
    positives = sorted([r for r in reviews if r.get('sentiment_score', 0) > 0], 
                       key=lambda x: x.get('sentiment_score', 0), reverse=True)[:5]
    criticals = sorted([r for r in reviews if r.get('sentiment_score', 0) < 0], 
                       key=lambda x: x.get('sentiment_score', 0))[:5]
    
    # 2. Format feedback for the LLM
    pos_texts = "\n".join([f"- {r.get('text', '')[:200]}..." for r in positives])
    crit_texts = "\n".join([f"- {r.get('text', '')[:200]}..." for r in criticals])
    review_texts = f"POSITIVE FEEDBACK:\n{pos_texts if pos_texts else 'None'}\n\nCRITICAL FEEDBACK:\n{crit_texts if crit_texts else 'None'}"

    # 3. Load template
    prompt_template = Path(config.PROMPTS_DIR / "summarize.txt").read_text(encoding='utf-8')
    
    # 4. Populate template variables
    # We use .format() to safely inject variables into the template text
    full_prompt = prompt_template.format(
        product_name=product_name,
        len_reviews=len(reviews),
        top_phrases=', '.join([p[0] for p in top_phrases]),
        review_texts=review_texts
    )
 
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠ LLM generation failed: {e}")
        return None

def summarize_product_reviews(filepath, product_name):
    reviews = load_product_reviews(filepath)
    if not reviews: return None
    product_id = filepath.stem
    avg_score = sum(r.get('score', 0) for r in reviews) / len(reviews)
    top_phrases = extract_key_phrases(reviews)[:5]
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
    config.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = config.SUMMARIES_DIR / f"{product_id}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_file

def process_summaries(skip_existing=True):
    """Processes product reviews and skips items with no review data."""
    with open(config.PRODUCTS_FILE, 'r', encoding='utf-16') as f:
        products = json.load(f)
    
    product_map = {f"product{i+1}": p.get('Product_Name') for i, p in enumerate(products)}
    review_files = list(config.REVIEWS_DIR.glob("product*.json"))
    total_files = len(review_files)

    for idx, review_file in enumerate(review_files):
        product_id = review_file.stem
        product_name = product_map.get(product_id, "Unknown Product")
        
        # Check for existing summary
        if skip_existing and (config.SUMMARIES_DIR / f"{product_id}_summary.json").exists():
            continue
        
        # Load and check for reviews
        reviews = load_product_reviews(review_file)
        if not reviews:
            print(f"  ⚠ No reviews found for {product_name} ({product_id}). Skipping.")
            continue
            
        print(f"[{idx + 1}/{total_files}] 📝 Summarizing {product_name}...")
        summary = summarize_product_reviews(review_file, product_name)
        
        if summary and summary.get('llm_summary'):
            save_summary(product_id, summary)
            print(f"  ✓ Saved summary for {product_id}")
        
        if idx < total_files - 1:
            time.sleep(4)
    
    print(f"\n✅ All available reviews processed.")

if __name__ == "__main__":
    process_summaries()