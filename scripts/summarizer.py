import json
import time
from pathlib import Path
from collections import Counter
import google.generativeai as genai
import config

# Configure Model once globally
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

def load_product_reviews(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure we return a list even if file is weird
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []
    
def extract_key_phrases(reviews, top_n=10):
    words = []
    for review in reviews:
        text = (review.get('title', '') + " " + review.get('text', '')).lower()
        words.extend([w for w in text.split() 
                      if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'have', 'just', 'like']])
    return Counter(words).most_common(top_n)

def generate_llm_summary(product_name, reviews, top_phrases):
    if not reviews: return None
    
    # Pre-filter for efficiency
    positives = sorted([r for r in reviews if r.get('sentiment_score', 0) > 0], 
                       key=lambda x: x.get('sentiment_score', 0), reverse=True)[:5]
    criticals = sorted([r for r in reviews if r.get('sentiment_score', 0) < 0], 
                       key=lambda x: x.get('sentiment_score', 0))[:5]
    
    # Format texts - Using a generator expression inside join is faster
    pos_texts = "\n".join(f"- {r.get('text', '')[:200]}..." for r in positives)
    crit_texts = "\n".join(f"- {r.get('text', '')[:200]}..." for r in criticals)
    
    review_texts = f"POSITIVE FEEDBACK:\n{pos_texts or 'None'}\n\nCRITICAL FEEDBACK:\n{crit_texts or 'None'}"

    try:
        
        review_count = len(reviews) if isinstance(reviews, list) else 0
        prompt_template = Path(config.PROMPTS_DIR / "summarize.txt").read_text(encoding='utf-8')
        
        full_prompt = prompt_template.format(
            product_name=product_name,
            len_reviews=review_count,
            top_phrases=', '.join([p[0] for p in top_phrases]),
            review_texts=review_texts
        )
        response = model.generate_content(full_prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"  ⚠ LLM generation failed: {e}")
        return None

def process_summaries(skip_existing=True):
    with open(config.PRODUCTS_FILE, 'r', encoding='utf-16') as f:
        products = json.load(f)
    
    product_map = {f"product{i+1}": p.get('Product_Name', "Unknown") for i, p in enumerate(products)}
    review_files = list(config.REVIEWS_DIR.glob("product*.json"))
    
    for idx, review_file in enumerate(review_files):
        product_id = review_file.stem
        summary_file = config.SUMMARIES_DIR / f"{product_id}_summary.json"
        
        # 1. Smarter Skip Logic
        if skip_existing and summary_file.exists():
            if review_file.stat().st_mtime <= summary_file.stat().st_mtime:
                continue

        # 2. Robust Loading
        reviews = load_product_reviews(review_file)
        if not reviews:
            print(f"  ⚠ Skipping {product_id}: No valid review list found.")
            continue
            
        print(f"[{idx + 1}/{len(review_files)}] 📝 Summarizing {product_map.get(product_id)}...")
        
        # 3. Inline Summary Logic (Avoids double-loading)
        top_phrases = [p[0] for p in extract_key_phrases(reviews)]
        summary_text = generate_llm_summary(product_map.get(product_id), reviews, top_phrases)
        
        if summary_text:
            output = {
                'product_id': product_id,
                'total_reviews': len(reviews),
                'llm_summary': summary_text,
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            config.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
            summary_file = config.SUMMARIES_DIR / f"{product_id}_summary.json"

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            time.sleep(4) # Rate limit