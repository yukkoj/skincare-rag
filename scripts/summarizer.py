"""
Review summarizer
Reads reviews and creates summaries
"""
import json
from pathlib import Path
from collections import Counter
import google.generativeai as genai
import config


genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


def load_product_reviews(product_id):
    """Load reviews for a specific product"""
    review_file = config.REVIEWS_DIR / f"product_{product_id}.json"
    
    if not review_file.exists():
        return None
    
    with open(review_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_key_phrases(reviews, top_n=10):
    """Extract most common phrases from reviews"""
    words = []
    for review in reviews:
        text = review['text'].lower()
        # Remove common words
        words.extend([w for w in text.split() 
                     if len(w) > 3 and w not in ['this', 'that', 'with', 'from']])
    
    return Counter(words).most_common(top_n)


def calculate_analytics(data, reviews):
    """Calculate analytics from reviews"""
    avg_score = sum(r['score'] for r in reviews) / len(reviews) if reviews else 0
    top_phrases = extract_key_phrases(reviews)
    
    # Get sample reviews by sentiment
    positive_samples = [r['text'] for r in reviews if r['sentiment'] == 'positive'][:3]
    negative_samples = [r['text'] for r in reviews if r['sentiment'] == 'negative'][:3]
    
    analytics = {
        'product_id': data['product_id'],
        'product_name': data['product_name'],
        'brand': data['brand'],
        'total_reviews': data['total_reviews'],
        'sentiment_summary': data['summary'],
        'avg_reddit_score': round(avg_score, 1),
        'top_phrases': [{'phrase': phrase, 'count': count} 
                       for phrase, count in top_phrases[:5]],  # Top 5 only
        'sample_positive': positive_samples,
        'sample_negative': negative_samples,
        'subreddit_distribution': dict(Counter(r['subreddit'] for r in reviews))
    }
    
    return analytics


def generate_llm_summary(analytics):
    """Use Gemini to generate natural language summary from analytics"""
    
    # Format the prompt
    prompt = f"""You are a skincare product review analyst. 
                Based on the following analytics from Reddit reviews, write a concise 
                2-3 sentence summary that captures the overall sentiment and key themes.

Product: {analytics['product_name']} by {analytics['brand']}
Total Reviews: {analytics['total_reviews']}
Sentiment Distribution: {analytics['sentiment_summary']['positive_count']} positive, {analytics['sentiment_summary']
                        ['negative_count']} negative, {analytics['sentiment_summary']['neutral_count']} neutral
Average Reddit Score: {analytics['avg_reddit_score']}
Top Mentioned Words: {', '.join([p['phrase'] for p in analytics['top_phrases']])}

Sample Positive Reviews:
{chr(10).join('- ' + review for review in analytics['sample_positive']) if analytics['sample_positive'] else 'None'}

Sample Negative Reviews:
{chr(10).join('- ' + review for review in analytics['sample_negative']) if analytics['sample_negative'] else 'None'}

Write a summary that:
1. Starts with overall sentiment
2. Mentions key benefits users love
3. Notes any common complaints
4. Is concise and specific to this product

Summary:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠ LLM generation failed: {e}")
        return None


def summarize_product_reviews(product_id):
    """Create summary with analytics and LLM-generated text"""
    data = load_product_reviews(product_id)
    
    if not data:
        print(f"No reviews found for product {product_id}")
        return None
    
    reviews = data['reviews']
    
    # Calculate analytics
    analytics = calculate_analytics(data, reviews)
    
    # Generate LLM summary from analytics
    llm_summary = generate_llm_summary(analytics)
    
    # Combine analytics and LLM summary
    full_summary = {
        **analytics,  # Include all analytics
        'llm_summary': llm_summary,  # Add natural language summary
        'generated_at': data['last_updated']
    }
    
    return full_summary


def save_summary(product_id, summary):
    """Save summary to file"""
    summaries_dir = config.REVIEWS_DIR / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    summary_file = summaries_dir / f"product_{product_id}_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary_file


def summarize_all_products():
    """Generate summaries for all products with reviews"""
    review_files = list(config.REVIEWS_DIR.glob("product_*.json"))
    
    print(f"\nGenerating LLM summaries for {len(review_files)} products...\n")
    
    for review_file in review_files:
        # Extract product ID from filename
        product_id = review_file.stem.replace('product_', '')
        
        print(f"Processing product {product_id}...")
        
        summary = summarize_product_reviews(product_id)
        if summary:
            save_summary(product_id, summary)
            print(f"✓ Summarized: {summary['product_name']}")
            if summary['llm_summary']:
                print(f"  📝 {summary['llm_summary']}\n")
    
    print(f"\n✓ All summaries generated!\n")


if __name__ == "__main__":
    summarize_all_products()