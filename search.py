"""
search.py
Interactive terminal script to search your skincare database.
Loads all product attributes dynamically and feeds them to Gemini for context-rich answers.
"""
import json
import datetime
import profile
from urllib import response
import config
import google.generativeai as genai
from scripts import embeddings

# Set up Gemini
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
# Lazy-loading of the full database and indices to avoid repeated disk reads
full_db = None
chunks, bm25_index, faiss_index = None, None, None

def load_semantic_profiles():
    """Loads the compiled semantic_profiles.json master file."""
    try:
        profiles_path = config.GENERATED_DIR / "semantic_profiles.json"
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            # Create a lookup map: { "1": {...profile_data...}, "2": {...} }
            return {str(p["Product_ID"]): p["Semantic_Profile"] for p in profiles}
    except Exception as e:
        print(f"  ⚠ Error loading semantic profiles: {e}")
        return {}

def generate_ai_recommendation(query, top_products, full_db):
    """Feeds structured product profiles to Gemini for a high-accuracy response."""
    
    # 1. Load the prompt template
    try:
        prompt_template = (config.PROMPTS_DIR / "recommendation.txt").read_text(encoding='utf-8')
    except:
        prompt_template = "Act as a skincare expert. Provide a concise recommendation based on the data below."

    # 2. Build context from structured profiles
    context_lines = []
    context_lines.append(f"User Query: '{query}'\n\nTop Product Matches:\n")
    
    for product_id, distance in top_products[:5]:
        profile = full_db.get(str(product_id))
        if profile:
            context_lines.append(f"--- Product: {profile.get('Product_Name')} ---")
            for key, value in profile.items():
                context_lines.append(f"{key.replace('_', ' ')}: {value}")
            context_lines.append("") # Empty line separator

    full_prompt = f"{prompt_template}\n\n" + "\n".join(context_lines)

    try:
        response = model.generate_content(full_prompt, stream=True)
        full_text_response = ""
        for chunk in response:
            # Check if chunk has valid text to avoid potential errors
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_text_response += chunk.text # Collect text for the history logger
        print("\n") # Print a final newline after streaming completes
        return full_text_response # Return the full compiled string
    except Exception as e:
        return f" ⚠ Could not generate AI summary: {e}"

def save_search_history(query, response, top_products, full_db):
    """
    Saves the search session to a local JSON file with product names 
    and a cleaned AI response.
    """
    # 1. Map IDs to Names for a readable history log
    readable_results = []
    for pid, dist in top_products:
        # Fetch the full profile from your master DB
        profile = full_db.get(str(pid))
        readable_results.append({
            "product_id": pid,
            "name": profile.get("Product_Name", "Unknown") if profile else "Not Found",
            "distance": round(dist, 4)
        })

    # 2. Format the AI response (stripping excessive whitespace)
    # This keeps your JSON file clean and prevents huge blocks of empty space
    formatted_response = response.strip()

    history_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "results_summary": readable_results,
        # Split the response by double-newlines to create a list of paragraphs
        "ai_response_paragraphs": [p.strip() for p in response.split('\n\n') if p.strip()]
    }

    # 3. Append to history file (Creates it automatically if it doesn't exist)
    log_path = config.GENERATED_DIR / "search_history.jsonl"  # Changed extension to .jsonl to match format

    # Ensure the parent directory exists first, just in case
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Opening with 'a' creates the file automatically if it's missing
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

def main():
    global full_db, chunks, bm25_index, faiss_index
    
    print("\n" + "="*60)
    print("✨ AI SKINCARE PRODUCT SEARCH ✨")
    print("="*60)

    while True:
        query = input("\nSearch (or 'q' to quit): ")
        if query.lower() in ['quit', 'q', 'exit']: break
        if not query.strip(): continue

        # LOADS DATA ONLY ON THE FIRST SEARCH RUN
        if full_db is None:
            print("📦 Initializing database and indices (First time setup)...")
            full_db = load_semantic_profiles()
            chunks, bm25_index, faiss_index = embeddings.load_all_indices()
            
        print(f"🔍 Searching...")
        
        # 1. Retrieve raw chunks from embeddings
        raw_results = embeddings.search_products(
            query, 
            chunks=chunks, 
            bm25=bm25_index, 
            index=faiss_index, 
            k=20)
        
        # 2. Aggregate chunks to unique IDs
        arregated_products = embeddings.aggregate_chunk_hits(raw_results)
        top_products = arregated_products[:10]  # Limit to top 10 for AI context
        # 3. Generate recommendation using structured profiles
        ai_response = generate_ai_recommendation(query, top_products, full_db)
        save_search_history(query, ai_response, top_products, full_db)

if __name__ == "__main__":
    main()