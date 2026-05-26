"""
search.py (Cloud API Version)
Interactive terminal script to search your skincare database using ChromaDB & Gemini.
"""
import json
import datetime
import config
import google.generativeai as genai
import chromadb
from scripts import embeddings

# Set up Gemini for the text generation
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Lazy-loading globals
full_db = None
chroma_client = None
collection = None

def load_semantic_profiles():
    """Loads the compiled semantic_profiles.json master file."""
    try:
        profiles_path = config.GENERATED_DIR / "semantic_profiles.json"
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            return {str(p["Product_ID"]): p["Semantic_Profile"] for p in profiles}
    except Exception as e:
        print(f"  ⚠ Error loading semantic profiles: {e}")
        return {}

def generate_ai_recommendation(query, top_products, full_db):
    """Feeds structured product profiles to Gemini for a high-accuracy response."""
    try:
        prompt_template = (config.PROMPTS_DIR / "recommendation.txt").read_text(encoding='utf-8')
    except:
        prompt_template = "Act as a skincare expert. Provide a concise recommendation based on the data below."

    context_lines = []
    context_lines.append(f"User Query: '{query}'\n\nTop Product Matches:\n")
    
    for product_id, distance in top_products[:5]:
        profile = full_db.get(str(product_id))
        if profile:
            context_lines.append(f"--- Product: {profile.get('Product_Name')} ---")
            for key, value in profile.items():
                context_lines.append(f"{key.replace('_', ' ')}: {value}")
            context_lines.append("") 

    full_prompt = f"{prompt_template}\n\n" + "\n".join(context_lines)

    try:
        # TEMPERATURE 0.0 FIX: Forces Gemini to act deterministically like a search engine
        response = model.generate_content(
            full_prompt, 
            stream=True,
            generation_config={"temperature": 0.0} 
        )
        full_text_response = ""
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_text_response += chunk.text 
        print("\n") 
        return full_text_response 
    except Exception as e:
        return f" ⚠ Could not generate AI summary: {e}"

def save_search_history(query, response, top_products, full_db):
    """Saves the search session to a local JSONL file."""
    readable_results = []
    for pid, dist in top_products:
        profile = full_db.get(str(pid))
        readable_results.append({
            "product_id": pid,
            "name": profile.get("Product_Name", "Unknown") if profile else "Not Found",
            "distance": round(dist, 4)
        })

    formatted_response = response.strip()

    history_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "results_summary": readable_results,
        "ai_response_paragraphs": [p.strip() for p in response.split('\n\n') if p.strip()]
    }

    log_path = config.GENERATED_DIR / "search_history.jsonl" 
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

def main():
    global full_db, chroma_client, collection
    
    print("\n" + "="*60)
    print("✨ AI SKINCARE PRODUCT SEARCH ✨")
    print("="*60)

    while True:
        # BUFFER FLUSH FIX: Ensures the prompt actually paints to the screen
        print("\nSearch (or 'q' to quit): ", end="", flush=True)
        query = input()
        
        if query.lower() in ['quit', 'q', 'exit']: break
        if not query.strip(): continue

        # Connects to the database instantly
        if full_db is None:
            full_db = load_semantic_profiles()
            chroma_client = chromadb.PersistentClient(path=str(config.GENERATED_DIR / "chroma_db"))
            collection = chroma_client.get_collection(
                name="skincare_products", 
                embedding_function=None 
            )
            
        print(f"🔍 Searching...")
        
        # 1. Ask Gemini API for the vector
        query_vector = embeddings.get_sentence_embeddings(query)
        
        # 2. Search ChromaDB
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        
        # 3. Reformat the results for the AI prompt
        top_products = []
        if search_results['ids'] and len(search_results['ids'][0]) > 0:
            result_ids = search_results['ids'][0]
            result_distances = search_results['distances'][0]
            for pid, dist in zip(result_ids, result_distances):
                top_products.append((pid, dist))
        
        if not top_products:
            print("No matching products found in the database.")
            continue

        # 4. Generate recommendation
        ai_response = generate_ai_recommendation(query, top_products, full_db)
        save_search_history(query, ai_response, top_products, full_db)

if __name__ == "__main__":
    main()