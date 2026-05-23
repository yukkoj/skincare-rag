"""
search.py
Interactive terminal script to search your skincare database.
Loads all product attributes dynamically and feeds them to Gemini for context-rich answers.
"""
import json
import config
import google.generativeai as genai
from scripts import embeddings

# Set up Gemini
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def load_all_product_properties():
    """Loads EVERY property for every product dynamically from your products file."""
    database = {}
    try:
        if config.PRODUCTS_FILE.exists():
            with open(config.PRODUCTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle List structure
                if isinstance(data, list):
                    for item in data:
                        pid = item.get('product_id', item.get('id', ''))
                        if pid:
                            database[str(pid)] = item
                            
                # Handle Dictionary structure
                elif isinstance(data, dict):
                    for pid, details in data.items():
                        if isinstance(details, dict):
                            database[str(pid)] = details
                        else:
                            database[str(pid)] = {"product_name": details}
                            
    except Exception as e:
        print(f"  ⚠ Error reading products.json properties: {e}")
        
    return database

def generate_ai_recommendation(query, top_products, results, full_db):
    """Feeds text chunks AND all product properties to Gemini for an intelligent, detailed breakdown."""
    
    prompt = f"A user is looking for skincare advice based on this search query: '{query}'\n\n"
    prompt += "Here are the top matches retrieved from our database, including full product specs:\n\n"
    
    # --- FIX: We now pass the top 6 products so the AI can list 3 Top + 3 "Also Considered" ---
    for product_id, score in top_products[:6]:
        properties = full_db.get(product_id, {})
        
        # Get the name safely
        name = properties.get('product_name', properties.get('name', product_id.replace('_', ' ').title()))
        
        prompt += f"=== PRODUCT: {name} ===\n"
        prompt += "Product Specifications:\n"
        
        # Dump every single attribute
        for key, value in properties.items():
            if key in ['product_id', 'id', 'product_name', 'name']:
                continue
            clean_key = key.replace('_', ' ').title()
            prompt += f"- {clean_key}: {value}\n"
            
        prompt += "User Review Highlights:\n"
        
        for r in results:
            if r['product_id'] == product_id:
                clean_text = r['text'].replace(product_id, name).replace(product_id.title(), name)
                prompt += f"- {clean_text}\n"
        prompt += "\n"
        
    # --- NEW: Your updated prompt rules ---
    prompt += """Based on BOTH the product specifications and the user review highlights, write a highly professional, friendly, and structured response for the user. 
    
    Strict Formatting Rules:
    1. Start with a warm, brief 1-sentence intro acknowledging their specific query. Eliminate products with no name (e.g., "unknown") from your response.
    2a. Analyze the reviews and choose the top 3 products with positive reviews that best match their needs based on performance and price.
    2b. List the next 3 products in order of recommendation under a heading called "Also Considered".
    2c. Use a bulleted list for all recommendations. Bold the **Product Name** AND explicitly state its price right next to it (e.g., **Product Name** - $15.00). Add a blank line between each recommendation for clarity.
    3. Explain exactly *why* it fits their query based on what users said and its specific ingredients/features.
    4. NEVER mention internal IDs like "product1", "product_id", or "unknown".
    5. Do not say "database mentions" or "chunk hits" — frame it naturally as "Reviewers mention..." or "Users found...".
    6. Keep it concise, organized, and helpful.
    7. End with a friendly closing line encouraging them to explore the products."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"  ⚠ Could not generate AI summary: {e}"

def main():
    # Load the entire properties dictionary once at startup
    full_db = load_all_product_properties()
    
    print("\n" + "="*60)
    print("✨ AI SKINCARE PRODUCT SEARCH ✨")
    print("="*60)
    print("Type what you are looking for (e.g., 'gentle cream for acne').")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Search: ")
        
        if query.lower() in ['quit', 'q', 'exit']:
            print("Exiting search...")
            break
            
        if not query.strip():
            continue
            
        print(f"\n🔍 Searching database for: '{query}'...")
        
        try:
            # --- FIX: Increased 'k' to 20 so the database pulls plenty of reviews to analyze ---
            results = embeddings.find_similar_chunks(query, k=20)
            
            if not results:
                print("No matches found for that query.")
                continue
                
            top_products = embeddings.aggregate_chunk_hits(results)
            
            print("🧠 Analyzing reviews and selecting top matches...\n")
            ai_response = generate_ai_recommendation(query, top_products, results, full_db)
            
            print(ai_response)
            print("\n" + "-"*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Search error: {e}")
            print("Did you run `main.py` first to generate the embeddings?")

if __name__ == "__main__":
    main()