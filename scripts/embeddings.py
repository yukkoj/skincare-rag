"""
embeddings.py (Cloud API Version)
Uses the Gemini API for fast, zero-RAM vector generation.
Includes a one-time setup function to build the Chroma database.
"""
import json
import time
import chromadb
from google import genai
import config

# Initialize the new Gemini GenAI Client
client = genai.Client(api_key=config.GOOGLE_API_KEY)

def get_sentence_embeddings(query: str):
    """Fetches a single vector for the user's search query."""
    response = client.models.embed_content(
        model="gemini-embedding-2", 
        contents=query
    )
    # ChromaDB expects a standard Python list of floats, which this returns
    return response.embeddings[0].values

def build_chroma_database():
    """Reads semantic_profiles.json and builds the Chroma database incrementally."""
    print("📦 Checking ChromaDB status...")
    
    # 1. Setup ChromaDB Local Storage
    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.GENERATED_DIR / "chroma_db"))
    
    collection = chroma_client.get_or_create_collection(
        name="skincare_products",
        embedding_function=None 
    )

    # 2. Load Master Profiles
    profiles_path = config.GENERATED_DIR / "semantic_profiles.json"
    if not profiles_path.exists():
        print("⚠ No semantic_profiles.json found!")
        return

    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # 3. The "Smart Delta" Check
    # Ask Chroma for existing IDs (without downloading the heavy vectors)
    existing_data = collection.get(include=[]) 
    existing_ids = set(existing_data['ids'])

    # Filter the master list to ONLY include products not yet in ChromaDB
    new_profiles = [p for p in profiles if str(p.get("Product_ID")) not in existing_ids]

    if not new_profiles:
        print(f"✓ Database is fully up to date ({len(existing_ids)} products total).")
        return

    print(f"Found {len(new_profiles)} brand new products to embed (Skipping {len(existing_ids)} existing ones)...")
    
    # 4. Batch processing on NEW profiles only
    batch_size = 100 
    
    for i in range(0, len(new_profiles), batch_size):
        batch = new_profiles[i : i + batch_size]
        
        ids = []
        texts = []
        metadatas = []
        
        for p in batch:
            pid = str(p.get("Product_ID"))
            semantic_data = p.get("Semantic_Profile", {})
            
            full_text = (f"Product: {semantic_data.get('Product_Name', 'Unknown')}. "
                         f"Specs: {json.dumps(semantic_data)}. "
                         f"Consensus: {semantic_data.get('Customer_Consensus', '')}")
            
            ids.append(pid)
            texts.append(full_text)
            metadatas.append({"name": semantic_data.get('Product_Name', 'Unknown')})
            
        print(f"  -> Fetching embeddings from Gemini for items {i+1} to {min(i+batch_size, len(new_profiles))}...")
        
        # Get vectors from Gemini
        response = client.models.embed_content(
            model="gemini-embedding-2", 
            contents=texts
        )
        embeddings_list = [emb.values for emb in response.embeddings]
        
        # Save directly to ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        time.sleep(2) 
        
    print("✅ ChromaDB incremental build complete!")

if __name__ == "__main__":
    # When you run this file directly, it builds the database.
    build_chroma_database()