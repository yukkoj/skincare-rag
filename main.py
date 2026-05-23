"""
Main entry point for skincare search pipeline
Runs: scraper → summarizer → embeddings
"""
from scripts import scraper, summarizer, embeddings


def main():
    """Run the full pipeline: scrape → summarize → embed"""
    
    print("\n" + "="*60)
    print("SKINCARE SEARCH PIPELINE")
    print("="*60 + "\n")
    
    """  
    # ==================== STEP 1: SCRAPE REVIEWS ====================
    print("="*60)
    print("STEP 1: Scraping Reddit Reviews")
    print("="*60 + "\n")
    
    try:
        scrape_stats = scraper.scrape_all_products(
            max_products=10,  # Change to 5 for testing
            skip_existing=True,  # Skip products that already have reviews
        )
        
        print(f"\n✓ Scraping complete!")
        print(f"  Scraped: {scrape_stats['scraped']} products")
        print(f"  Skipped: {scrape_stats['skipped']} products")
        print(f"  Errors: {scrape_stats['errors']} products\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure data/products/products.json exists!")
        return
    except Exception as e:
        print(f"\n❌ Scraping error: {e}")
        return
    
    # ==================== STEP 2: GENERATE SUMMARIES ====================
    print("="*60)
    print("STEP 2: Generating LLM Summaries")
    print("="*60 + "\n")
    
    try:
        summarizer.summarize_all_products(skip_existing=True)
        print("✓ Summaries generated!\n")
        
    except Exception as e:
        print(f"\n❌ Summarization error: {e}")
        print("Continuing to embeddings...\n")
    """
    # ==================== STEP 3: CREATE EMBEDDINGS ====================
    print("="*60)
    print("STEP 3: Creating Embeddings & Search Index")
    print("="*60 + "\n")
    
    try:
        import json
        import config
        from pathlib import Path

        # 1. Load the products list that core_embeddings needs
        with open(config.PRODUCTS_FILE, 'r', encoding='utf-16') as f:
            products_data = json.load(f)

        # 2. Define the paths and model name your function expects
        model_name = 'all-MiniLM-L6-v2'
        embeddings_file = str(config.EMBED_DIR / "embeddings.npy")
        metadata_file = str(config.EMBED_DIR / "chunks.pkl")

        print("Processing text chunks into embeddings...")
        embeddings.ensure_chunk_embeddings()
        
        # 3. Build the FAISS database so the embeddings are searchable
        print("Building FAISS search index...")
        embeddings.build_faiss_index()
        
        print("\n✓ Embeddings and search index created successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Embedding error: {e}\n")
    
    # ==================== PIPELINE COMPLETE ====================
    print("="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)



if __name__ == "__main__":
    main()