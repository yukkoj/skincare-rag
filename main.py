"""
Main entry point for skincare search pipeline
Runs: scraper → summarizer → semantic_profiler → embeddings
"""
from scripts import scraper, summarizer, semantic_profile, embeddings

def main():
    """Run the full pipeline: scrape → summarize → profile → embed"""
    
    print("\n" + "="*60)
    print("SKINCARE SEARCH PIPELINE")
    print("="*60 + "\n")
    
    # ==================== STEP 1: SCRAPE REVIEWS ====================
    print("="*60)
    print("STEP 1: Scraping Reddit Reviews")
    print("="*60 + "\n")
    
    try:
        scrape_stats = scraper.scrape_all_products(
            max_products=None,  # Change to 5 for testing
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
        print("Continuing to semantic profiling...\n")

    # ==================== STEP 3: COMPILE SEMANTIC PROFILES ====================
    print("="*60)
    print("STEP 3: Compiling Semantic Profiles")
    print("="*60 + "\n")
    
    try:
        semantic_profile.compile_profiles()
        print("✓ Semantic profiles compiled successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Profiling error: {e}")
        print("Cannot proceed to embeddings without profiles. Exiting.\n")
        return

    # ==================== STEP 4: CREATE EMBEDDINGS ====================
    print("="*60)
    print("STEP 4: Creating Embeddings & Search Index")
    print("="*60 + "\n")
    
    try:
        print("Processing semantic profiles into vector chunks...")
        embeddings.ensure_chunk_embeddings()
        
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