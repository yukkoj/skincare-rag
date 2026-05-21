"""
Main entry point for Reddit scraper
"""
from scripts import scraper

def main():
    """Run the scraper"""
    try:
        # Run scraper
        stats = scraper.scrape_all_products(
            max_products=None,  # Change to 5 for testing
            skip_existing=True
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("✓ Scraping complete!")
        print(f"  Scraped: {stats['scraped']} products")
        print(f"  Skipped: {stats['skipped']} products")
        print(f"  Errors: {stats['errors']} products")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure data/products/products.json exists!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()