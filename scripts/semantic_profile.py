import json
import config

def build_semantic_dict(product: dict, llm_summary: str) -> dict:
    """Returns a structured dictionary instead of a raw text block."""
    return {
        "Product_Name": product.get("Product_Name"),
        "Brand": product.get("Brand"),
        "Category": product.get("Category"),
        "Price_USD": product.get("Price_USD"),
        "Texture": product.get("Texture"),
        "Skin_Types": product.get("Skin_Type", []),
        "Key_Ingredients": product.get("Key_Ingredients", []),
        "Fragrance_Free": product.get("Fragrance_Free"),
        "Non_Comedogenic": product.get("Non_Comedogenic"),
        "SPF": product.get("SPF"),
        "Rating": product.get("Rating"),
        "Customer_Consensus": llm_summary if llm_summary else "No summary available"
    }

def compile_profiles(
    products_file = config.PRODUCTS_FILE, 
    summaries_dir = config.SUMMARIES_DIR, 
    output_file   = config.GENERATED_DIR / "semantic_profiles.json"
):
    # 1. Load the original specs using pathlib's built in .exists()
    if not products_file.exists():
        print(f"Error: Could not find {products_file}")
        return
        
    with open(products_file, 'r', encoding='utf-16') as f:
        products = json.load(f)

    final_profiles = []
    
    # Check if summaries directory exists
    if not summaries_dir.exists():
        print(f"Error: The directory '{summaries_dir}' does not exist.")
        return

    # 2. Merge them together
    print("Stitching specs and summaries together...")
    for prod in products:
        pid = prod.get("Product_ID")
        summary_filename = f"product{pid}_summary.json"
        summary_file = summaries_dir / summary_filename
        summary_text = "" 
        
        # 3. Load the individual LLM summary
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as sf:
                    summary_data = json.load(sf)
                    
                    llm_text = summary_data.get('llm_summary', '')
                    phrases = [p['phrase'] for p in summary_data.get('top_phrases', [])]
                    
                    if llm_text or phrases:
                        summary_text = f"{llm_text}"
                        if phrases:
                            summary_text += f"Key themes: {', '.join(phrases)}."
            except Exception as e:
                print(f"  ⚠ Error reading {summary_filename}: {e}")
        else:
            print(f"  ℹ No summary file found for '{pid}'. Profiling without summary.")
        
        # 4. Build the structured dictionary
        # Now summary_text is guaranteed to be empty if no file was found
        profile_data = build_semantic_dict(prod, summary_text.strip())
    
        final_profiles.append({
            "Product_ID": pid,
            "Semantic_Profile": profile_data 
        })

    # 5. Save the final compiled file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_profiles, f, indent=2)
        
    print(f"\n✅ Success! Compiled {len(final_profiles)} full semantic profiles into {output_file}.")
    
# Run the compiler
if __name__ == "__main__":
    compile_profiles()