from convert import convert_pymupdfllm, convert, compact_chunk

import os

from tqdm import tqdm

def run_pipeline():
    print(" ==== Running Pipeline ==== ")
    
    # Step 1: Convert all the documents to Markdown
    for filename in tqdm(os.listdir("docs"), desc="Converting documents", unit="file", position=0, leave=True):
        # Skip files that are already converted
        base_name = filename.replace('.pdf', '')
        md_path = os.path.join("docs_md", f"{base_name}_0.md")
        #if os.path.exists(md_path):
            #print(f" ==== Skipping {filename}, already converted. ====")
            #continue
        
        if filename.endswith(".pdf"):
            source = os.path.join("docs", filename)
            out = os.path.join("docs_md", filename.replace(".pdf", ""))
            #convert(source, out)
            
            out = "docs_md/"
            compact_chunk(filename.replace(".pdf", ""), out)

    # Step 2: Profit
    print(" ==== Pipeline completed successfully! ==== ")
    
if __name__ == "__main__":
    run_pipeline()