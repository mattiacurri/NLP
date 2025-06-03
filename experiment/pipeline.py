from scrape import scrape 
from convert import convert

import os

from tqdm import tqdm

def run_pipeline():
    print(" ==== Running Pipeline ==== ")
    # Step 1: Scrape the documents
    scrape()
    
    # Step 2: Convert all the documents to Markdown
    for filename in tqdm(os.listdir("docs"), desc="Converting documents", unit="file", position=0, leave=True):
        if filename.endswith(".pdf"):
            source = os.path.join("docs", filename)
            out = os.path.join("docs_md", filename.replace(".pdf", ".md"))
            convert(source, out)

    # Step 3: Profit
    print(" ==== Pipeline completed successfully! ==== ")
    
if __name__ == "__main__":
    run_pipeline()