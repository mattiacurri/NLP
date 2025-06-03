from docling.document_converter import DocumentConverter

import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

def convert(source, out):
    print(" ==== Converting Document to Markdown ==== ")
    converter = DocumentConverter()
    result = converter.convert(source)

    # Save the converted document to a Markdown file
    if os.path.exists(out):
        print(f" ==== Skipping {out}, already exists. ====")
        return    
    
    with open(out, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
    print(f"Markdown file saved as '{out}'.")