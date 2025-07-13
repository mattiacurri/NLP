from docling.document_converter import DocumentConverter

import os

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import smolvlm_picture_description

import pymupdf

# Docling parameters
pipeline_options = PdfPipelineOptions()
# pipeline_options.do_formula_enrichment = True
# pipeline_options.generate_picture_images = False
# pipeline_options.images_scale = 2
# pipeline_options.do_picture_classification = False
# pipeline_options.do_picture_description = False
# pipeline_options.picture_description_options = smolvlm_picture_description

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )

# CPU goes brrr
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

def convert(source, out):
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    # Step 2. Validate the source file
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file '{source}' does not exist.")

    if not source.endswith(".pdf"):
        raise ValueError(f"Source file '{source}' is not a PDF document.")    
    
    print(f" ==== Converting {source} to {out} ==== ")
    # Step 3. Convert the PDF document to Markdown
    doc = converter.convert(source).document
    
    title = doc.name
    
    EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    tokenizer: BaseTokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID, padding_side='left', max_tokens=8192),
    )
    chunker = HybridChunker(tokenizer=tokenizer,
        serializer_provider=MDTableSerializerProvider())
    chunk_iter = chunker.chunk(dl_doc=doc)
    ll = list(chunk_iter)  # consume the iterator to get the length
    for i, chunk in enumerate(ll):
        ser_txt = chunker.contextualize(chunk=chunk)
        # add at the start of every chunk the title of the document
        ser_txt = f"# {title} (Parte {i + 1}/{len(ll)})\n\n" + ser_txt
        with open(f"{out}_{i}.md", "a", encoding="utf-8") as f:
            f.write(ser_txt + "\n\n")

import glob
import re
import os

def compact_chunk(source, out, group_size=3):
    """
    Compact the chunks by removing empty lines and unnecessary whitespace.
    """
    
    # Source : name of the file
    # Find all chunks, that are named source_*.md
    
    # I need to find all files that match the pattern source_*.md
    # ignore out, it's not useful
    base = os.path.basename(source)
    pattern = f"{base}_*.md"
    files = glob.glob(os.path.join(out, pattern))
    
    # sort files by name
    files.sort(key=lambda x: int(re.search(r'_(\d+)\.md$', x).group(1)) if re.search(r'_(\d+)\.md$', x) else 0)

    if not files:
        print(f"No chunk files found matching {pattern}")
        return

    out_dir = os.path.join(out, "out_compacted")
    os.makedirs(out_dir, exist_ok=True)
    if len(files) < group_size:
        for fname in files:
            with open(fname, "r", encoding="utf-8") as f:
                lines = [line.rstrip() for line in f if line.strip()]
            compacted_text = "\n".join(lines) + "\n"
            out_file = os.path.join(out_dir, f"{source}_compacted.md")
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(compacted_text)
    else:
        for idx in range(0, len(files), group_size):
            group = files[idx:idx+group_size]
            compacted = []
            for fname in group:
                with open(fname, "r", encoding="utf-8") as f:
                    lines = [line.rstrip() for line in f if line.strip()]
                    compacted.extend(lines)
            compacted_text = "\n".join(compacted) + "\n"
            out_file = os.path.join(out_dir, f"{base}_compacted_{idx}.md")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(compacted_text)
    

def convert_pymupdf(source, out):
    doc = pymupdf.open(source)  # open a document
    with open(out, "wb") as fout:  # create a text output
        for page in doc:  # iterate the document pages
            text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
            fout.write(text)  # write text of page
            
            
def convert_pymupdfllm(source, out):
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(source)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md_text)
        