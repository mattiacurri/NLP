from kg_gen import KGGen

import os

# Initialize KGGen with optional configuration
kg = KGGen(
  model="ollama_chat/gemma3:4b",  # Default model
  temperature=0.0,        # Default temperature
)

# text_input, take a file from the folder docs_md

with open(os.path.join("docs_md", "2006_0163_codice_allegati.md"), "r", encoding="utf-8") as file:
    text_input = file.read()

print(" ==== Generating Knowledge Graph ==== ")
graph_1 = kg.generate(
  input_data=text_input,
)

print(graph_1)