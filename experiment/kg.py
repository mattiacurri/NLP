import os

import tqdm
import json

from prompts import EXTRACTION_SYSTEM_PROMPT

import concurrent.futures

from google import genai

from dotenv import load_dotenv

from pydantic import BaseModel

from rateguard import rate_limit

class Triple(BaseModel):
  entita1: str
  relazione: str
  entita2: str
  fonte: str

class KnowledgeGraph(BaseModel):
  titolo_documento: str
  triples: list[Triple]

load_dotenv()


@rate_limit(rpm=9)
def generate_knowledge_graph(document: str) -> dict:
    client = genai.Client(
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{document[0]}",
        config={
        "response_mime_type": "application/json",
        "response_schema": KnowledgeGraph,
        "temperature": 0.0,
        "system_instruction": EXTRACTION_SYSTEM_PROMPT,
        "thinking_config": {
          "thinking_budget": 0
        }
      },
    )
    return response.text

if not os.path.exists("docs_kg"):
    os.makedirs("docs_kg")
    
list_of_apis = os.getenv("API_KEYS").split(",")

files = []
for i, file in enumerate(tqdm.tqdm(os.listdir("docs_md/out_compacted"), desc="Processing knowledge graphs", unit="file")):
    #chunks = []
    if os.path.exists(os.path.join("docs_kg", f"{file.replace('.md', '.json')}")):
      print(f"{file} has already been processed. Skipping...")
      continue
    else:
      print(f"Processing {file}...")
      with open(os.path.join("docs_md/out_compacted", file), "r", encoding="utf-8") as f:
          example_document = f.read()
          files.append((example_document, file))

files = [files[0:50], files[50:100], files[100:]]

for i, file in enumerate(files):
  print(f"Total files to process: {len(file)}")     
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
      # Submit the task to the executor
      results = list(tqdm.tqdm(executor.map(generate_knowledge_graph, file), total=len(file), desc="Generating knowledge graphs", unit="file"))
      # Wait for the result
  results = list(results)

  for i, kg in enumerate(results):
    with open(os.path.join("docs_kg", f"{file[i][1].replace('.md', '.json')}"), "w", encoding="utf-8") as f:
        # format in json
        try:
          j = json.loads(kg)
          for triple in j["triples"]:
            if j['titolo_documento'] not in triple["fonte"]:
              triple["fonte"] = f"[{j['titolo_documento']}]: {triple['fonte']}"
          #chunks.append(j)
          json.dump(j, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
          print(f"Error decoding JSON for {file[i][1]}: {e}")
          continue
      
def aggregate_knowledge_graphs(files: list[str]) -> dict:
    combined_triples = []
    for file in files:
        print(f"Aggregating knowledge graph from {file}...")
        with open(os.path.join("docs_kg", file), "r", encoding="utf-8") as f:
            kg = json.load(f)
            combined_triples.extend(kg["triples"])
    
    # collect entities and relations
    entities = []
    relations = []
    
    for triple in combined_triples:
        entities.append(triple["entita1"])
        entities.append(triple["entita2"])
        relations.append(triple["relazione"])
    
    # Remove duplicate triples (dicts) by serializing to tuple of sorted items
    seen = set()
    unique_triples = []
    for triple in combined_triples:
      # Convert dict to a tuple of sorted items for hashing
      triple_tuple = tuple(sorted(triple.items()))
      if triple_tuple not in seen:
        seen.add(triple_tuple)
        unique_triples.append(triple)
    combined_triples = unique_triples
    # # Remove triples with the same subject and object 
    combined_triples = [edge for edge in combined_triples if edge["entita1"] != edge["entita2"]]
    # Create a new knowledge graph with the combined triples
    with open("docs_kg/aggregated_knowledge_graph.json", "w", encoding="utf-8") as f:
        aggregated_kg = {
            "entities": list(set(entities)),
            "relations": list(set(relations)),
            "triples": combined_triples
        }
        json.dump(aggregated_kg, f, indent=2, ensure_ascii=False)
    # Save a version of the aggregated KG without the "fonte" field in triples
    triples_no_fonte = [
      {k: v for k, v in triple.items() if k != "fonte"}
      for triple in combined_triples
    ]

    aggregated_kg_no_fonte = {
      "entities": list(set(entities)),
      "relations": list(set(relations)),
      "triples": triples_no_fonte
    }
    with open("docs_kg/aggregated_knowledge_graph_no_fonte.json", "w", encoding="utf-8") as f_no_fonte:
      json.dump(aggregated_kg_no_fonte, f_no_fonte, indent=2, ensure_ascii=False)

aggregate_knowledge_graphs(os.listdir("docs_kg"))

print(" ==== Knowledge Graphs generated and saved successfully! ==== ")