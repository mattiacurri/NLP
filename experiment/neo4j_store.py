import json
from neo4j import GraphDatabase

# Configuration
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
JSON_PATH = "docs_kg/aggregated_knowledge_graph_normalized.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def store_graph(driver, data):
    data['entities'] = list(set(data['entities']))  # Remove duplicates in entities
    data['triples'] = list(set(tuple(triple.items()) for triple in data['triples']))  # Remove duplicates in triples
    data['triples'] = [dict(triple) for triple in data['triples']]  # Convert back to list of dicts
    with driver.session() as session:
        # Optional: clear database
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
        
        print(f"Storing {len(data['entities'])} entities and {len(data['triples'])} relationships in Neo4j...")
        # Create nodes
        for i, node in enumerate(data['entities']):
            session.run(
                "CREATE (n:Node {id: $id, label: $label, properties: $properties})",
                id=node,
                label=node,
                properties=None
            )
        print(f"Created {len(data['entities'])} nodes.")
        
        print(f"Creating {len(data['triples'])} relationships...")
        # Create relationships
        for triple in data['triples']:
            session.run(
                """
                MATCH (a:Node {id: $start}), (b:Node {id: $end})
                CREATE (a)-[r:REL {type: $type, properties: $properties}]->(b)
                """,
                start=triple["entita1"],
                end=triple["entita2"],
                type=triple["relazione"],
                properties=triple["fonte"]
            )

def main():
    data = load_json(JSON_PATH)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    store_graph(driver, data)
    driver.close()
    print("Graph stored in Neo4j.")
    

if __name__ == "__main__":
    main()