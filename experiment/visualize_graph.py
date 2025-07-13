import json
import networkx as nx
import matplotlib.pyplot as plt

import tqdm

print("Loading JSON data and constructing the graph...")
# Carica JSON da file
with open('docs_kg/aggregated_knowledge_graph_normalized_no_fonte.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print("JSON data loaded successfully.")

# Costruzione grafo
G = nx.DiGraph()

# Aggiunta nodi
print("Adding nodes to the graph...")
for entity in tqdm.tqdm(data["entities"], desc="Adding nodes", unit="node"):
    G.add_node(entity, label=entity)
print(f"Total nodes added: {len(data['entities'])}")

# Aggiunta archi (relations come [soggetto, relazione, oggetto])
print("Adding edges to the graph...")
for relation in tqdm.tqdm(data["triples"], desc="Adding edges", unit="edge"):
    G.add_edge(relation['entita1'], relation['entita2'], label=relation['relazione'])
print(f"Total edges added: {len(data['triples'])}")

# Disegno grafo
print("Saving the graph...")
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')
edge_labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(150, 60))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', font_size=8, font_color='black', font_weight='bold', arrows=True, arrowsize=5)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.savefig('grafo_cleaned_normalized_3.jpg', format='jpg', dpi=300)
plt.close()
print("Graph saved.")