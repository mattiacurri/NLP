import json

from json import load
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

from sklearn.cluster import DBSCAN

from pydantic import BaseModel, Field

class Entities(BaseModel):
    entities: list[str] = Field(
        ...,
        description="Lista di entità uniche e normalizzate."
    )

class Normalizer:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', eps=0.05):
        """
        - model_name: Model name from Hugging Face (e.g. "Qwen/Qwen3-Embedding-4B").
        - eps: DBSCAN epsilon.
        """
        
        self.model = SentenceTransformer(model_name)

        self.eps = eps
        self.new_entities = set()  # Per tenere traccia delle nuove entità aggiunte
        
        # Stato del sistema
        self.centroids = {}
        self.representatives = {}
        self.normalization_map = {}
        self.noise_embeddings = np.array([])
        self.noise_words = []
        self.next_cluster_id = 0

    def _get_embedding(self, word):
        try:
            return self.model.encode(word)
        except Exception:
            return None

    def _compute_distance_matrix(self, embeddings):
        """
        Calcola la matrice delle distanze pairwise per DBSCAN.
        La distanza usata è 1 - cosine_similarity.
        """
        similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
        
        distance_matrix = 1 - similarity_matrix

        np.clip(distance_matrix, 0, None, out=distance_matrix)

        return distance_matrix 

    def initialize(self, initial_vocabulary):
        """
        Step 1: Perform initial clustering on a vocabulary.
        - initial_vocabulary: a list of unique words/nodes.
        """
        print(f"Initialization with a vocabulary of {len(initial_vocabulary)} words.")

        valid_words = [word for word in initial_vocabulary if self._get_embedding(word) is not None]
        embeddings = np.array([self._get_embedding(word) for word in valid_words])

        dbscan = DBSCAN(metric='precomputed', eps=self.eps, min_samples=2, n_jobs=-1).fit(self._compute_distance_matrix(embeddings))

        # Initial clustering results
        unique_labels = set(dbscan.labels_)
        for label in unique_labels:
            indices = np.where(dbscan.labels_ == label)[0]
            cluster_words = [valid_words[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            print(label, cluster_words, cluster_embeddings.shape)
            if label == -1: # Noise
                self.noise_words.extend(cluster_words)
                self.noise_embeddings = cluster_embeddings
                for word in cluster_words:
                    self.normalization_map[word] = word
            else: # Valid Cluster
                centroid = np.mean(cluster_embeddings, axis=0)
                # Word near the centroid
                similarities = util.cos_sim(cluster_embeddings, centroid.reshape(1, -1))
                representative = cluster_words[np.argmax(similarities)]

                self.centroids[label] = centroid
                self.representatives[label] = representative
                self.next_cluster_id = max(self.next_cluster_id, label + 1)
                print(f"Cluster {label}: {representative} con {len(cluster_words)} parole.")
                for word in cluster_words:
                    self.normalization_map[word] = representative
        
        print("Initialization completed.")
        print(f"Found {len(self.centroids)} clusters and {len(self.noise_words)} noise words.")

    def normalize(self, new_word):
        """
        Step 2: Normalize a new word.
        """
        # 1. Cache
        if new_word in self.normalization_map:
            return self.normalization_map[new_word]

        # 2. Embedding generation
        new_embedding = self._get_embedding(new_word)
        if new_embedding is None:
            self.normalization_map[new_word] = new_word
            return new_word

        # 3. Matching with existing clusters
        if self.centroids:
            max_sim = -1
            best_cluster = None
            for label, centroid in self.centroids.items():
                sim = util.cos_sim(new_embedding.reshape(1, -1), centroid.reshape(1, -1)).item()
                if sim > max_sim:
                    max_sim = sim
                    best_cluster = label

            if 1-max_sim <= self.eps:
                representative = self.representatives[best_cluster]
                self.normalization_map[new_word] = representative
                # Opzionale: aggiornare il centroide
                return representative

        # 4. Noise word handling
        if len(self.noise_words) > 0:
            # Cosim for each noise word
            for i, noise_word in enumerate(self.noise_words):
                sim = util.cos_sim(new_embedding.reshape(1, -1), self.noise_embeddings[i].reshape(1, -1)).item()
                if sim > 1 - self.eps:  # Soglia per considerare simile
                    # If noise word is similar, replace it
                    self.normalization_map[new_word] = noise_word
                    # Remove the noise word from the list
                    self.noise_words.pop(i)
                    self.noise_embeddings = np.delete(self.noise_embeddings, i, axis=0)
                    return noise_word

        # 5. New noise classification
        self.noise_words.append(new_word)
        if len(self.noise_embeddings) == 0:
            self.noise_embeddings = new_embedding.reshape(1, -1)
        else:
            self.noise_embeddings = np.vstack([self.noise_embeddings, new_embedding])
        self.normalization_map[new_word] = new_word
        
        return new_word
    
    def save_state(self, filepath):
        state = {
            "centroids": self.centroids,
            "representatives": self.representatives,
            "normalization_map": self.normalization_map,
            "noise_embeddings": self.noise_embeddings,
            "noise_words": self.noise_words,
            "next_cluster_id": self.next_cluster_id,
            "eps": self.eps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to {filepath}")

    def load_state(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.centroids = state["centroids"]
        self.representatives = state["representatives"]
        self.normalization_map = state["normalization_map"]
        self.noise_embeddings = state["noise_embeddings"]
        self.noise_words = state["noise_words"]
        self.next_cluster_id = state["next_cluster_id"]
        self.eps = state["eps"]
        print(f"State loaded from {filepath}")

    def view_state(self):
        print(f"Centroids: {self.centroids}")
        print(f"Representatives: {self.representatives}")
        print(f"Normalization Map: {self.normalization_map}")
        print(f"Noise Words: {self.noise_words}")
        print(f"Next Cluster ID: {self.next_cluster_id}")
        print(f"Epsilon: {self.eps}")
    
# --- ESEMPIO DI UTILIZZO ---

model_name = 'Qwen/Qwen3-Embedding-0.6B'

# 1. Loading the graph
with open('docs_kg/aggregated_knowledge_graph.json', 'r', encoding='utf-8') as f:
    graph_dict = load(f)
    
with open('docs_kg/aggregated_knowledge_graph_no_fonte.json', 'r', encoding='utf-8') as f:  
    graph_dict_no_fonte = load(f)
relation_normalizer = Normalizer(model_name=model_name)
#relation_normalizer.initialize(graph_dict['relations'])
#relation_normalizer.save_state('relation_normalizer_state.pkl')

relation_normalizer.load_state('relation_normalizer_state.pkl')

# we explore graph_dict['relations'] and normalize them
for relation in graph_dict['relations']:
    normalized_relation = relation_normalizer.normalize(relation)
    if relation != normalized_relation:
        # replace in the graph_dict
        graph_dict['relations'][graph_dict['relations'].index(relation)] = normalized_relation
        graph_dict_no_fonte['relations'][graph_dict_no_fonte['relations'].index(relation)] = normalized_relation
    
print(f"Total relations before normalization: {len(graph_dict['relations'])}")
graph_dict['relations'] = list(set(graph_dict['relations']))  # Remove duplicates
graph_dict_no_fonte['relations'] = list(set(graph_dict_no_fonte['relations']))  # Remove duplicates
print(f"Total relations after normalization: {len(graph_dict['relations'])}")

# we explore graph_dict['triples'] and normalize them
for triple in graph_dict['triples']:
    normalized_relation = relation_normalizer.normalize(triple['relazione'])
    if triple['relazione'] != normalized_relation:
        triple['relazione'] = normalized_relation

for triple in graph_dict_no_fonte['triples']:
    normalized_relation = relation_normalizer.normalize(triple['relazione'])
    if triple['relazione'] != normalized_relation:
        triple['relazione'] = normalized_relation

print(f"Total triples before normalization: {len(graph_dict['triples'])}")

# remove triples exactly the same
unique_triples = []
for triple in graph_dict['triples']:
    if triple not in unique_triples:
        unique_triples.append(triple)
graph_dict['triples'] = unique_triples

print(f"Total triples after normalization: {len(graph_dict['triples'])}")

with open('docs_kg/aggregated_knowledge_graph_normalized.json', 'w', encoding='utf-8') as f:
    json.dump(graph_dict, f, ensure_ascii=False, indent=4)
    
with open('docs_kg/aggregated_knowledge_graph_normalized_no_fonte.json', 'w', encoding='utf-8') as f:
    json.dump(graph_dict_no_fonte, f, ensure_ascii=False, indent=4)