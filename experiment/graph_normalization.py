"""
Questa pipeline è progettata per normalizzare le entità di un Knowledge Graph in un sistema RAG che viene aggiornato online. Utilizza un approccio a due fasi: una fase di inizializzazione offline per costruire la base di conoscenza iniziale e una fase di aggiornamento online per gestire in modo efficiente i nuovi dati.
Descrizione della Pipeline di Clustering Adattivo

L'obiettivo è creare e mantenere un sistema che raggruppi nodi semanticamente simili (es. "auto", "automobile", "vettura") sotto un unico rappresentante canonico (es. "auto").
Fase 1: Inizializzazione (Offline)

Questa fase viene eseguita una sola volta su un corpus di dati iniziale per creare le fondamenta del nostro sistema di normalizzazione.

    Estrazione del Vocabolario: Dal corpus iniziale, estrai tutti i nodi unici del Knowledge Graph per formare un vocabolario.

    Generazione degli Embeddings: Per ogni parola del vocabolario, calcola il suo vettore di embedding usando un modello pre-addestrato (es. FastText o Word2Vec).

    Clustering Iniziale (HDBSCAN/DBSCAN): Esegui un algoritmo di clustering basato sulla densità (HDBSCAN è spesso preferibile perché non richiede eps) su tutti gli embeddings. Questo raggrupperà le parole simili e identificherà le parole isolate (rumore).

    Creazione dello Stato Iniziale: Sulla base dei risultati del clustering, costruisci e salva gli "artefatti" del sistema:

        centroids: Un dizionario che mappa l'ID di ogni cluster al suo vettore centroide (la media degli embeddings dei suoi membri).

        representatives: Un dizionario che mappa l'ID di ogni cluster alla sua parola rappresentativa (es. la più vicina al centroide).

        normalization_map: Una mappa completa che associa ogni parola del vocabolario iniziale al suo rappresentante canonico. Le parole "rumore" mappano a se stesse.

        noise_embeddings e noise_words: Due liste sincronizzate contenenti gli embeddings e le parole classificate come rumore. Queste saranno la base per scoprire nuovi cluster online.

Fase 2: Aggiornamento (Online)

Questa è la funzione che viene chiamata ogni volta che una nuova parola (nodo) viene estratta da un nuovo documento.

    Input: Una nuova parola new_word.

    Controllo Cache: Verifica se new_word è già presente in normalization_map. Se sì, restituisce immediatamente la sua forma canonica.

    Generazione Embedding: Calcola l'embedding per new_word. Se la parola non è nel modello, viene considerata rumore e mappata a se stessa.

    Matching con Cluster Esistenti: Calcola la similarità (coseno) tra l'embedding di new_word e tutti i centroidi dei cluster esistenti. Se la similarità con il cluster più vicino supera una threshold_cluster, la parola viene assegnata a quel cluster, la normalization_map viene aggiornata e il processo termina.

    Matching con Parole Isolate (Rumore): Se la parola non corrisponde a nessun cluster, si cerca un "partner" tra le parole precedentemente etichettate come rumore. Si calcola la similarità con tutte le parole in noise_words.

    Creazione di un Nuovo Cluster: Se la similarità con la parola rumore più vicina supera una threshold_new_cluster (tipicamente più restrittiva), viene creato un nuovo cluster:

        Si sceglie un rappresentante per la nuova coppia.

        Si calcola il nuovo centroide.

        I nuovi dati vengono aggiunti a centroids e representatives.

        La normalization_map viene aggiornata per entrambe le parole.

        La parola "partner" viene rimossa dalla lista del rumore.

    Classificazione come Nuovo Rumore: Se la parola non trova corrispondenze né con i cluster né con altre parole isolate, viene aggiunta alle liste noise_embeddings e noise_words per essere considerata in futuri confronti.
"""


from json import load
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

from sklearn.cluster import DBSCAN

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

class Entities(BaseModel):
    entities: list[str] = Field(
        ...,
        description="Lista di entità uniche e normalizzate."
    )

class Normalizer:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', eps=0.05):
        """
        Inizializza il normalizzatore.
        - model_name: Nome del modello da Hugging Face (es. "Qwen/Qwen3-Embedding-4B").
        - eps: Soglia per DBSCAN.
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
        # Calcola la similarità cosenica tra tutti gli embedding in modo efficiente
        similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
        
        # La distanza è 1 - similarità
        distance_matrix = 1 - similarity_matrix

        # Assicura che non ci siano valori negativi dovuti a imprecisioni numeriche
        np.clip(distance_matrix, 0, None, out=distance_matrix)

        return distance_matrix 

    def initialize(self, initial_vocabulary):
        """
        FASE 1: Esegue il clustering iniziale su un vocabolario.
        - initial_vocabulary: una lista di parole/nodi unici.
        """
        print(f"Inizializzazione con un vocabolario di {len(initial_vocabulary)} parole.")
        
        valid_words = [word for word in initial_vocabulary if self._get_embedding(word) is not None]
        embeddings = np.array([self._get_embedding(word) for word in valid_words])

        dbscan = DBSCAN(metric='precomputed', eps=self.eps, min_samples=2, n_jobs=-1).fit(self._compute_distance_matrix(embeddings))

        # Popola lo stato iniziale
        unique_labels = set(dbscan.labels_)
        for label in unique_labels:
            indices = np.where(dbscan.labels_ == label)[0]
            cluster_words = [valid_words[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            print(label, cluster_words, cluster_embeddings.shape)
            if label == -1: # Rumore
                self.noise_words.extend(cluster_words)
                self.noise_embeddings = cluster_embeddings
                for word in cluster_words:
                    self.normalization_map[word] = word
            else: # Cluster validi
                centroid = np.mean(cluster_embeddings, axis=0)
                # Scegli il rappresentante (parola più vicina al centroide)
                similarities = util.cos_sim(cluster_embeddings, centroid.reshape(1, -1))
                representative = cluster_words[np.argmax(similarities)]

                self.centroids[label] = centroid
                self.representatives[label] = representative
                self.next_cluster_id = max(self.next_cluster_id, label + 1)
                print(f"Cluster {label}: {representative} con {len(cluster_words)} parole.")
                for word in cluster_words:
                    self.normalization_map[word] = representative
        
        print("Inizializzazione completata.")
        print(f"Trovati {len(self.centroids)} cluster e {len(self.noise_words)} parole isolate.")

    def normalize(self, new_word):
        """
        FASE 2: Normalizza una nuova parola usando la logica online.
        """
        # 1. Controllo cache
        if new_word in self.normalization_map:
            return self.normalization_map[new_word]

        # 2. Generazione embedding
        new_embedding = self._get_embedding(new_word)
        if new_embedding is None:
            self.normalization_map[new_word] = new_word
            return new_word

        # 3. Matching con cluster esistenti
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

        # 4. Matching con parole isolate (rumore)
        if len(self.noise_words) > 0:
            # Calcola similarità con ogni parola rumore
            for i, noise_word in enumerate(self.noise_words):
                sim = util.cos_sim(new_embedding.reshape(1, -1), self.noise_embeddings[i].reshape(1, -1)).item()
                if sim > 1 - self.eps:  # Soglia per considerare simile
                    # Se trova una parola rumore simile, aggiorna la normalization_map
                    self.normalization_map[new_word] = noise_word
                    # Rimuove la parola rumore dalla lista
                    self.noise_words.pop(i)
                    self.noise_embeddings = np.delete(self.noise_embeddings, i, axis=0)
                    # print(f"Parola '{new_word}' normalizzata a '{noise_word}'.")
                    return noise_word

        # 5. Classificazione come nuovo rumore
        self.noise_words.append(new_word)
        if len(self.noise_embeddings) == 0:
            self.noise_embeddings = new_embedding.reshape(1, -1)
        else:
            self.noise_embeddings = np.vstack([self.noise_embeddings, new_embedding])
        self.normalization_map[new_word] = new_word
        
        return new_word
    
    def save_state(self, filepath):
        """Salva lo stato corrente del normalizzatore su file."""
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
        print(f"Stato salvato in {filepath}")

    def load_state(self, filepath):
        """Carica uno stato precedentemente salvato."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.centroids = state["centroids"]
        self.representatives = state["representatives"]
        self.normalization_map = state["normalization_map"]
        self.noise_embeddings = state["noise_embeddings"]
        self.noise_words = state["noise_words"]
        self.next_cluster_id = state["next_cluster_id"]
        self.eps = state["eps"]
        print(f"Stato caricato da {filepath}")

    def view_state(self):
        """Visualizza lo stato corrente del normalizzatore."""
        print(f"Centroids: {self.centroids}")
        print(f"Representatives: {self.representatives}")
        print(f"Normalization Map: {self.normalization_map}")
        print(f"Noise Words: {self.noise_words}")
        print(f"Next Cluster ID: {self.next_cluster_id}")
        print(f"Epsilon: {self.eps}")
    
# --- ESEMPIO DI UTILIZZO ---

# Imposta il percorso del tuo modello
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

import json

with open('docs_kg/aggregated_knowledge_graph_normalized.json', 'w', encoding='utf-8') as f:
    json.dump(graph_dict, f, ensure_ascii=False, indent=4)
    
with open('docs_kg/aggregated_knowledge_graph_normalized_no_fonte.json', 'w', encoding='utf-8') as f:
    json.dump(graph_dict_no_fonte, f, ensure_ascii=False, indent=4)