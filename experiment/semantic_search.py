import os
import json
import faiss
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

class Search:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', index_file=None, triples_file=None, entities_file=None, entities_index=None):
        """
        Inizializza la classe di ricerca con il modello e l'indice FAISS.
        """
        self.model_name = model_name
        self.index_file = index_file
        self.index = self._get_embeddings_from_file(index_file) if index_file else None
        self.model = SentenceTransformer(model_name)
        self.triples_file = triples_file
        self.triples = self.extract_triples_from_graph(triples_file, save=False) if triples_file else []
        self.entities = self.extract_entities_from_graph(entities_index) if entities_index else []
        self.entities_index = self._get_embeddings_from_file(entities_file) if entities_file else None
        self.entities_file = entities_index
        
    def extract_triples_from_graph(self, graph_path, save=True):
        """
        Carica un knowledge graph da un file JSON e lo converte in una lista di triple.
        Il formato atteso è un dizionario/adjacency list: {"head1": [{"relation": r1, "target": t1}], ...}
        """
        if not os.path.exists(graph_path):
            print(f"Errore: Il file {graph_path} non è stato trovato.")
            return []
            
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        triples = []
        for triple in graph_data['triples']:
            triples.append((triple['entita1'], triple['relazione'], triple['entita2'], triple['fonte']))

        print(f"Estratte {len(triples)} triple dal grafo.")
        if save:
            # Salva le triple in un file per il retrieval
            with open("triples.json", 'w', encoding='utf-8') as f:
                json.dump(triples, f, ensure_ascii=False, indent=4)
            print("Triple salvate in triples.json.")
        self.triples = triples
        return triples
    
    def extract_entities_from_graph(self, graph_path):
        """
        Carica un knowledge graph da un file JSON e lo converte in una lista di triple.
        Il formato atteso è un dizionario/adjacency list: {"head1": [{"relation": r1, "target": t1}], ...}
        """
        if not os.path.exists(graph_path):
            print(f"Errore: Il file {graph_path} non è stato trovato.")
            return []
            
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        entities = []
        for entity in graph_data['entities']:
            entities.append(entity)

        print(f"Estratte {len(entities)} entità dal grafo.")
        
        # Salva le entità in un file per il retrieval
        #with open("entities.json", 'w', encoding='utf-8') as f:
        #    json.dump(entities, f, ensure_ascii=False, indent=4)
        #print("Entità salvate in entities.json.")
        self.entities = entities
        return entities
    
    def vectorize_and_index_triples(self, source=True):
        """
        Vettorizza le triple del grafo (convertite in frasi) e le indicizza con FAISS.
        """
        if not self.triples:
            print("Nessuna tripla da vettorizzare.")
            return None, None, None

        # 1. Converti le triple in frasi
        if source:
          sentences = [f"{s} {r} {o}: {f}" for s, r, o, f in self.triples]
        else:
          print("Source not embedded")
          sentences = [f"{s} {r} {o}" for s, r, o, f in self.triples]
        print("Vettorizzazione delle triple in corso (potrebbe richiedere tempo)...")
        # Qwen
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        # Normalizza gli embedding per la ricerca di similarità coseno
        faiss.normalize_L2(embeddings)

        # 3. Crea e popola l'indice FAISS
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Adatto per similarità coseno su vettori normalizzati
        self.index.add(embeddings)

        print(f"Creato indice FAISS con {self.index.ntotal} vettori.")

    def vectorize_and_index_entities(self):
        """
        Vettorizza le triple del grafo (convertite in frasi) e le indicizza con FAISS.
        """
        if not self.entities:
            print("Nessuna tripla da vettorizzare.")
            return None, None, None

        # 1. Converti le triple in frasi
        print("Vettorizzazione delle triple in corso (potrebbe richiedere tempo)...")
        # Qwen
        embeddings = self.model.encode(self.entities, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        # Normalizza gli embedding per la ricerca di similarità coseno
        faiss.normalize_L2(embeddings)

        # 3. Crea e popola l'indice FAISS
        embedding_dim = embeddings.shape[1]
        self.entities_index = faiss.IndexFlatIP(embedding_dim)  # Adatto per similarità coseno su vettori normalizzati
        self.entities_index.add(embeddings)

        print(f"Creato indice FAISS con {self.entities_index.ntotal} vettori.")

    def save_embeddings_to_file(self, file_path):
        """
        Salva gli embeddings vettoriali su file in formato binario.
        """
        faiss.write_index(self.index, file_path)
        print(f"Embeddings salvati in {file_path}.")

    def _get_embeddings_from_file(self, file_path):
        print(file_path)
        """
        Carica gli embeddings vettoriali da un file in formato binario.
        """
        if not os.path.exists(file_path):
            print(f"Errore: Il file {file_path} non è stato trovato.")
            return None

        self.index = faiss.read_index(file_path)
        print(f"Embeddings caricati da {file_path}. Numero di vettori: {self.index.ntotal}.")
        return self.index

    def search_semantic_triples(self, query, cosine_threshold=0.6, top_k=546):
        """
        Cerca le triple più simili a una query testuale utilizzando l'indice FAISS.
        """
        # Vettorizza la query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Cerca nell'indice
        distances, indices = self.index.search(query_embedding, top_k)

        # Restituisci i risultati
        results = []
        print(f"\nRisultati della ricerca per '{query}':")
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                head, relation, target, source = self.triples[idx]
                similarity = distances[0][i]
                if similarity > cosine_threshold:
                    results.append(((head, relation, target, source), similarity))
        print(len(results), "risultati trovati.")
        if len(results) > 50:
            # sort by similarity and take the top 50
            results = sorted(results, key=lambda x: x[1], reverse=True)[:50]

        # Reranker
        # from qwen3_reranker_transformers import Qwen3Reranker
        # reranker = Qwen3Reranker(model_name_or_path='Qwen/Qwen3-Reranker-0.6B', instruction="Retrieval document that can answer user's query", max_length=2048)
        # rr = [(query, results[i][0]) for i in range(len(results))]
        # reranked_results = []
        # for r in tqdm(rr):
        #     reranked_results.append(reranker.compute_scores([r], instruction="Given the user query, retrieval the relevant passages"))

        # print(reranked_results[:10])

        return results

if __name__ == '__main__':
    s = Search()
    
    # Percorso del file del grafo
    graph_file_path = 'aggregated_knowledge_graph_normalized.json'
    
    # 1. Estrai le triple dal grafo
    knowledge_triples = s.extract_triples_from_graph(graph_file_path)
    
    # 2. Vettorizza e indicizza le triple
    s.vectorize_and_index_triples()
    
    # 3. Salva gli embeddings su file
    embeddings_file_path = 'EMB_withoutsource.index'
    s.save_embeddings_to_file(embeddings_file_path)

    # Test della ricerca semantica
    query = "Qual è l'obbligo degli Stati membri secondo la Direttiva 2014/23/UE?"
    r = s.search_semantic_triples(query)
    print(r)