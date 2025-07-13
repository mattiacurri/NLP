import os
import json
import faiss
from sentence_transformers import SentenceTransformer


class Search:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', index_file=None, triples_file=None, entities_file=None, entities_index=None):
        """
        Initialize the search class with the model and FAISS index.
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
        Load a knowledge graph from a JSON file and convert it to a list of triples.
        Expected format is a dictionary/adjacency list: {"head1": [{"relation": r1, "target": t1}], ...}
        """
        if not os.path.exists(graph_path):
            print(f"Error: File {graph_path} not found.")
            return []
            
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        triples = []
        for triple in graph_data['triples']:
            triples.append((triple['entita1'], triple['relazione'], triple['entita2'], triple['fonte']))

        print(f"Extracted {len(triples)} triples from the graph.")
        if save:
            # Save triples to a file for retrieval
            with open("triples.json", 'w', encoding='utf-8') as f:
                json.dump(triples, f, ensure_ascii=False, indent=4)
            print("Triples saved to triples.json.")
        self.triples = triples
        return triples
    
    def extract_entities_from_graph(self, graph_path):
        """
        Load a knowledge graph from a JSON file and convert it to a list of entities.
        Expected format is a dictionary/adjacency list: {"head1": [{"relation": r1, "target": t1}], ...}
        """
        if not os.path.exists(graph_path):
            print(f"Error: File {graph_path} not found.")
            return []
            
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        entities = []
        for entity in graph_data['entities']:
            entities.append(entity)

        print(f"Extracted {len(entities)} entities from the graph.")
        
        # Save entities to a file for retrieval
        #with open("entities.json", 'w', encoding='utf-8') as f:
        #    json.dump(entities, f, ensure_ascii=False, indent=4)
        #print("Entities saved to entities.json.")
        self.entities = entities
        return entities
    
    def vectorize_and_index_triples(self, source=True):
        """
        Vectorize graph triples (converted to sentences) and index them with FAISS.
        """
        if not self.triples:
            print("No triples to vectorize.")
            return None, None, None

        # 1. Convert triples to sentences
        if source:
            sentences = [f"{s} {r} {o}: {f}" for s, r, o, f in self.triples]
        else:
            print("Source not embedded")
            sentences = [f"{s} {r} {o}" for s, r, o, f in self.triples]
        print("Vectorizing triples in progress (this may take time)...")
        # Qwen
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        # Normalize embeddings for cosine similarity search
        faiss.normalize_L2(embeddings)

        # 3. Create and populate FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Suitable for cosine similarity on normalized vectors
        self.index.add(embeddings)

        print(f"Created FAISS index with {self.index.ntotal} vectors.")

    def vectorize_and_index_entities(self):
        """
        Vectorize graph entities and index them with FAISS.
        """
        if not self.entities:
            print("No entities to vectorize.")
            return None, None, None

        # 1. Convert entities to vectors
        print("Vectorizing entities in progress (this may take time)...")
        # Qwen
        embeddings = self.model.encode(self.entities, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        # Normalize embeddings for cosine similarity search
        faiss.normalize_L2(embeddings)

        # 3. Create and populate FAISS index
        embedding_dim = embeddings.shape[1]
        self.entities_index = faiss.IndexFlatIP(embedding_dim)  # Suitable for cosine similarity on normalized vectors
        self.entities_index.add(embeddings)

        print(f"Created FAISS index with {self.entities_index.ntotal} vectors.")

    def save_embeddings_to_file(self, file_path):
        """
        Save vector embeddings to file in binary format.
        """
        faiss.write_index(self.index, file_path)
        print(f"Embeddings saved to {file_path}.")

    def _get_embeddings_from_file(self, file_path):
        print(file_path)
        """
        Load vector embeddings from a file in binary format.
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return None

        self.index = faiss.read_index(file_path)
        print(f"Embeddings loaded from {file_path}. Number of vectors: {self.index.ntotal}.")
        return self.index

    def search_semantic_triples(self, query, cosine_threshold=0.6, top_k=546):
        """
        Search for triples most similar to a text query using the FAISS index.
        """
        # Vettorizza la query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Cerca nell'indice
        distances, indices = self.index.search(query_embedding, top_k)

        # Restituisci i risultati
        results = []
        print(f"\nSearch results for '{query}':")
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                head, relation, target, source = self.triples[idx]
                similarity = distances[0][i]
                if similarity > cosine_threshold:
                    results.append(((head, relation, target, source), similarity))
        print(len(results), "results found.")
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
    
    # Graph file path
    graph_file_path = 'aggregated_knowledge_graph_normalized.json'
    
    # 1. Extract triples from graph
    knowledge_triples = s.extract_triples_from_graph(graph_file_path)
    
    # 2. Vectorize and index triples
    s.vectorize_and_index_triples()
    
    # 3. Save embeddings to file
    embeddings_file_path = 'EMB_withoutsource.index'
    s.save_embeddings_to_file(embeddings_file_path)

    # Test semantic search
    query = "Qual è l'obbligo degli Stati membri secondo la Direttiva 2014/23/UE?"
    r = s.search_semantic_triples(query)
    print(r)