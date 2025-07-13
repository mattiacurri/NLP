import os
import json
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from typing import List

from google.genai import Client

from prompts import HUBS_SYSTEM_PROMPT, ISOLATED_SYSTEM_PROMPT

from pydantic import BaseModel
import time

# --- CONFIGURAZIONE ---

from dotenv import load_dotenv
load_dotenv()

client = Client()
# Neo4j Database
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "skill")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "issue")

# Configurazione del Dataset
DATASET_DISTRIBUTION = {
    "isolated": 8,
    "1_hop": 8,
    "2_hop": 8,
    "hubs": 3,
    "totalmente_fuori_contesto": 6,
}

# --- FUNZIONI DI INTERAZIONE CON LLM ---

class AnswerEntry(BaseModel):
    answer: str
    analysis: str

class DatasetEntry(BaseModel):
    question: str
    answer: AnswerEntry

class HubEntry(BaseModel):
    subquestions: List[str]
    subanswers: List[AnswerEntry]

def generate_qa_from_prompt(system_instruction, prompt_text, response_schema=DatasetEntry, model="gemini-2.5-flash"):
    """
    Funzione generica per inviare un prompt a Gemini e ricevere una risposta JSON.
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text,
            config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.0,
            "system_instruction": system_instruction,
            "thinking_config": {
                "thinking_budget": 0
            }
          },
        )
        # Assumendo che l'output sia un oggetto JSON contenente una lista
        return response.text
    except Exception as e:
        raise RuntimeError(f"Errore durante la generazione della risposta: {e}")

# --- FUNZIONI DI CAMPIONAMENTO DA NEO4J ---

class Neo4jSampler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _run_query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def sample_isolated(self, limit=20):
        query = """
        MATCH (s)-[r]->(o)
        WHERE 
            // This part correctly identifies the starting nodes 's'
            count {(s)-->()} = 1 AND 
            count {(s)<--()} = 0 and count {(o)-->()} = 0
            and rand() < 0.1
            
        // Group by the starting node 's' to make it unique
        WITH s, 
            // For each 's', just take the first path found
            head(collect({r: r, o: o})) AS path_data

        // Now, expand the collected data back into columns
        RETURN
            elementId(s) AS source_id,
            s.id AS entita1,
            path_data.r.type AS relation,          // Corrected: use type()
            path_data.r.properties AS source,     // Corrected: use properties()
            path_data.o.id AS entita2,
            elementId(path_data.o) AS target_id
        LIMIT $limit
        """
        return self._run_query(query, {'limit': limit})

    def sample_one_hop(self, limit=DATASET_DISTRIBUTION["1_hop"]):
        query = """
        MATCH (s)-[r]->(o)
        WHERE 
            // This part correctly identifies the starting nodes 's'
            count {(s)-->()} = 1
            and rand() < 0.1
            
        // Group by the starting node 's' to make it unique
        WITH s, 
            // For each 's', just take the first path found
            head(collect({r: r, o: o})) AS path_data

        // Now, expand the collected data back into columns
        RETURN
            elementId(s) AS source_id,
            s.id AS entita1,
            path_data.r.type AS relation,          // Corrected: use type()
            path_data.r.properties AS source,     // Corrected: use properties()
            path_data.o.id AS entita2,
            elementId(path_data.o) AS target_id

        LIMIT $limit
        """
        return self._run_query(query, {'limit': limit})

    def sample_two_hops(self, limit=DATASET_DISTRIBUTION["2_hop"]):
        query = """
        MATCH (s)-[r]->(o)-[r1]->(e)
        WHERE 
        // This part correctly identifies the starting nodes 's'
        count {(o)-->()} = 1
        and rand() < 0.1

        // Group by the starting node 's' to make it unique
        WITH s, 
            // For each 's', just take the first path found
            head(collect({r: r, o: o, r1: r1, e: e})) AS path_data

        // Now, expand the collected data back into columns
        RETURN
        elementId(s) AS source_id,
        s.id AS entita1,
        path_data.r.type AS relation,          // Corrected: use type()
        path_data.r.properties AS source,     // Corrected: use properties()
        path_data.o.id AS entita2,
        elementId(path_data.o) AS target_id,
        path_data.r1.type as relation1,
        path_data.r1.properties as source1,
        path_data.e.id as entita3,
        elementId(path_data.e) as target_id_1
        LIMIT $limit
        """
        return self._run_query(query, {'limit': limit})
    
    def sample_hubs(self, limit=DATASET_DISTRIBUTION["hubs"]):
        # Query per trovare nodi "hub" e raccogliere il CONTESTO dei nodi collegati.
        # Assumiamo che il contesto sia nella proprietà 'name'.
        query = """
        MATCH (hub)
        WITH hub, count {(hub)--()} as degree
        ORDER BY degree DESC
//        where degree > 2 and degree <= 5
        limit 1000

        // Campiona casualmente dai top hub trovati
        WITH hub
        limit $limit

        // Raccogli il contesto e gli ID dei nodi collegati
        MATCH (hub)-[r]-(o)
        WHERE o.id IS NOT NULL
        RETURN 
            elementId(hub) as hub_id,
            hub.id as hub_name, 
            collect({
                relation: r.type,
                source: r.properties, 
                connected_node_name: o.id,
                connected_node_id: elementId(o)
            }) as connections;
        """
        return self._run_query(query, {'limit': limit})
        
#     def sample_for_negative(self, limit=10):
#         # Query per trovare una coppia di nodi sorgente (s1, s2) che puntano a oggetti diversi (o1, o2)
#         # ma che potrebbero essere confusi tra loro.
#         query = """
#         // Fase 1: Trova un nodo hub importante in modo casuale
# MATCH (hub)
# WITH hub, count {(hub)--()} as degree
# ORDER BY degree DESC
# LIMIT 100 // Prendi i primi 100 nodi più connessi come "candidati importanti"


# // Salta un numero casuale di candidati per non prendere sempre il primo
# WITH hub SKIP toInteger(rand() * 20) 
# LIMIT 1 // Seleziona un singolo hub per questa iterazione

# // Fase 2: Raccogli tutte le informazioni VERE associate a questo hub
# MATCH (hub)-[r]->(entity)
# // Assumiamo, come da nostra ultima conversazione, che il testo sia nell'unica proprietà della relazione
# //WHERE size(keys(r)) = 1 
# WITH hub, r, keys(r) as text_key, entity
# LIMIT $limit

# // Fase 3: Restituisci i dati necessari per il prompt
# RETURN
#   hub.label as hub_label,    // Le proprietà del nostro soggetto (es. {"id": "...", "label": "..."})
#   r.type as relation,
#   entity.label as entita,
#   r.properties as source    // La lista di tutti i testi veri che abbiamo su questo hub
#         """
#         return self._run_query(query, {'limit': limit})

# --- FUNZIONI DI GENERAZIONE DEL DATASET ---

def create_dataset():
    sampler = Neo4jSampler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    dataset = []
    
    # 1. Domande Facili (con testo del nodo)
    samples = sampler.sample_isolated(limit=DATASET_DISTRIBUTION["isolated"])
    for i, sample in enumerate(tqdm(samples, desc="Facili")):
        prompt = f"""
        `{sample['entita1']} {sample['relation']} {sample['entita2']} - {sample['source']}`
        """
        if (i + 1) % 10 == 0:
            time.sleep(50)
        qa = generate_qa_from_prompt(ISOLATED_SYSTEM_PROMPT, prompt)
        j = json.loads(qa)
        dataset.append({
            "question": j['question'],
            "answer": j['answer']['answer'],
            "analysis": j['answer']['analysis'],
            "difficulty": "isolated",
            "triples": [f"{sample['entita1']} {sample['relation']} {sample['entita2']} - {sample['source']}"],
        })
    time.sleep(50)
    #2. Domande Medie (con testo dei nodi)
    samples = sampler.sample_one_hop(limit=DATASET_DISTRIBUTION["1_hop"])
    for i, sample in enumerate(tqdm(samples, desc="Medie")):
        prompt = f"""
        `{sample['entita1']} {sample['relation']} {sample['entita2']} - {sample['source']}`
        """
        
        if (i + 1) % 10 == 0:
            time.sleep(50)
        qa = generate_qa_from_prompt(ISOLATED_SYSTEM_PROMPT, prompt)
        j = json.loads(qa)
        dataset.append({
            "question": j["question"],
            "answer": j["answer"]["answer"],
            "analysis": j["answer"]["analysis"],
            "difficulty": "1_hop", 
            "triples": [
                f"{sample['entita1']} {sample['relation']} {sample['entita2']} - {sample['source']}"
            ]
        })
    time.sleep(50)
#    3. Domande Difficili (Aggregazione)
    samples = sampler.sample_hubs()
    for i, sample in enumerate(tqdm(samples, desc="Difficili")):
        contexts_for_prompt = [f"{sample['hub_name']} {c['relation']} {c['connected_node_name']} - {c['source']}" for c in sample['connections']]
        contexts_str = "\n- ".join(contexts_for_prompt)
        
        prompt = f"""
        Concetto: `{sample['hub_name']}`
        
        Contesto:
        
        `{contexts_str}`
        """
        if (i + 1) % 10 == 0:
            time.sleep(50)
        qas = generate_qa_from_prompt(HUBS_SYSTEM_PROMPT, prompt, response_schema=HubEntry)
        
        """
        subquestions : ["", ""]
        subanswers : {"answer": "", "analysis": ""}, {"answer": "", "analysis": ""}]
        subtriples: [["", ""], ["", ""]]
        general_question: ""
        general_answer: {"answer": "", "analysis": ""}
        """

        j = json.loads(qas)
        # Raccogli tutti gli ID dei nodi collegati come ground truth per il retriever
        for i, subquestion in enumerate(j["subquestions"]):
            dataset.append({
                "question": subquestion,
                "answer": j["subanswers"][i]["answer"],
                "analysis": j["subanswers"][i]["analysis"],
                "difficulty": "hubs",
                "triples": contexts_for_prompt,
            })
            
    time.sleep(50)

    # 4. Domande "2_hop"
    samples = sampler.sample_two_hops()
    for i, sample in enumerate(tqdm(samples, desc="2_hop")):
        contexts_for_prompt = [
            f"{sample['entita1']} {sample['relation']} {sample['entita2']} - {sample['source']}",
            f"{sample['entita2']} {sample['relation1']} {sample['entita3']} - {sample['source1']}"
        ]
        contexts_str = "\n- ".join(contexts_for_prompt)

        prompt = f"""
        `{contexts_str}`
        """
        
        if (i + 1) % 10 == 0:
            time.sleep(50)
        qas = generate_qa_from_prompt(ISOLATED_SYSTEM_PROMPT, prompt)
        
        j = json.loads(qas)
        # La risposta corretta è la negazione, basata sul contesto vero
        dataset.append({
            "question": j['question'],
            "answer": j["answer"]["answer"],
            "analysis": j["answer"]["analysis"],
            "difficulty": "2_hop",
            "triples": contexts_for_prompt,
        })

    # 5. Domande "Totalmente Fuori Contesto"
    print("\nGenerando domande TOTALMENTE FUORI CONTESTO...")
    far_ood_questions = [
        "Qual è la ricetta per la pizza margherita?",
        "Spiegami le regole del fuorigioco nel calcio.",
        "Chi ha scritto 'I Promessi Sposi'?",
        "Come si diventa avvocati?",
        "Come funziona un motore a scoppio per un motore da 200 cavalli?",
        "Quali sono le cause dell'effetto serra?",
    ]
    for q in far_ood_questions:
        dataset.append({
            "question": q,
            "answer": "Non ho informazioni su questo argomento.",
            "analysis": "Questa domanda è completamente fuori contesto rispetto al dominio legale e amministrativo.",
            "difficulty": "totalmente_fuori_contesto",
            "triples": [],
        })

    sampler.close()
    return dataset


# --- ESECUZIONE ---
if __name__ == "__main__":
    final_dataset = create_dataset()
    
    # Salva il dataset in un file JSON
    with open("EmPULIA-QA.json", "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    # (Opzionale) Salva anche come CSV per una isolated visualizzazione
    df = pd.DataFrame(final_dataset)
    df.to_csv("EmPULIA-QA.csv", index=False, encoding='utf-8-sig')

    print(f"\nDataset di valutazione creato con successo! Contiene {len(final_dataset)} domande.")