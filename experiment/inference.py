import ollama 
from google import genai

from pydantic import BaseModel, Field

from semantic_search import Search

from dotenv import load_dotenv
import os

import json

from prompts import RAG_PROMPT, rag_prompt, ENTITIES_RELATIONS_EXTRACTION_SYSTEM_PROMPT, COMBINE_ANSWERS_PROMPT, DECOMPOSE_PROMPT, ENTITIES_EXTRACTION_SYSTEM_PROMPT

load_dotenv()

NEO4J_URL = os.getenv('NEO4J_URL', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'skill')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'issue')


class MultiQuery(BaseModel):
    """A model for multi-query prompts."""
    queries: list[str] = Field(
        ...,
        description="A list of queries that are rephrased versions of the original prompt."
    )

class Response(BaseModel):
    answer: str = Field(
        ...,
        description="The answer to the question based on the context provided."
    )
    analysis: str = Field(
        ...,
        description="A detailed analysis of the answer, explaining the reasoning and connections made."
    )
    sources: list[str] = Field(
        ...,
        description="A list of sources cited in the answer, providing references to the entities and relationships in the graph."
    )

class Entities(BaseModel):
    entities: list[str] = Field(
        ...,
        description="A list of entity names extracted from the text."
    )

class EntitiesRelations(BaseModel):
    entities: list[str] = Field(
        ...,
        description="A list of entity names extracted from the text."
    )
    relations: list[str] = Field(
        ...,
        description="A list of relation names extracted from the text."
    )
    triples: list[list[str]] = Field(
        ...,
        description="A list of triples in the form [[subject, predicate, object], ...] extracted from the text."
    )
    
class OllamaInference():
    def __init__(self, model_name: str = "qwen3:4b", index= None, triples=None, entities_file=None, entities_index=None):
        self.model_name = model_name
        self.search = Search(index_file=index, triples_file=triples, entities_file=entities_file, entities_index=entities_index)

    def GraphRAG(self, prompt):
        # connect to the Neo4j database
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # take the prompt and extract entities and relations
        entities_and_relations = self.extract_entities(prompt)
        entities = entities_and_relations['entities']
        graph_entities = []
        for entity in entities:
        # semantic search for the entities
          s = self.search.search_semantic_triples(entity, cosine_threshold=0.85)
          if len(s) > 0:
            # take the first element of the list
            graph_entities.append(s[0])
        
        # take only entities from the graph_entities
        graph_entities = [entity[0] for entity in graph_entities]
        
        # Trova percorsi o random walk tra le entità
        query = """
        MATCH (n)
        WHERE n.id IN $graph_entities
        WITH COLLECT(n) AS sourceNodes
        UNWIND sourceNodes AS src
        CALL gds.randomWalk.stream(
        'myGraph',
        {
            sourceNodes: [src],
            walksPerNode: 1000,
            randomSeed: 42,
            concurrency: 4
        }
        )
        YIELD path
        WITH src.id AS startId, nodes(path) AS nodeList
        WHERE SIZE(nodeList) = SIZE(apoc.coll.toSet(nodeList))  // no cycles
        WITH startId, nodeList AS longestPath, RANGE(0, SIZE(nodeList)-2) AS idxs
        UNWIND idxs AS i
        WITH startId, longestPath, i, longestPath[i] AS fromNode, longestPath[i+1] AS toNode
        MATCH (fromNode)-[r]->(toNode)
        WITH startId, longestPath, COLLECT({i: i, rel: r}) AS relInfos
        WITH startId,
            [i IN RANGE(0, SIZE(longestPath)-1) |
            CASE
                WHEN i < SIZE(longestPath) - 1 THEN
                [longestPath[i].id, head([x IN relInfos WHERE x.i = i]).rel.type]
                ELSE
                [longestPath[i].id]
            END
            ] AS chunks
        RETURN startId, REDUCE(acc = [], chunk IN chunks | acc + chunk) AS walk
        """
        with driver.session() as session:
            result = session.run(query, graph_entities=graph_entities)
            paths = [record["walk"] for record in result]
        context = []
        for path in paths:
            # Format the path as a string
            # Format as: entity1 -[relation1]-> entity2 -[relation2]-> entity3 ...
            formatted_path = ""
            for i in range(len(path)):
                if i % 2 == 0:
                    if i == 0:
                        formatted_path += f"{path[i]}"
                    else:
                        formatted_path += f" -> {path[i]}"
                else:
                    formatted_path += f" -[{path[i]}]-"
            context.append(formatted_path)
            
        context = "\n\n".join([f"- {c}" for c in context])
        
        response = self.generate_content(rag_prompt.format(context=context, prompt=prompt))
        return response, context
    
    def format_context(self, context):
        # Format the context as bullet points
        # print(context)
        return "\n".join([f"- {triple[0][0]} {triple[0][1]} {triple[0][2]} (Fonte: {triple[0][3]})" for triple in context])
    
    def RAG(self, prompt, strategy='default', return_context=False):
        if strategy == "graphrag":
            return self.GraphRAG(prompt)
        if strategy == 'default-extraction':
            entities_and_relations = self.extract_entities_and_relations(prompt)
            entities_and_relations = json.loads(entities_and_relations)

            query_triples = entities_and_relations['triples']
            
            context = []
            for triple in query_triples:
                context.append(self.search.search_semantic_triples(triple))
                
            # Flatten the context list
            context = [item for sublist in context for item in sublist]

            context = self.format_context(context)
            response = self.generate_content(rag_prompt.format(context=context, prompt=prompt))
            if return_context:
                return response, context
            else:
                return response
        elif strategy == 'default':
            context = self.search.search_semantic_triples(prompt)

            context = self.format_context(context)
            response = self.generate_content(rag_prompt.format(context=context, prompt=prompt))
            if return_context:
                return response, context
            else:
                return response
        elif 'multiquery' in strategy:
            queries = self.generate_content_multi_query(prompt)
            contexts = []
            for query in queries['queries']:
                context = []
                if 'extraction' in strategy:
                    entities_and_relations = self.extract_entities_and_relations(prompt)
                    entities_and_relations = json.loads(entities_and_relations)

                    query_triples = entities_and_relations['triples']
                    
                    for triple in query_triples:
                        # Format triple
                        triple = f"{triple[0]} {triple[1]} {triple[2]}"
                    
                        retrieved_list = self.search.search_semantic_triples(triple)
                        # append every element of the retrieved list
                        for triple, score in retrieved_list:
                            # Check if the triple already exists in context
                            found = False
                            for i, (existing_triple, existing_score) in enumerate(context):
                                if existing_triple == triple:
                                    # Replace the score with the highest one
                                    context[i] = (existing_triple, max(existing_score, score))
                                    found = True
                                    break
                            if not found:
                                context.append((triple, score))
                    # Flatten the context list
                    context = sorted(context, key=lambda x: x[1], reverse=True)
                    
                    contexts.append(context)
                else:
                    retrieved_list = self.search.search_semantic_triples(query)
                    # append every element of the retrieved list
                    for triple, score in retrieved_list:
                        # Check if the triple already exists in context
                        found = False
                        for i, (existing_triple, existing_score) in enumerate(context):
                            if existing_triple == triple:
                                # Replace the score with the highest one
                                context[i] = (existing_triple, max(existing_score, score))
                                found = True
                                break
                        if not found:
                            context.append((triple, score))
                    # Flatten the context list
                    context = sorted(context, key=lambda x: x[1], reverse=True)
                    
                    contexts.append(context)
            # Initialize top5 as an empty list to accumulate top entries from all contexts
            top5 = []
            
            # Process each context to extract top entries
            for context in contexts:
                if len(context) > 5:
                    # Get the top 5 triples for this context and add them to top5
                    for item in context[:5]:
                        top5.append(item)  
                else:
                    # If there are less than 5 triples, add all of them
                    for item in context:
                        top5.append(item)  
            
            # flatten contexts to get all triples for further processing
            all_context_triples = [item for sublist in contexts for item in sublist]
            
            # Filter out triples that are already in top5
            filtered_triples = [triple for triple in all_context_triples if triple not in top5]
            
            # Get the most common triples from the filtered list
            from collections import Counter
            most_common = Counter(filtered_triples).most_common(100 - len(top5))
            most_common_triples = [item[0] for item in most_common]
            
            # final context: most_common + top5
            final_context = most_common_triples + top5
            
            # Remove duplicates while preserving order based on the triple
            seen_triples = set()
            unique_context = []
            for triple, score in final_context:
                if tuple(triple) not in seen_triples:
                    unique_context.append((triple, score))
                    seen_triples.add(tuple(triple))
            final_context = unique_context
            
            
            final_context = self.format_context(final_context)
            
            response = self.generate_content(rag_prompt.format(context=final_context, prompt=query))
            
            if return_context:
                return response, final_context
            else:
                return response
        elif 'decomposition' in strategy:
            response = self.generate_content_query_decomposition(prompt)
            # now answer separately each query in the list
            responses = []
            for query in set(response['queries']):
                if 'extraction' in strategy:
                    responses.append(self.RAG(query, strategy='default-extraction'))
                else:
                    responses.append(self.RAG(query, strategy='default'))

            # then, combine the results into a single response
            response = self.generate_content(prompt=COMBINE_ANSWERS_PROMPT.format(
                question=prompt,
                questions_and_answers="\n".join([f"Domanda: {q}\n\nRisposta: \n\n - {a['answer']}\n\nAnalisi:\n\n - {a['analysis']}\n\nFonti:\n\n - {a['sources']}" for q, a in zip(response['queries'], responses)])
            ))
            return response
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            

    def extract_entities(self, prompt: str):
        client = genai.Client(
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
            "response_mime_type": "application/json",
            "response_schema": Entities,
            "temperature": 0.0,
            "system_instruction": ENTITIES_EXTRACTION_SYSTEM_PROMPT,
        },
        )
        return json.loads(response.text)
    
    def extract_entities_and_relations(self, prompt: str):
        client = genai.Client(
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
            "response_mime_type": "application/json",
            "response_schema": EntitiesRelations,
            "temperature": 0.0,
            "system_instruction": ENTITIES_RELATIONS_EXTRACTION_SYSTEM_PROMPT,
        },
        )
        return response.text
    
    def generate_content(self, prompt: str) -> str:
        r = ""
        for part in ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": RAG_PROMPT},
                {"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,  # Adjust the creativity of the response
                "min_p": 0.0,
                "num_ctx": 8192,
                #"repeat_penalty": 1.1,  # Adjust the penalty for repeating phrases
            },
            think=True,  # Enable thinking mode for more complex responses
            stream=True,  # Enable streaming for real-time response
            format=Response.model_json_schema()
        ):
            #print(part['message']['content'], end='', flush=True)
            r += part['message']['content']
            
        if r.startswith('{'):
            r = r[1:]
        return json.loads(r)
    
    def generate_content_query_decomposition(self, prompt: str) -> str:
        # Decompose the query into smaller sub-queries

        response = ""
        for part in ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "min_p": 0.0,
                "num_ctx": 8192
            },
            think=True,
            format=MultiQuery.model_json_schema(),
            stream=True
        ):
            # print(part['message']['content'], end='', flush=True)
            response += part['message']['content']
        # remove the first character if it is a {
        if response.startswith('{'):
            response = response[1:]
        
        return json.loads(response)
            
    def generate_content_multi_query(self, prompt):
        # Use the Multi Query technique to generate content
        response = ""
        for part in ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": "Given the following prompt, create 5 prompts that are equal to the original but rephrased: " + prompt + "Give me the 5 prompts in a list. No explanations, just the list."}],
            options={
                "temperature": 0.0,
                "min_p": 0.0,
                "num_ctx": 8192
            },
            think=True,
            stream=True,
            format=MultiQuery.model_json_schema()
        ):
            # print(part['message']['content'], end='', flush=True)
            response += part['message']['content']
        # remove the first character if it is a {
        if response.startswith('{'):
            response = response[1:]
        return json.loads(response)