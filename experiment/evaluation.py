from tqdm import tqdm
from inference import OllamaInference

from metrics import context_faithfulness, context_precision, context_recall, answer_accuracy

import os
import json
import time

class RetrievalEval:
    def __init__(self, model="Qwen3:4B", index_file=None, triples_file=None, entities_file=None, entities_index=None):

        self.inference = OllamaInference(model_name=model, index=index_file, triples=triples_file, entities_file=entities_file, entities_index=entities_index)
        
    def retrieve(self, query):
        """
        Retrieve relevant triples from the knowledge graph based on the query.
        """
        context = self.inference.search.search_semantic_triples(query)
        
        context = [f"{triple[0][0]} {triple[0][1]} {triple[0][2]} - {triple[0][3]}" for triple in context]
        
        return context
    
    def evaluate(self, dataset, results):
        ev_results = {
            "1_hop": {
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "2_hop": {
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "isolated": {
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "hubs": {
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "totalmente_fuori_contesto": {
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        }
        
        for diff in ev_results:
            cp = []
            cr = []
            query_pool = results[diff]
            for i, query in enumerate(query_pool):
                query_id = query.get("id", 0)  # Use the query's ID to find the matching dataset item
                gt = dataset[query_id]["triples"]
                retrieval = query["retrieval"]
                cp.append(context_precision(retrieval, gt))
                cr.append(context_recall(retrieval, gt))
            
            ev_results[diff]["context_precision"] = sum(cp) / len(cp) if cp else 0.0
            ev_results[diff]["context_recall"] = sum(cr) / len(cr) if cr else 0.0

        return ev_results

class GenerationEval:
    def __init__(self, model="Qwen3:4B", index_file=None, triples_file=None, entities_file=None, entities_index=None):
        self.inference = OllamaInference(model_name=model, index=index_file, triples=triples_file, entities_file=entities_file, entities_index=entities_index)
        
    def generate(self, query, strategy='default'):
        """
        Generate an answer based on the query and the retrieved context.
        """
        response, context = self.inference.RAG(query, strategy=strategy, return_context=True)
        return response, context
    
    def evaluate(self, dataset, results, output_filename=None):
        if output_filename is None:
            output_filename = "generation_evaluation_results.json"
        
        # Initialize the results structure
        ev_results = {
            "1_hop": {
                "avg_context_faithfulness": 0.0,
                "avg_analysis_faithfulness": 0.0,
                "avg_answer_accuracy": 0.0,
                "context_faithfulness_explanations": [],
                "analysis_faithfulness_explanations": [],
                "answer_accuracy_explanations": [],
            },
            "2_hop": {
                "avg_context_faithfulness": 0.0,
                "avg_analysis_faithfulness": 0.0,
                "avg_answer_accuracy": 0.0,
                "context_faithfulness_explanations": [],
                "analysis_faithfulness_explanations": [],
                "answer_accuracy_explanations": [],
            },
            "isolated": {
                "avg_context_faithfulness": 0.0,
                "avg_analysis_faithfulness": 0.0,
                "avg_answer_accuracy": 0.0,
                "context_faithfulness_explanations": [],
                "analysis_faithfulness_explanations": [],
                "answer_accuracy_explanations": [],
            },
            "hubs": {
                "avg_context_faithfulness": 0.0,
                "avg_analysis_faithfulness": 0.0,
                "avg_answer_accuracy": 0.0,
                "context_faithfulness_explanations": [],
                "analysis_faithfulness_explanations": [],
                "answer_accuracy_explanations": [],
            },
            "totalmente_fuori_contesto": {
                "avg_context_faithfulness": 0.0,
                "avg_analysis_faithfulness": 0.0,
                "avg_answer_accuracy": 0.0,
                "context_faithfulness_explanations": [],
                "analysis_faithfulness_explanations": [],
                "answer_accuracy_explanations": [],
            }
        }
        
        # Try to load existing results if the file exists
        if os.path.exists(output_filename):
            try:
                with open(output_filename, "r", encoding="utf-8") as f:
                    ev_results = json.load(f)
                print(f"Loaded existing results from {output_filename}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading existing results: {e}. Starting fresh.")
        
        # Keep track of processed queries to avoid re-evaluation
        processed_queries = set()
        for diff in ev_results:
            # Count the number of explanations to determine how many queries were processed
            processed_queries.add((diff, len(ev_results[diff]["context_faithfulness_explanations"])))
        
        total_queries_processed = sum(count for _, count in processed_queries)
        
        for diff in tqdm(results, desc="Processing difficulty levels"):
            # Get current number of processed queries for this difficulty
            current_processed = len(ev_results[diff]["context_faithfulness_explanations"])
            
            faithfulness_scores = []
            analysis_faithfulness_scores = []
            answer_accuracy_scores = []
            
            # Load existing scores if any
            if current_processed > 0:
                # Reconstruct scores from averages (approximate)
                if ev_results[diff]["avg_context_faithfulness"] > 0:
                    faithfulness_scores = [ev_results[diff]["avg_context_faithfulness"]] * current_processed
                if ev_results[diff]["avg_analysis_faithfulness"] > 0:
                    analysis_faithfulness_scores = [ev_results[diff]["avg_analysis_faithfulness"]] * current_processed
                if ev_results[diff]["avg_answer_accuracy"] > 0:
                    answer_accuracy_scores = [ev_results[diff]["avg_answer_accuracy"] * 4] * current_processed
            
            for i, item in enumerate(tqdm(results[diff], desc=f"Evaluating {diff} queries")):
                # Skip if this query was already processed
                if i < current_processed:
                    print(f"Skipping already processed query {i+1}/{len(results[diff])} for {diff}")
                    continue
                
                query = item["query"]
                generated_answer = item["generation"]["answer"]
                generated_context = item["retrieval"]
                
                query_id = item.get("id", 0)  # Use the item's ID to find the matching dataset item
                ground_truths = dataset[query_id]["answer"]

                # Faithfulness
                statements, faithfulness_score, explanation = context_faithfulness(query, generated_answer, generated_context)
                faithfulness_scores.append(faithfulness_score)
                ev_results[diff]["context_faithfulness_explanations"].append([(s, e) for s, e in zip(statements, explanation)])

                ground_truths = dataset[query_id]["analysis"]
                generated_answer = item["generation"]["analysis"]
                
                statements, faithfulness_score, explanation = context_faithfulness(query, generated_answer, generated_context, analysis=True)
                analysis_faithfulness_scores.append(faithfulness_score)
                
                ev_results[diff]["analysis_faithfulness_explanations"].append([(s, e) for s, e in zip(statements, explanation)])
                time.sleep(20)
                
                # Answer accuracy
                answer_score, answer_explanation = answer_accuracy(query, ground_truths, generated_answer)
                answer_accuracy_scores.append(answer_score)
                ev_results[diff]["answer_accuracy_explanations"].append(answer_explanation)
                time.sleep(20)
                
                # Update averages after each query
                ev_results[diff]["avg_context_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
                ev_results[diff]["avg_analysis_faithfulness"] = sum(analysis_faithfulness_scores) / len(analysis_faithfulness_scores) if analysis_faithfulness_scores else 0.0
                ev_results[diff]["avg_answer_accuracy"] = sum(answer_accuracy_scores) / (4 * len(answer_accuracy_scores)) if answer_accuracy_scores else 0.0
                time.sleep(20)
                
                # Save results after each query evaluation
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(ev_results, f, indent=4, ensure_ascii=False)
                    print(f"Saved progress: {diff} - Query {i+1}/{len(results[diff])}")
                except Exception as e:
                    print(f"Error saving results: {e}")
                
                time.sleep(10)
            
        return ev_results


if __name__ == "__main__":
    import argparse
    
    # Define the experiment strategies
    STRATEGIES = [
        #"source-default", # F-R
        #"source-multiquery", # F-R
        "nosource-default", # F-R
        #"nosource-multiquery", # F-R
        #"source-multiquery-extraction", # F-R
        #"source-default-extraction", # F-R
        #"nosource-default-extraction", # F-R
        #"graphrag"
    ]
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES,
                        help="Retrieval and generation strategy to use", default="source-default")
    parser.add_argument("--model", type=str, default="Qwen3:4B",
                        help="Model name for Ollama inference")
    parser.add_argument("--index", type=str, default="gold_embedding/EMB_withSource.index",
                        help="Path to the index file")
    parser.add_argument("--triples", type=str, 
                        default="docs_kg/aggregated_knowledge_graph_normalized.json",
                        help="Path to the triples file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename prefix (default: based on strategy)")
    args = parser.parse_args()
    
    # Set output filenames based on strategy if not specified
    output_prefix = args.output if args.output else f"results_{args.strategy}"
    
    # Parse strategy components
    use_source = args.strategy.startswith("source")
    extraction = "extraction" in args.strategy
    if "extraction" in args.strategy:
        # nosource-default-extraction => "default-extraction"
        # source-multiquery-extraction => "multiquery-extraction"
        retrieval_strategy = args.strategy.split("-")[1] + "-extraction"
    elif "multiquery" in args.strategy:
        retrieval_strategy = "multiquery"
    elif "decomposition" in args.strategy:
        retrieval_strategy = "decomposition"
    elif "default" in args.strategy:
        retrieval_strategy = "default"
    elif "graphrag" in args.strategy:
        retrieval_strategy = "graphrag"
    
    # print(f"Running experiment with strategy: {args.strategy}")
    # print(f"- Use source: {use_source}")
    # print(f"- Extraction: {extraction}")
    # print(f"- Retrieval strategy: {retrieval_strategy}")
    
    # Load the dataset
    with open("gold_dataset/EmPULIA-QA.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    retrieval_eval = RetrievalEval(model=args.model, index_file=args.index, triples_file=args.triples)
    generation_eval = GenerationEval(model=args.model, index_file=args.index, triples_file=args.triples)
    
    # Initialize evaluators with command-line arguments
    # for strategy in tqdm(STRATEGIES, desc="Running strategies"):
    #     # Set output filenames based on strategy if not specified
    #     output_prefix = args.output if args.output else f"results_{strategy}"
        
    #     output_filename = f"results/{output_prefix}.json"
        
    #     if "nosource" in strategy:
    #         args.index = "gold_embedding/EMB_withoutSource.index"
    #         retrieval_eval = RetrievalEval(model=args.model, index_file=args.index, triples_file=args.triples)
    #         generation_eval = GenerationEval(model=args.model, index_file=args.index, triples_file=args.triples)
    #     elif "source" in strategy:
    #         args.index = "gold_embedding/EMB_withSource.index"
    #         retrieval_eval = RetrievalEval(model=args.model, index_file=args.index, triples_file=args.triples)
    #         generation_eval = GenerationEval(model=args.model, index_file=args.index, triples_file=args.triples)
    #     elif "graphrag" in strategy:
    #         args.entities_file = "gold_embedding/EMB_entities.index"
    #         args.entities_index = "docs_kg/aggregated_knowledge_graph.json"
        
    #         retrieval_eval = RetrievalEval(model=args.model, index_file=args.index, triples_file=args.triples, entities_file=args.entities_file, entities_index=args.entities_index)
    #         generation_eval = GenerationEval(model=args.model, index_file=args.index, triples_file=args.triples, entities_file=args.entities_file, entities_index=args.entities_index)
        
    #     if "extraction" in strategy:
    #     # nosource-default-extraction => "default-extraction"
    #     # source-multiquery-extraction => "multiquery-extraction"
    #         retrieval_strategy = strategy.split("-")[1] + "-extraction"
    #     elif "multiquery" in strategy:
    #         retrieval_strategy = "multiquery"
    #     elif "decomposition" in strategy:
    #         retrieval_strategy = "decomposition"
    #     elif "default" in strategy:
    #         retrieval_strategy = "default"
    #     elif "graphrag" in strategy:
    #         retrieval_strategy = "graphrag"
        
        
    #     print(f"Running experiment with strategy: {strategy}")
    #     print(f"- Use source: {use_source}")
    #     print(f"- Extraction: {extraction}")
    #     print(f"- Retrieval strategy: {retrieval_strategy}")
        
    #     if os.path.exists(output_filename):
    #         print(f"Resuming from existing results file: {output_filename}")
    #         with open(output_filename, "r", encoding="utf-8") as f:
    #             results = json.load(f)
    #     else:
    #         results = {
    #             "1_hop": [],
    #             "2_hop": [],
    #             "isolated": [],
    #             "hubs": [],
    #             "totalmente_fuori_contesto": []
    #         }

    #     processed_ids = set()
    #     for diff_list in results.values():
    #         for res in diff_list:
    #             processed_ids.add(res["id"])

    #     for i, item in enumerate(tqdm(dataset, desc="Processing dataset items")):
    #         if i in processed_ids:
    #             print(f"Item {i} already processed. Skipping...")
    #             continue

    #         query = item["question"]
            
    #         generation_results, context = generation_eval.generate(query, strategy=retrieval_strategy)
            
    #         # context is a list of triples in bullet points (-)
    #         # e.g., "- subject predicate object (Fonte: source) - subject predicate object (Fonte: source)"
    #         if not context:
    #             context = []
    #         else:
    #             context = context.split("\n")
    #             context = [c[2:] for c in context]
    #             # replace (Fonte:with -, then the final character is removed
    #             context = [c.replace("(Fonte:", "-")[:-1] for c in context if c.strip()]
                
    #         results[item["difficulty"]].append({
    #             "id": i,
    #             "query": query,
    #             "retrieval": context,
    #             "generation": generation_results
    #         })
            
    #         # Save the results with strategy in filenames
    #         with open(output_filename, "w", encoding="utf-8") as f:
    #             json.dump(results, f, indent=4, ensure_ascii=False)
        
    #     del retrieval_eval
    #     del generation_eval
    #     gc.collect()
    
    ## EVALUATION
    with open(f"results/{output_prefix}.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    results = {
        "1_hop": results["1_hop"],
        "2_hop": results["2_hop"],
        "isolated": results["isolated"],
        "hubs": results["hubs"],
        "totalmente_fuori_contesto": results["totalmente_fuori_contesto"]
    }

    retrieval_results = retrieval_eval.evaluate(
        dataset,
        results
    )
    
    with open(f"results/{output_prefix}_retrieval_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=4, ensure_ascii=False)
    
    generation_results = generation_eval.evaluate(
        dataset,
        results,
        output_filename=f"results/{output_prefix}_generation_evaluation_results.json"
    )
    
    with open(f"results/{output_prefix}_generation_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(generation_results, f, indent=4, ensure_ascii=False)