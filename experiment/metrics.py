from typing import List, Tuple, Literal
from google.genai import Client
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import util

import json

from pydantic import BaseModel

import time

from prompts import VERIFICATION_SYSTEM_INSTRUCTION, ANALYSIS_VERIFICATION_SYSTEM_INSTRUCTION, LLM_JURY_SYSTEM_INSTRUCTION

from dotenv import load_dotenv
load_dotenv()

class FaithfulnessMetric(BaseModel):
    statements: List[Literal["Si", "No"]]
    explanations: List[str]

class StatementExtraction(BaseModel):
    statements: List[str]

class AnswerAccuracy(BaseModel):
    score: Literal["0", "2", "4"]
    explanation: str

def context_precision(retrieved, ground_truth):
    """
    Computes context precision as described:
    For each element in retrieved, if it is in ground_truth, save (number of ground_truth elements found so far) / (position in retrieved, 1-based).
    Returns the average of these precision values.
    """
    if not retrieved and not ground_truth:
        return 1.0
    if not retrieved or not ground_truth:
        return 0.0
    found = 0
    precisions = []
    for idx, item in enumerate(retrieved):
        if item in ground_truth:
            found += 1
            precisions.append(found / (idx + 1))
    return sum(precisions) / len(precisions) if precisions else 0.0

def context_recall(retrieved, ground_truth):
    """
    Computes context recall as described:
    For each element in ground_truth, if it is in retrieved, save (number of retrieved elements found so far) / (position in ground_truth, 1-based).
    Returns the average of these recall values.
    """
    if not retrieved and not ground_truth:
        return 1.0
    if not retrieved or not ground_truth:
        return 0.0

    found = 0
    recalls = []
    for idx, item in enumerate(ground_truth, 1):
        if item in retrieved:
            found += 1
            recalls.append(found / idx)
    return sum(recalls) / len(recalls) if recalls else 0.0

def setup_gemini_api():
    """
    Set up the Gemini API with the API key from environment variables.
    """
    return Client()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_statements(model, question: str, answer: str) -> List[str]:
    """
    Extract individual statements from an answer using Gemini API.
    
    Args:
        model: Initialized Gemini model
        question: The original question
        answer: The answer to analyze
        
    Returns:
        A list of extracted statements
    """
    prompt = (f"Data una domanda e una risposta, crea una o più affermazioni da ciascuna frase "
              f"della risposta fornita.\n\nDomanda: {question}\nRisposta: {answer}")

    response = model.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": StatementExtraction,
            "temperature": 0.0,
            "thinking_config": {
                "thinking_budget": 0
            },
        })
    
    response = json.loads(response.text)

    return response['statements']

def format_context(context):
    if len(context) == 0:
        return "Nessun contesto disponibile."
    # Format the context as bullet points
    return "\n".join([f"- {triple}" for triple in context])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def verify_statements(model, statements: List[str], context: str, analysis=False, answer=None) -> Tuple[int, List[bool]]:
    """
    Verify which statements are supported by the context using Gemini API.
    
    Args:
        model: Initialized Gemini model
        statements: List of statements to verify
        context: The retrieval context
        
    Returns:
        A tuple containing:
        - The number of supported statements
        - A list of boolean values indicating whether each statement is supported
    """
    
    if not analysis:
        formatted_statements = "\n".join(statements)
        prompt = f"""
        **Contesto:**
        
        {context}
        

        **Affermazioni da valutare:**
        
        {statements}
        """
        prompt = prompt.format(context=format_context(context), statements=formatted_statements)
    else:
        formatted_statements = "\n".join(statements)
        prompt = f"""
        **Contesto:**
        
        {context}
        
        **Risposta:**
        
        {answer}
        

        **Analisi da valutare:**
        
        {statements}
        """
        
        prompt = prompt.format(context=format_context(context), answer=answer, statements=formatted_statements)
    
    # print length in token of the prompt
    response = model.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
        config={
            "thinking_config": {
            "thinking_budget": 0
        },
            "response_mime_type": "application/json",
            "response_schema": FaithfulnessMetric,
            "temperature": 0.0,
            "system_instruction": VERIFICATION_SYSTEM_INSTRUCTION if not analysis else ANALYSIS_VERIFICATION_SYSTEM_INSTRUCTION,
        }
    )
    total_tokens = model.models.count_tokens(
        model="gemini-2.5-flash", contents=response.text
    )
    try:
        explanations = json.loads(response.text)['explanations']
        response = json.loads(response.text)['statements']
    except json.JSONDecodeError as e:
        print("total_tokens response: ", total_tokens, " se è 65523 probabilmente la risposta è troppo lunga ed è stata troncata.")
        import os
        if not os.path.exists("Problematic_Responses"):
            os.makedirs("Problematic_Responses")
        if not os.path.exists("Problematic_Responses/responses.json"):
            with open("Problematic_Responses/responses.json", "w") as f:
                f.write("")
        with open("Problematic_Responses/responses.json", "a") as f:
            f.write(response.text)
        print(f"Error decoding JSON: {e}")
        print(f"Response text: {response.text}")
        return 0, [], []
    supported_statements = []
    supported_count = 0

    for i, (resp, stmt) in enumerate(zip(response, statements)):
        resp = resp.lower().strip()
        is_supported = (resp == "si" or resp == "sì")

        supported_statements.append(is_supported)
        if is_supported:
            supported_count += 1
    time.sleep(20)
    return supported_count, supported_statements, explanations

def context_faithfulness(question: str, generated_answer: str, contexts: List[str], analysis=False) -> float:
    """
    Computes the faithfulness metric.
    
    Args:
        questions: List of questions
        answers: List of answers to evaluate
        contexts: List of contexts used to generate the answers
        
    Returns:
        The average faithfulness score across all answers
    """
    model = setup_gemini_api()

    if not contexts and generated_answer == "Non ho informazioni su questo argomento.":
        return [], 1.0, []
    
    statements = extract_statements(model, question, generated_answer)
    
    if not statements:
        return [], 0.0, []

    if analysis:
        supported_count, _, explanations = verify_statements(model, statements, contexts, analysis=True, answer=generated_answer)
    else:
        supported_count, _, explanations = verify_statements(model, statements, contexts)
    
    return statements, supported_count / len(statements) if statements else 0.0, explanations

def answer_accuracy(question, ground_truth: str, generated_answer: str) -> Tuple[float, List[str]]:
    """
    Computes the accuracy of generated answers against ground truth answers.
    
    Args:
        ground_truths: List of ground truth answers.
        generated_answers: List of generated answers to evaluate.
        model_name: Name of the embedding model to use.
        
    Returns:
        A tuple containing:
        - The average accuracy score (between 0 and 1).
        - A list of explanations for each answer's accuracy.
    """
    model1 = setup_gemini_api()
    model2 = setup_gemini_api()
    
    # give the two answer to the model and ask to rate them
    # Two LLM
    # Average the scores
    prompt1 = (f"Domanda: {question}\n"
              f"Ground Truth: {ground_truth}\n"
              f"Risposta Generata: {generated_answer}\n"
              f"Valutazione: ")
    
    # We swap ground_truth and generated_answer to get a different perspective
    # This is to ensure that both models evaluate the same content but from different angles
    prompt2 = (f"Domanda: {question}\n"
              f"Ground Truth: {generated_answer}\n"
              f"Risposta Generata: {ground_truth}\n"
              f"Valutazione: ")
    
    # LLM Jury

    response1 = model1.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt1,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "response_schema": AnswerAccuracy,
            "system_instruction": LLM_JURY_SYSTEM_INSTRUCTION,
            "thinking_config": {
                "thinking_budget": 0
            }
        }
    )
    response2 = model2.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt2,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "response_schema": AnswerAccuracy,
            "system_instruction": LLM_JURY_SYSTEM_INSTRUCTION,
            "thinking_config": {
                "thinking_budget": 0
            }
        }
    )
    response1 = json.loads(response1.text)
    response2 = json.loads(response2.text)
    score = (int(response1['score']) + int(response2['score'])) / 2
    explanation = {
        "response1": response1['explanation'],
        "response2": response2['explanation'],
    }

    return score, explanation

def answer_semantic_similarity(ground_truths: List[str], generated_answers: List[str], model=None) -> List[float]:
    """
    Computes the semantic similarity between ground truth answers and generated answers.
    
    Args:
        ground_truths: List of ground truth answers.
        generated_answers: List of generated answers to evaluate.
        model_name: Name of the embedding model to use.
        
    Returns:
        A list of similarity scores (between 0 and 1) for each answer pair.
    """
    if not model:
        raise ValueError("Embedding model must be provided.")
    
    # Calculate similarities
    # Step 1 & 2: Vectorize both texts
    truth_embedding = model.encode(ground_truths, convert_to_tensor=True)
    answer_embedding = model.encode(generated_answers, convert_to_tensor=True)
    
    # Step 3: Compute cosine similarity
    return util.cos_sim(truth_embedding, answer_embedding).item()

def average_semantic_similarity(ground_truths: List[str], generated_answers: List[str], model=None) -> float:
    """
    Computes the average semantic similarity across all ground truth and generated answer pairs.
    
    Args:
        ground_truths: List of ground truth answers.
        generated_answers: List of generated answers to evaluate.
        model_name: Name of the embedding model to use.
        
    Returns:
        The average semantic similarity score.
    """
    similarities = answer_semantic_similarity(ground_truths, generated_answers, model)
    return sum(similarities) / len(similarities) if similarities else 0.0
