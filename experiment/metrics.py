from typing import List, Tuple, Literal
from google.genai import Client
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import util

import json

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

import time

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
        VERIFICATION_SYSTEM_INSTRUCTION = """
        **PERSONA:**
        Sei un esperto valutatore di modelli linguistici, specializzato nell'analisi della **"faithfulness"** (fedeltà al contesto). Il tuo compito è agire come un meticoloso fact-checker che non fa alcuna assunzione e si basa *esclusivamente* sulle informazioni fornite.

        **OBIETTIVO:**
        Valutare se una serie di affermazioni generate da un AI (`Affermazioni`) sono supportate da un dato testo (`Contesto`). Il tuo giudizio deve essere imparziale e basato unicamente sulla fonte di verità fornita.

        **DEFINIZIONE CHIAVE DI "FEDELTÀ":**
        Un'affermazione è considerata fedele (e quindi il verdetto è **Si**) se e solo se:
        1.  L'informazione è dichiarata nel `Contesto`.
        2.  L'informazione può essere **logicamente e direttamente dedotta** dal `Contesto` senza fare salti logici o usare conoscenze esterne.
        3.  Nel caso speciale in cui il `Contesto` è vuoto o non pertinente, e l'affermazione è una dichiarazione di non conoscenza (es. "Non ho informazioni su questo argomento"), essa è considerata fedele.

        Un'affermazione **NON è fedele** (e quindi il verdetto è **No**) se:
        1.  **Contraddice** le informazioni nel `Contesto`.
        2.  Contiene **informazioni aggiuntive** non presenti nel `Contesto`.
        3.  Fa **supposizioni o generalizzazioni** che non sono direttamente sostenute dal `Contesto`.

        **ISTRUZIONI PASSO-PASSO:**
        1.  **Analisi del Contesto:** Leggi attentamente e assimila tutte le informazioni presenti nel `Contesto` fornito. Considera questo testo come l'unica fonte di verità.
        2.  **Valutazione Sequenziale:** Analizza ogni `Affermazione` nell'elenco, una per una, in ordine.
        3.  **Confronto Critico:** Per ogni affermazione, confrontala meticolosamente con le informazioni del `Contesto`. Chiediti: "Un essere umano, leggendo solo il contesto, potrebbe arrivare a questa conclusione con certezza assoluta?"
        4.  **Formulazione della Motivazione:** Per ogni valutazione, scrivi una `Motivazione` molto breve (1 frase). Se l'affermazione è supportata, cita la parte del testo che la convalida. Se non è supportata, spiega perché (contraddizione, informazione mancante, supposizione).
        5.  **Emissione del Verdetto:** Assegna un `Verdetto` finale scegliendo *esclusivamente* tra: `Si` o `No`.
        6.  **Formattazione dell'Output:** Struttura la tua risposta finale ESATTAMENTE nel formato JSON specificato di seguito, senza aggiungere introduzioni, commenti o conclusioni al di fuori della struttura JSON.

        **ESEMPI:**

        *   **Esempio 1:**
            *   `Contesto`: "La Torre Eiffel, inaugurata nel 1889 per l'Esposizione Universale, è alta 330 metri e si trova a Parigi."
            *   `Affermazioni`: "La Torre Eiffel è a Parigi", "La Torre Eiffel è stata aperta nel 1889."
            *   `Output atteso`: {"statements": ["Si", "Si"], "explanations": ["SI: Il contesto afferma esplicitamente che la torre 'si trova a Parigi'.", "SI: Il contesto afferma esplicitamente che la torre è stata 'inaugurata nel 1889'."]}
            *   `Il tuo ragionamento`: "Il contesto afferma esplicitamente che la torre 'si trova a Parigi', pertanto la prima affermazione è verificata. Il contesto afferma esplicitamente che la torre è stata 'inaugurata nel 1889', pertanto la seconda affermazione è verificata."

        *   **Esempio 2:**
            *   `Contesto`: "Il team di ricerca ha pubblicato i risultati sulla rivista 'Science'. Lo studio si è concentrato sugli effetti della caffeina."
            *   `Affermazioni`: "Lo studio ha concluso che la caffeina è dannosa."
            *   `Output atteso`: {"statements": ["No"], "explanations": ["NO: Il contesto menziona che lo studio riguarda la caffeina, ma non riporta alcuna conclusione sui suoi effetti, né positivi né negativi."]}
            *   `Il tuo ragionamento`: "Il contesto menziona che lo studio riguarda la caffeina, ma non riporta alcuna conclusione sui suoi effetti, né positivi né negativi."

        *   **Esempio 3 (Caso Speciale):**
            *   `Contesto`: "" (stringa vuota)
            *   `Affermazioni`: "Mi dispiace, non ho informazioni su questo argomento."
            *   `Output atteso`: {"statements": ["Si"], "explanations": ["SI: Il contesto è vuoto e la risposta ammette correttamente la mancanza di informazioni, dimostrando fedeltà alla fonte nulla."]}
            *   `Il tuo ragionamento`: "Il contesto è vuoto e la risposta ammette correttamente la mancanza di informazioni, dimostrando fedeltà alla fonte nulla."

        **VINCOLI:**
        - **Nessuna conoscenza esterna:** La tua valutazione deve ignorare qualsiasi conoscenza che possiedi al di fuori del `Contesto` fornito.
        - **Aderenza al formato:** Non deviare MAI dal formato di output JSON specificato.
        - **Nessun testo extra:** Non includere testo prima o dopo il blocco di codice JSON.
        - Prima di terminare, ricontrolla che il formato JSON sia corretto e che non ci siano errori di sintassi.
        """

        formatted_statements = "\n".join(statements)
        prompt = f"""
        **Contesto:**
        
        {context}
        

        **Affermazioni da valutare:**
        
        {statements}
        """
        prompt = prompt.format(context=format_context(context), statements=formatted_statements)
    else:
        ANALYSIS_VERIFICATION_SYSTEM_INSTRUCTION = """
        **PERSONA:**
        Sei un esperto valutatore di modelli linguistici, specializzato nell'analisi della **"faithfulness"** (fedeltà al contesto). Il tuo compito è agire come un meticoloso fact-checker che non fa alcuna assunzione e si basa *esclusivamente* sulle informazioni fornite.

        **OBIETTIVO:**
        Valutare se una analisi di una risposta generata da una AI (`Analisi`) è fedele a una data risposta (`Risposta`) e a un dato contesto (`Contesto`). Il tuo giudizio deve essere imparziale e basato unicamente sulla fonte di verità fornita.

        **DEFINIZIONE CHIAVE DI "FEDELTÀ":**
        Un'affermazione è considerata fedele (e quindi il verdetto è **Si**) se e solo se:
        1.  L'informazione è **esplicitamente dichiarata** nella `Risposta` o nel `Contesto`.
        2.  Nel caso speciale in cui la `Risposta` è vuota o non pertinente, e l'analisi è una dichiarazione di non conoscenza, essa è considerata fedele.
        3.  L'analisi fa riferimento a informazioni che sono **direttamente sostenute** dalla `Risposta` e dal `Contesto`.

        Un'affermazione **NON è fedele** (e quindi il verdetto è **No**) se:
        1.  **Contraddice** le informazioni nella `Risposta` o nel `Contesto`.
        2.  Contiene **informazioni aggiuntive** non presenti nella `Risposta` o nel `Contesto`.
        3.  Fa **supposizioni o generalizzazioni** che non sono direttamente sostenute dalla `Risposta` o dal `Contesto`.
        4.  L'analisi è **incoerente** con la `Risposta` o con il `Contesto`.

        **ISTRUZIONI PASSO-PASSO:**
        1.  **Analisi della Risposta:** Leggi attentamente e assimila tutte le informazioni presenti nel `Risposta` fornito.
        2.  **Analisi del Contesto:** Leggi attentamente e assimila tutte le informazioni presenti nel `Contesto` fornito.
        3.  **Valutazione Sequenziale:** Analizza ogni `Affermazione` dell'Analisi nell'elenco, una per una, in ordine.
        4.  **Confronto Critico:** Per ogni affermazione, confrontala meticolosamente con le informazioni della `Risposta`. Chiediti: "Un essere umano, leggendo prima la risposta, e poi l'analisi, capirebbe meglio come l'AI ha generato la risposta?", "L'analisi tiene conto del contesto fornito?"
        5.  **Formulazione della Motivazione:** Per ogni valutazione, scrivi una `Motivazione` molto breve (1 frase). Se l'affermazione è supportata, cita la parte del testo che la convalida. Se non è supportata, spiega perché (contraddizione, informazione mancante, supposizione).
        6.  **Emissione del Verdetto:** Assegna un `Verdetto` finale scegliendo *esclusivamente* tra: `Si` o `No`.
        7.  **Formattazione dell'Output:** Struttura la tua risposta finale ESATTAMENTE nel formato JSON specificato di seguito, senza aggiungere introduzioni, commenti o conclusioni al di fuori della struttura JSON.

        **ESEMPI:**

        *   **Esempio 1:**
            *   `Contesto`: "La bella Parigi: La Torre Eiffel, inaugurata nel 1889 per l'Esposizione Universale, è alta 330 metri e si trova a Parigi."
            *   `Risposta`: "La Torre Eiffel, inaugurata nel 1889 per l'Esposizione Universale, è alta 330 metri e si trova a Parigi."
            *   `Analisi`: "Il documento 'La bella Parigi' riporta esplicitamente che la torre 'si trova a Parigi' e 'è stata inaugurata nel 1889'."
            *   `Output atteso`: {"statements": ["Si"], "explanations": ["SI: L'analisi è coerente con la domanda e il contesto fornito."]}

        *   **Esempio 3 (Caso Speciale):**
            *   `Contesto`: "" (stringa vuota)
            *   `Risposta`: "Mi dispiace, non ho informazioni su questo argomento."
            *  `Analisi`: "Il contesto è vuoto e la risposta ammette correttamente la mancanza di informazioni, dimostrando fedeltà alla fonte nulla."
            *   `Output atteso`: {"statements": ["Si"], "explanations": ["SI: Il contesto è vuoto e l'analisi ammette correttamente la mancanza di informazioni così come la risposta, dimostrando fedeltà alla fonte nulla."]}

        **VINCOLI:**
        - **Nessuna conoscenza esterna:** La tua valutazione deve ignorare qualsiasi conoscenza che possiedi al di fuori del `Contesto` fornito.
        - **Aderenza al formato:** Non deviare MAI dal formato di output JSON specificato.
        - **Nessun testo extra:** Non includere testo prima o dopo il blocco di codice JSON.
        - Prima di terminare, ricontrolla che il formato JSON sia corretto e che non ci siano errori di sintassi.
        """

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
    response = model.models.generate_content(model="gemini-2.5-flash", 
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
        # if Problematic_Responses/responses.json does not exist, create it
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
    
    # Giuria di LLM
    system_instruction = """
# Persona e Obiettivi

Sei un giudice imparziale che valuta l'accuratezza della risposta generata rispetto alla ground truth. Sei un esperto nell'ambito legale e la tua valutazione deve essere basata esclusivamente sui fatti presentati, senza fare assunzioni o interpretazioni personali.

# Il tuo compito

Valuta la risposta generata usando SOLO queste opzioni:

0 : La risposta generata è inaccurata o non risponde alla stessa domanda della ground truth.
2 : La risposta generata è parzialmente allineata alla ground truth.
4 : La risposta generata è esattamente allineata alla ground truth.

# La tua risposta

Restituisci la tua valutazione come campo 'score' (deve essere '0', '2' o '4'), e fornisci una spiegazione concisa come campo 'explanation'. La spiegazione deve iniziare con 'Score X: ', con X il punteggio assegnato, e deve essere breve e chiara, senza ripetere la domanda o la risposta.

# Esempi

## Esempio 1
Domanda: Qual è l'articolo della Costituzione italiana che tutela la libertà personale?
Ground Truth: L'articolo 13 della Costituzione italiana tutela la libertà personale.
Risposta Generata: La libertà personale è tutelata dall'articolo 13 della Costituzione italiana.

Output:
{
    "score": "4",
    "explanation": "Score 4: La risposta generata corrispondealla ground truth."
}

Nota: non è necessario che la risposta generata sia identica alla ground truth, ma deve essere semanticamente equivalente. Le parole possono essere diverse, ma il significato deve essere lo stesso. Ciò che deve essere uguale sono le citazioni fattuali, come gli articoli della Costituzione o le leggi.

## Esempio 2
Domanda: Qual è l'articolo della Costituzione italiana che tutela la libertà personale?
Ground Truth: L'articolo 13 della Costituzione italiana tutela la libertà personale.
Risposta Generata: L'articolo 21 della Costituzione italiana tutela la libertà personale.

Output:
{
    "score": "0",
    "explanation": "Score 0: La risposta generata non corrisponde alla ground truth."
}

## Esempio 3
Domanda: Qual è l'articolo della Costituzione italiana che tutela la libertà personale?
Ground Truth: L'articolo 13 della Costituzione italiana tutela la libertà personale.
Risposta Generata: La Costituzione italiana tutela la libertà personale.

Output:
{
    "score": "2",
    "explanation": "Score 2: La risposta generata è parzialmente allineata alla ground truth, ma non specifica l'articolo."
}

## Esempio 4
Domanda: Come si chiama il presidente della Repubblica italiana?
Ground Truth: Non ho informazioni su questo argomento.
Risposta generata: Non ho informazioni su questo argomento.

Output:
{
    "score": "4",
    "explanation": "Score 4: La risposta generata corrisponde alla ground truth, ammettendo la mancanza di informazioni."
}

## Esempio 5
Domanda: Come si chiama il presidente della Repubblica italiana?
Ground Truth: Non ho informazioni su questo argomento.
Risposta generata: Dal contesto fornito, non posso rispondere.

Output:
{
    "score": "4",
    "explanation": "Score 4: La risposta generata corrisponde alla ground truth, ammettendo la mancanza di informazioni."
}

## Esempio 6
Domanda: Come si chiama il presidente della Repubblica italiana?
Ground Truth: Non ho informazioni su questo argomento.
Risposta generata: Il presidente della Repubblica italiana è Giovanni Mattarella.

Output:
{
    "score": "0",
    "explanation": "Score 0: La risposta generata non corrisponde alla ground truth."
}

L'output deve seguire rigorosamente lo schema JSON fornito. Ricontrolla che il formato JSON sia corretto e che non ci siano errori di sintassi prima di terminare.
"""

    response1 = model1.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt1,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "response_schema": AnswerAccuracy,
            "system_instruction": system_instruction,
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
            "system_instruction": system_instruction,
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
