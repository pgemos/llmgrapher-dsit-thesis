# ==============================================================================
# FINAL EXPERIMENT SCRIPT: LLM vs. Baselines for Knowledge Graph Generation
# ==============================================================================
# SCRIPT DESCRIPTION
# ==============================================================================
# This script runs the "Triplet Context" experiment against a FILTERED Knowledge Base.
#
# Key Characteristics:
# 1. Filtered KB: Uses `full_freebase_df_filtered.parquet`, which excludes "junk" 
#    entities (short names, stopwords) to reduce retrieval noise.
# 2. Context: Uses "Context Triplets" (neighboring nodes/edges) for LLM reranking.
# 3. Goal: Evaluates if cleaning the target KB improves the precision of the 
#    standardization pipeline.
# ==============================================================================

import os
import json
import pandas as pd
import ollama
import spacy
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from wikigraphs.data import paired_dataset, io_tools
from itertools import combinations

pd.set_option('display.max_colwidth', 300)
pd.set_option('display.width', 120) # For better terminal display of DataFrames

DEBUG=False

# --- 1. Configuration ---
SAMPLES_TO_RUN = 2000
LLM_MODEL = "llama3.1:8b"
LINKING_STRATEGY = "simple"
BATCH_RERANKING_SIZE = 20
WIKIGRAPHS_DATA_DIR = "data/wikigraphs/"
CHECKPOINT_DIR = "checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
UPDATE_DISTANT_SUPERVISION_ONLY_IN_CHECKPOINTS = True
NAME_DB_PATH = "name_embeddings"
PREDICATE_DB_PATH = "predicate_embeddings"

# --- 2. Prompt Engineering: Thorough Prompts for Generation and Evaluation ---
LLM_PROMPT_VECTORDB_FOR_CR = """You are a knowledge graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```) Your task is to extract the ontology 
of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n
Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n
\tTerms may include object, entity, location, organization, person, \n
\tcondition, acronym, documents, service, concept, etc.\n
\tTerms should be as atomistic as possible\n\n
Thought 2: Think about how these terms can have one on one relation with other terms.\n
\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n
\tTerms can be related to many other terms\n\n
Thought 3: Find out the relation between each such related pair of terms. \n\n
Format your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the follwing: \n
[
   {
       "node_1": "A concept from extracted ontology",\n
       "node_2": "A related concept from extracted ontology",\n
       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n
   }, {...}\n
]\n
Do not add any other comment before or after the json. Respond ONLY with a well formed json that can be directly read by a program."""

LLM_PROMPT_INHERENT_CR = """You are a knowledge graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```). Your task is to extract the ontology 
of terms mentioned in the given context and the relationships between them.\n\nThought 1: First, read through the text to identify the core entities. These are the main people, organizations, locations, and concepts being discussed.\n\nThought 2: Pay close attention to coreferences. If an entity is mentioned multiple times with different names (e.g., 'Valkyria Chronicles III', 'the game', 'it'), I must identify the most complete and descriptive name (e.g., 'Valkyria Chronicles III') and use it consistently for all triplets involving that entity.\n\nThought 3: I must also resolve pronouns. If the text says 'it was developed by Sega', the pronoun 'it' must be resolved to the specific entity it refers to from the preceding text. The final triplet should not contain pronouns like 'he', 'she', or 'it'.\n\nThought 4: For each sentence, I will extract relationships as `(node_1, edge, node_2)` triplets. The `edge` should be a concise, verb-oriented phrase describing the relationship.\n\nFormat your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the follwing: \n
[
   {
       "node_1": "Canonical Entity Name",\n
       "node_2": "Another Canonical Entity Name",\n
       "edge": "A concise relationship phrase"\n
   }, {...}\n
]\n
Respond ONLY with a well-formed JSON list. Do not include any introductory text, comments, or explanations in your response."""

LLM_JUDGE_SYS_PROMPT_CATEGORICAL = """You are an expert evaluator for knowledge graph relations. Your task is to determine if two items are semantically equivalent.
You will be given Item A and Item B.

CRITERIA:
- 'High Confidence': They refer to the exact same concept without ambiguity (e.g., 'Sega' and 'Sega Wow'; 'developed by' and 'developer').
- 'Plausible': They are closely related and could be considered equivalent in a broader context, but are not identical (e.g., 'game' and 'tactical role-playing game'; 'is part of' and 'member of').
- 'No Match': The items refer to different concepts.

Your response MUST be a single phrase from the list: ['High Confidence', 'Plausible', 'No Match']. Do not provide any other text."""

LLM_RERANKER_SYS_PROMPT = """You are a highly intelligent entity linking system. Your task is to identify the correct real-world entity from a list of candidates that best fits the given context.

I will provide you with:
1. A "Sentence Context" where an ambiguous entity was found.
2. The "Ambiguous Mention" from that sentence.
3. A numbered list of "Candidates" from a knowledge base.

Your job is to analyze the context and determine which candidate is the most appropriate match for the ambiguous mention.

Your response MUST be only the number of the correct candidate. If no candidate is a good match, respond with the number 0. Do not provide any explanation or any other text.
"""

LLM_RERANKER_BATCH_MODE_SYS_PROMPT = """You are a highly intelligent and precise entity linking system. Your sole task is to identify the correct real-world entity from a list of candidates that best fits the given context.

I will provide you with a batch of items to process. Each item will contain:
1. An "Ambiguous Mention" found in a text.
2. A list of "Context Triplets" showing how this mention was used.
3. A numbered list of "Candidates" from a knowledge base.

Your Thought Process for Each Item:
1.  First, I will read the "Ambiguous Mention".
2.  Next, I will carefully analyze the "Context Triplets" to understand the semantic role of the ambiguous mention.
3.  Then, I will review the numbered list of "Candidates".
4.  I will choose the single best candidate string that fits the context.
5.  If none of the candidates are a good semantic fit, I will use the original "Ambiguous Mention" string as the result.

Your final output MUST be a single, valid JSON dictionary. The keys of the dictionary must be the original "Ambiguous Mention" strings, and the values must be the full string of the candidate you chose.

Example Input Format:
Item 1:
- Ambiguous Mention: "Paris"
- Context Triplets:
- (achilles, fought, paris)
- Candidates:
  1. "Paris, Texas"
  2. "Paris"
  3. "Paris Saint-Germain F.C."
Item 2:
...

Example Response Format:
{
  "Paris": "Paris",
  ...
}

Do not provide any explanation, comments, or any text outside of the single JSON dictionary in your response.
"""

# --- 3. Fixed values

METADATA_PREDICATE_BLACKLIST = [
    'ns/type.object.name',
    'ns/common.topic.description',
    'key/wikipedia.en',
    'ns/common.topic.alias',
    'ns/type.object.id',
    'ns/common.topic.image'
]

# --- 4. Helper and Generation Functions ---

def load_wikigraphs_data(data_root, subset='train', version='max256'):
    print("Loading WikiGraphs dataset...")
    paired_dataset.DATA_ROOT = data_root
    dataset = paired_dataset.ParsedDataset(
        subset=subset, shuffle_data=False, data_dir=None, version=version
    )
    return list(dataset)

def get_ground_truth_graph(pair, predicate_blacklist=METADATA_PREDICATE_BLACKLIST):
    """
    Converts a WikiGraphs pair object into a pandas DataFrame of triplets.
    The function is setup to filter out a blacklist of non-relational metadata predicates. ***
    """
    g = pair.graph
    df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
    df["subject"] = df["src"].apply(lambda node_id: g.nodes()[node_id])
    df["object"] = df["tgt"].apply(lambda node_id: g.nodes()[node_id])
    df = df[["subject", "edge", "object"]]
    df.rename(columns={"edge": "predicate"}, inplace=True)
    
    # --- NEW: Filter out the predicate blacklist ---
    if predicate_blacklist:
        original_len = len(df)
        df = df[~df['predicate'].isin(predicate_blacklist)]
        if len(df) < original_len:
            print(f"Filtered {original_len - len(df)} metadata triplets from ground truth.")
            
    return df.drop_duplicates().reset_index(drop=True)

def fix_prompt_output(text):
    starting_characters = ('"', '{', '}', '[', ']')
    lines = text.splitlines()
    filtered_lines = [line for line in lines if any(line.lstrip().startswith(char) for char in starting_characters)]
    return "\n".join(filtered_lines)

def generate_graph_from_text(text, system_prompt, model=LLM_MODEL):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, length_function=len)
    pages = splitter.split_text(text)
    all_triplets = []
    print(f"Processing text in {len(pages)} chunks...")
    for page in tqdm(pages, desc="Generating graph with LLM", leave=False):
        if len(page.strip()) < 50: continue
        user_prompt = f"context: ```{page}``` \n\n output: "
        try:
            response_dict = ollama.generate(model=model, system=system_prompt, prompt=user_prompt)
            response_text = response_dict["response"]
            cleaned_response = fix_prompt_output(response_text)
            triplets = json.loads(cleaned_response)
            if isinstance(triplets, list) and all(isinstance(i, dict) for i in triplets):
                all_triplets.extend(triplets)
        except Exception:
            continue
    if not all_triplets: return pd.DataFrame(columns=['node_1', 'node_2', 'edge'])
    valid_triplets = [t for t in all_triplets if all(k in t for k in ['node_1', 'node_2', 'edge'])]
    df = pd.DataFrame(valid_triplets)
    return df.drop_duplicates().reset_index(drop=True)

def generate_graph_with_distant_supervision(text, name_to_id_map, relations_lookup, id_to_name_map, nlp, debug_sentence_limit=3 if DEBUG else 0):
    """
    Generates a graph using a true, highly optimized distant supervision heuristic.
    Includes verbose debugging print statements to trace the matching logic.
    """
    if nlp is None: return pd.DataFrame(columns=['subject', 'object', 'predicate'])
    print("Generating graph using Optimized Distant Supervision baseline (DEBUG MODE)...")
    
    all_known_names = set(name_to_id_map.keys())
    doc = nlp(text)
    discovered_triplets = set()
    sentences_debugged = 0

    for sentence in tqdm(doc.sents, desc="Scanning sentences for DS", leave=False):
        sentence_text = sentence.text.lower()
        
        # --- Start Debugging Block ---
        print_debug = (debug_sentence_limit is None) or (sentences_debugged < debug_sentence_limit)
        if print_debug:
            print("\n" + "="*80)
            print(f"ANALYZING SENTENCE: '{sentence.text[:150]}...'")
            print("="*80)

        entities_in_sentence = set()
        for name in all_known_names:
            # Normalize the name from the KG for searching
            search_name = name.strip('"').lower()
            if search_name in sentence_text:
                entities_in_sentence.add(name)
        
        if print_debug:
            print(f"  Found {len(entities_in_sentence)} known entities in this sentence.")
            if entities_in_sentence:
                print(f"  Entities found: {list(entities_in_sentence)[:5]}...") # Print a sample

        if len(entities_in_sentence) < 2:
            if print_debug:
                print("  -> Skipping sentence (fewer than 2 entities found).")
            sentences_debugged += 1
            continue

        for name1, name2 in combinations(entities_in_sentence, 2):
            if print_debug:
                print(f"\n  - Checking pair: ('{name1}', '{name2}')")
            
            ids1 = name_to_id_map.get(name1, [])
            ids2 = name_to_id_map.get(name2, [])
            
            if not ids1 or not ids2:
                if print_debug: print("    -> One or both names not in ID map. Skipping.")
                continue

            for id1 in ids1:
                for id2 in ids2:
                    key = tuple(sorted((id1, id2)))
                    if key in relations_lookup:
                        if print_debug:
                            print(f"    -> SUCCESS: Found relation(s) for ID pair ({id1}, {id2}) in lookup table.")
                        for triplet_record in relations_lookup[key]:
                            if (triplet_record.subject == id1 and triplet_record.object == id2) or \
                               (triplet_record.subject == id2 and triplet_record.object == id1):
                                if print_debug:
                                    print(f"      -> MATCHED: {triplet_record}")
                                discovered_triplets.add((triplet_record.subject, triplet_record.object, triplet_record.predicate))
        
        sentences_debugged += 1

    if not discovered_triplets: 
        print("\nOptimized Distant Supervision found no relations.")
        return pd.DataFrame(columns=['subject', 'object', 'predicate'])
        
    generated_df_with_ids = pd.DataFrame(list(discovered_triplets), columns=['subject', 'object', 'predicate'])
    human_readable_df = create_human_readable_ground_truth(generated_df_with_ids, id_to_name_map)
    
    print(f"\nOptimized Distant Supervision generated {len(human_readable_df)} unique, human-readable triplets.")
    return human_readable_df.reset_index(drop=True)

def generate_graph_with_pattern_matching(text, nlp):
    if nlp is None: return pd.DataFrame(columns=['node_1', 'node_2', 'edge'])
    print("Generating raw graph using Pattern-Matching (spaCy SVO)... ")
    doc = nlp(text)
    discovered_triplets = []
    for token in tqdm(doc, desc="Parsing dependencies for SVO", leave=False):
        if token.pos_ == 'VERB':
            subjects = [child.text for child in token.children if child.dep_ == 'nsubj']
            objects = [child.text for child in token.children if child.dep_ == 'dobj']
            if subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        discovered_triplets.append({"node_1": subj, "node_2": obj, "edge": token.lemma_})
    if not discovered_triplets: return pd.DataFrame(columns=['node_1', 'node_2', 'edge'])
    return pd.DataFrame(discovered_triplets).drop_duplicates().reset_index(drop=True)

def create_human_readable_ground_truth(truth_df, id_to_name_map):
    readable_df = truth_df.copy()
    readable_df['subject'] = readable_df['subject'].apply(lambda x: id_to_name_map.get(x, x))
    readable_df['object'] = readable_df['object'].apply(lambda x: id_to_name_map.get(x, x))
    return readable_df

class VectorDatabase:
    """
    A robust wrapper for ChromaDB that correctly handles both persistent (on-disk)
    and ephemeral (in-memory) clients.
    """
    def __init__(self, collection_name, texts, path=None, similarity="cosine", batch_size=4000):
        import chromadb
        
        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": similarity}
        )
        
        # Populate the collection if it's empty, now with batching and a progress bar
        if self.collection.count() == 0 and texts:
            print(f"Populating new VectorDB '{collection_name}' with {len(texts)} items...")
            
            # Process in batches to show progress
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Building DB '{collection_name}'"):
                batch_texts = texts[i:i+batch_size]
                
                # Create IDs for the current batch
                batch_ids = [f"id{j}" for j in range(i, i + len(batch_texts))]
                
                if batch_texts: # Ensure the batch is not empty
                    self.collection.add(
                        documents=batch_texts,
                        ids=batch_ids
                    )
            print(f"VectorDB '{collection_name}' built successfully.")

    def query(self, query_text, n_results):
        if self.collection.count() == 0: return None
        results = self.collection.query(query_texts=[str(query_text)], n_results=n_results)
        return results['documents'][0] if results and results.get('documents') and results['documents'][0] else None


def standardize_graph_items_simple(graph_df, vector_db, item_type='entity'):
    """
    Performs simple, top-1 nearest neighbor standardization for entities or predicates.
    """

    if item_type not in ("entity", "predicate"):
        raise ValueError(f"Wrong type: `{item_type}`")
    
    if graph_df.empty: return graph_df
    
    print(f"Running SIMPLE standardization for type: `{item_type}`...")
    
    # Determine which columns to process
    if item_type == 'entity':
        columns_to_standardize = ['node_1', 'node_2']
        items_to_standardize = pd.concat([graph_df[col] for col in columns_to_standardize]).unique()
    else: # predicate
        columns_to_standardize = ['edge']
        items_to_standardize = graph_df['edge'].unique()

    item_mapping = {}
    for item in tqdm(items_to_standardize, desc=f"Linking {item_type}s (Simple)", leave=False):
        best_match_list = vector_db.query(query_text=str(item), n_results=1)
        if best_match_list and isinstance(best_match_list, list):
            item_mapping[item] = best_match_list[0]
        else:
            # For predicates, we drop them if no match. For entities, we keep the original.
            item_mapping[item] = item if item_type == 'entity' else None

    std_df = graph_df.copy()
    for col in columns_to_standardize:
        std_df[col] = std_df[col].map(item_mapping)
    
    if item_type == 'predicate':
        std_df.dropna(subset=['edge'], inplace=True)
        
    return std_df.drop_duplicates().reset_index(drop=True)


def standardize_graph_items_with_batch_reranking(
    graph_df, 
    vector_db, 
    item_type='entity',
    batch_size=5, 
    top_k_candidates=5, 
    num_context_triplets=3
):
    """
    A generalized function to standardize entities OR predicates using a mini-batch reranking pipeline.
    """
    if graph_df.empty: return graph_df

    if item_type not in ("entity", "predicate"):
        raise ValueError(f"Wrong type: `{item_type}`")
    
    print(f"Running BATCH RERANKING standardization for type: `{item_type}`...")
    
    item_mapping = {}
    
    # Determine which columns to process
    if item_type == 'entity':
        columns_to_standardize = ['node_1', 'node_2']
        items_to_standardize = list(pd.concat([graph_df[col] for col in columns_to_standardize]).unique())
    else: # predicate
        columns_to_standardize = ['edge']
        items_to_standardize = list(graph_df['edge'].unique())

    for i in tqdm(range(0, len(items_to_standardize), batch_size), desc=f"Linking {item_type}s (Batch Rerank)"):
        batch_items = items_to_standardize[i:i+batch_size]
        
        prompt_body = ""
        candidate_map = {}
        
        for j, item in enumerate(batch_items):
            candidates = vector_db.query(query_text=str(item), n_results=top_k_candidates)
            if not candidates:
                item_mapping[item] = item if item_type == 'entity' else None
                continue
            
            candidate_map[item] = candidates
            
            # Build context for the item
            if item_type == 'entity':
                sample_triplets = graph_df[(graph_df['node_1'] == item) | (graph_df['node_2'] == item)].head(num_context_triplets)
                context_str = "\\n".join([f"- ({row.node_1}, {row.edge}, {row.node_2})" for row in sample_triplets.itertuples()])
                context_label = "Context Triplets"
            else: # predicate
                sample_triplet = graph_df[graph_df['edge'] == item].iloc[0]
                context_str = f"This relationship connects '{sample_triplet.node_1}' to '{sample_triplet.node_2}'."
                context_label = "Context"

            candidate_list_str = "\\n".join([f"  {k+1}. {c}" for k, c in enumerate(candidates)])
            
            prompt_body += (f"Item {j+1}:\\n"
                             f"- Ambiguous Mention: \\\"{item}\\\"\\n"
                             f"- {context_label}:\\n{context_str}\\n"
                             f"- Candidates:\\n{candidate_list_str}\\n\\n")

        if not candidate_map: continue

        try:
            response = ollama.generate(model=LLM_MODEL, system=LLM_RERANKER_BATCH_MODE_SYS_PROMPT, prompt=prompt_body, options={'temperature': 0.0})
            answer_json = json.loads(fix_prompt_output(response['response']))
            
            for item in batch_items:
                if item in answer_json and answer_json[item] in candidate_map.get(item, []):
                    item_mapping[item] = answer_json[item]
                elif item in candidate_map:
                    item_mapping[item] = candidate_map[item][0]
        
        except Exception as e:
            print(f"Warning: Batch LLM call failed ({e}). Falling back to top-1 for this batch.")
            for item in batch_items:
                if item in candidate_map:
                    item_mapping[item] = candidate_map[item][0]

    std_df = graph_df.copy()
    for col in columns_to_standardize:
        std_df[col] = std_df[col].map(item_mapping)
    
    if item_type == 'predicate':
        std_df.dropna(subset=['edge'], inplace=True)
        
    return std_df.drop_duplicates().reset_index(drop=True)


def orchestrate_standardization(
    raw_graph_df, 
    name_db, 
    predicate_db, 
    strategy="simple", 
    batch_size=5  # <-- NEW PARAMETER
):
    """
    A generalized wrapper that runs the full standardization pipeline.

    Args:
        raw_graph_df (pd.DataFrame): The input graph to standardize.
        name_db (VectorDatabase): The vector database for entities.
        predicate_db (VectorDatabase): The vector database for predicates.
        strategy (str): The linking strategy. Choose from:
                        'simple' -> Top-1 vector search.
                        'reranking' -> Sequential LLM reranking (batch_size=1).
                        'batch_reranking' -> Batched LLM reranking.
        batch_size (int): The batch size to use for 'reranking' and 'batch_reranking' strategies.
    """
    print(f"\\n--- Running full standardization pipeline with strategy: '{strategy}' (Batch Size: {batch_size if 'reranking' in strategy else 'N/A'}) ---")
    
    if strategy == "simple":
        std_entity_df = standardize_graph_items_simple(raw_graph_df, name_db, item_type='entity')
        fully_std_df = standardize_graph_items_simple(std_entity_df, predicate_db, item_type='predicate')
    
    elif strategy == "reranking":
        # "reranking" is just batch_reranking with a batch size of 1.
        std_entity_df = standardize_graph_items_with_batch_reranking(raw_graph_df, name_db, item_type='entity', batch_size=1)
        fully_std_df = standardize_graph_items_with_batch_reranking(std_entity_df, predicate_db, item_type='predicate', batch_size=1)

    elif strategy == "batch_reranking":
        std_entity_df = standardize_graph_items_with_batch_reranking(raw_graph_df, name_db, item_type='entity', batch_size=batch_size)
        fully_std_df = standardize_graph_items_with_batch_reranking(std_entity_df, predicate_db, item_type='predicate', batch_size=batch_size)
        
    else:
        raise ValueError(f"Unknown standardization strategy: '{strategy}'. Choose from 'simple', 'reranking', or 'batch_reranking'.")
        
    return fully_std_df
    

# --- Evaluation Functions --- 
llm_judge_cache = {}
def ask_llm_judge_categorical(item_a, item_b):
    cache_key = tuple(sorted((str(item_a).lower(), str(item_b).lower())))
    if cache_key in llm_judge_cache: return llm_judge_cache[cache_key]
    prompt = f"Item A: \\\"{item_a}\\\"\\nItem B: \\\"{item_b}\\\""
    try:
        response = ollama.generate(model=LLM_MODEL, system=LLM_JUDGE_SYS_PROMPT_CATEGORICAL, prompt=prompt)
        answer = response['response'].strip().lower()
        result = 'High Confidence' if 'high confidence' in answer else 'Plausible' if 'plausible' in answer else 'No Match'
        llm_judge_cache[cache_key] = result
        return result
    except Exception: return 'No Match'

def _calculate_metrics_from_counts(true_positives, generated_count, truth_count):
    if generated_count == 0 and truth_count == 0: return {"Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0}
    precision = true_positives / generated_count if generated_count > 0 else 0
    recall = true_positives / truth_count if truth_count > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score}

# ==============================================================================
# CORRECTED EVALUATION FUNCTIONS (WITH DEBUGGING)
# ==============================================================================

def evaluate_pairs(generated_df, truth_df, resilient=False, debug=False, debug_name=""):
    """Evaluates connected node pairs with optional debugging."""
    if generated_df.empty: return {"Precision": 0, "Recall": 0, "F1-Score": 0}
    # Copy and rename generated_df to the canonical schema ('subject', 'object')
    gen_norm = generated_df.rename(columns={'node_1': 'subject', 'node_2': 'object'})
    # Ensure truth_df has the canonical schema (it already does, but this is safe)
    truth_norm = truth_df.rename(columns={'subject': 'subject', 'object': 'object'})
    gen_norm = gen_norm[['subject', 'object']].apply(lambda x: x.astype(str).str.lower().str.strip())
    truth_norm = truth_norm[['subject', 'object']].apply(lambda x: x.astype(str).str.lower().str.strip())
    gen_records = gen_norm.to_records(index=False)
    truth_records = truth_norm.to_records(index=False)
    if resilient:
        generated_set = {tuple(sorted(tuple(p))) for p in gen_records}
        truth_set = {tuple(sorted(tuple(p))) for p in truth_records}
    else:
        generated_set = set(map(tuple, gen_records))
        truth_set = set(map(tuple, truth_records))
    tp = len(generated_set.intersection(truth_set))
    
    if debug:
        print("\n" + "-"*20 + f" DEBUGGING: {debug_name} Pairs " + "-"*20)
        print(f"Generated Set Size: {len(generated_set)}")
        print(f"Ground Truth Set Size: {len(truth_set)}")
        print(f"Intersection (TP): {tp}")
        print("\n--- Sample of Generated Pairs (Sorted) ---")
        for item in sorted(list(generated_set))[:20]: print(f"  -> {item}")
        print("\n--- Sample of Truth Pairs (Sorted) ---")
        for item in sorted(list(truth_set))[:20]: print(f"  -> {item}")
        print("\n--- Intersection Content ---")
        intersection = generated_set.intersection(truth_set)
        if not intersection: print("  -> INTERSECTION IS EMPTY.")
        for item in sorted(list(intersection))[:20]: print(f"  -> {item}")
        print("-"*70)

    return _calculate_metrics_from_counts(tp, len(gen_norm), len(truth_norm))

def evaluate_triplets(generated_df, truth_df, resilient=False, debug=False, debug_name=""):
    """Evaluates full triplets with optional debugging."""
    if generated_df.empty: return {"Precision": 0, "Recall": 0, "F1-Score": 0}

    gen_norm = generated_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'})
    gen_norm = gen_norm.apply(lambda x: x.astype(str).str.lower().str.strip())
    truth_norm = truth_df.apply(lambda x: x.astype(str).str.lower().str.strip())
    
    if resilient:
        gen_set = { (tuple(sorted((p.subject, p.object))), p.predicate) for p in gen_norm.itertuples(index=False) }
        truth_set = { (tuple(sorted((p.subject, p.object))), p.predicate) for p in truth_norm.itertuples(index=False) }
    else:
        gen_set = set(map(tuple, gen_norm[['subject', 'predicate', 'object']].to_records(index=False)))
        truth_set = set(map(tuple, truth_norm[['subject', 'predicate', 'object']].to_records(index=False)))
        
    tp = len(gen_set.intersection(truth_set))

    if debug:
        print("\n" + "-"*20 + f" DEBUGGING: {debug_name} Triplets " + "-"*20)
        print(f"Generated Set Size: {len(gen_set)}")
        print(f"Ground Truth Set Size: {len(truth_set)}")
        print(f"Intersection (TP): {tp}")
        print("Sample of Generated Triplets:", sorted(list(gen_set), key=str)[:5])
        print("Sample of Truth Triplets:", sorted(list(truth_set), key=str)[:5])
        print("-"*60)

    return _calculate_metrics_from_counts(tp, len(gen_norm), len(truth_norm))

def run_comprehensive_evaluation(generated_df, truth_df, debug=False, debug_name=""):
    """
    Runs a multi-layered evaluation, calculating four distinct sets of metrics.
    Includes verbose debugging print statements to trace the matching logic.
    """
    if generated_df.empty: return {
        "Strict Triplets": {"Precision": 0, "Recall": 0, "F1-Score": 0},
        "Resilient Triplets": {"Precision": 0, "Recall": 0, "F1-Score": 0},
        "Semantic (High Confidence)": {"Precision": 0, "Recall": 0, "F1-Score": 0},
        "Semantic (Plausible)": {"Precision": 0, "Recall": 0, "F1-Score": 0}}
    
    # --- 1. Data Prep ---
    gen_norm = generated_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'})
    gen_norm = gen_norm.apply(lambda x: x.astype(str).str.lower().str.strip())
    truth_norm = truth_df.apply(lambda x: x.astype(str).str.lower().str.strip())
    num_gen, num_truth = len(gen_norm), len(truth_norm)

    # --- 2. Strict Triplet Evaluation ---
    gen_strict_set = set(map(tuple, gen_norm[['subject', 'predicate', 'object']].to_records(index=False)))
    truth_strict_set = set(map(tuple, truth_norm[['subject', 'predicate', 'object']].to_records(index=False)))
    strict_triplet_tp = len(gen_strict_set.intersection(truth_strict_set))

    if debug:
        print("\n" + "-"*20 + f" DEBUGGING: {debug_name} (Strict Triplets) " + "-"*20)
        print(f"Generated Set Size: {len(gen_strict_set)}")
        print(f"Ground Truth Set Size: {len(truth_strict_set)}")
        print(f"Intersection (TP): {strict_triplet_tp}")
        print("\n--- Sample of Generated Triplets (Sorted) ---")
        for item in sorted(list(gen_strict_set), key=str)[:20]: print(f"  -> {item}")
        print("\n--- Sample of Truth Triplets (Sorted) ---")
        for item in sorted(list(truth_strict_set), key=str)[:20]: print(f"  -> {item}")
        print("\n--- Intersection Content ---")
        strict_intersection = gen_strict_set.intersection(truth_strict_set)
        if not strict_intersection: print("  -> INTERSECTION IS EMPTY.")
        for item in sorted(list(strict_intersection), key=str)[:20]: print(f"  -> {item}")
        print("-"*70)

    # --- 3. Resilient Triplet Evaluation ---
    gen_resilient_set = { (tuple(sorted((p.subject, p.object))), p.predicate) for p in gen_norm.itertuples(index=False) }
    truth_resilient_set = { (tuple(sorted((p.subject, p.object))), p.predicate) for p in truth_norm.itertuples(index=False) }
    resilient_triplet_tp = len(gen_resilient_set.intersection(truth_resilient_set))

    if debug:
        print("\n" + "-"*20 + f" DEBUGGING: {debug_name} (Resilient Triplets) " + "-"*20)
        print(f"Generated Set Size: {len(gen_resilient_set)}")
        print(f"Ground Truth Set Size: {len(truth_resilient_set)}")
        print(f"Intersection (TP): {resilient_triplet_tp}")
        print("\n--- Sample of Generated Triplets (Sorted) ---")
        for item in sorted(list(gen_resilient_set), key=str)[:20]: print(f"  -> {item}")
        print("\n--- Sample of Truth Triplets (Sorted) ---")
        for item in sorted(list(truth_resilient_set), key=str)[:20]: print(f"  -> {item}")
        print("\n--- Intersection Content ---")
        resilient_intersection = gen_resilient_set.intersection(truth_resilient_set)
        if not resilient_intersection: print("  -> INTERSECTION IS EMPTY.")
        for item in sorted(list(resilient_intersection), key=str)[:20]: print(f"  -> {item}")
        print("-"*70)

    # --- 4. Semantic Evaluation (LLM Judge) ---
    high_confidence_tp = 0; plausible_tp = 0
    unmatched_truth_set = truth_resilient_set.copy()
    for gen_triplet in tqdm(gen_resilient_set, desc="Running Semantic Evaluation", leave=False):
        gen_nodes, gen_pred = gen_triplet
        best_match_found = 'No Match'
        matched_truth_triplet = None
        for truth_triplet in unmatched_truth_set:
            truth_nodes, truth_pred = truth_triplet
            if gen_nodes == truth_nodes:
                judgment = ask_llm_judge_categorical(gen_pred, truth_pred)
                if judgment == 'High Confidence':
                    best_match_found = 'High Confidence'
                    matched_truth_triplet = truth_triplet
                    break
                elif judgment == 'Plausible':
                    best_match_found = 'Plausible'
                    matched_truth_triplet = truth_triplet
        if matched_truth_triplet:
            if best_match_found == 'High Confidence': high_confidence_tp += 1
            plausible_tp += 1
            unmatched_truth_set.remove(matched_truth_triplet)
            
    # --- 5. Calculate all metrics ---
    results = {
        "Strict Triplets": _calculate_metrics_from_counts(strict_triplet_tp, num_gen, num_truth),
        "Resilient Triplets": _calculate_metrics_from_counts(resilient_triplet_tp, num_gen, num_truth),
        "Semantic (High Confidence)": _calculate_metrics_from_counts(high_confidence_tp, num_gen, num_truth),
        "Semantic (Plausible)": _calculate_metrics_from_counts(plausible_tp, num_gen, num_truth)}
    return results

# --- 5. Data Preparation (One-Time Setup) ---

def get_or_build_freebase_dataframe():
    """
    Loads the full Freebase DataFrame from a fast Parquet cache. If the cache
    doesn't exist, it builds the DataFrame from the raw source data, performs
    critical cleaning by filtering out junk entities and most metadata predicates,
    and then creates the cache for future runs.
    It intentionally leaves 'ns/type.object.name' for the mapping step.
    """
    freebase_parquet_path = Path('full_freebase_df_filtered.parquet') 
    
    if freebase_parquet_path.exists():
        print(f"Loading filtered Freebase DataFrame from cache '{freebase_parquet_path}'...")
        full_freebase_df = pd.read_parquet(freebase_parquet_path)
    else:
        print("Filtered Freebase cache not found. Building DataFrame from raw data (this may take over 10 minutes)...\n")
        # Load raw data
        freebase_graphs_generator = io_tools.graphs_from_file("data/freebase/max1024/whole.gz")
        freebase_graphs = [paired_dataset.Graph.from_edges(g.edges) for g in list(freebase_graphs_generator)]
        df_list = []
        for g in tqdm(freebase_graphs, desc="Processing Freebase graphs"):
            df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
            node_map = {i: node_val for i, node_val in enumerate(g.nodes())}
            df["subject"] = df["src"].map(node_map)
            df["object"] = df["tgt"].map(node_map)
            df_list.append(df[["subject", "edge", "object"]])
        
        unfiltered_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
        unfiltered_df.rename(columns={"edge": "predicate"}, inplace=True)

        # --- Integrated Filtering Logic ---
        
        print(f"\nOriginal size of Freebase DataFrame: {len(unfiltered_df)}")

        # 1. Identify Junk Entity IDs using the full, unfiltered data
        object_name_df = unfiltered_df[unfiltered_df["predicate"] == "ns/type.object.name"]
        names = object_name_df['object'].str.strip('"')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        junk_names_df = object_name_df[
            (names.str.len() <= 2) | 
            (names.str.lower().isin(stop_words))
        ]
        junk_entity_ids = set(junk_names_df['subject'].unique())
        print(f"Identified {len(junk_entity_ids)} junk entity IDs to be removed.")
    
        # 2. Filter out triplets containing junk entities
        full_freebase_df = unfiltered_df[
            ~unfiltered_df['subject'].isin(junk_entity_ids) &
            ~unfiltered_df['object'].isin(junk_entity_ids)
        ]
        print(f"Filtered Junk Entities. Removed {len(unfiltered_df) - len(full_freebase_df)} triplets.")

        # 3. Filter out all blacklisted predicates EXCEPT 'ns/type.object.name'
        blacklist = [p for p in METADATA_PREDICATE_BLACKLIST if p != 'ns/type.object.name']
        original_len = len(full_freebase_df)
        full_freebase_df = full_freebase_df[~full_freebase_df['predicate'].isin(blacklist)]
        print(f"Filtered Metadata Predicates. Removed {original_len - len(full_freebase_df)} triplets. Final size: {len(full_freebase_df)}")
    
        print(f"Saving filtered DataFrame to Parquet cache at '{freebase_parquet_path}'...")
        full_freebase_df.to_parquet(freebase_parquet_path)
    
    print("Filtered Freebase DataFrame loaded (still contains 'ns/type.object.name').")
    return full_freebase_df

def load_or_build_id_to_name_map(full_freebase_df):
    """
    Loads the ID-to-Name map from a JSON cache. If the cache doesn't exist,
    it builds the map from the provided full_freebase_df.
    """
    id_map_path = Path('id_to_name_map.json')
    
    if id_map_path.exists():
        print(f"Loading existing ID-to-Name map from '{id_map_path}'...")
        with open(id_map_path, 'r') as f:
            id_to_name = json.load(f)
    else:
        print("ID-to-Name map not found. Building from the provided Freebase DataFrame...")
        # Use the DataFrame that was already loaded, avoiding a second file read
        object_name_df = full_freebase_df[full_freebase_df["predicate"] == "ns/type.object.name"].drop_duplicates()
        id_to_name = {row['subject']: str(row['object']) for _, row in object_name_df.iterrows()}
        
        print(f"Saving ID-to-Name map with {len(id_to_name)} entries to '{id_map_path}'...")
        with open(id_map_path, 'w') as f:
            json.dump(id_to_name, f)
            
    print(f"ID-to-Name map loaded with {len(id_to_name)} entries.")
    return id_to_name

# --- 6. Main Experiment Loop with Checkpointing ---

def run_batch_experiment(nlp, id_to_name, full_freebase_df, name_to_id_map, relations_lookup, name_db, predicate_db, update_distant_supervision=UPDATE_DISTANT_SUPERVISION_ONLY_IN_CHECKPOINTS):
    parsed_pairs = load_wikigraphs_data(WIKIGRAPHS_DATA_DIR)
    num_samples = SAMPLES_TO_RUN if SAMPLES_TO_RUN is not None else len(parsed_pairs)
    
    all_raw_graphs = []
    all_final_graphs = []

    for i in tqdm(range(num_samples), desc="Processing All Samples"):
        sample_pair = parsed_pairs[i]
        sample_title = "".join(c for c in sample_pair.title if c.isalnum() or c in (' ', '_')).rstrip()
        checkpoint_path = Path(CHECKPOINT_DIR) / f"sample_{i}_{sample_title}.json"

        if checkpoint_path.exists():
            print(f"\n--- Sample {i}: '{sample_pair.title}' --- Checkpoint found, loading results...")
            with open(checkpoint_path, 'r') as f:
                raw_graphs_dict = json.load(f)
            
            # Load the results we want to keep from the existing checkpoint
            raw_graph_default_df = pd.DataFrame(raw_graphs_dict['llm_vectordb_cr']['data'], columns=raw_graphs_dict['llm_vectordb_cr']['columns'])
            raw_graph_user_df = pd.DataFrame(raw_graphs_dict['llm_inherent_cr']['data'], columns=raw_graphs_dict['llm_inherent_cr']['columns'])
            raw_graph_pattern_based_df = pd.DataFrame(raw_graphs_dict['pattern_based']['data'], columns=raw_graphs_dict['pattern_based']['columns'])

            if update_distant_supervision:
                # Re-run only the Distant Supervision baseline to get its new results
                print("Re-running improved Distant Supervision baseline to update checkpoint...")
                original_text = sample_pair.text
                sample_text = original_text.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
                
                graph_distant_supervision_df = generate_graph_with_distant_supervision(sample_text, name_to_id_map, relations_lookup, id_to_name_map, nlp)
                
                # Overwrite the 'distant_supervision' key in our loaded dictionary
                raw_graphs_dict['distant_supervision'] = graph_distant_supervision_df.to_dict(orient='split')
                
                # Save the entire updated dictionary back to the checkpoint file
                with open(checkpoint_path, 'w') as f:
                    json.dump(raw_graphs_dict, f, indent=4)
                print(f"Updated checkpoint file at {checkpoint_path}")
            else:
                graph_distant_supervision_df = pd.DataFrame(raw_graphs_dict['distant_supervision']['data'],
                                                            columns=raw_graphs_dict['distant_supervision']['columns'])

        else:
            print(f"\n--- Sample {i}: '{sample_pair.title}' --- No checkpoint, running generation...")
            original_text = sample_pair.text
            sample_text = original_text.replace(' @-@ ', '-')
            
            raw_graph_default_df = generate_graph_from_text(sample_text, LLM_PROMPT_VECTORDB_FOR_CR)
            raw_graph_user_df = generate_graph_from_text(sample_text, LLM_PROMPT_INHERENT_CR)
            raw_graph_pattern_based_df = generate_graph_with_pattern_matching(sample_text, nlp)

            graph_distant_supervision_df = generate_graph_with_distant_supervision(sample_text, name_to_id_map, relations_lookup, id_to_name_map, nlp)
            
            checkpoint_data = {
                'llm_vectordb_cr': raw_graph_default_df.to_dict(orient='split'),
                'llm_inherent_cr': raw_graph_user_df.to_dict(orient='split'),
                'pattern_based': raw_graph_pattern_based_df.to_dict(orient='split'),
                'distant_supervision': graph_distant_supervision_df.to_dict(orient='split'), # Save the readable version
            }
            with open(checkpoint_path, 'w') as f: json.dump(checkpoint_data, f, indent=4)
            print(f"Saved checkpoint to {checkpoint_path}")

        all_raw_graphs.append({
            "LLM Prompt with VectorDB for CR": raw_graph_default_df,
            "LLM Prompt with Inherent CR": raw_graph_user_df
        })
        
        ground_truth_df_original = get_ground_truth_graph(sample_pair)
        ground_truth_df_readable = create_human_readable_ground_truth(ground_truth_df_original, id_to_name)
        
        fully_std_graph_default_df = orchestrate_standardization(
            raw_graph_default_df, name_db, predicate_db, strategy=LINKING_STRATEGY, batch_size=BATCH_RERANKING_SIZE
        )
        fully_std_graph_user_df = orchestrate_standardization(
            raw_graph_user_df, name_db, predicate_db, strategy=LINKING_STRATEGY, batch_size=BATCH_RERANKING_SIZE
        )
        fully_std_graph_pattern_based_df = orchestrate_standardization(
            raw_graph_pattern_based_df, name_db, predicate_db, strategy=LINKING_STRATEGY, batch_size=BATCH_RERANKING_SIZE
        )

        # The Distant Supervision dataframe does not need standardization because it is already standardized
        
        all_final_graphs.append({
            "LLM Prompt with VectorDB for CR": fully_std_graph_default_df,
            "LLM Prompt with Inherent CR": fully_std_graph_user_df,
            "Baseline (SVO + Embeddings)": fully_std_graph_pattern_based_df,
            "Baseline (Distant Supervision)": graph_distant_supervision_df,
            "ground_truth": ground_truth_df_readable
        })
        
    return all_raw_graphs, all_final_graphs

def load_all_dependencies(name_db_path=NAME_DB_PATH, predicate_db_path=PREDICATE_DB_PATH):
    """
    Loads all heavy, one-time dependencies: spaCy model, ID maps, the full Freebase KG,
    the relations_lookup dictionary and the name & predicate vector DBs.
    """
    print("="*30 + " LOADING DEPENDENCIES " + "="*30)
    
    print("Loading spaCy model 'en_core_web_lg'...")
    try:
        nlp = spacy.load("en_core_web_lg")
        print("spaCy model loaded successfully.")
    except OSError:
        print("ERROR: spaCy model not found. Please run: python -m spacy download en_core_web_lg")
        exit()

    full_freebase_df = get_or_build_freebase_dataframe()
    
    id_to_name = load_or_build_id_to_name_map(full_freebase_df)
    
    print("Loading full Freebase DataFrame for Distant Supervision...")

    print("Inverting ID-to-Name map for Distant Supervision...")
    name_to_id_map = {}
    for _id, name in id_to_name.items():
        name_to_id_map.setdefault(name, []).append(_id)

    # --- *** Build the Relation Lookup Dictionary ONCE *** ---
    print("Pre-building the relations lookup dictionary for DS (one-time operation)...")
    relations_lookup = {}
    # We store the full triplet to preserve directionality
    for triplet in tqdm(full_freebase_df.itertuples(index=False), total=len(full_freebase_df), desc="Building relation lookup"):
        key = tuple(sorted((triplet.subject, triplet.object)))
        if key not in relations_lookup:
            relations_lookup[key] = set()
        relations_lookup[key].add(triplet) # Add the full named tuple
    print("Relation lookup built.")

    # --- Build Name and Predicate Vector Databases from Full Freebase KG *** ---
    print("\nInitializing Name and Predicate Vector Databases from full Freebase vocabulary...")
    
    # 1. Get all unique entity names from the entire KG
    name_vocab = list(name_to_id_map.keys())
    
    # 2. Get all unique predicates from the entire KG apart from "ns/type.object.name"
    predicate_vocab = list(full_freebase_df['predicate'].unique())
    predicate_vocab = [p for p in predicate_vocab if p != 'ns/type.object.name']

    # 3. Initialize the persistent Vector Databases
    if os.path.exists(name_db_path):
        print(f"Loading existing name_db from '{name_db_path}'...")
        # If it exists, initialize without texts to prevent re-building
        name_db = VectorDatabase(collection_name="name_embeddings", texts=None, path=name_db_path)
    else:
        print(f"Building new name_db with {len(name_vocab)} entities at '{name_db_path}'...")
        # If it doesn't exist, initialize with texts to build it
        name_db = VectorDatabase(collection_name="name_embeddings", texts=name_vocab, path=name_db_path)

    if os.path.exists(predicate_db_path):
        print(f"Loading existing predicate_db from '{predicate_db_path}'...")
        predicate_db = VectorDatabase(collection_name="predicate_embeddings", texts=None, path=predicate_db_path)
    else:
        print(f"Building new predicate_db with {len(predicate_vocab)} predicates at '{predicate_db_path}'...")
        predicate_db = VectorDatabase(collection_name="predicate_embeddings", texts=predicate_vocab, path=predicate_db_path)
    
    print("Name and Predicate Vector Databases are ready.")
    
    print("="*30 + " DEPENDENCIES LOADED " + "="*30 + "\n")
    return nlp, id_to_name, full_freebase_df, name_to_id_map, relations_lookup, name_db, predicate_db

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    
    start_time = datetime.now()

    nlp_model, id_to_name_map, full_freebase_df, name_to_id_map, relations_lookup, name_db, predicate_db = load_all_dependencies()
    
    all_raw_graphs, all_final_graphs = run_batch_experiment(
        nlp=nlp_model, 
        id_to_name=id_to_name_map, 
        full_freebase_df=full_freebase_df, 
        name_to_id_map=name_to_id_map,
        relations_lookup=relations_lookup,
        name_db=name_db,
        predicate_db=predicate_db
    )

    end_time = datetime.now()
    duration_delta = end_time - start_time
    duration_minutes = int(duration_delta.total_seconds() / 60)
    
    # --- Aggregate and Report Final Results ---
    
    quant_results = {name: [] for name in all_final_graphs[0].keys() if name != 'ground_truth'}
    for sample in all_final_graphs:
        for name, df in sample.items():
            if name == 'ground_truth': continue
            quant_results[name].append(len(df))
    avg_triplets = {name: pd.Series(counts).mean() for name, counts in quant_results.items()}

    coref_results = {'LLM Prompt with VectorDB for CR': [], 'LLM Prompt with Inherent CR': []}
    for sample in all_raw_graphs:
        if not sample['LLM Prompt with VectorDB for CR'].empty:
            coref_results['LLM Prompt with VectorDB for CR'].append(len(pd.concat([sample['LLM Prompt with VectorDB for CR']['node_1'], sample['LLM Prompt with VectorDB for CR']['node_2']]).unique()))
        if not sample['LLM Prompt with Inherent CR'].empty:
            coref_results['LLM Prompt with Inherent CR'].append(len(pd.concat([sample['LLM Prompt with Inherent CR']['node_1'], sample['LLM Prompt with Inherent CR']['node_2']]).unique()))
    avg_coref = {name: pd.Series(counts).mean() for name, counts in coref_results.items()}

    print("\n" + "="*30 + " FINAL AGGREGATED RESULTS " + "="*30)
    print(f"\n### 7.1 Aggregated Quantitative and Coreference Analysis (Based on {len(all_final_graphs)} samples) ###\n")
    print("Average Triplets Generated:", avg_triplets)
    print("Average Unique Entities (Coreference):", avg_coref)

    final_metrics = {name: [] for name in all_final_graphs[0].keys() if name != 'ground_truth'}
    llm_judge_cache.clear()
    for sample in tqdm(all_final_graphs, desc="Aggregating Final Evaluations"):
        gt = sample['ground_truth']
        for name, df in sample.items():
            if name == 'ground_truth': continue
            
            strict_pairs = evaluate_pairs(df, gt, resilient=False, debug=False, debug_name=name)
            resilient_pairs = evaluate_pairs(df, gt, resilient=True, debug=False, debug_name=name)
            
            if "Distant Supervision" in name:
                triplet_metrics = run_comprehensive_evaluation(df, gt, debug=False, debug_name=name)
            else:
                triplet_metrics = run_comprehensive_evaluation(df, gt)
            
            final_metrics[name].append({
                "Strict Pairs": strict_pairs,
                "Resilient Pairs": resilient_pairs,
                **triplet_metrics
            })

    aggregated_results = {}
    for name, results_list in final_metrics.items():
        metric_types = results_list[0].keys()
        temp_agg = {}
        for mtype in metric_types:
            df = pd.DataFrame([res[mtype] for res in results_list])
            for col in df.columns:
                temp_agg[f"{col} ({mtype})"] = df[col].mean()
        aggregated_results[name] = temp_agg

    final_df = pd.DataFrame.from_dict(aggregated_results, orient='index')
    
    final_df_percent = final_df * 100
    
    def format_as_percentage(val):
        return f"{val:.2f}%"

    print("\n\n### 7.2 Aggregated Performance Metrics vs. Ground Truth ###\n")
    
    pair_cols = [col for col in final_df.columns if 'Pairs' in col]
    triplet_cols = [col for col in final_df.columns if 'Triplets' in col]
    semantic_cols = [col for col in final_df.columns if 'Semantic' in col and 'Pairs' not in col]

    print("\n--- Performance: Connected Node Pairs (Mean) ---")
    if pair_cols:
        print(final_df_percent[pair_cols].to_string(formatters={col: format_as_percentage for col in pair_cols}))
    
    print("\n\n--- Performance: Strict/Resilient Triplets (Mean) ---")
    if triplet_cols:
        print(final_df_percent[triplet_cols].to_string(formatters={col: format_as_percentage for col in triplet_cols}))
    
    print("\n\n--- Performance: Semantic Evaluations (Mean) ---")
    if semantic_cols:
        print(final_df_percent[semantic_cols].to_string(formatters={col: format_as_percentage for col in semantic_cols}))

    start_time_str = start_time.strftime('%Y%m%d-%H.%M')
    num_texts = len(all_final_graphs)
    output_filename = f"llmgrapher_experiment_filtered_vectordb-llm_linking-triplets_{num_texts}_{start_time_str}_{duration_minutes}min.csv"
    
    final_df.to_csv(output_filename)
    print(f"\n\nFull aggregated results saved to '{output_filename}'")
