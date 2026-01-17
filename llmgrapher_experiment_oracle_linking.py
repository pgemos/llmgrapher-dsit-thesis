# ==============================================================================
# SCRIPT DESCRIPTION
# ==============================================================================
# This script performs the "Oracle Linking" (Gold Standard) evaluation.
#
# Key Characteristics:
# 1. Oracle Scope: The standardization step is restricted strictly to the vocabulary 
#    (entities and predicates) found in the Ground Truth of the current text.
# 2. Purpose: To establish an upper-bound performance benchmark by eliminating 
#    retrieval errors caused by the massive size of the full Freebase KG.
# 3. Implementation: Re-initializes the VectorDBs for every sample using only 
#    the target vocabulary, ensuring the best possible match is theoretically retrievable.
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

# --- 1. Configuration ---
SAMPLES_TO_RUN = 200
LLM_MODEL = "llama3.1:8b"
WIKIGRAPHS_DATA_DIR = "data/wikigraphs/"
CHECKPOINT_DIR = "checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
UPDATE_DISTANT_SUPERVISION_ONLY_IN_CHECKPOINTS = False

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

# --- 4. Helper and Generation Functions ---

def load_wikigraphs_data(data_root, subset='train', version='max256'):
    print("Loading WikiGraphs dataset...")
    paired_dataset.DATA_ROOT = data_root
    dataset = paired_dataset.ParsedDataset(
        subset=subset, shuffle_data=False, data_dir=None, version=version
    )
    return list(dataset)

def get_ground_truth_graph(pair):
    g = pair.graph
    df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
    df["subject"] = df["src"].apply(lambda node_id: g.nodes()[node_id])
    df["object"] = df["tgt"].apply(lambda node_id: g.nodes()[node_id])
    df = df[["subject", "edge", "object"]]
    df.rename(columns={"edge": "predicate"}, inplace=True)
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

def generate_graph_with_distant_supervision(text, name_to_id_map, relations_lookup, id_to_name_map, nlp, debug_sentence_limit=3):
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
    def __init__(self, collection_name, texts, path=None, similarity="cosine"):
        import chromadb
        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": similarity})
        if self.collection.count() == 0 and texts:
            ids = [f"id{i}" for i in range(len(texts))]
            if ids: self.collection.add(documents=texts, ids=ids)

    def query(self, query_text, n_results):
        if self.collection.count() == 0: return None
        results = self.collection.query(query_texts=[str(query_text)], n_results=n_results)
        return results['documents'][0] if results and results.get('documents') and results['documents'][0] else None

def standardize_graph_entities(raw_graph_df, vector_db, truth_entities):
    if raw_graph_df.empty: return raw_graph_df
    generated_entities = pd.concat([raw_graph_df['node_1'], raw_graph_df['node_2']]).unique()
    name_mapping = {}
    for entity in tqdm(generated_entities, desc="Linking entities", leave=False):
        best_match_list = vector_db.query(query_text=str(entity), n_results=1)
        if best_match_list and isinstance(best_match_list, list):
            name_mapping[entity] = best_match_list[0]
        else:
            name_mapping[entity] = entity
    std_graph_df = raw_graph_df.copy()
    std_graph_df['node_1'] = std_graph_df['node_1'].map(name_mapping)
    std_graph_df['node_2'] = std_graph_df['node_2'].map(name_mapping)
    return std_graph_df.drop_duplicates().reset_index(drop=True)

def standardize_graph_predicates(entity_std_graph_df, vector_db, truth_predicates_vocab):
    if entity_std_graph_df.empty: return entity_std_graph_df
    generated_edges = entity_std_graph_df['edge'].unique()
    predicate_mapping = {}
    for edge in tqdm(generated_edges, desc="Linking predicates", leave=False):
        best_match_list = vector_db.query(query_text=str(edge), n_results=1)
        if best_match_list and isinstance(best_match_list, list):
            predicate_mapping[edge] = best_match_list[0]
        else:
            predicate_mapping[edge] = None
    fully_std_graph_df = entity_std_graph_df.copy()
    fully_std_graph_df['edge'] = fully_std_graph_df['edge'].map(predicate_mapping)
    fully_std_graph_df.dropna(inplace=True)
    return fully_std_graph_df.drop_duplicates().reset_index(drop=True)

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

def get_or_build_freebase_dataframe(filter_two_letter_names=True):
    """
    Loads the full Freebase DataFrame from a fast Parquet cache. If the cache
    doesn't exist, it builds the DataFrame from the raw source data and creates
    the cache for future runs. This is the single source for the expensive I/O.
    """
    freebase_parquet_path = Path('full_freebase_df.parquet')
    
    if freebase_parquet_path.exists():
        print(f"Loading full Freebase DataFrame from cache '{freebase_parquet_path}'...")
        full_freebase_df = pd.read_parquet(freebase_parquet_path)
    else:
        print("Freebase cache not found. Building DataFrame from raw data (this may take over 10 minutes)...\n")
        freebase_graphs_generator = io_tools.graphs_from_file("data/freebase/max1024/whole.gz")
        freebase_graphs = [paired_dataset.Graph.from_edges(g.edges) for g in list(freebase_graphs_generator)]
        df_list = []
        for g in tqdm(freebase_graphs, desc="Processing Freebase graphs"):
            df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
            node_map = {i: node_val for i, node_val in enumerate(g.nodes())}
            df["subject"] = df["src"].map(node_map)
            df["object"] = df["tgt"].map(node_map)
            df_list.append(df[["subject", "edge", "object"]])
        
        full_freebase_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
        full_freebase_df.rename(columns={"edge": "predicate"}, inplace=True)

        if filter_two_letter_names:
            # --- *** CRITICAL FIX: Filter Junk Entities from the Knowledge Base *** ---
            print(f"Original size of Freebase DataFrame: {len(full_freebase_df)}")
            
            # 1. Get all object names that are very short or are common English stop words
            object_name_df = full_freebase_df[full_freebase_df["predicate"] == "ns/type.object.name"]
            # Strip quotes for length check
            names = object_name_df['object'].str.strip('"')
            # Define junk names: less than 3 chars OR a common stop word (e.g., 'the', 'a', 'is')
            stop_words = spacy.lang.en.stop_words.STOP_WORDS
            junk_names = object_name_df[
                (names.str.len() <= 2) | 
                (names.str.lower().isin(stop_words))
            ]
            
            # 2. Get the IDs of these junk entities
            junk_entity_ids = set(junk_names['subject'].unique())
            print(f"Identified {len(junk_entity_ids)} junk entity IDs to be removed.")
        
            # 3. Filter the entire Freebase DataFrame to remove any triplet involving a junk ID
            original_len = len(full_freebase_df)
            full_freebase_df = full_freebase_df[
                ~full_freebase_df['subject'].isin(junk_entity_ids) &
                ~full_freebase_df['object'].isin(junk_entity_ids)
            ]
            print(f"Filtered Freebase DataFrame. Removed {original_len - len(full_freebase_df)} triplets. New size: {len(full_freebase_df)}")
    
        # --- Build the Cleaned ID-to-Name and Name-to-ID Maps ---
        print("Building cleaned ID-to-Name and Name-to-ID maps...")
        object_name_df_clean = full_freebase_df[full_freebase_df["predicate"] == "ns/type.object.name"]
        id_to_name = {row['subject']: str(row['object']) for _, row in object_name_df_clean.iterrows()}
        
        name_to_id_map = {}
        for _id, name in id_to_name.items():
            name_to_id_map.setdefault(name, []).append(_id)
        print(f"Cleaned maps built. Total unique entity names: {len(name_to_id_map)}")
        
        print(f"Saving full DataFrame to Parquet cache at '{freebase_parquet_path}'...")
        full_freebase_df.to_parquet(freebase_parquet_path)
    
    print("Full Freebase DataFrame loaded.")
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

def run_batch_experiment(nlp, id_to_name, full_freebase_df, name_to_id_map, relations_lookup, update_distant_supervision=UPDATE_DISTANT_SUPERVISION_ONLY_IN_CHECKPOINTS):
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
                sample_text = original_text.replace(' @-@ ', '-')
                # The new DS function needs the full_freebase_df and name_to_id_map
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
        
        truth_entities_vocab = list(set(pd.concat([ground_truth_df_readable['subject'], ground_truth_df_readable['object']]).unique()))
        truth_predicates_vocab = list(ground_truth_df_readable['predicate'].unique())
        
        name_db = VectorDatabase(collection_name=f"name_db_for_sample_{i}", texts=truth_entities_vocab)
        predicate_db = VectorDatabase(collection_name=f"predicate_db_for_sample_{i}", texts=truth_predicates_vocab)

        # HERE
        std_entity_graph_default_df = standardize_graph_entities(raw_graph_default_df, name_db, truth_entities_vocab)
        fully_std_graph_default_df = standardize_graph_predicates(std_entity_graph_default_df, predicate_db, truth_predicates_vocab)
        
        std_entity_graph_user_df = standardize_graph_entities(raw_graph_user_df, name_db, truth_entities_vocab)
        fully_std_graph_user_df = standardize_graph_predicates(std_entity_graph_user_df, predicate_db, truth_predicates_vocab)
        
        std_entity_graph_pattern_based_df = standardize_graph_entities(raw_graph_pattern_based_df, name_db, truth_entities_vocab)
        fully_std_graph_pattern_based_df = standardize_graph_predicates(std_entity_graph_pattern_based_df, predicate_db, truth_predicates_vocab)

        # The Distant Supervision dataframe does not need standardization because it is already standardized
        
        all_final_graphs.append({
            "LLM Prompt with VectorDB for CR": fully_std_graph_default_df,
            "LLM Prompt with Inherent CR": fully_std_graph_user_df,
            "Baseline (SVO + Embeddings)": fully_std_graph_pattern_based_df,
            "Baseline (Distant Supervision)": graph_distant_supervision_df,
            "ground_truth": ground_truth_df_readable
        })
        
    return all_raw_graphs, all_final_graphs

def load_all_dependencies():
    """
    Loads all heavy, one-time dependencies: spaCy model, ID maps, the full Freebase KG
    and the relations_lookup dictionary.
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
    
    print("="*30 + " DEPENDENCIES LOADED " + "="*30 + "\n")
    return nlp, id_to_name, full_freebase_df, name_to_id_map, relations_lookup

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    
    start_time = datetime.now()

    nlp_model, id_to_name_map, full_freebase_df, name_to_id_map, relations_lookup = load_all_dependencies()
    
    all_raw_graphs, all_final_graphs = run_batch_experiment(
        nlp=nlp_model, 
        id_to_name=id_to_name_map, 
        full_freebase_df=full_freebase_df, 
        name_to_id_map=name_to_id_map,
        relations_lookup=relations_lookup
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
    output_filename = f"llmgrapher_experiment_oracle_linking_{num_texts}_{start_time_str}_{duration_minutes}min.csv"
    
    final_df.to_csv(output_filename)
    print(f"\n\nFull aggregated results saved to '{output_filename}'")