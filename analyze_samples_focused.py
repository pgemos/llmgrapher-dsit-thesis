# ==============================================================================
# FOCUSED SAMPLE ANALYSIS & LLM-BASED PRUNING SCRIPT (V3)
# ==============================================================================
# This script performs a multi-stage, in-depth analysis on a small text snippet
# from selected samples.
#
# Pipeline:
# 1. Extracts the first N paragraphs from the selected samples.
# 2. Generates a raw graph and saves it to a 'checkpoints_focused' directory.
# 3. Standardizes the raw graph.
# 4. Generates a full qualitative PDF report containing:
#    a. The raw, unstandardized graph (highlighted).
#    b. The standardized, unfiltered graph (highlighted).
#    c. Performance metrics for the unfiltered graph.
#    d. A pruning analysis detailing removed triplets.
#    e. **NEW**: Performance metrics for the final pruned graph.
# 5. Prints a final before-and-after comparison to the console.
# ==============================================================================

# --- 0. Installation Notes ---
# pip install reportlab pandas tqdm spacy chromadb ollama
# Ensure you have the spaCy model: python -m spacy download en_core_web_lg

import os
import json    
import spacy
import pandas as pd
import re
from tqdm.auto import tqdm
from pathlib import Path
import ollama

# PDF Generation Library
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor

from wikigraphs.data import paired_dataset, io_tools
import analyze_samples as ansa

# --- 1. Configuration ---
SAMPLES_TO_ANALYZE = list(range(100))  # <-- EDIT THIS LIST of sample indexes
MIN_NODE_DEGREE_FOR_FILTERING = 3 # The degree for an entity to be considered for pruning
NUMBER_OF_PARAGRAPHS = 2

# Directory Paths
WIKIGRAPHS_DATA_DIR = "data/wikigraphs/"
FREEBASE_DATA_PATH = "data/freebase/max1024/whole.gz" # Path to the Freebase data
CHECKPOINT_DIR = "checkpoints_focused"
ANALYSIS_OUTPUT_DIR = "analysis_reports_focused"
ansa.ANALYSIS_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR # Override the imported module's setting
NAME_DB_PATH = "name_embeddings"
PREDICATE_DB_PATH = "predicate_embeddings"

Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
Path(ANALYSIS_OUTPUT_DIR).mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = "llama3.1:8b"

# Color Scheme
GEN_COLOR_LIGHT, GEN_COLOR_NORMAL, GEN_COLOR_DARK = HexColor('#FFFACD'), HexColor('#FFEE75'), HexColor('#FFD700')
TRUTH_COLOR_LIGHT, TRUTH_COLOR_NORMAL, TRUTH_COLOR_DARK = HexColor('#E0FFFF'), HexColor('#ADD8E6'), HexColor('#4682B4')
BOTH_COLOR_LIGHT, BOTH_COLOR_NORMAL, BOTH_COLOR_DARK = HexColor('#DFF0D8'), HexColor('#90EE90'), HexColor('#228B22')

# Sample Files Naming
SAMPLE_FILENAME_ZERO_PADDING_WIDTH = len(str(len(SAMPLES_TO_ANALYZE)))

# ==============================================================================
# SECTION 2: PROMPT ENGINEERING
# ==============================================================================

LLM_PREDICATE_FILTER_PROMPT = """You are a Knowledge Graph specialist with expert domain knowledge. Your task is to refine a generated knowledge graph by comparing its extracted relations to a trusted, external knowledge base (Freebase).

You will be given:
1.  **Entity**: The central entity being analyzed.
2.  **Generated Edges**: A numbered list of relationships that were extracted from a short text about this entity.
3.  **Known Freebase Predicates**: A list of trusted, canonical predicates known to be associated with this entity in a large knowledge base.

Your Task:
Analyze each "Generated Edge". If its meaning is semantically equivalent, a direct subset, or a very close hyponym of any of the "Known Freebase Predicates", you should approve it. Discard any generated edges that are too vague, factually incorrect, or cannot be mapped to the known predicates.

Your response MUST be a single, valid JSON list containing only the NUMBERS of the "Generated Edges" that you approve and NOTHING else.

Example:
- Entity: "Sega"
- Generated Edges:
  1. "is a developer of video games"
  2. "is headquartered in Japan"
  3. "has a famous logo"
- Known Freebase Predicates:
  - "ns/organization.organization.headquarters"
  - "ns/business.business_operation.industry"
  - "ns/common.topic.official_website"

Correct Response:
[1, 2]
"""

LLM_PROMPT_INHERENT_CR = """You are a knowledge graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```). Your task is to extract the ontology 
of terms mentioned in the given context and the relationships between them.


Thought 1: First, read through the text to identify the core entities. These are the main people, organizations, locations, and concepts being discussed.

Thought 2: Pay close attention to coreferences. If an entity is mentioned multiple times with different names (e.g., 'Valkyria Chronicles III', 'the game', 'it'), I must identify the most complete and descriptive name (e.g., 'Valkyria Chronicles III') and use it consistently for all triplets involving that entity.

Thought 3: I must also resolve pronouns. If the text says 'it was developed by Sega', the pronoun 'it' must be resolved to the specific entity it refers to from the preceding text. The final triplet should not contain pronouns like 'he', 'she', or 'it'.

Thought 4: For each sentence, I will extract relationships as `(node_1, edge, node_2)` triplets. The `edge` should be a concise, verb-oriented phrase describing the relationship.

Format your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the follwing:

[
   {
       "node_1": "Canonical Entity Name",
       "node_2": "Another Canonical Entity Name",
       "edge": "A concise relationship phrase"
   }, {...}
]

Respond ONLY with a well-formed JSON list. Do not include any introductory text, comments, or explanations in your response."""

GENERATION_PROMPT = LLM_PROMPT_INHERENT_CR

# ==============================================================================
# SECTION 3: HELPER FUNCTIONS
# ==============================================================================

# --- Text and Data Loading ---
def get_first_n_paragraphs(text, nlp, n=NUMBER_OF_PARAGRAPHS):
    """
    Extracts the first n paragraphs from a text string, skipping any paragraphs
    that contain only a single sentence.
    """
    paragraphs = text.split('\n')
    non_empty_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    multi_sentence_paragraphs = []
    for p in non_empty_paragraphs:
        # Use spaCy to process the paragraph and count sentences
        doc = nlp(p)
        if len(list(doc.sents)) > 1:
            multi_sentence_paragraphs.append(p)
    
    # Take the first n paragraphs from the *filtered* list
    return '\n\n'.join(multi_sentence_paragraphs[:n])

def load_wikigraphs_data(data_root, subset='train', version='max256'):
    paired_dataset.DATA_ROOT = data_root
    return list(paired_dataset.ParsedDataset(subset=subset, shuffle_data=False, data_dir=None, version=version))

def get_or_build_freebase_dataframe(freebase_path):
    freebase_parquet_path = Path('full_freebase_df_max1024.parquet')
    if freebase_parquet_path.exists():
        print(f"Loading Freebase DF from cache '{freebase_parquet_path}'...")
        return pd.read_parquet(freebase_parquet_path)
    print("Building Freebase DF from raw data (one-time operation)...")
    df_list = []
    for g in tqdm(list(io_tools.graphs_from_file(freebase_path)), desc="Processing Freebase graphs"):
        graph = paired_dataset.Graph.from_edges(g.edges)
        df = pd.DataFrame(graph.edges(), columns=["src", "tgt", "edge"])
        node_map = {i: node_val for i, node_val in enumerate(graph.nodes())}
        df["subject"], df["object"] = df["src"].map(node_map), df["tgt"].map(node_map)
        df_list.append(df[["subject", "edge", "object"]])
    full_df = pd.concat(df_list, ignore_index=True).drop_duplicates().rename(columns={"edge": "predicate"})
    full_df.to_parquet(freebase_parquet_path)
    return full_df

def get_ground_truth_graph(pair):
    g = pair.graph
    df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
    df["subject"], df["object"] = df["src"].apply(lambda n: g.nodes()[n]), df["tgt"].apply(lambda n: g.nodes()[n])
    df = df[["subject", "edge", "object"]].rename(columns={"edge": "predicate"})
    return df[~df['predicate'].isin(ansa.METADATA_PREDICATE_BLACKLIST)].drop_duplicates().reset_index(drop=True)

def create_human_readable_ground_truth(truth_df, id_to_name_map):
    readable_df = truth_df.copy()
    readable_df['subject'] = readable_df['subject'].apply(lambda x: id_to_name_map.get(x, x))
    readable_df['object'] = readable_df['object'].apply(lambda x: id_to_name_map.get(x, x))
    return readable_df

def load_or_build_id_to_name_map():
    id_map_path = Path('id_to_name_map.json')
    if not id_map_path.exists(): raise FileNotFoundError("id_to_name_map.json not found.")
    with open(id_map_path, 'r') as f: return json.load(f)

# --- Graph Generation, Standardization, and Evaluation ---
class VectorDatabase:
    def __init__(self, collection_name, path):
        import chromadb
        if not os.path.exists(path): raise FileNotFoundError(f"VectorDB path not found: '{path}'.")
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(name=collection_name)
    def query(self, query_text, n_results):
        if self.collection.count() == 0: return None
        results = self.collection.query(query_texts=[str(query_text)], n_results=n_results)
        return results['documents'][0] if results and results.get('documents') and results['documents'][0] else None

def fix_json_output(text):
    text = text.strip()
    # First, try to find a JSON list specifically wrapped in markdown code fences
    match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        # If found, return the content inside the fences (the first captured group)
        return match.group(1)

    # If not in a code fence, fall back to finding the first bracketed expression.
    # The non-greedy `.*?` is crucial here to stop at the first closing bracket.
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return match.group(0)
    
    # If no list is found at all, return the original text
    return text

def generate_graph_from_text(text, system_prompt):
    if len(text.strip()) < 20: return pd.DataFrame(columns=['node_1', 'node_2', 'edge'])
    user_prompt = f"context: ```{text}``` \n\n output: "
    try:
        response = ollama.generate(model=LLM_MODEL, system=system_prompt, prompt=user_prompt)
        triplets = json.loads(fix_json_output(response["response"]))
        if isinstance(triplets, list):
            return pd.DataFrame([t for t in triplets if all(k in t for k in ['node_1', 'node_2', 'edge'])]).drop_duplicates().reset_index(drop=True)
    except Exception as e: print(f"  [Warning] LLM generation failed: {e}")
    return pd.DataFrame(columns=['node_1', 'node_2', 'edge'])

def standardize_graph_items_simple(graph_df, vector_db, item_type='entity'):
    """MODIFIED: Now returns the mapping dictionary along with the DataFrame."""
    if graph_df.empty: return graph_df, {}
    columns = ['node_1', 'node_2'] if item_type == 'entity' else ['edge']
    items = pd.concat([graph_df[col] for col in columns if col in graph_df]).unique()
    mapping = {item: (vector_db.query(str(item), 1)[0] if vector_db.query(str(item), 1) else (item if item_type == 'entity' else None)) for item in items}
    std_df = graph_df.copy()
    for col in columns:
        if col in std_df: std_df[col] = std_df[col].map(mapping)
    if item_type == 'predicate' and 'edge' in std_df: std_df.dropna(subset=['edge'], inplace=True)
    return std_df.drop_duplicates().reset_index(drop=True), mapping

def orchestrate_standardization(raw_graph_df, name_db, predicate_db):
    """MODIFIED: Now returns a unified name map for highlighting raw triplets."""
    std_entity_df, entity_map = standardize_graph_items_simple(raw_graph_df, name_db, item_type='entity')
    fully_std_df, pred_map = standardize_graph_items_simple(std_entity_df, predicate_db, item_type='predicate')
    # Merge maps. In case of conflict, entity map takes precedence (less likely to be an issue)
    name_map = {**pred_map, **entity_map}
    return fully_std_df, name_map

def run_comprehensive_evaluation(generated_df, truth_df):
    return ansa.run_comprehensive_evaluation(generated_df, truth_df)

# ==============================================================================
# SECTION 4: CORE LOGIC FOR ANALYSIS AND PRUNING
# ==============================================================================

def find_high_degree_nodes(df, min_degree=3):
    if df.empty: return []
    node_counts = pd.concat([df['node_1'], df['node_2']]).value_counts()
    return node_counts[node_counts >= min_degree].index.tolist()

def find_freebase_predicates_for_nodes(nodes, freebase_df, name_to_id_map):
    node_predicates_map = {node: {"predicates": set()} for node in nodes}
    print(f"  Querying Freebase for predicates of {len(nodes)} high-degree nodes...")
    for node_name in tqdm(nodes, desc="  Finding predicates", leave=False):
        search_name = str(node_name).lower().strip()
        freebase_ids = name_to_id_map.get(search_name, [])
        if not freebase_ids: continue
        related_triplets = freebase_df[freebase_df['subject'].isin(freebase_ids) | freebase_df['object'].isin(freebase_ids)]
        node_predicates_map[node_name]["predicates"].update(set(related_triplets['predicate'].unique()))
    return node_predicates_map

def filter_graph_with_llm(graph_df, node_predicates_map):
    pruning_stats = {"total_removed": 0, "removed_by_entity": {}}
    high_degree_nodes = list(node_predicates_map.keys())
    final_triplets_df = graph_df[~graph_df['node_1'].isin(high_degree_nodes) & ~graph_df['node_2'].isin(high_degree_nodes)].copy()
    print(f"  Starting LLM-based pruning for {len(high_degree_nodes)} nodes...")
    for node, data in node_predicates_map.items():
        node_triplets = graph_df[graph_df['node_1'] == node]
        if node_triplets.empty: continue
        known_predicates = data['predicates']
        if not known_predicates:
            final_triplets_df = pd.concat([final_triplets_df, node_triplets], ignore_index=True)
            continue
        generated_edges = node_triplets['edge'].tolist()
        prompt = f"**Entity**: \"{node}\"\n\n**Generated Edges**:\n" + "\n".join([f"  {i+1}. \"{e}\"" for i, e in enumerate(generated_edges)]) + f"\n\n**Known Freebase Predicates**:\n" + "\n".join([f"  - \"{p}\"" for p in sorted(list(known_predicates))])
        try:
            response = ollama.generate(model=LLM_MODEL, system=LLM_PREDICATE_FILTER_PROMPT, prompt=prompt)
            approved_indices = {i - 1 for i in json.loads(fix_json_output(response['response'])) if isinstance(i, int) and 0 < i <= len(generated_edges)}
            approved_triplets = node_triplets.iloc[list(approved_indices)]
            final_triplets_df = pd.concat([final_triplets_df, approved_triplets], ignore_index=True)
            removed_indices = set(range(len(node_triplets))) - approved_indices
            if removed_indices:
                removed_triplets = node_triplets.iloc[list(removed_indices)]
                pruning_stats["removed_by_entity"][node] = [f"({r.node_1}, {r.edge}, {r.node_2})" for _, r in removed_triplets.iterrows()]
                pruning_stats["total_removed"] += len(removed_triplets)
            print(f"    - Node '{node}': Kept {len(approved_triplets)} of {len(node_triplets)} triplets.")
        except Exception as e:
            print(f"    - [Warning] LLM filtering for node '{node}' failed: {e}. Keeping all original triplets.")
            final_triplets_df = pd.concat([final_triplets_df, node_triplets], ignore_index=True)
    return final_triplets_df.drop_duplicates().reset_index(drop=True), pruning_stats

# ==============================================================================
# SECTION 5: FOCUSED PDF REPORT GENERATION (with Pruning Stats)
# ==============================================================================

def highlight_raw_triplet(triplet_row, name_map, item_to_color_map):
    """Highlights a raw triplet by mapping its components to their standardized colors."""
    s, p, o = triplet_row.node_1, triplet_row.edge, triplet_row.node_2
    # Find the standardized name, then find the color for that standardized name
    s_std, p_std, o_std = name_map.get(s, s), name_map.get(p, p), name_map.get(o, o)
    s_color, p_color, o_color = item_to_color_map.get(str(s_std).lower().strip()), item_to_color_map.get(str(p_std).lower().strip()), item_to_color_map.get(str(o_std).lower().strip())
    s_html = f'<font backColor="{s_color.hexval()}">{s}</font>' if s_color else str(s)
    p_html = f'<font backColor="{p_color.hexval()}">{p}</font>' if p_color else str(p)
    o_html = f'<font backColor="{o_color.hexval()}">{o}</font>' if o_color else str(o)
    return f"({s_html}, &nbsp; {p_html}, &nbsp; {o_html})"

def create_focused_pdf_report(sample_index, sample_title, method_name, unfiltered_metrics, pruned_metrics, match_counts, annotated_text, raw_df, gen_df, truth_df, item_to_color_map, name_map, pruning_stats=None):
    """Creates a comprehensive PDF report including raw graphs and pruning metrics."""
    sample_dir = Path(ANALYSIS_OUTPUT_DIR) / f"Sample_{sample_index:0{SAMPLE_FILENAME_ZERO_PADDING_WIDTH}d}_{sample_title}"
    sample_dir.mkdir(exist_ok=True, parents=True)
    filename = sample_dir / f"{sample_title}-Error_Analysis_Report-Focused.pdf"
    doc = SimpleDocTemplate(str(filename))
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(f"Analysis for Sample {sample_index}: {sample_title}", styles['h1']))
    story.append(Paragraph(f"Method: {method_name}", styles['h2']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Performance Metrics (Standardized - Unfiltered):", styles['h3']))
    for mtype, values in unfiltered_metrics.items():
        story.append(Paragraph(f"<b>{mtype}:</b> " + ", ".join([f"{k}: {v*100:.2f}%" for k, v in values.items()]), styles['Normal']))
    story.append(Spacer(1, 6))

    if pruned_metrics:
        story.append(Paragraph("Performance Metrics (Pruned Graph):", styles['h3']))
        for mtype, values in pruned_metrics.items():
            story.append(Paragraph(f"<b>{mtype}:</b> " + ", ".join([f"{k}: {v*100:.2f}%" for k, v in values.items()]), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detailed Match Statistics:", styles['h3']))
    for category, counts in match_counts.items(): story.append(Paragraph(f"<b>{category}:</b> In Both: {counts['both']}, Gen Only: {counts['gen_only']}, Truth Only: {counts['truth_only']}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    if pruning_stats and pruning_stats["total_removed"] > 0:
        story.append(Paragraph("Pruning Analysis:", styles['h3']))
        story.append(Paragraph(f"<b>Total Triplets Removed: {pruning_stats['total_removed']}</b>", styles['Normal']))
        no_indent_code_style = ParagraphStyle(name='CodeNoIndent', parent=styles['Code'], leftIndent=0)
        for entity, removed_list in pruning_stats["removed_by_entity"].items():
            story.append(Paragraph(f"<u>Removed for Entity: '{entity}'</u>", styles['Normal']))
            for triplet_str in removed_list: story.append(Paragraph(f"- {triplet_str}", no_indent_code_style))
        story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"Generated Triplets (Raw - Before Standardization): (Total: {len(raw_df)})", styles['h3']))
    if raw_df.empty: story.append(Paragraph("<i>None</i>", styles['Italic']))
    else:
        for row in raw_df.itertuples(index=False): story.append(Paragraph(highlight_raw_triplet(row, name_map, item_to_color_map), styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"Generated Triplets (Standardized - Unfiltered): (Total: {len(gen_df)})", styles['h3']))
    if gen_df.empty: story.append(Paragraph("<i>None</i>", styles['Italic']))
    else:
        for row in gen_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'}).itertuples(): story.append(Paragraph(ansa.highlight_triplet(row, item_to_color_map), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Ground Truth Triplets: (Total: {len(truth_df)})", styles['h3']))
    if truth_df.empty: story.append(Paragraph("<i>None</i>", styles['Italic']))
    else:
        for row in truth_df.itertuples(): story.append(Paragraph(ansa.highlight_triplet(row, item_to_color_map), styles['Normal']))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Annotation Legend:", styles['h3']))
    story.append(Paragraph("<b><u>'Both' Items (Green Shades)</u></b>", styles['Normal']))
    story.append(Paragraph(f'<font backColor="{BOTH_COLOR_DARK.hexval()}">Item is part of a full Triplet match.</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{BOTH_COLOR_NORMAL.hexval()}">Item is part of a Pair match (but not a full triplet match).</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{BOTH_COLOR_LIGHT.hexval()}">Item exists in both graphs (but not in stronger matches).</font>', styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b><u>'Generated Only' Items (Yellow Shades)</u></b>", styles['Normal']))
    story.append(Paragraph(f'<font backColor="{GEN_COLOR_DARK.hexval()}">Item is part of a Triplet found only by this Method.</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{GEN_COLOR_NORMAL.hexval()}">Item is part of a Pair found only by this Method.</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{GEN_COLOR_LIGHT.hexval()}">Item exists only in this Method\'s graph.</font>', styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b><u>'Ground Truth Only' Items (Blue Shades)</u></b>", styles['Normal']))
    story.append(Paragraph(f'<font backColor="{TRUTH_COLOR_DARK.hexval()}">Item is part of a Triplet found only in Ground Truth.</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{TRUTH_COLOR_NORMAL.hexval()}">Item is part of a Pair found only in Ground Truth.</font>', styles['Normal']))
    story.append(Paragraph(f'<font backColor="{TRUTH_COLOR_LIGHT.hexval()}">Item exists only in the Ground Truth graph.</font>', styles['Normal']))
    story.append(Spacer(1, 24))
    
    story.append(Paragraph(f"Annotated Text (First {NUMBER_OF_PARAGRAPHS} Paragraphs):", styles['h3']))
    story.append(Paragraph(annotated_text, styles['Normal']))
    
    doc.build(story)
    print(f"  Saved full report to {filename}")

# ==============================================================================
# SECTION 6: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    print("="*30 + " Starting Focused Analysis & Pruning " + "="*30)

    print("\nLoading heavy dependencies...")
    # Add SpaCy model loading here
    try:
        nlp = spacy.load("en_core_web_lg")
        print("SpaCy model 'en_core_web_lg' loaded for sentence tokenization.")
    except OSError:
        print("ERROR: SpaCy model 'en_core_web_lg' not found.")
        print("Please run: python -m spacy download en_core_web_lg")
        exit()
        
    wikigraphs_dataset = load_wikigraphs_data(WIKIGRAPHS_DATA_DIR)
    id_to_name = load_or_build_id_to_name_map()
    name_to_id_map = {}
    for _id, name in id_to_name.items(): name_to_id_map.setdefault(str(name).lower().strip(), []).append(_id)
    full_freebase_df = get_or_build_freebase_dataframe(FREEBASE_DATA_PATH)
    name_db = VectorDatabase(collection_name="name_embeddings", path=NAME_DB_PATH)
    predicate_db = VectorDatabase(collection_name="predicate_embeddings", path=PREDICATE_DB_PATH)
    print("Dependencies loaded successfully.")

    for sample_index in tqdm(SAMPLES_TO_ANALYZE, desc="Processing All Selected Samples"):
        if sample_index >= len(wikigraphs_dataset): continue

        sample_pair = wikigraphs_dataset[sample_index]
        sample_title_clean = "".join(c for c in sample_pair.title if c.isalnum() or c in (' ', '_')).rstrip()
        
        print(f"\n{'='*20} SAMPLE {sample_index}: {sample_pair.title} {'='*20}")
        
        cleaned_text = sample_pair.text.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
        focused_text = get_first_n_paragraphs(cleaned_text, nlp=nlp)
        ground_truth_df_original = get_ground_truth_graph(sample_pair)
        ground_truth_df_readable = create_human_readable_ground_truth(ground_truth_df_original, id_to_name)
        
        checkpoint_path = Path(CHECKPOINT_DIR) / f"sample_{sample_index:0{SAMPLE_FILENAME_ZERO_PADDING_WIDTH}d}_{sample_title_clean}.json"
        if checkpoint_path.exists():
            print(f"  Loading focused graphs from checkpoint...")
            with open(checkpoint_path, 'r') as f: checkpoint_data = json.load(f)
            raw_graph_df = pd.DataFrame(checkpoint_data['llm_raw']['data'], columns=checkpoint_data['llm_raw']['columns'])
        else:
            print(f"  Checkpoint not found. Generating focused graphs...")
            raw_graph_df = generate_graph_from_text(focused_text, GENERATION_PROMPT)
            checkpoint_data = {'llm_raw': {'data': raw_graph_df.to_dict('split')['data'], 'columns': raw_graph_df.columns.tolist()}}
            with open(checkpoint_path, 'w') as f: json.dump(checkpoint_data, f, indent=4)

        unfiltered_df, name_map = orchestrate_standardization(raw_graph_df, name_db, predicate_db)
        
        pruning_stats, pruned_metrics = None, None
        
        high_degree_nodes = find_high_degree_nodes(unfiltered_df, min_degree=MIN_NODE_DEGREE_FOR_FILTERING)
        if not high_degree_nodes:
            print(f"  No nodes with degree >= {MIN_NODE_DEGREE_FOR_FILTERING} found. Skipping pruning.")
            pruned_df = unfiltered_df.copy()
        else:
            node_predicates_map = find_freebase_predicates_for_nodes(high_degree_nodes, full_freebase_df, name_to_id_map)
            pruned_df, pruning_stats = filter_graph_with_llm(unfiltered_df, node_predicates_map)
        
        unfiltered_metrics = run_comprehensive_evaluation(unfiltered_df, ground_truth_df_readable)
        pruned_metrics = run_comprehensive_evaluation(pruned_df, ground_truth_df_readable)
        
        print("  Generating full qualitative report...")
        annotation_sets = ansa.get_annotation_sets(unfiltered_df, ground_truth_df_readable)
        item_to_color_map = ansa._get_kg_item_to_color_map(annotation_sets)
        match_counts = {"Full Triplets": ansa.get_match_counts(annotation_sets['gen_triplets'], annotation_sets['truth_triplets'])}
        annotated_html = ansa.annotate_text(focused_text, item_to_color_map)
        
        create_focused_pdf_report(
            sample_index=sample_index, sample_title=sample_title_clean, method_name="LLM Generated (Focused)", 
            unfiltered_metrics=unfiltered_metrics, pruned_metrics=pruned_metrics,
            match_counts=match_counts, annotated_text=annotated_html,
            raw_df=raw_graph_df, gen_df=unfiltered_df, truth_df=ground_truth_df_readable, 
            item_to_color_map=item_to_color_map, name_map=name_map, pruning_stats=pruning_stats
        )
        
        print("\n" + "-"*15 + " PRUNING IMPACT ANALYSIS " + "-"*15)
        orig_p = unfiltered_metrics['Resilient Triplets']['Precision'] * 100
        orig_r = unfiltered_metrics['Resilient Triplets']['Recall'] * 100
        orig_f1 = unfiltered_metrics['Resilient Triplets']['F1-Score'] * 100
        prun_p = pruned_metrics['Resilient Triplets']['Precision'] * 100
        prun_r = pruned_metrics['Resilient Triplets']['Recall'] * 100
        prun_f1 = pruned_metrics['Resilient Triplets']['F1-Score'] * 100
        print(f"                     | {'Original Graph':<15} | {'Pruned Graph':<15} | {'Change':<10}")
        print(f"---------------------|-----------------|-----------------|-----------")
        print(f"Precision            | {orig_p:14.2f}% | {prun_p:14.2f}% | {(prun_p - orig_p):+9.2f}%")
        print(f"Recall               | {orig_r:14.2f}% | {prun_r:14.2f}% | {(prun_r - orig_r):+9.2f}%")
        print(f"F1-Score             | {orig_f1:14.2f}% | {prun_f1:14.2f}% | {(prun_f1 - orig_f1):+9.2f}%")
        print(f"Triplet Count        | {len(unfiltered_df):<15} | {len(pruned_df):<15} | {len(pruned_df) - len(unfiltered_df):+9}")
        print("-" * 65)

    print("\n" + "="*30 + " Focused Analysis Complete " + "="*30)