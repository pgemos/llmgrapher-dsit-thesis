# ==============================================================================
# QUALITATIVE ANALYSIS SCRIPT for LLMGrapher Experiment (V6)
# ==============================================================================
# This script loads results from experiment checkpoints and generates detailed,
# annotated PDF reports with a fully corrected 9-color highlighting scheme.
#
# V6 FIXES & FEATURES:
# - HIERARCHY CORRECTION: The core color-mapping logic has been re-engineered
#   to give "Both" (Green) matches absolute priority over "Generated Only"
#   (Yellow) and "Truth Only" (Blue) matches. Within each color, darker
#   shades have priority. This fixes the bug where green shades were being
#   incorrectly overridden.
# - The script now correctly produces the full range of 9 colors as intended.
#
# INSTRUCTIONS:
# 1. Place this script in the same directory as your main experiment script.
# 2. Edit `SAMPLES_TO_ANALYZE` to select which samples to process.
# 3. Run the script: `python qualitative_analyzer_v6.py`
# ==============================================================================

# --- 0. Installation Notes ---
# pip install reportlab pandas tqdm spacy chromadb ollama
# Ensure you have the spaCy model: python -m spacy download en_core_web_lg

import os
import json
import pandas as pd
import re
from tqdm.auto import tqdm
from pathlib import Path
import ollama

# PDF Generation Library
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import HexColor

# --- 1. Configuration ---
SAMPLES_TO_ANALYZE = list(range(10))  # <-- EDIT THIS LIST with the sample indexes you want to investigate

# Directory Paths
WIKIGRAPHS_DATA_DIR = "data/wikigraphs/"
CHECKPOINT_DIR = "checkpoints"
ANALYSIS_OUTPUT_DIR = "analysis_reports"
NAME_DB_PATH = "name_embeddings"
PREDICATE_DB_PATH = "predicate_embeddings"

Path(ANALYSIS_OUTPUT_DIR).mkdir(exist_ok=True)

# Define the 3-level, 9-color scheme
GEN_COLOR_LIGHT = HexColor('#FFFACD')   # Light Yellow
GEN_COLOR_NORMAL = HexColor('#FFEE75')  # Normal Yellow
GEN_COLOR_DARK = HexColor('#FFD700')    # Dark Yellow

TRUTH_COLOR_LIGHT = HexColor('#E0FFFF')  # Light Blue
TRUTH_COLOR_NORMAL = HexColor('#ADD8E6') # Normal Blue
TRUTH_COLOR_DARK = HexColor('#4682B4')   # Dark Blue

BOTH_COLOR_LIGHT = HexColor('#DFF0D8')   # Light Green
BOTH_COLOR_NORMAL = HexColor('#90EE90')  # Normal Green
BOTH_COLOR_DARK = HexColor('#228B22')    # Dark Green


# ==============================================================================
# SECTION 2: COPIED HELPER FUNCTIONS FROM MAIN EXPERIMENT SCRIPT
# ==============================================================================

from wikigraphs.data import paired_dataset

METADATA_PREDICATE_BLACKLIST = [
    'ns/type.object.name', 'ns/common.topic.description', 'key/wikipedia.en',
    'ns/common.topic.alias', 'ns/type.object.id', 'ns/common.topic.image'
]

def load_wikigraphs_data(data_root, subset='train', version='max256'):
    paired_dataset.DATA_ROOT = data_root
    dataset = paired_dataset.ParsedDataset(subset=subset, shuffle_data=False, data_dir=None, version=version)
    return list(dataset)

def get_ground_truth_graph(pair, predicate_blacklist=METADATA_PREDICATE_BLACKLIST):
    g = pair.graph
    df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
    df["subject"] = df["src"].apply(lambda node_id: g.nodes()[node_id])
    df["object"] = df["tgt"].apply(lambda node_id: g.nodes()[node_id])
    df = df[["subject", "edge", "object"]].rename(columns={"edge": "predicate"})
    if predicate_blacklist:
        df = df[~df['predicate'].isin(predicate_blacklist)]
    return df.drop_duplicates().reset_index(drop=True)

def create_human_readable_ground_truth(truth_df, id_to_name_map):
    readable_df = truth_df.copy()
    readable_df['subject'] = readable_df['subject'].apply(lambda x: id_to_name_map.get(x, x))
    readable_df['object'] = readable_df['object'].apply(lambda x: id_to_name_map.get(x, x))
    return readable_df

def load_or_build_id_to_name_map():
    id_map_path = Path('id_to_name_map.json')
    if not id_map_path.exists():
        raise FileNotFoundError(f"Critical file not found: '{id_map_path}'. Please run the main experiment script once to generate it.")
    with open(id_map_path, 'r') as f:
        return json.load(f)

class VectorDatabase:
    def __init__(self, collection_name, path):
        import chromadb
        if not os.path.exists(path):
            raise FileNotFoundError(f"VectorDB path not found: '{path}'. Please run the main experiment script to build the databases.")
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(name=collection_name)

    def query(self, query_text, n_results):
        if self.collection.count() == 0: return None
        results = self.collection.query(query_texts=[str(query_text)], n_results=n_results)
        return results['documents'][0] if results and results.get('documents') and results['documents'][0] else None

def standardize_graph_items_simple(graph_df, vector_db, item_type='entity'):
    if graph_df.empty: return graph_df
    columns = ['node_1', 'node_2'] if item_type == 'entity' else ['edge']
    items = pd.concat([graph_df[col] for col in columns if col in graph_df]).unique()
    mapping = {item: (vector_db.query(str(item), 1)[0] if vector_db.query(str(item), 1) else (item if item_type == 'entity' else None)) for item in items}
    std_df = graph_df.copy()
    for col in columns:
        if col in std_df:
            std_df[col] = std_df[col].map(mapping)
    if item_type == 'predicate' and 'edge' in std_df:
        std_df.dropna(subset=['edge'], inplace=True)
    return std_df.drop_duplicates().reset_index(drop=True)

def orchestrate_standardization(raw_graph_df, name_db, predicate_db):
    std_entity_df = standardize_graph_items_simple(raw_graph_df, name_db, item_type='entity')
    return standardize_graph_items_simple(std_entity_df, predicate_db, item_type='predicate')

def _calculate_metrics_from_counts(tp, gen_count, truth_count):
    if gen_count == 0 and truth_count == 0: return {"Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0}
    p = tp / gen_count if gen_count > 0 else 0.0
    r = tp / truth_count if truth_count > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"Precision": p, "Recall": r, "F1-Score": f1}

def run_comprehensive_evaluation(generated_df, truth_df):
    if generated_df.empty and truth_df.empty:
        metrics = {"Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0}
        return {k: metrics for k in ["Strict Triplets", "Resilient Triplets"]}
    if generated_df.empty:
        metrics = {"Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0}
        return {k: metrics for k in ["Strict Triplets", "Resilient Triplets"]}

    gen_norm = generated_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'})
    gen_norm = gen_norm.apply(lambda x: x.astype(str).str.lower().str.strip())
    truth_norm = truth_df.apply(lambda x: x.astype(str).str.lower().str.strip())
    num_gen, num_truth = len(gen_norm), len(truth_norm)

    gen_strict_set = set(map(tuple, gen_norm[['subject', 'predicate', 'object']].to_records(index=False)))
    truth_strict_set = set(map(tuple, truth_norm[['subject', 'predicate', 'object']].to_records(index=False)))
    strict_tp = len(gen_strict_set.intersection(truth_strict_set))

    gen_resilient_set = {(tuple(sorted((p.subject, p.object))), p.predicate) for p in gen_norm.itertuples(index=False)}
    truth_resilient_set = {(tuple(sorted((p.subject, p.object))), p.predicate) for p in truth_norm.itertuples(index=False)}
    resilient_tp = len(gen_resilient_set.intersection(truth_resilient_set))

    return {
        "Strict Triplets": _calculate_metrics_from_counts(strict_tp, num_gen, num_truth),
        "Resilient Triplets": _calculate_metrics_from_counts(resilient_tp, num_gen, num_truth)
    }

# ==============================================================================
# SECTION 3: PDF GENERATION AND ANNOTATION LOGIC (V6 - CORRECTED)
# ==============================================================================

def get_annotation_sets(generated_df, truth_df):
    """Creates sets of nodes, predicates, pairs, and triplets for annotation."""
    gen_norm = generated_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'})
    gen_norm = gen_norm.apply(lambda x: x.astype(str).str.lower().str.strip())
    truth_norm = truth_df.apply(lambda x: x.astype(str).str.lower().str.strip())

    return {
        "gen_nodes": set(pd.concat([gen_norm['subject'], gen_norm['object']]).unique()) if not gen_norm.empty else set(),
        "truth_nodes": set(pd.concat([truth_norm['subject'], truth_norm['object']]).unique()) if not truth_norm.empty else set(),
        "gen_predicates": set(gen_norm['predicate'].unique()) if not gen_norm.empty else set(),
        "truth_predicates": set(truth_norm['predicate'].unique()) if not truth_norm.empty else set(),
        "gen_pairs": {tuple(sorted(p)) for p in gen_norm[['subject', 'object']].to_records(index=False)} if not gen_norm.empty else set(),
        "truth_pairs": {tuple(sorted(p)) for p in truth_norm[['subject', 'object']].to_records(index=False)} if not truth_norm.empty else set(),
        "gen_triplets": {(tuple(sorted((p.subject, p.object))), p.predicate) for p in gen_norm.itertuples(index=False)} if not gen_norm.empty else set(),
        "truth_triplets": {(tuple(sorted((p.subject, p.object))), p.predicate) for p in truth_norm.itertuples(index=False)} if not truth_norm.empty else set(),
    }

def get_match_counts(gen_set, truth_set):
    """Calculates counts for intersection and differences between two sets."""
    return { "both": len(gen_set.intersection(truth_set)), "gen_only": len(gen_set - truth_set), "truth_only": len(truth_set - gen_set) }

def clean_kg_name_for_search(name):
    """Removes KG artifacts like quotes and language tags for text searching."""
    name = str(name).strip()
    if name.startswith('"') and name.endswith('"'): name = name[1:-1]
    return re.sub(r'@..$', '', name)

def _get_kg_item_to_color_map(annotation_sets):
    """
    Builds a master dictionary mapping each KG item to its correctly-tiered, 9-category color.
    HIERARCHY: Green > Yellow > Blue. Within each color: Dark > Normal > Light.
    """
    item_to_color = {}

    # Define match sets for easier access
    both_triplets = annotation_sets['gen_triplets'].intersection(annotation_sets['truth_triplets'])
    gen_only_triplets = annotation_sets['gen_triplets'] - both_triplets
    truth_only_triplets = annotation_sets['truth_triplets'] - both_triplets
    
    both_pairs = annotation_sets['gen_pairs'].intersection(annotation_sets['truth_pairs'])
    gen_only_pairs = annotation_sets['gen_pairs'] - both_pairs
    truth_only_pairs = annotation_sets['truth_pairs'] - both_pairs

    both_nodes = annotation_sets['gen_nodes'].intersection(annotation_sets['truth_nodes'])
    gen_only_nodes = annotation_sets['gen_nodes'] - both_nodes
    truth_only_nodes = annotation_sets['truth_nodes'] - both_nodes
    
    both_preds = annotation_sets['gen_predicates'].intersection(annotation_sets['truth_predicates'])
    gen_only_preds = annotation_sets['gen_predicates'] - both_preds
    truth_only_preds = annotation_sets['truth_predicates'] - both_preds

    # --- 1. GREEN PASS (Absolute Priority) ---
    for nodes, pred in both_triplets:
        for node in nodes: item_to_color[node] = BOTH_COLOR_DARK
        item_to_color[pred] = BOTH_COLOR_DARK
    for pair in both_pairs:
        for node in pair:
            if node not in item_to_color: item_to_color[node] = BOTH_COLOR_NORMAL
    for node in both_nodes:
        if node not in item_to_color: item_to_color[node] = BOTH_COLOR_LIGHT
    for pred in both_preds:
        if pred not in item_to_color: item_to_color[pred] = BOTH_COLOR_LIGHT

    # --- 2. YELLOW PASS (For uncolored items) ---
    for nodes, pred in gen_only_triplets:
        for node in nodes:
            if node not in item_to_color: item_to_color[node] = GEN_COLOR_DARK
        if pred not in item_to_color: item_to_color[pred] = GEN_COLOR_DARK
    for pair in gen_only_pairs:
        for node in pair:
            if node not in item_to_color: item_to_color[node] = GEN_COLOR_NORMAL
    for node in gen_only_nodes:
        if node not in item_to_color: item_to_color[node] = GEN_COLOR_LIGHT
    for pred in gen_only_preds:
        if pred not in item_to_color: item_to_color[pred] = GEN_COLOR_LIGHT
        
    # --- 3. BLUE PASS (For remaining uncolored items) ---
    for nodes, pred in truth_only_triplets:
        for node in nodes:
            if node not in item_to_color: item_to_color[node] = TRUTH_COLOR_DARK
        if pred not in item_to_color: item_to_color[pred] = TRUTH_COLOR_DARK
    for pair in truth_only_pairs:
        for node in pair:
            if node not in item_to_color: item_to_color[node] = TRUTH_COLOR_NORMAL
    for node in truth_only_nodes:
        if node not in item_to_color: item_to_color[node] = TRUTH_COLOR_LIGHT
    for pred in truth_only_preds:
        if pred not in item_to_color: item_to_color[pred] = TRUTH_COLOR_LIGHT

    return item_to_color

def annotate_text(text, item_to_color_map):
    """Applies hierarchical background color tagging to the text."""
    search_term_to_color = {clean_kg_name_for_search(k): v for k, v in item_to_color_map.items()}
    sorted_phrases = sorted(search_term_to_color.keys(), key=len, reverse=True)
    annotated_text = text
    for phrase in sorted_phrases:
        if not phrase: continue
        color = search_term_to_color[phrase]
        try:
            annotated_text = re.sub(
                f"\\b({re.escape(phrase)})\\b", f'<font backColor="{color.hexval()}">\\1</font>', 
                annotated_text, flags=re.IGNORECASE
            )
        except re.error:
            annotated_text = re.sub(
                f"({re.escape(phrase)})", f'<font backColor="{color.hexval()}">\\1</font>', 
                annotated_text, flags=re.IGNORECASE
            )
    return annotated_text.replace('\n', '<br/>')

def highlight_triplet(triplet_row, item_to_color_map):
    """Returns an HTML string for a single triplet with its parts highlighted."""
    s, p, o = triplet_row.subject, triplet_row.predicate, triplet_row.object
    s_norm, p_norm, o_norm = str(s).lower().strip(), str(p).lower().strip(), str(o).lower().strip()
    
    s_color, p_color, o_color = item_to_color_map.get(s_norm), item_to_color_map.get(p_norm), item_to_color_map.get(o_norm)
    
    s_html = f'<font backColor="{s_color.hexval()}">{s}</font>' if s_color else str(s)
    p_html = f'<font backColor="{p_color.hexval()}">{p}</font>' if p_color else str(p)
    o_html = f'<font backColor="{o_color.hexval()}">{o}</font>' if o_color else str(o)
    
    return f"({s_html}, &nbsp; {p_html}, &nbsp; {o_html})"

def create_pdf_report(sample_index, sample_title, method_name, metrics, match_counts, annotated_text, gen_df, truth_df, item_to_color_map):
    """Creates a PDF report with metrics, stats, legend, annotated text, and highlighted triplet lists."""
    sample_dir = Path(ANALYSIS_OUTPUT_DIR) / f"Sample_{sample_index}_{sample_title}"
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    clean_method_name = "".join(c for c in method_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    filename = sample_dir / f"{clean_method_name}.pdf"
    doc = SimpleDocTemplate(str(filename))
    
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(f"Analysis for Sample {sample_index}: {sample_title}", styles['h1']))
    story.append(Paragraph(f"Method: {method_name}", styles['h2']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Performance Metrics (for this sample):", styles['h3']))
    for metric_type, values in metrics.items():
        story.append(Paragraph(f"<b>{metric_type}:</b>", styles['Normal']))
        for key, val in values.items():
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{key}: {val*100:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detailed Match Statistics:", styles['h3']))
    for category, counts in match_counts.items():
        story.append(Paragraph(f"<b>{category}:</b>", styles['Normal']))
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;- In Both Graphs: {counts['both']}", styles['Normal']))
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;- Only in Generated Graph: {counts['gen_only']}", styles['Normal']))
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;- Only in Ground Truth: {counts['truth_only']}", styles['Normal']))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"Generated Triplets: (Total: {len(gen_df)})", styles['h3']))
    if gen_df.empty: story.append(Paragraph("<i>None</i>", styles['Italic']))
    else:
        for row in gen_df.itertuples(): story.append(Paragraph(highlight_triplet(row, item_to_color_map), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Ground Truth Triplets: (Total: {len(truth_df)})", styles['h3']))
    if truth_df.empty: story.append(Paragraph("<i>None</i>", styles['Italic']))
    else:
        for row in truth_df.itertuples(): story.append(Paragraph(highlight_triplet(row, item_to_color_map), styles['Normal']))
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
    
    story.append(Paragraph("Annotated Text:", styles['h3']))
    story.append(Paragraph(annotated_text, styles['Normal']))
    
    doc.build(story)
    print(f"Saved report to {filename}")

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    print("="*30 + " Starting Qualitative Analysis (V6) " + "="*30)

    print("\nLoading dependencies...")
    wikigraphs_dataset = load_wikigraphs_data(WIKIGRAPHS_DATA_DIR)
    id_to_name = load_or_build_id_to_name_map()
    name_db = VectorDatabase(collection_name="name_embeddings", path=NAME_DB_PATH)
    predicate_db = VectorDatabase(collection_name="predicate_embeddings", path=PREDICATE_DB_PATH)
    print("Dependencies loaded successfully.")

    for sample_index in tqdm(SAMPLES_TO_ANALYZE, desc="Processing Samples"):
        if sample_index >= len(wikigraphs_dataset): continue

        sample_pair = wikigraphs_dataset[sample_index]
        sample_title_clean = "".join(c for c in sample_pair.title if c.isalnum() or c in (' ', '_')).rstrip()
        checkpoint_path = Path(CHECKPOINT_DIR) / f"sample_{sample_index}_{sample_title_clean}.json"

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint '{checkpoint_path}' not found. Skipping."); continue
            
        print(f"\n--- Analyzing Sample {sample_index}: '{sample_pair.title}' ---")
        
        with open(checkpoint_path, 'r') as f: checkpoint_data = json.load(f)
            
        original_text = sample_pair.text.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
        ground_truth_df_original = get_ground_truth_graph(sample_pair)
        ground_truth_df_readable = create_human_readable_ground_truth(ground_truth_df_original, id_to_name)

        method_name_map = {
            'llm_vectordb_cr': 'LLM Prompt with VectorDB for CR', 'llm_inherent_cr': 'LLM Prompt with Inherent CR',
            'pattern_based': 'Baseline (SVO + Embeddings)', 'distant_supervision': 'Baseline (Distant Supervision)'
        }

        for key, friendly_name in method_name_map.items():
            if key not in checkpoint_data: continue

            print(f"  -> Processing method: {friendly_name}")
            graph_data = checkpoint_data[key]
            raw_df = pd.DataFrame(graph_data['data'], columns=graph_data['columns'])
            
            final_df = raw_df if "Distant Supervision" in friendly_name else orchestrate_standardization(raw_df, name_db, predicate_db)
            final_df.rename(columns={'subject': 'node_1', 'object': 'node_2', 'predicate': 'edge'}, inplace=True, errors='ignore')

            per_sample_metrics = run_comprehensive_evaluation(final_df, ground_truth_df_readable)
            annotation_sets = get_annotation_sets(final_df, ground_truth_df_readable)
            item_to_color_map = _get_kg_item_to_color_map(annotation_sets)
            
            match_counts = {
                "Single Entities (Nodes)": get_match_counts(annotation_sets['gen_nodes'], annotation_sets['truth_nodes']),
                "Single Predicates": get_match_counts(annotation_sets['gen_predicates'], annotation_sets['truth_predicates']),
                "Entity Pairs": get_match_counts(annotation_sets['gen_pairs'], annotation_sets['truth_pairs']),
                "Full Triplets": get_match_counts(annotation_sets['gen_triplets'], annotation_sets['truth_triplets'])
            }
            
            annotated_html = annotate_text(original_text, item_to_color_map)
            
            final_df_report = final_df.rename(columns={'node_1': 'subject', 'node_2': 'object', 'edge': 'predicate'})
            truth_df_report = ground_truth_df_readable.rename(columns={'subject': 'subject', 'object': 'object', 'predicate': 'predicate'})
            
            create_pdf_report(
                sample_index=sample_index, sample_title=sample_title_clean, method_name=friendly_name, 
                metrics=per_sample_metrics, match_counts=match_counts, annotated_text=annotated_html,
                gen_df=final_df_report, truth_df=truth_df_report, item_to_color_map=item_to_color_map
            )
            
    print("\n" + "="*30 + " Qualitative Analysis Complete " + "="*30)