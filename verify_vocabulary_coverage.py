# ==============================================================================
# SCRIPT: verify_vocabulary_coverage.py
# ==============================================================================
# This script performs a crucial validation step for the thesis experiment.
# It verifies what percentage of the ground truth entities and predicates from the
# WikiGraphs dataset are actually present in the global Freebase knowledge graph
# used for standardization and linking.

import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from wikigraphs.data import paired_dataset, io_tools

# --- Configuration ---
WIKIGRAPHS_DATA_DIR = "data/wikigraphs/"

# --- Helper Functions (from the main script) ---

def get_or_build_freebase_dataframe():
    """
    Loads the full Freebase DataFrame from a fast Parquet cache. If the cache
    doesn't exist, it builds the DataFrame from the raw source data and creates
    the cache for future runs.
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
        
        print(f"Saving full DataFrame to Parquet cache at '{freebase_parquet_path}'...")
        full_freebase_df.to_parquet(freebase_parquet_path)
    
    print("Full Freebase DataFrame loaded.")
    return full_freebase_df

def get_ground_truth_graph(pair):
    g = pair.graph
    df = pd.DataFrame(g.edges(), columns=["src", "tgt", "edge"])
    df["subject"] = df["src"].apply(lambda node_id: g.nodes()[node_id])
    df["object"] = df["tgt"].apply(lambda node_id: g.nodes()[node_id])
    df = df[["subject", "edge", "object"]]
    df.rename(columns={"edge": "predicate"}, inplace=True)
    return df.drop_duplicates().reset_index(drop=True)

def load_wikigraphs_data(data_root, subset='train', version='max256'):
    print("Loading WikiGraphs dataset...")
    paired_dataset.DATA_ROOT = data_root
    dataset = paired_dataset.ParsedDataset(
        subset=subset, shuffle_data=False, data_dir=None, version=version
    )
    return list(dataset)

# --- Main Verification Logic ---

if __name__ == '__main__':
    print("="*30 + " STARTING VOCABULARY COVERAGE VERIFICATION " + "="*30)

    # 1. Load the global Freebase knowledge graph
    full_freebase_df = get_or_build_freebase_dataframe()
    
    # 2. Create sets of all known entities and predicates for fast lookup
    global_entities = set(pd.concat([full_freebase_df['subject'], full_freebase_df['object']]).unique())
    global_predicates = set(full_freebase_df['predicate'].unique())
    
    print(f"\nGlobal KG contains {len(global_entities)} unique entity IDs and {len(global_predicates)} unique predicates.")

    # 3. Load the WikiGraphs dataset
    parsed_pairs = load_wikigraphs_data(WIKIGRAPHS_DATA_DIR)
    
    # 4. Initialize sets to store missing items
    missing_entities = set()
    missing_predicates = set()
    total_gt_entities = set()
    total_gt_predicates = set()

    # 5. Iterate through every sample and check for missing items
    print(f"\nVerifying vocabulary coverage for {len(parsed_pairs)} ground truth graphs...")
    for pair in tqdm(parsed_pairs, desc="Verifying Samples"):
        gt_graph = get_ground_truth_graph(pair)
        
        # Get entities and predicates from this specific ground truth graph
        gt_entities_in_sample = set(pd.concat([gt_graph['subject'], gt_graph['object']]).unique())
        gt_predicates_in_sample = set(gt_graph['predicate'].unique())
        
        # Update total sets
        total_gt_entities.update(gt_entities_in_sample)
        total_gt_predicates.update(gt_predicates_in_sample)
        
        # Find items in this sample that are NOT in the global sets
        missing_entities.update(gt_entities_in_sample - global_entities)
        missing_predicates.update(gt_predicates_in_sample - global_predicates)

    # 6. Report the final, aggregated results
    print("\n" + "="*30 + " VOCABULARY COVERAGE REPORT " + "="*30)
    print(f"Checked {len(parsed_pairs)} ground truth graphs from the WikiGraphs dataset.")
    
    print("\n--- Entity Coverage ---")
    print(f"Total unique entities found across all ground truth graphs: {len(total_gt_entities)}")
    print(f"Number of ground truth entities MISSING from the global Freebase KG: {len(missing_entities)}")
    if total_gt_entities:
        coverage_percent = (1 - len(missing_entities) / len(total_gt_entities)) * 100
        print(f"Entity Vocabulary Coverage: {coverage_percent:.2f}%")
    if missing_entities:
        print("Sample of missing entities:", list(missing_entities)[:10])

    print("\n--- Predicate Coverage ---")
    print(f"Total unique predicates found across all ground truth graphs: {len(total_gt_predicates)}")
    print(f"Number of ground truth predicates MISSING from the global Freebase KG: {len(missing_predicates)}")
    if total_gt_predicates:
        coverage_percent = (1 - len(missing_predicates) / len(total_gt_predicates)) * 100
        print(f"Predicate Vocabulary Coverage: {coverage_percent:.2f}%")
    if missing_predicates:
        print("Sample of missing predicates:", list(missing_predicates)[:10])
        
    print("\n" + "="*72)