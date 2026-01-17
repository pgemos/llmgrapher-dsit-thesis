# LLMGrapher Project Structure

This repository contains the codebase, data, and experimental results for **LLMGrapher**, a research project investigating the capabilities of Large Language Models (LLMs) to generate Knowledge Graphs from text, specifically targeting the WikiGraphs dataset and aligning with the Freebase ontology.

## 📂 Directory Overview

### 🧪 Experiment Scripts (Core Pipeline)
These scripts contain the main pipelines for generating graphs from text, standardizing entities/predicates, and evaluating against ground truth. They vary by linking strategy and context usage.

| File Name | Description |
| :--- | :--- |
| `llmgrapher_experiment_vectordb-only_linking.py` | **Basic Strategy:** Uses only VectorDB retrieval (top-1) to link extracted terms to Freebase entities/predicates without a secondary LLM verification step. |
| `llmgrapher_experiment_vectordb-llm_linking-triplets.py` | **Context Variation:** Uses VectorDB for candidate generation, followed by an LLM Reranker provided with "Triplet Context" (neighboring nodes/edges) to aid disambiguation. |
| `llmgrapher_experiment_vectordb-llm_linking-sentences.py` | **Context Variation:** Provides the LLM reranker with the *original sentence* from the source text as context for disambiguation. |
| `llmgrapher_experiment_filtered_vectordb-llm_linking-triplets.py` | **Optimization:** Runs the triplet-context strategy against a *Filtered* version of Freebase (removing noise/junk entities) to improve retrieval precision. |
| `llmgrapher_experiment_filtered_vectordb-llm_linking-sentences.py`| **Optimization:** Runs the sentence-context strategy against the *Filtered* Freebase knowledge base. |
| `llmgrapher_experiment_oracle_linking.py` | **Gold Standard:** An "Oracle" experiment where the standardization search space is restricted *only* to entities present in the Ground Truth, establishing an upper bound for extraction performance. |
| `llmgrapher_experiment_withPairs.py` | An iteration of the experiment focused on evaluating connected node pairs in addition to triplets. |
| `llmgrapher_experiment_...-debug.py` | Debug versions of the scripts containing verbose logging to trace the internal logic of the LLM and standardization steps. |

### 🔍 Analysis & Diagnostic Tools
Scripts used to deep-dive into the results, generate visual reports, and perform error analysis.

| File Name | Description |
| :--- | :--- |
| `analyze_samples_focused.py` | **Qualitative Diagnostic Investigation:** The primary tool for the thesis's "Qualitative Error Analysis". It conducts a deep-dive diagnosis on truncated text segments (first N paragraphs) to isolate specific failure modes, producing detailed pdf reports. In addition, it implements **Knowledge-Based Pruning** to filter hallucinations against Freebase, generating detailed visual comparisons to assess the impact of semantic refinement. |
| `analyze_samples.py` | A counterpart to the `analyze_samples_focused.py` that processes the *full* source text to calculate aggregate performance metrics and producing detailed pdf reports that assist for conducting qualitative error analysis. Unlike the focused analysis, this script evaluates the raw generation **without Knowledge-Based Pruning** or text truncation. |
| `results-visulizer.ipynb` | Jupyter notebook for visualizing quantitative results (charts, tables of precision/recall/F1). |
| `test_correference_llm.ipynb` | Notebook for the validation and testing of intrinsic LLM-prompted coreference resolution. |
| `exploration.ipynb` | Scratchpad notebook for initial data exploration and code testing. |

### 📦 Data & Caching
Files used to store the Knowledge Base, cached dataframes to speed up loading, and embeddings.

| File/Folder | Description |
| :--- | :--- |
| `data/` | Contains the raw WikiGraphs dataset and pruned versions of Freebase as constructed by Wikigraphs authors. |
| `full_freebase_df.parquet` | **Critical Cache:** A Parquet file containing the processed Freebase Knowledge Graph (Subject-Predicate-Object). Used for fast loading. |
| `full_freebase_df_max1024.parquet` | The cached max1024 version of Freebase. |
| `full_freebase_df_filtered.parquet` | A cleaned version of the Freebase cache with "junk" entities (short names, stopwords) removed. |
| `id_to_name_map.json` | A mapping file to translate Freebase IDs to human-readable names. |
| `name_embeddings/` | ChromaDB folder containing vector embeddings for Freebase entity names. |
| `predicate_embeddings/` | ChromaDB folder containing vector embeddings for Freebase predicates. |
| `description_embeddings/` | ChromaDB folder containing vector embeddings for Freebase entity descriptions. |

### 💾 Checkpoints & Outputs
Directories where the system saves progress and final reports.

| Folder | Description |
| :--- | :--- |
| `checkpoints/` | JSON files containing the raw output of LLM generation for full experiments. Prevents re-running expensive generation. |
| `checkpoints_focused/` | JSON files specific to the "Focused" analysis (truncated text runs). |
| `analysis_reports/` | PDF reports generated by the standard analysis pipeline. |
| `analysis_reports_focused/` | **Diagnostic Reports:** Detailed PDF case studies on the truncated text runs including highlighted text, pruning statistics, and error analysis for specific samples. |
| `results/` | CSV files containing aggregate quantitative metrics. |

### 🛠️ Utilities
Helper scripts for maintenance and verification.

| File Name | Description |
| :--- | :--- |
| `verify_vocabulary_coverage.py` | This script performs a crucial validation step for the thesis experiment. It verifies what percentage of the ground truth entities and predicates from the WikiGraphs dataset are actually present in the global Freebase knowledge graph used for standardization and linking. |
| `upgrade_checkpoints.py` | Utility to update the format of existing JSON checkpoint files if the schema changes. |
| `final_experiment_suite.ipynb` | Experimental notebook used initially as a preparation ground for running the final suite of experiments which are now done via the `llmgrapher_experiment_*` files. |
| `versioning/` | Contains archived or deprecated scripts. |

---

### 📝 Note on "Focused" vs. Standard Scripts
The `*_focused.py` scripts and directories relate to the qualitative error analysis phase of the thesis. They operate on a subset of data (first two paragraphs) to allow for manual inspection, visual highlighting of errors (hallucinations vs. mapping errors), and testing of the pruning hypothesis.