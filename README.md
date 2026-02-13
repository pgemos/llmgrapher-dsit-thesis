# LLMGrapher: Document-Level Knowledge Graph Construction

**LLMGrapher** is a modular research pipeline designed to evaluate the capabilities of Large Language Models (LLMs) in constructing Knowledge Graphs (KGC) from unstructured text.

This repository contains the source code, data handling scripts, and experimental results for the Master's Thesis: **"LLMGrapher: Transforming Texts to Knowledge Graphs with the Power of Large Language Models"**.

## 📖 Overview

The project investigates the **Schema Alignment Bottleneck**—the difficulty zero-shot LLMs face in mapping extracted facts to rigid Knowledge Base identifiers (e.g., Freebase). It proposes a hybrid pipeline separating **Generation** (via Llama 3.1) from **Normalization** (via VectorDBs and Contextual Reranking).

### Key Features
*   **Zero-Shot Extraction:** Utilizing prompt engineering to extract entities and relations without fine-tuning.
*   **Modular Linking:** Pluggable modules for Candidate Retrieval (ChromaDB) and Contextual Reranking.
*   **Recall-Centric Evaluation:** A multi-layered metric suite (Strict vs. Resilient vs. Semantic).
*   **Diagnostic Tools:** Automated PDF reporting for qualitative error analysis and Knowledge-Based Pruning.

---

## 🚀 Setup & Installation

### Prerequisites
*   **Python 3.9+**
*   **Ollama:** Ensure [Ollama](https://ollama.com/) is installed and running.
*   **Model:** Pull the Llama 3.1 model: `ollama pull llama3.1:8b`
*   **System Tools:** `wget`, `tar`, `unzip`, `git` (standard on Linux/Mac).

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pgemos/llmgrapher-dsit-thesis.git
    cd llmgrapher-dsit-thesis
    ```

2.  **Setup the WikiGraphs Library:**
    This project relies on the DeepMind `wikigraphs` data loaders. If not already present in the root, setup the module:
    ```bash
    # Clone and arrange the DeepMind library (Linux/Mac)
    git clone https://github.com/google-deepmind/deepmind-research.git temp_dm
    mv temp_dm/wikigraphs/wikigraphs .
    mv temp_dm/wikigraphs/scripts wikigraphs/
    rm -rf temp_dm
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download SpaCy model:**
    ```bash
    python -m spacy download en_core_web_lg
    ```
    
5. **Download Datasets:**
    Run the included script to automatically fetch WikiGraphs and Freebase data:
    ```bash
    chmod +x ./wikigraphs/scripts/download.sh
    ./wikigraphs/scripts/download.sh
    ```

6.  **Build the WikiGraphs Dataset:**
    The WikiGraphs dataset must be constructed locally by pairing the downloaded text with the knowledge graphs. This project uses the tools provided by the [DeepMind WikiGraphs repository](https://github.com/google-deepmind/deepmind-research/tree/master/wikigraphs).

    Run the construction script (located in `wikigraphs/scripts`):
    ```bash
    # Example command to build the 'max256' version for all splits
    python wikigraphs/scripts/freebase_preprocess.py \
      --wikitext_dir=data/wikitext-103 \
      --freebase_dir=data/freebase/max256 \
      --output_dir=data/wikigraphs/max256
    ```
    *This process aligns the unstructured text with the structured subgraphs to create the Ground Truth.*

### ⚠️ Important Note on First Run
The first time you run *any* experiment script (see below), the system will automatically:
1.  Parse the raw Freebase data (`whole.gz`) and create `full_freebase_df.parquet`.
2.  Build the persistent Vector Databases (`name_embeddings` and `predicate_embeddings`).

**This initialization process takes significant time (approx. 10-20 minutes).** Subsequent runs will use these cached files and be much faster.

---

## 📂 Repository Contents & File Description

### 📝 Thesis Manuscript
| Folder | Description |
| :--- | :--- |
| `pgemos_dsit_thesis/` | **Thesis Manuscript:** Contains the full LaTeX source code, chapters, and figures for the thesis document. |

### 🧪 Experiment Scripts (Core Pipeline)
These scripts contain the main pipelines for generating graphs from text, standardizing entities/predicates, and evaluating against ground truth.

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
Files used to store the Knowledge Base, cached dataframes to speed up loading, and embeddings. (*created during and after execution of project code files*)

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
Directories where the system saves progress and final reports. (*created during and after execution of project code files*)

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

---

## 📊 Evaluation Logic

To rigorously assess the quality of generated graphs, the pipeline employs a **multi-layered evaluation strategy** moving from rigid syntactic matching to flexible semantic assessment.

### 1. Strict Matching (Syntactic)
*   **Definition:** Requires an exact, case-insensitive string match for the triplet `(Subject, Predicate, Object)`.
*   **Purpose:** Measures the model's ability to perfectly adhere to the specific Freebase schema vocabulary and directionality.
*   **Limitation:** Often penalizes valid synonyms (e.g., *“wrote”* vs *“author of”*).

### 2. Resilient Matching (Structural)
*   **Definition:** Checks if the pair of entities `{Subject, Object}` exists in the ground truth, regardless of the edge label or direction.
*   **Purpose:** Isolates **Entity Linking** performance from **Relation Extraction**. It determines if the model correctly identified that *a* relationship exists, even if it got the specific predicate wrong.

### 3. Semantic Matching (LLM-as-a-Judge)
*   **Definition:** If the entities match but the predicate differs, a second LLM acts as a judge to compare the semantic meaning of the generated predicate against the ground truth predicate.
*   **Categories:**
    *   **High Confidence:** The predicates are semantically identical (e.g., *“married to”* $\approx$ *“spouse”*).
    *   **Plausible:** The predicate is reasonable but less specific.
*   **Purpose:** Addresses the **Schema Alignment Bottleneck** by distinguishing between hallucinated relations and valid relations expressed in natural language.

---

## 🧪 Reproducing Experiments

To reproduce the results reported in the thesis, run the following scripts directly from the root directory. Results will be saved to CSV files in the root folder and detailed JSON logs in `checkpoints/`.

### 1. Baselines & Upper Bounds
Establish the theoretical limits and the baseline performance.

*   **Oracle Linking (Upper Bound):**
    ```bash
    python llmgrapher_experiment_oracle_linking.py
    ```
    *Tests extraction quality when linking is perfect (restricted search space).*

*   **VectorDB-Only (Lower Bound):**
    ```bash
    python llmgrapher_experiment_vectordb-only_linking.py
    ```
    *Tests standard semantic search without any LLM reranking.*

### 2. Contextual Reranking (Filtered KB) - **Main Results**
These scripts run the proposed pipeline using the **Cleaned/Filtered Knowledge Base** (removing stopwords/junk entities) to maximize precision.

*   **Linguistic Context (Sentences):**
    ```bash
    python llmgrapher_experiment_filtered_vectordb-llm_linking-sentences.py
    ```
    *Uses the original source sentence to disambiguate entities.*

*   **Structural Context (Triplets):**
    ```bash
    python llmgrapher_experiment_filtered_vectordb-llm_linking-triplets.py
    ```
    *Uses neighboring graph nodes/edges to disambiguate entities.*

### 3. Ablation Studies (Unfiltered)
To demonstrate the impact of KB filtering, you can run the standard (unfiltered) versions:
*   `llmgrapher_experiment_vectordb-llm_linking-sentences.py`
*   `llmgrapher_experiment_vectordb-llm_linking-triplets.py`

---

## 📄 Citation

If you use this code or thesis in your research, please cite:

```bibtex
@mastersthesis{gemos2025llmgrapher,
  author  = {Panagiotis-Konstantinos Gemos},
  title   = {LLMGrapher: Transforming Texts to Knowledge Graphs with the Power of Large Language Models},
  school  = {National and Kapodistrian University of Athens},
  year    = {2025},
  month   = {October}
}
```

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
