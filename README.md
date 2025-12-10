# Intent Expansion Pipeline

This project provides a pipeline for analyzing customer messages and proposing new or refined conversational AI intents using clustering and (optionally) LLM-based summarization.

## Features

- Vectorizes messages using Gemini, sentence-transformers, or TF-IDF+SVD.
- Clusters messages using HDBSCAN or KMeans.
- Analyzes clusters to propose new or split secondary intents.
- Optionally uses Gemini or OpenAI LLMs to generate intent names and descriptions.
- Outputs cluster reports and suggested intents for review.

## Requirements

- Python 3.7+
- Required: `numpy`, `pandas`, `scikit-learn`
- Optional for better embeddings: `sentence-transformers`
- Optional for better clustering: `hdbscan`
- Optional for LLM naming: `google-generativeai` (Gemini) or `openai`

Install requirements:
```bash
pip install numpy pandas scikit-learn
pip install sentence-transformers hdbscan google-generativeai openai  # as needed
```

## Usage

```bash
python intent_expansion_pipeline.py --input path/to/inputs_for_assignment.json --outdir output_dir
```

### Optional arguments

- `--use_gemini` and `--gemini_api_key <KEY>`: Use Gemini for embeddings and LLM naming.
- `--use_openai` and `--openai_api_key <KEY>`: Use OpenAI for LLM naming.
- `--no_sentence_transformers`: Do not use sentence-transformers even if installed.
- `--prefer_hdbscan`: Prefer HDBSCAN for clustering.
- `--min_cluster_size <N>`: Minimum cluster size for proposals.
- `--t_split_ratio <FLOAT>`: Ratio for split heuristic (default 0.2).
- `--t_new_cluster_pct <FLOAT>`: Cluster percent threshold for new intent (default 0.03).

## Outputs

- `intent_expansion_report.csv`: Cluster-level analytics.
- `suggested_intents.json`: Proposed new or split intents.
- `cluster_examples.csv`: Example messages per cluster.

## Input Format

The input JSON should contain:
- `customer_messages`: List of message dicts, each with at least a `current_human_message` or `current_message`.
- `intent_mapper`: (Optional) Existing intent taxonomy.

## Example

```bash
python intent_expansion_pipeline.py --input inputs_for_assignment.json --outdir out_intent_expansion --use_gemini --gemini_api_key YOUR_API_KEY
```

## License

MIT License.
