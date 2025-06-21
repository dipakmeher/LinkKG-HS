# LLM-Based Construction of Knowledge Graphs for the Analysis of Human Smuggling Networks

## Abstract

Human smuggling networks are complex and constantly evolving, making them difficult to analyze using traditional methods. Legal case documents offer rich factual and procedural information but are often lengthy, unstructured, and filled with ambiguous or shifting references—posing major challenges for automated knowledge graph (KG) construction. Existing approaches often neglect coreference resolution or fail to scale beyond short text spans, resulting in fragmented graphs and inconsistent entity linking. We propose LINK-KG, a scalable, modular framework that combines a three-stage, LLM-guided coreference resolution pipeline with structured KG extraction. At the core of LINK-KG is a type-specific Prompt Cache that enables consistent reference tracking across document chunks, producing clean, disambiguated narratives suitable for KG construction from both short and long legal texts. Compared to baselines, LINK-KG reduces node duplication by 49.16% and noisy nodes by 41.25%, resulting in significantly cleaner and more coherent graph structures—making it a strong foundation for analyzing complex criminal networks.

## Instructions to Run the Code

The repository includes the complete implementation of both the LINK-KG pipeline and its baselines (CORE-KG and GraphRAG).

### Requirements
- Python: 3.12
- Ollama (for local LLM inference): https://ollama.com/download
- GraphRAG version 0.3.2 (for KG construction baseline): https://github.com/microsoft/graphrag

## Commands

### 1. Run Coreference Resolution (LINK-KG)

Use the following command template to run the coreference resolution pipeline. This resolves entity mentions in a legal document using a type-specific LLM prompt.

```bash
python run_pipeline5.py \
  --input-file <path_to_input_text> \
  --input-file-name <file_id_without_extension> \
  --entity-type <entity_type> \
  --max-tokens 300 \
  --min-last-chunk-words 50 \
  --use-tokenizer \
  --ner-prompt-file <path_to_ner_prompt> \
  --ner-model-name <model_name_in_ollama> \
  --coref-prompt-file <path_to_coref_prompt> \
  --coref-model-name <model_name_in_ollama> \
  --resolve-prompt-file <path_to_resolve_prompt> \
  --resolve-model-name <model_name_in_ollama> \
  --run-stages chunk ner coref resolve
```

#### Argument Descriptions
- `--input-file`: Path to the raw document (e.g., `.txt` file)
- `--input-file-name`: Unique name used to create output folders
- `--entity-type`: One of person, location, routes, etc.
- `--run-stages`: Sequence of pipeline stages to run (chunk, ner, coref, resolve)

### LINK-KG Prompt Files

- Coreference Prompts: `link-kg/prompts/`
- KG Construction Prompts: `link-kg/kgconstruction/ragtest/prompts/`

### 2. Run KG Construction (GraphRAG-based)

```bash
python index.py --root ./ragtest
```

Make sure you are in the correct directory and have installed GraphRAG dependencies:  
https://github.com/microsoft/graphrag

## Running Baselines

### 1. Run Coreference Resolution (CORE-KG)

```bash
python resolve_coref_pipeline.py \
  --input-file <input_txt_file> \
  --output-folder <output_folder> \
  --person-prompt <path_to_person_prompt> \
  --routes-prompt <path_to_routes_prompt> \
  --location-prompt <path_to_location_prompt> \
  --mot-prompt <path_to_transportation_prompt> \
  --moc-prompt <path_to_communication_prompt> \
  --organization-prompt <path_to_organization_prompt> \
  --smuggleditems-prompt <path_to_smuggleditems_prompt> \
  --model <model_name_in_ollama>
```

#### CORE-KG Prompt Files

- Coreference Prompts: `core-kg/coreference-resolution/prompts/`
- KG Construction Prompts: `core-kg/kgconstruction/ragtest/prompts/`

### 2. Run KG Construction (CORE-KG or Baseline)

```bash
python index.py --root ./ragtest
```

Ensure directory structure is consistent with GraphRAG format.

## Baseline: GraphRAG Only

To run GraphRAG as a standalone baseline:

```bash
python index.py --root ./ragtest
```

- Prompts: Baseline prompt templates are located in `baseline/ragtest/prompts/`

## Conclusion

LINK-KG introduces a robust and scalable approach to constructing knowledge graphs from complex legal documents. It combines instruction-tuned LLMs with a type-specific Prompt Cache to enable consistent, high-precision coreference resolution. Unlike previous methods, LINK-KG effectively handles long-range dependencies, plural and ambiguous references, and role-shifting entities. Evaluated on real-world human smuggling case documents, LINK-KG produces cleaner, more coherent graph structures than existing baselines, supporting advanced downstream analysis such as group detection, role attribution, temporal modeling, and event prediction from legal narratives.
