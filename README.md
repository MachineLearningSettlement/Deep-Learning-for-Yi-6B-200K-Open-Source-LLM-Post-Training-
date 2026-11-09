# Deep Learning Framework for Post-Training the Yi-6B-200K Open-Source LLM

This repository provides a modular framework for preparing narrative datasets, curating training examples, and post-training **Yi-6B-200K** with supervised fine-tuning (SFT) and continual learning.

The framework has two main parts:

1. **Data curation:** learns sample representations and evaluates relevance, prompt-response compatibility, redundancy, diversity, and narrative consistency.
2. **Model post-training:** fine-tunes Yi-6B-200K with LoRA. The continual-learning trainer uses replay to mix narrative examples with examples representing previously acquired capabilities, and adapts the replay ratio using evaluation metrics.

The default model identifier is [`01-ai/Yi-6B-200K`](https://huggingface.co/01-ai/Yi-6B-200K). The model and tokenizer are loaded through Hugging Face Transformers. The training scripts use a default maximum sequence length of 2,048 tokens; this can be changed with `--max-seq-length`.

## Repository structure

```text
Deep-Learning-for-Yi-6B-200K-Open-Source-LLM-Post-Training/
│
├── data/
│   └── dataset_curation/
│       ├── data_preprocessor.py
│       ├── dataset_curator.py
│       ├── autoencoder.py
│       ├── siamese_network.py
│       ├── variational_autoencoder.py
│       ├── relevance_mlp.py
│       └── transformer_encoder.py
│
├── training/
│   ├── sft_trainer.py
│   └── continual_learning.py
│
├── evaluation/
│   ├── forgetting_metrics.py
│   ├── stability_metrics.py
│   └── plasticity_metrics.py
│
├── docs/
│   └── yi_model_setup_guide.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Environment setup

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies from the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `bitsandbytes` dependency is optional and is only needed when using the 4-bit loading option. Install it separately when required:

```bash
pip install bitsandbytes
```

A CUDA-capable GPU is recommended for model training. The scripts expose device and precision options; the model's memory requirements depend on the selected precision, sequence length, and batch configuration.

## 2. Dataset preparation and curation

### 2.1 Raw narrative dataset

Create a JSONL file in which each line is a JSON object with the following fields:

```json
{"sample_id":"S001","prompt":"Write the opening scene of a mystery story.","response":"The old station clock stopped at midnight...","relevance_label":1,"compatibility_label":1}
```

Required fields:

| Field | Description |
|---|---|
| `sample_id` | Unique identifier for the sample. |
| `prompt` | Narrative instruction or prompt. |
| `response` | Corresponding response or narrative continuation. |
| `relevance_label` | Binary relevance annotation (`0` or `1`). |
| `compatibility_label` | Binary prompt-response compatibility annotation (`0` or `1`). |

The preprocessing script requires both classes (`0` and `1`) to be represented in the relevance and compatibility labels, with at least two samples per class. More examples are needed for meaningful training and evaluation.

### 2.2 Pair annotations

Create a second JSONL file containing pairwise annotations:

```json
{"sample_id_a":"S001","sample_id_b":"S002","similarity_label":1,"consistency_label":0}
```

Required fields:

| Field | Description |
|---|---|
| `sample_id_a`, `sample_id_b` | IDs of two different samples in the raw narrative dataset. |
| `similarity_label` | `1` if the samples are semantically similar; otherwise `0`. |
| `consistency_label` | `1` if the pair is narratively consistent; otherwise `0`. |

The pair annotations must reference existing sample IDs. The preprocessing pipeline requires at least four unique pairs and at least two pairs in each class for both labels.

### 2.3 Generate the preprocessing archive

`data_preprocessor.py` validates the records, creates text embeddings using `sentence-transformers/all-MiniLM-L6-v2`, constructs sample and pair representations, and saves the arrays expected by `dataset_curator.py` in a compressed NPZ archive.

Run from the repository root:

```bash
python data/dataset_curation/data_preprocessor.py \
  --input data/raw/narrative_dataset.jsonl \
  --pairs data/raw/pair_annotations.jsonl \
  --output data/processed/post_training_dataset.npz
```

The embedding model is downloaded through Transformers on first use unless it is already available in the local cache. You can override it with `--embedding-model`, and configure `--batch-size`, `--max-seq-length`, and `--device`.

### 2.4 Run dataset curation

Run the curation pipeline on the generated NPZ archive:

```bash
python data/dataset_curation/dataset_curator.py \
  --input data/processed/post_training_dataset.npz \
  --output-dir outputs/dataset_curation
```

The curator runs the Autoencoder, Siamese Network, Variational Autoencoder, relevance MLP, prompt-response compatibility MLP, and Transformer Encoder. It aggregates their outputs into a sample-level curation score. The default curation threshold is `0.5` and can be changed with `--curation-threshold`.

Important outputs are saved under:

```text
outputs/dataset_curation/
└── curation/
    ├── curated_sample_indices.npy
    ├── review_sample_indices.npy
    ├── curated_feature_matrix.npy
    ├── curated_sample_ids.npy
    ├── curation_scores.npy
    ├── sample_curation_results.csv
    └── curation_report.json
```

The `curated_sample_indices.npy` file contains indices into the original raw JSONL dataset. Keep the original JSONL file and its record order unchanged when passing these indices to the SFT trainer.

## 3. Supervised fine-tuning

`sft_trainer.py` loads the base Yi-6B-200K model, applies LoRA, and performs SFT on prompt-response records. Prompt tokens are masked from the language-model loss so that the response tokens provide the training targets. The trainer includes validation, early stopping, gradient accumulation, mixed precision, optional 4-bit loading, and adapter saving.

To run baseline SFT using all records:

```bash
python training/sft_trainer.py \
  --dataset data/raw/narrative_dataset.jsonl \
  --output-dir outputs/sft
```

To train only on the curated examples:

```bash
python training/sft_trainer.py \
  --dataset data/raw/narrative_dataset.jsonl \
  --curated-indices outputs/dataset_curation/curation/curated_sample_indices.npy \
  --output-dir outputs/sft_curated
```

The SFT dataset must contain `prompt` and `response` fields. The `sample_id` field is optional for training. By default, the trainer uses LoRA rank `16`, alpha `32`, dropout `0.05`, learning rate `2e-4`, batch size `1`, gradient accumulation `16`, and `3` epochs. See `python training/sft_trainer.py --help` for all options.

## 4. Continual learning with adaptive replay during SFT

`continual_learning.py` implements the continual-learning training loop. It imports and reuses the SFT components from `sft_trainer.py` (model and tokenizer loading, LoRA setup, SFT dataset, data collator, JSONL loading, and seed initialization), then runs its own training loop with a mixture of narrative and retention examples.

**For continual SFT, run `continual_learning.py` directly. Do not run `sft_trainer.py` as a separate prerequisite.** The continual-learning script initializes the base model and performs SFT with replay in the same training process.

Prepare four JSONL files. Each line must contain at least `prompt` and `response`:

- `narrative_train.jsonl`: narrative examples used for adaptation.
- `retention_train.jsonl`: examples representing skills to retain, including logical-reasoning tasks when preservation of reasoning is being evaluated.
- `narrative_eval.jsonl`: held-out narrative examples for measuring adaptation.
- `retention_eval.jsonl`: held-out retention examples for measuring capability preservation.

Example record:

```json
{"prompt":"Solve the following reasoning problem...","response":"First, identify the relevant constraints..."}
```

Run from the repository root:

```bash
python training/continual_learning.py \
  --narrative-train data/continual/narrative_train.jsonl \
  --retention-train data/continual/retention_train.jsonl \
  --narrative-eval data/continual/narrative_eval.jsonl \
  --retention-eval data/continual/retention_eval.jsonl \
  --output-dir outputs/continual_learning
```

The default initial replay ratio is `0.25`. The trainer adjusts the ratio within the configured minimum and maximum bounds according to the evaluation signals. It monitors the following metrics:

- **Capability Deviation:** relative deterioration in retention evaluation loss compared with the initial retention baseline. Lower is better.
- **Stability:** consistency of capability deviation between consecutive rounds. Higher is better.
- **Plasticity:** relative reduction in narrative evaluation loss compared with the initial narrative baseline. Higher is better.

The training loop uses these operational metric definitions to adjust replay and check convergence. Their values are evaluation signals; preservation of logical reasoning and successful adaptation must be established by evaluating suitable held-out datasets.

By default, the trainer allows up to `10` rounds and requires the configured convergence conditions to be satisfied for `2` consecutive rounds. See `python training/continual_learning.py --help` for all replay, metric-threshold, LoRA, optimization, and runtime options.

## 5. Evaluation modules

The `evaluation/` directory also provides standalone implementations of the metrics used by the continual-learning workflow:

- `forgetting_metrics.py` — Capability Deviation.
- `stability_metrics.py` — Stability.
- `plasticity_metrics.py` — Plasticity.

`continual_learning.py` currently contains its own implementations of these calculations. The standalone files are optional utilities and do not need to be run separately for continual-learning training.

## 6. Model setup guide

See [`docs/yi_model_setup_guide.pdf`](docs/yi_model_setup_guide.pdf) for the local Yi-6B-200K model setup instructions. By default, the training scripts use the Hugging Face model identifier `01-ai/Yi-6B-200K`; a local model directory can be supplied through `--model-name-or-path`.

## 7. Outputs

Training outputs are written to the selected output directories. The SFT trainer saves the LoRA adapter, tokenizer, training history, and training configuration. The continual-learning trainer saves the final adapter, tokenizer, continual-learning configuration, and round-by-round metric history.

Generated datasets, model checkpoints, and training outputs should be stored outside version control unless there is a specific reason to include them.

## 8. Reproducibility

Both training scripts and the data pipelines expose seed options. The default seed is `42`. Record the dataset versions, model identifier, command-line arguments, software versions, and hardware configuration when running experiments.

This repository provides the implementation framework. Training results, convergence, and capability preservation depend on the datasets, annotations, compute environment, and actual execution; no experimental result is implied by the presence of the scripts alone.
