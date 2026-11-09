
"""
Data Preprocessing Pipeline for Narrative Dataset Curation
===========================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    1. Load raw narrative datasets in JSONL format.
    2. Extract prompts and responses.
    3. Generate numerical text embeddings.
    4. Construct sample-level feature representations.
    5. Organize relevance and compatibility annotations.
    6. Load semantic similarity and consistency annotations.
    7. Generate token-level embeddings for Transformer processing.
    8. Construct the complete NPZ dataset expected by dataset_curator.py.

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Output:
    post_training_dataset.npz

Requirements:
    pip install torch numpy transformers

Author:
    PlotDot-LLM Project
"""

import os
import json
import argparse

from typing import Dict, List, Tuple

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. JSONL DATA LOADING
# ============================================================

def load_jsonl(path: str) -> List[Dict]:

    """Load records from a JSONL file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}."
        )

    return records


# ============================================================
# 3. BINARY LABEL VALIDATION
# ============================================================

def validate_binary_label(
    value,
    field_name: str
) -> int:

    """Validate an annotation containing either 0 or 1."""

    if isinstance(value, bool):
        return int(value)

    if value not in (0, 1):
        raise ValueError(
            f"{field_name} must be either 0 or 1."
        )

    return int(value)


# ============================================================
# 4. RAW SAMPLE VALIDATION
# ============================================================

def validate_samples(
    records: List[Dict]
) -> Dict[str, np.ndarray]:

    """
    Extract and validate raw narrative samples.

    Required fields:

        sample_id
        prompt
        response
        relevance_label
        compatibility_label
    """

    sample_ids = []
    prompts = []
    responses = []

    relevance_labels = []
    compatibility_labels = []

    seen_ids = set()

    for index, record in enumerate(records):

        required_fields = [
            "sample_id",
            "prompt",
            "response",
            "relevance_label",
            "compatibility_label"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing_fields:

            raise ValueError(
                f"Sample {index} is missing fields: "
                f"{missing_fields}"
            )

        sample_id = str(
            record["sample_id"]
        )

        prompt = record["prompt"]
        response = record["response"]

        if not sample_id.strip():
            raise ValueError(
                f"Sample {index} has an empty sample_id."
            )

        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample_id: {sample_id}"
            )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Invalid prompt in sample {sample_id}."
            )

        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                f"Invalid response in sample {sample_id}."
            )

        relevance = validate_binary_label(
            record["relevance_label"],
            "relevance_label"
        )

        compatibility = validate_binary_label(
            record["compatibility_label"],
            "compatibility_label"
        )

        sample_ids.append(sample_id)

        prompts.append(prompt)

        responses.append(response)

        relevance_labels.append(relevance)

        compatibility_labels.append(compatibility)

        seen_ids.add(sample_id)

    relevance_labels = np.asarray(
        relevance_labels,
        dtype=np.float32
    )

    compatibility_labels = np.asarray(
        compatibility_labels,
        dtype=np.float32
    )

    for name, labels in [
        ("relevance_label", relevance_labels),
        ("compatibility_label", compatibility_labels)
    ]:

        counts = np.bincount(
            labels.astype(np.int64),
            minlength=2
        )

        if np.any(counts < 2):
            raise ValueError(
                f"{name} requires at least two samples from "
                "each class (0 and 1) for stratified training "
                "and validation."
            )

    return {

        "sample_ids": np.asarray(
            sample_ids,
            dtype=str
        ),

        "prompts": prompts,

        "responses": responses,

        "relevance_labels": relevance_labels,

        "compatibility_labels": compatibility_labels

    }


# ============================================================
# 5. PAIR ANNOTATION LOADING
# ============================================================

def load_pair_annotations(
    path: str,
    sample_ids: np.ndarray
) -> Dict[str, np.ndarray]:

    """
    Load annotations connecting narrative samples.

    Required fields:

        sample_id_a
        sample_id_b
        similarity_label
        consistency_label

    similarity_label:
        1 = Semantically similar.
        0 = Semantically dissimilar.

    consistency_label:
        1 = Narratively consistent.
        0 = Narratively inconsistent.
    """

    records = load_jsonl(path)

    id_to_index = {
        sample_id: index
        for index, sample_id in enumerate(sample_ids)
    }

    siamese_pairs = []
    consistency_pairs = []

    similarity_labels = []
    consistency_labels = []

    seen_pairs = set()

    for record_index, record in enumerate(records):

        required_fields = [
            "sample_id_a",
            "sample_id_b",
            "similarity_label",
            "consistency_label"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing_fields:

            raise ValueError(
                f"Pair annotation {record_index} is missing "
                f"fields: {missing_fields}"
            )

        id_a = str(
            record["sample_id_a"]
        )

        id_b = str(
            record["sample_id_b"]
        )

        if id_a not in id_to_index:
            raise ValueError(
                f"Unknown sample_id_a: {id_a}"
            )

        if id_b not in id_to_index:
            raise ValueError(
                f"Unknown sample_id_b: {id_b}"
            )

        index_a = id_to_index[id_a]
        index_b = id_to_index[id_b]

        if index_a == index_b:
            raise ValueError(
                "A sample cannot be paired with itself."
            )

        pair_key = tuple(
            sorted((index_a, index_b))
        )

        if pair_key in seen_pairs:
            raise ValueError(
                f"Duplicate pair annotation: {id_a}, {id_b}"
            )

        similarity = validate_binary_label(
            record["similarity_label"],
            "similarity_label"
        )

        consistency = validate_binary_label(
            record["consistency_label"],
            "consistency_label"
        )

        siamese_pairs.append([
            index_a,
            index_b,
            similarity
        ])

        consistency_pairs.append([
            index_a,
            index_b
        ])

        similarity_labels.append(
            similarity
        )

        consistency_labels.append(
            consistency
        )

        seen_pairs.add(pair_key)

    siamese_pairs = np.asarray(
        siamese_pairs,
        dtype=np.float32
    )

    consistency_pairs = np.asarray(
        consistency_pairs,
        dtype=np.int64
    )

    similarity_labels = np.asarray(
        similarity_labels,
        dtype=np.float32
    )

    consistency_labels = np.asarray(
        consistency_labels,
        dtype=np.float32
    )

    if len(siamese_pairs) < 4:
        raise ValueError(
            "At least four annotated pairs are required."
        )

    for name, labels in [
        ("similarity_label", similarity_labels),
        ("consistency_label", consistency_labels)
    ]:

        counts = np.bincount(
            labels.astype(np.int64),
            minlength=2
        )

        if np.any(counts < 2):
            raise ValueError(
                f"{name} requires at least two pairs from "
                "each class (0 and 1)."
            )

    return {

        "siamese_pairs": siamese_pairs,

        "consistency_pair_indices": consistency_pairs,

        "consistency_labels": consistency_labels

    }


# ============================================================
# 6. PRETRAINED TEXT EMBEDDING MODEL
# ============================================================

class TextEmbeddingGenerator:

    """
    Generate pooled and token-level representations using
    a pretrained Transformer language model.

    Pooled representations:
        Mean-pooled token embeddings.

    Token-level representations:
        Last hidden states used by the Transformer
        consistency model.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_seq_length: int = 256,
        device: str = None
    ):

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if max_seq_length < 2:
            raise ValueError(
                "max_seq_length must be at least 2."
            )

        if device is not None:

            self.device = torch.device(device)

        else:

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        self.batch_size = batch_size

        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        self.embedding_dim = (
            self.model.config.hidden_size
        )

    # --------------------------------------------------------
    # 6.1 MODEL INFERENCE
    # --------------------------------------------------------

    def _forward(
        self,
        encoded
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Extract last hidden states and attention masks.
        """

        model_inputs = {

            "input_ids": encoded["input_ids"].to(self.device),

            "attention_mask": encoded["attention_mask"].to(self.device)

        }

        if (
            "token_type_ids" in encoded
            and "token_type_ids" in self.model.forward.__code__.co_varnames
        ):

            model_inputs["token_type_ids"] = (
                encoded["token_type_ids"].to(self.device)
            )

        outputs = self.model(
            **model_inputs
        )

        return (
            outputs.last_hidden_state,
            model_inputs["attention_mask"]
        )

    # --------------------------------------------------------
    # 6.2 SINGLE-TEXT EMBEDDING
    # --------------------------------------------------------

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:

        """
        Generate mean-pooled embeddings for individual texts.

        Returns:
            (n_texts, embedding_dim)
        """

        if not texts:
            raise ValueError(
                "Cannot encode an empty text collection."
            )

        all_embeddings = []

        for start in range(
            0,
            len(texts),
            self.batch_size
        ):

            batch_texts = texts[
                start:start + self.batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            hidden_states, attention_mask = self._forward(
                encoded
            )

            mask = attention_mask.unsqueeze(-1).to(
                hidden_states.dtype
            )

            pooled = (
                (hidden_states * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )

            all_embeddings.append(
                pooled.cpu().numpy()
            )

        return np.concatenate(
            all_embeddings,
            axis=0
        ).astype(np.float32)

    # --------------------------------------------------------
    # 6.3 TOKEN-LEVEL NARRATIVE EMBEDDINGS
    # --------------------------------------------------------

    @torch.inference_mode()
    def encode_narratives(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Encode complete prompt-response narratives.

        The tokenizer processes each prompt and its corresponding
        response as one paired input.

        Returns:

            token_embeddings:
                (n_samples, max_seq_length, embedding_dim)

            attention_masks:
                (n_samples, max_seq_length)
        """

        if len(prompts) != len(responses):
            raise ValueError(
                "Prompt and response collections must have equal lengths."
            )

        if not prompts:
            raise ValueError(
                "Narrative collection cannot be empty."
            )

        all_embeddings = []
        all_masks = []

        for start in range(
            0,
            len(prompts),
            self.batch_size
        ):

            batch_prompts = prompts[
                start:start + self.batch_size
            ]

            batch_responses = responses[
                start:start + self.batch_size
            ]

            encoded = self.tokenizer(

                batch_prompts,

                batch_responses,

                padding="max_length",

                truncation=True,

                max_length=self.max_seq_length,

                return_tensors="pt"

            )

            hidden_states, attention_mask = self._forward(
                encoded
            )

            all_embeddings.append(
                hidden_states.cpu().numpy()
            )

            all_masks.append(
                attention_mask.cpu().numpy()
            )

        token_embeddings = np.concatenate(
            all_embeddings,
            axis=0
        ).astype(np.float32)

        attention_masks = np.concatenate(
            all_masks,
            axis=0
        ).astype(np.int64)

        return token_embeddings, attention_masks


# ============================================================
# 7. COMPLETE PREPROCESSING PIPELINE
# ============================================================

class NarrativeDataPreprocessor:

    """
    Complete preprocessing pipeline.

    Pipeline:

        Raw JSONL
            |
            v
        Data Validation
            |
            v
        Prompt and Response Extraction
            |
            v
        Text Embedding Generation
            |
            v
        Token-Level Embedding Generation
            |
            v
        Pair Annotation Mapping
            |
            v
        NPZ Dataset Construction
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_seq_length: int = 256,
        device: str = None,
        seed: int = 42
    ):

        self.seed = seed

        set_seed(seed)

        self.embedding_generator = TextEmbeddingGenerator(

            model_name=model_name,

            batch_size=batch_size,

            max_seq_length=max_seq_length,

            device=device

        )

    # --------------------------------------------------------
    # 7.1 LOAD RAW DATA
    # --------------------------------------------------------

    def load_raw_dataset(
        self,
        input_path: str
    ) -> Dict[str, np.ndarray]:

        records = load_jsonl(
            input_path
        )

        return validate_samples(
            records
        )

    # --------------------------------------------------------
    # 7.2 GENERATE EMBEDDINGS
    # --------------------------------------------------------

    def generate_embeddings(
        self,
        samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        prompts = samples["prompts"]

        responses = samples["responses"]

        print("Generating prompt embeddings...")

        prompt_embeddings = (
            self.embedding_generator.encode_texts(
                prompts
            )
        )

        print("Generating response embeddings...")

        response_embeddings = (
            self.embedding_generator.encode_texts(
                responses
            )
        )

        print("Generating token-level narrative embeddings...")

        narrative_embeddings, narrative_masks = (
            self.embedding_generator.encode_narratives(
                prompts,
                responses
            )
        )

        # Concatenate prompt and response representations
        # to construct one feature vector per sample.
        features = np.concatenate(
            [
                prompt_embeddings,
                response_embeddings
            ],
            axis=1
        ).astype(np.float32)

        return {

            "features": features,

            "prompt_embeddings": prompt_embeddings,

            "response_embeddings": response_embeddings,

            "narrative_embeddings": narrative_embeddings,

            "narrative_masks": narrative_masks

        }

    # --------------------------------------------------------
    # 7.3 MAP PAIR-LEVEL EMBEDDINGS
    # --------------------------------------------------------

    @staticmethod
    def construct_pair_embeddings(
        narrative_embeddings: np.ndarray,
        narrative_masks: np.ndarray,
        consistency_pair_indices: np.ndarray
    ) -> Dict[str, np.ndarray]:

        """
        Construct sequence-level embedding pairs for the
        Transformer Encoder using original sample indices.
        """

        index_a = consistency_pair_indices[:, 0]

        index_b = consistency_pair_indices[:, 1]

        return {

            "consistency_embeddings_a":
                narrative_embeddings[index_a],

            "consistency_embeddings_b":
                narrative_embeddings[index_b],

            "consistency_attention_mask_a":
                narrative_masks[index_a],

            "consistency_attention_mask_b":
                narrative_masks[index_b]

        }

    # --------------------------------------------------------
    # 7.4 BUILD FINAL NPZ DATASET
    # --------------------------------------------------------

    def build_dataset(
        self,
        input_path: str,
        pair_annotations_path: str
    ) -> Dict[str, np.ndarray]:

        """
        Construct the complete dataset expected by dataset_curator.py.
        """

        print("\nLoading raw narrative dataset...")

        samples = self.load_raw_dataset(
            input_path
        )

        print(
            f"Loaded {len(samples['sample_ids'])} samples."
        )

        print("\nLoading pair annotations...")

        pair_data = load_pair_annotations(

            pair_annotations_path,

            samples["sample_ids"]

        )

        print(
            f"Loaded {len(pair_data['siamese_pairs'])} annotated pairs."
        )

        print("\nGenerating numerical representations...")

        embeddings = self.generate_embeddings(
            samples
        )

        # ----------------------------------------------------
        # Map original prompt-response records
        # ----------------------------------------------------

        n_samples = len(
            samples["sample_ids"]
        )

        prompt_response_sample_indices = np.column_stack([
            np.arange(n_samples),
            np.arange(n_samples)
        ]).astype(np.int64)

        # ----------------------------------------------------
        # Construct Transformer sequence pairs
        # ----------------------------------------------------

        consistency_data = self.construct_pair_embeddings(

            narrative_embeddings=embeddings["narrative_embeddings"],

            narrative_masks=embeddings["narrative_masks"],

            consistency_pair_indices=pair_data[
                "consistency_pair_indices"
            ]

        )

        # ----------------------------------------------------
        # Assemble final dataset
        # ----------------------------------------------------

        dataset = {

            "features":
                embeddings["features"],

            "relevance_labels":
                samples["relevance_labels"],

            "siamese_pairs":
                pair_data["siamese_pairs"],

            "prompt_embeddings":
                embeddings["prompt_embeddings"],

            "response_embeddings":
                embeddings["response_embeddings"],

            "prompt_response_labels":
                samples["compatibility_labels"],

            "prompt_response_sample_indices":
                prompt_response_sample_indices,

            "consistency_embeddings_a":
                consistency_data["consistency_embeddings_a"],

            "consistency_embeddings_b":
                consistency_data["consistency_embeddings_b"],

            "consistency_attention_mask_a":
                consistency_data["consistency_attention_mask_a"],

            "consistency_attention_mask_b":
                consistency_data["consistency_attention_mask_b"],

            "consistency_labels":
                pair_data["consistency_labels"],

            "consistency_pair_indices":
                pair_data["consistency_pair_indices"],

            "sample_ids":
                samples["sample_ids"]

        }

        return dataset

    # --------------------------------------------------------
    # 7.5 SAVE PREPROCESSED DATASET
    # --------------------------------------------------------

    def save_dataset(
        self,
        dataset: Dict[str, np.ndarray],
        output_path: str
    ) -> None:

        """
        Save every preprocessing output into a compressed NPZ archive.
        """

        os.makedirs(
            os.path.dirname(
                os.path.abspath(output_path)
            ),
            exist_ok=True
        )

        np.savez_compressed(
            output_path,
            **dataset
        )

        print("\nPreprocessing completed.")

        print(f"Output: {output_path}")

        print("\nGenerated arrays:")

        for name, array in dataset.items():

            print(
                f"{name:40s} {array.shape}"
            )

    # --------------------------------------------------------
    # 7.6 COMPLETE EXECUTION
    # --------------------------------------------------------

    def run(
        self,
        input_path: str,
        pair_annotations_path: str,
        output_path: str
    ) -> Dict[str, np.ndarray]:

        dataset = self.build_dataset(

            input_path=input_path,

            pair_annotations_path=pair_annotations_path

        )

        self.save_dataset(
            dataset,
            output_path
        )

        return dataset


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw narrative data into numerical "
            "representations for the PlotDot dataset curation pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw narrative JSONL file."
    )

    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to the annotated sample-pairs JSONL file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/post_training_dataset.npz",
        help="Output NPZ dataset path."
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # Initialize preprocessing pipeline.
    preprocessor = NarrativeDataPreprocessor(

        model_name=args.embedding_model,

        batch_size=args.batch_size,

        max_seq_length=args.max_seq_length,

        device=args.device,

        seed=args.seed

    )

    # Execute preprocessing.
    preprocessor.run(

        input_path=args.input,

        pair_annotations_path=args.pairs,

        output_path=args.output

    )


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
