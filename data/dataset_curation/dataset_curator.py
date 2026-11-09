
"""
Unified Data-Centric Dataset Curation Pipeline
===============================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    Coordinate five deep learning architectures for dataset curation:

    1. Autoencoder:
       Sample representation learning and redundancy detection.

    2. Siamese Network:
       Semantic similarity and duplicate detection.

    3. Variational Autoencoder:
       Latent-space diversity analysis.

    4. Multi-Layer Perceptron:
       Sample relevance and prompt-response compatibility scoring.

    5. Transformer Encoder:
       Semantic relationship modeling and narrative consistency.

Pipeline:
    Input Dataset
         |
         v
    Feature Representations
         |
         +--------------------------+
         |                          |
         v                          v
     Autoencoder              Siamese Network
         |                          |
         v                          v
    Representations            Similarity Scores
    Redundancy Scores          Duplicate Candidates
         |                          |
         +------------+-------------+
                      |
                      v
              Variational Autoencoder
                      |
                      v
                Diversity Scores
                      |
           +----------+----------+
           |                     |
           v                     v
       Relevance MLP       Prompt-Response MLP
           |                     |
           v                     v
      Relevance Scores    Compatibility Scores
           |                     |
           +----------+----------+
                      |
                      v
              Transformer Encoder
                      |
                      v
             Narrative Consistency
                      |
                      v
             Score Aggregation
                      |
                      v
              Dataset Curation
                      |
                      v
             Curated Sample Indices
               and Feature Matrix

Input:
    One NPZ file containing:

        features:
            (N, D)

        relevance_labels:
            (N,)

        siamese_pairs:
            (P, 3)
            Columns: sample_i, sample_j, label.

        prompt_embeddings:
            (M, E)

        response_embeddings:
            (M, E)

        prompt_response_labels:
            (M,)

        prompt_response_sample_indices:
            (M, 2)
            Columns: prompt_sample_index, response_sample_index.

        consistency_embeddings_a:
            (Q, L1, T)

        consistency_embeddings_b:
            (Q, L2, T)

        consistency_attention_mask_a:
            (Q, L1)

        consistency_attention_mask_b:
            (Q, L2)

        consistency_labels:
            (Q,)

        consistency_pair_indices:
            (Q, 2)
            Columns: sample_i, sample_j.

        sample_ids:
            (N,) [Optional]

Output:
    - Five trained model checkpoints.
    - Latent representations.
    - Reconstruction errors.
    - Redundancy candidates.
    - Diversity scores.
    - Relevance scores.
    - Prompt-response compatibility scores.
    - Narrative consistency scores.
    - Unified sample-level curation scores.
    - Curated sample indices and feature matrix.

Requirements:
    pip install torch numpy

Author:
    PlotDot-LLM Project
"""

import os
import csv
import json
import argparse

from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from .autoencoder import AutoencoderPipeline

    from .siamese_network import SiamesePipeline

    from .variational_autoencoder import VAEPipeline

    from .relevance_mlp import (
        MLPScoringPipeline,
        build_prompt_response_features
    )

    from .transformer_encoder import TransformerConsistencyPipeline

except ImportError:
    from autoencoder import AutoencoderPipeline

    from siamese_network import SiamesePipeline

    from variational_autoencoder import VAEPipeline

    from relevance_mlp import (
        MLPScoringPipeline,
        build_prompt_response_features
    )

    from transformer_encoder import TransformerConsistencyPipeline


# ============================================================
# 1. VALIDATION AND GENERAL UTILITIES
# ============================================================

def validate_binary_labels(
    labels: np.ndarray,
    expected_size: int,
    name: str
) -> np.ndarray:

    labels = np.asarray(
        labels,
        dtype=np.float32
    ).reshape(-1)

    if len(labels) != expected_size:
        raise ValueError(
            f"{name}: expected {expected_size} labels, "
            f"received {len(labels)}."
        )

    if not np.isfinite(labels).all():
        raise ValueError(
            f"{name} contains invalid values."
        )

    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(
            f"{name} must contain binary labels."
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            f"{name} must contain both classes: 0 and 1."
        )

    return labels


def validate_sample_indices(
    indices: np.ndarray,
    n_samples: int,
    name: str
) -> np.ndarray:

    indices = np.asarray(indices)

    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (n_pairs, 2)."
        )

    if not np.isfinite(indices).all():
        raise ValueError(
            f"{name} contains invalid values."
        )

    if not np.all(indices == np.floor(indices)):
        raise ValueError(
            f"{name} must contain integer indices."
        )

    indices = indices.astype(np.int64)

    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError(
            f"{name} contains indices outside the dataset."
        )

    return indices


def parse_dimensions(value: str) -> List[int]:
    """Parse comma-separated hidden layer dimensions."""

    dimensions = [
        int(item.strip())
        for item in value.split(",")
    ]

    if not dimensions or any(dim < 1 for dim in dimensions):
        raise ValueError(
            "Hidden dimensions must be positive integers."
        )

    return dimensions


def normalize_minmax(
    values: np.ndarray
) -> np.ndarray:

    """
    Normalize sample-level diversity scores to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    minimum = np.min(values)
    maximum = np.max(values)

    if np.isclose(maximum, minimum):
        return np.full(
            len(values),
            0.5,
            dtype=np.float64
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def save_json(
    path: str,
    content: dict
) -> None:

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            content,
            file,
            indent=4,
            allow_nan=False
        )


def save_array(
    path: str,
    array: np.ndarray
) -> None:

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True
    )

    np.save(
        path,
        array
    )


# ============================================================
# 2. REDUNDANCY SCORE AGGREGATION
# ============================================================

def build_redundancy_profile(
    n_samples: int,
    redundant_pairs: List[Tuple[int, int, float]]
) -> np.ndarray:

    """
    Calculate maximum detected similarity for each sample.

    Each redundant pair contributes its similarity score
    to both participating samples.

    Returns:
        Per-sample redundancy scores.
    """

    redundancy = np.zeros(
        n_samples,
        dtype=np.float32
    )

    for sample_i, sample_j, similarity in redundant_pairs:

        similarity = float(
            np.clip(similarity, 0.0, 1.0)
        )

        redundancy[sample_i] = max(
            redundancy[sample_i],
            similarity
        )

        redundancy[sample_j] = max(
            redundancy[sample_j],
            similarity
        )

    return redundancy


# ============================================================
# 3. PAIR-LEVEL TO SAMPLE-LEVEL SCORE AGGREGATION
# ============================================================

def aggregate_pair_scores(
    n_samples: int,
    pair_indices: np.ndarray,
    pair_scores: np.ndarray,
    endpoints: Tuple[int, ...] = (0, 1)
) -> np.ndarray:

    """
    Aggregate pair-level predictions into sample-level scores.

    Each sample receives the mean score of the pairs in which
    it participates.

    endpoints:
        (0, 1): Assign scores to both samples.
        (1,): Assign scores only to the second sample.
    """

    score_sum = np.zeros(
        n_samples,
        dtype=np.float64
    )

    score_count = np.zeros(
        n_samples,
        dtype=np.int64
    )

    for pair_index, score in zip(
        pair_indices,
        pair_scores
    ):

        for endpoint in endpoints:

            sample_index = int(
                pair_index[endpoint]
            )

            score_sum[sample_index] += float(score)

            score_count[sample_index] += 1

    result = np.full(
        n_samples,
        np.nan,
        dtype=np.float64
    )

    valid = score_count > 0

    result[valid] = (
        score_sum[valid]
        / score_count[valid]
    )

    return result


# ============================================================
# 4. UNIFIED CURATION SCORE
# ============================================================

def calculate_curation_scores(
    relevance_scores: np.ndarray,
    compatibility_scores: np.ndarray,
    diversity_scores: np.ndarray,
    autoencoder_redundancy: np.ndarray,
    siamese_redundancy: np.ndarray,
    consistency_scores: np.ndarray
) -> Tuple[np.ndarray, Dict]:

    """
    Calculate final sample-level curation scores.

    Higher values indicate stronger retention priority.

    Components:

        Relevance:
            Measures the predicted relevance of a sample.

        Compatibility:
            Measures prompt-response compatibility.

        Diversity:
            Measures relative latent-space diversity.

        Autoencoder distinctiveness:
            Penalizes samples identified as redundant.

        Siamese distinctiveness:
            Penalizes semantically redundant samples.

        Narrative consistency:
            Measures consistency with paired narrative samples.

    Missing pair-level scores are excluded from the corresponding
    sample's weighted aggregation.
    """

    weights = {

        "relevance": 0.30,

        "compatibility": 0.20,

        "diversity": 0.15,

        "autoencoder_distinctiveness": 0.10,

        "siamese_distinctiveness": 0.10,

        "narrative_consistency": 0.15

    }

    signals = {

        "relevance": relevance_scores,

        "compatibility": compatibility_scores,

        "diversity": diversity_scores,

        "autoencoder_distinctiveness":
            1.0 - autoencoder_redundancy,

        "siamese_distinctiveness":
            1.0 - siamese_redundancy,

        "narrative_consistency": consistency_scores

    }

    n_samples = len(
        relevance_scores
    )

    weighted_sum = np.zeros(
        n_samples,
        dtype=np.float64
    )

    available_weight = np.zeros(
        n_samples,
        dtype=np.float64
    )

    coverage = {}

    for name, values in signals.items():

        values = np.asarray(
            values,
            dtype=np.float64
        )

        valid = np.isfinite(values)

        weighted_sum[valid] += (
            values[valid] * weights[name]
        )

        available_weight[valid] += weights[name]

        coverage[name] = float(
            np.mean(valid)
        )

    scores = np.divide(
        weighted_sum,
        available_weight,
        out=np.zeros_like(weighted_sum),
        where=available_weight > 0
    )

    scores = np.clip(
        scores,
        0.0,
        1.0
    )

    return scores, {
        "weights": weights,
        "signal_coverage": coverage
    }


# ============================================================
# 5. REDUNDANCY CANDIDATE EXPORT
# ============================================================

def export_redundancy_candidates(
    path: str,
    pairs: List[Tuple[int, int, float]],
    sample_ids: np.ndarray
) -> None:

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "sample_id_1",
            "sample_id_2",
            "cosine_similarity"
        ])

        for sample_i, sample_j, similarity in pairs:

            writer.writerow([
                str(sample_ids[sample_i]),
                str(sample_ids[sample_j]),
                float(similarity)
            ])


# ============================================================
# 6. MAIN DATASET CURATOR
# ============================================================

class DatasetCurator:
    """
    Orchestrates all five architectures into one dataset
    curation framework.

    The pipeline:

        1. Learns sample representations using an Autoencoder.
        2. Detects semantic redundancy using a Siamese Network.
        3. Analyzes latent-space diversity using a VAE.
        4. Scores sample relevance using an MLP.
        5. Scores prompt-response compatibility using an MLP.
        6. Evaluates narrative consistency using a Transformer.
        7. Aggregates the model outputs.
        8. Exports curated sample indices and feature vectors.
    """

    def __init__(self, args):

        self.args = args

        self.output_dir = args.output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.device = args.device

        self.sample_ids = None

        self.component_results = {}

    # --------------------------------------------------------
    # 6.1 LOAD AND VALIDATE DATA
    # --------------------------------------------------------

    def validate_dataset(
        self,
        data: Dict
    ) -> Dict:

        required_fields = {

            "features",

            "relevance_labels",

            "siamese_pairs",

            "prompt_embeddings",

            "response_embeddings",

            "prompt_response_labels",

            "prompt_response_sample_indices",

            "consistency_embeddings_a",

            "consistency_embeddings_b",

            "consistency_attention_mask_a",

            "consistency_attention_mask_b",

            "consistency_labels",

            "consistency_pair_indices"

        }

        missing_fields = (
            required_fields - set(data.keys())
        )

        if missing_fields:

            raise KeyError(
                f"Missing dataset fields: {sorted(missing_fields)}"
            )

        # ----------------------------------------------------
        # Global sample features
        # ----------------------------------------------------

        features = np.asarray(
            data["features"],
            dtype=np.float32
        )

        if features.ndim != 2:
            raise ValueError(
                "features must have shape (n_samples, input_dim)."
            )

        if features.shape[0] < 4:
            raise ValueError(
                "At least four samples are required."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "features contains invalid values."
            )

        n_samples = len(features)

        # ----------------------------------------------------
        # Sample-level relevance labels
        # ----------------------------------------------------

        relevance_labels = validate_binary_labels(
            data["relevance_labels"],
            n_samples,
            "relevance_labels"
        )

        # ----------------------------------------------------
        # Siamese labeled pairs
        # ----------------------------------------------------

        siamese_pairs = np.asarray(
            data["siamese_pairs"]
        )

        if siamese_pairs.ndim != 2 or siamese_pairs.shape[1] != 3:
            raise ValueError(
                "siamese_pairs must have shape (n_pairs, 3)."
            )

        if len(siamese_pairs) < 4:
            raise ValueError(
                "At least four Siamese pairs are required."
            )

        # Validate index columns and labels.
        siamese_indices = validate_sample_indices(
            siamese_pairs[:, :2],
            n_samples,
            "siamese_pairs"
        )

        siamese_labels = validate_binary_labels(
            siamese_pairs[:, 2],
            len(siamese_pairs),
            "siamese_pairs labels"
        )

        siamese_pairs = np.column_stack([
            siamese_indices,
            siamese_labels
        ]).astype(np.float32)

        # ----------------------------------------------------
        # Prompt-response embeddings
        # ----------------------------------------------------

        prompt_embeddings = np.asarray(
            data["prompt_embeddings"],
            dtype=np.float32
        )

        response_embeddings = np.asarray(
            data["response_embeddings"],
            dtype=np.float32
        )

        if prompt_embeddings.ndim != 2:
            raise ValueError(
                "prompt_embeddings must be a 2D matrix."
            )

        if response_embeddings.ndim != 2:
            raise ValueError(
                "response_embeddings must be a 2D matrix."
            )

        if prompt_embeddings.shape != response_embeddings.shape:
            raise ValueError(
                "Prompt and response embeddings must have identical shapes."
            )

        if not np.isfinite(prompt_embeddings).all():
            raise ValueError(
                "prompt_embeddings contains invalid values."
            )

        if not np.isfinite(response_embeddings).all():
            raise ValueError(
                "response_embeddings contains invalid values."
            )

        prompt_response_labels = validate_binary_labels(
            data["prompt_response_labels"],
            len(prompt_embeddings),
            "prompt_response_labels"
        )

        prompt_response_indices = validate_sample_indices(
            data["prompt_response_sample_indices"],
            n_samples,
            "prompt_response_sample_indices"
        )

        if len(prompt_response_indices) != len(prompt_embeddings):
            raise ValueError(
                "Prompt-response sample indices must align with embeddings."
            )

        # ----------------------------------------------------
        # Transformer sequence embeddings
        # ----------------------------------------------------

        transformer_a = np.asarray(
            data["consistency_embeddings_a"],
            dtype=np.float32
        )

        transformer_b = np.asarray(
            data["consistency_embeddings_b"],
            dtype=np.float32
        )

        mask_a = np.asarray(
            data["consistency_attention_mask_a"]
        )

        mask_b = np.asarray(
            data["consistency_attention_mask_b"]
        )

        if transformer_a.ndim != 3 or transformer_b.ndim != 3:
            raise ValueError(
                "Transformer embeddings must have shape (Q, L, T)."
            )

        if transformer_a.shape[0] != transformer_b.shape[0]:
            raise ValueError(
                "Both Transformer arrays must contain the same pair count."
            )

        if transformer_a.shape[2] != transformer_b.shape[2]:
            raise ValueError(
                "Both Transformer arrays must have the same embedding dimension."
            )

        if not np.isfinite(transformer_a).all():
            raise ValueError(
                "consistency_embeddings_a contains invalid values."
            )

        if not np.isfinite(transformer_b).all():
            raise ValueError(
                "consistency_embeddings_b contains invalid values."
            )

        consistency_labels = validate_binary_labels(
            data["consistency_labels"],
            len(transformer_a),
            "consistency_labels"
        )

        consistency_indices = validate_sample_indices(
            data["consistency_pair_indices"],
            n_samples,
            "consistency_pair_indices"
        )

        if len(consistency_indices) != len(transformer_a):
            raise ValueError(
                "Consistency pair indices must align with Transformer inputs."
            )

        # ----------------------------------------------------
        # Sample identifiers
        # ----------------------------------------------------

        if "sample_ids" in data:

            sample_ids = np.asarray(
                data["sample_ids"]
            ).reshape(-1)

            if len(sample_ids) != n_samples:
                raise ValueError(
                    "sample_ids must contain one ID per sample."
                )

        else:

            sample_ids = np.arange(
                n_samples
            ).astype(str)

        # ----------------------------------------------------
        # Validated dataset
        # ----------------------------------------------------

        validated_data = {

            "features": features,

            "relevance_labels": relevance_labels,

            "siamese_pairs": siamese_pairs,

            "prompt_embeddings": prompt_embeddings,

            "response_embeddings": response_embeddings,

            "prompt_response_labels": prompt_response_labels,

            "prompt_response_sample_indices": prompt_response_indices,

            "consistency_embeddings_a": transformer_a,

            "consistency_embeddings_b": transformer_b,

            "consistency_attention_mask_a": mask_a,

            "consistency_attention_mask_b": mask_b,

            "consistency_labels": consistency_labels,

            "consistency_pair_indices": consistency_indices,

            "sample_ids": sample_ids

        }

        return validated_data

    # --------------------------------------------------------
    # 6.2 TRAIN AUTOENCODER
    # --------------------------------------------------------

    def run_autoencoder(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 1: AUTOENCODER")

        print("=" * 75)

        X = data["features"]

        pipeline = AutoencoderPipeline(

            input_dim=X.shape[1],

            hidden_dims=self.args.ae_hidden_dims,

            latent_dim=self.args.ae_latent_dim,

            dropout=self.args.dropout,

            learning_rate=self.args.learning_rate,

            batch_size=self.args.batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            validation_split=self.args.validation_split,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(X)

        latent = pipeline.encode(X)

        reconstruction_errors = pipeline.reconstruction_error(
            X
        )

        redundant_pairs = pipeline.find_redundant_pairs(
            X,
            threshold=self.args.autoencoder_threshold
        )

        redundancy_profile = build_redundancy_profile(
            len(X),
            redundant_pairs
        )

        stage_dir = os.path.join(
            self.output_dir,
            "autoencoder"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        # Serialize preprocessing statistics as Python lists
        # for compatibility with the AutoencoderPipeline loader.
        pipeline.mean = np.asarray(
            pipeline.mean
        ).tolist()

        pipeline.std = np.asarray(
            pipeline.std
        ).tolist()

        pipeline.save(
            os.path.join(
                stage_dir,
                "autoencoder_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "latent_representations.npy"
            ),
            latent
        )

        save_array(
            os.path.join(
                stage_dir,
                "reconstruction_errors.npy"
            ),
            reconstruction_errors
        )

        save_array(
            os.path.join(
                stage_dir,
                "redundancy_profile.npy"
            ),
            redundancy_profile
        )

        export_redundancy_candidates(
            os.path.join(
                stage_dir,
                "redundancy_candidates.csv"
            ),
            redundant_pairs,
            data["sample_ids"]
        )

        print(
            f"Learned representations: {latent.shape}"
        )

        print(
            f"Redundant pairs: {len(redundant_pairs)}"
        )

        return {

            "latent_representations": latent,

            "reconstruction_errors": reconstruction_errors,

            "redundancy_profile": redundancy_profile,

            "redundant_pairs": redundant_pairs

        }

    # --------------------------------------------------------
    # 6.3 TRAIN SIAMESE NETWORK
    # --------------------------------------------------------

    def run_siamese(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 2: SIAMESE NETWORK")

        print("=" * 75)

        X = data["features"]

        pairs = data["siamese_pairs"]

        pipeline = SiamesePipeline(

            input_dim=X.shape[1],

            hidden_dims=self.args.siamese_hidden_dims,

            embedding_dim=self.args.siamese_embedding_dim,

            dropout=self.args.dropout,

            learning_rate=self.args.learning_rate,

            batch_size=self.args.batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            margin=self.args.margin,

            validation_split=self.args.validation_split,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(
            X,
            pairs
        )

        embeddings = pipeline.encode(X)

        redundant_pairs = pipeline.find_redundant_pairs(
            X,
            threshold=self.args.siamese_threshold
        )

        redundancy_profile = build_redundancy_profile(
            len(X),
            redundant_pairs
        )

        stage_dir = os.path.join(
            self.output_dir,
            "siamese_network"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        pipeline.save(
            os.path.join(
                stage_dir,
                "siamese_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "siamese_embeddings.npy"
            ),
            embeddings
        )

        save_array(
            os.path.join(
                stage_dir,
                "redundancy_profile.npy"
            ),
            redundancy_profile
        )

        export_redundancy_candidates(
            os.path.join(
                stage_dir,
                "redundancy_candidates.csv"
            ),
            redundant_pairs,
            data["sample_ids"]
        )

        print(
            f"Learned embeddings: {embeddings.shape}"
        )

        print(
            f"Redundant pairs: {len(redundant_pairs)}"
        )

        return {

            "embeddings": embeddings,

            "redundancy_profile": redundancy_profile,

            "redundant_pairs": redundant_pairs

        }

    # --------------------------------------------------------
    # 6.4 TRAIN VARIATIONAL AUTOENCODER
    # --------------------------------------------------------

    def run_vae(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 3: VARIATIONAL AUTOENCODER")

        print("=" * 75)

        X = data["features"]

        pipeline = VAEPipeline(

            input_dim=X.shape[1],

            hidden_dims=self.args.vae_hidden_dims,

            latent_dim=self.args.vae_latent_dim,

            dropout=self.args.dropout,

            learning_rate=self.args.learning_rate,

            batch_size=self.args.batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            beta=self.args.vae_beta,

            validation_split=self.args.validation_split,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(X)

        mu, logvar = pipeline.encode(
            X,
            sample=False,
            return_logvar=True
        )

        reconstruction_errors = pipeline.reconstruction_error(
            X
        )

        (
            diversity_report,
            diversity_scores,
            posterior_uncertainty
        ) = pipeline.analyze_latent_diversity(
            X,
            max_pairs=self.args.max_diversity_pairs,
            seed=self.args.seed
        )

        stage_dir = os.path.join(
            self.output_dir,
            "variational_autoencoder"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        pipeline.save(
            os.path.join(
                stage_dir,
                "vae_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "latent_means.npy"
            ),
            mu
        )

        save_array(
            os.path.join(
                stage_dir,
                "latent_log_variances.npy"
            ),
            logvar
        )

        save_array(
            os.path.join(
                stage_dir,
                "reconstruction_errors.npy"
            ),
            reconstruction_errors
        )

        save_array(
            os.path.join(
                stage_dir,
                "latent_diversity_scores.npy"
            ),
            diversity_scores
        )

        save_array(
            os.path.join(
                stage_dir,
                "posterior_uncertainty.npy"
            ),
            posterior_uncertainty
        )

        save_json(
            os.path.join(
                stage_dir,
                "diversity_report.json"
            ),
            diversity_report
        )

        print(
            f"Latent representations: {mu.shape}"
        )

        print(
            f"Active dimensions: "
            f"{diversity_report['active_latent_dimensions']}"
        )

        print(
            f"Mean latent diversity: "
            f"{np.mean(diversity_scores):.6f}"
        )

        return {

            "latent_means": mu,

            "latent_log_variances": logvar,

            "reconstruction_errors": reconstruction_errors,

            "diversity_scores": diversity_scores,

            "normalized_diversity": normalize_minmax(
                diversity_scores
            ),

            "posterior_uncertainty": posterior_uncertainty,

            "diversity_report": diversity_report

        }

    # --------------------------------------------------------
    # 6.5 TRAIN SAMPLE RELEVANCE MLP
    # --------------------------------------------------------

    def run_relevance_mlp(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 4A: SAMPLE RELEVANCE MLP")

        print("=" * 75)

        X = data["features"]

        y = data["relevance_labels"]

        pipeline = MLPScoringPipeline(

            input_dim=X.shape[1],

            hidden_dims=self.args.mlp_hidden_dims,

            dropout=self.args.mlp_dropout,

            learning_rate=self.args.learning_rate,

            batch_size=self.args.batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            validation_split=self.args.validation_split,

            threshold=self.args.relevance_threshold,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(
            X,
            y
        )

        scores = pipeline.predict_scores(X)

        predictions = pipeline.predict(
            X,
            threshold=self.args.relevance_threshold
        )

        stage_dir = os.path.join(
            self.output_dir,
            "relevance_mlp"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        pipeline.save(
            os.path.join(
                stage_dir,
                "relevance_mlp_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "relevance_scores.npy"
            ),
            scores
        )

        save_array(
            os.path.join(
                stage_dir,
                "relevance_predictions.npy"
            ),
            predictions
        )

        print(
            f"Scored samples: {len(scores)}"
        )

        print(
            f"Mean relevance score: {np.mean(scores):.6f}"
        )

        return {

            "pipeline": pipeline,

            "scores": scores,

            "predictions": predictions

        }

    # --------------------------------------------------------
    # 6.6 TRAIN PROMPT-RESPONSE COMPATIBILITY MLP
    # --------------------------------------------------------

    def run_compatibility_mlp(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 4B: PROMPT-RESPONSE COMPATIBILITY MLP")

        print("=" * 75)

        prompt_embeddings = data["prompt_embeddings"]

        response_embeddings = data["response_embeddings"]

        labels = data["prompt_response_labels"]

        # Construct joint representations.
        pair_features = build_prompt_response_features(
            prompt_embeddings,
            response_embeddings
        )

        pipeline = MLPScoringPipeline(

            input_dim=pair_features.shape[1],

            hidden_dims=self.args.mlp_hidden_dims,

            dropout=self.args.mlp_dropout,

            learning_rate=self.args.learning_rate,

            batch_size=self.args.batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            validation_split=self.args.validation_split,

            threshold=self.args.compatibility_threshold,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(
            pair_features,
            labels
        )

        compatibility_scores = pipeline.predict_scores(
            pair_features
        )

        predictions = pipeline.predict(
            pair_features,
            threshold=self.args.compatibility_threshold
        )

        # Assign each compatibility score to its response sample.
        sample_scores = aggregate_pair_scores(

            n_samples=len(data["features"]),

            pair_indices=data["prompt_response_sample_indices"],

            pair_scores=compatibility_scores,

            endpoints=(1,)

        )

        stage_dir = os.path.join(
            self.output_dir,
            "prompt_response_mlp"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        pipeline.save(
            os.path.join(
                stage_dir,
                "prompt_response_mlp_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "pair_compatibility_scores.npy"
            ),
            compatibility_scores
        )

        save_array(
            os.path.join(
                stage_dir,
                "pair_predictions.npy"
            ),
            predictions
        )

        save_array(
            os.path.join(
                stage_dir,
                "sample_compatibility_scores.npy"
            ),
            sample_scores
        )

        print(
            f"Scored prompt-response pairs: {len(compatibility_scores)}"
        )

        print(
            f"Mean compatibility score: "
            f"{np.mean(compatibility_scores):.6f}"
        )

        return {

            "pair_scores": compatibility_scores,

            "pair_predictions": predictions,

            "sample_scores": sample_scores

        }

    # --------------------------------------------------------
    # 6.7 TRAIN TRANSFORMER ENCODER
    # --------------------------------------------------------

    def run_transformer(
        self,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 5: TRANSFORMER ENCODER")

        print("=" * 75)

        embeddings_a = data["consistency_embeddings_a"]

        embeddings_b = data["consistency_embeddings_b"]

        mask_a = data["consistency_attention_mask_a"]

        mask_b = data["consistency_attention_mask_b"]

        labels = data["consistency_labels"]

        pipeline = TransformerConsistencyPipeline(

            input_dim=embeddings_a.shape[2],

            d_model=self.args.transformer_d_model,

            num_heads=self.args.transformer_num_heads,

            num_layers=self.args.transformer_num_layers,

            dim_feedforward=self.args.transformer_feedforward,

            dropout=self.args.dropout,

            learning_rate=self.args.transformer_learning_rate,

            batch_size=self.args.transformer_batch_size,

            epochs=self.args.epochs,

            patience=self.args.patience,

            validation_split=self.args.validation_split,

            threshold=self.args.consistency_threshold,

            device=self.device,

            seed=self.args.seed

        )

        pipeline.fit(

            embeddings_a=embeddings_a,

            embeddings_b=embeddings_b,

            attention_mask_a=mask_a,

            attention_mask_b=mask_b,

            labels=labels

        )

        consistency_scores = pipeline.predict_scores(

            embeddings_a=embeddings_a,

            embeddings_b=embeddings_b,

            attention_mask_a=mask_a,

            attention_mask_b=mask_b

        )

        predictions = (
            consistency_scores >= self.args.consistency_threshold
        ).astype(np.int64)

        contextual_representations = pipeline.encode_pairs(

            embeddings_a=embeddings_a,

            embeddings_b=embeddings_b,

            attention_mask_a=mask_a,

            attention_mask_b=mask_b

        )

        # Assign pair-level consistency scores to both samples.
        sample_scores = aggregate_pair_scores(

            n_samples=len(data["features"]),

            pair_indices=data["consistency_pair_indices"],

            pair_scores=consistency_scores,

            endpoints=(0, 1)

        )

        stage_dir = os.path.join(
            self.output_dir,
            "transformer_encoder"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        pipeline.save(
            os.path.join(
                stage_dir,
                "transformer_encoder_checkpoint.pth"
            )
        )

        save_array(
            os.path.join(
                stage_dir,
                "consistency_scores.npy"
            ),
            consistency_scores
        )

        save_array(
            os.path.join(
                stage_dir,
                "consistency_predictions.npy"
            ),
            predictions
        )

        save_array(
            os.path.join(
                stage_dir,
                "contextual_pair_representations.npy"
            ),
            contextual_representations
        )

        save_array(
            os.path.join(
                stage_dir,
                "sample_consistency_scores.npy"
            ),
            sample_scores
        )

        print(
            f"Processed narrative pairs: {len(consistency_scores)}"
        )

        print(
            f"Mean consistency score: "
            f"{np.mean(consistency_scores):.6f}"
        )

        return {

            "pair_scores": consistency_scores,

            "pair_predictions": predictions,

            "contextual_representations": contextual_representations,

            "sample_scores": sample_scores

        }

    # --------------------------------------------------------
    # 6.8 UNIFIED SCORE AGGREGATION
    # --------------------------------------------------------

    def aggregate_results(
        self,
        results: Dict,
        data: Dict
    ) -> Dict:

        print("\n" + "=" * 75)

        print("STAGE 6: UNIFIED SCORE AGGREGATION")

        print("=" * 75)

        # ----------------------------------------------------
        # Individual model scores
        # ----------------------------------------------------

        relevance_scores = results[
            "relevance"
        ]["scores"]

        compatibility_scores = results[
            "compatibility"
        ]["sample_scores"]

        diversity_scores = results[
            "vae"
        ]["normalized_diversity"]

        autoencoder_redundancy = results[
            "autoencoder"
        ]["redundancy_profile"]

        siamese_redundancy = results[
            "siamese"
        ]["redundancy_profile"]

        consistency_scores = results[
            "transformer"
        ]["sample_scores"]

        # ----------------------------------------------------
        # Calculate final score
        # ----------------------------------------------------

        final_scores, aggregation_report = calculate_curation_scores(

            relevance_scores=relevance_scores,

            compatibility_scores=compatibility_scores,

            diversity_scores=diversity_scores,

            autoencoder_redundancy=autoencoder_redundancy,

            siamese_redundancy=siamese_redundancy,

            consistency_scores=consistency_scores

        )

        # ----------------------------------------------------
        # Sample selection
        # ----------------------------------------------------

        retained_mask = (
            final_scores >= self.args.curation_threshold
        )

        retained_indices = np.where(
            retained_mask
        )[0]

        review_indices = np.where(
            ~retained_mask
        )[0]

        stage_dir = os.path.join(
            self.output_dir,
            "curation"
        )

        os.makedirs(
            stage_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save final scores
        # ----------------------------------------------------

        save_array(
            os.path.join(
                stage_dir,
                "curation_scores.npy"
            ),
            final_scores
        )

        save_array(
            os.path.join(
                stage_dir,
                "curated_sample_indices.npy"
            ),
            retained_indices
        )

        save_array(
            os.path.join(
                stage_dir,
                "review_sample_indices.npy"
            ),
            review_indices
        )

        save_array(
            os.path.join(
                stage_dir,
                "curated_feature_matrix.npy"
            ),
            data["features"][retained_indices]
        )

        save_array(
            os.path.join(
                stage_dir,
                "curated_sample_ids.npy"
            ),
            data["sample_ids"][retained_indices]
        )

        # ----------------------------------------------------
        # Build sample-level results table
        # ----------------------------------------------------

        csv_path = os.path.join(
            stage_dir,
            "sample_curation_results.csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "sample_id",

                "relevance_score",

                "prompt_response_compatibility",

                "latent_diversity",

                "autoencoder_redundancy",

                "siamese_redundancy",

                "narrative_consistency",

                "curation_score",

                "curation_decision"

            ])

            for i in range(len(final_scores)):

                decision = (
                    "retain"
                    if retained_mask[i]
                    else "review"
                )

                writer.writerow([

                    str(data["sample_ids"][i]),

                    float(relevance_scores[i]),

                    "" if not np.isfinite(
                        compatibility_scores[i]
                    ) else float(
                        compatibility_scores[i]
                    ),

                    float(diversity_scores[i]),

                    float(autoencoder_redundancy[i]),

                    float(siamese_redundancy[i]),

                    "" if not np.isfinite(
                        consistency_scores[i]
                    ) else float(
                        consistency_scores[i]
                    ),

                    float(final_scores[i]),

                    decision

                ])

        # ----------------------------------------------------
        # Aggregation report
        # ----------------------------------------------------

        report = {

            "total_samples": int(len(final_scores)),

            "retained_samples": int(len(retained_indices)),

            "review_samples": int(len(review_indices)),

            "retention_ratio": float(
                len(retained_indices) / len(final_scores)
            ),

            "curation_threshold": float(
                self.args.curation_threshold
            ),

            "mean_curation_score": float(
                np.mean(final_scores)
            ),

            "median_curation_score": float(
                np.median(final_scores)
            ),

            "min_curation_score": float(
                np.min(final_scores)
            ),

            "max_curation_score": float(
                np.max(final_scores)
            ),

            "aggregation": aggregation_report

        }

        save_json(
            os.path.join(
                stage_dir,
                "curation_report.json"
            ),
            report
        )

        print(
            f"Total samples: {len(final_scores)}"
        )

        print(
            f"Retained samples: {len(retained_indices)}"
        )

        print(
            f"Samples requiring review: {len(review_indices)}"
        )

        print(
            f"Mean curation score: {np.mean(final_scores):.6f}"
        )

        print(
            f"Retention ratio: "
            f"{report['retention_ratio']:.2%}"
        )

        return {

            "scores": final_scores,

            "retained_indices": retained_indices,

            "review_indices": review_indices,

            "report": report

        }

    # --------------------------------------------------------
    # 6.9 COMPLETE CURATION EXECUTION
    # --------------------------------------------------------

    def run(
        self,
        dataset: Dict
    ) -> Dict:

        """
        Execute the complete dataset curation framework.

        All five models are trained, their outputs are
        integrated, and the final curation results are exported.
        """

        data = self.validate_dataset(
            dataset
        )

        self.sample_ids = data["sample_ids"]

        print("\n" + "=" * 75)

        print("PLOTDOT-LLM DATASET CURATION FRAMEWORK")

        print("=" * 75)

        print(
            f"Number of samples: {len(data['features'])}"
        )

        print(
            f"Feature dimension: {data['features'].shape[1]}"
        )

        # ----------------------------------------------------
        # Execute the five architectures
        # ----------------------------------------------------

        results = {}

        results["autoencoder"] = self.run_autoencoder(
            data
        )

        results["siamese"] = self.run_siamese(
            data
        )

        results["vae"] = self.run_vae(
            data
        )

        results["relevance"] = self.run_relevance_mlp(
            data
        )

        results["compatibility"] = self.run_compatibility_mlp(
            data
        )

        results["transformer"] = self.run_transformer(
            data
        )

        # ----------------------------------------------------
        # Aggregate outputs
        # ----------------------------------------------------

        results["curation"] = self.aggregate_results(
            results,
            data
        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        summary = {

            "framework": "PlotDot-LLM Dataset Curation",

            "models": [

                "Autoencoder",

                "Siamese Network",

                "Variational Autoencoder",

                "Relevance MLP",

                "Prompt-Response Compatibility MLP",

                "Transformer Encoder"

            ],

            "total_samples": int(
                len(data["features"])
            ),

            "retained_samples": int(
                len(results["curation"]["retained_indices"])
            ),

            "review_samples": int(
                len(results["curation"]["review_indices"])
            ),

            "curation_threshold": float(
                self.args.curation_threshold
            )

        }

        save_json(
            os.path.join(
                self.output_dir,
                "pipeline_summary.json"
            ),
            summary
        )

        print("\n" + "=" * 75)

        print("DATASET CURATION PIPELINE COMPLETED")

        print("=" * 75)

        print(
            f"Output directory: {self.output_dir}"
        )

        return results


# ============================================================
# 7. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Unified Data-Centric Dataset Curation Pipeline "
            "for Yi-6B-200K post-training."
        )
    )

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input NPZ dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/dataset_curation"
    )

    # --------------------------------------------------------
    # General optimization
    # --------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None
    )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    parser.add_argument(
        "--ae-hidden-dims",
        type=parse_dimensions,
        default=[512, 256]
    )

    parser.add_argument(
        "--ae-latent-dim",
        type=int,
        default=64
    )

    parser.add_argument(
        "--autoencoder-threshold",
        type=float,
        default=0.95
    )

    # --------------------------------------------------------
    # Siamese Network
    # --------------------------------------------------------

    parser.add_argument(
        "--siamese-hidden-dims",
        type=parse_dimensions,
        default=[512, 256]
    )

    parser.add_argument(
        "--siamese-embedding-dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--siamese-threshold",
        type=float,
        default=0.90
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1.0
    )

    # --------------------------------------------------------
    # Variational Autoencoder
    # --------------------------------------------------------

    parser.add_argument(
        "--vae-hidden-dims",
        type=parse_dimensions,
        default=[512, 256]
    )

    parser.add_argument(
        "--vae-latent-dim",
        type=int,
        default=64
    )

    parser.add_argument(
        "--vae-beta",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max-diversity-pairs",
        type=int,
        default=100000
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    parser.add_argument(
        "--mlp-hidden-dims",
        type=parse_dimensions,
        default=[256, 128, 64]
    )

    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--compatibility-threshold",
        type=float,
        default=0.5
    )

    # --------------------------------------------------------
    # Transformer Encoder
    # --------------------------------------------------------

    parser.add_argument(
        "--transformer-d-model",
        type=int,
        default=256
    )

    parser.add_argument(
        "--transformer-num-heads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--transformer-num-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--transformer-feedforward",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--transformer-learning-rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.5
    )

    # --------------------------------------------------------
    # Final curation
    # --------------------------------------------------------

    parser.add_argument(
        "--curation-threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Global initialization
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    with np.load(
        args.input,
        allow_pickle=False
    ) as archive:

        dataset = {
            key: archive[key]
            for key in archive.files
        }

    # --------------------------------------------------------
    # Execute curation framework
    # --------------------------------------------------------

    curator = DatasetCurator(
        args
    )

    curator.run(
        dataset
    )


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
