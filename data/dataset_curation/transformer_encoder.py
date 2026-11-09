
"""
Transformer Encoder for Semantic Relationship Modeling
and Narrative Consistency Analysis
========================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    1. Model contextual relationships between narrative samples.
    2. Capture semantic dependencies using multi-head self-attention.
    3. Analyze narrative consistency between pairs of samples.
    4. Learn contextual representations across narrative sequences.
    5. Predict whether two narrative samples are semantically consistent.

Architecture:
    - Input projection.
    - Sinusoidal positional encoding.
    - Segment embeddings.
    - Multi-head self-attention.
    - Transformer Encoder layers.
    - Feed-forward neural networks.
    - CLS-based sequence-pair representation.
    - Binary classification head.

Input:
    Token-level numerical embeddings for two narrative sequences.

    embeddings_a:
        (n_pairs, seq_len_a, input_dim)

    embeddings_b:
        (n_pairs, seq_len_b, input_dim)

    attention_mask_a:
        (n_pairs, seq_len_a)

    attention_mask_b:
        (n_pairs, seq_len_b)

    labels:
        (n_pairs,)

        1 = Semantically consistent pair.
        0 = Semantically inconsistent pair.

Output:
    - Trained Transformer Encoder checkpoint.
    - Contextual pair representations.
    - Narrative consistency probabilities.
    - Binary consistency predictions.
    - Evaluation metrics.

Requirements:
    pip install torch numpy

Author:
    PlotDot-LLM Project
"""

import os
import csv
import copy
import json
import math
import random
import argparse

from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. SINUSOIDAL POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Injects positional information into token representations
    because self-attention alone does not encode token order.

    Encoding:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))

        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Supports variable sequence lengths.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1
    ):

        super().__init__()

        if d_model < 1:
            raise ValueError(
                "d_model must be positive."
            )

        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_len, d_model = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=x.dtype
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                device=x.device,
                dtype=x.dtype
            ) * (-math.log(10000.0) / d_model)
        )

        positional_encoding = torch.zeros(
            seq_len,
            d_model,
            device=x.device,
            dtype=x.dtype
        )

        positional_encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        positional_encoding[:, 1::2] = torch.cos(
            positions * div_term[:d_model // 2]
        )

        positional_encoding = positional_encoding.unsqueeze(0)

        x = x + positional_encoding

        return self.dropout(x)


# ============================================================
# 3. TRANSFORMER ENCODER ARCHITECTURE
# ============================================================

class TransformerConsistencyEncoder(nn.Module):
    """
    Transformer Encoder for semantic relationship modeling
    and narrative consistency classification.

    Two narrative sequences are processed jointly.

    Architecture:

        Narrative A embeddings ----+
                                   |
                                   v
                             Input Projection
                                   |
        Narrative B embeddings ---+
                                   |
                                   v
                         Segment Embeddings
                                   |
                                   v
                        Positional Encoding
                                   |
                                   v
                            CLS Token
                                   |
                                   v
                      Transformer Encoder
                                   |
                                   v
                      Contextual CLS Vector
                                   |
                                   v
                         Classification Head
                                   |
                                   v
                     Narrative Consistency Logit

    Joint processing enables cross-attention between tokens
    belonging to the two narrative sequences.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):

        super().__init__()

        if input_dim < 1 or d_model < 1:
            raise ValueError(
                "Input and model dimensions must be positive."
            )

        if num_heads < 1 or num_layers < 1:
            raise ValueError(
                "num_heads and num_layers must be positive."
            )

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        if dim_feedforward < 1:
            raise ValueError(
                "dim_feedforward must be positive."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.input_dim = input_dim

        self.d_model = d_model

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.input_norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Segment embeddings
        # ----------------------------------------------------

        # Segment 0: CLS token and narrative A.
        # Segment 1: narrative B.

        self.segment_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=d_model
        )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02
        )

        # ----------------------------------------------------
        # Transformer Encoder layers
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Narrative consistency classification head
        # ----------------------------------------------------

        self.classification_head = nn.Sequential(

            nn.Linear(
                d_model,
                d_model // 2
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                d_model // 2,
                1
            )

        )

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        attention_mask_a: torch.Tensor,
        attention_mask_b: torch.Tensor
    ) -> torch.Tensor:

        """
        Process two narrative sequences jointly.

        Returns:
            One consistency logit per sequence pair.
        """

        batch_size = embeddings_a.size(0)

        # ----------------------------------------------------
        # Project input embeddings
        # ----------------------------------------------------

        projected_a = self.input_projection(
            embeddings_a
        )

        projected_b = self.input_projection(
            embeddings_b
        )

        projected_a = self.input_norm(
            projected_a
        )

        projected_b = self.input_norm(
            projected_b
        )

        # ----------------------------------------------------
        # Create CLS token
        # ----------------------------------------------------

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # ----------------------------------------------------
        # Concatenate both narrative sequences
        # ----------------------------------------------------

        sequence = torch.cat(
            [
                cls,
                projected_a,
                projected_b
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Construct segment identifiers
        # ----------------------------------------------------

        segment_a = torch.zeros(
            batch_size,
            projected_a.size(1) + 1,
            dtype=torch.long,
            device=sequence.device
        )

        segment_b = torch.ones(
            batch_size,
            projected_b.size(1),
            dtype=torch.long,
            device=sequence.device
        )

        segment_ids = torch.cat(
            [
                segment_a,
                segment_b
            ],
            dim=1
        )

        sequence = sequence + self.segment_embedding(
            segment_ids
        )

        # ----------------------------------------------------
        # Add positional encoding
        # ----------------------------------------------------

        sequence = self.position_encoding(
            sequence
        )

        # ----------------------------------------------------
        # Construct attention masks
        # ----------------------------------------------------

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=sequence.device
        )

        valid_tokens = torch.cat(
            [
                cls_mask,
                attention_mask_a.bool(),
                attention_mask_b.bool()
            ],
            dim=1
        )

        # PyTorch expects True for padded positions.
        padding_mask = ~valid_tokens

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        contextual_representations = self.transformer_encoder(
            sequence,
            src_key_padding_mask=padding_mask
        )

        # ----------------------------------------------------
        # Extract contextual CLS representation
        # ----------------------------------------------------

        cls_representation = contextual_representations[:, 0]

        cls_representation = self.output_norm(
            cls_representation
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classification_head(
            cls_representation
        )

        return logits.squeeze(-1)

    def encode_pair(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        attention_mask_a: torch.Tensor,
        attention_mask_b: torch.Tensor
    ) -> torch.Tensor:

        """Extract contextual representations of sequence pairs."""

        batch_size = embeddings_a.size(0)

        projected_a = self.input_norm(
            self.input_projection(embeddings_a)
        )

        projected_b = self.input_norm(
            self.input_projection(embeddings_b)
        )

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        sequence = torch.cat(
            [
                cls,
                projected_a,
                projected_b
            ],
            dim=1
        )

        segment_a = torch.zeros(
            batch_size,
            projected_a.size(1) + 1,
            dtype=torch.long,
            device=sequence.device
        )

        segment_b = torch.ones(
            batch_size,
            projected_b.size(1),
            dtype=torch.long,
            device=sequence.device
        )

        segment_ids = torch.cat(
            [
                segment_a,
                segment_b
            ],
            dim=1
        )

        sequence = sequence + self.segment_embedding(
            segment_ids
        )

        sequence = self.position_encoding(
            sequence
        )

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=sequence.device
        )

        valid_tokens = torch.cat(
            [
                cls_mask,
                attention_mask_a.bool(),
                attention_mask_b.bool()
            ],
            dim=1
        )

        padding_mask = ~valid_tokens

        contextual_representations = self.transformer_encoder(
            sequence,
            src_key_padding_mask=padding_mask
        )

        return self.output_norm(
            contextual_representations[:, 0]
        )


# ============================================================
# 4. NARRATIVE CONSISTENCY LOSS
# ============================================================

class NarrativeConsistencyLoss(nn.Module):
    """
    Binary classification objective using BCEWithLogitsLoss.

    Labels:

        1 = Semantically consistent pair.
        0 = Semantically inconsistent pair.

    The positive class weight compensates for class imbalance.
    """

    def __init__(
        self,
        positive_weight: float = 1.0
    ):

        super().__init__()

        if positive_weight <= 0:
            raise ValueError(
                "positive_weight must be positive."
            )

        self.positive_weight = positive_weight

        self.loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [positive_weight],
                dtype=torch.float32
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        return self.loss_function(
            logits,
            labels
        )


# ============================================================
# 5. TRANSFORMER TRAINING PIPELINE
# ============================================================

class TransformerConsistencyPipeline:
    """
    Complete Transformer Encoder training and inference pipeline.

    Responsibilities:

        - Input validation.
        - Sequence and mask validation.
        - Stratified training/validation splitting.
        - Transformer optimization.
        - Class imbalance handling.
        - Validation metrics.
        - Early stopping.
        - Consistency prediction.
        - Contextual pair representation extraction.
        - Model saving and loading.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 50,
        patience: int = 8,
        validation_split: float = 0.2,
        threshold: float = 0.5,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        seed: int = 42
    ):

        if input_dim < 1:
            raise ValueError(
                "input_dim must be positive."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError(
                "Training parameters must be positive."
            )

        if not 0 < validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1."
            )

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if gradient_clip <= 0:
            raise ValueError(
                "gradient_clip must be positive."
            )

        self.input_dim = input_dim

        self.d_model = d_model

        self.num_heads = num_heads

        self.num_layers = num_layers

        self.dim_feedforward = dim_feedforward

        self.dropout = dropout

        self.learning_rate = learning_rate

        self.batch_size = batch_size

        self.epochs = epochs

        self.patience = patience

        self.validation_split = validation_split

        self.threshold = threshold

        self.gradient_clip = gradient_clip

        self.seed = seed

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = TransformerConsistencyEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        self.criterion = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        self.positive_weight = None

        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_precision": [],
            "validation_recall": [],
            "validation_f1": []
        }

        self.is_fitted = False

    # --------------------------------------------------------
    # 5.1 INPUT VALIDATION
    # --------------------------------------------------------

    def _validate_inputs(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        embeddings_a = np.asarray(
            embeddings_a,
            dtype=np.float32
        )

        embeddings_b = np.asarray(
            embeddings_b,
            dtype=np.float32
        )

        attention_mask_a = np.asarray(
            attention_mask_a
        )

        attention_mask_b = np.asarray(
            attention_mask_b
        )

        if embeddings_a.ndim != 3:
            raise ValueError(
                "embeddings_a must have shape (N, L, D)."
            )

        if embeddings_b.ndim != 3:
            raise ValueError(
                "embeddings_b must have shape (N, L, D)."
            )

        if embeddings_a.shape[0] != embeddings_b.shape[0]:
            raise ValueError(
                "Both embedding arrays must contain the same number of pairs."
            )

        if embeddings_a.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim} "
                f"for embeddings_a, received {embeddings_a.shape[2]}."
            )

        if embeddings_b.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim} "
                f"for embeddings_b, received {embeddings_b.shape[2]}."
            )

        if embeddings_a.shape[1] < 1 or embeddings_b.shape[1] < 1:
            raise ValueError(
                "Both sequences must have at least one position."
            )

        if not np.isfinite(embeddings_a).all():
            raise ValueError(
                "embeddings_a contains NaN or infinite values."
            )

        if not np.isfinite(embeddings_b).all():
            raise ValueError(
                "embeddings_b contains NaN or infinite values."
            )

        expected_mask_a = embeddings_a.shape[:2]
        expected_mask_b = embeddings_b.shape[:2]

        if attention_mask_a.shape != expected_mask_a:
            raise ValueError(
                "attention_mask_a must have shape (N, seq_len_a)."
            )

        if attention_mask_b.shape != expected_mask_b:
            raise ValueError(
                "attention_mask_b must have shape (N, seq_len_b)."
            )

        if not np.isfinite(attention_mask_a).all():
            raise ValueError(
                "attention_mask_a contains invalid values."
            )

        if not np.isfinite(attention_mask_b).all():
            raise ValueError(
                "attention_mask_b contains invalid values."
            )

        if not np.isin(attention_mask_a, [0, 1]).all():
            raise ValueError(
                "attention_mask_a must contain only 0 and 1."
            )

        if not np.isin(attention_mask_b, [0, 1]).all():
            raise ValueError(
                "attention_mask_b must contain only 0 and 1."
            )

        attention_mask_a = attention_mask_a.astype(bool)
        attention_mask_b = attention_mask_b.astype(bool)

        if not attention_mask_a.any(axis=1).all():
            raise ValueError(
                "Every sequence A must contain at least one valid token."
            )

        if not attention_mask_b.any(axis=1).all():
            raise ValueError(
                "Every sequence B must contain at least one valid token."
            )

        return (
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

    def _validate_labels(
        self,
        labels: np.ndarray,
        n_samples: int
    ) -> np.ndarray:

        labels = np.asarray(
            labels,
            dtype=np.float32
        ).reshape(-1)

        if len(labels) != n_samples:
            raise ValueError(
                "Number of labels must match number of sequence pairs."
            )

        if not np.isfinite(labels).all():
            raise ValueError(
                "Labels contain invalid values."
            )

        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(
                "Labels must be binary: 0 or 1."
            )

        return labels

    # --------------------------------------------------------
    # 5.2 STRATIFIED DATA SPLITTING
    # --------------------------------------------------------

    def _split_indices(
        self,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = np.random.default_rng(
            self.seed
        )

        train_indices = []
        val_indices = []

        for label in [0.0, 1.0]:

            class_indices = np.where(
                labels == label
            )[0]

            if len(class_indices) < 2:
                raise ValueError(
                    "At least two pairs per class are required."
                )

            rng.shuffle(
                class_indices
            )

            n_val = max(
                1,
                int(
                    len(class_indices)
                    * self.validation_split
                )
            )

            n_val = min(
                n_val,
                len(class_indices) - 1
            )

            val_indices.extend(
                class_indices[:n_val]
            )

            train_indices.extend(
                class_indices[n_val:]
            )

        train_indices = np.asarray(
            train_indices,
            dtype=np.int64
        )

        val_indices = np.asarray(
            val_indices,
            dtype=np.int64
        )

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        return train_indices, val_indices

    # --------------------------------------------------------
    # 5.3 DATASET CONSTRUCTION
    # --------------------------------------------------------

    @staticmethod
    def _build_dataset(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        labels: np.ndarray
    ) -> TensorDataset:

        return TensorDataset(

            torch.tensor(
                embeddings_a,
                dtype=torch.float32
            ),

            torch.tensor(
                embeddings_b,
                dtype=torch.float32
            ),

            torch.tensor(
                mask_a,
                dtype=torch.bool
            ),

            torch.tensor(
                mask_b,
                dtype=torch.bool
            ),

            torch.tensor(
                labels,
                dtype=torch.float32
            )

        )

    # --------------------------------------------------------
    # 5.4 TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, list]:

        """
        Train the Transformer Encoder.

        Parameters:
            embeddings_a:
                Token-level embeddings of narrative A.

            embeddings_b:
                Token-level embeddings of narrative B.

            attention_mask_a:
                Binary mask for narrative A.

            attention_mask_b:
                Binary mask for narrative B.

            labels:
                1 = Consistent.
                0 = Inconsistent.
        """

        set_seed(self.seed)

        (
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        ) = self._validate_inputs(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        labels = self._validate_labels(
            labels,
            len(embeddings_a)
        )

        if len(np.unique(labels)) != 2:
            raise ValueError(
                "Training requires consistent and inconsistent pairs."
            )

        # ----------------------------------------------------
        # Training / validation split
        # ----------------------------------------------------

        train_indices, val_indices = self._split_indices(
            labels
        )

        train_dataset = self._build_dataset(

            embeddings_a[train_indices],

            embeddings_b[train_indices],

            attention_mask_a[train_indices],

            attention_mask_b[train_indices],

            labels[train_indices]

        )

        val_dataset = self._build_dataset(

            embeddings_a[val_indices],

            embeddings_b[val_indices],

            attention_mask_a[val_indices],

            attention_mask_b[val_indices],

            labels[val_indices]

        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda"
        )

        # ----------------------------------------------------
        # Class imbalance handling
        # ----------------------------------------------------

        train_labels = labels[train_indices]

        n_positive = np.sum(
            train_labels == 1
        )

        n_negative = np.sum(
            train_labels == 0
        )

        self.positive_weight = float(
            n_negative / n_positive
        )

        self.criterion = NarrativeConsistencyLoss(
            positive_weight=self.positive_weight
        ).to(self.device)

        # ----------------------------------------------------
        # Early stopping initialization
        # ----------------------------------------------------

        best_val_loss = float("inf")

        best_weights = None

        epochs_without_improvement = 0

        print(f"Training device: {self.device}")

        print(f"Training pairs: {len(train_dataset)}")

        print(f"Validation pairs: {len(val_dataset)}")

        print(
            f"Positive class weight: {self.positive_weight:.4f}"
        )

        print("-" * 95)

        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            # ================================================
            # TRAINING PHASE
            # ================================================

            for (
                batch_a,
                batch_b,
                mask_a,
                mask_b,
                batch_labels
            ) in train_loader:

                batch_a = batch_a.to(self.device)

                batch_b = batch_b.to(self.device)

                mask_a = mask_a.to(self.device)

                mask_b = mask_b.to(self.device)

                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(
                    batch_a,
                    batch_b,
                    mask_a,
                    mask_b
                )

                loss = self.criterion(
                    logits,
                    batch_labels
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

                self.optimizer.step()

                train_loss += (
                    loss.item() * batch_a.size(0)
                )

            train_loss /= len(train_dataset)

            # ================================================
            # VALIDATION PHASE
            # ================================================

            (
                val_loss,
                metrics
            ) = self._evaluate_loader(
                val_loader
            )

            # Record training history.
            self.history["train_loss"].append(
                train_loss
            )

            self.history["validation_loss"].append(
                val_loss
            )

            self.history["validation_accuracy"].append(
                metrics["accuracy"]
            )

            self.history["validation_precision"].append(
                metrics["precision"]
            )

            self.history["validation_recall"].append(
                metrics["recall"]
            )

            self.history["validation_f1"].append(
                metrics["f1"]
            )

            print(
                f"Epoch [{epoch + 1:03d}/{self.epochs}] "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f} "
                f"| Accuracy: {metrics['accuracy']:.4f} "
                f"| F1: {metrics['f1']:.4f}"
            )

            # ================================================
            # EARLY STOPPING
            # ================================================

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_weights = copy.deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:

                print("Early stopping triggered.")

                break

        if best_weights is None:
            raise RuntimeError(
                "Training did not produce a valid checkpoint."
            )

        # Restore the best validation checkpoint.
        self.model.load_state_dict(
            best_weights
        )

        self.model.eval()

        self.is_fitted = True

        print("-" * 95)

        print(
            f"Best validation loss: {best_val_loss:.6f}"
        )

        print(
            "Transformer Encoder training completed."
        )

        return self.history

    # --------------------------------------------------------
    # 5.5 VALIDATION METRICS
    # --------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        labels: np.ndarray,
        probabilities: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:

        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        labels = labels.astype(
            np.int64
        )

        true_positive = int(
            np.sum(
                (labels == 1) & (predictions == 1)
            )
        )

        true_negative = int(
            np.sum(
                (labels == 0) & (predictions == 0)
            )
        )

        false_positive = int(
            np.sum(
                (labels == 0) & (predictions == 1)
            )
        )

        false_negative = int(
            np.sum(
                (labels == 1) & (predictions == 0)
            )
        )

        accuracy = (
            true_positive + true_negative
        ) / max(len(labels), 1)

        precision = (
            true_positive
            / max(true_positive + false_positive, 1)
        )

        recall = (
            true_positive
            / max(true_positive + false_negative, 1)
        )

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-12)
        )

        return {

            "accuracy": float(accuracy),

            "precision": float(precision),

            "recall": float(recall),

            "f1": float(f1),

            "true_positive": true_positive,

            "true_negative": true_negative,

            "false_positive": false_positive,

            "false_negative": false_negative

        }

    def _evaluate_loader(
        self,
        loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:

        self.model.eval()

        total_loss = 0.0

        all_labels = []

        all_probabilities = []

        with torch.no_grad():

            for (
                batch_a,
                batch_b,
                mask_a,
                mask_b,
                batch_labels
            ) in loader:

                batch_a = batch_a.to(self.device)

                batch_b = batch_b.to(self.device)

                mask_a = mask_a.to(self.device)

                mask_b = mask_b.to(self.device)

                batch_labels = batch_labels.to(self.device)

                logits = self.model(
                    batch_a,
                    batch_b,
                    mask_a,
                    mask_b
                )

                loss = self.criterion(
                    logits,
                    batch_labels
                )

                probabilities = torch.sigmoid(
                    logits
                )

                total_loss += (
                    loss.item() * batch_a.size(0)
                )

                all_labels.extend(
                    batch_labels.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

        total_loss /= len(loader.dataset)

        metrics = self._compute_metrics(
            np.asarray(all_labels),
            np.asarray(all_probabilities),
            self.threshold
        )

        return total_loss, metrics

    # --------------------------------------------------------
    # 5.6 CONSISTENCY SCORE PREDICTION
    # --------------------------------------------------------

    def predict_scores(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:

        """
        Compute narrative consistency probabilities.

        Returns:
            An array containing probabilities between 0 and 1.

        Higher values indicate stronger predicted consistency.
        """

        self._check_fitted()

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        (
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        ) = self._validate_inputs(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        dataset = TensorDataset(

            torch.tensor(
                embeddings_a,
                dtype=torch.float32
            ),

            torch.tensor(
                embeddings_b,
                dtype=torch.float32
            ),

            torch.tensor(
                attention_mask_a,
                dtype=torch.bool
            ),

            torch.tensor(
                attention_mask_b,
                dtype=torch.bool
            )

        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda"
        )

        self.model.eval()

        all_scores = []

        with torch.no_grad():

            for (
                batch_a,
                batch_b,
                mask_a,
                mask_b
            ) in loader:

                batch_a = batch_a.to(self.device)

                batch_b = batch_b.to(self.device)

                mask_a = mask_a.to(self.device)

                mask_b = mask_b.to(self.device)

                logits = self.model(
                    batch_a,
                    batch_b,
                    mask_a,
                    mask_b
                )

                probabilities = torch.sigmoid(
                    logits
                )

                all_scores.append(
                    probabilities.cpu().numpy()
                )

        return np.concatenate(
            all_scores,
            axis=0
        )

    # --------------------------------------------------------
    # 5.7 BINARY CONSISTENCY PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:

        """
        Generate binary narrative consistency predictions.

        Returns:
            1 = Consistent.
            0 = Inconsistent.
        """

        if threshold is None:
            threshold = self.threshold

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        scores = self.predict_scores(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        return (
            scores >= threshold
        ).astype(np.int64)

    # --------------------------------------------------------
    # 5.8 CONTEXTUAL REPRESENTATION EXTRACTION
    # --------------------------------------------------------

    def encode_pairs(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:

        """
        Extract joint contextual representations for sequence pairs.

        Returns:
            (n_pairs, d_model)
        """

        self._check_fitted()

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        (
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        ) = self._validate_inputs(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        dataset = TensorDataset(

            torch.tensor(
                embeddings_a,
                dtype=torch.float32
            ),

            torch.tensor(
                embeddings_b,
                dtype=torch.float32
            ),

            torch.tensor(
                attention_mask_a,
                dtype=torch.bool
            ),

            torch.tensor(
                attention_mask_b,
                dtype=torch.bool
            )

        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        representations = []

        with torch.no_grad():

            for (
                batch_a,
                batch_b,
                mask_a,
                mask_b
            ) in loader:

                batch_a = batch_a.to(self.device)

                batch_b = batch_b.to(self.device)

                mask_a = mask_a.to(self.device)

                mask_b = mask_b.to(self.device)

                contextual_representation = self.model.encode_pair(
                    batch_a,
                    batch_b,
                    mask_a,
                    mask_b
                )

                representations.append(
                    contextual_representation.cpu().numpy()
                )

        return np.concatenate(
            representations,
            axis=0
        )

    # --------------------------------------------------------
    # 5.9 MODEL EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        attention_mask_a: np.ndarray,
        attention_mask_b: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:

        """Evaluate the trained model on independent labeled pairs."""

        self._check_fitted()

        (
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        ) = self._validate_inputs(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        labels = self._validate_labels(
            labels,
            len(embeddings_a)
        )

        scores = self.predict_scores(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b
        )

        return self._compute_metrics(
            labels,
            scores,
            self.threshold
        )

    # --------------------------------------------------------
    # 5.10 MODEL SAVING
    # --------------------------------------------------------

    def save(
        self,
        path: str
    ) -> None:

        """Save model weights and architecture configuration."""

        self._check_fitted()

        os.makedirs(
            os.path.dirname(
                os.path.abspath(path)
            ),
            exist_ok=True
        )

        checkpoint = {

            "input_dim": self.input_dim,

            "d_model": self.d_model,

            "num_heads": self.num_heads,

            "num_layers": self.num_layers,

            "dim_feedforward": self.dim_feedforward,

            "dropout": self.dropout,

            "model_state_dict": self.model.state_dict(),

            "positive_weight": self.positive_weight,

            "learning_rate": self.learning_rate,

            "batch_size": self.batch_size,

            "epochs": self.epochs,

            "patience": self.patience,

            "validation_split": self.validation_split,

            "threshold": self.threshold,

            "gradient_clip": self.gradient_clip,

            "seed": self.seed,

            "history": self.history

        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"Transformer checkpoint saved to: {path}"
        )

    # --------------------------------------------------------
    # 5.11 MODEL LOADING
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None
    ):

        """Load a previously trained Transformer Encoder."""

        try:

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

        except TypeError:

            checkpoint = torch.load(
                path,
                map_location="cpu"
            )

        pipeline = cls(

            input_dim=checkpoint["input_dim"],

            d_model=checkpoint["d_model"],

            num_heads=checkpoint["num_heads"],

            num_layers=checkpoint["num_layers"],

            dim_feedforward=checkpoint["dim_feedforward"],

            dropout=checkpoint["dropout"],

            learning_rate=checkpoint["learning_rate"],

            batch_size=checkpoint["batch_size"],

            epochs=checkpoint["epochs"],

            patience=checkpoint["patience"],

            validation_split=checkpoint["validation_split"],

            threshold=checkpoint["threshold"],

            gradient_clip=checkpoint["gradient_clip"],

            seed=checkpoint["seed"],

            device=device

        )

        pipeline.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        pipeline.positive_weight = checkpoint["positive_weight"]

        pipeline.history = checkpoint["history"]

        if pipeline.positive_weight is not None:

            pipeline.criterion = NarrativeConsistencyLoss(
                positive_weight=pipeline.positive_weight
            ).to(pipeline.device)

        pipeline.model.eval()

        pipeline.is_fitted = True

        return pipeline

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _validate_labels(
        self,
        labels: np.ndarray,
        n_samples: int
    ) -> np.ndarray:

        labels = np.asarray(
            labels,
            dtype=np.float32
        ).reshape(-1)

        if len(labels) != n_samples:
            raise ValueError(
                "Number of labels must match the number of pairs."
            )

        if not np.isfinite(labels).all():
            raise ValueError(
                "Labels contain NaN or infinite values."
            )

        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(
                "Labels must be binary: 0 or 1."
            )

        return labels

    def _check_fitted(self) -> None:

        if not self.is_fitted:

            raise RuntimeError(
                "The Transformer Encoder must be trained "
                "or loaded before inference."
            )


# ============================================================
# 6. EXPORT RESULTS
# ============================================================

def export_results(
    pipeline: TransformerConsistencyPipeline,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    attention_mask_a: np.ndarray,
    attention_mask_b: np.ndarray,
    output_dir: str,
    pair_ids: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None
) -> None:

    """
    Export:

        - Narrative consistency probabilities.
        - Binary consistency predictions.
        - Joint contextual representations.
        - Pair-level classification results.
        - Evaluation metrics.
        - Training history.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Consistency scoring
    # --------------------------------------------------------

    scores = pipeline.predict_scores(
        embeddings_a,
        embeddings_b,
        attention_mask_a,
        attention_mask_b
    )

    predictions = (
        scores >= pipeline.threshold
    ).astype(np.int64)

    # --------------------------------------------------------
    # Contextual representation extraction
    # --------------------------------------------------------

    representations = pipeline.encode_pairs(
        embeddings_a,
        embeddings_b,
        attention_mask_a,
        attention_mask_b
    )

    np.save(
        os.path.join(
            output_dir,
            "contextual_pair_representations.npy"
        ),
        representations
    )

    # --------------------------------------------------------
    # Export scores and predictions
    # --------------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            "consistency_scores.npy"
        ),
        scores
    )

    np.save(
        os.path.join(
            output_dir,
            "consistency_predictions.npy"
        ),
        predictions
    )

    # --------------------------------------------------------
    # Validate pair identifiers
    # --------------------------------------------------------

    n_pairs = len(scores)

    if pair_ids is None:

        pair_ids = np.column_stack([
            np.arange(n_pairs),
            np.arange(n_pairs)
        ]).astype(str)

    else:

        pair_ids = np.asarray(
            pair_ids
        )

        if pair_ids.shape != (n_pairs, 2):
            raise ValueError(
                "pair_ids must have shape (n_pairs, 2)."
            )

    if labels is not None:

        labels = pipeline._validate_labels(
            labels,
            n_pairs
        )

    # --------------------------------------------------------
    # Export CSV
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "narrative_consistency_results.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "sample_id_a",
            "sample_id_b",
            "consistency_score",
            "predicted_label",
            "ground_truth"
        ])

        for i in range(n_pairs):

            ground_truth = (
                int(labels[i])
                if labels is not None
                else ""
            )

            writer.writerow([
                str(pair_ids[i, 0]),
                str(pair_ids[i, 1]),
                float(scores[i]),
                int(predictions[i]),
                ground_truth
            ])

    # --------------------------------------------------------
    # Export evaluation metrics
    # --------------------------------------------------------

    if labels is not None:

        metrics = pipeline.evaluate(
            embeddings_a,
            embeddings_b,
            attention_mask_a,
            attention_mask_b,
            labels
        )

        with open(
            os.path.join(
                output_dir,
                "evaluation_metrics.json"
            ),
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

    # --------------------------------------------------------
    # Export training history
    # --------------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "training_history.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pipeline.history,
            file,
            indent=4
        )

    print("\nTransformer results exported successfully.")

    print(
        f"Number of narrative pairs: {n_pairs}"
    )

    print(
        f"Contextual representations: {representations.shape}"
    )

    print(
        f"Mean consistency score: {np.mean(scores):.4f}"
    )

    print(
        f"Predicted consistent pairs: {np.sum(predictions)}"
    )

    print(
        f"Results directory: {output_dir}"
    )


# ============================================================
# 7. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Transformer Encoder for semantic relationship "
            "modeling and narrative consistency analysis."
        )
    )

    # --------------------------------------------------------
    # Input and output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the NPZ file containing sequence embeddings."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/transformer_encoder"
    )

    # --------------------------------------------------------
    # Transformer architecture
    # --------------------------------------------------------

    parser.add_argument(
        "--d-model",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0
    )

    # --------------------------------------------------------
    # Reproducibility and device
    # --------------------------------------------------------

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

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Load input arrays
    # --------------------------------------------------------

    with np.load(
        args.input,
        allow_pickle=False
    ) as data:

        required_fields = {
            "embeddings_a",
            "embeddings_b",
            "attention_mask_a",
            "attention_mask_b",
            "labels"
        }

        missing_fields = (
            required_fields - set(data.files)
        )

        if missing_fields:
            raise KeyError(
                f"Missing input fields: {sorted(missing_fields)}"
            )

        embeddings_a = data["embeddings_a"]

        embeddings_b = data["embeddings_b"]

        attention_mask_a = data["attention_mask_a"]

        attention_mask_b = data["attention_mask_b"]

        labels = data["labels"]

        pair_ids = (
            data["pair_ids"]
            if "pair_ids" in data.files
            else None
        )

    # --------------------------------------------------------
    # Initialize Transformer pipeline
    # --------------------------------------------------------

    pipeline = TransformerConsistencyPipeline(

        input_dim=embeddings_a.shape[2],

        d_model=args.d_model,

        num_heads=args.num_heads,

        num_layers=args.num_layers,

        dim_feedforward=args.dim_feedforward,

        dropout=args.dropout,

        learning_rate=args.learning_rate,

        batch_size=args.batch_size,

        epochs=args.epochs,

        patience=args.patience,

        validation_split=args.validation_split,

        threshold=args.threshold,

        gradient_clip=args.gradient_clip,

        device=args.device,

        seed=args.seed

    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(

        embeddings_a=embeddings_a,

        embeddings_b=embeddings_b,

        attention_mask_a=attention_mask_a,

        attention_mask_b=attention_mask_b,

        labels=labels

    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    pipeline.save(
        os.path.join(
            args.output_dir,
            "transformer_encoder_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # Export results
    # --------------------------------------------------------

    export_results(

        pipeline=pipeline,

        embeddings_a=embeddings_a,

        embeddings_b=embeddings_b,

        attention_mask_a=attention_mask_a,

        attention_mask_b=attention_mask_b,

        output_dir=args.output_dir,

        pair_ids=pair_ids,

        labels=labels

    )


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
