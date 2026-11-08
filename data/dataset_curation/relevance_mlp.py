
"""
MLP for Sample Relevance and Prompt-Response Compatibility Scoring
====================================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    1. Estimate the relevance of individual narrative samples.
    2. Evaluate prompt-response compatibility.
    3. Learn nonlinear relationships between input features.
    4. Generate relevance and compatibility scores.
    5. Support dataset filtering and prioritization.

Architecture:
    - Multi-Layer Perceptron.
    - Fully connected layers.
    - Layer Normalization.
    - ReLU activation.
    - Dropout regularization.
    - Sigmoid probability estimation.

Two supported applications:

    A. Sample relevance scoring:

       Input:
           Numerical representation of a narrative sample.

       Output:
           Relevance probability between 0 and 1.

    B. Prompt-response compatibility scoring:

       Input:
           Combined prompt and response representations.

       Output:
           Compatibility probability between 0 and 1.

Input:
    A numerical feature matrix:
        X.shape = (n_samples, input_dim)

    Binary labels:
        y.shape = (n_samples,)

        1 = Relevant / Compatible.
        0 = Irrelevant / Incompatible.

Requirements:
    pip install torch numpy

Author:
    PlotDot-LLM Project
"""

import os
import csv
import json
import copy
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
# 2. PROMPT-RESPONSE FEATURE CONSTRUCTION
# ============================================================

def build_prompt_response_features(
    prompt_embeddings: np.ndarray,
    response_embeddings: np.ndarray
) -> np.ndarray:

    """
    Construct a joint feature representation for a prompt-response pair.

    Both inputs must be numerical embeddings of equal dimension.

    The combined representation includes:

        1. Prompt embedding.
        2. Response embedding.
        3. Absolute difference between embeddings.
        4. Element-wise product between embeddings.

    If the original embedding dimension is D, the resulting
    feature dimension is 4D.

    Returns:
        Combined feature matrix of shape (n_pairs, 4 * D).
    """

    prompt_embeddings = np.asarray(
        prompt_embeddings,
        dtype=np.float32
    )

    response_embeddings = np.asarray(
        response_embeddings,
        dtype=np.float32
    )

    if prompt_embeddings.ndim != 2:
        raise ValueError(
            "Prompt embeddings must be a 2D matrix."
        )

    if response_embeddings.ndim != 2:
        raise ValueError(
            "Response embeddings must be a 2D matrix."
        )

    if prompt_embeddings.shape != response_embeddings.shape:
        raise ValueError(
            "Prompt and response embeddings must have identical shapes."
        )

    if not np.isfinite(prompt_embeddings).all():
        raise ValueError(
            "Prompt embeddings contain invalid values."
        )

    if not np.isfinite(response_embeddings).all():
        raise ValueError(
            "Response embeddings contain invalid values."
        )

    difference = np.abs(
        prompt_embeddings - response_embeddings
    )

    interaction = (
        prompt_embeddings * response_embeddings
    )

    combined_features = np.concatenate(
        [
            prompt_embeddings,
            response_embeddings,
            difference,
            interaction
        ],
        axis=1
    )

    return combined_features.astype(np.float32)


# ============================================================
# 3. MULTI-LAYER PERCEPTRON ARCHITECTURE
# ============================================================

class MLPScorer(nn.Module):

    """
    Multi-Layer Perceptron for relevance and compatibility scoring.

    Architecture:

        Input features
              |
              v
        Fully Connected Layer
              |
              v
        Layer Normalization
              |
              v
        ReLU Activation
              |
              v
        Dropout
              |
              v
        Additional Hidden Layers
              |
              v
        Output Linear Layer
              |
              v
        Prediction Logit

    A sigmoid function is applied during inference to obtain
    probabilities between 0 and 1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2
    ):

        super().__init__()

        if input_dim < 1:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError(
                "hidden_dims must contain positive dimensions."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.extend([

                nn.Linear(
                    previous_dim,
                    hidden_dim
                ),

                nn.LayerNorm(
                    hidden_dim
                ),

                nn.ReLU(),

                nn.Dropout(
                    dropout
                )

            ])

            previous_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            *layers
        )

        # Final layer produces one scoring logit per sample.
        self.output_layer = nn.Linear(
            previous_dim,
            1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        features = self.feature_extractor(x)

        logits = self.output_layer(features)

        return logits.squeeze(-1)

    def predict_proba(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """Convert logits into probabilities."""

        logits = self.forward(x)

        return torch.sigmoid(logits)


# ============================================================
# 4. SCORING DATASET
# ============================================================

class ScoringDataset(TensorDataset):

    """
    Dataset containing numerical feature vectors and binary labels.

    Labels:
        1 = Relevant / Compatible.
        0 = Irrelevant / Incompatible.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):

        feature_tensor = torch.tensor(
            features,
            dtype=torch.float32
        )

        label_tensor = torch.tensor(
            labels,
            dtype=torch.float32
        )

        super().__init__(
            feature_tensor,
            label_tensor
        )


# ============================================================
# 5. MLP SCORING PIPELINE
# ============================================================

class MLPScoringPipeline:

    """
    Complete training and inference pipeline.

    Supports:

        - Sample relevance scoring.
        - Prompt-response compatibility scoring.
        - Feature standardization.
        - Binary classification.
        - Class imbalance handling.
        - Early stopping.
        - Validation metrics.
        - Probability estimation.
        - Model checkpoint management.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
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

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

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

        self.hidden_dims = hidden_dims

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

        # Initialize model.
        self.model = MLPScorer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Binary classification loss.
        self.criterion = nn.BCEWithLogitsLoss()

        # Adam optimizer.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.mean = None

        self.std = None

        self.pos_weight = None

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

    def _validate_features(
        self,
        X: np.ndarray,
        min_samples: int = 1
    ) -> np.ndarray:

        X = np.asarray(
            X,
            dtype=np.float32
        )

        if X.ndim != 2:
            raise ValueError(
                "Features must be a 2D numerical matrix."
            )

        if X.shape[0] < min_samples:
            raise ValueError(
                f"At least {min_samples} samples are required."
            )

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {X.shape[1]}."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        return X

    def _validate_labels(
        self,
        y: np.ndarray,
        n_samples: int
    ) -> np.ndarray:

        y = np.asarray(
            y,
            dtype=np.float32
        ).reshape(-1)

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels must match number of samples."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Labels contain invalid values."
            )

        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError(
                "Labels must be binary: 0 or 1."
            )

        return y

    # --------------------------------------------------------
    # 5.2 STRATIFIED TRAINING / VALIDATION SPLIT
    # --------------------------------------------------------

    def _split_indices(
        self,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Split samples while preserving both classes
        in training and validation sets.
        """

        rng = np.random.default_rng(
            self.seed
        )

        train_indices = []
        val_indices = []

        for label in [0.0, 1.0]:

            class_indices = np.where(
                y == label
            )[0]

            if len(class_indices) < 2:
                raise ValueError(
                    "At least two samples per class are required "
                    "for stratified training and validation."
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
    # 5.3 FEATURE STANDARDIZATION
    # --------------------------------------------------------

    def _standardize(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        return (
            X - self.mean
        ) / self.std

    def _prepare_tensor(
        self,
        X: np.ndarray
    ) -> torch.Tensor:

        X = self._validate_features(X)

        X = self._standardize(X)

        return torch.tensor(
            X,
            dtype=torch.float32
        )

    # --------------------------------------------------------
    # 5.4 TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, list]:

        """
        Train the MLP scoring model.

        Parameters:
            X:
                Numerical input feature matrix.

            y:
                Binary labels.

        The same training pipeline is used for:

            A. Individual sample relevance prediction.

            B. Prompt-response compatibility prediction.

        The distinction depends on the supplied feature matrix
        and corresponding labels.
        """

        set_seed(self.seed)

        X = self._validate_features(
            X,
            min_samples=4
        )

        y = self._validate_labels(
            y,
            len(X)
        )

        # Both classes must be present.
        if len(np.unique(y)) != 2:
            raise ValueError(
                "Training requires both positive and negative labels."
            )

        # ----------------------------------------------------
        # Stratified split
        # ----------------------------------------------------

        train_indices, val_indices = (
            self._split_indices(y)
        )

        X_train = X[train_indices]

        X_val = X[val_indices]

        y_train = y[train_indices]

        y_val = y[val_indices]

        # ----------------------------------------------------
        # Fit normalization on training data only
        # ----------------------------------------------------

        self.mean = X_train.mean(
            axis=0,
            keepdims=True
        )

        self.std = X_train.std(
            axis=0,
            keepdims=True
        )

        self.std[self.std < 1e-8] = 1.0

        X_train = self._standardize(
            X_train
        )

        X_val = self._standardize(
            X_val
        )

        # ----------------------------------------------------
        # Handle class imbalance
        # ----------------------------------------------------

        n_positive = np.sum(
            y_train == 1
        )

        n_negative = np.sum(
            y_train == 0
        )

        self.pos_weight = float(
            n_negative / n_positive
        )

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [self.pos_weight],
                dtype=torch.float32,
                device=self.device
            )
        )

        # ----------------------------------------------------
        # Build datasets
        # ----------------------------------------------------

        train_dataset = ScoringDataset(
            X_train,
            y_train
        )

        val_dataset = ScoringDataset(
            X_val,
            y_val
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # ----------------------------------------------------
        # Early stopping initialization
        # ----------------------------------------------------

        best_val_loss = float("inf")

        best_weights = None

        epochs_without_improvement = 0

        print(f"Training device: {self.device}")

        print(f"Training samples: {len(train_dataset)}")

        print(f"Validation samples: {len(val_dataset)}")

        print(f"Positive class weight: {self.pos_weight:.4f}")

        print("-" * 85)

        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            # ================================================
            # TRAINING PHASE
            # ================================================

            for features, labels in train_loader:

                features = features.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                self.optimizer.zero_grad()

                logits = self.model(
                    features
                )

                loss = self.criterion(
                    logits,
                    labels
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

                self.optimizer.step()

                train_loss += (
                    loss.item() * features.size(0)
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

        # Restore best validation checkpoint.
        self.model.load_state_dict(
            best_weights
        )

        self.model.eval()

        self.is_fitted = True

        print("-" * 85)

        print(
            f"Best validation loss: {best_val_loss:.6f}"
        )

        print(
            "MLP scoring model training completed."
        )

        return self.history

    # --------------------------------------------------------
    # 5.5 VALIDATION AND EVALUATION
    # --------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:

        """Compute binary classification metrics."""

        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        y_true = y_true.astype(
            np.int64
        )

        true_positive = int(
            np.sum(
                (y_true == 1) & (predictions == 1)
            )
        )

        false_positive = int(
            np.sum(
                (y_true == 0) & (predictions == 1)
            )
        )

        false_negative = int(
            np.sum(
                (y_true == 1) & (predictions == 0)
            )
        )

        true_negative = int(
            np.sum(
                (y_true == 0) & (predictions == 0)
            )
        )

        accuracy = (
            true_positive + true_negative
        ) / max(len(y_true), 1)

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

        """Evaluate the model on validation data."""

        self.model.eval()

        total_loss = 0.0

        all_probabilities = []

        all_labels = []

        with torch.no_grad():

            for features, labels in loader:

                features = features.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                logits = self.model(
                    features
                )

                loss = self.criterion(
                    logits,
                    labels
                )

                probabilities = torch.sigmoid(
                    logits
                )

                total_loss += (
                    loss.item() * features.size(0)
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

        total_loss /= len(loader.dataset)

        metrics = self._compute_metrics(
            np.asarray(all_labels),
            np.asarray(all_probabilities),
            self.threshold
        )

        return total_loss, metrics

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:

        """Evaluate the trained model on an independent dataset."""

        self._check_fitted()

        X = self._validate_features(X)

        y = self._validate_labels(
            y,
            len(X)
        )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Evaluation requires both positive and negative labels."
            )

        probabilities = self.predict_scores(X)

        return self._compute_metrics(
            y,
            probabilities,
            self.threshold
        )

    # --------------------------------------------------------
    # 5.6 RELEVANCE / COMPATIBILITY SCORE PREDICTION
    # --------------------------------------------------------

    def predict_scores(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:

        """
        Predict probabilities between 0 and 1.

        Higher values indicate stronger predicted relevance
        or prompt-response compatibility.
        """

        self._check_fitted()

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        tensor = self._prepare_tensor(
            X
        )

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        all_scores = []

        with torch.no_grad():

            for (features,) in loader:

                features = features.to(
                    self.device
                )

                probabilities = self.model.predict_proba(
                    features
                )

                all_scores.append(
                    probabilities.cpu().numpy()
                )

        return np.concatenate(
            all_scores,
            axis=0
        )

    # --------------------------------------------------------
    # 5.7 BINARY PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:

        """
        Convert predicted probabilities into binary decisions.

        Returns:
            1 = Relevant / Compatible.
            0 = Irrelevant / Incompatible.
        """

        if threshold is None:
            threshold = self.threshold

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        scores = self.predict_scores(
            X
        )

        return (
            scores >= threshold
        ).astype(np.int64)

    # --------------------------------------------------------
    # 5.8 MODEL SAVING
    # --------------------------------------------------------

    def save(
        self,
        path: str
    ) -> None:

        """Save model parameters and preprocessing configuration."""

        self._check_fitted()

        os.makedirs(
            os.path.dirname(
                os.path.abspath(path)
            ),
            exist_ok=True
        )

        checkpoint = {

            "input_dim": self.input_dim,

            "hidden_dims": self.hidden_dims,

            "dropout": self.dropout,

            "model_state_dict": self.model.state_dict(),

            "mean": torch.tensor(
                self.mean
            ),

            "std": torch.tensor(
                self.std
            ),

            "pos_weight": self.pos_weight,

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
            f"MLP checkpoint saved to: {path}"
        )

    # --------------------------------------------------------
    # 5.9 MODEL LOADING
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None
    ):

        """Load a previously trained MLP scoring model."""

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

            hidden_dims=checkpoint["hidden_dims"],

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

        pipeline.mean = (
            checkpoint["mean"].cpu().numpy()
        )

        pipeline.std = (
            checkpoint["std"].cpu().numpy()
        )

        pipeline.pos_weight = checkpoint["pos_weight"]

        pipeline.history = checkpoint["history"]

        if pipeline.pos_weight is not None:

            pipeline.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [pipeline.pos_weight],
                    dtype=torch.float32,
                    device=pipeline.device
                )
            )

        pipeline.model.eval()

        pipeline.is_fitted = True

        return pipeline

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _check_fitted(self) -> None:

        if not self.is_fitted:

            raise RuntimeError(
                "The MLP must be trained or loaded before inference."
            )


# ============================================================
# 6. EXPORT SCORING RESULTS
# ============================================================

def export_results(
    pipeline: MLPScoringPipeline,
    X: np.ndarray,
    output_dir: str,
    sample_ids: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None
) -> None:

    """
    Export:
        - Predicted relevance / compatibility probabilities.
        - Binary predictions.
        - Scored sample CSV.
        - Evaluation metrics when labels are supplied.
        - Training history.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    X = pipeline._validate_features(X)

    # --------------------------------------------------------
    # Generate scores
    # --------------------------------------------------------

    scores = pipeline.predict_scores(X)

    predictions = (
        scores >= pipeline.threshold
    ).astype(np.int64)

    # --------------------------------------------------------
    # Export scores
    # --------------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            "scores.npy"
        ),
        scores
    )

    np.save(
        os.path.join(
            output_dir,
            "predictions.npy"
        ),
        predictions
    )

    # --------------------------------------------------------
    # Validate identifiers
    # --------------------------------------------------------

    if sample_ids is None:

        sample_ids = np.arange(
            len(X)
        ).astype(str)

    else:

        sample_ids = np.asarray(
            sample_ids
        ).reshape(-1)

        if len(sample_ids) != len(X):
            raise ValueError(
                "Number of sample IDs must match the number of features."
            )

    if labels is not None:

        labels = pipeline._validate_labels(
            labels,
            len(X)
        )

    # --------------------------------------------------------
    # Export CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        output_dir,
        "scoring_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "sample_id",
            "predicted_score",
            "predicted_label",
            "ground_truth"
        ])

        for i in range(len(X)):

            ground_truth = (
                int(labels[i])
                if labels is not None
                else ""
            )

            writer.writerow([
                str(sample_ids[i]),
                float(scores[i]),
                int(predictions[i]),
                ground_truth
            ])

    # --------------------------------------------------------
    # Export evaluation metrics
    # --------------------------------------------------------

    if labels is not None:

        metrics = pipeline.evaluate(
            X,
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

    print("\nScoring results exported successfully.")

    print(
        f"Number of scored samples: {len(scores)}"
    )

    print(
        f"Mean predicted score: {np.mean(scores):.4f}"
    )

    print(
        f"Number classified as positive: {np.sum(predictions)}"
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
            "Train an MLP for sample relevance and "
            "prompt-response compatibility scoring."
        )
    )

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an NPZ file containing features and labels."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/relevance_mlp"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "relevance",
            "prompt_response"
        ],
        default="relevance",
        help="Scoring task represented by the input features."
    )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128,64"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    # --------------------------------------------------------
    # Optimization
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
    # Reproducibility / device
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device: cpu or cuda."
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Load input data
    # --------------------------------------------------------

    with np.load(
        args.input,
        allow_pickle=False
    ) as data:

        if "features" not in data:
            raise KeyError(
                "Input NPZ must contain a 'features' array."
            )

        if "labels" not in data:
            raise KeyError(
                "Input NPZ must contain a 'labels' array."
            )

        X = data["features"]

        y = data["labels"]

        sample_ids = (
            data["ids"]
            if "ids" in data
            else None
        )

    if X.ndim != 2:
        raise ValueError(
            "Features must be a 2D numerical matrix."
        )

    # --------------------------------------------------------
    # Parse hidden dimensions
    # --------------------------------------------------------

    hidden_dims = [
        int(dim.strip())
        for dim in args.hidden_dims.split(",")
    ]

    # --------------------------------------------------------
    # Initialize MLP pipeline
    # --------------------------------------------------------

    pipeline = MLPScoringPipeline(

        input_dim=X.shape[1],

        hidden_dims=hidden_dims,

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
        X=X,
        y=y
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
            f"{args.task}_mlp_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # Export scoring results
    # --------------------------------------------------------

    export_results(
        pipeline=pipeline,
        X=X,
        output_dir=args.output_dir,
        sample_ids=sample_ids,
        labels=y
    )


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
