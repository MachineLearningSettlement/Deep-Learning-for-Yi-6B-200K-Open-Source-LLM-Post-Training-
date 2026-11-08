
"""
Siamese Neural Network for Semantic Similarity and Duplicate Detection
======================================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K Open-Source LLM

Purpose:
    1. Learn semantic representations of narrative samples.
    2. Measure semantic similarity between pairs of samples.
    3. Identify duplicate and near-duplicate samples.
    4. Support data redundancy detection during dataset curation.

Architecture:
    - Two shared-weight neural network branches.
    - Multi-layer perceptron encoder.
    - L2-normalized latent representations.
    - Contrastive loss for similarity learning.

Input:
    features.npy:
        Numerical feature matrix of shape (n_samples, input_dim).

    pairs.csv:
        Labeled sample pairs with three columns:
        sample_i, sample_j, label

        label = 1: semantically similar pair.
        label = 0: semantically dissimilar pair.

Output:
    - Trained Siamese Network checkpoint.
    - Learned sample representations.
    - Similarity scores.
    - Duplicate and near-duplicate candidate pairs.

Requirements:
    pip install torch numpy scikit-learn

Author:
    PlotDot-LLM Project
"""

import os
import csv
import copy
import argparse
import random

from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. SIAMESE NETWORK ARCHITECTURE
# ============================================================

class SampleEncoder(nn.Module):
    """
    Shared neural network encoder.

    Maps numerical sample representations into a compact
    latent embedding space.

    Input:
        (batch_size, input_dim)

    Output:
        (batch_size, embedding_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        dropout: float = 0.1
    ):

        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, embedding_dim)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embedding = self.network(x)

        # Normalize embeddings to enable cosine similarity.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network.

    Both branches share the same encoder and therefore
    the same trainable parameters.

    Two input samples are mapped into a common embedding
    space where their semantic similarity can be measured.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        dropout: float = 0.1
    ):

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        if input_dim < 1 or embedding_dim < 1:
            raise ValueError(
                "input_dim and embedding_dim must be positive."
            )

        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError(
                "hidden_dims must contain positive dimensions."
            )

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        # One encoder shared by both branches.
        self.encoder = SampleEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embedding_a = self.encoder(sample_a)
        embedding_b = self.encoder(sample_b)

        return embedding_a, embedding_b

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for individual samples."""

        return self.encoder(x)


# ============================================================
# 3. CONTRASTIVE LOSS
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss.

    Labels:
        1 = Similar pair.
        0 = Dissimilar pair.

    The loss:
        Similar pairs are encouraged to have small distances.

        Dissimilar pairs are encouraged to have distances
        greater than the specified margin.

    Formula:

        L = y * D^2
            + (1-y) * max(margin-D, 0)^2

    where:
        D = Euclidean distance between embeddings.
    """

    def __init__(self, margin: float = 1.0):

        super().__init__()

        if margin <= 0:
            raise ValueError("margin must be positive.")

        self.margin = margin

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        distances = F.pairwise_distance(
            embedding_a,
            embedding_b,
            p=2
        )

        similar_loss = labels * distances.pow(2)

        dissimilar_loss = (
            (1.0 - labels)
            * F.relu(self.margin - distances).pow(2)
        )

        return (similar_loss + dissimilar_loss).mean()


# ============================================================
# 4. PAIR DATASET
# ============================================================

class SiamesePairDataset(Dataset):
    """
    Dataset for Siamese Network training.

    Each item contains:
        - First sample feature vector.
        - Second sample feature vector.
        - Similarity label.
    """

    def __init__(
        self,
        features: np.ndarray,
        pairs: np.ndarray
    ):

        self.features = torch.tensor(
            features,
            dtype=torch.float32
        )

        self.pairs = torch.tensor(
            pairs[:, :2],
            dtype=torch.long
        )

        self.labels = torch.tensor(
            pairs[:, 2],
            dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):

        index_a, index_b = self.pairs[index]

        sample_a = self.features[index_a]
        sample_b = self.features[index_b]

        label = self.labels[index]

        return sample_a, sample_b, label


# ============================================================
# 5. SIAMESE NETWORK PIPELINE
# ============================================================

class SiamesePipeline:
    """
    Complete Siamese Network training and inference pipeline.

    Responsibilities:
        - Input validation.
        - Feature standardization.
        - Pair-based training.
        - Validation.
        - Early stopping.
        - Embedding extraction.
        - Similarity computation.
        - Redundancy detection.
        - Checkpoint saving and loading.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        margin: float = 1.0,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        seed: int = 42
    ):

        if input_dim < 1 or embedding_dim < 1:
            raise ValueError("Dimensions must be positive.")

        if hidden_dims is None:
            hidden_dims = [512, 256]

        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError("Invalid hidden dimensions.")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError("Training parameters must be positive.")

        if not 0 < validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.margin = margin

        self.validation_split = validation_split
        self.seed = seed

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = SiameseNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        ).to(self.device)

        self.criterion = ContrastiveLoss(
            margin=margin
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.mean = None
        self.std = None

        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_f1": []
        }

        self.is_fitted = False

    # --------------------------------------------------------
    # 5.1 INPUT VALIDATION
    # --------------------------------------------------------

    def _validate_features(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(
                "Features must be a 2D numerical matrix."
            )

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {X.shape[1]}."
            )

        if X.shape[0] < 2:
            raise ValueError("At least two samples are required.")

        if not np.isfinite(X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        return X

    def _validate_pairs(
        self,
        pairs: np.ndarray,
        n_samples: int
    ) -> np.ndarray:

        pairs = np.asarray(pairs)

        if pairs.ndim != 2 or pairs.shape[1] != 3:
            raise ValueError(
                "Pairs must have three columns: "
                "sample_i, sample_j, label."
            )

        if len(pairs) < 2:
            raise ValueError("At least two labeled pairs are required.")

        if not np.issubdtype(pairs.dtype, np.number):
            raise ValueError("Pair data must be numerical.")

        if not np.isfinite(pairs).all():
            raise ValueError("Pair data contains invalid values.")

        indices = pairs[:, :2]

        if not np.all(indices == np.floor(indices)):
            raise ValueError("Sample indices must be integers.")

        indices = indices.astype(np.int64)

        if np.any(indices < 0) or np.any(indices >= n_samples):
            raise ValueError(
                "Pair indices refer to samples outside the feature matrix."
            )

        labels = pairs[:, 2].astype(np.float32)

        if not np.isin(labels, [0, 1]).all():
            raise ValueError(
                "Labels must be 0 (dissimilar) or 1 (similar)."
            )

        if np.any(indices[:, 0] == indices[:, 1]):
            raise ValueError(
                "A sample cannot be paired with itself."
            )

        result = np.column_stack([
            indices,
            labels
        ]).astype(np.float32)

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Training requires both similar and dissimilar pairs."
            )

        return result

    # --------------------------------------------------------
    # 5.2 FEATURE PREPROCESSING
    # --------------------------------------------------------

    def _standardize(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        return (X - self.mean) / self.std

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
    # 5.3 TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        pairs: np.ndarray
    ) -> Dict[str, list]:

        """
        Train the Siamese Network using labeled sample pairs.

        Parameters:
            X:
                Feature matrix of shape (n_samples, input_dim).

            pairs:
                Matrix of shape (n_pairs, 3).

                Columns:
                    sample_i, sample_j, label

                Label 1 = similar.
                Label 0 = dissimilar.
        """

        set_seed(self.seed)

        X = self._validate_features(X)

        pairs = self._validate_pairs(
            pairs,
            n_samples=len(X)
        )

        # Pair-level split into training and validation.
        rng = np.random.default_rng(self.seed)

        indices = rng.permutation(len(pairs))

        n_val = max(
            1,
            int(len(pairs) * self.validation_split)
        )

        n_val = min(n_val, len(pairs) - 1)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_pairs = pairs[train_indices]
        val_pairs = pairs[val_indices]

        # Both classes must be represented in each split.
        if (
            len(np.unique(train_pairs[:, 2])) < 2
            or len(np.unique(val_pairs[:, 2])) < 2
        ):
            raise ValueError(
                "Both training and validation splits must contain "
                "similar and dissimilar pairs. Provide more balanced "
                "labeled pairs or adjust validation_split."
            )

        # Fit normalization parameters on samples used in
        # training pairs only.
        training_sample_indices = np.unique(
            train_pairs[:, :2].astype(np.int64)
        )

        X_train_samples = X[training_sample_indices]

        self.mean = X_train_samples.mean(
            axis=0,
            keepdims=True
        )

        self.std = X_train_samples.std(
            axis=0,
            keepdims=True
        )

        self.std[self.std < 1e-8] = 1.0

        X_scaled = self._standardize(X)

        train_dataset = SiamesePairDataset(
            X_scaled,
            train_pairs
        )

        val_dataset = SiamesePairDataset(
            X_scaled,
            val_pairs
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

        best_val_loss = float("inf")
        best_weights = None

        epochs_without_improvement = 0

        print(f"Training device: {self.device}")
        print(f"Training pairs: {len(train_dataset)}")
        print(f"Validation pairs: {len(val_dataset)}")
        print("-" * 75)

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            for sample_a, sample_b, labels in train_loader:

                sample_a = sample_a.to(self.device)
                sample_b = sample_b.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                embedding_a, embedding_b = self.model(
                    sample_a,
                    sample_b
                )

                loss = self.criterion(
                    embedding_a,
                    embedding_b,
                    labels
                )

                loss.backward()

                self.optimizer.step()

                train_loss += (
                    loss.item() * sample_a.size(0)
                )

            train_loss /= len(train_dataset)

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_loss, val_accuracy, val_f1 = self._evaluate_loader(
                val_loader
            )

            self.history["train_loss"].append(train_loss)

            self.history["validation_loss"].append(val_loss)

            self.history["validation_accuracy"].append(
                val_accuracy
            )

            self.history["validation_f1"].append(val_f1)

            print(
                f"Epoch [{epoch + 1:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Accuracy: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

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

        self.model.load_state_dict(best_weights)

        self.model.eval()

        self.is_fitted = True

        print("-" * 75)
        print(
            f"Best validation loss: {best_val_loss:.6f}"
        )
        print("Siamese Network training completed.")

        return self.history

    # --------------------------------------------------------
    # 5.4 VALIDATION METRICS
    # --------------------------------------------------------

    def _evaluate_loader(
        self,
        loader: DataLoader,
        threshold: float = 0.5
    ) -> Tuple[float, float, float]:

        self.model.eval()

        total_loss = 0.0

        all_labels = []
        all_predictions = []

        with torch.no_grad():

            for sample_a, sample_b, labels in loader:

                sample_a = sample_a.to(self.device)
                sample_b = sample_b.to(self.device)
                labels = labels.to(self.device)

                embedding_a, embedding_b = self.model(
                    sample_a,
                    sample_b
                )

                loss = self.criterion(
                    embedding_a,
                    embedding_b,
                    labels
                )

                total_loss += (
                    loss.item() * sample_a.size(0)
                )

                similarities = F.cosine_similarity(
                    embedding_a,
                    embedding_b,
                    dim=1
                )

                predictions = (
                    similarities >= threshold
                ).float()

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

        total_loss /= len(loader.dataset)

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        return total_loss, accuracy, f1

    # --------------------------------------------------------
    # 5.5 EMBEDDING EXTRACTION
    # --------------------------------------------------------

    def encode(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:

        """
        Generate a latent embedding for every sample.

        Returns:
            Matrix of shape (n_samples, embedding_dim).
        """

        self._check_fitted()

        tensor = self._prepare_tensor(X)

        loader = DataLoader(
            torch.utils.data.TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        embeddings = []

        with torch.no_grad():

            for (batch,) in loader:

                batch = batch.to(self.device)

                latent = self.model.encode(batch)

                embeddings.append(
                    latent.cpu().numpy()
                )

        return np.concatenate(
            embeddings,
            axis=0
        )

    # --------------------------------------------------------
    # 5.6 PAIRWISE SIMILARITY
    # --------------------------------------------------------

    def similarity(
        self,
        X: np.ndarray,
        sample_i: int,
        sample_j: int
    ) -> float:

        """
        Calculate cosine similarity between two samples.

        Returns a value between -1 and 1.
        """

        self._check_fitted()

        X = self._validate_features(X)

        n_samples = len(X)

        if not (
            0 <= sample_i < n_samples
            and 0 <= sample_j < n_samples
        ):
            raise IndexError(
                "Sample indices are outside the feature matrix."
            )

        if sample_i == sample_j:
            return 1.0

        embeddings = self.encode(
            X[[sample_i, sample_j]]
        )

        similarity_score = np.dot(
            embeddings[0],
            embeddings[1]
        )

        return float(
            np.clip(similarity_score, -1.0, 1.0)
        )

    # --------------------------------------------------------
    # 5.7 REDUNDANCY DETECTION
    # --------------------------------------------------------

    def find_redundant_pairs(
        self,
        X: np.ndarray,
        threshold: float = 0.90,
        chunk_size: int = 512
    ) -> List[Tuple[int, int, float]]:

        """
        Detect potentially redundant samples.

        Similarity is calculated using cosine similarity
        between learned sample embeddings.

        Returns:
            (sample_i, sample_j, similarity_score)

        Chunking limits the memory required for pairwise
        similarity calculations.
        """

        self._check_fitted()

        if not -1 <= threshold <= 1:
            raise ValueError(
                "threshold must be between -1 and 1."
            )

        if chunk_size < 1:
            raise ValueError(
                "chunk_size must be positive."
            )

        embeddings = self.encode(X)

        n_samples = len(embeddings)

        redundant_pairs = []

        # Embeddings are already L2-normalized.
        for start in range(
            0,
            n_samples,
            chunk_size
        ):

            end = min(
                start + chunk_size,
                n_samples
            )

            similarities = (
                embeddings[start:end] @ embeddings.T
            )

            for local_i in range(end - start):

                i = start + local_i

                # Exclude self-matches and avoid duplicate pairs.
                candidate_indices = np.where(
                    similarities[
                        local_i,
                        i + 1:
                    ] >= threshold
                )[0] + i + 1

                for j in candidate_indices:

                    score = float(
                        similarities[local_i, j]
                    )

                    redundant_pairs.append(
                        (i, int(j), score)
                    )

        return redundant_pairs

    # --------------------------------------------------------
    # 5.8 SAVE MODEL
    # --------------------------------------------------------

    def save(self, path: str) -> None:

        """Save trained weights and preprocessing parameters."""

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

            "embedding_dim": self.embedding_dim,

            "dropout": self.dropout,

            "margin": self.margin,

            "model_state_dict": self.model.state_dict(),

            "mean": torch.tensor(self.mean),

            "std": torch.tensor(self.std),

            "history": self.history

        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"Model checkpoint saved: {path}"
        )

    # --------------------------------------------------------
    # 5.9 LOAD MODEL
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None
    ):

        """Load a previously trained Siamese Network."""

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
            embedding_dim=checkpoint["embedding_dim"],
            dropout=checkpoint["dropout"],
            margin=checkpoint["margin"],
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

        pipeline.history = checkpoint["history"]

        pipeline.model.eval()

        pipeline.is_fitted = True

        return pipeline

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _check_fitted(self):

        if not self.is_fitted:

            raise RuntimeError(
                "The Siamese Network must be trained "
                "or loaded before inference."
            )


# ============================================================
# 6. EXPORT RESULTS
# ============================================================

def export_results(
    pipeline: SiamesePipeline,
    X: np.ndarray,
    output_dir: str,
    threshold: float = 0.90
) -> None:

    """
    Export:
        - Learned latent representations.
        - Similarity candidate pairs.
        - Redundancy detection results.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract latent representations
    # --------------------------------------------------------

    embeddings = pipeline.encode(X)

    np.save(
        os.path.join(
            output_dir,
            "siamese_embeddings.npy"
        ),
        embeddings
    )

    # --------------------------------------------------------
    # Identify redundant samples
    # --------------------------------------------------------

    redundant_pairs = pipeline.find_redundant_pairs(
        X,
        threshold=threshold
    )

    output_path = os.path.join(
        output_dir,
        "redundancy_candidates.csv"
    )

    with open(
        output_path,
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

        writer.writerows(
            redundant_pairs
        )

    print("\nExport completed.")

    print(
        f"Learned embeddings: {embeddings.shape}"
    )

    print(
        f"Redundancy candidates: {len(redundant_pairs)}"
    )

    print(
        f"Results directory: {output_dir}"
    )


# ============================================================
# 7. LOAD LABELED PAIRS
# ============================================================

def load_pairs_csv(path: str) -> np.ndarray:

    """
    Load labeled pairs from CSV.

    Expected columns:
        sample_i, sample_j, label

    The first row must contain column headers.
    """

    pairs = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "sample_i",
            "sample_j",
            "label"
        }

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):

            raise ValueError(
                "CSV must contain: sample_i, sample_j, label."
            )

        for row in reader:

            pairs.append([
                int(row["sample_i"]),
                int(row["sample_j"]),
                int(row["label"])
            ])

    if not pairs:
        raise ValueError("No labeled pairs found in CSV.")

    return np.asarray(
        pairs,
        dtype=np.float32
    )


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Siamese Network for semantic similarity "
            "and duplicate detection."
        )
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to the numerical feature matrix (.npy)."
    )

    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to the labeled pairs CSV file."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/siamese_network"
    )

    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,256"
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

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
        "--margin",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # Load feature matrix
    # --------------------------------------------------------

    X = np.load(args.features)

    if X.ndim != 2:
        raise ValueError(
            "The input feature file must contain a 2D matrix."
        )

    # --------------------------------------------------------
    # Load labeled sample pairs
    # --------------------------------------------------------

    pairs = load_pairs_csv(args.pairs)

    # --------------------------------------------------------
    # Parse hidden dimensions
    # --------------------------------------------------------

    hidden_dims = [
        int(dim.strip())
        for dim in args.hidden_dims.split(",")
    ]

    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    pipeline = SiamesePipeline(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        margin=args.margin,
        validation_split=args.validation_split,
        seed=args.seed
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X=X,
        pairs=pairs
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
            "siamese_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # Export representations and redundancy results
    # --------------------------------------------------------

    export_results(
        pipeline=pipeline,
        X=X,
        output_dir=args.output_dir,
        threshold=args.threshold
    )


if __name__ == "__main__":

    main()
