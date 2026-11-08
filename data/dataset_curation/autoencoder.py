
"""
Autoencoder for Sample Representation Learning and Redundancy Detection
======================================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K Open-Source LLM

Purpose:
    1. Learn compact latent representations of dataset samples.
    2. Reconstruct input feature vectors.
    3. Measure reconstruction errors.
    4. Identify potentially redundant samples using cosine similarity
       between learned latent representations.

Input:
    A numerical feature matrix X of shape (n_samples, n_features).
    For textual data, feature vectors should first be generated using
    a text embedding model.

Output:
    - Trained Autoencoder checkpoint.
    - Compressed latent representations.
    - Reconstruction errors.
    - Redundancy candidate pairs.

Requirements:
    pip install torch numpy
"""

import os
import csv
import copy
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
# 2. AUTOENCODER ARCHITECTURE
# ============================================================

class Encoder(nn.Module):
    """
    Encoder:
        Maps input feature vectors into a compact latent space.

    Input:
        (batch_size, input_dim)

    Output:
        (batch_size, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
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

        layers.append(nn.Linear(previous_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder:
        Reconstructs the original input from its latent representation.

    Input:
        (batch_size, latent_dim)

    Output:
        (batch_size, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        decoder_dims = list(reversed(hidden_dims))

        previous_dim = latent_dim

        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, input_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Autoencoder(nn.Module):
    """
    Complete Autoencoder architecture.

    The encoder learns compressed representations,
    while the decoder reconstructs the original samples.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract compact latent representations."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct samples from latent representations."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        reconstructed = self.decode(z)

        return reconstructed, z


# ============================================================
# 3. AUTOENCODER TRAINING PIPELINE
# ============================================================

class AutoencoderPipeline:
    """
    Handles preprocessing, training, validation,
    representation extraction and redundancy detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        seed: int = 42
    ):
        if input_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive.")

        if hidden_dims is None:
            hidden_dims = [512, 256]

        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")

        if batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError("batch_size, epochs and patience must be positive.")

        if not 0 < validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.seed = seed

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.mean = None
        self.std = None
        self.history = {
            "train_loss": [],
            "validation_loss": []
        }

        self.is_fitted = False

    # --------------------------------------------------------
    # 3.1 DATA VALIDATION AND PREPROCESSING
    # --------------------------------------------------------

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("Input must be a 2D feature matrix.")

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {X.shape[1]}."
            )

        if X.shape[0] < 2:
            raise ValueError("At least two samples are required.")

        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values.")

        return X

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def _prepare_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = self._validate_data(X)
        X = self._standardize(X)

        return torch.tensor(X, dtype=torch.float32)

    # --------------------------------------------------------
    # 3.2 TRAINING
    # --------------------------------------------------------

    def fit(self, X: np.ndarray) -> dict:
        """
        Train the Autoencoder.

        X:
            Numerical feature matrix (n_samples, input_dim).
        """

        set_seed(self.seed)

        X = self._validate_data(X)

        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        n_val = max(
            1,
            int(n_samples * self.validation_split)
        )

        n_val = min(n_val, n_samples - 1)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X[train_indices]
        X_val = X[val_indices]

        # Fit preprocessing parameters on training data only.
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

        X_train = self._standardize(X_train)
        X_val = self._standardize(X_val)

        train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        val_tensor = torch.tensor(
            X_val,
            dtype=torch.float32
        )

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        print(f"Training on: {self.device}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print("-" * 55)

        for epoch in range(self.epochs):

            # Training phase
            self.model.train()
            train_loss = 0.0

            for (batch,) in train_loader:

                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                reconstructed, _ = self.model(batch)

                loss = self.criterion(reconstructed, batch)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch.size(0)

            train_loss /= len(train_tensor)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for (batch,) in val_loader:

                    batch = batch.to(self.device)

                    reconstructed, _ = self.model(batch)

                    loss = self.criterion(reconstructed, batch)

                    val_loss += loss.item() * batch.size(0)

            val_loss /= len(val_tensor)

            self.history["train_loss"].append(train_loss)
            self.history["validation_loss"].append(val_loss)

            print(
                f"Epoch [{epoch + 1:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: {val_loss:.6f}"
            )

            # Early stopping
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
            raise RuntimeError("Training did not produce a valid checkpoint.")

        self.model.load_state_dict(best_weights)
        self.model.eval()

        self.is_fitted = True

        print("-" * 55)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("Autoencoder training completed.")

        return self.history

    # --------------------------------------------------------
    # 3.3 LATENT REPRESENTATION EXTRACTION
    # --------------------------------------------------------

    def encode(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:
        """
        Transform input samples into compact latent vectors.

        Returns:
            (n_samples, latent_dim)
        """

        self._check_fitted()

        tensor = self._prepare_tensor(X)

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()
        representations = []

        with torch.no_grad():

            for (batch,) in loader:

                batch = batch.to(self.device)

                z = self.model.encode(batch)

                representations.append(
                    z.cpu().numpy()
                )

        return np.concatenate(representations, axis=0)

    # --------------------------------------------------------
    # 3.4 SAMPLE RECONSTRUCTION
    # --------------------------------------------------------

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input samples in their original scale."""

        self._check_fitted()

        tensor = self._prepare_tensor(X).to(self.device)

        self.model.eval()

        with torch.no_grad():
            reconstructed, _ = self.model(tensor)

        reconstructed = reconstructed.cpu().numpy()

        return reconstructed * self.std + self.mean

    # --------------------------------------------------------
    # 3.5 RECONSTRUCTION ERROR
    # --------------------------------------------------------

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample mean squared reconstruction error.

        Larger errors indicate samples that are less accurately
        represented by the learned Autoencoder.
        """

        self._check_fitted()

        X = self._validate_data(X)
        X_scaled = self._standardize(X)

        reconstructed = (
            self.reconstruct(X) - self.mean
        ) / self.std

        errors = np.mean(
            np.square(X_scaled - reconstructed),
            axis=1
        )

        return errors

    # --------------------------------------------------------
    # 3.6 REDUNDANCY DETECTION
    # --------------------------------------------------------

    def find_redundant_pairs(
        self,
        X: np.ndarray,
        threshold: float = 0.95,
        chunk_size: int = 512
    ) -> List[Tuple[int, int, float]]:
        """
        Identify potentially redundant samples using cosine
        similarity between their latent representations.

        threshold:
            Similarity threshold between 0 and 1.

        Returns:
            List of (sample_i, sample_j, similarity).
        """

        self._check_fitted()

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        latent = self.encode(X)

        norms = np.linalg.norm(latent, axis=1, keepdims=True)

        normalized = latent / np.maximum(norms, 1e-12)

        n_samples = len(normalized)
        redundant_pairs = []

        # Chunking reduces peak memory usage.
        for start in range(0, n_samples, chunk_size):

            end = min(start + chunk_size, n_samples)

            similarities = (
                normalized[start:end] @ normalized.T
            )

            for local_i in range(end - start):

                i = start + local_i

                # Keep each pair only once and exclude self-matches.
                candidate_indices = np.where(
                    similarities[local_i, i + 1:] >= threshold
                )[0] + i + 1

                for j in candidate_indices:

                    similarity = float(
                        similarities[local_i, j]
                    )

                    redundant_pairs.append(
                        (i, int(j), similarity)
                    )

        return redundant_pairs

    # --------------------------------------------------------
    # 3.7 MODEL SAVING
    # --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model parameters and preprocessing configuration."""

        self._check_fitted()

        os.makedirs(
            os.path.dirname(os.path.abspath(path)),
            exist_ok=True
        )

        checkpoint = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "model_state_dict": self.model.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "history": self.history
        }

        torch.save(checkpoint, path)

        print(f"Model saved to: {path}")

    # --------------------------------------------------------
    # 3.8 MODEL LOADING
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None
    ):
        """Load a previously trained Autoencoder."""

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
            latent_dim=checkpoint["latent_dim"],
            dropout=checkpoint["dropout"],
            device=device
        )

        pipeline.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        pipeline.mean = checkpoint["mean"]
        pipeline.std = checkpoint["std"]
        pipeline.history = checkpoint["history"]

        pipeline.model.eval()
        pipeline.is_fitted = True

        return pipeline

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "The Autoencoder must be trained or loaded first."
            )


# ============================================================
# 4. EXPORT RESULTS
# ============================================================

def export_results(
    pipeline: AutoencoderPipeline,
    X: np.ndarray,
    output_dir: str,
    threshold: float = 0.95
) -> None:
    """
    Export latent representations, reconstruction errors
    and redundancy candidate pairs.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Extract representations
    latent = pipeline.encode(X)

    np.save(
        os.path.join(output_dir, "latent_representations.npy"),
        latent
    )

    # Compute reconstruction errors
    errors = pipeline.reconstruction_error(X)

    np.save(
        os.path.join(output_dir, "reconstruction_errors.npy"),
        errors
    )

    # Detect redundant samples
    pairs = pipeline.find_redundant_pairs(
        X,
        threshold=threshold
    )

    pairs_path = os.path.join(
        output_dir,
        "redundancy_candidates.csv"
    )

    with open(pairs_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "sample_id_1",
            "sample_id_2",
            "cosine_similarity"
        ])

        writer.writerows(pairs)

    print("\nExport completed:")
    print(f"Latent representations: {latent.shape}")
    print(f"Redundancy candidates: {len(pairs)}")
    print(f"Output directory: {output_dir}")


# ============================================================
# 5. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train an Autoencoder for sample representation "
            "learning and redundancy detection."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .npy feature matrix."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/autoencoder",
        help="Directory for model and exported results."
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64
    )

    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,256"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for redundancy candidates."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Load numerical feature matrix
    X = np.load(args.input)

    hidden_dims = [
        int(dim.strip())
        for dim in args.hidden_dims.split(",")
    ]

    # Initialize pipeline
    pipeline = AutoencoderPipeline(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed
    )

    # Train
    pipeline.fit(X)

    # Save trained model
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline.save(
        os.path.join(
            args.output_dir,
            "autoencoder_checkpoint.pth"
        )
    )

    # Export learned representations and redundancy candidates
    export_results(
        pipeline,
        X,
        args.output_dir,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()

"""

1. How to execute the script:

Once a numerical feature matrix is available as features.npy, execute:
python dataset_curation/autoencoder.py \
    --input data/features.npy \
    --latent-dim 64 \
    --hidden-dims 512,256 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --threshold 0.95
    
2. Expected outputs

After execution:
outputs/
└── autoencoder/
    ├── autoencoder_checkpoint.pth
    ├── latent_representations.npy
    ├── reconstruction_errors.npy
    └── redundancy_candidates.csv
    
"""
