
"""
Variational Autoencoder for Latent-Space Diversity Analysis
============================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    1. Learn probabilistic latent representations of narrative samples.
    2. Model the distribution of samples in a continuous latent space.
    3. Analyze latent-space diversity.
    4. Measure posterior uncertainty.
    5. Identify how samples are distributed around the latent-space
       centroid.
    6. Generate new feature representations by sampling from
       the learned latent distribution.

Architecture:
    - Variational Encoder.
    - Latent mean (mu) estimation.
    - Latent log-variance (logvar) estimation.
    - Reparameterization trick.
    - Variational Decoder.
    - Beta-ELBO objective (reconstruction + KL divergence).

Input:
    A numerical feature matrix:
        X.shape = (n_samples, input_dim)

    For textual narrative data, feature vectors must first be
    generated using a text embedding model.

Output:
    - Trained VAE checkpoint.
    - Latent mean representations.
    - Latent log-variance representations.
    - Reconstruction errors.
    - Latent diversity scores.
    - Posterior uncertainty scores.
    - Diversity analysis report.

Requirements:
    pip install torch numpy

Author:
    PlotDot-LLM Project
"""

import os
import json
import copy
import argparse
import random

from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 2. VARIATIONAL ENCODER
# ============================================================

class VariationalEncoder(nn.Module):
    """
    Encoder producing the parameters of a Gaussian
    latent distribution.

    Instead of directly learning a single latent vector,
    the encoder estimates:

        mu     : Latent mean.
        logvar : Latent log-variance.

    This defines the approximate posterior distribution:

        q(z|x) = N(mu, diag(exp(logvar)))

    Input:
        (batch_size, input_dim)

    Outputs:
        mu     : (batch_size, latent_dim)
        logvar : (batch_size, latent_dim)
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

        self.feature_extractor = nn.Sequential(*layers)

        # Separate projections for the distribution parameters.
        self.fc_mu = nn.Linear(
            previous_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            previous_dim,
            latent_dim
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.feature_extractor(x)

        mu = self.fc_mu(features)

        logvar = self.fc_logvar(features)

        # Prevent numerical instability during exponentiation.
        logvar = torch.clamp(
            logvar,
            min=-20.0,
            max=10.0
        )

        return mu, logvar


# ============================================================
# 3. VARIATIONAL DECODER
# ============================================================

class VariationalDecoder(nn.Module):
    """
    Decoder reconstructing input samples from latent vectors.

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

        # Linear output for continuous numerical features.
        layers.append(
            nn.Linear(previous_dim, input_dim)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        return self.network(z)


# ============================================================
# 4. COMPLETE VARIATIONAL AUTOENCODER
# ============================================================

class VariationalAutoencoder(nn.Module):
    """
    Complete VAE architecture.

    Pipeline:

        Input sample
             |
             v
        Variational Encoder
             |
          +--+--+
          |     |
         mu   logvar
          |     |
          +--+--+
             |
      Reparameterization
             |
             v
         Latent z
             |
             v
         Decoder
             |
             v
       Reconstruction

    The reparameterization trick allows gradients to propagate
    through stochastic latent sampling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        dropout: float = 0.1
    ):

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        if input_dim < 1 or latent_dim < 1:
            raise ValueError(
                "input_dim and latent_dim must be positive."
            )

        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError(
                "hidden_dims must contain positive dimensions."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.encoder = VariationalEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = VariationalDecoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.input_dim = input_dim
        self.latent_dim = latent_dim

    # --------------------------------------------------------
    # 4.1 ENCODING
    # --------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Estimate the parameters of q(z|x)."""

        return self.encoder(x)

    # --------------------------------------------------------
    # 4.2 REPARAMETERIZATION TRICK
    # --------------------------------------------------------

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:

        """
        Sample a latent vector using:

            z = mu + sigma * epsilon

            epsilon ~ N(0, I)
            sigma = exp(0.5 * logvar)

        This formulation preserves differentiability
        with respect to mu and logvar.
        """

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        return z

    # --------------------------------------------------------
    # 4.3 DECODING
    # --------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        """Reconstruct feature vectors from latent samples."""

        return self.decoder(z)

    # --------------------------------------------------------
    # 4.4 FORWARD PASS
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu, logvar = self.encode(x)

        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            # Deterministic reconstruction from posterior mean.
            z = mu

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    # --------------------------------------------------------
    # 4.5 PRIOR SAMPLING
    # --------------------------------------------------------

    def sample_prior(
        self,
        n_samples: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:

        """
        Sample latent vectors from the standard Gaussian prior:

            p(z) = N(0, I)
        """

        if n_samples < 1:
            raise ValueError(
                "n_samples must be positive."
            )

        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(
            n_samples,
            self.latent_dim,
            device=device
        )

        return self.decode(z)


# ============================================================
# 5. VAE LOSS FUNCTION
# ============================================================

class VAELoss(nn.Module):
    """
    Variational Autoencoder objective.

    Objective:

        L = L_reconstruction + beta * L_KL

    Reconstruction loss:

        L_reconstruction =
            E_q(z|x) [ -log p(x|z) ]

    Implemented using squared reconstruction error
    for continuous numerical features.

    KL divergence:

        KL(q(z|x) || p(z))

        = -0.5 * SUM(
            1 + logvar - mu^2 - exp(logvar)
        )

    where:

        q(z|x) = N(mu, diag(exp(logvar)))
        p(z)   = N(0, I)

    beta:
        Controls the relative importance of latent
        regularization.
    """

    def __init__(self, beta: float = 1.0):

        super().__init__()

        if beta < 0:
            raise ValueError(
                "beta must be non-negative."
            )

        self.beta = beta

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = target.size(0)

        # ----------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------

        reconstruction_loss = F.mse_loss(
            reconstruction,
            target,
            reduction="sum"
        ) / batch_size

        # ----------------------------------------------------
        # KL divergence
        # ----------------------------------------------------

        kl_divergence = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / batch_size

        # ----------------------------------------------------
        # Total objective
        # ----------------------------------------------------

        total_loss = (
            reconstruction_loss
            + self.beta * kl_divergence
        )

        return (
            total_loss,
            reconstruction_loss,
            kl_divergence
        )


# ============================================================
# 6. VAE TRAINING PIPELINE
# ============================================================

class VAEPipeline:
    """
    Complete training and inference pipeline.

    Includes:

        - Input validation.
        - Feature standardization.
        - Training/validation split.
        - VAE optimization.
        - Reconstruction and KL loss monitoring.
        - Early stopping.
        - Latent representation extraction.
        - Reconstruction.
        - Latent diversity analysis.
        - Posterior uncertainty analysis.
        - Prior sampling.
        - Model checkpoint management.
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
        beta: float = 1.0,
        validation_split: float = 0.2,
        min_delta: float = 0.0,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        seed: int = 42
    ):

        if input_dim < 1 or latent_dim < 1:
            raise ValueError(
                "Dimensions must be positive."
            )

        if hidden_dims is None:
            hidden_dims = [512, 256]

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError(
                "batch_size, epochs and patience must be positive."
            )

        if not 0 < validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative."
            )

        if gradient_clip <= 0:
            raise ValueError(
                "gradient_clip must be positive."
            )

        set_seed(seed)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.epochs = epochs
        self.patience = patience
        self.beta = beta

        self.validation_split = validation_split
        self.min_delta = min_delta
        self.gradient_clip = gradient_clip

        self.seed = seed

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        # Initialize architecture.
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        ).to(self.device)

        # Variational objective.
        self.criterion = VAELoss(
            beta=beta
        )

        # Optimizer.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Training statistics.
        self.mean = None
        self.std = None

        self.history = {
            "train_loss": [],
            "train_reconstruction_loss": [],
            "train_kl_loss": [],
            "validation_loss": [],
            "validation_reconstruction_loss": [],
            "validation_kl_loss": []
        }

        self.is_fitted = False

    # --------------------------------------------------------
    # 6.1 INPUT VALIDATION
    # --------------------------------------------------------

    def _validate_data(
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
                "Input must be a 2D numerical feature matrix."
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
                "Input contains NaN or infinite values."
            )

        return X

    # --------------------------------------------------------
    # 6.2 FEATURE STANDARDIZATION
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

        X = self._validate_data(X)

        X = self._standardize(X)

        return torch.tensor(
            X,
            dtype=torch.float32
        )

    # --------------------------------------------------------
    # 6.3 TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        X: np.ndarray
    ) -> Dict[str, list]:

        """
        Train the Variational Autoencoder.

        Parameters:
            X:
                Numerical feature matrix.

        The training procedure includes:
            - Random training/validation splitting.
            - Standardization.
            - VAE optimization.
            - KL divergence regularization.
            - Early stopping.
        """

        set_seed(self.seed)

        X = self._validate_data(
            X,
            min_samples=2
        )

        # ----------------------------------------------------
        # Training / validation split
        # ----------------------------------------------------

        n_samples = len(X)

        indices = np.random.permutation(
            n_samples
        )

        n_val = max(
            1,
            int(n_samples * self.validation_split)
        )

        n_val = min(
            n_val,
            n_samples - 1
        )

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X[train_indices]
        X_val = X[val_indices]

        # Fit normalization parameters only on training data.
        self.mean = X_train.mean(
            axis=0,
            keepdims=True
        )

        self.std = X_train.std(
            axis=0,
            keepdims=True
        )

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

        # ----------------------------------------------------
        # Early stopping initialization
        # ----------------------------------------------------

        best_val_loss = float("inf")

        best_weights = None

        epochs_without_improvement = 0

        print(f"Training device: {self.device}")
        print(f"Training samples: {len(train_tensor)}")
        print(f"Validation samples: {len(val_tensor)}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Beta: {self.beta}")
        print("-" * 85)

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(self.epochs):

            self.model.train()

            train_total = 0.0
            train_reconstruction = 0.0
            train_kl = 0.0

            # ================================================
            # TRAINING PHASE
            # ================================================

            for (batch,) in train_loader:

                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                reconstruction, mu, logvar = self.model(
                    batch,
                    sample=True
                )

                total_loss, recon_loss, kl_loss = (
                    self.criterion(
                        reconstruction,
                        batch,
                        mu,
                        logvar
                    )
                )

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

                self.optimizer.step()

                current_batch_size = batch.size(0)

                train_total += (
                    total_loss.item() * current_batch_size
                )

                train_reconstruction += (
                    recon_loss.item() * current_batch_size
                )

                train_kl += (
                    kl_loss.item() * current_batch_size
                )

            train_total /= len(train_tensor)

            train_reconstruction /= len(train_tensor)

            train_kl /= len(train_tensor)

            # ================================================
            # VALIDATION PHASE
            # ================================================

            (
                val_total,
                val_reconstruction,
                val_kl
            ) = self._evaluate_loader(val_loader)

            # Record statistics.
            self.history["train_loss"].append(
                train_total
            )

            self.history["train_reconstruction_loss"].append(
                train_reconstruction
            )

            self.history["train_kl_loss"].append(
                train_kl
            )

            self.history["validation_loss"].append(
                val_total
            )

            self.history["validation_reconstruction_loss"].append(
                val_reconstruction
            )

            self.history["validation_kl_loss"].append(
                val_kl
            )

            print(
                f"Epoch [{epoch + 1:03d}/{self.epochs}] "
                f"| Train: {train_total:.5f} "
                f"| Val: {val_total:.5f} "
                f"| Recon: {val_reconstruction:.5f} "
                f"| KL: {val_kl:.5f}"
            )

            # ================================================
            # EARLY STOPPING
            # ================================================

            if val_total < best_val_loss - self.min_delta:

                best_val_loss = val_total

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

        print("-" * 85)

        print(
            f"Best validation objective: {best_val_loss:.6f}"
        )

        print(
            "Variational Autoencoder training completed."
        )

        return self.history

    # --------------------------------------------------------
    # 6.4 VALIDATION
    # --------------------------------------------------------

    def _evaluate_loader(
        self,
        loader: DataLoader
    ) -> Tuple[float, float, float]:

        """
        Evaluate the VAE using deterministic posterior-mean
        reconstruction for stable validation and checkpoint
        selection.
        """

        self.model.eval()

        total_loss = 0.0
        reconstruction_total = 0.0
        kl_total = 0.0

        with torch.no_grad():

            for (batch,) in loader:

                batch = batch.to(self.device)

                reconstruction, mu, logvar = self.model(
                    batch,
                    sample=False
                )

                total, recon, kl = self.criterion(
                    reconstruction,
                    batch,
                    mu,
                    logvar
                )

                current_batch_size = batch.size(0)

                total_loss += (
                    total.item() * current_batch_size
                )

                reconstruction_total += (
                    recon.item() * current_batch_size
                )

                kl_total += (
                    kl.item() * current_batch_size
                )

        n_samples = len(loader.dataset)

        return (
            total_loss / n_samples,
            reconstruction_total / n_samples,
            kl_total / n_samples
        )

    # --------------------------------------------------------
    # 6.5 LATENT REPRESENTATION EXTRACTION
    # --------------------------------------------------------

    def encode(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        sample: bool = False,
        return_logvar: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        """
        Extract latent representations.

        By default:
            Returns posterior means (mu).

        If sample=True:
            Returns stochastic samples from q(z|x).

        If return_logvar=True:
            Also returns posterior log-variances.
        """

        self._check_fitted()

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        tensor = self._prepare_tensor(X)

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        latent_vectors = []
        log_variances = []

        with torch.no_grad():

            for (batch,) in loader:

                batch = batch.to(self.device)

                mu, logvar = self.model.encode(batch)

                if sample:

                    z = self.model.reparameterize(
                        mu,
                        logvar
                    )

                else:

                    z = mu

                latent_vectors.append(
                    z.cpu().numpy()
                )

                log_variances.append(
                    logvar.cpu().numpy()
                )

        latent_vectors = np.concatenate(
            latent_vectors,
            axis=0
        )

        if return_logvar:

            log_variances = np.concatenate(
                log_variances,
                axis=0
            )

            return latent_vectors, log_variances

        return latent_vectors

    # --------------------------------------------------------
    # 6.6 SAMPLE RECONSTRUCTION
    # --------------------------------------------------------

    def reconstruct(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:

        """
        Reconstruct samples using posterior means.

        Returns reconstructed features in their original
        input scale.
        """

        self._check_fitted()

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        X = self._validate_data(X)

        tensor = self._prepare_tensor(X)

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        reconstructions = []

        with torch.no_grad():

            for (batch,) in loader:

                batch = batch.to(self.device)

                reconstruction, _, _ = self.model(
                    batch,
                    sample=False
                )

                reconstructions.append(
                    reconstruction.cpu().numpy()
                )

        reconstruction = np.concatenate(
            reconstructions,
            axis=0
        )

        # Reverse standardization.
        reconstruction = (
            reconstruction * self.std + self.mean
        )

        return reconstruction

    # --------------------------------------------------------
    # 6.7 RECONSTRUCTION ERROR
    # --------------------------------------------------------

    def reconstruction_error(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        """
        Compute the mean squared reconstruction error
        for each sample.

        Returns:
            Array of shape (n_samples,).
        """

        self._check_fitted()

        X = self._validate_data(X)

        X_scaled = self._standardize(X)

        reconstructed = self.reconstruct(X)

        reconstructed_scaled = (
            reconstructed - self.mean
        ) / self.std

        errors = np.mean(
            np.square(
                X_scaled - reconstructed_scaled
            ),
            axis=1
        )

        return errors

    # --------------------------------------------------------
    # 6.8 LATENT-SPACE DIVERSITY ANALYSIS
    # --------------------------------------------------------

    def analyze_latent_diversity(
        self,
        X: np.ndarray,
        max_pairs: int = 100000,
        active_threshold: float = 0.01,
        seed: int = 42
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:

        """
        Analyze the distribution and diversity of latent
        representations.

        Metrics:

            1. Mean pairwise Euclidean distance.
            2. Median pairwise Euclidean distance.
            3. Mean pairwise cosine distance.
            4. Mean latent dimension variance.
            5. Number of active latent dimensions.
            6. Average posterior standard deviation.
            7. Distance of samples from the latent centroid.

        Random pair sampling is used to avoid allocating
        a full N x N distance matrix.

        Returns:
            report:
                Dictionary containing global diversity metrics.

            diversity_scores:
                Per-sample distance from the latent centroid.

            posterior_uncertainty:
                Per-sample average posterior standard deviation.
        """

        self._check_fitted()

        if max_pairs < 1:
            raise ValueError(
                "max_pairs must be positive."
            )

        if active_threshold < 0:
            raise ValueError(
                "active_threshold must be non-negative."
            )

        X = self._validate_data(X)

        # Extract posterior statistics.
        mu, logvar = self.encode(
            X,
            sample=False,
            return_logvar=True
        )

        n_samples = mu.shape[0]

        # ----------------------------------------------------
        # Latent dimension variance
        # ----------------------------------------------------

        dimension_variance = np.var(
            mu,
            axis=0
        )

        mean_latent_variance = float(
            np.mean(dimension_variance)
        )

        active_dimensions = int(
            np.sum(
                dimension_variance > active_threshold
            )
        )

        # ----------------------------------------------------
        # Posterior uncertainty
        # ----------------------------------------------------

        posterior_std = np.exp(
            0.5 * np.clip(
                logvar,
                -20.0,
                10.0
            )
        )

        posterior_uncertainty = np.mean(
            posterior_std,
            axis=1
        )

        # ----------------------------------------------------
        # Sample diversity relative to latent centroid
        # ----------------------------------------------------

        centroid = np.mean(
            mu,
            axis=0,
            keepdims=True
        )

        diversity_scores = np.sqrt(
            np.mean(
                np.square(mu - centroid),
                axis=1
            )
        )

        # ----------------------------------------------------
        # Random pairwise distance estimation
        # ----------------------------------------------------

        if n_samples >= 2:

            n_pairs = min(
                max_pairs,
                max(1, n_samples * (n_samples - 1) // 2)
            )

            rng = np.random.default_rng(seed)

            # Generate distinct indices without creating
            # a complete pairwise matrix.
            index_a = rng.integers(
                0,
                n_samples,
                size=n_pairs
            )

            offsets = rng.integers(
                1,
                n_samples,
                size=n_pairs
            )

            index_b = (
                index_a + offsets
            ) % n_samples

            vectors_a = mu[index_a]
            vectors_b = mu[index_b]

            differences = vectors_a - vectors_b

            euclidean_distances = np.linalg.norm(
                differences,
                axis=1
            )

            dot_products = np.sum(
                vectors_a * vectors_b,
                axis=1
            )

            norms_a = np.linalg.norm(
                vectors_a,
                axis=1
            )

            norms_b = np.linalg.norm(
                vectors_b,
                axis=1
            )

            cosine_similarity = dot_products / (
                np.maximum(norms_a * norms_b, 1e-12)
            )

            cosine_similarity = np.clip(
                cosine_similarity,
                -1.0,
                1.0
            )

            cosine_distances = (
                1.0 - cosine_similarity
            )

            mean_euclidean = float(
                np.mean(euclidean_distances)
            )

            median_euclidean = float(
                np.median(euclidean_distances)
            )

            mean_cosine = float(
                np.mean(cosine_distances)
            )

        else:

            n_pairs = 0

            mean_euclidean = 0.0
            median_euclidean = 0.0
            mean_cosine = 0.0

        # ----------------------------------------------------
        # Diversity report
        # ----------------------------------------------------

        report = {

            "n_samples": int(n_samples),

            "latent_dim": int(self.latent_dim),

            "sampled_pair_count": int(n_pairs),

            "mean_pairwise_euclidean_distance":
                mean_euclidean,

            "median_pairwise_euclidean_distance":
                median_euclidean,

            "mean_pairwise_cosine_distance":
                mean_cosine,

            "mean_latent_dimension_variance":
                mean_latent_variance,

            "active_latent_dimensions":
                active_dimensions,

            "active_dimension_threshold":
                float(active_threshold),

            "mean_posterior_standard_deviation":
                float(np.mean(posterior_uncertainty)),

            "mean_sample_distance_from_centroid":
                float(np.mean(diversity_scores)),

            "latent_centroid": centroid.flatten().tolist()

        }

        return (
            report,
            diversity_scores,
            posterior_uncertainty
        )

    # --------------------------------------------------------
    # 6.9 PRIOR-BASED GENERATION
    # --------------------------------------------------------

    def sample_prior(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:

        """
        Generate new feature representations by sampling:

            z ~ N(0, I)

        Then decode the sampled latent vectors.

        Returns generated features in the original input scale.

        Note:
            These are generated feature vectors, not guaranteed
            valid movie scripts or decoded text.
        """

        self._check_fitted()

        if n_samples < 1:
            raise ValueError(
                "n_samples must be positive."
            )

        self.model.eval()

        generator = None

        if seed is not None:

            generator = torch.Generator(
                device=self.device
            )

            generator.manual_seed(seed)

        with torch.no_grad():

            z = torch.randn(
                n_samples,
                self.latent_dim,
                device=self.device,
                generator=generator
            )

            generated = self.model.decode(z)

        generated = generated.cpu().numpy()

        # Reverse feature standardization.
        generated = (
            generated * self.std + self.mean
        )

        return generated

    # --------------------------------------------------------
    # 6.10 SAVE MODEL CHECKPOINT
    # --------------------------------------------------------

    def save(self, path: str) -> None:

        """
        Save:
            - Model architecture configuration.
            - Model parameters.
            - Preprocessing parameters.
            - Training configuration.
            - Training history.
        """

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

            "latent_dim": self.latent_dim,

            "dropout": self.dropout,

            "model_state_dict": self.model.state_dict(),

            "mean": torch.tensor(self.mean),

            "std": torch.tensor(self.std),

            "learning_rate": self.learning_rate,

            "batch_size": self.batch_size,

            "epochs": self.epochs,

            "patience": self.patience,

            "beta": self.beta,

            "validation_split": self.validation_split,

            "min_delta": self.min_delta,

            "gradient_clip": self.gradient_clip,

            "seed": self.seed,

            "history": self.history

        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"VAE checkpoint saved to: {path}"
        )

    # --------------------------------------------------------
    # 6.11 LOAD MODEL CHECKPOINT
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None
    ):

        """
        Restore a previously trained VAE and its
        preprocessing configuration.
        """

        try:

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

        except TypeError:

            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                path,
                map_location="cpu"
            )

        pipeline = cls(

            input_dim=checkpoint["input_dim"],

            hidden_dims=checkpoint["hidden_dims"],

            latent_dim=checkpoint["latent_dim"],

            dropout=checkpoint["dropout"],

            learning_rate=checkpoint["learning_rate"],

            batch_size=checkpoint["batch_size"],

            epochs=checkpoint["epochs"],

            patience=checkpoint["patience"],

            beta=checkpoint["beta"],

            validation_split=checkpoint["validation_split"],

            min_delta=checkpoint["min_delta"],

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
                "The VAE must be trained or loaded before "
                "using inference and analysis methods."
            )


# ============================================================
# 7. EXPORT RESULTS
# ============================================================

def export_results(
    pipeline: VAEPipeline,
    X: np.ndarray,
    output_dir: str,
    max_pairs: int = 100000,
    active_threshold: float = 0.01,
    seed: int = 42
) -> None:

    """
    Export trained VAE representations and diversity analysis.

    Generated files:

        - latent_means.npy
        - latent_log_variances.npy
        - reconstruction_errors.npy
        - latent_diversity_scores.npy
        - posterior_uncertainty.npy
        - diversity_report.json
        - training_history.json
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract posterior representations
    # --------------------------------------------------------

    mu, logvar = pipeline.encode(
        X,
        sample=False,
        return_logvar=True
    )

    np.save(
        os.path.join(
            output_dir,
            "latent_means.npy"
        ),
        mu
    )

    np.save(
        os.path.join(
            output_dir,
            "latent_log_variances.npy"
        ),
        logvar
    )

    # --------------------------------------------------------
    # Reconstruction errors
    # --------------------------------------------------------

    errors = pipeline.reconstruction_error(X)

    np.save(
        os.path.join(
            output_dir,
            "reconstruction_errors.npy"
        ),
        errors
    )

    # --------------------------------------------------------
    # Latent diversity analysis
    # --------------------------------------------------------

    (
        report,
        diversity_scores,
        posterior_uncertainty
    ) = pipeline.analyze_latent_diversity(
        X,
        max_pairs=max_pairs,
        active_threshold=active_threshold,
        seed=seed
    )

    np.save(
        os.path.join(
            output_dir,
            "latent_diversity_scores.npy"
        ),
        diversity_scores
    )

    np.save(
        os.path.join(
            output_dir,
            "posterior_uncertainty.npy"
        ),
        posterior_uncertainty
    )

    # --------------------------------------------------------
    # Export diversity report
    # --------------------------------------------------------

    report_path = os.path.join(
        output_dir,
        "diversity_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Export training history
    # --------------------------------------------------------

    history_path = os.path.join(
        output_dir,
        "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pipeline.history,
            file,
            indent=4
        )

    print("\nVAE results exported successfully.")

    print(
        f"Latent representations: {mu.shape}"
    )

    print(
        f"Reconstruction errors: {errors.shape}"
    )

    print(
        f"Active latent dimensions: "
        f"{report['active_latent_dimensions']}"
    )

    print(
        f"Mean pairwise Euclidean distance: "
        f"{report['mean_pairwise_euclidean_distance']:.6f}"
    )

    print(
        f"Mean pairwise cosine distance: "
        f"{report['mean_pairwise_cosine_distance']:.6f}"
    )

    print(
        f"Results directory: {output_dir}"
    )


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Variational Autoencoder for latent-space "
            "diversity analysis."
        )
    )

    # --------------------------------------------------------
    # Input and output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the numerical feature matrix (.npy)."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/variational_autoencoder"
    )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

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
        "--beta",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0
    )

    # --------------------------------------------------------
    # Diversity analysis
    # --------------------------------------------------------

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.01
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
        default=None,
        help="Optional device: cpu or cuda."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # Load numerical features
    # --------------------------------------------------------

    X = np.load(
        args.input,
        allow_pickle=False
    )

    if X.ndim != 2:
        raise ValueError(
            "Input file must contain a 2D feature matrix."
        )

    # --------------------------------------------------------
    # Parse architecture configuration
    # --------------------------------------------------------

    hidden_dims = [
        int(dim.strip())
        for dim in args.hidden_dims.split(",")
    ]

    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    pipeline = VAEPipeline(

        input_dim=X.shape[1],

        hidden_dims=hidden_dims,

        latent_dim=args.latent_dim,

        dropout=args.dropout,

        learning_rate=args.learning_rate,

        batch_size=args.batch_size,

        epochs=args.epochs,

        patience=args.patience,

        beta=args.beta,

        validation_split=args.validation_split,

        min_delta=args.min_delta,

        gradient_clip=args.gradient_clip,

        device=args.device,

        seed=args.seed

    )

    # --------------------------------------------------------
    # Train VAE
    # --------------------------------------------------------

    pipeline.fit(X)

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
            "vae_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # Export representations and diversity metrics
    # --------------------------------------------------------

    export_results(

        pipeline=pipeline,

        X=X,

        output_dir=args.output_dir,

        max_pairs=args.max_pairs,

        active_threshold=args.active_threshold,

        seed=args.seed

    )


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
