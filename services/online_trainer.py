import torch
import yaml
import numpy as np
import pickle
from pathlib import Path
from .vae_model import AdaptiveVAE


class OnlineTrainer:
    """
    Improved Online VAE trainer with:
    - Warm-up phase
    - Clean threshold estimation
    - Stable sliding error buffer
    """

    WARMUP_STEPS = 200           # Number of updates before anomaly detection
    MAX_ERRORS = 500             # Sliding window size for thresholds
    THRESHOLD_PERCENTILE = 95    # Recommended: 90-99

    def __init__(self, config_path="config/hyperparameters-config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["vae"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AdaptiveVAE(
            input_dim=self.cfg["input_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            latent_dim=self.cfg["latent_dim"],
            dropout=self.cfg["dropout"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg["learning_rate"]
        )

        # Per-room error buffers & counters
        self.error_buffer = {}   # rid → list of recent errors
        self.counts = {}         # rid → update count

    def update(self, sequence_tensor, rid):
        """
        Online update:
        - train
        - compute reconstruction error
        - warm up
        - stable anomaly detection
        """
        self.model.train()
        sequence_tensor = sequence_tensor.to(self.device)

        # Forward pass
        recon, mu, logvar = self.model(sequence_tensor)
        loss = self.model.reconstruction_loss(
            sequence_tensor, recon, mu, logvar, beta=self.cfg["beta"]
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reconstruction error (scalar)
        recon_error = torch.mean((sequence_tensor - recon) ** 2).item()

        # Initialize structures for this room
        if rid not in self.counts:
            self.counts[rid] = 0
            self.error_buffer[rid] = []

        self.counts[rid] += 1

        # --------------------------
        # 1) WARM-UP PHASE
        # --------------------------
        if self.counts[rid] < self.WARMUP_STEPS:
            self.error_buffer[rid].append(recon_error)
            return recon_error, False, None

        # --------------------------
        # 2) NORMAL OPERATION
        # --------------------------
        # Use sliding window of errors
        self.error_buffer[rid].append(recon_error)
        if len(self.error_buffer[rid]) > self.MAX_ERRORS:
            self.error_buffer[rid] = self.error_buffer[rid][-self.MAX_ERRORS:]

        # Compute threshold from buffer
        threshold = np.percentile(
            self.error_buffer[rid],
            self.THRESHOLD_PERCENTILE
        )

        # Flag anomaly
        is_anomaly = recon_error > threshold

        # Optional improvement:
        # only update thresholds using non-anomalous errors
        # if not is_anomaly:
        #     self.error_buffer[rid].append(recon_error)

        return recon_error, is_anomaly, threshold

    def save_model(self, resourceid):
        torch.save(self.model.state_dict(), f"models/vae_{resourceid}.pt")
