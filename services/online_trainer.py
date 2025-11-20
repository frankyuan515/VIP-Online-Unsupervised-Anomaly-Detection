import torch
import yaml
import numpy as np
import pickle
from pathlib import Path
from .vae_model import AdaptiveVAE


class OnlineTrainer:
    """Manages per-room VAE updates and thresholds."""

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
        self.thresholds = {}  # {rid: list of errors}
        self.scalers_path = Path("models/scaler_per_room.pkl")

    def load_scalers(self):
        with open(self.scalers_path, "rb") as f:
            return pickle.load(f)

    def update(self, sequence_tensor, resourceid):
        """Online update: forward → loss → backprop → threshold."""
        self.model.train()
        sequence_tensor = sequence_tensor.to(self.device)
        recon, mu, logvar = self.model(sequence_tensor)
        loss = self.model.reconstruction_loss(
            sequence_tensor, recon, mu, logvar, beta=self.cfg["beta"]
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        recon_error = torch.mean((sequence_tensor - recon) ** 2).item()
        if resourceid not in self.thresholds:
            self.thresholds[resourceid] = []
        self.thresholds[resourceid].append(recon_error)
        if len(self.thresholds[resourceid]) > 1000:
            self.thresholds[resourceid] = self.thresholds[resourceid][-1000:]
        threshold = np.percentile(self.thresholds[resourceid], 95)
        is_anomaly = recon_error > threshold
        return recon_error, is_anomaly, threshold

    def save_model(self, resourceid):
        torch.save(self.model.state_dict(), f"models/vae_{resourceid}.pt")
