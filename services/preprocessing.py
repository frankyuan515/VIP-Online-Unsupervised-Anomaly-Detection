import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
from config.features_config import SENSOR_COLS  # Load from YAML if needed

SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]
WINDOW_SIZE = 10


class StreamPreprocessor:
    """Handles scaling and windowing for streaming data."""

    def __init__(self):
        self.scalers_path = Path("models/scaler_per_room.pkl")
        with open(self.scalers_path, "rb") as f:
            self.scalers = pickle.load(f)

    def preprocess_stream(self, data_dict, resourceid):
        """Scale new data and build window (mock buffer for real streaming)."""
        scaler = self.scalers.get(resourceid)
        if scaler is None:
            raise ValueError(f"No scaler for {resourceid}")

        # Extract sensors
        sensors = np.array([data_dict[col] for col in SENSOR_COLS]).reshape(1, -1)
        scaled = scaler.transform(sensors)

        # Build window (in real: sliding buffer)
        window = np.tile(scaled, (WINDOW_SIZE, 1))  # Mock; use buffer in prod
        return torch.tensor(window.flatten(), dtype=torch.float32).unsqueeze(
            0
        )  # (1, 60)
    
    #add for isolation forest
    def to_torch_window(self, data_dict, resourceid):
        """Existing function for VAE – returns (1, 60) torch tensor."""
        seq_np = self._build_window_np(data_dict, resourceid)
        import torch
        return torch.tensor(seq_np, dtype=torch.float32)

    def to_numpy_window(self, data_dict, resourceid):
        """New helper for Isolation Forest – returns (1, 60) numpy array."""
        return self._build_window_np(data_dict, resourceid)

    def _build_window_np(self, data_dict, resourceid):
        scaler = self.scalers.get(resourceid)
        if scaler is None:
            raise ValueError(f"No scaler for {resourceid}")

        sensors = np.array([data_dict[col] for col in SENSOR_COLS]).reshape(1, -1)
        scaled = scaler.transform(sensors)

        # TODO: replace this fake tiling with a real sliding buffer
        window = np.tile(scaled, (WINDOW_SIZE, 1))   # (10, 6)
        flat = window.flatten().reshape(1, -1)       # (1, 60)
        return flat
