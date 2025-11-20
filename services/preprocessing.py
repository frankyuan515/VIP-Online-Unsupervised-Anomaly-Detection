import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch

# Sensor columns expected in incoming JSON messages
SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]

WINDOW_SIZE = 10  # 10 timesteps -> 6 * 10 = 60 features


class StreamPreprocessor:
    """
    Handles:
    - Loading per-room MinMaxScalers from models/scaler_per_room.pkl
    - Maintaining per-room sliding windows
    - Returning tensors/arrays in shape (1, 60) for VAE/IF
    """

    def __init__(self, scaler_path: str = "models/scaler_per_room.pkl"):
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        with scaler_path.open("rb") as f:
            self.scalers: dict[str, MinMaxScaler] = pickle.load(f)

        # per-room buffers of recent scaled rows
        self.buffers: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_torch_window(self, data_dict: dict, resourceid: str) -> torch.Tensor:
        """
        Returns (1, 60) torch.float32 tensor for VAE.
        """
        seq_np = self._build_window_np(data_dict, resourceid)
        return torch.tensor(seq_np, dtype=torch.float32)

    def to_numpy_window(self, data_dict: dict, resourceid: str) -> np.ndarray:
        """
        Returns (1, 60) numpy array for Isolation Forest.
        """
        return self._build_window_np(data_dict, resourceid)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _build_window_np(self, data_dict: dict, resourceid: str) -> np.ndarray:
        """
        - Take one sensor snapshot (6 values)
        - Scale using room-specific MinMaxScaler
        - Push into per-room buffer
        - Build a 10-step window (pad with first element if not full yet)
        - Flatten to shape (1, 60)
        """
        scaler = self.scalers.get(resourceid)
        if scaler is None:
            raise ValueError(f"No scaler for {resourceid}")

        # Extract in correct order
        sensors = np.array([data_dict[col] for col in SENSOR_COLS]).reshape(1, -1)

        # Apply scaler (ignore feature names warning)
        scaled = scaler.transform(sensors)  # shape (1, 6)

        # Initialize buffer if needed
        if resourceid not in self.buffers:
            self.buffers[resourceid] = []
        buf = self.buffers[resourceid]

        buf.append(scaled[0])
        if len(buf) < WINDOW_SIZE:
            # pad with the first element until we have WINDOW_SIZE
            pad = [buf[0]] * (WINDOW_SIZE - len(buf))
            window = pad + buf
        else:
            window = buf[-WINDOW_SIZE:]

        window = np.stack(window, axis=0)  # (10, 6)
        flat = window.flatten().reshape(1, -1)  # (1, 60)
        return flat
