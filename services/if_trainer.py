import numpy as np
import yaml
from .adaptive_iforest import AdaptiveIsolationForest


class IFOnlineTrainer:
    """
    Online trainer/wrapper around AdaptiveIsolationForest.
    Mirrors the API of OnlineTrainer (VAE) so we can compare easily.

    update(sequence_np, resourceid) -> (score, is_anomaly, threshold)
    """

    def __init__(self, config_path: str = "config/hyperparameters-config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg_if = cfg.get("iforest", {})
        self.model = AdaptiveIsolationForest()  # use internal defaults

        # per-room score history for adaptive thresholds
        self.scores_history = {}  # resourceid -> list[float]
        self.max_history = self.cfg_if.get("max_history", 1000)
        self.quantile = self.cfg_if.get("threshold_quantile", 0.95)

        # track whether we have done an initial fit using this room's data
        self.initialized_rooms = set()

    def update(self, sequence_np: np.ndarray, resourceid: str):
        """
        sequence_np: shape (1, n_features), already scaled (same as VAE input).
        Returns: (score, is_anomaly, threshold)
        """
        sequence_np = np.asarray(sequence_np)
        if sequence_np.ndim == 1:
            sequence_np = sequence_np.reshape(1, -1)

        # First time we see any room -> initial fit
        if resourceid not in self.initialized_rooms:
            self.model.fit(sequence_np)
            self.initialized_rooms.add(resourceid)
        else:
            self.model.update(sequence_np)

        # Score: higher = more anomalous
        scores = self.model.score_samples(sequence_np)  # shape (1,)
        score = float(scores[0])

        # Rolling history for adaptive per-room threshold
        if resourceid not in self.scores_history:
            self.scores_history[resourceid] = []
        hist = self.scores_history[resourceid]
        hist.append(score)
        if len(hist) > self.max_history:
            self.scores_history[resourceid] = hist[-self.max_history:]
            hist = self.scores_history[resourceid]

        # Compute threshold
        if len(hist) < 10:
            threshold = float(np.quantile(hist, self.quantile)) if hist else score + 1e-6
        else:
            threshold = float(np.quantile(hist, self.quantile))

        is_anomaly = score > threshold
        return score, is_anomaly, threshold
