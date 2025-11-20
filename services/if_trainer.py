import numpy as np
import yaml
from pathlib import Path
from .adaptive_iforest import AdaptiveIsolationForest

class IFOnlineTrainer:
    """
    Online trainer/wrapper around AdaptiveIsolationForest.
    Mirrors the API of OnlineTrainer (VAE) so we can compare easily.
    """

    def __init__(self, config_path="config/hyperparameters-config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg_if = cfg.get("iforest", {})  # add this section in YAML

        self.model = AdaptiveIsolationForest(
            n_estimators=self.cfg_if.get("n_estimators", 50),
            replace_rate=self.cfg_if.get("replace_rate", 0.1),
            random_state=self.cfg_if.get("random_state", 42),
        )

        # per-room score history for adaptive thresholds
        self.scores_history = {}  # resourceid -> list[float]
        self.max_history = self.cfg_if.get("max_history", 1000)
        self.quantile = self.cfg_if.get("threshold_quantile", 0.95)

    def update(self, sequence_np: np.ndarray, resourceid: str):
        """
        sequence_np: shape (1, n_features), already scaled (same as VAE input).
        Returns: (score, is_anomaly, threshold)
        """
        # 1. Update the IF model with new data (depends on your class’ API)
        #    e.g. if your AdaptiveIsolationForest has fit_partial or partial_fit:
        self.model.fit_partial(sequence_np)

        # 2. Get anomaly score (the higher, the more abnormal)
        scores = self.model.score_samples(sequence_np)  # shape (1,)
        score = float(scores[0])

        # 3. Update rolling history for this room
        if resourceid not in self.scores_history:
            self.scores_history[resourceid] = []
        hist = self.scores_history[resourceid]
        hist.append(score)
        if len(hist) > self.max_history:
            self.scores_history[resourceid] = hist[-self.max_history:]
            hist = self.scores_history[resourceid]

        # 4. Adaptive threshold (same logic as VAE but on IF scores)
        if len(hist) < 10:
            # not enough history yet – be conservative
            threshold = float(np.quantile(hist, self.quantile)) if hist else score + 1e-6
        else:
            threshold = float(np.quantile(hist, self.quantile))

        is_anomaly = score > threshold
        return score, is_anomaly, threshold
