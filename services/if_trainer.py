import numpy as np
import yaml
from .adaptive_iforest import AdaptiveIsolationForest

'''
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
    
'''
class IFOnlineTrainer:
    """
    Manages an AdaptiveIsolationForest and per-room thresholds.

    - For each room (resourceid), we keep:
        - count of updates
        - sliding window of recent scores

    - We use a warm-up phase before we start flagging anomalies.
    """

    WARMUP_STEPS = 100       # number of windows before trusting threshold
    MAX_SCORES = 500         # sliding buffer length per room
    THRESH_PCT = 98          # percentile for anomaly threshold

    def __init__(self):
        self.model = AdaptiveIsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination="auto",
            random_state=42,
        )

        # per-room state
        self.counts = {}        # rid -> number of updates
        self.score_buffer = {}  # rid -> list of recent scores

    def update(self, sequence_np, rid):
        """
        sequence_np: numpy array of shape (1, feature_dim) or (feature_dim,)
        rid: room/resourceid string

        Returns: (score, is_anomaly, threshold_or_None)
        """
        X = np.asarray(sequence_np, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize per-room structures
        if rid not in self.counts:
            self.counts[rid] = 0
            self.score_buffer[rid] = []

        # "Online" fit: in this simple version we refit on this batch
        self.model.partial_fit(X)

        # Score this window
        scores = self.model.score_samples(X)  # shape (1,)
        score = float(scores[0])

        # Update per-room stats
        self.counts[rid] += 1
        self.score_buffer[rid].append(score)
        if len(self.score_buffer[rid]) > self.MAX_SCORES:
            self.score_buffer[rid] = self.score_buffer[rid][-self.MAX_SCORES :]

        # Warm-up: do not flag anomalies yet, threshold = None
        if self.counts[rid] < self.WARMUP_STEPS:
            return score, False, None

        # Compute threshold from recent scores
        scores_arr = np.array(self.score_buffer[rid], dtype=float)
        if len(scores_arr) < 10:
            return score, False, None

        threshold = float(np.percentile(scores_arr, self.THRESH_PCT))
        is_anomaly = bool(score > threshold)

        return score, is_anomaly, threshold