'''
import numpy as np
import yaml
from .adaptive_iforest import AdaptiveIsolationForest


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
'''
#only aif
# services/if_trainer.py

import numpy as np
from .adaptive_iforest import AdaptiveIsolationForest


class IFOnlineTrainer:
    """
    Online trainer for AdaptiveIsolationForest.

    Per room (resourceid) we keep:
      - models[rid]: AdaptiveIsolationForest instance
      - counts[rid]: how many updates seen
      - score_history[rid]: recent anomaly scores (sliding history)

    We:
      - use a warm-up phase before flagging anomalies
      - compute a threshold as a percentile of recent scores
    """

    # You can tune these three for sensitivity:
    WARMUP_STEPS = 5          # very short warmup for testing
    MAX_HISTORY = 200         # how many scores we keep per room
    THRESH_PCT = 60           # percentile: 60 = quite sensitive, many anomalies

    def __init__(self):
        # One AdaptiveIsolationForest per room/resourceid
        self.models = {}         # rid -> AdaptiveIsolationForest
        self.counts = {}         # rid -> int
        self.score_history = {}  # rid -> list[float]

    def update(self, sequence_np, rid):
        """
        sequence_np: numpy array of shape (1, feature_dim) or (feature_dim,)
        rid: room/resourceid string

        Returns:
            score: float
            is_anomaly: bool
            threshold: float or None (if still warming up)
        """
        # Ensure numpy array with shape (1, -1)
        X = np.asarray(sequence_np, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialise per-room structures
        if rid not in self.models:
            self.models[rid] = AdaptiveIsolationForest(
                n_estimators=100,
                max_samples="auto",
                contamination="auto",
                random_state=42,
                max_history=self.MAX_HISTORY,
            )
            self.counts[rid] = 0
            self.score_history[rid] = []

        model = self.models[rid]

        # "Online" fit: refit on room-specific sliding history (inside AIF)
        model.partial_fit(X)

        # Score this window
        scores = model.score_samples(X)  # shape (1,)
        score = float(scores[0])

        # Update counters and history
        self.counts[rid] += 1
        self.score_history[rid].append(score)
        if len(self.score_history[rid]) > self.MAX_HISTORY:
            self.score_history[rid] = self.score_history[rid][-self.MAX_HISTORY :]

        # Warm-up phase: don't flag anomalies yet, threshold is unknown
        if self.counts[rid] < self.WARMUP_STEPS or len(self.score_history[rid]) < self.WARMUP_STEPS:
            return score, False, None

        # Compute threshold from per-room history
        scores_arr = np.array(self.score_history[rid], dtype=float)
        threshold = float(np.percentile(scores_arr, self.THRESH_PCT))

        is_anomaly = bool(score > threshold)

        return score, is_anomaly, threshold
